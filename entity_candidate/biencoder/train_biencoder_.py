# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import argparse
import pickle
import torch
import json
import sys
import io
import random
import time
import numpy as np

from multiprocessing.pool import ThreadPool

from tqdm import tqdm, trange
from collections import OrderedDict

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers.optimization import WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.modeling_utils import WEIGHTS_NAME

from blink.biencoder.biencoder_ import BiEncoderRanker, load_biencoder
import logging

import blink.candidate_ranking.utils as utils
import blink.biencoder.data_process_ as data
from blink.biencoder.zeshel_utils import DOC_PATH, WORLDS, world_to_id
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import BlinkParser


logger = None

# The evaluate function during training uses in-batch negatives:
# for a batch of size B, the labels from the batch are used as label candidates
# B is controlled by the parameter eval_batch_size
def evaluate(reranker, eval_dataloader, params, device, logger,):
    # 将模型设置为评估模式，即关闭Dropout和Batch Normalization层。
    reranker.model.eval()
    # 是否显示评估进度条
    if params["silent"]:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")
    results = {}
    # 评估准确率
    eval_accuracy = 0.0
    # 评估样本数量
    nb_eval_examples = 0
    # 评估步骤数量
    nb_eval_steps = 0
    # 遍历每个批次
    for step, batch in enumerate(iter_):
        # 将批次中的数据移动到设备上
        batch = tuple(t.to(device) for t in batch)
        # 将批次中的数据解析为context_input和candidate_input
        context_input, candidate_input, _ = batch
        # 关闭梯度计算
        with torch.no_grad():
            # 计算验证集上的损失和预测结果
            eval_loss, logits = reranker(context_input, candidate_input)
        # 将预测结果从张量转换为NumPy数组，并移动到CPU上
        logits = logits.detach().cpu().numpy()
        # 创建一个与评估批次大小相同的标签张量   [0 1 2 3 4 5 6 7]
        label_ids = torch.LongTensor(torch.arange(params["eval_batch_size"])).numpy()
        # 计算批次中预测正确的数量
        tmp_eval_accuracy, _ = utils.accuracy(logits, label_ids)
        # print(tmp_eval_accuracy)
        # 累加预测正确的数量
        eval_accuracy += tmp_eval_accuracy
        # 累加评估样本数量
        nb_eval_examples += context_input.size(0)
        # 累加评估步骤数量
        nb_eval_steps += 1
    # 计算评估准确率
    normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)
    # 将评估准确率存储到结果字典中
    results["normalized_accuracy"] = normalized_eval_accuracy
    return results
def evaluate_(reranker, eval_dataloader, params, device, logger,):
    # 将模型设置为评估模式，即关闭Dropout和Batch Normalization层。
    reranker.model.eval()
    # 是否显示评估进度条
    if params["silent"]:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")
    results = {}
    # 评估准确率
    eval_accuracy = 0.0
    # 评估样本数量
    nb_eval_examples = 0
    # 评估步骤数量
    nb_eval_steps = 0
    # 遍历每个批次
    for step, batch in enumerate(iter_):
        # 将批次中的数据移动到设备上
        batch = tuple(t.to(device) for t in batch)
        # 将批次中的数据解析为context_input和candidate_input
        context_input, candidate_input_A, candidate_input_B, label = batch
        # 关闭梯度计算
        with torch.no_grad():
            # 计算验证集上的损失和预测结果
            logits = reranker(context_input, candidate_input_A, candidate_input_B, label)
            # 将预测结果从张量转换为NumPy数组，并移动到CPU上
            outputs = np.argmax(logits.cpu().detach().numpy(), axis=1).flatten()
            seq_labels = np.argmax(label.cpu().detach().numpy(), axis=1).flatten()
        # 计算批次中预测正确的数量
        tmp_eval_accuracy = np.sum(outputs == seq_labels)
        # print(tmp_eval_accuracy)
        # 累加预测正确的数量
        eval_accuracy += tmp_eval_accuracy
        # 累加评估样本数量
        nb_eval_examples += context_input.size(0)
        # 累加评估步骤数量
        nb_eval_steps += 1
    # 计算评估准确率
    normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)
    # 将评估准确率存储到结果字典中
    results["normalized_accuracy"] = normalized_eval_accuracy
    return results

def get_optimizer(model, params):
    return get_bert_optimizer(
        [model],
        params["type_optimization"],
        params["learning_rate"],
        fp16=params.get("fp16"),
    )

def get_optimizer(model, params):
    return get_bert_optimizer(
        [model],
        params["type_optimization"],
        params["learning_rate"],
        fp16=params.get("fp16"),
    )


def get_scheduler(params, optimizer, len_train_data, logger):
    batch_size = params["train_batch_size"]
    grad_acc = params["gradient_accumulation_steps"]
    epochs = params["num_train_epochs"]

    num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * params["warmup_proportion"])

    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps,
    )
    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler


def main(params):
    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    logger = utils.get_logger(params["output_path"])

    # Init model
    reranker = BiEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model

    device = reranker.device
    n_gpu = reranker.n_gpu

    if params["gradient_accumulation_steps"] < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                params["gradient_accumulation_steps"]
            )
        )

    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
    # args.gradient_accumulation_steps = args.gradient_accumulation_steps // n_gpu
    params["train_batch_size"] = (
        params["train_batch_size"] // params["gradient_accumulation_steps"]
    )
    train_batch_size = params["train_batch_size"]
    eval_batch_size = params["eval_batch_size"]
    grad_acc_steps = params["gradient_accumulation_steps"]

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # Load train data
    # train_samples, valid_samples = utils.new_read_dataset("train", params["data_path"])
    train_samples, valid_samples = utils.new_read_dataset("train0", params["data_path"])
    logger.info("Read %d train samples." % len(train_samples))
   
    train_data, train_tensor_data = data.process_mention_data(
        train_samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
    )
    if params["shuffle"]:
        train_sampler = RandomSampler(train_tensor_data)
    else:
        train_sampler = SequentialSampler(train_tensor_data)

    train_dataloader = DataLoader(
        train_tensor_data, sampler=train_sampler, batch_size=train_batch_size
    )

    # Load eval data
    # TODO: reduce duplicated code here
    logger.info("Read %d valid samples." % len(valid_samples))
    valid_data, valid_tensor_data = data.process_mention_data_(
        valid_samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
    )
    if params["shuffle"]:
        valid_sampler = RandomSampler(valid_tensor_data)
    else:
        valid_sampler = SequentialSampler(valid_tensor_data)
    valid_dataloader = DataLoader(
        valid_tensor_data, sampler=valid_sampler, batch_size=eval_batch_size
    )

    # evaluate before training
    # results = evaluate(
    #     reranker, valid_dataloader, params, device=device, logger=logger,
    # )


    time_start = time.time()

    utils.write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )

    logger.info("Starting training")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False)
    )

    optimizer = get_optimizer(model, params)
    scheduler = get_scheduler(params, optimizer, len(train_tensor_data), logger)

    # 设置模型为训练模式
    model.train()

    # 初始化最佳轮次id和最佳得分
    best_epoch_idx = -1
    best_score = -1

    # 获取训练的轮次
    num_train_epochs = params["num_train_epochs"]
    # 迭代训练轮次
    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        # 初始化训练损失和结果
        tr_loss = 0
        results = None

        # 判断是否显示进度条
        if params["silent"]:
            iter_ = train_dataloader
        else:
            iter_ = tqdm(train_dataloader, desc="Batch")

        # 迭代训练数据集的批次
        for step, batch in enumerate(iter_):
            # 将批次中的数据移动到指定的设备上（GPU或CPU）
            batch = tuple(t.to(device) for t in batch)
            # 从批次中获取上下文输入和候选输入
            context_input, candidate_input_A, candidate_input_B= batch
            # context_input, candidate_input, _, _ = batch
            # 执行模型的前向传播，计算损失
            loss = reranker(context_input, candidate_input_A,  candidate_input_B)
            # if n_gpu > 1:
            #     loss = loss.mean()    # mean() to average on multi-gpu.

            # 如果梯度累积的步数大于1，则将损失除以梯度累积的步数
            # 通过将损失值除以梯度累积步数，可以得到每个梯度累积步骤的平均损失值
            # grad_acc_steps = params["gradient_accumulation_steps"] = 1
            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps

            # 累积训练损失
            tr_loss += loss.item()

            # 如果达到打印间隔，则打印平均损失
            if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
                logger.info(
                    "Step {} - epoch {} average loss: {}\n".format(
                        step,
                        epoch_idx,
                        tr_loss / (params["print_interval"] * grad_acc_steps),
                    )
                )
                tr_loss = 0

            # 反向传播，计算梯度
            loss.backward()

            # 如果达到梯度累积的步数，则执行梯度裁剪、优化器的更新和学习率调度器的更新
            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), params["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # 如果达到评估间隔，则进行评估
            if (step + 1) % (params["eval_interval"] * grad_acc_steps) == 0:
                logger.info("Evaluation on the development dataset")
                evaluate_(reranker, valid_dataloader, params, device=device, logger=logger,)
                model.train()
                logger.info("\n")

        logger.info("***** Saving fine - tuned model *****")
        # 构建当前轮次的输出文件夹路径
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        # 保存模型和分词器到指定的输出文件夹
        utils.save_model(model, tokenizer, epoch_output_folder_path)

        # 构建评估结果的输出文件路径
        output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")
        # 进行评估
        results = evaluate_(
            reranker, valid_dataloader, params, device=device, logger=logger,
        )

        # 构建最佳得分和最佳轮次的列表
        ls = [best_score, results["normalized_accuracy"]]
        li = [best_epoch_idx, epoch_idx]

        # 更新最佳得分和最佳轮次
        best_score = ls[np.argmax(ls)]
        best_epoch_idx = li[np.argmax(ls)]
        logger.info("\n")

    # 计算训练时间并写入文件
    execution_time = (time.time() - time_start) / 60
    utils.write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )
    logger.info("The training took {} minutes\n".format(execution_time))

    # save the best model in the parent_dir
    # 通过最佳轮次得到最佳模型的参数
    # 路径由模型输出路径（model_output_path）、"epoch_{best_epoch_idx}"和WEIGHTS_NAME组成
    logger.info("Best performance in epoch: {}".format(best_epoch_idx))
    params["path_to_model"] = os.path.join(
        model_output_path, 
        "epoch_{}".format(best_epoch_idx),
        WEIGHTS_NAME,
    )
    # 加载最佳模型
    reranker = load_biencoder(params)
    # 保存最佳模型
    utils.save_model(reranker.model, tokenizer, model_output_path)

    # # 如果evaluate参数为True，加载模型，然后调用evaluate函数进行评估
    # if params["evaluate"]:
    #     params["path_to_model"] = model_output_path
    #     evaluate(params, logger=logger)


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    parser.add_eval_args()

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
