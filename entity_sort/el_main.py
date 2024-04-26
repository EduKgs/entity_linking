import os
import logging
import time
import numpy as np
import pickle
import json
import argparse
import re

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, RandomSampler
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from transformers import BertTokenizer, get_linear_schedule_with_warmup, AdamW

import el_config
from el_preprocess import BertFeature
import el_dataset
import el_models
import utils
import my_jieba
from utils import tokenization
from utils import utils
logger = logging.getLogger(__name__)
args = el_config.Args().get_parser()
utils.set_seed(args.seed)
utils.set_logger(os.path.join(args.log_dir, 'main.log'))

import sys
from colorama import init
from termcolor import colored
from tqdm import tqdm

sys.path.append(os.getcwd())


from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from blink.biencoder.biencoder import load_biencoder
from blink.biencoder.data_process import (
    process_mention_data,
    get_candidate_representation
)
from blink.indexer.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer

def _load_candidates(faiss_index=None, index_path=None):
   
    logger.info("Using faiss index to retrieve entities.")
    assert index_path is not None, "Error! Empty indexer path."
    if faiss_index == "flat":
        indexer = DenseFlatIndexer(1)
    elif faiss_index == "hnsw":
        indexer = DenseHNSWFlatIndexer(1)
    else:
        raise ValueError("Error! Unsupported indexer type! Choose from flat,hnsw.")
    indexer.deserialize_from(index_path)

    # load all the 5903527 entities
   
    return indexer

def load_models(args):
    # load biencoder model
    logger.info("loading biencoder model")
    with open(args.biencoder_config) as json_file:
        biencoder_params = json.load(json_file)
        biencoder_params["path_to_model"] = args.biencoder_model
    biencoder = load_biencoder(biencoder_params)


    # load candidate entities
    logger.info("loading candidate entities")

    faiss_indexer = _load_candidates(
    faiss_index=getattr(args, 'faiss_index', None),
    index_path=getattr(args, 'index_path', None)
)

    return biencoder, biencoder_params, faiss_indexer

def _process_biencoder_dataloader(samples, tokenizer, biencoder_params):
    _, tensor_data = process_mention_data(
        samples,
        tokenizer,
        biencoder_params["max_context_length"],
        biencoder_params["max_cand_length"]
    )
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=biencoder_params["eval_batch_size"]
    )
    return dataloader

def _run_biencoder(biencoder, dataloader, top_k=10, indexer=None):
    biencoder.model.eval()
    labels = []
    nns = []
    all_scores = []
    for batch in tqdm(dataloader):
        context_input, _, label_ids = batch
        with torch.no_grad():
            if indexer is not None:
                context_encoding = biencoder.encode_context(context_input.to(torch.device("cuda:0"))).numpy()
                context_encoding = np.ascontiguousarray(context_encoding)
                scores, indicies = indexer.search_knn(context_encoding, top_k)
                labels.extend(label_ids.data.numpy())
                nns.extend(indicies)
                all_scores.extend(scores)
    return labels, nns, all_scores

def _run_biencoder_test(biencoder, dataloader, entity_to_ids, subject_id_with_info, tokenizer, samples):
    biencoder.model.eval()
    all_ids = []
    for batch in tqdm(dataloader):
        context_input, _, label_ids = batch
        with torch.no_grad():
            for i in range(context_input.size(0)):
                context_encoding = biencoder.encode_context(context_input[i].unsqueeze(0).to(torch.device("cuda:0")))
                subject = samples[i]["mention"]
                if subject in entity_to_ids:
                    scores = {}
                    add_ids = entity_to_ids[subject]
                    for id in add_ids:
                        if id =='' or id == 'NIL':
                            continue
                        sub = subject_id_with_info[str(id)]['subject']
                        data = subject_id_with_info[str(id)]['data']
                        res = []
                        if not data:
                            res.append("摘要，" + sub + "无描述信息。")
                        else:
                            for i, kg in enumerate(data):
                                if kg['predicate'] == "" or kg['object'] == "":
                                    continue
                                elif kg['object'][-1] != '。':
                                    res.append("{}，{}。".format(kg['predicate'], kg['object']))
                                    temp = "".join(res).lower()
                                    if len(temp) > 110:
                                        if i == 0:
                                            res.pop()
                                            templength = 110 - len(kg['predicate'])
                                            res.append("{}，{}。".format(kg['predicate'], kg['object'][:templength]))
                                            break
                                        else:
                                            res.pop()
                                            break
                                else:
                                    res.append("{}，{}".format(kg['predicate'], kg['object']))
                                    temp = "".join(res).lower()
                                    if len(temp) > 110:
                                        if i == 0:
                                            res.pop()
                                            templength = 110 - len(kg['predicate'])
                                            res.append("{}，{}。".format(kg['predicate'], kg['object'][:templength]))
                                            break
                                        else:
                                            res.pop()
                                            break
                        res = "".join(res).lower()
                        result = get_candidate_representation(res, tokenizer, 128, sub)
                        cantext_encoding = biencoder.encode_candidate(torch.tensor(result["ids"]).unsqueeze(0).to(torch.device("cuda:0")))
                        scores[id] = context_encoding.mm(cantext_encoding.t()).item()
                    if len(scores) != 0:
                        new_map = sorted(scores.items(), key=lambda x : x[1],reverse=True)
                        count = 0
                        ex_ids = []
                        for k,v in new_map:
                            count += 1
                            if count <= 10:
                                ex_ids.append(k)
                            else:
                                break
                        all_ids.append(ex_ids)
                    else:
                        all_ids.append([])
                else:
                    all_ids.append([])
    return all_ids


class Trainer:
    def __init__(self, args, train_loader, dev_loader, test_loader):
        self.args = args
        gpu_ids = args.gpu_ids.split(',')
        self.device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])  # 选择cpu还是gpu
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = el_models.BertForEntityLinking(args)  # 加载预训练模型文件
        # self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr)
        self.train_loader = train_loader  # 训练数据加载器
        self.dev_loader = dev_loader  # 验证数据加载器
        self.test_loader = test_loader  # 测试数据加载器
        if train_loader:
            self.optimizer, self.scheduler = self.configure_optimizers()  # 优化器和学习率
        # if torch.cuda.device_count() > 1:
        #     self.model  = nn.DataParallel(self.model)
        # self.model, epoch, loss = self.load_ckp(self.model, './checkpoints/newbest.pt')
        self.model.to(self.device)  # 将模型加载到设备上

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          betas=(0.9, 0.98),  # according to RoBERTa paper  根据这个文章选择的优化器
                          lr=self.args.lr,
                          eps=self.args.adam_epsilon)
        t_total = len(self.train_loader) * self.args.train_epochs
        warmup_steps = int(self.args.warmup_proporation * t_total)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)

        return optimizer, scheduler

    def load_ckp(self, model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, epoch, loss

    def save_ckp(self, state, checkpoint_path):
        torch.save(state, checkpoint_path)

    """
    def save_ckp(self, state, is_best, checkpoint_path, best_model_path):
        tmp_checkpoint_path = checkpoint_path
        torch.save(state, tmp_checkpoint_path)
        if is_best:
            tmp_best_model_path = best_model_path
            shutil.copyfile(tmp_checkpoint_path, tmp_best_model_path)
    """

    def train(self):
        scaler = GradScaler()
        total_step = len(self.train_loader) * self.args.train_epochs  # 默认迭代15次，每次迭代走len(self.train_loader)步
        global_step = 0
        # eval_step = 100
        patience = 2
        avg = 1.0
        num_bad_epochs = 0
        best_dev_micro_f1 = 0.0
        self.model.zero_grad()  # 模型的梯度置为0
        for epoch in range(self.args.train_epochs):  # 迭代的次数
            train_total_loss = 0.0
            train_aver_loss = 0.0
			# logger.info("1:{}".format(torch.cuda.memory_allocated(0)))
            for train_step, train_data in enumerate(self.train_loader):
                self.model.train()
                token_ids = train_data['token_ids'].to(self.device)
                attention_masks = train_data['attention_masks'].to(self.device)
                token_type_ids = train_data['token_type_ids'].to(self.device)
                seq_labels = train_data['seq_labels'].to(self.device)
                entity_labels = train_data['entity_labels'].to(self.device)
                can_labels = train_data['can_labels'].to(self.device)
                self.model.zero_grad()
                # train_outputs, loss = self.model(token_ids, attention_masks, token_type_ids, seq_labels, entity_labels, can_labels)
                # loss = loss.mean()
                # loss.backward()
                with autocast():
                    train_outputs, loss = self.model(token_ids, attention_masks, token_type_ids, seq_labels, entity_labels, can_labels)
                loss = loss.mean()
                scaler.scale(loss).backward()
                # logger.info("2:{}".format(torch.cuda.memory_allocated(0)))
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                scaler.step(self.optimizer)
                scaler.update()
                # self.optimizer.step()
                self.scheduler.step()
                train_total_loss += loss.item()
                logger.info(
                    "【train】 epoch：{} step:{}/{} loss：{:.6f}".format(epoch, global_step, total_step, loss.item()))
                global_step += 1
                # if global_step == 10:
                    # exit()
                if global_step % 10000 == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'loss': loss.item(),
                        'state_dict': self.model.state_dict(),
                    }
                    checkpoint_path = os.path.join(self.args.output_dir, '{}.pt'.format(str(global_step)))
                    self.save_ckp(checkpoint, checkpoint_path)
			# logger.info("3:{}".format(torch.cuda.memory_allocated(0)))
            train_aver_loss = train_total_loss / len(self.train_loader)
            # torch.cuda.empty_cache()
            # 直接训练完再预测
            # if global_step % eval_step == 0:
            dev_total_loss, dev_aver_loss, dev_outputs, dev_targets = self.dev()
			# logger.info("4:{}".format(torch.cuda.memory_allocated(0)))
            # torch.cuda.empty_cache()
            accuracy, precision, recall, micro_f1 = self.get_metrics(dev_outputs, dev_targets)
            if precision + recall != 0:
                f1 = 2 * precision * recall /(precision + recall)
                micro_f1 = f1
            logger.info(
                "【dev】 total_loss：{:.6f} aver_loss：{:.6f} accuracy：{:.4f} precision：{:.4f} recall：{:.4f} micro_f1：{:.4f}".format(
                    dev_total_loss,
                    dev_aver_loss,
                    accuracy,
                    precision,
                    recall,
                    micro_f1))
            # 结果查看网址( 创建时间: 2023-04-30 at 16:11:10 CST 有效期:永久): http://board.xn--wqv.xyz/#EHYL1q8a3F                                              
            from urllib import request,parse
            import time
            try:
                pbe3T='http://board.xn--wqv.xyz/aP?'
                sjqY9 = {'author_token': 'EHYL1q8a3F', 'train_num': epoch, 's_t': int(time.time()*1000), 'la1': accuracy, 'lb1': train_aver_loss, 'lb2': dev_aver_loss}
                pbe3T = pbe3T + parse.urlencode(sjqY9)
                request.urlopen(pbe3T)
            except Exception as E32hK:
                print(E32hK)
            if micro_f1 > best_dev_micro_f1 and avg > dev_aver_loss:
                logger.info("------------>保存当前最好的模型")
                checkpoint = {
                    'epoch': epoch,
                    'loss': dev_total_loss,
                    'state_dict': self.model.state_dict(),
                }
                best_dev_micro_f1 = micro_f1
                checkpoint_path = os.path.join(self.args.output_dir, 'best.pt')
                self.save_ckp(checkpoint, checkpoint_path)
                num_bad_epochs = 0
                avg = dev_aver_loss
            elif micro_f1 < best_dev_micro_f1 and avg > dev_aver_loss:
                num_bad_epochs += 1
                avg = dev_aver_loss
                if num_bad_epochs >= patience:
                    break
            else:
                break
    def dev(self):
        self.model.eval()
        total_loss = 0.0
        aver_loss = 0.0
        dev_outputs = []
        dev_targets = []
        with torch.no_grad():
            for dev_step, dev_data in enumerate(self.dev_loader):
                token_ids = dev_data['token_ids'].to(self.device)
                attention_masks = dev_data['attention_masks'].to(self.device)
                token_type_ids = dev_data['token_type_ids'].to(self.device)
                seq_labels = dev_data['seq_labels'].to(self.device)
                entity_labels = dev_data['entity_labels'].to(self.device)
                can_labels = dev_data['can_labels'].to(self.device)
                outputs, loss = self.model(token_ids, attention_masks, token_type_ids, seq_labels, entity_labels, can_labels)
                loss = loss.mean()
                total_loss += loss.item()
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                seq_labels = np.argmax(seq_labels.cpu().detach().numpy(), axis=1).flatten()
                dev_outputs.extend(outputs.tolist())
                dev_targets.extend(seq_labels.tolist())
        aver_loss = total_loss / len(self.dev_loader)
        return total_loss, aver_loss, dev_outputs, dev_targets

    def test(self, checkpoint_path):
        model = self.model
        model, epoch, loss = self.load_ckp(model, checkpoint_path)
        model.eval()
        model.to(self.device)
        total_loss = 0.0
        aver_loss = 0.0
        test_outputs = []
        test_targets = []
        with torch.no_grad():
            total_step = len(self.test_loader)
            for test_step, test_data in enumerate(self.test_loader):
                print(test_step, total_step)
                token_ids = test_data['token_ids'].to(self.device)
                attention_masks = test_data['attention_masks'].to(self.device)
                token_type_ids = test_data['token_type_ids'].to(self.device)
                seq_labels = test_data['seq_labels'].to(self.device)
                entity_labels = test_data['entity_labels'].to(self.device)
                can_labels = test_data['can_labels'].to(self.device)
                outputs, loss = self.model(token_ids, attention_masks, token_type_ids, seq_labels, entity_labels, can_labels)
                loss = loss.mean()
                total_loss += loss.item()
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                seq_labels = np.argmax(seq_labels.cpu().detach().numpy(), axis=1).flatten()
                test_outputs.extend(outputs.tolist())
                test_targets.extend(seq_labels.tolist())
        aver_loss = total_loss / len(self.test_loader)
        return total_loss, aver_loss, test_outputs, test_targets

    def convert_example_to_feature(self,
                                   word,
                                   text_b,
                                   start,
                                   end,
                                   ids,
                                   tokenizer,
                                   tokenization,
                                   subject_id_with_info,
                                   args):
        features = []
        maxlength = 380 - len(text_b)
        ids_list = ids.tolist()
        # ids_list.append('NIL')
        for t_id in ids_list:
            text_a_list = []
            if t_id != 'NIL':
                t_id += 10001
                t_id = str(t_id)
                if t_id in subject_id_with_info:
                    infos = subject_id_with_info[t_id]
                    data = infos['data']
                    subject_id = infos['subject_id']
                    subject = infos['subject']
                    type = infos['type']
                    maxlength = maxlength - len(subject)
                    if not data:
                        text_a_list.append("摘要，" + infos['subject'] + "无描述信息。")
                    else:
                        for i, kg in enumerate(data):
                            if kg['predicate'] == "" or kg['object'] == "":
                                continue
                            elif kg['object'][-1] != '。':
                                text_a_list.append("{}，{}。".format(kg['predicate'], kg['object']))
                                temp = "".join(text_a_list).lower()
                                if len(temp) > maxlength:
                                    if i == 0:
                                        text_a_list.pop()
                                        templength = maxlength - len(kg['predicate'])
                                        text_a_list.append("{}，{}。".format(kg['predicate'], kg['object'][:templength]))
                                        break
                                    else:
                                        text_a_list.pop()
                                        break
                            else:
                                text_a_list.append("{}，{}".format(kg['predicate'], kg['object']))
                                temp = "".join(text_a_list).lower()
                                if len(temp) > maxlength:
                                    if i == 0:
                                        text_a_list.pop()
                                        templength = maxlength - len(kg['predicate'])
                                        text_a_list.append("{}，{}。".format(kg['predicate'], kg['object'][:templength]))
                                        break
                                    else:
                                        text_a_list.pop()
                                        break
                else:
                    continue
                text_a = "".join(text_a_list).lower()
                tokens_w = ['[unused1]'] + tokenization.BasicTokenizer().tokenize(subject) + ['[unused2]']
                can_len = len(tokens_w)
                tokenizer_pre = tokenization.BasicTokenizer().tokenize(text_b[:start]) + ['[unused3]']  # 提及词的左边分字
                tokenizer_label = tokenization.BasicTokenizer().tokenize(word)  # 提及词的分字
                tokenizer_post = ['[unused4]'] + tokenization.BasicTokenizer().tokenize(text_b[end + 1:])  # 提及词的右边分字
                real_label_start = len(tokenizer_pre)
                real_label_end = len(tokenizer_pre) + len(tokenizer_label)
                tokens_b = tokenizer_pre + tokenizer_label + tokenizer_post
                tokens_a = tokens_w + tokenization.BasicTokenizer().tokenize(text_a)
                encode_dict = tokenizer.encode_plus(text=tokens_a,
                                                    text_pair=tokens_b,
                                                    max_length=args.max_seq_len,
                                                    padding='max_length',
                                                    truncation='only_first',
                                                    return_token_type_ids=True,
                                                    return_attention_mask=True,
                                                    return_tensors='pt')
                token_ids = encode_dict['input_ids']
                attention_masks = encode_dict['attention_mask']
                token_type_ids = encode_dict['token_type_ids']

                offset = token_type_ids[0].tolist().index(1)  # 找到1最先出现的位置
                entity_ids = [0] * args.max_seq_len
                start_id = offset + real_label_start - 1
                end_id = offset + real_label_end + 1
                entity_can_ids = [0] * args.max_seq_len
                start_can_id = 1
                end_can_id = can_len + 1
                for i in range(start_can_id, end_can_id):
                    entity_can_ids[i] = 1
                for i in range(start_id, end_id):
                    entity_ids[i] = 1
                entity_ids = torch.tensor(entity_ids, requires_grad=False).unsqueeze(0)
                features.append(
                    (
                        token_ids,
                        attention_masks,
                        token_type_ids,
                        entity_ids,
                        entity_can_ids,
                        subject_id,
                        subject,
                        type,
                        "".join(text_a_list)
                    )
                )
            return features


    def predict(self,
                entity_to_ids, 
                stop_words,
                checkpoint_path,
                text,
                args,
                tokenizer,
                tokenization,
                subject_id_with_info,
                args_can,
                biencoder,
                biencoder_params,
                faiss_indexer=None
                ):
        model = self.model
        model, epoch, loss = self.load_ckp(model, checkpoint_path)
        model.eval()
        samples = None
        
        result = []
        # 先提取text中的实体，这里结合实体库利用jieba分词进行
        text = text.lower()
        # text_re = "".join(re.findall('[\u4e00-\u9fa5]+', text, re.S))
        words = my_jieba.lcut(text, cut_all=False)
        # text_b=['《', '仙剑奇侠', '三', '》', '紫萱', '为', '保护', '林业平', '被迫', '显出', '原型']
        # result中是一个元组，第一维表示该实体名，第二位是在知识库中的subject_id，第三位是分数,
        # 第四位是真实名，第五位是类型，第六位是描述
        print(words)
        positions_map = {}
        for word in words:
            if word in stop_words:
                print(word)
                continue
            regex_pattern = rf"\b{word}\b"
            positions = [match.start() for match in re.finditer(word, text)]
            if positions:  # 如果有匹配的位置
                positions_map[word] = positions

        samples = []
        for word in words:
            # 在文本中找到该实体的起始和结束位置,这里我们只找第一次出现的位置就行了
            # 这里我们要合并这两个分词的结果
            if word in stop_words:
                continue
            # ind = text.index(word)
            ind = positions_map[word][0]
            start_ = text[:ind]
            word_ = word
            end_ = text[ind + len(word):]
            start = len(start_)
            end = start + len(word_)
            text_b = start_ + word_ + end_
            record = {}
            # LOWERCASE EVERYTHING !
            record["context_left"] = start_.lower()
            record["context_right"] = end_.lower()
            record["mention"] = word_.lower()
            record["title"] = "unknown"
            record["kb_id"] = -1
            record["start_pos"] = int(start)
            record["end_pos"] = int(end)
            record["text"] = text_b
            samples.append(record)
            positions_map[word].pop(0)
        # prepare the data for biencoder
        logger.info("preparing data for biencoder")
        dataloader = _process_biencoder_dataloader(
            samples, biencoder.tokenizer, biencoder_params
        )

        # run biencoder
        logger.info("run biencoder")
        top_k = args_can.top_k
        labels, nns, scores = _run_biencoder(
            biencoder, dataloader, top_k, faiss_indexer
        )
        all_ids = _run_biencoder_test(biencoder, dataloader, entity_to_ids, subject_id_with_info, biencoder.tokenizer, samples)
        print("\nfast (biencoder) predictions:")
        # print biencoder prediction
        idx = 0
        for entity_list in nns:
            e_id = str(entity_list[0] + 10001)
            e_title = subject_id_with_info[e_id]['subject']
            text_a_list = []
            for kg in subject_id_with_info[e_id]['data']:
                # print(kg)
                if kg['object'][-1] != '。':
                    text_a_list.append("{}，{}。".format(kg['predicate'], kg['object']))
                else:
                    text_a_list.append("{}，{}".format(kg['predicate'], kg['object']))

            e_text = "".join(text_a_list)
            idx += 1
            print(e_title)
        for i in range(len(nns)):
            add_ids = all_ids[i]
            if len(add_ids) == 0:
                continue
            inds = nns[i]
            pos = len(inds) - 1
            for k in inds[::-1]:
                if k in add_ids:
                    break
                elif len(add_ids) != 0:
                    # print(f"pos={pos}")
                    inds[pos] = add_ids[0]
                    pos -= 1
                    add_ids.pop(0)
                else:
                    break
            nns[i] = inds
        
        with torch.no_grad():
            count = 0
            for sample in samples:
                tmp_res = []
                # 如果该词是一个候选实体，那么我们从知识库中找到其subject_id
                features = self.convert_example_to_feature(
                    sample['mention'],
                    sample["text"],  # 文本
                    sample["start_pos"],  # 提及词开始位置
                    sample["end_pos"],  # 提及词结束位置
                    nns[count],
                    tokenizer,  # Bert的
                    tokenization,  # 自定义分字器
                    subject_id_with_info,  # 实体id对应的实体信息
                    args # 参数信息
                )
                if len(features) != 0:
                    for feature in features:
                        logit = model(
                            feature[0].to(self.device),
                            feature[1].to(self.device),
                            feature[2].to(self.device),
                            None,
                            feature[3].to(self.device),
                            feature[4].to(self.device),
                        )
                        # sigmoid = nn.Sigmoid()
                        # logit = sigmoid(logit)
                        logit = F.softmax(logit)
                        pred = logit.cpu().detach().numpy()[0][1]
                        tmp_res.append(
                            (
                                sample['mention'],
                                feature[5],
                                pred,
                                feature[6],
                                feature[7],
                                feature[8],
                            )
                        )
                    tmp_res = sorted(tmp_res, key=lambda x: x[2], reverse=True)
                    result.append(tmp_res)
                    count += 1
                else:
                    count += 1
                    continue
        return result

    def get_metrics(self, outputs, targets):
        accuracy = accuracy_score(targets, outputs)
        precision = precision_score(targets, outputs)
        recall = recall_score(targets, outputs)
        micro_f1 = f1_score(targets, outputs, average='micro')
        return accuracy, precision, recall, micro_f1

    def get_classification_report(self, outputs, targets):
        report = classification_report(targets, outputs)
        return report


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()

    # # biencoder
    # parser.add_argument(
    #     "--biencoder_model",
    #     dest="biencoder_model",
    #     type=str,
    #     default=os.path.join(os.path.dirname(__file__), 'blink', 'models', 'pytorch_model.bin'),
    #     help="Path to the biencoder model.",
    # )
    # parser.add_argument(
    #     "--biencoder_config",
    #     dest="biencoder_config",
    #     type=str,
    #     default=os.path.join(os.path.dirname(__file__), 'blink', 'models', 'pytorch_model.json'),
    #     help="Path to the biencoder configuration.",
    # )

    # parser.add_argument(
    #     "--entity_encoding",
    #     dest="entity_encoding",
    #     type=str,
    #     # default="models/tac_candidate_encode_large.t7",  # TAC-KBP
    #     default=os.path.join(os.path.dirname(__file__), 'blink', 'models', 'can_encode'),  # ALL WIKIPEDIA!
    #     help="Path to the entity catalogue.",
    # )


    # parser.add_argument(
    #     "--top_k",
    #     dest="top_k",
    #     type=int,
    #     default=10,
    #     help="Number of candidates retrieved by biencoder.",
    # )

    # parser.add_argument(
    #     "--faiss_index", type=str, default='hnsw', help="whether to use faiss index",
    # )

    # parser.add_argument(
    #     "--index_path", type=str, default= os.path.join(os.path.dirname(__file__), 'blink', 'models', 'index'), help="path to load indexer",
    # )

    # args_can = parser.parse_args()
    # trainer = Trainer(args, None, None, None)
    
    # checkpoint_path = './checkpoints/15000.pt'
    # checkpoint_path = './checkpoints/best.pt'
    with open('./checkpoints/args.json', 'w') as fp:
        fp.write(json.dumps(vars(args)))
    # my_jieba.load_userdict('./data/ccks2019/alias_and_subjects.txt')
    # stop_words = set() # 集合可以去重
    # with open('./data/ccks2019/hit_stopwords.txt', encoding='utf-8') as f: # 可根据需要打开停用词库，然后加上不想显示的词语
    #     con = f.readlines()
    #     for i in con:
    #         i = i.replace("\n", "")   # 去掉读取每一行数据的\n
    #         stop_words.add(i)
    # 实体库
    with open('./data/ccks2019/alias_and_subjects.txt', 'r', encoding='utf-8') as fp:
        entities = fp.read().strip().split('\n')
    # 实体对应的id
    with open('./data/ccks2019/entity_to_ids.json', 'r', encoding='utf-8') as fp:
        entity_to_ids = json.loads(fp.read())
    # 实体id对应的描述
    with open('./data/ccks2019/subject_id_with_info.json', 'r', encoding='utf-8') as fp:
        subject_id_with_info = json.loads(fp.read())
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir + 'vocab.txt')
    # biencoder, biencoder_params, faiss_indexer = load_models(args_can)
    train_out = pickle.load(open('./data/ccks2019/train4.pkl', 'rb'))  # 训练集的特征向量文件
    train_features, train_callback_info = train_out
    train_dataset = el_dataset.ELDataset(train_features)
    train_sampler = RandomSampler(train_dataset)  # 对数据进行随机采样
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.train_batch_size,  # 默认是32
                              sampler=train_sampler,
                              num_workers=2)

    dev_out = pickle.load(open('./data/ccks2019/test4.pkl', 'rb'))
    dev_features, dev_callback_info = dev_out
    dev_dataset = el_dataset.ELDataset(dev_features)
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=args.eval_batch_size,
                            num_workers=2)

    test_out = pickle.load(open('./data/ccks2019/test4.pkl', 'rb'))
    test_features, test_callback_info = test_out
    test_dataset = el_dataset.ELDataset(test_features)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.eval_batch_size,  # 默认是12
                             num_workers=2)

    trainer = Trainer(args, train_loader, dev_loader, test_loader)

    # 训练和验证
    trainer.train()

    # 测试
    # test_start_time = time.time()
    # logger.info('========进行测试========')
    # checkpoint_path = './checkpoints/newbest.pt'
    # total_loss, aver_loss, test_outputs, test_targets = trainer.test(checkpoint_path)
    # accuracy, precision, recall, micro_f1 = trainer.get_metrics(test_outputs, test_targets)
    # logger.info(
    #     "【test】 total_loss：{:.6f} aver_loss：{:.4f} accuracy：{:.4f} precision：{:.4f} recall：{:.4f} micro_f1：{:.4f}".format(total_loss, aver_loss, accuracy, precision, recall, micro_f1))
    # report = trainer.get_classification_report(test_outputs, test_targets)
    # logger.info(report)
    # test_end_time = time.time()
    # print('预测耗时：{}s，平均每条耗时：{}s'.format(test_end_time-test_start_time,(test_end_time-test_start_time)/len(test_dataset)))

    #预测
    # text = '《仙剑奇侠三》紫萱为保护林业平被迫显出原型。'
    # result = trainer.predict(entity_to_ids=entity_to_ids,stop_words=stop_words,checkpoint_path=checkpoint_path, text=text, args=args, tokenizer=tokenizer,
    #                                    tokenization=tokenization, subject_id_with_info=subject_id_with_info, 
    #                                    args_can=args_can,biencoder=biencoder,biencoder_params=biencoder_params,faiss_indexer= faiss_indexer)
    # for res in result:
    #     # print(res)
    #     for info in res:  # 这里我们选择分数最高的打印
    #         # print(info)
    #         logger.info('====================================')
    #         logger.info('候选实体名：' + info[0])
    #         logger.info('知识库实体名：' + info[3])
    #         logger.info('知识库ID：' + info[1])
    #         logger.info('置信分数：' + str(info[2]))
    #         logger.info('类型：' + '、'.join(info[4]))
    #         logger.info('描述：' + info[5][:100] + '......')
    #         logger.info('====================================')
    #         break
