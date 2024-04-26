# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import io
import sys
import json
import torch
import logging

import numpy as np

from collections import OrderedDict, defaultdict
from pytorch_transformers.modeling_utils import CONFIG_NAME, WEIGHTS_NAME
from tqdm import tqdm

from blink.candidate_ranking.bert_reranking import BertReranker
from blink.biencoder.biencoder import BiEncoderRanker


def new_read_dataset(dataset_name, preprocessed_json_data_parent_folder, debug=False):
    # 拼接生成文件名
    file_name = "/{}.json".format(dataset_name)
    txt_file_path = preprocessed_json_data_parent_folder + file_name
    kb_data_file_path = "/home/ubuntu/Desktop/BLINK-main/blink/biencoder/data/ccks2019/kb_data"
    #print(txt_file_path)
    #print(kb_data_file_path)
    # 用于保存每个subject_id对应的信息
    subject_id_with_info = defaultdict(dict)
    if not os.path.exists(preprocessed_json_data_parent_folder + '/subject_id_with_info.json'):
        subject_id_with_info_file = open(preprocessed_json_data_parent_folder + '/subject_id_with_info.json', 'w', encoding='utf-8')
        with open(kb_data_file_path, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
            total = len(lines) - 1
            for i, line in enumerate(lines):
                print(i, total)
                line = eval(line)
                subject_id_with_info[line['subject_id']] = line
        subject_id_with_info_file.write(json.dumps(subject_id_with_info, ensure_ascii=False))
    else:
        with open(preprocessed_json_data_parent_folder + '/subject_id_with_info.json', 'r', encoding='utf-8') as fp:
            subject_id_with_info = json.loads(fp.read())
    samples = []
    with open(txt_file_path, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for i, line in enumerate(lines):
            line = eval(line)
            mention_datas = line['mention_data']
            text = line['text']
            # if i <= 2:
            #     print(text)
            for mention_data in mention_datas:
                sample = []
                mention_text = mention_data['mention'].lower()
                start_pos = int(mention_data['offset'])
                end_pos = start_pos + int(len(mention_text))
                context_left = text[:start_pos].lower()
                context_right = text[end_pos:].lower()
                kb_id = mention_data['kb_id']
                description = ''
                if kb_id != 'NIL':
                    infos = subject_id_with_info[kb_id]
                    des = infos['data']
                    description = ''
                    if len(des) == 0:
                        description = "摘要，" + infos['subject'] + "无描述信息。"
                    else:
                        for kb_text in des:
                            str1 = kb_text['predicate']
                            str2 = str(kb_text['object'])
                            flag = str2.endswith("。") | str2.endswith(".")
                            if flag:
                                description = description + str1 + ':' + str2
                            else:
                                description = description + str1 + ':' + str2 + '。'
                    sample = {
                        "mention_text": mention_text,
                        "start_pos": start_pos,
                        "end_pos": end_pos,
                        "context_left": context_left,
                        "context_right": context_right,
                        "title": mention_text,
                        "text": description,
                        "kb_id": kb_id
                    }
                    # if i <= 2:
                    #     print(sample)
                    samples.append(sample)
                if debug and len(samples) >= 200:
                    break
    total = len(samples)  # 整理好的训练集的正负样本数量
    train_total = int(total * 0.7)  # 选取训练集数量
    train_examples = samples[:train_total]
    test_examples = samples[train_total:]
    return train_examples,test_examples


def filter_samples(samples, top_k, gold_key="gold_pos"):
    if top_k == None:
        return samples

    filtered_samples = [
        sample
        for sample in samples
        if sample[gold_key] > 0 and sample[gold_key] <= top_k
    ]
    return filtered_samples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def eval_precision_bm45_dataloader(dataloader, ks=[1, 5, 10], number_of_samples=None):
    label_ids = torch.cat([label_ids for _, _, _, label_ids, _ in dataloader])
    label_ids = label_ids + 1
    p = {}

    for k in ks:
        p[k] = 0

    for label in label_ids:
        if label > 0:
            for k in ks:
                if label <= k:
                    p[k] += 1

    for k in ks:
        if number_of_samples is None:
            p[k] /= len(label_ids)
        else:
            p[k] /= number_of_samples

    return p


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels), outputs == labels


def remove_module_from_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        name = "".join(key.split(".module"))
        new_state_dict[name] = value
    return new_state_dict


def save_model(model, tokenizer, output_dir):
    """Saves the model and the tokenizer used in the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_to_save = model.module if hasattr(model, "module") else model
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)


def get_logger(output_dir=None):
    if output_dir != None:
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(
                    "{}/log.txt".format(output_dir), mode="a", delay=False
                ),
                logging.StreamHandler(sys.stdout),
            ],
        )
    else:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    logger = logging.getLogger('Blink')
    logger.setLevel(10)
    return logger


def write_to_file(path, string, mode="w"):
    with open(path, mode) as writer:
        writer.write(string)


def get_reranker(parameters):
    return BertReranker(parameters)


def get_biencoder(parameters):
    return BiEncoderRanker(parameters)
