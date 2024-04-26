# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import torch
import json
import random
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset

from pytorch_transformers.tokenization_bert import BertTokenizer

from blink.biencoder.zeshel_utils import world_to_id
from blink.common.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG
from blink.utils import tokenization

def select_field(data, key1, key2=None):
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]


def get_context_representation(
    sample,
    tokenizer,
    max_seq_length,
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
):
    mention_tokens = []

    # 根据提及键从样本中提取出提及文本
    if sample["mention_text"] and len(sample["mention_text"]) > 0:
        # 使用tokenizer将其分割为单词列表
        mention_tokens = tokenization.BasicTokenizer().tokenize(sample["mention_text"])
        # 实体起始标记和实体结束标记添加到单词列表的开头和结尾
        mention_tokens = [ent_start_token] + mention_tokens + [ent_end_token]

    # 提取提及左侧和右侧文本
    # 将提及左侧和右侧文本分割为单词列表
    context_left = sample["context_left"]
    context_right = sample["context_right"]
    context_left = tokenization.BasicTokenizer().tokenize(context_left)
    context_right = tokenization.BasicTokenizer().tokenize(context_right)

    # 根据最大长度，计算左侧和右侧文本可以使用的单词数量并进行调整
    # 计算出的提及左侧和右侧单词列表长度
    left_quota = (max_seq_length - len(mention_tokens)) // 2 - 1  # -1 为减去句子开始/结束标记
    right_quota = max_seq_length - len(mention_tokens) - left_quota - 2  # -2 为减去句子开始和结束标记
    # 实际提及左侧和右侧单词列表长度
    left_add = len(context_left)
    right_add = len(context_right)
    # 如果提及左侧有剩余空间并且右侧空间不够，将左侧剩余大小增加给右侧
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    # 如果提及左侧空间不够并且右侧有剩余空间，将右侧剩余大小增加给左侧
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add

    # 拼接生成上下文的token
    context_tokens = (
            context_left[-left_quota:] + mention_tokens + context_right[:right_quota]
    )
    context_tokens = ["[CLS]"] + context_tokens + ["[SEP]"]

    # 将分词结果转换为数字序列
    input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    # context_tokens
    return {
        "tokens": context_tokens,
        "ids": input_ids,
    }



def get_candidate_representation(
    candidate_desc, 
    tokenizer, 
    max_seq_length, 
    candidate_title=None,
    title_tag=ENT_TITLE_TAG,
):
    # 得到CLS和SEP的token
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    # 得到候选实体描述的token
    cand_tokens = tokenization.BasicTokenizer().tokenize(str(candidate_desc))
    # 判断是否存在候选实体标题
    if candidate_title is not None:
        # 如果存在，则将标题分割为单词列表，
        title_tokens = tokenization.BasicTokenizer().tokenize(str(candidate_title))
        # 将标题添加到候选实体描述的前面，添加一个特殊的标题标记
        cand_tokens = title_tokens + [title_tag] + cand_tokens
        # cand_tokens = title_tokens + ['。'] + cand_tokens
    # 根据最大长度进行调整
    cand_tokens = cand_tokens[: max_seq_length - 2]  # -2 为cls_token和sep_token
    # 拼接生成候选实体的token
    cand_tokens = [cls_token] + cand_tokens + [sep_token]
    # 将分词结果转换为数字序列
    input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": cand_tokens,
        "ids": input_ids,
    }

def process_mention_data(
    samples,
    tokenizer,
    max_context_length,
    max_cand_length,
    silent,
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    title_token=ENT_TITLE_TAG,
    debug=False,
    logger=None,
):
    processed_samples = []

    # 根据debug参数的值，判断是否只处理前200个提及样本
    # 如果debug为True，则只处理前200个提及样本
    if debug:
        samples = samples[:200]

    # 根据silent参数的值，判断是否在处理过程中显示进度条
    # 如果silent为True，则不显示进度条，直接将samples列表赋值给iter_
    # 否则，使用tqdm函数创建一个进度条，并将samples列表作为参数传递给iter_
    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples)
    with open('/media/ubuntu/A4CAA455CAA42610/BLINK-main/blink/biencoder/data/ccks2019/subject_id_with_info.json','r',encoding = 'utf-8') as fp:
        id2text = json.loads(fp.read())
    with open('/media/ubuntu/A4CAA455CAA42610/BLINK-main/blink/biencoder/data/ccks2019/id_sim.json','r',encoding = 'utf-8') as fp:
        id2_sim = json.loads(fp.read())
    with open('/media/ubuntu/A4CAA455CAA42610/BLINK-main/blink/biencoder/data/ccks2019/wikipedia_id2local_id.json','r',encoding = 'utf-8') as fp:
        wikipedia_id2local_id = json.loads(fp.read())
    with open('/media/ubuntu/A4CAA455CAA42610/BLINK-main/blink/biencoder/data/ccks2019/local_id2wikipedia_id.json','r',encoding = 'utf-8') as fp:
        local_id2wikipedia_id= json.loads(fp.read())
    # 遍历每个样本
    for idx, sample in enumerate(iter_):
        # 得到上下文的tokens和ids
        context_tokens = get_context_representation(
            sample, tokenizer, max_context_length, ent_start_token, ent_end_token,
        )

        title = sample["title"]
        text = sample["text"]
        # 得到候选实体的tokens和ids
        label_tokens = get_candidate_representation(
            text, tokenizer, max_cand_length, title,
        )
        entity_idx = int(wikipedia_id2local_id[str(sample["kb_id"])])

        id_ = id2_sim[str(entity_idx)][0]
        id_ = local_id2wikipedia_id[str(id_)]
        # if sample["mention_text"] not in id2_sim.keys():
        #     continue
        # for id in id2_sim[sample["mention_text"]]:
        #     if id != sample["kb_id"]:
        #         id_ = id
        #         break
        # if id_ not in id2text.keys():
        #     continue
        infos =  id2text[str(id_)]
        title_f = infos['subject']
        description = ''
        if len(infos['data']) == 0:
            description = "摘要，" + title_f + "无描述信息。"
        else:
            for data in infos['data']:
                str1 = data['predicate']
                str2 = data['object']
                flag = str2.endswith("。") | str2.endswith(".")
                if flag:
                    description = description + str1 + ':' + str2
                else:
                    description = description + str1 + ':' + str2 + '。'
        
        label_tokens_f = get_candidate_representation(
        description[:100], tokenizer, max_cand_length, title_f
    )
        record = {
                "context": context_tokens,
                "cand_A": label_tokens,
                "cand_B": label_tokens_f
            }
        processed_samples.append(record)

    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"), dtype=torch.long,
    )
    cand_vecs_A = torch.tensor(
        select_field(processed_samples, "cand_A", "ids"), dtype=torch.long,
    )
    cand_vecs_B = torch.tensor(
        select_field(processed_samples, "cand_B", "ids"), dtype=torch.long,
    )

    data = {
        "context_vecs": context_vecs,
        "cand_vecs_A": cand_vecs_A,
        "cand_vecs_B": cand_vecs_B
    }
    tensor_data = TensorDataset(context_vecs, cand_vecs_A,cand_vecs_B)
    return data, tensor_data


def process_mention_data_(
   samples,
    tokenizer,
    max_context_length,
    max_cand_length,
    silent,
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    title_token=ENT_TITLE_TAG,
    debug=False,
    logger=None,
):
    processed_samples = []

    # 根据debug参数的值，判断是否只处理前200个提及样本
    # 如果debug为True，则只处理前200个提及样本
    if debug:
        samples = samples[:200]

    # 根据silent参数的值，判断是否在处理过程中显示进度条
    # 如果silent为True，则不显示进度条，直接将samples列表赋值给iter_
    # 否则，使用tqdm函数创建一个进度条，并将samples列表作为参数传递给iter_
    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples)
    with open('/media/ubuntu/A4CAA455CAA42610/BLINK-main/blink/biencoder/data/ccks2019/subject_id_with_info.json','r',encoding = 'utf-8') as fp:
        id2text = json.loads(fp.read())
    with open('/media/ubuntu/A4CAA455CAA42610/BLINK-main/blink/biencoder/data/ccks2019/id_sim.json','r',encoding = 'utf-8') as fp:
        id2_sim = json.loads(fp.read())
    with open('/media/ubuntu/A4CAA455CAA42610/BLINK-main/blink/biencoder/data/ccks2019/wikipedia_id2local_id.json','r',encoding = 'utf-8') as fp:
        wikipedia_id2local_id = json.loads(fp.read())
    with open('/media/ubuntu/A4CAA455CAA42610/BLINK-main/blink/biencoder/data/ccks2019/local_id2wikipedia_id.json','r',encoding = 'utf-8') as fp:
        local_id2wikipedia_id= json.loads(fp.read())

    # 遍历每个样本
    for idx, sample in enumerate(iter_):
        # 得到上下文的tokens和ids
        context_tokens = get_context_representation(
            sample, tokenizer, max_context_length, ent_start_token, ent_end_token,
        )

        title = sample["title"]
        text = sample["text"]
        # 得到候选实体的tokens和ids
        label_tokens = get_candidate_representation(
            text, tokenizer, max_cand_length, title,
        )
        entity_idx = int(wikipedia_id2local_id[sample["kb_id"]])
        if sample["mention_text"] not in id2_sim.keys():
            continue
        id_vec = [local_id2wikipedia_id[str(id_)] for id_ in id2_sim[str(entity_idx)][:3]]
        # id_vec = []
        # for id in id2_sim[sample["mention_text"]]:
        #     if id != sample["kb_id"]:
        #         if id not in id2text.keys():
        #             continue
        #         id_vec.append(id)
        #         if len(id_vec) == 3:
        #             break
        label_tokens_ = [label_tokens, 1]

        # result = [[] for i in id_vec] 
        result = [[] ,[],[]] 
        for i, id_ in enumerate(id_vec):
            infos =  id2text[str(id_)]
            title_f = infos['subject']
            description = ''
            if len(infos['data']) == 0:
                description = "摘要，" + title_f + "无描述信息。"
            else:
                for data in infos['data']:
                    str1 = data['predicate']
                    str2 = data['object']
                    flag = str2.endswith("。") | str2.endswith(".")
                    if flag:
                        description = description + str1 + ':' + str2
                    else:
                        description = description + str1 + ':' + str2 + '。'
        
            label_tokens_f = get_candidate_representation(
            description[:100], tokenizer, max_cand_length, title_f
        )
           
            result[i].append(label_tokens_)
            result[i].append([label_tokens_f, 0])
        
        for subarr in result:
            random.shuffle(subarr)
        
        for i in result:
            record = {
                    "context": context_tokens,
                    "label_A": i[0][0],
                    "label_B": i[1][0],
                    "label": [i[0][1], i[1][1]]
                }
            processed_samples.append(record)


    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"), dtype=torch.long,
    )
    cand_vecs_A = torch.tensor(
        select_field(processed_samples, "label_A", "ids"), dtype=torch.long,
    )
    cand_vecs_B = torch.tensor(
        select_field(processed_samples, "label_B", "ids"), dtype=torch.long,
    )
    label_vec = torch.tensor(
        select_field(processed_samples, "label"), dtype=torch.long,
    )

    data = {
        "context_vecs": context_vecs,
        "cand_vecs_A": cand_vecs_A,
        "cand_vecs_B": cand_vecs_B,
        "label_vec": label_vec
    }

    tensor_data = TensorDataset(context_vecs, cand_vecs_A, cand_vecs_B, label_vec)
    return data, tensor_data

def process_mention_data__(
    samples,
    tokenizer,
    max_context_length,
    max_cand_length,
    silent,
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    title_token=ENT_TITLE_TAG,
    debug=False,
    logger=None,
):
    processed_samples = []

    # 根据debug参数的值，判断是否只处理前200个提及样本
    # 如果debug为True，则只处理前200个提及样本
    if debug:
        samples = samples[:200]

    # 根据silent参数的值，判断是否在处理过程中显示进度条
    # 如果silent为True，则不显示进度条，直接将samples列表赋值给iter_
    # 否则，使用tqdm函数创建一个进度条，并将samples列表作为参数传递给iter_
    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples)
    with open('/home/ubuntu/Desktop/BLINK-main/blink/biencoder/data/ccks2019/wikipedia_id2local_id.json','r',encoding = 'utf-8') as fp:
        wikipedia_id2local_id = json.loads(fp.read())
    # 遍历每个样本
    for idx, sample in enumerate(iter_):
        # 得到上下文的tokens和ids
        context_tokens = get_context_representation(
            sample, tokenizer, max_context_length, ent_start_token, ent_end_token,
        )

        title = sample["title"]
        text = sample["text"]
        # 得到候选实体的tokens和ids
        entity_tokens = get_candidate_representation(
            text, tokenizer, max_cand_length, title,
        )
        # entity_idx = int(sample["kb_id"]) - 10001
        entity_idx =  wikipedia_id2local_id[sample["kb_id"]]
        record = {
                "context": context_tokens,
                "entity_tokens": entity_tokens,
                "entity_idx": [entity_idx]
            }
        processed_samples.append(record)

    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"), dtype=torch.long,
    )
    cand_vecs = torch.tensor(
        select_field(processed_samples, "entity_tokens", "ids"), dtype=torch.long,
    )
    entity_idx = torch.tensor(
        select_field(processed_samples, "entity_idx"), dtype=torch.long,
    )
    data = {
        "context_vecs": context_vecs,
        "entity_tokens": cand_vecs,
        "entity_idx": entity_idx
    }
    tensor_data = TensorDataset(context_vecs, cand_vecs, entity_idx)
    return data, tensor_data