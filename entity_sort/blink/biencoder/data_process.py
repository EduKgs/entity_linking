# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import torch
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset

from pytorch_transformers.tokenization_bert import BertTokenizer

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
    if sample["mention"] and len(sample["mention"]) > 0:
        # 使用tokenizer将其分割为单词列表
        mention_tokens = tokenization.BasicTokenizer().tokenize(sample["mention"])
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
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    title_token=ENT_TITLE_TAG,
   
):
    processed_samples = []
    iter_ = samples
       

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
        entity_idx = int(sample["kb_id"]) - 10001

        record = {
            "context": context_tokens,
            "entity": entity_tokens,
            "entity_idx": [entity_idx],
        }
        processed_samples.append(record)

    # 将处理后的数据转换为PyTorch张量数据
    # processed_samples中context中的ids转化为context_vecs
    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"), dtype=torch.long,
    )
    # processed_samples中label中的ids转化为cand_vecs
    cand_vecs = torch.tensor(
        select_field(processed_samples, "entity", "ids"), dtype=torch.long,
    )
    # 将processed_samples中的entity_idx转换为entity_idx
    entity_idx = torch.tensor(
        select_field(processed_samples, "entity_idx"), dtype=torch.long,
    )

    data = {
        "context_vecs": context_vecs,
        "cand_vecs": cand_vecs,
        "entity_idx": entity_idx,
    }

    # 将上下文向量、实体向量和标签向量打包为一个TensorDataset对象
    tensor_data = TensorDataset(context_vecs, cand_vecs, entity_idx)

    return data, tensor_data