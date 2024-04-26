# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from collections import defaultdict
import json
import logging
import os
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from pytorch_transformers.tokenization_bert import BertTokenizer

from blink.biencoder.biencoder_ import BiEncoderRanker
import blink.biencoder.data_process_ as data
import blink.biencoder.nn_prediction_ as nnquery
import blink.candidate_ranking.utils as utils
from blink.biencoder.zeshel_utils import WORLDS, load_entity_dict_zeshel, Stats
from blink.common.params import BlinkParser


def load_entity_dict(logger, params, is_zeshel):
    if is_zeshel:
        return load_entity_dict_zeshel(logger, params)

    path = params.get("entity_dict_path", None)
    # 获取实体文件路径
    path = "/media/ubuntu/A4CAA455CAA42610/BLINK-main/blink/biencoder/data/ccks2019/subject_id_with_info.json"
    assert path is not None, "Error! entity_dict_path is empty."
    # 从指定路径的文件中逐行加载实体的标题和文本描述，并将它们存储在一个列表中返回
    entity_list = []
    logger.info("Loading entity description from path: " + path)
    with open(path, 'r', encoding='utf-8') as f:
        js = json.load(f)
    cnt = len(js)
    for k,v in js.items():
        print(cnt)
        cnt -= 1
        line = v
        title = line['subject']
        des = line['data']
        description = ''
        for text in des:
            str1 = text['predicate']
            str2 = str(text['object'])
            flag = str2.endswith("。") | str2.endswith(".")
            if flag:
                description = description + str1 + ':' + str2
            else:
                description = description + str1 + ':' + str2 + '。'
        entity_list.append((title, description[:100]))
        # 在调试模式下，可以通过设置 debug 标志来限制加载的实体数量
        if params["debug"] and len(entity_list) >= 200:
            break
    return entity_list


# zeshel version of get candidate_pool_tensor
def get_candidate_pool_tensor_zeshel(
    entity_dict,
    tokenizer,
    max_seq_length,
    logger,
):
    candidate_pool = {}
    for src in range(len(WORLDS)):
        if entity_dict.get(src, None) is None:
            continue
        logger.info("Get candidate desc to id for pool %s" % WORLDS[src])
        candidate_pool[src] = get_candidate_pool_tensor(
            entity_dict[src],
            tokenizer,
            max_seq_length,
            logger,
        )

    return candidate_pool


def get_candidate_pool_tensor_helper(
    entity_desc_list,
    tokenizer,
    max_seq_length,
    logger,
    is_zeshel,
):
    if is_zeshel:
        return get_candidate_pool_tensor_zeshel(
            entity_desc_list,
            tokenizer,
            max_seq_length,
            logger,
        )
    else:
        return get_candidate_pool_tensor(
            entity_desc_list,
            tokenizer,
            max_seq_length,
            logger,
        )


def get_candidate_pool_tensor(
    entity_desc_list,
    tokenizer,
    max_seq_length,
    logger,
):
    # TODO: add multiple thread process
    logger.info("Convert candidate text to id")
    cand_pool = [] 
    # 从列表中依次读取每个实体的标题和文本描述
    for entity_desc in tqdm(entity_desc_list):
        if type(entity_desc) is tuple:
            title, entity_text = entity_desc
        else:
            title = None
            entity_text = entity_desc
        # 通过get_candidate_representation方法进行数据处理，生成每个实体的tokens和ids
        rep = data.get_candidate_representation(      # rep 包含tokens和ids
                entity_text, 
                tokenizer, 
                max_seq_length,
                title,
        )
        # cand_pool包含所有实体的ids
        cand_pool.append(rep["ids"])
    # 将数据转换为PyTorch张量
    cand_pool = torch.LongTensor(cand_pool) 
    return cand_pool


def encode_candidate(
    reranker,
    candidate_pool,
    encode_batch_size,
    silent,
    logger,
    is_zeshel,
):
    if is_zeshel:
        src = 0
        cand_encode_dict = {}
        for src, cand_pool in candidate_pool.items():
            logger.info("Encoding candidate pool %s" % WORLDS[src])
            cand_pool_encode = encode_candidate(
                reranker,
                cand_pool,
                encode_batch_size,
                silent,
                logger,
                is_zeshel=False,
            )
            cand_encode_dict[src] = cand_pool_encode
        return cand_encode_dict
        
    reranker.model.eval()
    device = reranker.device
    sampler = SequentialSampler(candidate_pool)
    data_loader = DataLoader(
        candidate_pool, sampler=sampler, batch_size=encode_batch_size
    )
    if silent:
        iter_ = data_loader
    else:
        iter_ = tqdm(data_loader)

    cand_encode_list = None
    for step, batch in enumerate(iter_):
        cands = batch
        cands = cands.to(device)
        # cand_encode = reranker.encode_candidate(cands)
        cand_encode = reranker.encode_context(cands)
        if cand_encode_list is None:
            cand_encode_list = cand_encode
        else:
            cand_encode_list = torch.cat((cand_encode_list, cand_encode))

    return cand_encode_list


def load_or_generate_candidate_pool(
    tokenizer,
    params,
    logger,
    cand_pool_path,
):
    candidate_pool = None
    is_zeshel = params.get("zeshel", None)
    if cand_pool_path is not None:
        # try to load candidate pool from file
        try:
            logger.info("Loading pre-generated candidate pool from: ")
            logger.info(cand_pool_path)
            candidate_pool = torch.load(cand_pool_path)
        except:
            logger.info("Loading failed. Generating candidate pool")

    if candidate_pool is None:
        # compute candidate pool from entity list
        entity_desc_list = load_entity_dict(logger, params, is_zeshel)
        candidate_pool = get_candidate_pool_tensor_helper(
            entity_desc_list,
            tokenizer,
            params["max_cand_length"],
            logger,
            is_zeshel,
        )

        if cand_pool_path is not None:
            logger.info("Saving candidate pool.")
            torch.save(candidate_pool, cand_pool_path)

    return candidate_pool


def make_ids(data_path):
    kb_data_file = data_path + '/kb_data'
    train_data_file = data_path + '/agr_train_8627.json'
    alias_and_subjects = []
    alias_and_subjects_file = open(data_path + '/alias_and_subjects.txt', 'w', encoding='utf-8')
    # 保存每个实体在知识库中对应的id（一对多），
    # 格式为名字：{'subject_id':'','subject_name':'}
    entity_to_ids = defaultdict(list)
    entity_to_ids_files = open(data_path + '/entity_to_ids.json', 'w', encoding='utf-8')
    # 用于保存每个实体的类型
    entity_type = []
    entity_type_file = open(data_path + '/entity_type.txt', 'w', encoding='utf-8')
    # 用于保存每个subject_id对应的信息
    subject_id_with_info = defaultdict(dict)
    subject_id_with_info_file = open(data_path + '/subject_id_with_info.json', 'w', encoding='utf-8')

    with open(kb_data_file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        total = len(lines) - 1
        for i, line in enumerate(lines):
            # print(i, total)
            line = eval(line)
            # 实体类型
            for e_type in line['type']:
                entity_type.append(e_type)
            # line['alias']为实体所有别名
            for word in line['alias']:
                word = word.lower()
                # 别名列表
                alias_and_subjects.append(word)
                # 实体到id映射--实体名称：[]
                if line['subject_id'] not in entity_to_ids[word]:
                    entity_to_ids[word].append(line['subject_id'])
            subject_id_with_info[line['subject_id']] = line

    # print("===============================")
    with open(train_data_file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        total = len(lines) - 1
        for i, line in enumerate(lines):
            print(i, total)
            line = eval(line)
            mention_datas = line['mention_data']
            for mention_data in mention_datas:
                # 提及词
                word = mention_data['mention'].lower()
                # 提及词加入别名列表
                alias_and_subjects.append(word)
                # 提及词对应的实体id加入实体和id映射文件
                if mention_data['kb_id'] not in entity_to_ids[word]:
                    entity_to_ids[word].append(mention_data['kb_id'])

    entity_type = list(set(entity_type))  # 去重类型转换为list
    entity_type_str = "\n".join(entity_type)  # 每行一个类型
    alias_and_subjects = sorted(list(set(alias_and_subjects)), key=lambda x: len(x), reverse=True)  # 去重别名，转换为list，按照长度进行降序排序
    alias_and_subjects_str = "\n".join(alias_and_subjects)

    alias_and_subjects_file.write(alias_and_subjects_str)
    entity_type_file.write(entity_type_str)
    entity_to_ids_files.write(json.dumps(entity_to_ids, ensure_ascii=False))
    subject_id_with_info_file.write(json.dumps(subject_id_with_info, ensure_ascii=False))

    alias_and_subjects_file.close()
    entity_type_file.close()
    entity_to_ids_files.close()
    subject_id_with_info_file.close()



def main(params):
    output_path = params["output_path"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logger = utils.get_logger(params["output_path"])

    # Init model 
    reranker = BiEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model
    
    device = reranker.device
    
    cand_encode_path = params.get("cand_encode_path", None)
    
    # candidate encoding is not pre-computed. 
    # load/generate candidate pool to compute candidate encoding.
    cand_pool_path = params.get("cand_pool_path", None)
    candidate_pool = load_or_generate_candidate_pool(
        tokenizer,
        params,
        logger,
        cand_pool_path,
    )       

    candidate_encoding = None
    if cand_encode_path is not None:
        # try to load candidate encoding from path
        # if success, avoid computing candidate encoding
        try:
            logger.info("Loading pre-generated candidate encode path.")
            candidate_encoding = torch.load(cand_encode_path)
        except:
            logger.info("Loading failed. Generating candidate encoding.")

    if candidate_encoding is None:
        candidate_encoding = encode_candidate(
            reranker,
            candidate_pool,
            params["encode_batch_size"],
            silent=params["silent"],
            logger=logger,
            is_zeshel=params.get("zeshel", None)
            
        )

        if cand_encode_path is not None:
            # Save candidate encoding to avoid re-compute
            logger.info("Saving candidate encoding to file " + cand_encode_path)
            torch.save(candidate_encoding, cand_encode_path)
    # make_ids(params['data_path'])
    # train_samples, valid_samples = utils.new_read_dataset("train", params["data_path"])
    _,train_samples= utils.new_read_dataset("train0", params["data_path"])
    logger.info("Read %d test samples." % len(train_samples))
    test_data, test_tensor_data = data.process_mention_data(
        train_samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"]
    )
    test_sampler = SequentialSampler(test_tensor_data)
    test_dataloader = DataLoader(
        test_tensor_data, 
        sampler=test_sampler, 
        batch_size=params["eval_batch_size"]
    )  

    # make_ids(params['data_path'])

    save_results = params.get("save_topk_result")
    with torch.no_grad():
        new_data = nnquery.get_topk_predictions(
            reranker,
            test_dataloader,
            candidate_pool,
            candidate_encoding,
            params["silent"],
            logger,
            params["top_k"],
            params.get("zeshel", None),
            save_results,
        )

    if save_results: 
        save_data_dir = os.path.join(
            params['output_path'],
            "top%d_candidates" % params['top_k'],
        )
        if not os.path.exists(save_data_dir):
            os.makedirs(save_data_dir)
        save_data_path = os.path.join(save_data_dir, "%s.t7" % params['mode'])
        torch.save(new_data, save_data_path)


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_eval_args()

    args = parser.parse_args()
    print(args)

    params = args.__dict__

    mode_list = params["mode"].split(',')
    for mode in mode_list:
        new_params = params
        new_params["mode"] = mode
        main(new_params)
