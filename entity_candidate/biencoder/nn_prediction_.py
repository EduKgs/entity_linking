# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import logging
import torch
from tqdm import tqdm

import blink.candidate_ranking.utils as utils
from blink.biencoder.zeshel_utils import WORLDS, Stats
from collections import OrderedDict, defaultdict


def get_topk_predictions(
    reranker,
    train_dataloader,
    candidate_pool,
    cand_encode_list,
    silent,
    logger,
    top_k=10,
    is_zeshel=False,
    save_predictions=False,
):
    # 将模型设置为评估模式
    reranker.model.eval()
    # 获取设备
    device = reranker.device
    logger.info("Getting top %d predictions." % top_k)
    # 是否显示评估进度条
    if silent:
        iter_ = train_dataloader
    else:
        iter_ = tqdm(train_dataloader)

    nn_context = []         # 存储上下文向量
    nn_candidates = []      # 存储候选实体向量
    nn_labels = []          # 存储标签
    stats = {}              # 存储统计信息

    candidate_pool = [candidate_pool]
    cand_encode_list = [cand_encode_list]

    stats[0] = Stats(top_k)
    id_sim = defaultdict(dict)
    id_sim_file = open('/home/ubuntu/Desktop/BLINK-main/blink/biencoder/data/ccks2019' + '/id_sim.json', 'w', encoding='utf-8')
    # 初始化一个计数器
    oid = 0
    # 遍历迭代器，获取每个批次的数据
    for step, batch in enumerate(iter_):
        # 将批次中的张量数据移动到设备上
        batch = tuple(t.to(device) for t in batch)
        # 从批次中读取出上下文输入（ids）和entity_idx（kb_id对应的local_id）
        # context_input：[8, 128]
        # entity_idx：[8, 1]
        context_input, _, label_ids = batch
        # 使用模型对候选实体进行评分，返回得分
        # scores：[8, 200]   由[8,768]和[200,768]得来
        # debug下候选实体个数为200
        scores,_,_ = reranker.score_candidate(
            context_input, 
            None, 
            None,
            cand_encs=cand_encode_list[0].to(device)
        )
        # 获取前k个最高分数的值和索引
        # 分数从高到低已排序，索引是每行分数前10数所在的列id
        # values：[8, 10]
        # indicies：[8, 10]
        values, indicies = scores.topk(top_k)
        # 遍历上下文输入的每个样本
        for i in range(context_input.size(0)):
            oid += 1
            # 获取当前样本的候选实体索引
            inds = indicies[i]

            # 初始化指针为-1
            pointer = -1
            # 遍历10个候选实体的索引
            tem = []
            for j in range(len(inds)):
                # 如果该行样本的某个候选实体索引与该行样本本身的id匹配，更新指针并退出循环
                if inds[j].item() == label_ids[i].item():
                    pointer = j
                    # break
                else:
                    tem.append(str(inds[j].item()))
            id_sim[str(label_ids[i].item())] = tem
            # 将指针添加到统计信息中
            stats[0].add(pointer)

            # 如果没有匹配则遍历下一个样本
            if pointer == -1:
                continue

            # 如果不存储预测结果则遍历下一个样本
            if not save_predictions:
                continue

            # 仅记录了成功匹配的样本
            # 获取当前样本的10个候选实体的ids张量
            cur_candidates = candidate_pool[0][inds]
            # 将当前样本的上下文向量添加到列表中
            nn_context.append(context_input[i].cpu().tolist())
            # 将当前样本的候选实体向量添加到列表中
            nn_candidates.append(cur_candidates.cpu().tolist())
            # 将指针添加到标签列表中
            nn_labels.append(pointer) 
    id_sim_file.write(json.dumps(id_sim, ensure_ascii=False))
    id_sim_file.close()
    # 输出统计信息
    logger.info(stats[0].output())

    # 将上下文向量，候选实体向量和标签转换为长整型的张量
    nn_context = torch.LongTensor(nn_context)
    nn_candidates = torch.LongTensor(nn_candidates)
    nn_labels = torch.LongTensor(nn_labels)
    # 创建字典，存储上下文向量、候选实体向量和标签
    nn_data = {
        'context_vecs': nn_context,
        'candidate_vecs': nn_candidates,
        'labels': nn_labels,
    }
    
    return nn_data

