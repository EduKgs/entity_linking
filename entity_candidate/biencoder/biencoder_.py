# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)

from pytorch_transformers.tokenization_bert import BertTokenizer

from blink.common.ranker_base import BertEncoder, get_model_obj
from blink.common.optimizer import get_bert_optimizer


def load_biencoder(params):
    # Init model
    biencoder = BiEncoderRanker(params)
    return biencoder


class BiEncoderModule(torch.nn.Module):
    def __init__(self, params):
        super(BiEncoderModule, self).__init__()
        ctxt_bert = BertModel.from_pretrained(os.path.join(os.path.dirname(os.path.dirname(__file__)), params["bert_model"]))
        # cand_bert = BertModel.from_pretrained(os.path.join(os.path.dirname(os.path.dirname(__file__)), params["bert_model"]))
        self.context_encoder = BertEncoder(
            ctxt_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        # self.cand_encoder = BertEncoder(
            # cand_bert,
            # params["out_dim"],
            # layer_pulled=params["pull_from_layer"],
            # add_linear=params["add_linear"],
        # )
        self.config = ctxt_bert.config

    def forward(
        self,
        token_idx_ctxt,
        segment_idx_ctxt,
        mask_ctxt,
        token_idx_cands,
        segment_idx_cands,
        mask_cands,
    ):
        embedding_ctxt = None
        if token_idx_ctxt is not None:
            embedding_ctxt = self.context_encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt
            )
        embedding_cands = None
        # if token_idx_cands is not None:
            # embedding_cands = self.cand_encoder(
                # token_idx_cands, segment_idx_cands, mask_cands
            # )
        return embedding_ctxt, embedding_cands


class BiEncoderRanker(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(BiEncoderRanker, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        self.tokenizer = BertTokenizer.from_pretrained(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), params["bert_model"]), do_lower_case=params["lowercase"]
        )
        # init model
        self.build_model()
        model_path = params.get("path_to_model", None)
        if model_path is not None:
            self.load_model(model_path)

        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)
        self.temperature = 0.05

    def load_model(self, fname, cpu=False):
        print("loading model...")
        if cpu:
            state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
        else:
            state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict)

    def build_model(self):
        self.model = BiEncoderModule(self.params)

    # def save_model(self, output_dir):
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #     model_to_save = get_model_obj(self.model) 
    #     output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    #     output_config_file = os.path.join(output_dir, CONFIG_NAME)
    #     torch.save(model_to_save.state_dict(), output_model_file)
    #     model_to_save.config.to_json_file(output_config_file)

    def get_optimizer(self, optim_states=None, saved_optim_type=None):
        return get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )
 
    def encode_context(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        embedding_context, _ = self.model(
            token_idx_cands, segment_idx_cands, mask_cands, None, None, None
        )
        return embedding_context.cpu().detach()

    def encode_candidate(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        return embedding_cands.cpu().detach()
        # TODO: why do we need cpu here?
        # return embedding_cands
    def score_candidate(
        self,
        text_vecs,
        cand_vecs_A,
        cand_vecs_B,
        random_negs1=True,
        random_negs2=True,
        cand_encs=None,  # pre-computed candidate encoding.
    ):
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        embedding_ctxt, _ = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None
        )

        # Candidate encoding is given, do not need to re-compute
        # Directly return the score of context encoding and candidate encoding
        if cand_encs is not None:
            return embedding_ctxt.mm(cand_encs.t()), None, None

        # Train time. We compare with all elements of the batch
        token_idx_cands_A, segment_idx_cands_A, mask_cands_A = to_bert_input(
            cand_vecs_A, self.NULL_IDX
        )
        embedding_cands_A,_= self.model(
            token_idx_cands_A, segment_idx_cands_A, mask_cands_A,None, None, None
        )
           
        if random_negs1 and random_negs2:
            # train on random negatives
            return embedding_ctxt.mm(embedding_cands_A.t()), None, None
        
        
        elif random_negs1 and not random_negs2:
            token_idx_cands_B, segment_idx_cands_B, mask_cands_B = to_bert_input(
            cand_vecs_B, self.NULL_IDX
            )
            embedding_cands_B,_ = self.model(
                token_idx_cands_B, segment_idx_cands_B, mask_cands_B,None, None, None
            )
            return embedding_ctxt, embedding_cands_A, embedding_cands_B
        
        else:
            token_idx_cands_B, segment_idx_cands_B, mask_cands_B = to_bert_input(
            cand_vecs_B, self.NULL_IDX
            )
            embedding_cands_B,_= self.model(
               token_idx_cands_B, segment_idx_cands_B, mask_cands_B, None, None, None
            )
            # train on hard negatives
            # embedding_ctxt = embedding_ctxt.unsqueeze(1)  # batchsize x 1 x embed_size
            # embedding_cands_A = embedding_cands_A.unsqueeze(2)  # batchsize x embed_size x 1
            # scores_A = torch.bmm(embedding_ctxt, embedding_cands_A)  # batchsize x 1 x 1
            # scores_A = torch.squeeze(scores_A).unsqueeze(1)
            # embedding_cands_B = embedding_cands_B.unsqueeze(2)  # batchsize x embed_size x 1
            # scores_B = torch.bmm(embedding_ctxt, embedding_cands_B)  # batchsize x 1 x 1
            # scores_B = torch.squeeze(scores_B).unsqueeze(1)
            scores_A = F.cosine_similarity(embedding_ctxt, embedding_cands_A, dim=1).unsqueeze(1)
            scores_B = F.cosine_similarity(embedding_ctxt, embedding_cands_B, dim=1).unsqueeze(1)
            scores = torch.cat((scores_A, scores_B), dim=1) 
            return scores, None, None
    # Score candidates given context input and label input
    # If cand_encs is provided (pre-computed), cand_ves is ignored
    # def score_candidate(
    #     self,
    #     text_vecs,
    #     cand_vecs_A,
    #     cand_vecs_B,
    #     random_negs1=True,
    #     random_negs2=True,
    #     cand_encs=None,  # pre-computed candidate encoding.
    # ):
    #     # Encode contexts first
    #     token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
    #         text_vecs, self.NULL_IDX
    #     )
    #     embedding_ctxt, _ = self.model(
    #         token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None
    #     )

    #     # Candidate encoding is given, do not need to re-compute
    #     # Directly return the score of context encoding and candidate encoding
    #     if cand_encs is not None:
    #         cosine_similarity = embedding_ctxt.mm(cand_encs.t())
    #         # cosine_similarity = F.cosine_similarity(embedding_ctxt.unsqueeze(1), cand_encs.unsqueeze(0), dim=2)
    #         return cosine_similarity, None, None

    #     # Train time. We compare with all elements of the batch
    #     token_idx_cands_A, segment_idx_cands_A, mask_cands_A = to_bert_input(
    #         cand_vecs_A, self.NULL_IDX
    #     )
    #     _, embedding_cands_A = self.model(
    #         None, None, None, token_idx_cands_A, segment_idx_cands_A, mask_cands_A
    #     )
           
    #     if random_negs1 and random_negs2:
    #         # train on random negatives
    #         # return embedding_ctxt.mm(embedding_cands_A.t()), None, None
    #         return embedding_ctxt, embedding_cands_A, None
        
    #     else:
    #         token_idx_cands_B, segment_idx_cands_B, mask_cands_B = to_bert_input(
    #         cand_vecs_B, self.NULL_IDX
    #         )
    #         _, embedding_cands_B = self.model(
    #             None, None, None, token_idx_cands_B, segment_idx_cands_B, mask_cands_B
    #         )
    #         # train on hard negatives
    #         # embedding_ctxt = embedding_ctxt.unsqueeze(1)  # batchsize x 1 x embed_size
    #         # embedding_cands_A = embedding_cands_A.unsqueeze(2)  # batchsize x embed_size x 1
    #         # scores_A = torch.bmm(embedding_ctxt, embedding_cands_A)  # batchsize x 1 x 1
    #         # scores_A = torch.squeeze(scores_A).unsqueeze(1)
    #         # embedding_cands_B = embedding_cands_B.unsqueeze(2)  # batchsize x embed_size x 1
    #         # scores_B = torch.bmm(embedding_ctxt, embedding_cands_B)  # batchsize x 1 x 1
    #         # scores_B = torch.squeeze(scores_B).unsqueeze(1)
    #         scores_A = F.cosine_similarity(embedding_ctxt, embedding_cands_A, dim=1).unsqueeze(1)
    #         scores_B = F.cosine_similarity(embedding_ctxt, embedding_cands_B, dim=1).unsqueeze(1)
    #         scores = torch.cat((scores_A, scores_B), dim=1) 
    #         return scores, None, None
    
    # label_input -- negatives provided
    # If label_input is None, train on in-batch negatives
    # def forward(self, context_input, cand_input_A, cand_input_B=None, label_input=None):
    #     flag1 = label_input is None
    #     flag2 = cand_input_B is None
    #     scores1, scores2, scores3 = self.score_candidate(context_input, cand_input_A, cand_input_B, flag1, flag2)
    #     if flag1 and flag2:
    #         """无监督的损失函数
    #         y_pred (tensor): bert的输出, [batch_size * 2, 768]
    #         """
    #         y_pred = torch.stack((scores1, scores2), dim=1).view(-1, 768)
    #         # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    #         y_true = torch.arange(y_pred.shape[0], device='cuda')
    #         y_true = (y_true - y_true % 2 * 2) + 1
    #         # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    #         sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    #         # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    #         sim = sim - torch.eye(y_pred.shape[0], device='cuda') * 1e12
    #         # 相似度矩阵除以温度系数
    #         sim = sim / self.temperature
    #         # 计算相似度矩阵与y_true的交叉熵损失
    #         # 计算交叉熵，每个case都会计算与其他case的相似度得分，得到一个得分向量，目的是使得该得分向量中正样本的得分最高，负样本的得分最低
    #         loss = F.cross_entropy(sim, y_true)
    #         return torch.mean(loss)
        
    #     else:
    #         return scores1
    def forward(self, context_input, cand_input_A, cand_input_B=None, label_input=None):
        flag1 = label_input is None
        flag2 = cand_input_B is None
        scores1, scores2, scores3 = self.score_candidate(context_input, cand_input_A, cand_input_B, flag1, flag2)
        if flag1 and not flag2:
            y_pred = torch.stack((scores1, scores2, scores3), dim=1).view(-1, 768)
            row = torch.arange(0,y_pred.shape[0],3,device='cuda') # [0,3]
            col = torch.arange(y_pred.shape[0], device='cuda') # [0,1,2,3,4,5]
            #这里[(0,1,2),(3,4,5)]代表二组样本，
            #其中0,1是相似句子，0,2是不相似的句子
            #其中3,4是相似句子，3,5是不相似的句子
            col = torch.where(col % 3 != 0)[0].cuda() # [1,2,4,5]
            y_true = torch.arange(0,len(col),2,device='cuda') # 生成真实的label  = [0,2]
            #计算各句子之间的相似度，形成下方similarities 矩阵，其中xij 表示第i句子和第j个句子的相似度
            #[[ x00,x01,x02,x03,x04 ,x05  ]
            # [ x10,x11,x12,x13,x14 ,x15  ]
            # [ x20,x21,x22,x23,x24 ,x25  ]
            # [ x30,x31,x32,x33,x34 ,x35  ]
            # [ x40,x41,x42,x43,x44 ,x45  ]
            # [ x50,x51,x52,x53,x54 ,x55  ]]
            similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
            #这里将similarities 做切片处理，形成下方矩阵
            #[[ x01,x02,x04 ,x05 ]  
            # [x31,x32,x34 ,x35 ]]
            similarities = torch.index_select(similarities,0,row)
            similarities = torch.index_select(similarities,1,col)
            #论文中除以 temperature 超参 
            similarities = similarities / self.temperature
            #下面这一行计算的是相似矩阵每一行和y_true = [0, 2] 的交叉熵损失
            #[[ x01,x02,x04 ,x05 ]   label = 0 含义：第0个句子应该和第1个句子的相似度最高,  即x01越接近1越好
            # [x31,x32,x34 ,x35 ]]  label = 2 含义：第3个句子应该和第4个句子的相似度最高   即x34越接近1越好
            #这行代码就是simsce的核心部分，和正例句子向量相似度应该越大 
            #越好，和负例句子之间向量的相似度越小越好
            loss = F.cross_entropy(similarities,y_true)
            return torch.mean(loss)
        
        else:
            return scores1

def to_bert_input(token_idx, null_idx):
    """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
    """
    segment_idx = token_idx * 0
    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask
