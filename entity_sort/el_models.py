from typing import Any

from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch.nn as nn
import torch
import torch.nn.functional as F
from el_preprocess import BertFeature
from el_dataset import ELDataset
import numpy as np

class BertForEntityLinking(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, args):
        super(BertForEntityLinking, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir)  # 预训练模型的路径
        self.bert_config = self.bert.config  # 预训练模型的配置文件
        self.out_dims = self.bert_config.hidden_size
        self.criterion = nn.CrossEntropyLoss()
        self.linear = nn.Sequential(
            torch.nn.Linear(2, args.num_tags),
        )

        self.fnn1 = nn.Sequential(
            torch.nn.Linear(3*768, 768),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(768, 1),
        )
        self.fnn2 = nn.Sequential(
            torch.nn.Linear(3*768, 768),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(768, 1),
        )
        self.fnn3 = nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(768, 1),
        )

    def find_entity_span(self, entity_mask, flag=1):
        """找到实体指代的开始和结束索引"""
        start = torch.where(entity_mask == flag)[0][0]
        end = torch.where(entity_mask == flag)[0][-1]
        return start, end
    
    
    def forward(self, token_ids, attention_masks, token_type_ids, seq_labels, entity_labels, can_labels):
        bert_outputs = self.bert(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )
        # 最后一层的所有token向量
        token_out = bert_outputs[0]  # [64,256,768]
        # CLS的向量
        seq_out = bert_outputs[1]  # [64,768]
        batch_out_entity = [] # 初始化结果张量
        batch_out_can = [] # 初始化结果张量
        batch_out_entity_des = [] # 初始化结果张量
        batch_out_can_des = []
        batch_out_cls = []
        for t_out, s_out, t_mask, c_mask, a_mask in zip(token_out, seq_out, entity_labels, can_labels):
            t_mask = t_mask == 1  # [256]
            c_mask = c_mask == 1  # [256]
            entity_out = t_out[t_mask]  # [2,768]
            can_out = t_out[c_mask]
            cls_out = s_out.unsqueeze(0)
            entity_out = entity_out.unsqueeze(0)  # [1,3,768]
            entity_out = F.adaptive_max_pool1d(entity_out.transpose(1, 2).contiguous(), output_size=1)   # [1,768]
            entity_out = entity_out.squeeze(-1)
            can_out = can_out.unsqueeze(0)  # [1,3,768]
            can_out = F.adaptive_max_pool1d(can_out.transpose(1, 2).contiguous(), output_size=1)   # [1,768]
            can_out = can_out.squeeze(-1)
            batch_out_entity.append(entity_out)
            batch_out_can.append(can_out)
            batch_out_cls.append(cls_out)

        batch_out_cls = torch.cat(batch_out_cls, dim=0)
        batch_out_cls = self.fnn3(batch_out_cls)

        batch_out_entity = torch.cat(batch_out_entity, dim=0)
        batch_out_can = torch.cat(batch_out_can, dim=0)
        batch_out_title = torch.cat([batch_out_entity, batch_out_can, torch.abs(batch_out_entity - batch_out_can)], dim=-1)
        batch_out_title = self.fnn1(batch_out_title)
        batch_out = torch.cat([batch_out_cls,batch_out_title], dim=-1)
        batch_out = self.linear(batch_out)
        if seq_labels is None:
            return batch_out
        loss = self.criterion(batch_out, seq_labels.float())
        return batch_out, loss

if __name__ == '__main__':
    class Args:
        bert_dir = './models/chinese-bert-wwm-ext/'
        num_tags = 2
        eval_batch_size = 4


    args = Args()
    import pickle

    tokenizer = BertTokenizer.from_pretrained(args.bert_dir + 'vocab.txt')
    test_out = pickle.load(open('./data/ccks2019/test.pkl', 'rb'))
    test_features, test_callback_info = test_out
    test_dataset = ELDataset(test_features)
    # for data in test_dataset:
    # text = tokenizer.convert_ids_to_tokens(data['token_ids'])
    # print(text)
    #     print(data['attention_masks'])
    #     print(data['token_type_ids'])
    #     print(data['seq_labels'])
    #     print(data['entity_labels'])
    #     break

    args.eval_batch_size = 4
    test_sampler = RandomSampler(test_dataset)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.eval_batch_size,
                             sampler=test_sampler,
                             num_workers=2)
    device = torch.device("cuda:0")
    model = BertForEntityLinking(args)
    model.to(device)
    for step, test_data in enumerate(test_loader):
        # print(test_data['token_ids'].shape)
        # print(test_data['attention_masks'].shape)
        # print(test_data['token_type_ids'].shape)
        # print(test_data['seq_labels'])
        # print(test_data['entity_labels'])
        for key in test_data:
            test_data[key] = test_data[key].to(device)
        _, loss = model(test_data['token_ids'],
                        test_data['attention_masks'],
                        test_data['token_type_ids'],
                        test_data['seq_labels'],
                        test_data['entity_labels'])
        print(loss.item())
        break
