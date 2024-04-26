import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
# 这里要显示的引入BertFeature，不然会报错
from el_preprocess import BertFeature
from el_preprocess import ELProcessor
from transformers import BertTokenizer
import el_config


class ELDataset(Dataset):
    def __init__(self, features):  # 特征向量
        self.nums = len(features)
        self.features = features

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        example = self.features[index]
        data = {'token_ids': torch.tensor(example.token_ids).long(),
                'attention_masks': torch.tensor(example.attention_masks).float(),
                'token_type_ids': torch.tensor(example.token_type_ids).long(),
                'seq_labels': torch.tensor(example.seq_labels).long(),
                'entity_labels': torch.tensor(example.entity_labels).long(),
                'can_labels': torch.tensor(example.can_labels).long()}
        
        return data


if __name__ == '__main__':
    args = el_config.Args().get_parser()
    args.log_dir = './logs/'
    args.max_seq_len = 256
    args.bert_dir = './models/chinese-bert-wwm-ext/'
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir + 'vocab.txt')

    import pickle

    test_out = pickle.load(open('data/ccks2019/test.pkl', 'rb'))
    test_features, test_callback_info = test_out
    test_dataset = ELDataset(test_features)
    for i, data in enumerate(test_dataset):
        print(data['token_ids'])
        print(tokenizer.convert_ids_to_tokens(data['token_ids']))
        print(data['attention_masks'])
        print(data['token_type_ids'])
        print(data['seq_labels'])
        print(data['entity_labels'])
        if i == 2:
            break

    args.eval_batch_size = 2
    test_sampler = RandomSampler(test_dataset)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.eval_batch_size,
                             sampler=test_sampler,
                             num_workers=2)
    for step, test_data in enumerate(test_loader):
        print(test_data['token_ids'].shape)
        print(test_data['attention_masks'].shape)
        print(test_data['token_type_ids'].shape)
        print(test_data['seq_labels'])
        print(test_data['entity_labels'])
        break
