import torch
import torch.nn as nn
import el_models



class Trainer:
    def __init__(self, args):
        self.args = args
        gpu_ids = args.gpu_ids.split(',')
        self.device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
        self.model = el_models.BertForEntityLinking(args)
        self.model.to(self.device)


    def load_ckp(self, model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, epoch, loss

    def save_ckp(self, state, checkpoint_path):
        torch.save(state, checkpoint_path)


    def convert_example_to_feature(self0,
                                   text_b,
                                   start,
                                   end,
                                   ids,
                                   tokenizer,
                                   tokenization,
                                   subject_id_with_info,
                                   args):
        features = []
        for t_id in ids:
            if t_id in subject_id_with_info:
                info = subject_id_with_info[t_id]
                text_a_list = []
                for kg in info['data']:
                    # print(kg)
                    if kg['object'][-1] != '。':
                        text_a_list.append("{}，{}。".format(kg['predicate'],kg['object']))
                    else:
                        text_a_list.append("{}，{}".format(kg['predicate'], kg['object']))

                text_a = "".join(text_a_list)
                text_a = tokenization.BasicTokenizer().tokenize(text_a)
                encode_dict = tokenizer.encode_plus(text=text_a,
                                                    text_pair=text_b,
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
                # print(start)
                # print(end)
                # print(offset)
                start_id = offset + start
                end_id = offset + end
                # print(start_id)
                # print(end_id)
                # print(tokenizer.convert_ids_to_tokens(token_ids[0]))
                # print(tokenizer.convert_ids_to_tokens(token_ids[0][start_id:end_id+1]))
                # print(len(token_ids[0]))
                # print(start_id, end_id)
                for i in range(start_id, end_id):
                    entity_ids[i] = 1
                entity_ids = torch.tensor(entity_ids, requires_grad=False).unsqueeze(0)
                features.append(
                    (
                        token_ids,
                        attention_masks,
                        token_type_ids,
                        entity_ids,
                        info['subject_id'],
                        info['subject'],
                        info['type'],
                        "".join(text_a_list),
                    )
                )
        return features


    def predict(self,
                checkpoint_path,
                text,
                args,
                tokenizer,
                tokenization,
                entities,
                entity_to_ids,
                subject_id_with_info,
                jieba_cut,
                ):
        model = self.model
        model, epoch, loss = self.load_ckp(model, checkpoint_path)
        model.eval()
        model.to(self.device)
        # 先提取text中的实体，这里结合实体库利用jieba分词进行
        text = text.lower()
        words = jieba_cut.lcut(text, cut_all=False)
        # text_b=['《', '仙剑奇侠', '三', '》', '紫萱', '为', '保护', '林业平', '被迫', '显出', '原型']
        # result中是一个元组，第一维表示该实体名，第二位是在知识库中的subject_id，第三位是分数,
        # 第四位是真实名，第五位是类型，第六位是描述
        result = []
        NIL_list = []
        with torch.no_grad():
            for word in words:
                # 如果该词是一个候选实体，那么我们从知识库中找到其subject_id
                if word in entities:
                    # print(word)
                    tmp_res = []
                    ids = entity_to_ids[word]
                    if len(ids) == 1 and ids[-1] == 'NIL':
                        NIL_list.append(word)
                    else:
                        # 在文本中找到该实体的起始和结束位置,这里我们只找第一次出现的位置就行了
                        # 这里我们要合并这两个分词的结果
                        ind = text.index(word)
                        start_ = tokenization.BasicTokenizer().tokenize(text[:ind])
                        word_ = tokenization.BasicTokenizer().tokenize(word)
                        end_ = tokenization.BasicTokenizer().tokenize(text[ind+len(word):])
                        start = len(start_)
                        end = start+len(word_)
                        text_b = start_ + word_ + end_
                        # print(text_b)
                        features = self.convert_example_to_feature(
                            text_b,
                            start,
                            end,
                            ids,
                            tokenizer,
                            tokenization,
                            subject_id_with_info,
                            args,
                        )
                        if len(features) != 0:
                            for feature in features:
                                logit = model(
                                    feature[0].to(self.device),
                                    feature[1].to(self.device),
                                    feature[2].to(self.device),
                                    None,
                                    feature[3].to(self.device),
                                )
                                # print(logit)
                                sigmoid = nn.Sigmoid()
                                logit = sigmoid(logit)
                                pred = logit.cpu().detach().numpy()[0][1]
                                # print(pred)
                                tmp_res.append(
                                    (
                                        word,
                                        feature[4],
                                        pred,
                                        feature[5],
                                        feature[6],
                                        feature[7],
                                    )
                                )
                            tmp_res = sorted(tmp_res, key=lambda x:x[2], reverse=True)
                            # print(tmp_res)
                            result.append(tmp_res)
                        else:
                            continue
        return result, NIL_list



