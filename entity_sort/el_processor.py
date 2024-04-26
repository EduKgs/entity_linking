"""
该文件主要是测试ELProcess类，
将知识库中的文本和query文本进行连接，
然后构建正负样本。
"""
import json
import random


class ELProcessor:
    def __init__(self):
        with open('./data/ccks2019/entity_to_ids.json', 'r', encoding='utf-8') as fp:
            self.entity_to_ids = json.loads(fp.read())
        with open('./data/ccks2019/subject_id_with_info.json', 'r', encoding='utf-8') as fp:
            self.subject_id_with_info = json.loads(fp.read())

    def read_json(self, path):  # 读取train.json
        with open(path, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
        return lines

    def get_result(self, lines):
        examples = []
        for i, line in enumerate(lines):
            line = eval(line)
            text = line['text'].lower()  # 每个训练数据的文本
            for mention_data in line['mention_data']:  # 每个训练数据的提及信息
                word = mention_data['mention'].lower()  # 提及词
                kb_id = mention_data['kb_id']  # 提及词对应的实体id
                start_id = int(mention_data['offset'])  # 提及词在文本中开始位置
                end_id = start_id + len(word) - 1  # 提及词在文本中的结束位置
                # print((kb_id, word, start_id, end_id))
                rel_texts = self.get_text_pair(word, kb_id, text)
                for i, rel_text in enumerate(rel_texts):
                    print('text：', rel_text)
                    print('entity_label：', (kb_id, word, start_id, end_id))
                    if i == 0:
                        print('seq_label：', 1)
                    else:
                        print('seq_label：', 0)

            if i == 1:
                break

    def get_text_pair(self, word, kb_id, text):
        """
        用于构建正负样本对，一个正样本，三个负样本
        :return:
        """
        results = []
        # 如果实体id不是不可链接实体id，并且提及词在实体-id映射可以找到
        if kb_id != 'NIL' and word in self.entity_to_ids:
            # 该实体id对应属性以及提及词本来的文本信息
            pos_example = self.get_info(kb_id) + '#;#' + text
            results.append(pos_example)

            # 获得提及词的id
            ids = self.entity_to_ids[word]
            # 如果id有NIL，就移除
            if 'NIL' in ids:
                ids.remove('NIL')
            ind = ids.index(kb_id)  # 找出提及词对应实体id列表中的该实体id位置
            ids = ids[:ind] + ids[ind + 1:]  # 取出除了该实体id的所有实体id
            if len(ids) >= 3:
                ids = random.sample(ids, 3)  # 如果超过三个随机选择三个
            for t_id in ids:
                info = self.get_info(t_id)  # 循环获得这些实体id对应的信息
                neg_example = info + '#;#' + text
                results.append(neg_example)
        return results

    def get_info(self, subject_id):
        """
        根据subject_id找到其描述文本，将predicate和object拼接
        :param subject_id:
        :return:
        """
        # 获得实体id对应的实体信息
        infos = self.subject_id_with_info[subject_id]
        data = infos['data']  # 该实体id对应的属性
        res = []
        for kg in data:
            # 形成预测和实体信息的一个句子
            if kg['object'][-1] != '。':
                res.append("{}，{}。".format(kg['predicate'], kg['object']))
            else:
                res.append("{}，{}".format(kg['predicate'], kg['object']))
        return "".join(res).lower()


if __name__ == '__main__':
    elProcessor = ELProcessor()
    data_path = './data/ccks2019'
    train_data = data_path + '/train.json'
    elProcessor.get_result(elProcessor.read_json(train_data))
    # print(elProcessor.get_info('10010'))
