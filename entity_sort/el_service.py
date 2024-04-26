import json
import sys
import traceback
import argparse
from ast import literal_eval
from flask import Flask, request, abort, Response

from transformers import BertTokenizer
from service_main import Trainer
from utils import tokenization
import my_jieba


# 字典转换为可以用.获取值得形式
def dict_to_obj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    class Obj:
        pass
    obj = Obj()
    for k in dictObj:
        obj.__dict__[k] = dict_to_obj(dictObj[k])
    return obj

class EntityLinking:
    def __init__(self):
        with open('./checkpoints/args.json', 'r') as fp:
            self.args = dict_to_obj(json.loads(fp.read()))
        self.checkpoint_path = './checkpoints/15000.pt'
        my_jieba.load_userdict('./data/ccks2019/alias_and_subjects.txt')
        self.jieba = my_jieba
        # 实体库
        with open('./data/ccks2019/alias_and_subjects.txt', 'r') as fp:
            self.entities = fp.read().strip().split('\n')
        # 实体对应的id
        with open('./data/ccks2019/entity_to_ids.json', 'r') as fp:
            self.entity_to_ids = json.loads(fp.read())
        # 实体id对应的描述
        with open('./data/ccks2019/subject_id_with_info.json', 'r') as fp:
            self.subject_id_with_info = json.loads(fp.read())
        self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_dir + 'vocab.txt')
        self.tokenization = tokenization
        self.trainer = Trainer(self.args)

    def predict(self, text):
        return self.trainer.predict(
            self.checkpoint_path,
            text,
            self.args,
            self.tokenizer,
            self.tokenization,
            self.entities,
            self.entity_to_ids,
            self.subject_id_with_info,
            self.jieba,
        )

    def parse_result(self, result):
        for res in result:
            for info in res:  # 这里我们选择分数最高的打印
                print('====================================')
                print('候选实体名：' + info[0])
                print('知识库实体名：' + info[3])
                print('知识库ID：' + info[1])
                print('置信分数：' + str(info[2]))
                print('类型：' + '、'.join(info[4]))
                print('描述：' + info[5][:100] + '......')
                print('====================================')
                break


app = Flask(__name__)

@app.route("/entity_linking", methods=['POST', 'GET'])
def get_result():
    if request.method == 'POST':
        text = request.data.decode("utf-8")
    else:
        text = request.args['text']
    try:
        result, NIL_list = el.predict(text)  # '''调用模型，返回结果'''
        el.parse_result(result)
    except Exception as e:
        result_error = {'errcode': -1}
        result = json.dumps(result_error, indent=4, ensure_ascii=False)
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        abort(Response("Failed!\n" + '\n\r\n'.join('' + line for line in lines)))

    return Response(str(result), mimetype='application/json')




if __name__ == '__main__':
    el = EntityLinking()
    service_parser = argparse.ArgumentParser()
    service_parser.add_argument('--ip', type=str,
                        help='service_ip',
                        default='0.0.0.0')
    service_parser.add_argument('--port', type=int,
                        help='service_port',
                        default=1080)
    service_args = service_parser.parse_args()
    app.run(host=service_args.ip, port=service_args.port, threaded=False)