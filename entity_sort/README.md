# pytorch_bert_entity_linking
基于bert的中文实体链接<br>
在hugging face上下载好预训练的权重：chinese-bert-wwm-ext<br>
这里在预测的时候选择用修改后的jieba分词+自定义实体词典的方式来获取候选实体，需要注意的地方是对英文词组的切分方式。

# 目录说明
--checkpoints：模型保存<br>
--data：数据<br>
--logs：日志<br>
--my_jieba：修改后的结巴分词，解决jieba分词不能将知识库中的kg ls正确分词<br>
--utils：辅助函数，里面值得注意的是tokenization，主要解决的是进行token化的时候将英文、数字等分开。<br>
--el_config.py：配置信息<br>
--el_dataset.py：转换数据为pytorch的格式<br>
--el_main.py：主运行文件，训练、验证测试和预测<br>
--el_main.sh：运行指令<br>
--el_models.py：模型<br>
--el_preprocess.py：处理数据为bert需要的格式<br>
--el_process.py：处理原始训练数据和知识库，得到一些中间文件<br>
--el_processor.py：测试el_preprocess中的处理器<br>
--el_service.py：进行起服务<br>
--service.log：服务日志<br>
--service_main.py：抽离主程序，用于起服务<br>
--start_service.sh：开始服务<br>
--stop_servie.sh：终止服务<br>
--test_jieba.py：测试my_jieba<br>
--test_service.py：测试调用起的服务<br>
--test_tokenizer.py：测试tokenizer<br>
同时，我们要注意数据的一些文件：在/data/ccks2019/下<br>
alias_and_subjects.txt：知识库中的实体名<br>
develop.json：用于预测<br>
entity_to_ids.json：实体以及对应知识库中的id<br>
entity_type.txt：实体的类型<br>
kb_data：知识库<br>
subject_id_with_info.json：知识库中实体id及其对应的相关信息<br>
test.pkl：测试二进制文件<br>
train.json：训练数据<br>
train.pkl：训练二进制文件<br>

# 流程
首先是el_process.py里面生成一些我们所需要的中间文件。然后是el_processor.py测试数据处理器。接着在el_preprocess.py里面处理数据为bert所需要的格式，并划分训练集和测试集，存储为相关二进制文件。在el_dataset.py里面转换为pytorch所需要的格式，最后在el_main.py里面调用。

# 依赖
```python
pytorch==1.6
transformers
sklearn
```

# 命令
```python
python el_main.py \
--bert_dir="../model_hub/chinese-bert-wwm-ext/" \
--data_dir="./data/ccsk2019/" \
--log_dir="./logs/" \
--output_dir="./checkpoints" \
--num_tags=2 \
--seed=123 \
--gpu_ids="0" \
--max_seq_len=256 \
--lr=2e-5 \
--other_lr=2e-4 \
--train_batch_size=32 \
--train_epochs=1 \
--eval_batch_size=32
```

# 起服务
```python
nohup python -u el_service.py --ip '0.0.0.0' --port '1080' > service.log 2>&1 &
```
# 测试服务
```python
import requests

text = '恶魔猎手吧-百度贴吧--《魔兽世界》恶魔猎手职业贴吧...'
text = text.encode('utf-8')
url = 'http://0.0.0.0:1080/entity_linking'
result = requests.post(url, data=text)
result = result.text
print(result)
```
# 终止服务
最后面要多一个空格。
```python
ps -ef|grep "el_service.py --ip ${ip}"|grep -v grep|awk '{print $2}'|xargs kill -9 
```
