# pytorch_bert_entity_linking
基于bert的中文实体链接<br>
在hugging face上下载好预训练的权重：chinese-bert-wwm-ext<br>
已经训练好的模型以及数据：<br>
链接：https://pan.baidu.com/s/17y29cBIqIuVzUMub60E53w <br>
提取码：g7og<br>
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

# 训练和测试
```python
2021-09-06 15:06:50,504 - INFO - el_main.py - train - 103 - 【train】 epoch：0 step:15038/15267 loss：0.345154
2021-09-06 15:06:51,283 - INFO - el_main.py - train - 103 - 【train】 epoch：0 step:15039/15267 loss：0.409021
2021-09-06 15:06:52,007 - INFO - el_main.py - train - 103 - 【train】 epoch：0 step:15040/15267 loss：0.313814
2021-09-06 15:06:52,859 - INFO - el_main.py - train - 103 - 【train】 epoch：0 step:15041/15267 loss：0.328172
========进行测试========
2021-09-06 16:04:21,522 - INFO - el_main.py - <module> - 252 - 【test】 loss：1076.141372 accuracy：0.9367 precision：0.9133 recall：0.9144 micro_f1：0.9367
2021-09-06 16:04:21,974 - INFO - el_main.py - <module> - 254 -               
              precision    recall  f1-score   support

           0       0.95      0.95      0.95    132491
           1       0.91      0.91      0.91     76887

    accuracy                           0.94    209378
   macro avg       0.93      0.93      0.93    209378
weighted avg       0.94      0.94      0.94    209378
```

# 预测
```python
text = '《仙剑奇侠三》紫萱为保护林业平被迫显出原型'
====================================
2021-09-07 11:35:54,624 - INFO - el_main.py - <module> - 394 - 候选实体名：《
2021-09-07 11:35:54,624 - INFO - el_main.py - <module> - 395 - 知识库实体名：书名号
2021-09-07 11:35:54,624 - INFO - el_main.py - <module> - 396 - 知识库ID：219092
2021-09-07 12:12:46,730 - DEBUG - __init__.py - initialize - 118 - Building prefix dict from the default dictionary ...
2021-09-07 12:12:46,730 - DEBUG - __init__.py - initialize - 138 - Loading model from cache /tmp/jieba.cache
2021-09-07 12:12:47,425 - DEBUG - __init__.py - initialize - 170 - Loading model cost 0.695 seconds.
2021-09-07 12:12:47,425 - DEBUG - __init__.py - initialize - 171 - Prefix dict has been built successfully.
2021-09-07 12:13:14,608 - INFO - el_main.py - <module> - 393 - ====================================
2021-09-07 12:13:14,608 - INFO - el_main.py - <module> - 394 - 候选实体名：《
2021-09-07 12:13:14,608 - INFO - el_main.py - <module> - 395 - 知识库实体名：书名号
2021-09-07 12:13:14,608 - INFO - el_main.py - <module> - 396 - 知识库ID：219092
2021-09-07 12:13:14,608 - INFO - el_main.py - <module> - 397 - 置信分数：0.1311657
2021-09-07 12:13:14,608 - INFO - el_main.py - <module> - 398 - 类型：Thing
2021-09-07 12:13:14,608 - INFO - el_main.py - <module> - 399 - 描述：摘要，书名号是用于标明书名、篇名、报刊名、文件名、戏曲名、歌曲名、图画名等的标点符号，亦用于歌曲、电影、电视剧等与书面媒介紧密相关的文艺作品。书名号分为双书名号(《》)和单书名号(〈〉)，书名号里还有......
2021-09-07 12:13:14,608 - INFO - el_main.py - <module> - 400 - ====================================
2021-09-07 12:13:14,609 - INFO - el_main.py - <module> - 393 - ====================================
2021-09-07 12:13:14,609 - INFO - el_main.py - <module> - 394 - 候选实体名：仙剑奇侠
2021-09-07 12:13:14,609 - INFO - el_main.py - <module> - 395 - 知识库实体名：仙剑奇侠传
2021-09-07 12:13:14,609 - INFO - el_main.py - <module> - 396 - 知识库ID：39813
2021-09-07 12:13:14,609 - INFO - el_main.py - <module> - 397 - 置信分数：0.38944265
2021-09-07 12:13:14,609 - INFO - el_main.py - <module> - 398 - 类型：Game
2021-09-07 12:13:14,609 - INFO - el_main.py - <module> - 399 - 描述：摘要，《仙剑奇侠传》是由中国台湾大宇资讯股份有限公司(简称“大宇资讯”或“大宇”)旗下发行的系列电脑游戏。仙剑故事以中国古代的仙妖神鬼传说为背景、以武侠和仙侠为题材，迄今已发行八款单机角色扮演游戏、一......
2021-09-07 12:13:14,609 - INFO - el_main.py - <module> - 400 - ====================================
2021-09-07 12:13:14,609 - INFO - el_main.py - <module> - 393 - ====================================
2021-09-07 12:13:14,609 - INFO - el_main.py - <module> - 394 - 候选实体名：三
2021-09-07 12:13:14,609 - INFO - el_main.py - <module> - 395 - 知识库实体名：三
2021-09-07 12:13:14,609 - INFO - el_main.py - <module> - 396 - 知识库ID：254618
2021-09-07 12:13:14,609 - INFO - el_main.py - <module> - 397 - 置信分数：0.3548399
2021-09-07 12:13:14,609 - INFO - el_main.py - <module> - 398 - 类型：Vocabulary
2021-09-07 12:13:14,609 - INFO - el_main.py - <module> - 399 - 描述：摘要，三，数名，二加一(在钞票和单据上常用大写“叁”代)。三维空间。三部曲。三国(中国古代一个历史时期)。外文名，ㄙㄢ three 3 Ⅲ。词性，数词、名词。拼音，san。笔画数，3。五笔，dggg。......
2021-09-07 12:13:14,609 - INFO - el_main.py - <module> - 400 - ====================================
2021-09-07 12:13:14,609 - INFO - el_main.py - <module> - 393 - ====================================
2021-09-07 12:13:14,610 - INFO - el_main.py - <module> - 394 - 候选实体名：》
2021-09-07 12:13:14,610 - INFO - el_main.py - <module> - 395 - 知识库实体名：书名号
2021-09-07 12:13:14,610 - INFO - el_main.py - <module> - 396 - 知识库ID：219092
2021-09-07 12:13:14,610 - INFO - el_main.py - <module> - 397 - 置信分数：0.08161632
2021-09-07 12:13:14,610 - INFO - el_main.py - <module> - 398 - 类型：Thing
2021-09-07 12:13:14,610 - INFO - el_main.py - <module> - 399 - 描述：摘要，书名号是用于标明书名、篇名、报刊名、文件名、戏曲名、歌曲名、图画名等的标点符号，亦用于歌曲、电影、电视剧等与书面媒介紧密相关的文艺作品。书名号分为双书名号(《》)和单书名号(〈〉)，书名号里还有......
2021-09-07 12:13:14,610 - INFO - el_main.py - <module> - 400 - ====================================
2021-09-07 12:13:14,610 - INFO - el_main.py - <module> - 393 - ====================================
2021-09-07 12:13:14,610 - INFO - el_main.py - <module> - 394 - 候选实体名：紫萱
2021-09-07 12:13:14,610 - INFO - el_main.py - <module> - 395 - 知识库实体名：紫萱
2021-09-07 12:13:14,610 - INFO - el_main.py - <module> - 396 - 知识库ID：141031
2021-09-07 12:13:14,610 - INFO - el_main.py - <module> - 397 - 置信分数：0.8763746
2021-09-07 12:13:14,610 - INFO - el_main.py - <module> - 398 - 类型：FictionalHuman
2021-09-07 12:13:14,610 - INFO - el_main.py - <module> - 399 - 描述：摘要，紫萱，改编自同名单机游戏的电视剧《仙剑奇侠传三》中的第三女主角。由内地著名女演员唐嫣饰演，冯骏骅配音。她是女娲族后裔，与徐长卿情牵三世，不离不弃，饱受情爱煎熬三生三世之苦，爱情是她的执着，等待是......
2021-09-07 12:13:14,610 - INFO - el_main.py - <module> - 400 - ====================================
2021-09-07 12:13:14,610 - INFO - el_main.py - <module> - 393 - ====================================
2021-09-07 12:13:14,610 - INFO - el_main.py - <module> - 394 - 候选实体名：保护
2021-09-07 12:13:14,611 - INFO - el_main.py - <module> - 395 - 知识库实体名：保护
2021-09-07 12:13:14,611 - INFO - el_main.py - <module> - 396 - 知识库ID：179940
2021-09-07 12:13:14,611 - INFO - el_main.py - <module> - 397 - 置信分数：0.9528888
2021-09-07 12:13:14,611 - INFO - el_main.py - <module> - 398 - 类型：Vocabulary
2021-09-07 12:13:14,611 - INFO - el_main.py - <module> - 399 - 描述：摘要，保护，指尽力照顾，使自身(或他人、或其他事物)的权益不受损害。语出《书·毕命》“分居里，成 周 郊” 孔 传：“分别民之居里，异其善恶；成定 东周 郊境，使有保护。近义词，保卫。注音，bǎo h......
2021-09-07 12:13:14,611 - INFO - el_main.py - <module> - 400 - ====================================
2021-09-07 12:13:14,611 - INFO - el_main.py - <module> - 393 - ====================================
2021-09-07 12:13:14,611 - INFO - el_main.py - <module> - 394 - 候选实体名：林业平
2021-09-07 12:13:14,611 - INFO - el_main.py - <module> - 395 - 知识库实体名：林业平
2021-09-07 12:13:14,611 - INFO - el_main.py - <module> - 396 - 知识库ID：32716
2021-09-07 12:13:14,611 - INFO - el_main.py - <module> - 397 - 置信分数：0.93222207
2021-09-07 12:13:14,611 - INFO - el_main.py - <module> - 398 - 类型：FictionalHuman
2021-09-07 12:13:14,611 - INFO - el_main.py - <module> - 399 - 描述：摘要，林业平，是电视剧《仙剑奇侠传三》角色。长安玄道观掌门道长。一心向道，但遇到紫萱之后，动了情愫，作品将其设定为一个性格十分坚韧的正面角色。女儿，林青儿。饰演，霍建华。其他名称，顾留芳、徐长卿、业平......
2021-09-07 12:13:14,611 - INFO - el_main.py - <module> - 400 - ====================================
2021-09-07 12:13:14,611 - INFO - el_main.py - <module> - 393 - ====================================
2021-09-07 12:13:14,611 - INFO - el_main.py - <module> - 394 - 候选实体名：原型
2021-09-07 12:13:14,611 - INFO - el_main.py - <module> - 395 - 知识库实体名：原型
2021-09-07 12:13:14,612 - INFO - el_main.py - <module> - 396 - 知识库ID：290312
2021-09-07 12:13:14,612 - INFO - el_main.py - <module> - 397 - 置信分数：0.9565517
2021-09-07 12:13:14,612 - INFO - el_main.py - <module> - 398 - 类型：CreativeWork
2021-09-07 12:13:14,612 - INFO - el_main.py - <module> - 399 - 描述：摘要，原型，汉语词语，读音为yuán xíng。指原来的类型或模型，特指文学艺术作品中塑造人物形象所依据的现实生活中的人。注释，指原来的类型或模型。外文名，model，prototype，archet......
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

# 讲在最后
- 该项目是对实体链接的一种尝试，采用的是BERT的句子对模式，即[CLS]实体描述[SEP]查询句子[SEP]，利用二分类的思想，取出实体所对应的向量和CLS进行拼接后再分类。在预测的时候采用的直接分词的方式，并判断每个词是否在实体库中，因此可能不大准确，可以利用命名实体识别的方式进行改进。
- 记得修改相关位置的路径为自己的路径。
