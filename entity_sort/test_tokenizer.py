from transformers import BertTokenizer
from utils import tokenization

tokenizer = BertTokenizer.from_pretrained('../model_hub/chinese-bert-wwm-ext/vocab.txt')
tokens_a = '摘要，《龙在天涯》是一部香港动作片，由邓衍成导演、李连杰与周星驰主演，并于1989年上映。影片讲述的是两位中国武术成员在美国经历惨遭陷害，最终冲出重围的故事。中国武术团出访美国，威哥叛逃，而师弟阿利也被扣留在了美国。威哥凭借自己的功夫加入了黑社会，谁知在一次百万美金的毒品交易中威哥失手了……。外文名，dragon fight ，dragon kickboxer。导演，邓衍成 billy tang。制片人，林小乐。编剧，阮世生 sally nichols。片长，96分钟。对白语言，粤语。主演，李连杰，jet，周星驰，stephen chow。中文名，龙在天涯。发行公司，argentina video home (avh) 阿根廷。上映时间，1989年9月1日。imdb编码，tt0095542。拍摄地点，美国三藩市。分级，阿根廷：13 瑞典：15 英国：18。类型，动作。在线播放平台，爱奇艺。色彩，彩色。出品公司，罗维影业有限公司 中国香港。义项描述，1989年邓衍成执导电影。标签，动作电影。标签，电影作品。标签，电影。'
tokens_b = '《龙在天涯》电影_龙在天涯（dragon fight）在线观看高清全集完整版-乐...'
labels = ('dragon fight', 14, 25)
print(tokens_b[14:25+1])
# res = tokenizer.tokenize(text)
# print(res)
# # res = tokenizer.encode(text)
# # res = tokenizer.convert_ids_to_tokens(res)
# # print(res)
# # print(len(text))
# # print(len(res))
#
tokens_a = tokenization.BasicTokenizer().tokenize(tokens_a)
tokens_b  = tokenization.BasicTokenizer().tokenize(tokens_b)
# print(res2)
# print(len(res2))
# res2 = tokenizer.encode(res2)
# res2 = tokenizer.convert_ids_to_tokens(res2)
# print(res2)
encode_dict = tokenizer.encode_plus(text=tokens_a,
                                    text_pair=tokens_b,
                                    max_length=256,
                                    padding='max_length',
                                    truncation_strategy='only_first',
                                    return_token_type_ids=True,
                                    return_attention_mask=True)
print(encode_dict['input_ids'])
print(encode_dict['attention_mask'])
print(encode_dict['token_type_ids'])
print(tokenizer.convert_ids_to_tokens(encode_dict['input_ids']))

offset = encode_dict['token_type_ids'].index(1)
print(offset)
print(tokenizer.convert_ids_to_tokens(encode_dict['input_ids'])[offset:offset+len(tokens_b)])