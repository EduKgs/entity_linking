import my_jieba
import json
from utils import tokenization


my_jieba.load_userdict('./data/ccks2019/alias_and_subjects.txt')
#
text = '巴南区东温泉镇地处巴南区东部，巴南旅游双环线上。2001年7月，乡镇建制调整后由原来东泉、五布镇、天赐镇、清和乡合并而成。东与南川市白沙镇接壤，南与石龙镇、姜家镇为邻，西连二圣镇，北接木洞、丰盛镇。幅员面积122.70平方公里，耕地面积50862亩，其中田25376亩。镇政府所在地距重庆市区50余公里。现辖14个行政村、4个居委会、136个村民小组，总户数11869户，总人口37856人，其中农业人口35188人，非农业人口2668人。'
text = text.lower()
res = my_jieba.lcut(text, cut_all=False)
res2 = tokenization.BasicTokenizer().tokenize(text)
print(res)
print(res2)






