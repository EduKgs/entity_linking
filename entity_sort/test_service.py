import requests

text = '恶魔猎手吧-百度贴吧--《魔兽世界》恶魔猎手职业贴吧...'
text = text.encode('utf-8')
url = 'http://0.0.0.0:1080/entity_linking'
result = requests.post(url, data=text)
result = result.text
print(result)