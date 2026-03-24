import requests

#
# curl -X POST -H "Content-Type: application/json" -d "{\"input\": {\"language\": \"英语\",\"content\":\"我爱北京天安门\"}}" http://192.168.10.14:60002/translate/invoke

response = requests.post(
    "http://127.0.0.1:60002/translate/invoke",
    json={'input': {'language': '英语','content':'我爱北京天安门'}}
)
rep = response.json()
print(rep["output"])