import requests
import base64


def image_to_base64(image_path,type):
    if type == 'text':
        return image_path
    with open(image_path, "rb") as image_file:
        # 读取二进制数据并编码为 Base64
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def rerank(query, candidates,query_type,answer_type):
    url = "http://localhost:6006/rank"
    payload = {
        "text_1": query,
        "text_2": candidates,
        "query_type": query_type,
        "answer_type": answer_type
    }
    response = requests.post(url, json=payload)
    return response.json()

query = "海滩上面有什么"
query_image = image_to_base64("./3.png",'image')

mixed_candidates = [
    {"id": 1, "type": "text", "content": "海鸥飞在海滩上"},
    {"id": 1, "type": "text", "content": "沙漠里面有骆驼"},
    {"id": 2, "type": "image", "content": "./1.png"}, # 假设存在这张图
    {"id": 3, "type": "text", "content": "椰子树在海边高高地长大"},
    {"id": 4, "type": "image", "content": "./2.png"}    # 假设存在这张图
]

print(len(mixed_candidates))
num = 1
for candidate in mixed_candidates:
    print(f"num: {num}")
    # results = rerank(query, image_to_base64(candidate['content'],candidate['type']), 'text', candidate['type'])
    results = rerank(query_image, image_to_base64(candidate['content'],candidate['type']), 'image', candidate['type'])
    print(results)
    num += 1


