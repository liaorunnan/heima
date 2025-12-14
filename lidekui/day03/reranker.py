import requests
from heima.conf import settings

def rank_score(text_1, text_2, model="bge-reranker-v2-m3"):
    url = f"{settings.vllm_rerank_url}/score"
    data = {
        "model": model,
        "encoding_format": "float",
        "text_1": text_1,
        "text_2": text_2
    }

    response = requests.post(url, json=data)
    score = response.json()['data'][0]['score']
    return score

def get_books_topk(query, books, topk=10):
    for i in range(len(books)):
        books[i].score = rank_score(query, books[i].parent)
    books.sort(key=lambda x: x.score, reverse=True)
    return books[:topk]

def get_laws_topk(query, laws, topk=10):
    for i in range(len(laws)):
        laws[i].score = rank_score(query, laws[i].embedding_text)
    laws.sort(key=lambda x: x.score, reverse=True)
    return laws[:topk]

if __name__ == '__main__':
    result = rank_score(
        "猴子",
        "猴子捞月"
    )
    print(result)
