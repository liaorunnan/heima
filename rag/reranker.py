import requests
from conf import settings

def rank(text_1, text_2, model="bge-reranker-v2-m3"):
    url = settings.Rank_url
    data = {
        "model": model,
        "encoding_format": "float",
        "text_1": text_1,
        "text_2": text_2
    }

    response = requests.post(url, json=data)
    score = response.json()['data'][0]['score']
    return score

if __name__ == '__main__':
    result = rank(
        "What is the capital of France?",
        "The capital of France is Paris."
    )
    print(result)