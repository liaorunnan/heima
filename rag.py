wav2vec2import numpy as np
import openai
from conf import settings



client = openai.Client(api_key=settings.API_KEY,base_url=settings.BASE_URL)

texts = [
    "我爱北京天安门",
    "我爱我的祖国",
    "我爱我的家乡",
    "北京天安门是中国的象征"
]

def match(query):
    # 创建一个空集合，用于存储所有不重复的单词
    word_dict = set()
    # 遍历所有文本，将每个文本中的单词添加到集合中
    for text in texts:
        for word in text:
            word_dict.add(word)

    # 将查询文本中的单词也添加到集合中
    for word in query:
        word_dict.add(word)

    # 将集合转换为列表，便于后续索引操作
    word_dict = list(word_dict)

    # 创建一个空列表，用于存储所有文本的向量表示
    vecs = []
    # 遍历每个文本，将其转换为词频向量
    for text in texts:
        # 创建一个与词表长度相同的零向量
        vec = np.zeros(len(word_dict))
        # 统计文本中每个词出现的频率
        for word in text:
            vec[word_dict.index(word)] += 1
        # 将生成的向量添加到向量列表中
        vecs.append(vec)


    # 创建查询文本的向量表示
    query_vec = np.zeros(len(word_dict))
    # 统计查询文本中每个词出现的频率
    for word in query:
        query_vec[word_dict.index(word)] += 1

    # 计算查询向量与所有文本向量的余弦相似度（点积）
    scores = np.dot(vecs,query_vec)
    # 将相似度得分从小到大排序，得到对应的索引
    indexs = np.argsort(scores)

    # 返回相似度最高的两个文本（使用逆序排序获取最高分）
    return texts[indexs[::-1][:2]]


            
            

def ranker(query,docs):
    pass
    

def gen(query,docs):
    prompt = f"根据以下文档来回答问题,如果找不到相关信息,请说'我找不到相关信息',不要自己编造答案： 文档1:{docs[0]},文档2:{docs[1]},问题:{query}"
    answer = client.chat.completions.create(
        model=settings.MODEL_NAME,
        messages=[
           
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
    )

    return answer.choices[0].message.content

if __name__ == '__main__':
    print(match("我的家是哪里?"))

