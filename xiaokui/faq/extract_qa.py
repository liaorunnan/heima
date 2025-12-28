import json
import os
import simple_pickle as sp
from tqdm import tqdm

from b_rag.xiaokui.tool.llm import chat




qas = []
path = "../../xiaokui/data/results/words_output2.txt"
# 读取这个txt文件，并且打印每一行
def get_pkl():
    with open(path, "r", encoding="utf-8") as f:
        for text in tqdm(f):
            system_prompt = '我们在做一个英语学习app，请根据四级词汇，输出学生们可能问的问题，并给出答案。格式为json:[{"query":"","answer":""}]，不要给多余的内容，不要解释,注意是学习英语的大学生会问的问题'
            user_prompt = f'以下为英语学习的教材：{text}'
            print(text)
            qas_tmp = chat(user_prompt, system_prompt=system_prompt)
            print(qas_tmp)
            qas_tmp = json.loads(qas_tmp)
            qas.extend(qas_tmp)

            sp.write_pickle(qas, "../../xiaokui/data/qas.pkl")


# 读取这个pkl文件，并且打印每一行
def read_pkl():
    qas = sp.read_pickle("../../xiaokui/data/qas.pkl")
    print(qas)
    # for qa in qas:
    #     print(qa)

if __name__ == '__main__':
    # get_pkl()
    read_pkl()


