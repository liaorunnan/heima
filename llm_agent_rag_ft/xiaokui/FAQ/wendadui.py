import json

from xiaokui.fqg_table import Wendadui
import os
import requests
from bs4 import BeautifulSoup
from xiaokui.llm import chat

def in_es(file_path):
    # 创建索引（如果不存在）
    Wendadui.init()
    i = 0
    # 读取 markdown 文件

    for filename in os.listdir(file_path):
        # print(filename)
        if filename.endswith(".md"):
            path = os.path.join(file_path, filename)

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                # print(content)

                # print(response.text)
                soup = BeautifulSoup(content, "html.parser")
                # 找到表格中的所有行（<tr>）
                rows = soup.find_all('tr')

                # 跳过第一行（表头），处理数据行
                vocabulary = []
                for row in rows[1:]:  # 从第二行开始
                    cells = row.find_all('td')
                    word = cells[1].get_text(strip=True)
                    pronunciation = cells[2].get_text(strip=True)
                    mean = cells[3].get_text(strip=True)


                    print("*"*30)


                    query = word+mean+'请根据这个单词生成问答对,比如这个单词是什么意思?这个单词怎么读怎么用等等,每一个单词都要有1个问答对,格式为json:[{"query":"","answer":""}]，不要给多余的内容，不要解释'

                    print("*" * 30)
                    print(query)
                    repose = chat(query,[])

                    qas_tmp = json.loads(repose)

                    for qas in qas_tmp:
                        print(qas)
                        i = i+1
                        word_info = Wendadui(
                            meta={'id': i},
                            query=qas['query'],
                            answer=qas['answer'],
                        )
                        word_info.save()


                    print(qas_tmp)

                    print("*" * 30)





if __name__ == '__main__':
    in_es("../four_level_md")
