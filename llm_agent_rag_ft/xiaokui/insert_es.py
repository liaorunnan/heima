from fqg_table import Wendadui
import os
import requests
from bs4 import BeautifulSoup

def in_es(file_path):
    # 创建索引（如果不存在）
    Word.init()
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
                    if len(cells) >= 4:
                        word = cells[1].get_text(strip=True)
                        pronunciation = cells[2].get_text(strip=True)
                        mean = cells[3].get_text(strip=True)
                        vocabulary.append({
                            'word': word,
                            'pronunciation': pronunciation,
                            'mean': mean
                        })

                # 打印结果示例
                for item in vocabulary[:5]:  # 只打印前5个
                    print(f"单词: {item['word']}")
                    print(f"注音: {item['pronunciation']}")
                    print(f"释义: {item['mean']}")
                    print("-" * 40)
                for item in vocabulary:
                    word_info = Word(
                        meta={'id': i},
                        word=item['word'],
                        pronunciation=item['pronunciation'],
                        mean=item['mean']
                    )
                    i += 1
                    word_info.save()




if __name__ == '__main__':
    in_es("four_level_md")
