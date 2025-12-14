import json
import os
import simple_pickle as sp
from tqdm import tqdm
from pathlib import Path

from rag.llm import chat
from lunwen_100 import keep_only_english_chars
from rag.text_splitter import get_html

path = "./data/wenzhang"
files = [file for file in os.listdir(path) if file.endswith('.md')]



qas = []
# for file in tqdm(files):
#     text = ''.join(sp.read_data(os.path.join(path, file)))

    
#     system_prompt = '我们在做英语培训app，请根据范文内容，输出学生可能询问的问题（有关于如何写作文的问题），并给出答案。格式为json:[{"query":"","answer":""}]，不要给多余的内容，不要解释,问题使用中文，后面跟上英文的翻译，答案使用中文,至少生成3个问答对。'
#     user_prompt = f'以下为课文内容：{keep_only_english_chars(text)}'
 
#     qas_tmp = chat(user_prompt,[], system_prompt=system_prompt)
#     print(qas_tmp)
#     qas_tmp = json.loads(qas_tmp)
#     qas.extend(qas_tmp)


folder_path = 'data/wenzhang/four_level_md'

p = Path(folder_path)
md_files = list(p.glob('*.md'))
all_data= []
num = 1
error_num = 1
for file_path in tqdm(md_files):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        source_name = file_path.stem 
        file_docs = get_html(content)
        for item in tqdm(file_docs):


            system_prompt = '我们在做英语培训app，请根据范文内容，输出学生可能询问的问题（查询单词的问题），并给出答案。格式为json:[{"query":"","answer":""}]，不要给多余的内容，问题必须要包含询问这个单词是什么意思、如何使用、或者他的中文用英语怎么说,问题和答案主体使用中文,至少生成3个问答对。'
            user_prompt = f'以下为课文内容：{item["parent"]}'
    
            qas_tmp = chat(user_prompt,[], system_prompt=system_prompt)
            try:
                qas_tmp = json.loads(qas_tmp)
                qas.extend(qas_tmp)
                sp.write_pickle(qas, "./qas_word.pkl")
            except Exception as e:
                print(file_docs)
                print(e)
                error_num += 1

                continue
            

        num += 1
print(f"total num: {num}, error num: {error_num}")