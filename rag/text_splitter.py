import re
from bs4 import BeautifulSoup
from pathlib import Path
import os

def split_text(text, max_length=512):
    chunks = []

    for para in re.split(r'\n+', text.strip()):
        if not para: continue
        if len(para) <= max_length:
            chunks.append(para)
            continue

        current = [para]
        for delim in ['[。！？…….!?]', '[；;]', '[，,]']:
            new = []
            for chunk in current:
                if len(chunk) <= max_length:
                    new.append(chunk)
                else:
                    parts = re.split(f'({delim})', chunk)
                    parts = [''.join(parts[i:i + 2]) for i in range(0, len(parts) - 1, 2)]
                    new.extend(parts)
            current = new

        for chunk in current:
            if len(chunk) <= max_length:
                chunks.append(chunk)
            else:
                chunks.extend([chunk[i:i + max_length] for i in range(0, len(chunk), max_length)])

    return [{
        'child': chunk.strip(),
        'parent': (chunks[i - 1] if i > 0 else '') + chunk + (chunks[i + 1] if i < len(chunks) - 1 else '')
    } for i, chunk in enumerate(chunks) if chunk.strip()]



def get_html(file_content):
    docs = []
    soup = BeautifulSoup(file_content, 'html.parser')
    rows = soup.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        col_list = [c.get_text(strip=True) for c in cols]
        if "单词" in col_list[1]:
            continue
        word = col_list[1]      
        yinbiao = col_list[2]  
        shiyi = col_list[3]
        child_text = f"{word} {shiyi}"
        parent_text = (
            f"【单词】: {word}\n"
            f"【音标】: {yinbiao}\n"
            f"【释义】: {shiyi}"
        )
        doc = {
            "child": child_text,
            "parent": parent_text,
            "source": "四级单词"
        }
        docs.append(doc)
            
    return docs




if __name__ == "__main__":
    pass
    # with open('data/wenzhang/1.md', 'r', encoding='utf-8') as f:
    #     text = f.read()
        
    #     result = split_text(text)
    #     print(result)
    #     exit()
        
    #     for i, item in enumerate(result,10):
    #         print(f"块{i + 1}: 子({len(item['child'])}): {item['child']}")
    #     exit()

    
    # folder_path = 'data/wenzhang/four_level_md'
    
    # p = Path(folder_path)
    # md_files = list(p.glob('*.md'))
    # all_data= []
    # for file_path in md_files:
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #         content = f.read()
    #     source_name = file_path.stem 
    #     file_docs = get_html(content)
    #     for doc in file_docs:
    #         print(doc)
    #         print(doc['child'])
    #         print(doc['parent'])
    #         print(doc['source'])
    #         exit()


    
        


    