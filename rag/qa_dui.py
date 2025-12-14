
import simple_pickle

file_path = "data/wenzhang/"
parent_data = ''

num = 1

for i in tqdm.tqdm(range(1,20)):
    filename = f'{i}.md'
    with open(file_path + filename, 'r') as f:
        data = f.read()
        data = split_text(data)
        
        for item in data:
            
            num += 1
            

filename_dirt = {'六级范文':'英语范文','六级范文2':'英语范文','四级范文':'英语范文','作文模版1':'英语作文模版','作文模版2':'英语作文模版','作文模版3':'英语作文模版','作文模版4':'英语作文模版'}


for filename, source in filename_dirt.items():
    pdf = PDF(file_path + filename+'.pdf')
    pdfdata = pdf.get_text()
    for page_data in tqdm.tqdm(pdfdata):
        data = split_text(page_data)

        for item in data:
            
            num += 1

folder_path = 'data/wenzhang/four_level_md'

p = Path(folder_path)
md_files = list(p.glob('*.md'))
all_data= []
for file_path in tqdm.tqdm(md_files):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    source_name = file_path.stem 
    file_docs = get_html(content)
    for item in file_docs:

        print(doc['child'])
        print(doc['parent'])
        exit()
        
        num += 1
        

print(f'已保存 {num} 条数据')