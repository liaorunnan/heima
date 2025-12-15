

from conf import settings
from elasticsearch_dsl import Document, Date, Integer, Keyword, Text, connections

connections.create_connection(hosts=settings.es_host, http_auth=(settings.es_user, settings.es_password),
                              verify_certs=False, ssl_assert_hostname=False)

from rag.items import YinyutlItem
from elasticsearch_dsl.query import Script


import urllib3
import warnings
import tqdm
from rag.pdf import PDF
from bs4 import BeautifulSoup
from pathlib import Path
import os




# 屏蔽 InsecureRequestWarning 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", message="Connecting to 'https://localhost:9200' using TLS")

from rag.text_splitter import split_text, get_html



class Yinyutl(Document):
    child = Text(analyzer="smartcn")
    parent = Text(analyzer="smartcn")
    source = Keyword()

    class Index:
        name = 'yinyutl'
        settings = {
            "number_of_shards": 2,
        }

    def query(self, item,type=''):
  
        items = self.search().query("match", child=item)[:10].execute()
        return [YinyutlItem(id=item.meta.id, child=item.child, parent=item.parent, source=item.source) for item in items]


    @classmethod
    def scan(self):
        s = self.search()
      
        for item in s.scan():
        
            yield item


if __name__ == '__main__':

    pass

    # Yinyutl.init()

    # ES_data = Yinyutl()

    
    # items = ES_data.scan()
    # for item in items:
    #     print(item.child)
    #     exit()
    





    exit()
    file_path = "data/wenzhang/"
    parent_data = ''

    num = 1

    for i in tqdm.tqdm(range(1,20)):
        filename = f'{i}.md'
        with open(file_path + filename, 'r') as f:
            data = f.read()
            data = split_text(data)
            
            for item in data:
                yinyutl = Yinyutl(meta={'id': num}, child=item['child'], parent=item['parent'], source=['四级英语听力'])
                yinyutl.save()
                num += 1

    filename_dirt = {'六级范文':'英语范文','六级范文2':'英语范文','四级范文':'英语范文','作文模版1':'英语作文模版','作文模版2':'英语作文模版','作文模版3':'英语作文模版','作文模版4':'英语作文模版'}

    
    for filename, source in filename_dirt.items():
        pdf = PDF(file_path + filename+'.pdf')
        pdfdata = pdf.get_text()
        for page_data in tqdm.tqdm(pdfdata):
            data = split_text(page_data)

            for item in data:
                yinyutl = Yinyutl(meta={'id': num}, child=item['child'], parent=item['parent'], source=[source])
                yinyutl.save()
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
            yinyutl = Yinyutl(meta={'id': num}, child=item['child'], parent=item['parent'], source=[item['source']])
            yinyutl.save()
            num += 1
            

    print(f'已保存 {num} 条数据')

              
        





