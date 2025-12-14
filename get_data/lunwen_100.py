from playwright.sync_api import sync_playwright
import re
from conf import settings


from sqlmodel import Field, Session, SQLModel, create_engine,select
from sqlalchemy import TEXT

from rag.llm import chat




class Wenzhang(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)  
    title: str  
    content: str = Field(sa_type=TEXT)

    class Config:
        table_name = "wenzhang"  # 可选：指定表名

    def insert(self):
        with Session(engine) as session:
            session.add(self)
            session.commit()

engine = create_engine(settings.url(), echo=True)  


def create_db_and_tables():  
    SQLModel.metadata.create_all(engine)  




def is_english(text):
    return all(ord(char) < 128 for char in text)

def is_gaokao_full_score_essay_title(text):
    pattern = r'^高考满分英语作文\s+\d+$'
    return bool(re.match(pattern, text))


def keep_only_english_chars(text):

    pattern = r'[^a-zA-Z0-9\s.,!?;:\'\"\-]'
    result = re.sub(pattern, ' ', text)

    result = ' '.join(result.split())
    
    return result

def extract_articles_from_content(content_element):
    """从.content元素中提取文章"""
    articles = []
    
    # 获取所有直接子元素
    children = content_element.locator("xpath=./*").all()
    
    current_title = None
    current_content = []
    
    for child in children:
        tag_name = child.evaluate("el => el.tagName.toLowerCase()")
        text = child.text_content().strip()
        
        if not text:  # 跳过空文本
            continue
            
        if tag_name == "h2":
            # 保存上一篇文章
            if current_title is not None:
                articles.append({
                    "title": current_title,
                    "content": "\n".join(current_content).strip(),
                    "is_english": all(is_english(p) for p in current_content)
                })
            
            # 开始新文章
            current_title = text
            current_content = []
        elif tag_name == "p" and current_title is not None and is_english(text):
            current_content.append(text)
        elif current_title is not None:
            # 处理其他标签（如div, span等）中的文本
            current_content.append(text)
    
    # 添加最后一篇文章
    if current_title is not None:
        articles.append({
            "title": current_title,
            "content": "\n".join(current_content).strip(),
            "is_english": all(is_english(p) for p in current_content)
        })
    
    return articles

def main():
    create_db_and_tables()
    url = "https://www.ruiwen.com/zuowen/gaokaoyingyuzuowen/8073108.html"
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")
        
        # 等待内容加载
        page.wait_for_selector(".content", timeout=10000)
        
        all_articles = []
        
        # 处理每个.content元素
        content_elements = page.locator(".content").all()
        
        for i, content_elem in enumerate(content_elements):
            print(f"处理第 {i+1} 个.content元素...")
            articles = extract_articles_from_content(content_elem)
            all_articles.extend(articles)
        
        # 输出结果
        print(f"\n共找到 {len(all_articles)} 篇文章:")
        print("=" * 60)
        


        
        for i, article in enumerate(all_articles):
            title = article['title']
            content = article['content']
            wenzhang = Wenzhang(title=title, content=content)

            wenzhang.insert()
            print(f"第 {i+1} 篇文章:")
            
        
        browser.close()



def change_data():
    with Session(engine) as session:
        wenzhangs = session.query(Wenzhang).all()
        for wenzhang in wenzhangs:

          
            content = keep_only_english_chars(wenzhang.content)

            prompt = f"请为接下来的文章生成一个摘要，要求精炼，包含提取到的关键词，只回答我摘要，不要包含别的内容，摘要使用中文：\n{content}"

            result = chat(prompt,[],system_prompt='你是一名摘要提取模型，你的任务是为给定的文章生成一个摘要，摘要应包含文章的主要内容和关键词，只回答我摘要，不要包含别的内容，摘要使用中文。')

            statement = select(Wenzhang).where(Wenzhang.id == wenzhang.id)
            results = session.exec(statement)
            wenzhang = results.one()

            wenzhang.content = content
            wenzhang.title = result
            session.add(wenzhang)
            session.commit()
            session.refresh(wenzhang)

            
            

if __name__ == "__main__":
    # change_data()
    pass
    
