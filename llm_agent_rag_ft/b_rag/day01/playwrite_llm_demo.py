import openai
from playwright.sync_api import sync_playwright
import html2text
from conf import settings
client = openai.Client(api_key=settings.api_key,base_url=settings.base_url)
html_converter = html2text.HTML2Text()
html_converter.ignore_links = True
html_converter.ignore_images = True

def chat(prompt):
    response = client.chat.completions.create(
        model=settings.model_name,
        messages=[
            {"role":"system","content":'你是一个信息抽取专家,请从提供的markdown中抽取标题和作者,结果用json返回,格式如下,[{"title":"","author":""}]'},
            {"role":"user","content":prompt},
        ],

        temperature=0
    )
    return response.choices[0].message.content

def crawler():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("http://localhost:8080/explore/recommend")
        page.wait_for_selector(".waterfall-item")
        html = page.content()
        browser.close()
        return html



if __name__ == '__main__':
    # print(chat("生存还是毁灭"))
    html = crawler()
    md = html_converter.handle(html)
    html =html_converter.handle(html)
    print(chat(md))