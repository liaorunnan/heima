from chat_llm import chat

from playwright.sync_api import sync_playwright
from prompt_bash import *

from tqdm import tqdm
from datetime import datetime, timedelta
# 使用相对导入访问上级目录中的email模块
from sendemail import send_email  
 

if __name__ == '__main__':

    email_context = ""
    
    # 获取当前日期和时间
    now = datetime.now()
    today = now.date()
    
    # 确定需要访问的since参数
    # 检查是否是每月最后一天
    is_last_day_of_month = (today + timedelta(days=1)).month != today.month
    # 检查是否是每周一（0表示周一，1表示周二，以此类推）
    is_monday = today.weekday() == 0
    
    # 根据日期条件选择需要访问的since参数列表
    since_params = []
    
    # 每天都需要访问daily
    since_params.append("daily")

    is_monday = True
    # 如果是周一，添加weekly
    if is_monday:
        since_params.append("weekly")

    is_last_day_of_month = True
    # 如果是每月最后一天，添加monthly
    if is_last_day_of_month:
        since_params.append("monthly")
    
    print(f"当前日期: {today}")
    print(f"需要访问的since参数: {since_params}")
    
    with sync_playwright() as p:
        browser = p.webkit.launch()  # 恢复无头模式
        
        # 遍历所有需要访问的since参数
        for since in since_params:
            email_context = f"\n\n# {since}\n"
            page = browser.new_page() 
            url = f"https://github.com/trending?since={since}"
            print(f"访问URL: {url}")
            page.goto(url)
            page.wait_for_load_state("networkidle")  # 等待页面完全加载
            
            Box_div = page.locator(".Box .Box-row").all()    
            print(f"找到 {len(Box_div)} 个热门仓库")
            
            # 为不同的since参数添加标题
            if since == "daily":
                since_text = "今日热门仓库"
            elif since == "weekly":
                since_text = "本周热门仓库"
            else:  # monthly
                since_text = "本月热门仓库"
            
            email_context += f"\n\n## {since_text}\n"
            
            # 写到文件中，添加since参数对应的标题
            with open("agent_bash/github_trending.md", "a", encoding="utf-8") as f:
                f.write(f"\n\n# {since_text}\n")
            
            for i, in_box in tqdm(enumerate(Box_div), desc=f"处理{since_text}"):  # 处理前10个仓库
                title_element = in_box.locator("h2 a")
                if title_element.count() > 0:
                    href = title_element.get_attribute("href")
                    text = title_element.inner_text().strip()
                    pa_href = f"https://github.com{href}"
                    summary = chat(pa_href, [], system_prompt=GITHUB_PROMPT,temperature=0.0)
                    
                    # 添加到邮件内容
                    email_context += f"\n{i+1}. {text}\n{summary}\n"
                    
                    # 写到文件中
                    with open(f"agent_bash/github_trending_{since}.md", "a", encoding="utf-8") as f:
                        f.write(f"## {i+1}. {text}\n")
                        f.write(f"{summary}\n")  

                # 发送邮件
            send_email(f"{since_text}趋势", email_context)    
            
            # 关闭当前页面
            page.close()
        
        # 关闭浏览器
        browser.close()

        