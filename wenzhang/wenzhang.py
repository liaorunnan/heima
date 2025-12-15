import openai
from conf import settings
import json
import csv



client = openai.OpenAI(
    api_key=settings.api_key,
    base_url=settings.base_url
)

import wenzhang.wenzhang_toollist as wenzhang_toollist

from translate_baidu import baidu_ai_translate



toolslist = [getattr(wenzhang_toollist,attr) for attr in wenzhang_toollist.__dir__() if attr.endswith('_tool')]





class SummarizeAgent:
    def __init__(self, train_csv_path='./train.csv'):
        self.history_examples = self._load_data(train_csv_path)
        self.system_prompt = f"你是一个专业的摘要助手。请阅读用户提供的文章，提炼核心观点你是一个专业的文章摘要助手。你的唯一任务是阅读用户提供的文章，提炼核心观点，并**必须通过调用提供的工具**来提交结果。请勿直接在对话中输出摘要文本，必须使用工具传输数据。并最后调用翻译工具来输出英文摘要。"

    def _load_data(self, path, limit=3):
        examples = []
    
        with open(path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            
            for i, row in enumerate(reader):
                if i >= limit: break
                if len(row) < 2: continue
                
                # 构造历史对话：模拟之前的 User 输入和 Assistant 的工具调用
                examples.append({"role": "user", "content": row[0]})
                examples.append({"role": "assistant", "content": f"已调用工具提交摘要：{row[1]}"})

        return examples

    def run(self, article: str):
        
       
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history_examples)
        messages.append({
            "role": "user", 
            "content": f"请总结以下文章：\n{article}"
        })


       
        response = client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=messages,
            tools=toolslist,
            tool_choice="auto",
            temperature=0.1,
        )

        response_message = response.choices[0].message

        print(response_message)

        if response_message.tool_calls:
            tool_call = response_message.tool_calls[0]
            function_args = json.loads(tool_call.function.arguments)
            summary = function_args.get("summary")
            return baidu_ai_translate(summary, "auto", "en")
        else:
            return response_message.content

       


if __name__ == '__main__':
    agent = SummarizeAgent()
    
    article_text = """丝绸之路不仅仅是一条连接东西方的贸易通道，它更是一张巨大的文明交流网络。起于中国古代的长安（今西安），经中亚、西亚，最终抵达欧洲的罗马，这条路线跨越了数千公里。在数千年的历史长河中，骆驼商队驮着的不仅是中国的丝绸、瓷器和西方的玻璃、香料，更有无形的文化财富。
通过丝绸之路，中国的造纸术、火药和印刷术传入了西方，极大地推动了欧洲文明的进程；与此同时，西方的佛教艺术、天文历法以及葡萄、胡萝卜等农作物也传入了中国，丰富了东方人的生活。可以说，丝绸之路是人类历史上第一次全球化浪潮的缩影，它证明了不同文明之间的交流与互鉴是推动历史进步的重要动力。"""
    
    result = agent.run(article_text)
    
    # 这里得到的一定是干净的字符串，没有 Markdown 符号，没有 JSON 括号
    print("Agent 输出的摘要结果：")
    print(result)