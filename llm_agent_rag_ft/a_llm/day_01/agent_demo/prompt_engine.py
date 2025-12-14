from unittest import TestCase
import openai
from conf import settings
import json

client = openai.Client(api_key=settings.api_key,base_url=settings.base_url)

class myTest(TestCase):
    def test_classification_fun1(self):
        user_prompt= """
你是一个金融专家，需要对输入的金融领域文本进行分析，将类型归类到['新闻报道', '财务报告', '公司公告', '分析师报告']，对于不在这四类中的数据，输出‘不清楚类型’，不要有多余的解释，只给答案。以下是几个示例分类：
今日，股市经历了一轮震荡，受到宏观经济数据和全球贸易紧张局势的影响。投资者密切关注美联储可能的政策调整，以适应市场的不确定性。是['新闻报道', '财务报告', '公司公告', '分析师报告']里的什么类别？
新闻报道
本公司年度财务报告显示，去年公司实现了稳步增长的盈利，同时资产负债表呈现强劲的状况。经济环境的稳定和管理层的有效战略执行为公司的健康发展奠定了基础。是['新闻报道', '财务报告', '公司公告', '分析师报告']里的什么类别？
财务报告
本公司高兴地宣布成功完成最新一轮并购交易，收购了一家在人工智能领域领先的公司。这一战略举措将有助于扩大我们的业务领域，提高市场竞争力是['新闻报道', '财务报告', '公司公告', '分析师报告']里的什么类别？
公司公告
最新的行业分析报告指出，科技公司的创新将成为未来增长的主要推动力。云计算、人工智能和数字化转型被认为是引领行业发展的关键因素，投资者应关注这些趋势是['新闻报道', '财务报告', '公司公告', '分析师报告']里的什么类别？
"""

        response = client.chat.completions.create(
            model=settings.model_name,
            messages=[{"role":"user","content":user_prompt}],
            temperature=0,
        )
        print(response.choices[0].message.content)

    def test_classification_fun2(self):
        system_prompt= "现在你需要帮助我完成文本匹配任务，当我给你两个句子时，你需要回答我这两句话语义是否相似。只需要回答是否相似，不要做多余的回答。"
        user_prompt="句子一: 股票市场今日大涨，投资者乐观。\n句子二: 持续上涨的市场让投资者感到满意。\n上面两句话是相似的语义吗？"
        history=[
    {'role': 'user', 'content': '句子一: 公司ABC发布了季度财报，显示盈利增长。\n句子二: 财报披露，公司ABC利润上升。\n上面两句话是相似的语义吗？'},
    {'role': 'assistant', 'content': '是'},
    {'role': 'user', 'content': '句子一: 黄金价格下跌，投资者抛售。\n句子二: 外汇市场交易额创下新高。\n上面两句话是相似的语义吗？'},
    {'role': 'assistant', 'content': '不是'},
    {'role': 'user', 'content': '句子一: 央行降息，刺激经济增长。\n句子二: 新能源技术的创新。\n上面两句话是相似的语义吗？'},
    {'role': 'assistant', 'content': '不是'},
        ]

        a=client.chat.completions.create(
            model=settings.model_name,
            messages=[{"role":"system","content":system_prompt},
                      *history,
                      {"role":"user","content":user_prompt }
                      ],
            temperature=0
        )
        print(a.choices[0].message.content)

    def test_demo3(self):
        system_prompt= "你是一个金融专家，需要对输入的金融领域文本进行分析，将类型归类到['新闻报道', '公司公告', '财务公告', '分析师报告'],对于不在这四类中的数据，输出‘不清楚类型’，不要有多余的解释，只给答案。"
        user_prompt="最新的分析报告指出，可再生能源行业预计将在未来几年经历持续增长，投资者应该关注这一领域的投资机会"
        history=[
            {"role":"system","content":system_prompt},
            {"role":"system","content":system_prompt},
            {"role":"system","content":system_prompt},
            {"role":"system","content":system_prompt},
            {"role":"system","content":system_prompt},
            {"role":"system","content":system_prompt}
        ]
        answer = client.chat.completions.create(
            model=settings.model_name,
            messages=[{"role": "system", "content": system_prompt},
                      {'role': 'user',
                       'content': '今日，央行发布公告宣布降低利率，以刺激经济增长。这一降息举措将影响贷款利率，并在未来几个季度内对金融市场产生影响。'},
                      {'role': 'assistant', 'content': '新闻报道'},
                      {'role': 'user',
                       'content': 'ABC公司今日发布公告称，已成功完成对XYZ公司股权的收购交易。本次交易是ABC公司在扩大业务范围、加强市场竞争力方面的重要举措。据悉，此次收购将进一步巩固ABC公司在行业中的地位，并为未来业务发展提供更广阔的发展空间。详情请见公司官方网站公告栏'},
                      {'role': 'assistant', 'content': '公司公告'},
                      {'role': 'user',
                       'content': '今日，央行发布公告宣布降低利率，以刺激经济增长。这一降息举措将影响贷款利率，并在未来几个季度内对金融市场产生影响。'},
                      {'role': 'assistant', 'content': '财务公告'},
                      {"role": "user", "content": user_prompt}],
            temperature=0
        )
        print(answer.choices[0].message.content)


    def test_into_exstract(self):
        system_prompt="你是信息提取专家，需要需要完成信息抽取任务。我会给你一个句子，你需要提取句子中的实体，并按照JSON格式输出，如果句子中有不存在的信息用['原文中未提及']来表示"
        user_prompt="2023-02-15，寓意吉祥的节日，股票佰笃[BD]美股开盘价10美元，虽然经历了波动，但最终以13美元收盘，成交量微幅增加至460,000，投资者情绪较为平稳。"
        history=[
    {'role': 'user', 'content': '2023-01-10，股市震荡。股票古哥-D[EOOE]美股今日开盘价100美元，一度飙升至105美元，随后回落至98美元，最终以102美元收盘，成交量达到520000。'},
    {'role': 'assistant', 'content': "{'日期':['2023-01-10'],'股票名称':['古哥-D[EOOE]美股'],'开盘价':['100美元'],'收盘价':['102美元'],'成交量':['520000']}"}
]
        schema={
            "type":"object",
            "properties":{
                "日期":{"type":"array","items":{"type":"string"}},
                "股票名称":{"type":"array","items":{"type":"string"}},
                "开盘价":{"type":"array","items":{"type":"string"}},
                "收盘价":{"type":"array","items":{"type":"string"}},
                "成交量":{"type":"array","items":{"type":"string"}}
            }
        }
        answer=client.chat.completions.create(
            model=settings.model_name,
            messages=[
                {"role":"system","content":system_prompt},
                {"role":"system","content":user_prompt},
                *history
            ],
            temperature=0,
            response_format={"type":"json_object","schema":schema}
        )
        print(json.loads(answer.choices[0].message.content))





