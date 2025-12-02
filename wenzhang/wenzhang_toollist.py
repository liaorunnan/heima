

summary_tool = {
    "type": "function",
    "function": {
        "name": "submit_summary",
        "description": "提交生成的文章摘要。当摘要生成完毕后，必须调用此函数。",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "精简后的文章摘要内容，包含主要观点。"
                }
            },
            "required": ["summary"]
        }
    }
}

translate_tool = {
    "type": "function",
    "function": {
        "name": "translate_result",
        "description": "当摘要生成完毕后，调用此函数提交翻译结果。",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "精简后的文章摘要内容，包含主要观点。"
                }
            },
            "required": ["summary"]
        }
    }
}

# import requests
# from conf import settings

# url = "https://wenxin.baidu.com/younger/portal/api/rest/1.0/ernie/3.0/zeus"

# payload={
#     'access_token': '24.9991a77ac60c6e969e176b776f74b09e.86400000.1653006451543.499f8f33b6821ebbf9ba1fbea525d6ae-6',
#     'text': '19号，印度一些主流媒体发布消息称，汉语普通话被批准成为巴基斯坦官方语言！消息称，巴基斯坦参议院19号通过将汉语普通话作为官方语言的议案，如果普通话成为巴基斯坦官方语言，中巴关系会进一步深化， 两国人民在中巴经济走廊建设中的沟通也会变得更简单。到底是不是真消息呢？据记者了解，事实上，该决议只是提到鼓励学习中国官方语言，并没有提到汉语普通话要成为巴基斯坦的官方语言。文章标题是：',
#     'seq_len': 32,
#     'task_prompt': 'Summarization',
#     'dataset_prompt': '',
#     'topk': 1,
#     'stop_token': '》'
#     }

# response = requests.request("POST", url, data=payload)