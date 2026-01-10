
"""

langgraph 使用浏览器搜索工具进行搜索，并返回搜索结果
添加了 限制工具调用次数：
    对于任何全自动的代理，合理控制调用次数都是至关重要的一环，对于LangGraph React Agent来说，
    只需要在Agent运行的时候设置{"recursion_limit": X}，即可限制智能体自主执行任务时的步数。
    设置recursion_limit为 4 ，即可限制智能体自主执行任务时的步数为 4 。

"""

from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError

from langchain_tavily import TavilySearch

from dotenv import load_dotenv 
load_dotenv(override=True)



search_tool = TavilySearch(max_results=5, topic="general")

tools = [search_tool]
model = init_chat_model(model="deepseek-chat", model_provider="deepseek")  

search_agent = create_react_agent(model=model, tools=tools)

try:
    response = search_agent.invoke(
        {"messages": [{"role": "user", "content": "请问北京今天天气如何？"}]},
        {"recursion_limit": 4},
    )
except GraphRecursionError:
    print("Agent stopped due to max iterations.")


print("=================================================    1    ===============================================================")
print(response["messages"])

print("=================================================    2    ===============================================================")
print(response["messages"][-1].content)



"""
response["messages"] 的结构：

[
HumanMessage(content='请问北京今天天气如何？', additional_kwargs={}, response_metadata={}, id='cb534851-fdfe-4fa2-96c6-c4f846b6c020'), 

AIMessage(content='我来帮您查询北京今天的天气情况。', additional_kwargs={'tool_calls': [{'id': 'call_00_c0DFK2FpcsdRTXmlDGfkhFtY', 'function': {'arguments': '{"query": "\\u5317\\u4eac\\u4eca\\u5929\\u5929\\u6c14", "time_range": "day", "search_depth": "basic"}', 'name': 'tavily_search'}, 'type': 'function', 'index': 0}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 57, 'prompt_tokens': 1804, 'total_tokens': 1861, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 1792}, 'prompt_cache_hit_tokens': 1792, 'prompt_cache_miss_tokens': 12}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_08f168e49b_prod0820_fp8_kvcache', 'id': 'fa978b9d-8d45-4286-9edb-35c011a9b5fb', 'service_tier': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run--56bff019-cac1-463e-b481-bee1d3064b8c-0', tool_calls=[{'name': 'tavily_search', 'args': {'query': '北京今天天气', 'time_range': 'day', 'search_depth':'basic'}, 'id': 'call_00_c0DFK2FpcsdRTXmlDGfkhFtY', 'type': 'tool_call'}], usage_metadata={'input_tokens': 1804, 'output_tokens': 57, 'total_tokens': 1861, 'input_token_details': {'cache_read': 1792}, 'output_token_details': {}}), 

ToolMessage(content='{"query": "北京今天天气", "follow_up_questions": null, "answer": null, "images": [], "results": [{"url": "http://bj.cma.gov.cn/fzlm/", "title": "北京市气象局", "content": "今天白天 最高气温26℃ 多云间阴，山区有阵雨 3、4级，阵风6、7级 偏北风 ·今天夜间 最低气温15℃ 多云转晴 1、2级 偏北风 · 明天白天 最高气温26℃晴 2、3级 北转南风", "score": 0.76279575, "raw_content": null}, {"url": "https://www.accuweather.com/zh/cn/beijing/101924/daily-weather-forecast/101924", "title": "北京, 北京市, 中國每日天氣", "content": "最大紫外線指數6.0 (不健康－敏感人士) 風東 5英里/小时 最大紫外線指數4.0 (溫和) 風東 4英里/小时 最大紫外線指數5.0 (溫和) 風東北偏北 6英里/小时 最大紫外線指數2.0 (良好) 風東 1英里/小时 最大紫外線指數5.0 (溫和) 風東北偏東 2英里/小时 最大紫外線指數5.0 (溫和) 風東北偏東 1英里/小时 最大紫外線指數4.0 (溫和) 最大紫外線指數5.0 (溫和) 最大紫外線指數5.0 (溫和) 風西北 2英里/小时 最大紫外線指數1.0 (良好) 風東北偏東 2英里/小时 最大紫外線指數5.0 (溫和) 最大紫外線指數5.0 (溫和) 風西北偏北 13英里/小时 最大紫外線指數5.0 (溫和) 風西北 4英里/小时 最大紫外線指數5.0 (溫和) 風東北 2英里/小时 最大紫外線指數3.0 (溫和) 風東北偏東 2英里/小时 最大紫外線指數2.0 (良好) 風東南偏東 2英里/小时 最大紫外線指數4.0 (溫和) 風西北偏北 4英里/小时 最大紫外線指 數4.0 (溫和) 風西北偏北 5英里/小时 最大紫外線指數1.0 (良好) 風東北 1英里/小时 最大紫外線指數2.0 (良好) 風西 1英里/小时 最大紫外線指數4.0 (溫和) 風北 6英里/小时 最大紫外線指數1.0 (良好) 風東北 2英里/小时 最大紫外線指數4.0 (溫和) 風西北 8英里/小时 最大紫外線指數4.0 (溫和) 最大紫外線指數4.0 (溫和) 最大紫外線指數4.0 (溫和) 風西北 5英里/小时 最大紫外線指數3.0 (溫和) 最大紫外線指數4.0 (溫和) 最大紫外線指數3.0 (溫和) 風西北偏北 7英里/小时 最大紫外線指數4.0 (溫和) 最大紫外線指數4.0 (溫和) 最大紫外線指數4.0 (溫和) 風東北偏東 4英里/小时 最大紫外線指數4.0 (溫和) 風 東北 9英里/小时 最大紫外線指數3.0 (溫和) 風東 2英里/小时 最大紫外線指數4.0 (溫和)", "score": 0.75809944, "raw_content": null}, {"url": "https://www.weather.com.cn/weather/101010100.shtml", "title": "预报- 北京", "content": "20日（今天）. 晴. 29/16℃. <3级. 21日（明天）. 多云", "score": 0.7418428, "raw_content": null}, {"url": "https://www.nmc.cn/publish/forecast/ABJ/beijing.html", "title": "北京-天气预报", "content": "北京 18.3℃ 西南风 微风 全球灾害性天气监测月报 环境气象公报 山洪灾害气象预警 地质灾害气象风险预警 中小河流洪水气象风险预警 渍涝风险气象预警  交通气象预报 森林火险预报 草原火险预报 台风海洋 台风快讯与报文 台风路径预报 台风公报 台风预警 海区预报 海事公报 海洋天气预报 近海海雾预报 + 北京 北京 14℃ 西南风 微风 西南风 微风 29℃ 16℃  微风 微风 26℃ 18℃ 微风 微风 25℃ 14℃ 西南风 微风 西南风 微风 25℃ 15℃ 西南风 微风 微风 27℃ 15℃ 西南风 微风 微风 27℃ 17℃ 微风 14.2℃ 19.5℃ 24.5℃ 28.8℃ 26.8℃ 西南风 21.9℃ 19.8℃ 18.7℃ 24.5℃ 28.8℃ 26.8℃ 西南风 21.9℃ 19.8℃ 18.7℃ 16.2℃ 19.6℃ 22.7℃ 25.8℃ 24.5℃ 21.7℃ 21.6℃ 18.2℃ 19.6℃ 20℃ 23.8℃ 24.8℃ 24.1℃ 22.8℃ 17.4℃ 西南风 15.7℃ 西南风 14.2℃ 19.3℃ 23.6℃ 24.8℃ 24.4℃ 西南风 19.4℃ 西南风 18℃ 西南风 17℃ 西南风 15.2℃ 18.8℃ 24.5℃ 26.8℃ 25.4℃ 22.5℃ 西南风 16.6℃ 西南风 15.4℃ 15.2℃ 18.6℃ 26.8℃ 24.8℃ 19.8℃ 18.7℃ 20℃", "score": 0.7180613, "raw_content": null}, {"url": "https://weather.cma.cn/web/weather/54511.html", "title": "中国气象局-天气预报- 北京", "content": "27℃ 17℃ 24℃ 17℃ 28℃ 28℃ 17℃ 27℃ 16℃ | 气温 | 25.5℃ | 26.8℃ | 24.4℃ | 22.1℃ | 20.7℃ | 19℃ | 17.3℃ | 17.2℃ | | 气温 | 17.2℃ | 22.8℃ | 23.8℃ | 23℃ | 20.5℃ | 19.2℃ | 17.9℃ | 17.2℃ | | 气温 | 19.3℃ | 18.8℃ | 20.8℃ | 19.8℃ | 16.4℃ | 15.4℃ | 14.2℃ | 14.4℃ | | 气温 | 17.4℃ | 24.5℃ | 27.8℃ | 26.2℃ | 20.8℃ | 18.7℃ | 17.4℃ | 15.2℃ | | 气温 | 20℃ | 25.5℃ | 27.2℃ | 27.8℃ | 22.4℃ | 18.3℃ | 17.8℃ | 16.2℃ | | 气温 | 18.9℃ | 23.2℃ | 27.8℃ | 27.8℃ | 21.1℃ | 17.8℃ | 17.2℃ | 17.2℃ |", "score": 0.6867278, "raw_content": null}], "response_time": 1.52, "request_id": "b2049356-24c8-451e-b5e0-843231ad1819"}', name='tavily_search', id='c82747e2-5521-47d1-a370-5a6c68b8b0f8', tool_call_id='call_00_c0DFK2FpcsdRTXmlDGfkhFtY'), 

AIMessage(content='根据最新的天气预报信息，北京今天的天气情况如下：\n\n**今天白天：**\n- 最高气温：26℃\n- 天气：多云间阴，山区有阵雨\n- 风力：3-4级偏北风，阵风可达6-7级\n\n**今天夜间：**\n- 最低气温：15℃\n- 天气：多云转晴\n- 风力：1-2级偏北风\n\n**明天天气：**\n- 最高气温：26℃\n- 天气：晴\n- 风力：2-3级北转南风\n\n今天整体天气比较舒适，但山区可能会有阵雨，外出建议携带雨具。风力较大，请注意防风。夜间温度较低，建议适当添加衣物。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 155, 'prompt_tokens': 3731, 'total_tokens': 3886, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 1792}, 'prompt_cache_hit_tokens': 1792, 'prompt_cache_miss_tokens': 1939}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_08f168e49b_prod0820_fp8_kvcache', 'id': 'bf864d5e-e3b4-44c6-af3f-3d6604ae33e9', 'service_tier': None, 'finish_reason': 'stop', 'logprobs': None}, id='run--f2237cf7-49b3-4eee-bfe7-cceedf25ca86-0', usage_metadata={'input_tokens': 3731, 'output_tokens': 155, 'total_tokens': 3886, 'input_token_details': {'cache_read': 1792}, 'output_token_details': {}})]

"""



























