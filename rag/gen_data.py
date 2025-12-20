import json  # 用于JSON数据的序列化和反序列化
import simple_pickle as sp  # 自定义的pickle序列化模块
from pydantic import BaseModel  # 用于数据验证和设置的数据模型基类
from tqdm import tqdm  # 用于显示进度条

from rag.match_keyword import Yinyutl  # 从自定义模块导入Book类，用于扫描上下文
from rag.llm import chat  # 导入LLM聊天功能
from rag.prompts_pinggu import *  # 导入所有提示词模板
from loguru import logger

from rag.rag_api import rag_query

logger.add("rag_gendata.log", rotation="500 MB", encoding="utf-8")


# 定义Item数据模型，用于存储生成的问答数据
class Item(BaseModel):
    question: str  # 问题文本
    answer: str  # 生成的答案（初始为空，后续填充）
    ground_truth: str  # 真实答案（从上下文提取）
    contexts: str  # 上下文信息


# 评估问答对质量的函数
def crite_qa(question, answer, context):
    # 评估问题与上下文的关联性（接地性）
    groundedness_score = chat(question_groundedness_critique_prompt.format(context=context, question=question))
    # 评估问题是否独立（是否无需上下文即可理解）
    not_standalone_score = chat(question_standalone_critique_prompt.format(question=question))
    # 答案评估部分被注释掉了，可能暂时不需要或未使用
    # answer_score = chat(evaluation_prompt.format(question=question, answer=answer, ground_truth=context))
    # 打印调试信息
    print(context, groundedness_score, not_standalone_score)
    # 返回两个评估分数
    return groundedness_score, not_standalone_score


# 生成问答对并评估质量的函数
def gen_and_crite_qa(context):
    # 使用LLM生成问答对
    qa = chat(qa_generation_prompt.format(context=context),[])
    # 解析JSON格式的返回结果
    qa = json.loads(qa)
    # 提取问题和答案
    q, a = qa["事实型问题"], qa["答案"]
    # 评估生成的问题质量
    groundedness_score, not_standalone_score = crite_qa(question=q.strip("事实型问题："), answer=a.strip("答案："), context=context)
    # 打印调试信息
    print(context)
    print("-" * 50)
    print(q, a, sep="\n")
    print("-" * 50)
    print(groundedness_score, not_standalone_score)
    # 解析评估分数的JSON格式
    groundedness_score, not_standalone_score = json.loads(groundedness_score), json.loads(not_standalone_score)
    # 检查分数是否达到阈值（>=5表示质量较差）
    if all([int(groundedness_score["总评分"]) >= 5, int(not_standalone_score["总评分"]) >= 5]):
        # 如果质量较差，返回Item对象（answer字段为空）
        return Item(question=q.strip("事实型问题："), answer="", contexts=context, ground_truth=a.strip("答案："))


# RAG（检索增强生成）函数，用于根据上下文回答问题
def rag(query, context, return_doc=False,context_array=False):

    context = [{"role": "assistant", "content": context}]
    answer,doc = rag_query(query, context, return_doc=return_doc)

    if context_array:
        context_str = ''
        for item in doc:

            context_str += " "+item.parent

        doc = context_str
            


    return answer,doc


# 主程序入口
if __name__ == '__main__':
    contexts = [item.parent for item in Yinyutl.scan_moban()]
    results = []  # 存储结果的列表


    num = 0
    
    # 遍历所有上下文，显示进度条
    # for context in tqdm(contexts):
    #
    #     # 对每个上下文尝试3次生成
    #     for _ in range(3):
    #         item = gen_and_crite_qa(context)
    #         if item:
    #             # 如果生成了有效的Item，添加到结果列表并跳出内层循环
    #             results.append(item)
    #             logger.info(item)
    #             break
    #     num += 1
    #
    #     if num ==100:
    #         # 将结果保存为pickle文件
    #         sp.write_pickle(results, 'data_pinggu.pkl')
    
    # # 重新读取pickle文件（用于验证和后续处理）
    # results = sp.read_pickle('./data_pinggu.pkl')
    # # 将结果保存为JSONL格式（每行一个JSON对象）
    # with open('data_pinggu.jsonl', 'w', encoding='utf-8') as f:
    #     for item in tqdm(results):
    #         # 使用RAG为每个问题生成答案
            
    #         item.answer = rag(item.question, item.contexts)

    #         # 打印Item信息
    #         logger.info(item.model_dump())
    #         # 写入JSONL文件
    #         json.dump(item.model_dump(), f, ensure_ascii=False)
    #         f.write('\n')

    # 用于评估生成阶段的数据
    results = sp.read_pickle('./data_pinggu.pkl')
    with open('data_gen_eval.jsonl', 'w', encoding='utf-8') as f:
        for item in tqdm(results):
            
            item.answer,_ = rag(item.question, item.contexts, return_doc=True)
            print(item.model_dump())
            json.dump(item.model_dump(), f, ensure_ascii=False)
            f.write('\n')

    # 用于评估检索阶段阶段


    results = sp.read_pickle('./data_pinggu.pkl')
    with open('data_search_eval.jsonl', 'w', encoding='utf-8') as f:
        for item in tqdm(results):
            item.answer,item.contexts = rag(item.question,item.contexts, return_doc=True,context_array=True)
            print(item.model_dump())
            json.dump(item.model_dump(), f, ensure_ascii=False)
            f.write('\n')