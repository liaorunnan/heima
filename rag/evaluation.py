import json

import pandas as pd
import simple_pickle as sp
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from conf import settings
from rag.gen_data import Item


def evaluate_ragas(data_path, metrics,context_string=True):
    data = sp.read_data(data_path)
    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in data[:50]:
        item = json.loads(item)
        for k in eval_data:
            if k == "contexts" and context_string:
                eval_data[k].append([item[k]])
            else:
                eval_data[k].append(item[k])

    dataset = Dataset.from_dict(eval_data)
    print(f'dataset size: {len(dataset)}')
    print(f'dataset[0]-->{dataset[0]}')

    llm = ChatOpenAI(base_url=settings.qw_api_url,
                     model=settings.qw_model,
                     api_key=settings.qw_api_key,
                     temperature=0,
                     n=3)
    embeddings = DashScopeEmbeddings(dashscope_api_key=settings.qw_api_key)

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings
    )

    print(f"RAGAS评估结果：\n{result}")
    result_df = pd.DataFrame([result])
    result_df.to_csv("ragas_evaluation_results.csv", index=False)


# RAGAS评估结果：
# {'faithfulness': 0.9218, 'answer_relevancy': 0.7608, 'context_precision': 1.0000, 'context_recall': 1.0000}

# 上下文相关性：上下文是否仅包含相关信息   # 上下文召回率：上下文是否包含所有必要信息
evaluate_ragas("./data_search_eval.jsonl", [context_precision, context_recall])
# 忠实度：答案是否基于上下文     # 答案相关性：答案与问题的匹配度
# evaluate_ragas("./data_gen_eval.jsonl", [faithfulness, answer_relevancy])
