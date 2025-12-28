import simple_pickle as sp

from b_rag.xiaokui.faq.indexing import VecIndex

from b_rag.xiaokui.tool.items import FAQItem,EnglishItem,BookItem


# path = "../../xiaokui/data/qas.pkl"
path = "../../xiaokui/data/qas.pkl"

qas = sp.read_pickle("../../xiaokui/data/qas.pkl")
items = [FAQItem(id="", query=qa["query"], answer=qa["answer"],source="英语词汇") for qa in qas]
# print(items)
VecIndex("faq").load(items)
