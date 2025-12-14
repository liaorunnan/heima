from heima.lidekui.day02.match_keyword import Law
from heima.lidekui.day03.indexing import VecIndex, VecIndexLaw, VecIndexFaq
from heima.lidekui.day03.embedding import get_embedding
from heima.lidekui.day02.llm import chat
from heima.lidekui.day03.reranker import get_books_topk, get_laws_topk
from heima.lidekui.day04.cache import get_redis_answer, set_redis_answer, run_migrate

# Book.init()
# 已停止迭代
# def my_book():
#     history = []
#     while True:
#         query = input("请输入问题：")
#         books = Book.query(query)
#         vec = get_embedding(query)
#         hits = VecIndex("children").search(vec)
#         books.extend(hits)
#         books = get_books_topk(query, books)
#         length = len(books)
#         alist = list(range(0, length if length % 2 == 0 else length+1, 2)) + list(range(length-1 if length % 2 == 0 else length, 0, -2))
#         # print(books)
#
#         references = ''.join([books[i].source+'\n'+books[i].parent for i in alist])
#         user_prompt = f'请参考以下内容：{references}。回答问题：{query}'
#         answer = chat(user_prompt, history, system_prompt='你是一个小学一年级的语文老师，请根据我提供给你的参考内容，回答我的问题。如果在参考内容中没有找到答案，请回答"没有找到答案"。')
#         print(answer)
#         history.extend([
#             {"role": "user", "content": query},
#             {"role": "assistant", "content": answer}
#         ])

def my_law():
    history = []
    run_migrate()
    while True:
        query = input("请输入问题：")
        answer = get_redis_answer(query)
        if answer:
            print('答案来自缓存')
            print('='*50)
            print(answer)
        else:
            vec = get_embedding(query)
            faqs = VecIndexFaq("laws_faq").search(vec, topk=1)[0]
            if faqs.score > 0.76:
                print('答案来自FAQ')
                print('='*50)
                answer = faqs.answer
                print(answer)
            else:
                if_search = chat(query, history=history, system_prompt="""请判断用户是否需要搜索知识库，知识库中包含了一些法律法规的
                内容，如果需要搜索知识库，请返回true，否则返回false。不要回复其他内容。""")
                if if_search.lower() == 'true':
                    laws = Law.query(query)

                    hits = VecIndexLaw("laws").search(vec)
                    laws.extend(hits)
                    laws = get_laws_topk(query, laws)
                    length = len(laws)
                    alist = list(range(0, length if length % 2 == 0 else length + 1, 2)) + list(
                        range(length - 1 if length % 2 == 0 else length, 0, -2))
                    # print(laws)

                    references = '\n'.join([laws[i].embedding_text for i in alist])
                    user_prompt = f'请参考以下内容：{references}。回答问题：{query}'
                else:
                    user_prompt = query
                answer = chat(user_prompt, history=history, system_prompt="""
                你是一个法律专家，我有时会给你提供一些参考内容，如果我提供了，请根据我提供给你的法律法条等内容，回答我的问题，并列出法律法条依据。
                如果在参考内容中没有找到，请回答\"没有找到法律法规依据\"。如果我没有提供参考内容，或者在参考内容中没有找到答案，请联网搜索答案
                或者其他方式，以回答我的问题""")
                print(answer)
            set_redis_answer(query, answer)
        history.extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer}
        ])


if __name__ == '__main__':
    my_law()
