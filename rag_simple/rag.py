from data_process.insert_to_es import ESLawChunk, ESWritChunk
from rag_simple.llm import chat


def choose_mode():
    while True:
        mode = input("请选择模式：1) 法律相关问答  2) 起诉状/答辩状文书  (输入q退出)：")
        if mode in ("1", "2", "q", "quit"):
            return mode
        print("请输入 1 / 2 / q")


def build_reference_law(docs):
    # 使用可用的文本字段构造参考，优先 content_text，其次 embedding_text
    parts = []
    for d in docs:
        text = getattr(d, "content_text", None) or getattr(d, "embedding_text", "")
        if text:
            parts.append(text)
    return ['\n' + part for part in [parts[i] for i in (list(range(0,len(parts),2)) + list(range(len(parts)-1,0,-2)))]]


def build_reference_writ(docs):
    parts = []
    for d in docs:
        parts.append(f"{d.indexbytitle}\n{d.context}")
    return "\n\n".join(parts)


def main():
    history = []
    mode = choose_mode()
    if mode in ("q", "quit"):
        return

    while True:
        query = input("请输入问题（输入q／quit退出，输入switch切换模式）: ")
        if query in ("q", "quit"):
            break
        if query == "switch":
            mode = choose_mode()
            if mode in ("q", "quit"):
                break
            continue

        if mode == "1":
            docs = ESLawChunk.query(query)
            reference = build_reference_law(docs)
        else:
            docs = ESWritChunk.query(query)
            reference = build_reference_writ(docs)

        prompt = f"请根据以下参考内容回答问题:\n{reference}\n问题: {query}\n答案: "
        print("召回资料: ", reference)
        answer = chat(prompt, history)
        print("回答: ", answer)
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
