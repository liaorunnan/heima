from demo_es import Article
from processors.d_text_splitter import split_text
import os


def in_es(file_path):
    # 创建索引（如果不存在）
    Article.init()
    i = 0
    # 读取 markdown 文件

    for filename in os.listdir(file_path):
        # print(filename)
        if filename.endswith(".md"):
            path = os.path.join(file_path, filename)

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                # print(content)
                contents = split_text(content)

                for item in contents:
                    article = Article(
                        meta={'id': i},
                        child=item['child'],
                        parents=item['parent']

                    )
                    i += 1
                    article.save()
                    print(f"✅ 已存入 ES: {i}")


if __name__ == '__main__':
    in_es(".\output_markdown")
