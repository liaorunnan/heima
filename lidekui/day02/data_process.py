from heima.lidekui.day02.c_ocr import Img2text
from heima.lidekui.day02.llm import chat
import os
from tqdm import tqdm
import simple_pickle as sp
import joblib


def img_ocr():
    img_paths = os.listdir("./images")
    img2text = Img2text()
    img2text.load_ocr()
    for img in tqdm(img_paths):
        img_path = os.path.join("./images", img)
        img2text.ocr(img_path)


def strip_text(text):
    text = text.replace("\n", "")
    text = text.replace(" ", "")
    text = text.replace("\\", "")
    text = text.replace("|", "")
    text = text.replace("-", "")
    text = text.replace("_", "")
    return text


def md2text():
    md_paths = [i for i in os.listdir("./output") if i.endswith(".md")]
    md0 = os.path.join("./output", md_paths[0])
    with open(md0, "r", encoding="utf-8") as f:
        text = f.read()
        text = strip_text(text)
    history = [
        {"role": "user", "content": text},
        {"role": "assistant", "content": """
            教育部审定
            二零一六年
            全国优秀教材特等奖
            义务教育教科书
            一年级上册
            语文
        """},
    ]
    md_contents = []
    for md in tqdm(md_paths):
        md_path = os.path.join("./output", md)
        with open(md_path, "r", encoding="utf-8") as f:
            text = f.read()
        text = strip_text(text)
        content = chat(text, history)
        history.extend([
            {"role": "user", "content": text},
            {"role": "assistant", "content": content}
        ])
        md_contents.append(content)
    sp.write_data(str(md_contents), "./data/md_contents.txt")
    joblib.dump(md_contents, "./data/md_contents.pkl")
    return md_contents


def read_md():
    md_contents = joblib.load("./data/md_contents.pkl")
    for md_content in md_contents:
        print(md_content)
    print(len(md_contents))


if __name__ == '__main__':
    read_md()
