import json
import os
import base64
from tqdm import tqdm
import simple_pickle as sp

from b_rag.day02.processors.b_pdf2img import pdf_to_images
from openai import OpenAI

from conf import settings

# client = OpenAI(
#     api_key=settings.dashscope_api_key,
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )


if __name__ == '__main__':
    pdf_to_images("data/文书示范文本.pdf", "data/legalinstructionimages")
