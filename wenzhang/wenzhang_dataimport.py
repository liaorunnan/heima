import openai
from conf import settings
import json

from datasets import load_dataset

# data_files 参数可以指定本地路径
# split="train" 表示加载后直接返回 dataset 对象，而不是字典
dataset = load_dataset("parquet", data_files={'train': 'train-00000-of-00001.parquet'}, split='train')

# 查看数据集结构
print(dataset)

# 查看第一条数据
# print(dataset[0])

# 将数据放到csv中
dataset.to_csv('train.csv', index=False,columns=['dialogue','summary'])


wenzhang = """
随着生成式人工智能（Generative AI）技术的飞速发展，2024年被视为“AI应用元年”。与过去仅仅能够进行简单对话的聊天机器人不同，新一代的 AI 代理（AI Agents）开始展现出自主规划和执行任务的能力。它们不仅能写代码、生成图片，还能协助人类预订机票、管理日程甚至自动处理复杂的办公流程。
然而，技术的进步也伴随着挑战。数据隐私安全、版权纠纷以及深度伪造（Deepfake）内容的泛滥，成为了监管机构和科技公司必须面对的难题。专家认为，未来 AI 的发展重心将从单纯追求模型参数的大小，转向模型的可解释性、安全性以及在垂直领域的深度应用。人类与 AI 的协作模式，也将从“人发出指令”转变为“人设定目标，AI 协助达成”。
"""


client = openai.OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url
)

system_prompt = """
你是一名专业的文章摘要员，负责为用户提供文章的摘要。请根据用户提供的文章，生成一篇简洁明了的文章摘要。摘要应包含文章的主要观点和关键信息。输出使用json格式,包括以下内容:
{
    "summary": 文章的标题,
    "main_points": ["主要观点1", "主要观点2", ...]
}
"""

wengzhang1 = """
在信息碎片化的时代，保持专注变得越来越难，而“番茄工作法”成为了许多人提升效率的秘密武器。这种时间管理方法的核心理念非常简单：将工作时间切割成一个个25分钟的片段，每个片段称为一个“番茄钟”。在这一段时间内，你需要全神贯注地工作，不做任何与任务无关的事情，然后休息5分钟。
这种“冲刺-休息”的循环模式之所以有效，是因为它符合大脑的认知规律。长时间的连续工作容易导致认知疲劳，而短暂且规律的休息能让大脑“回血”，从而维持长期的敏锐度。此外，番茄工作法还能有效对抗拖延症，因为25分钟的短目标比“完成整个项目”的宏大目标更容易让人产生行动的动力。"""

wengzhang2 = """
丝绸之路不仅仅是一条连接东西方的贸易通道，它更是一张巨大的文明交流网络。起于中国古代的长安（今西安），经中亚、西亚，最终抵达欧洲的罗马，这条路线跨越了数千公里。在数千年的历史长河中，骆驼商队驮着的不仅是中国的丝绸、瓷器和西方的玻璃、香料，更有无形的文化财富。
通过丝绸之路，中国的造纸术、火药和印刷术传入了西方，极大地推动了欧洲文明的进程；与此同时，西方的佛教艺术、天文历法以及葡萄、胡萝卜等农作物也传入了中国，丰富了东方人的生活。可以说，丝绸之路是人类历史上第一次全球化浪潮的缩影，它证明了不同文明之间的交流与互鉴是推动历史进步的重要动力。
"""

zaiyao1 = {
    "summary": '番茄工作法',
    "main_points": ["番茄工作法通过将时间切割为25分钟工作和5分钟休息来提升效率。", "规律的休息能缓解大脑认知疲劳，保持专注力。", "短时段的目标设定有助于克服拖延症"]
}

zaiyao2 = {
    "summary": '丝绸之路',
    "main_points": ["丝绸之路是连接东西方的重要贸易通道和文明交流网络。", "促进了商品（丝绸、香料）以及技术（造纸术）和文化（宗教）的双向传播。", "它是古代全球化的缩影，推动了不同文明的进步与互鉴。"]
}

history = [
    {"role": "user", "content": wengzhang1},
    {"role": "assistant", "content": json.dumps(zaiyao1)},
    {"role": "user", "content": wengzhang2},
    {"role": "assistant", "content": json.dumps(zaiyao2)},
]

user_prompt = f"""
请为以下文章生成摘要：
{wenzhang}
"""

message = [
    {"role": "system", "content": system_prompt},
    *history,
    {"role": "user", "content": user_prompt},
]


response = client.chat.completions.create(
        model=settings.MODEL_NAME,
        messages=message,
        temperature=0.0,
    )
print(response.choices[0].message.content)


def summarize_text(article: str):
    

    return article

