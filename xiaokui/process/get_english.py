import fitz  # PyMuPDF
import re

def remove_page_number(text): # 移除页码方法
    """
    移除类似 "第 XX 页 共 113 页" 的页码标记
    """
    # 正则表达式匹配模式
    pattern = r'第\s*\d+\s*页\s*共\s*113\s*页'
    # 替换为空字符串
    return re.sub(pattern, '', text).strip()
# 我的pdf文件路径是pdf_path = "../data/大学英语四级词汇完整版带音标-顺序版.pdf"
# 如何读取pdf内容，将序号seq_id,单词word,音标pronunciation，词性和意思part_of_speech and meaning写入csv文件

def read_pdf_pymupdf(pdf_path):
    """使用PyMuPDF读取PDF文本"""
    text = ""
    with fitz.open(pdf_path) as doc:
        print(f"PDF总页数: {doc.page_count}")

        # 逐页提取文本
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()

    return text


def format_text_to_csv(pdf_text):
    """将PDF文本格式化为CSV格式"""
    lines = pdf_text.strip().split('\n')

    result = []
    current_entry = []

    for line in lines:
        line = line.strip()

        # 跳过标题行和空行
        if line in ["序号单词", "注音", "释义"] or not line:
            continue

        # 检查是否是新条目开始（以数字开头）
        # 使用正则表达式匹配以数字开头的行，数字后面可能有空格、点号或单词
        if re.match(r'^\d+', line):
            # 处理之前的条目
            if current_entry:
                process_entry(current_entry, result)

            # 开始新的条目
            current_entry = [line]
        else:
            # 继续当前条目
            current_entry.append(line)

    # 处理最后一个条目
    if current_entry:
        process_entry(current_entry, result)

    return result


def process_entry(entry_lines, result_list):
    """处理一个单词条目"""
    if not entry_lines:
        return

    # 第一行可能包含序号，或者序号+单词
    first_line = entry_lines[0]

    # 使用正则表达式提取序号
    match = re.match(r'^(\d+)', first_line)
    if not match:
        return

    seq = match.group(1)

    # 移除第一行中的序号部分
    remaining = first_line[len(seq):].strip()

    # 判断格式类型
    if len(entry_lines) >= 3:
        # 格式1：序号单独一行，后面是单词、音标、释义
        if not remaining:  # 第一行只有序号
            if len(entry_lines) >= 4:
                word = entry_lines[1] if len(entry_lines) > 1 else ""
                phonetic = entry_lines[2] if len(entry_lines) > 2 else ""
                meaning = " ".join(entry_lines[3:]) if len(entry_lines) > 3 else ""
            else:
                # 不完整，跳过
                return
        else:
            # 格式2：序号和单词在同一行
            word = remaining
            if len(entry_lines) >= 3:
                phonetic = entry_lines[1] if len(entry_lines) > 1 else ""
                meaning = " ".join(entry_lines[2:]) if len(entry_lines) > 2 else ""
            else:
                # 不完整，跳过
                return
    else:
        # 条目不完整，跳过
        return

    # 清理数据
    seq = seq.strip()
    word = word.strip()
    phonetic = phonetic.strip()
    meaning = meaning.strip()
    meaning = remove_page_number(meaning)
    print(meaning)

    # 创建格式化条目
    if word and phonetic and meaning:
        formatted = f'{seq},"{word}","{phonetic}","{meaning}"'
        result_list.append(formatted)
    elif word and meaning:  # 如果没有音标，只有释义
        formatted = f'{seq},"{word}","","{meaning}"'
        result_list.append(formatted)


def clean_csv_data(csv_data):
    """清理CSV数据，确保引号和逗号正确处理"""
    cleaned_data = []
    for line in csv_data:
        # 处理可能存在的多余引号
        line = line.replace('""', '"')
        cleaned_data.append(line)
    return cleaned_data


if __name__ == '__main__':
    pdf_path = "../data/大学英语四级词汇完整版带音标-顺序版.pdf"

    # 读取PDF
    pdf_text = read_pdf_pymupdf(pdf_path)

    # 格式化文本
    formatted_data = format_text_to_csv(pdf_text)

    # 清理数据
    cleaned_data = clean_csv_data(formatted_data)

    # 输出示例（查看第90-110行）
    print("第90-110行示例:")
    for i in range(89, 110):  # 索引从0开始，所以89对应第90行
        if i < len(cleaned_data):
            print(cleaned_data[i])

    print(f"\n总共处理了 {len(cleaned_data)} 个单词")

    # 保存到文件
    with open("../data/words_output2.csv", "w", encoding="utf-8") as f:
        for line in cleaned_data:
            f.write(line + "\n")

    print("结果已保存到 words_output2.csv")