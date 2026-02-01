
import json
import os
import sys

# 将项目根目录添加到 sys.path 以便导入 conf.settings
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from tuijianxiton.llm_tagger import extract_user_tags

def run_tag_extraction():
    """
    运行标签提取主流程
    """
    # 1. 加载标签库
    tag_path = os.path.join(os.path.dirname(__file__), 'tag.json')
    with open(tag_path, 'r', encoding='utf-8') as f:
        tag_library = json.load(f)

    # 2. 加载聊天记录
    chat_path = os.path.join(os.path.dirname(__file__), 'chat.json')
    with open(chat_path, 'r', encoding='utf-8') as f:
        chat_data = json.load(f)

    # 3. 预处理聊天记录：转换为大模型易读的对话格式
    chat_lines = []
    for msg in chat_data:
        role_label = "用户" if msg['role'] == 'user' else "客服"
        chat_lines.append(f"{role_label}: {msg['content']}")
    
    chat_history_str = "\n".join(chat_lines)

    print("--- 正在提取标签 ---")
    
    # 4. 调用提取服务
    try:
        result_json_str = extract_user_tags(chat_history_str, tag_library)
        
        # 5. 解析并打印结果
        result = json.loads(result_json_str)
        print("\n--- 提取结果 ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 可选：保存结果到文件
        output_path = os.path.join(os.path.dirname(__file__), 'extracted_tags.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存至: {output_path}")

    except Exception as e:
        print(f"提取过程中发生错误: {e}")

if __name__ == "__main__":
    run_tag_extraction()
