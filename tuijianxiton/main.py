
import json
import os
import sys

# 将项目根目录添加到 sys.path 以便导入 conf.settings
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from tuijianxiton.llm_tagger import extract_user_tags
from tuijianxiton.user_persona import UserPersonaSystem

def run_tag_extraction():
    """
    运行标签提取与画像更新主流程
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
        # 在实际开发中，如果 extracted_tags.json 已经存在且最新，可以跳过 LLM 调用节省成本
        # 这里为了演示完整流程，每次都调用
        output_path = os.path.join(os.path.dirname(__file__), 'extracted_tags.json')
        
        # 优先使用本地缓存，避免 API 调用失败
        if os.path.exists(output_path):
            print(f"检测到本地已有标签文件: {output_path}，直接读取...")
            with open(output_path, 'r', encoding='utf-8') as f:
                extracted_tags = json.load(f)
        else:
            print("本地无缓存，调用 LLM 提取标签...")
            result_json_str = extract_user_tags(chat_history_str, tag_library)
            extracted_tags = json.loads(result_json_str)
            
            # 保存提取结果
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(extracted_tags, f, indent=2, ensure_ascii=False)
            print(f"\n结果已保存至: {output_path}")

        print("\n--- 提取结果 ---")
        print(json.dumps(extracted_tags, indent=2, ensure_ascii=False))

        # ==========================================
        # 6. 画像系统更新 (新增逻辑)
        # ==========================================
        print("\n--- 正在更新用户画像 ---")
        persona_system = UserPersonaSystem()
        
        # 模拟加载历史画像 (可选)
        # persona_system.long_term_profile = load_from_db(...)
        
        # 传入提取的标签进行更新
        # 注意：extracted_tags 是一个列表，格式如 [{"name": "...", "tier": "..."}]
        # UserPersonaSystem.update_persona 已经适配了这个格式
        if isinstance(extracted_tags, dict) and "tags" in extracted_tags:
             # 有些 LLM 输出可能会包一层 {"tags": [...]}
             tags_list = extracted_tags["tags"]
        elif isinstance(extracted_tags, list):
             tags_list = extracted_tags
        else:
             tags_list = []
             print("Warning: 无法识别的标签格式")

        persona_system.update_persona(tags_list)
        
        # 获取最终画像
        final_persona = persona_system.get_final_persona()
        print("\n--- 最终用户画像 (Short-term & Long-term) ---")
        print(json.dumps(final_persona, indent=2, ensure_ascii=False))
        
        # 保存画像结果
        persona_path = os.path.join(os.path.dirname(__file__), 'user_persona.json')
        with open(persona_path, 'w', encoding='utf-8') as f:
            json.dump(final_persona, f, indent=2, ensure_ascii=False)
        print(f"\n画像已保存至: {persona_path}")

    except Exception as e:
        print(f"流程执行过程中发生错误: {e}")

if __name__ == "__main__":
    run_tag_extraction()
