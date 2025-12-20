
import json
from rag.llm import chat
from get_data.tishici import *


# 添加错误处理和调试信息
try:
    response = chat('请按照提示词的要求生成英语app问答对', [], SYSTEM_PROMPT)
    
    # 调试：打印响应内容的前200个字符
    print("LLM Response (first 200 chars):", response[:200])
    print("..." if len(response) > 200 else "")
    
    # 清理响应内容，移除Markdown代码块标记
    cleaned_response = response.strip()
    if cleaned_response.startswith("```json"):
        # 移除开头的```json标记
        cleaned_response = cleaned_response[7:].strip()
    if cleaned_response.startswith("```"):
        # 移除开头的```标记
        cleaned_response = cleaned_response[3:].strip()
    if cleaned_response.endswith("```"):
        # 移除结尾的```标记
        cleaned_response = cleaned_response[:-3].strip()
    
    # 尝试解析JSON
    data = json.loads(cleaned_response)
    
    # 写入文件
    with open('wendadui.json', 'a', encoding='utf-8') as file:
        file.write(json.dumps(data, ensure_ascii=False, indent=4))
        file.write('\n')  # 添加换行符
        
    print("数据已成功写入 wendadui.json")
    
except json.JSONDecodeError as e:
    print(f"JSON解析错误: {e}")
    print("LLM返回的内容不是有效的JSON格式")
    print("Raw response:", response[:500] if response else "Empty response")
    
    # 尝试修复常见的JSON格式问题
    if response:
        # 尝试清理响应内容
        cleaned_response = response.strip()
        # 移除Markdown代码块标记
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:].strip()
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:].strip()
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3].strip()
            
        if not cleaned_response.startswith('{') and not cleaned_response.startswith('['):
            # 如果响应不以JSON开始符号开头，尝试找到JSON部分
            start_idx = cleaned_response.find('{')
            if start_idx != -1:
                cleaned_response = cleaned_response[start_idx:]
                
        # 尝试再次解析
        try:
            data = json.loads(cleaned_response)
            with open('wendadui.json', 'a', encoding='utf-8') as file:
                file.write(json.dumps(data, ensure_ascii=False, indent=4))
                file.write('\n')
            print("修复后数据已成功写入 wendadui.json")
        except json.JSONDecodeError:
            print("修复后仍然无法解析JSON")
            
except Exception as e:
    print(f"发生其他错误: {e}")
