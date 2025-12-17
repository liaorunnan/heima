import requests
import json

API_URL = "http://localhost:8003"

def test_sse():
    # 测试SSE流
    response = requests.post(f"{API_URL}/api/ragapi",
                             json={"query": "如何学习英语？", "history": []},
                             stream=True, timeout=30)
    
    print(f"Response status: {response.status_code}")
    print(f"Response headers: {response.headers}")
    
    answer = ""
    for line in response.iter_lines():
        # 调试：打印line的原始内容和长度
        print(f"Raw line: {line}, Length: {len(line)}, Type: {type(line)}")
        
        # 调试：打印解码后的内容
        if line:
            decoded_line = line.decode()
            print(f"Decoded line: '{decoded_line}', Starts with 'data: '? {decoded_line.startswith('data: ')}")
            print(f"Decoded line length: {len(decoded_line)}")
            
        if line and (s := line.decode()).startswith('data: '):
            print(f"Entered if block: {s}")
            chunk = json.loads(s[6:])
            print(f"Parsed chunk: {chunk}")
            
            token = chunk.get('token', '')
            if token:
                print(f"Got token: {token}")
                answer += token
            elif chunk.get('complete'):
                print(f"Complete flag found: {chunk}")
                break
    
    print(f"Final answer: {answer}")

if __name__ == "__main__":
    test_sse()
