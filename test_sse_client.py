"""简单的SSE客户端测试脚本"""

import asyncio
import httpx
import urllib.parse

async def test_sse_client():
    """测试SSE客户端连接和工具调用"""
    base_url = "http://127.0.0.1:3000"
    
    print("测试SSE客户端连接...")
    print(f"服务端地址: {base_url}")
    print()
    
    try:
        # 1. 连接到SSE端点获取session_id
        print("1. 连接到SSE端点...")
        sse_url = f"{base_url}/sse"
        session_id = None
        
        # 使用第一个客户端会话获取session_id
        async with httpx.AsyncClient(timeout=30.0) as client1:
            async with client1.stream("GET", sse_url) as response:
                response.raise_for_status()
                print(f"   SSE连接成功，状态码: {response.status_code}")
                
                # 解析SSE事件
                event_type = None
                
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        # 空行，事件结束
                        event_type = None
                        continue
                    elif line.startswith("event:"):
                        # 事件类型
                        event_type = line[6:].strip()
                        print(f"   收到SSE事件类型: {event_type}")
                    elif line.startswith("data:"):
                        # 事件数据
                        data = line[5:].strip()
                        print(f"   收到SSE数据: {data}")
                        
                        # 检查是否是endpoint事件，提取session_id
                        if event_type == "endpoint":
                            # 从数据中提取session_id
                            parsed = urllib.parse.urlparse(data)
                            query_params = urllib.parse.parse_qs(parsed.query)
                            if "session_id" in query_params:
                                session_id = query_params["session_id"][0]
                                print(f"   提取到session_id: {session_id}")
                                break
                    elif line.startswith(":"):
                        # 注释，忽略
                        continue
        
        if not session_id:
            print("   错误: 未能获取session_id")
            return
        
        print()
        print("2. 测试工具调用...")
        
        # 2. 使用session_id调用工具
        messages_url = f"{base_url}/messages/"
        tool_name = "add_numbers"
        tool_params = {"a": 5, "b": 3}
        
        # 构建请求体（JSON-RPC格式）
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": tool_name,
            "params": tool_params
        }
        
        # 使用第二个客户端会话调用工具
        async with httpx.AsyncClient(timeout=30.0) as client2:
            # 发送POST请求，session_id放在查询参数中
            request_url = f"{messages_url}?session_id={session_id}"
            print(f"   调用工具: {tool_name}")
            print(f"   参数: {tool_params}")
            print(f"   请求URL: {request_url}")
            print(f"   请求体: {payload}")
            
            response = await client2.post(
                request_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"   响应状态码: {response.status_code}")
            print(f"   响应内容: {response.text}")
            
            if response.status_code == 202:
                print(f"   工具调用请求已被接受 (202 Accepted)")
                print("   注意: 在SSE通信中，工具执行结果会通过SSE流返回")
                print("   由于SSE的特性，我们需要保持SSE连接打开来接收结果")
                print("   测试完成: 客户端已成功发送工具调用请求")
            elif response.status_code == 200:
                result = response.json()
                print(f"   工具调用成功，结果: {result}")
            else:
                print(f"   工具调用失败: {response.status_code}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("测试完成")

if __name__ == "__main__":
    print("简单的SSE客户端测试脚本")
    print("=" * 50)
    print("请确保MCP服务器已启动 (python mcp_server_example.py)")
    print()
    
    try:
        asyncio.run(test_sse_client())
    except KeyboardInterrupt:
        print("测试被用户中断")
    except Exception as e:
        print(f"测试错误: {e}")
