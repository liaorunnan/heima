"""测试MCP服务器是否正常运行"""

import httpx
import asyncio

async def test_server():
    """测试MCP服务器是否正常运行"""
    url = "http://127.0.0.1:3000"
    
    print(f"Testing MCP server at {url}...")
    
    try:
        # 测试服务器是否可以连接
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{url}/health")
            print(f"Health check status: {response.status_code}")
            print(f"Health check response: {response.text}")
            
        print("\nServer is running!")
        print("To test tool calls, you can use:")
        print('python -c "from mcp.client.streamable_http import streamable_http_client; client = streamable_http_client(\"http://127.0.0.1:3000\"); result = client.tools.add_numbers(a=5, b=3); print(result)"' )
    except Exception as e:
        print(f"Error testing server: {e}")
        print("Make sure the MCP server is running first!")

if __name__ == "__main__":
    asyncio.run(test_server())
