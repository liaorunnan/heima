from fastmcp import FastMCP

mcp = FastMCP("My MCP Server", host="0.0.0.0", port=3000)

@mcp.tool
def greet(name: str) -> str:
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run()