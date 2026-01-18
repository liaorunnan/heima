---
name: python-expert
description: Python 开发专家，提供性能优化、最佳实践和库选择建议
---

# Python 专家 Skill

## 何时使用

- Python 代码优化
- 选择合适的库和框架
- 性能调优
- 异步编程
- 类型注解和静态检查

## 核心建议

### 1. 性能优化
- 使用生成器代替列表（节省内存）
- 使用 `set` 进行快速查找
- 避免全局变量
- 使用 `__slots__` 减少内存占用

### 2. 代码风格
- 遵循 PEP 8 规范
- 使用类型注解（Type Hints）
- 编写文档字符串（Docstrings）
- 使用上下文管理器（Context Managers）

### 3. 常用库推荐
- 数据处理: `pandas`, `polars`
- 异步: `asyncio`, `aiohttp`
- 测试: `pytest`, `unittest`
- 类型检查: `mypy`, `pydantic`

## 示例代码

```python
# 推荐的函数定义方式
from typing import List, Optional

def process_data(
    items: List[str], 
    limit: Optional[int] = None
) -> List[str]:
    """
    处理数据列表
    
    Args:
        items: 输入数据列表
        limit: 可选的限制数量
    
    Returns:
        处理后的数据列表
    """
    return items[:limit] if limit else items
```
