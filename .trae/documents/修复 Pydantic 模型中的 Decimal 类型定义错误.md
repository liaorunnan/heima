## 问题分析

在 `product.py` 文件中，`price` 字段的定义存在错误：

```python
price: Decimal = Field(..., gt=0, decimal_places=2, description="商品价格")
```

错误原因：`decimal_places` 不是 Pydantic 中 `Field` 的有效约束参数。在 Pydantic 中，对于 `Decimal` 类型，应该使用类型注解来指定精度，而不是作为 `Field` 的参数。

## 修复方案

修改 `product.py` 文件，移除 `decimal_places=2` 约束，并正确定义 `Decimal` 类型。

### 修改步骤

1. **修改** **`ProductBase`** **类**：

   * 移除 `price` 字段的 `decimal_places=2` 参数

   * 保持 `gt=0` 约束确保价格为正数

2. **修改** **`ProductUpdate`** **类**：

   * 同样移除 `price` 字段的 `decimal_places=2` 参数

   * 保持 `gt=0` 约束

### 修复后代码

```python
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from decimal import Decimal


class ProductBase(BaseModel):
    """商品基础模型"""
    name: str = Field(..., min_length=1, max_length=100, description="商品名称")
    description: str = Field(..., description="商品描述")
    price: Decimal = Field(..., gt=0, description="商品价格")


class ProductCreate(ProductBase):
    """商品创建模型"""
    pass


class ProductUpdate(BaseModel):
    """商品更新模型"""
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="商品名称")
    description: Optional[str] = Field(None, description="商品描述")
    price: Optional[Decimal] = Field(None, gt=0, description="商品价格")


class Product(ProductBase):
    """商品响应模型"""
    id: int
    merchant_id: int
    created_at: datetime

    class Config:
        from_attributes = True
```

## 验证

修复后，系统应该能够正常启动，不再出现 `ValueError: Unknown constraint decimal_places` 错误。

运行前后端功能，确保可以运行

## 注意事项

* Pydantic 中 `Decimal` 类型的精度控制通常在数据库层面或业务逻辑中处理

* 保持 `gt=0` 约束确保价格为正数，符合业务需求

* 修复后需要重新启动后端服务以应用更改

