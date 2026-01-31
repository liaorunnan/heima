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