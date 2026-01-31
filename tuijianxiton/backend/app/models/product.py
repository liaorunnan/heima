from sqlalchemy import Column, Integer, String, Text, Numeric, ForeignKey, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.database import Base


class Product(Base):
    """商品模型"""
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True, comment="商品ID")
    name = Column(String(100), nullable=False, comment="商品名称")
    description = Column(Text, nullable=False, comment="商品描述")
    price = Column(Numeric(10, 2), nullable=False, comment="商品价格")
    merchant_id = Column(Integer, ForeignKey("users.id"), nullable=False, comment="商家ID")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")

    # 关系
    merchant = relationship("User", back_populates="products")
    chat_messages = relationship("ChatMessage", back_populates="product")