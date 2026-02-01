from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.database import Base


class User(Base):
    """用户模型"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, comment="用户ID")
    username = Column(String(50), unique=True, nullable=False, index=True, comment="用户名")
    password_hash = Column(String(255), nullable=False, comment="密码哈希")
    role = Column(String(20), nullable=False, comment="角色(customer/merchant)")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")

    # 关系
    customer_chat_rooms = relationship("ChatRoom", foreign_keys="ChatRoom.customer_id", back_populates="customer")
    merchant_chat_rooms = relationship("ChatRoom", foreign_keys="ChatRoom.merchant_id", back_populates="merchant")
    sent_messages = relationship("ChatMessage", foreign_keys="ChatMessage.sender_id", back_populates="sender")
    products = relationship("Product", foreign_keys="Product.merchant_id", back_populates="merchant")
    tags = relationship("UserTag", back_populates="user", cascade="all, delete-orphan")
    profile = relationship("UserProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")