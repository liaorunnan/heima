from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.database import Base


class ChatRoom(Base):
    """聊天室模型"""
    __tablename__ = "chat_rooms"

    id = Column(Integer, primary_key=True, index=True, comment="聊天室ID")
    customer_id = Column(Integer, ForeignKey("users.id"), nullable=False, comment="客户ID")
    merchant_id = Column(Integer, ForeignKey("users.id"), nullable=False, comment="商家ID")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")

    # 关系
    customer = relationship("User", foreign_keys=[customer_id], back_populates="customer_chat_rooms")
    merchant = relationship("User", foreign_keys=[merchant_id], back_populates="merchant_chat_rooms")
    messages = relationship("ChatMessage", back_populates="chat_room", cascade="all, delete-orphan")


class ChatMessage(Base):
    """聊天消息模型"""
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True, comment="消息ID")
    chat_room_id = Column(Integer, ForeignKey("chat_rooms.id"), nullable=False, comment="聊天室ID")
    sender_id = Column(Integer, ForeignKey("users.id"), nullable=False, comment="发送者ID")
    content = Column(Text, nullable=False, comment="消息内容")
    message_type = Column(String(20), nullable=False, comment="消息类型(text/product)")
    product_id = Column(Integer, ForeignKey("products.id"), nullable=True, comment="商品ID(消息类型为product时使用)")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="发送时间")

    # 关系
    chat_room = relationship("ChatRoom", back_populates="messages")
    sender = relationship("User", back_populates="sent_messages")
    product = relationship("Product", back_populates="chat_messages")