from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.database import Base


class StandardTag(Base):
    """标准标签库模型"""
    __tablename__ = "standard_tags"

    id = Column(Integer, primary_key=True, index=True, comment="标签ID")
    name = Column(String(100), nullable=False, unique=True, comment="标签名称")
    description = Column(Text, nullable=True, comment="标签描述")
    parent_id = Column(Integer, ForeignKey("standard_tags.id"), nullable=True, comment="父标签ID")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment="更新时间")

    # 关系
    parent = relationship("StandardTag", remote_side=[id], backref="children")
    user_tags = relationship("UserTag", back_populates="standard_tag")
    chat_room_tags = relationship("ChatRoomTag", back_populates="standard_tag")


class UserTag(Base):
    """用户标签模型"""
    __tablename__ = "user_tags"

    id = Column(Integer, primary_key=True, index=True, comment="标签ID")
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, comment="用户ID")
    tag_id = Column(Integer, ForeignKey("standard_tags.id"), nullable=False, comment="标准标签ID")
    intent_level = Column(Integer, nullable=False, comment="意图等级")
    score = Column(Float, nullable=False, comment="标签得分")
    create_type = Column(String(20), nullable=False, default="auto", comment="创建类型(auto/user)")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment="更新时间")

    # 关系
    user = relationship("User", back_populates="tags")
    standard_tag = relationship("StandardTag", back_populates="user_tags")


class UserProfile(Base):
    """用户画像模型"""
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True, comment="画像ID")
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False, comment="用户ID")
    profile_data = Column(Text, nullable=False, comment="画像数据(JSON格式)")
    tag_count = Column(Integer, nullable=False, default=0, comment="标签数量")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment="更新时间")

    # 关系
    user = relationship("User", back_populates="profile")


class ChatRoomTag(Base):
    """聊天室标签模型"""
    __tablename__ = "chat_room_tags"

    id = Column(Integer, primary_key=True, index=True, comment="标签ID")
    chat_room_id = Column(Integer, ForeignKey("chat_rooms.id"), nullable=False, comment="聊天室ID")
    tag_id = Column(Integer, ForeignKey("standard_tags.id"), nullable=False, comment="标准标签ID")
    intent_level = Column(Integer, nullable=False, comment="意图等级")
    score = Column(Float, nullable=False, comment="标签得分")
    create_type = Column(String(20), nullable=False, default="auto", comment="创建类型(auto/user)")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment="更新时间")

    # 关系
    chat_room = relationship("ChatRoom", back_populates="tags")
    standard_tag = relationship("StandardTag", back_populates="chat_room_tags")


class ChatRoomProfile(Base):
    """聊天室画像模型"""
    __tablename__ = "chat_room_profiles"

    id = Column(Integer, primary_key=True, index=True, comment="画像ID")
    chat_room_id = Column(Integer, ForeignKey("chat_rooms.id"), unique=True, nullable=False, comment="聊天室ID")
    profile_data = Column(Text, nullable=False, comment="画像数据(JSON格式)")
    tag_count = Column(Integer, nullable=False, default=0, comment="标签数量")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment="更新时间")

    # 关系
    chat_room = relationship("ChatRoom", back_populates="profile")
