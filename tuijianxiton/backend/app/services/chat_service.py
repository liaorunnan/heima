from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.models.chat import ChatRoom, ChatMessage
from app.models.user import User
from app.schemas.chat import ChatRoomCreate, ChatMessageCreate


def create_chat_room(db: Session, chat_room_create: ChatRoomCreate) -> ChatRoom:
    """创建聊天室"""
    # 检查聊天室是否已存在
    existing_room = db.query(ChatRoom).filter(
        (ChatRoom.customer_id == chat_room_create.customer_id) &
        (ChatRoom.merchant_id == chat_room_create.merchant_id)
    ).first()
    if existing_room:
        return existing_room

    # 验证用户角色
    customer = db.query(User).filter(User.id == chat_room_create.customer_id).first()
    merchant = db.query(User).filter(User.id == chat_room_create.merchant_id).first()
    
    if not customer or not merchant:
        raise ValueError("指定的用户不存在")
    
    if customer.role != "customer":
        raise ValueError("第一个用户必须是客户角色")
    
    if merchant.role != "merchant":
        raise ValueError("第二个用户必须是商家角色")

    # 创建新聊天室
    db_chat_room = ChatRoom(
        customer_id=chat_room_create.customer_id,
        merchant_id=chat_room_create.merchant_id
    )
    db.add(db_chat_room)
    db.commit()
    db.refresh(db_chat_room)
    return db_chat_room


def get_chat_room_by_id(db: Session, chat_room_id: int) -> Optional[ChatRoom]:
    """根据ID获取聊天室"""
    return db.query(ChatRoom).filter(ChatRoom.id == chat_room_id).first()


def get_chat_rooms_by_user_id(db: Session, user_id: int, role: str) -> List[ChatRoom]:
    """根据用户ID获取聊天室列表"""
    if role == "customer":
        chat_rooms = db.query(ChatRoom).filter(ChatRoom.customer_id == user_id).all()
    else:  # merchant
        chat_rooms = db.query(ChatRoom).filter(ChatRoom.merchant_id == user_id).all()

    # 为每个聊天室添加最后一条消息和时间
    for chat_room in chat_rooms:
        last_message = db.query(ChatMessage).filter(
            ChatMessage.chat_room_id == chat_room.id
        ).order_by(desc(ChatMessage.created_at)).first()
        if last_message:
            chat_room.last_message = last_message.content
            chat_room.last_message_time = last_message.created_at

    return chat_rooms


def create_chat_message(db: Session, chat_room_id: int, sender_id: int, message_create: ChatMessageCreate) -> ChatMessage:
    """创建聊天消息"""
    # 检查聊天室是否存在
    chat_room = get_chat_room_by_id(db, chat_room_id)
    if not chat_room:
        raise ValueError("聊天室不存在")

    # 检查发送者是否在聊天室中
    if sender_id != chat_room.customer_id and sender_id != chat_room.merchant_id:
        raise ValueError("发送者不在聊天室中")

    # 获取发送者角色
    sender = db.query(User).filter(User.id == sender_id).first()
    if not sender:
        raise ValueError("发送者不存在")

    # 创建新消息
    db_message = ChatMessage(
        chat_room_id=chat_room_id,
        sender_id=sender_id,
        content=message_create.content,
        message_type=message_create.message_type,
        product_id=message_create.product_id
    )
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    
    # 添加发送者角色信息
    db_message.sender_role = sender.role
    
    # 为商品消息添加价格信息
    if message_create.message_type == "product" and message_create.product_id:
        from app.models.product import Product
        product = db.query(Product).filter(Product.id == message_create.product_id).first()
        if product:
            db_message.product_price = product.price
    
    return db_message


def get_chat_messages_by_room_id(db: Session, chat_room_id: int) -> List[ChatMessage]:
    """根据聊天室ID获取聊天消息"""
    messages = db.query(ChatMessage).filter(
        ChatMessage.chat_room_id == chat_room_id
    ).order_by(ChatMessage.created_at).all()
    
    # 为每条消息添加发送者角色信息和商品价格信息
    for message in messages:
        # 获取发送者角色
        sender = db.query(User).filter(User.id == message.sender_id).first()
        if sender:
            message.sender_role = sender.role
        
        # 为商品消息添加价格信息
        if message.message_type == "product" and message.product_id:
            # 尝试获取商品信息
            from app.models.product import Product
            product = db.query(Product).filter(Product.id == message.product_id).first()
            if product:
                # 添加商品价格信息
                message.product_price = product.price
    
    return messages