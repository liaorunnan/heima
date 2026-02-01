from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from sqlalchemy import and_
from sqlalchemy.orm import Session
import json

from app.models.chat import ChatMessage, ChatRoom
from app.models.user import User
from app.models.profile import UserTag, UserProfile, ChatRoomTag, ChatRoomProfile, StandardTag
from app.services.llm_service import analyze_user_intent


def get_user_chat_history(user_id: int, start_time: datetime, end_time: datetime, db: Session) -> List[Dict[str, Any]]:
    """获取用户在指定时间范围内的聊天记录
    
    Args:
        user_id: 用户ID
        start_time: 开始时间
        end_time: 结束时间
        db: 数据库会话
        
    Returns:
        聊天记录列表
    """
    # 查询用户参与的聊天室
    chat_rooms = db.query(ChatRoom).filter(
        (ChatRoom.customer_id == user_id) | (ChatRoom.merchant_id == user_id)
    ).all()
    
    chat_room_ids = [room.id for room in chat_rooms]
    
    # 查询这些聊天室中的消息
    messages = db.query(ChatMessage).filter(
        and_(
            ChatMessage.chat_room_id.in_(chat_room_ids),
            ChatMessage.created_at >= start_time,
            ChatMessage.created_at <= end_time
        )
    ).order_by(ChatMessage.created_at).all()
    
    # 格式化聊天记录
    chat_history = []
    for message in messages:
        chat_history.append({
            "id": message.id,
            "chat_room_id": message.chat_room_id,
            "sender_id": message.sender_id,
            "content": message.content,
            "message_type": message.message_type,
            "product_id": message.product_id,
            "created_at": message.created_at
        })
    
    return chat_history


def generate_chat_summary(chat_history: List[Dict[str, Any]]) -> str:
    """生成聊天记录摘要
    
    Args:
        chat_history: 聊天记录列表
        
    Returns:
        聊天记录摘要
    """
    if not chat_history:
        return ""
    
    # 合并聊天内容
    content = " ".join([msg["content"] for msg in chat_history if msg["content"]])
    
    # 限制摘要长度
    max_length = 1000
    if len(content) > max_length:
        content = content[:max_length] + "..."
    
    return content


def get_previous_day_range() -> tuple:
    """获取前一天的时间范围
    
    Returns:
        (start_time, end_time) 元组
    """
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    
    start_time = datetime(yesterday.year, yesterday.month, yesterday.day, 0, 0, 0)
    end_time = datetime(yesterday.year, yesterday.month, yesterday.day, 23, 59, 59)
    
    return start_time, end_time


def process_user_profile(user_id: int, db: Session) -> Dict[str, Any]:
    """处理用户画像，包括聊天记录打标和画像构建
    
    Args:
        user_id: 用户ID
        db: 数据库会话
        
    Returns:
        处理结果，包含标签和画像信息
    """
    # 获取前一天的时间范围
    start_time, end_time = get_previous_day_range()
    
    # 获取用户聊天记录
    chat_history = get_user_chat_history(user_id, start_time, end_time, db)
    
    if not chat_history:
        return {
            "user_id": user_id,
            "status": "no_chat_history",
            "message": "用户前一天没有聊天记录",
            "tags": [],
            "profile": {}
        }
    
    # 生成聊天记录摘要
    summary = generate_chat_summary(chat_history)
    
    if not summary:
        return {
            "user_id": user_id,
            "status": "no_summary",
            "message": "无法生成聊天记录摘要",
            "tags": [],
            "profile": {}
        }
    
    # 分析用户意图，提取标签
    # 这里使用简化的标签库，实际应用中应该从数据库或配置文件中加载
    standard_tags = [
        {"id": 1, "name": "产品咨询", "description": "用户咨询产品相关信息"},
        {"id": 2, "name": "价格咨询", "description": "用户咨询价格相关信息"},
        {"id": 3, "name": "售后服务", "description": "用户咨询售后服务相关信息"},
        {"id": 4, "name": "订单问题", "description": "用户咨询订单相关问题"},
        {"id": 5, "name": "物流咨询", "description": "用户咨询物流相关信息"}
    ]
    
    # 分析用户意图
    extracted_tags = analyze_user_intent([{"summary": summary}], standard_tags, user_id)
    
    # 构建用户画像
    profile = build_user_profile(user_id, extracted_tags, db)
    
    return {
        "user_id": user_id,
        "status": "success",
        "message": "用户画像处理成功",
        "tags": extracted_tags,
        "profile": profile
    }


def build_user_profile(user_id: int, tags: List[Dict[str, Any]], db: Session) -> Dict[str, Any]:
    """构建用户画像
    
    Args:
        user_id: 用户ID
        tags: 用户标签列表
        db: 数据库会话
        
    Returns:
        用户画像
    """
    # 获取用户基本信息
    user = db.query(User).filter(User.id == user_id).first()
    
    # 保存用户标签
    for tag in tags:
        # 检查标签是否已存在
        existing_tag = db.query(UserTag).filter(
            (UserTag.user_id == user_id) & (UserTag.tag_id == tag.get("id", 0))
        ).first()
        
        if existing_tag:
            # 更新现有标签
            existing_tag.intent_level = tag.get("intent_level", 0)
            existing_tag.score = tag.get("score", 0.0)
            existing_tag.updated_at = datetime.now()
        else:
            # 创建新标签
            new_tag = UserTag(
                user_id=user_id,
                tag_id=tag.get("id", 0),
                tag_name=tag.get("name", ""),
                intent_level=tag.get("intent_level", 0),
                score=tag.get("score", 0.0),
                description=tag.get("description", ""),
                create_type="auto"
            )
            db.add(new_tag)
    
    # 保存用户画像
    profile_data = {
        "user_id": user_id,
        "username": user.username if user else "",
        "role": user.role if user else "",
        "tags": tags,
        "tag_count": len(tags),
        "updated_at": datetime.now().isoformat()
    }
    
    # 检查画像是否已存在
    existing_profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
    
    if existing_profile:
        # 更新现有画像
        existing_profile.profile_data = json.dumps(profile_data, ensure_ascii=False)
        existing_profile.tag_count = len(tags)
        existing_profile.updated_at = datetime.now()
    else:
        # 创建新画像
        new_profile = UserProfile(
            user_id=user_id,
            profile_data=json.dumps(profile_data, ensure_ascii=False),
            tag_count=len(tags)
        )
        db.add(new_profile)
    
    # 提交事务
    db.commit()
    
    return profile_data


def get_user_tags(user_id: int, db: Session) -> List[Dict[str, Any]]:
    """获取用户标签
    
    Args:
        user_id: 用户ID
        db: 数据库会话
        
    Returns:
        用户标签列表
    """
    # 从数据库中查询用户标签
    tags = db.query(UserTag).filter(UserTag.user_id == user_id).all()
    
    # 格式化标签数据
    formatted_tags = []
    for tag in tags:
        formatted_tags.append({
            "id": tag.tag_id,
            "tag_name": tag.tag_name,
            "intent_level": tag.intent_level,
            "score": tag.score,
            "description": tag.description,
            "create_type": tag.create_type,
            "created_at": tag.created_at.isoformat() if tag.created_at else "",
            "updated_at": tag.updated_at.isoformat() if tag.updated_at else ""
        })
    
    return formatted_tags


def get_user_profile(user_id: int, db: Session) -> Dict[str, Any]:
    """获取用户画像
    
    Args:
        user_id: 用户ID
        db: 数据库会话
        
    Returns:
        用户画像信息
    """
    # 从数据库中查询用户画像
    profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
    
    if profile:
        # 解析画像数据
        try:
            profile_data = json.loads(profile.profile_data)
            return profile_data
        except Exception as e:
            print(f"解析用户画像失败: {str(e)}")
    
    # 如果画像不存在，返回默认值
    return {
        "user_id": user_id,
        "tags": [],
        "tag_count": 0,
        "updated_at": datetime.now().isoformat()
    }


def get_chat_room_chat_history(chat_room_id: int, start_time: datetime, end_time: datetime, db: Session) -> List[Dict[str, Any]]:
    """获取聊天室在指定时间范围内的聊天记录
    
    Args:
        chat_room_id: 聊天室ID
        start_time: 开始时间
        end_time: 结束时间
        db: 数据库会话
        
    Returns:
        聊天记录列表
    """
    # 查询聊天室中的消息
    messages = db.query(ChatMessage).filter(
        and_(
            ChatMessage.chat_room_id == chat_room_id,
            ChatMessage.created_at >= start_time,
            ChatMessage.created_at <= end_time
        )
    ).order_by(ChatMessage.created_at).all()
    
    # 格式化聊天记录
    chat_history = []
    for message in messages:
        chat_history.append({
            "id": message.id,
            "chat_room_id": message.chat_room_id,
            "sender_id": message.sender_id,
            "content": message.content,
            "message_type": message.message_type,
            "product_id": message.product_id,
            "created_at": message.created_at
        })
    
    return chat_history


def process_chat_room_profile(chat_room_id: int, db: Session) -> Dict[str, Any]:
    """处理聊天室画像，包括聊天记录打标和画像构建
    
    Args:
        chat_room_id: 聊天室ID
        db: 数据库会话
        
    Returns:
        处理结果，包含标签和画像信息
    """
    print(f"开始处理聊天室画像，聊天室ID: {chat_room_id}")
    
    try:
        print(f"步骤1: 获取聊天室信息")
        # 获取聊天室信息
        chat_room = db.query(ChatRoom).filter(ChatRoom.id == chat_room_id).first()
        if not chat_room:
            return {
                "chat_room_id": chat_room_id,
                "status": "chat_room_not_found",
                "message": "聊天室不存在",
                "tags": [],
                "profile": {}
            }
        
        customer_id = chat_room.customer_id
        print(f"获取到聊天室信息，客户ID: {customer_id}")
        
        print(f"步骤2: 获取前一天的时间范围")
        # 获取前一天的时间范围
        start_time, end_time = get_previous_day_range()
        print(f"时间范围: {start_time} 到 {end_time}")
        
        print(f"步骤3: 获取聊天室聊天记录")
        # 获取聊天室聊天记录
        chat_history = get_chat_room_chat_history(chat_room_id, start_time, end_time, db)
        print(f"获取到 {len(chat_history)} 条聊天记录")
        
        if not chat_history:
            return {
                "chat_room_id": chat_room_id,
                "status": "no_chat_history",
                "message": "聊天室前一天没有聊天记录",
                "tags": [],
                "profile": {}
            }
        
        print(f"步骤4: 过滤客户聊天记录")
        # 过滤只保留客户的聊天记录
        customer_chat_history = [msg for msg in chat_history if msg["sender_id"] == customer_id]
        print(f"过滤后客户聊天记录: {len(customer_chat_history)} 条")
        
        if not customer_chat_history:
            return {
                "chat_room_id": chat_room_id,
                "status": "no_customer_messages",
                "message": "聊天室前一天没有客户消息",
                "tags": [],
                "profile": {}
            }
        
        print(f"步骤5: 生成聊天记录摘要")
        # 生成聊天记录摘要
        summary = generate_chat_summary(customer_chat_history)
        print(f"生成的摘要: {summary}")
        
        if not summary:
            return {
                "chat_room_id": chat_room_id,
                "status": "no_summary",
                "message": "无法生成聊天记录摘要",
                "tags": [],
                "profile": {}
            }
        
        print(f"步骤4: 加载标准标签库")
        # 从数据库中加载标准标签库
        standard_tags_db = db.query(StandardTag).all()
        
        # 如果标准标签库为空，创建默认标签
        if not standard_tags_db:
            print("标准标签库为空，创建默认标签")
            default_tags = [
                {"name": "产品咨询", "description": "用户咨询产品相关信息"},
                {"name": "价格咨询", "description": "用户咨询价格相关信息"},
                {"name": "售后服务", "description": "用户咨询售后服务相关信息"},
                {"name": "订单问题", "description": "用户咨询订单相关问题"},
                {"name": "物流咨询", "description": "用户咨询物流相关信息"}
            ]
            
            for tag_data in default_tags:
                tag = StandardTag(
                    name=tag_data["name"],
                    description=tag_data["description"]
                )
                db.add(tag)
            
            db.commit()
            standard_tags_db = db.query(StandardTag).all()
        
        # 转换标签格式
        standard_tags = []
        for tag in standard_tags_db:
            standard_tags.append({
                "id": tag.id,
                "name": tag.name,
                "description": tag.description
            })
        
        print(f"加载了 {len(standard_tags)} 个标准标签")
        
        print(f"步骤5: 分析用户意图")
        # 分析用户意图
        extracted_tags = analyze_user_intent([{"summary": summary}], standard_tags, str(chat_room_id))
        print(f"提取到 {len(extracted_tags)} 个标签")
        
        print(f"步骤6: 构建聊天室画像")
        # 构建聊天室画像
        profile = build_chat_room_profile(chat_room_id, extracted_tags, db, customer_id)
        print(f"成功构建聊天室画像")
        
        print(f"步骤7: 返回结果")
        return {
            "chat_room_id": chat_room_id,
            "customer_id": customer_id,
            "status": "success",
            "message": "聊天室画像处理成功",
            "tags": extracted_tags,
            "profile": profile
        }
    except Exception as e:
        # 添加详细的错误日志
        print(f"处理聊天室画像失败: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise


def build_chat_room_profile(chat_room_id: int, tags: List[Dict[str, Any]], db: Session, customer_id: int) -> Dict[str, Any]:
    """构建聊天室画像
    
    Args:
        chat_room_id: 聊天室ID
        tags: 聊天室标签列表
        db: 数据库会话
        customer_id: 客户ID
        
    Returns:
        聊天室画像
    """
    # 保存聊天室标签
    for tag in tags:
        # 检查标签是否已存在
        existing_tag = db.query(ChatRoomTag).filter(
            (ChatRoomTag.chat_room_id == chat_room_id) & (ChatRoomTag.tag_id == tag.get("id", 0))
        ).first()
        
        if existing_tag:
            # 更新现有标签
            existing_tag.intent_level = tag.get("intent_level", 0)
            existing_tag.score = tag.get("score", 0.0)
            existing_tag.updated_at = datetime.now()
        else:
            # 创建新标签
            new_tag = ChatRoomTag(
                chat_room_id=chat_room_id,
                tag_id=tag.get("id", 0),
                intent_level=tag.get("intent_level", 0),
                score=tag.get("score", 0.0),
                create_type="auto"
            )
            db.add(new_tag)
    
    # 保存客户标签
    for tag in tags:
        # 检查客户标签是否已存在
        existing_user_tag = db.query(UserTag).filter(
            (UserTag.user_id == customer_id) & (UserTag.tag_id == tag.get("id", 0))
        ).first()
        
        if existing_user_tag:
            # 更新现有客户标签
            existing_user_tag.intent_level = tag.get("intent_level", 0)
            existing_user_tag.score = tag.get("score", 0.0)
            existing_user_tag.updated_at = datetime.now()
        else:
            # 创建新客户标签
            new_user_tag = UserTag(
                user_id=customer_id,
                tag_id=tag.get("id", 0),
                intent_level=tag.get("intent_level", 0),
                score=tag.get("score", 0.0),
                create_type="auto"
            )
            db.add(new_user_tag)
    
    # 保存聊天室画像
    profile_data = {
        "chat_room_id": chat_room_id,
        "customer_id": customer_id,
        "tags": tags,
        "tag_count": len(tags),
        "updated_at": datetime.now().isoformat()
    }
    
    # 检查画像是否已存在
    existing_profile = db.query(ChatRoomProfile).filter(ChatRoomProfile.chat_room_id == chat_room_id).first()
    
    if existing_profile:
        # 更新现有画像
        existing_profile.profile_data = json.dumps(profile_data, ensure_ascii=False)
        existing_profile.tag_count = len(tags)
        existing_profile.updated_at = datetime.now()
    else:
        # 创建新画像
        new_profile = ChatRoomProfile(
            chat_room_id=chat_room_id,
            profile_data=json.dumps(profile_data, ensure_ascii=False),
            tag_count=len(tags)
        )
        db.add(new_profile)
    
    # 提交事务
    db.commit()
    
    return profile_data


def get_chat_room_tags(chat_room_id: int, db: Session) -> List[Dict[str, Any]]:
    """获取聊天室标签
    
    Args:
        chat_room_id: 聊天室ID
        db: 数据库会话
        
    Returns:
        聊天室标签列表
    """
    # 从数据库中查询聊天室标签，包含标准标签的信息
    from sqlalchemy.orm import joinedload
    tags = db.query(ChatRoomTag).options(
        joinedload(ChatRoomTag.standard_tag)
    ).filter(ChatRoomTag.chat_room_id == chat_room_id).all()
    
    # 格式化标签数据
    formatted_tags = []
    for tag in tags:
        formatted_tags.append({
            "id": tag.tag_id,
            "tag_name": tag.standard_tag.name if tag.standard_tag else "",
            "intent_level": tag.intent_level,
            "score": tag.score,
            "description": tag.standard_tag.description if tag.standard_tag else "",
            "create_type": tag.create_type,
            "created_at": tag.created_at.isoformat() if tag.created_at else "",
            "updated_at": tag.updated_at.isoformat() if tag.updated_at else ""
        })
    
    return formatted_tags


def get_chat_room_profile(chat_room_id: int, db: Session) -> Dict[str, Any]:
    """获取聊天室画像
    
    Args:
        chat_room_id: 聊天室ID
        db: 数据库会话
        
    Returns:
        聊天室画像信息
    """
    # 从数据库中查询聊天室画像
    profile = db.query(ChatRoomProfile).filter(ChatRoomProfile.chat_room_id == chat_room_id).first()
    
    if profile:
        # 解析画像数据
        try:
            profile_data = json.loads(profile.profile_data)
            return profile_data
        except Exception as e:
            print(f"解析聊天室画像失败: {str(e)}")
    
    # 如果画像不存在，返回默认值
    return {
        "chat_room_id": chat_room_id,
        "tags": [],
        "tag_count": 0,
        "updated_at": datetime.now().isoformat()
    }
