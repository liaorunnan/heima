from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base
from app.models.profile import UserTag, StandardTag
from app.models.chat import ChatRoom
from app.models.user import User

# 数据库连接配置
DATABASE_URL = "postgresql://myuser:123456@127.0.0.1/mydb"

# 创建数据库引擎
engine = create_engine(DATABASE_URL)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建数据库会话
db = SessionLocal()

try:
    print("=== 检查客户标签保存情况 ===")
    
    # 1. 查询聊天室信息，获取客户ID
    chat_room = db.query(ChatRoom).filter(ChatRoom.id == 2).first()
    if chat_room:
        customer_id = chat_room.customer_id
        print(f"聊天室ID: 2, 客户ID: {customer_id}")
        
        # 2. 查询客户信息
        customer = db.query(User).filter(User.id == customer_id).first()
        if customer:
            print(f"客户信息: ID={customer.id}, 用户名={customer.username}, 角色={customer.role}")
        else:
            print("未找到客户信息")
        
        # 3. 查询客户标签
        user_tags = db.query(UserTag).filter(UserTag.user_id == customer_id).all()
        print(f"\n客户标签数量: {len(user_tags)}")
        
        if user_tags:
            print("\n客户标签列表:")
            for tag in user_tags:
                # 获取标准标签信息
                standard_tag = db.query(StandardTag).filter(StandardTag.id == tag.tag_id).first()
                tag_name = standard_tag.name if standard_tag else "未知标签"
                tag_description = standard_tag.description if standard_tag else ""
                
                print(f"- 标签ID: {tag.tag_id}, 标签名称: {tag_name}")
                print(f"  意图等级: {tag.intent_level}, 得分: {tag.score}")
                print(f"  创建类型: {tag.create_type}")
                print(f"  创建时间: {tag.created_at}")
                print(f"  更新时间: {tag.updated_at}")
                print(f"  标签描述: {tag_description}")
                print()
        else:
            print("客户无标签")
    else:
        print("未找到聊天室ID=2的信息")
        
    # 4. 查询所有用户标签，确保数据完整性
    all_user_tags = db.query(UserTag).all()
    print(f"\n=== 数据库中所有用户标签数量: {len(all_user_tags)} ===")
    
    if all_user_tags:
        print("\n所有用户标签概览:")
        for tag in all_user_tags:
            user = db.query(User).filter(User.id == tag.user_id).first()
            user_name = user.username if user else "未知用户"
            standard_tag = db.query(StandardTag).filter(StandardTag.id == tag.tag_id).first()
            tag_name = standard_tag.name if standard_tag else "未知标签"
            
            print(f"用户ID: {tag.user_id}, 用户名: {user_name}, 标签ID: {tag.tag_id}, 标签名称: {tag_name}")
    else:
        print("数据库中无用户标签")
        
finally:
    # 关闭数据库会话
    db.close()
    print("\n数据库查询完成")
