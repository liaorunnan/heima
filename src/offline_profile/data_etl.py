import datetime
from typing import Dict, List, Any, Optional

from src.tools.porstgreDB_tools import get_user_chat_records_db
from src.utils.llm_utils import get_default_llm
from langchain.agents import create_agent

from src.workflow.offline_profile.config1 import DATA_PROCESSING_CONFIG, logger
from src.workflow.offline_profile.utils import remove_pii, clean_text, calculate_time_diff, format_datetime


class Session:
    """会话类"""
    
    def __init__(self, session_id: str, start_time: datetime.datetime):
        self.session_id = session_id
        self.start_time = start_time
        self.end_time = start_time
        self.messages = []
    
    def add_message(self, message: Dict[str, Any]):
        """添加消息到会话"""
        self.messages.append(message)
        self.end_time = message['created_at']
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'session_id': self.session_id,
            'start_time': format_datetime(self.start_time),
            'end_time': format_datetime(self.end_time),
            'message_count': len(self.messages),
            'messages': self.messages
        }


def split_into_sessions(messages: List[Dict[str, Any]]) -> List[Session]:
    """将会话记录切分为会话块
    
    Args:
        messages: 消息列表
        
    Returns:
        会话列表
    """
    if not messages:
        return []
    
    sessions = []
    current_session = None
    timeout_minutes = DATA_PROCESSING_CONFIG['session_timeout_minutes']
    
    for message in sorted(messages, key=lambda x: x['created_at']):
        if not current_session:
            # 创建新会话
            session_id = f"session_{len(sessions)}"
            current_session = Session(session_id, message['created_at'])
            current_session.add_message(message)
        else:
            # 计算时间差
            time_diff = calculate_time_diff(current_session.end_time, message['created_at'])
            if time_diff > timeout_minutes:
                # 超时，创建新会话
                sessions.append(current_session)
                session_id = f"session_{len(sessions)}"
                current_session = Session(session_id, message['created_at'])
            current_session.add_message(message)
    
    # 添加最后一个会话
    if current_session:
        sessions.append(current_session)
    
    return sessions


def process_chat_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """处理单条聊天消息
    
    Args:
        message: 原始消息
        
    Returns:
        处理后的消息
    """
    # 清洗文本
    content = clean_text(message.get('content', ''))
    # 去除敏感信息
    content = remove_pii(content)
    
    return {
        'content': content,
        'created_at': message.get('created_at'),
        'chat_role': message.get('chat_role', 'user')
    }


def get_chat_history(user_id: str, conversation_id: int, start_time: datetime.datetime, end_time: datetime.datetime) -> List[Dict[str, Any]]:
    """获取用户聊天历史
    
    Args:
        user_id: 用户 ID
        conversation_id: 会话 ID
        start_time: 开始时间
        end_time: 结束时间
        
    Returns:
        聊天记录列表
    """
    try:
        messages = get_user_chat_records_db(
            exa_conversation_id=conversation_id,
            exa_customer_id=int(user_id),
            time_after=start_time
        )

        logger.info(f"聊天记录{messages}")
        
        # 处理消息
        processed_messages = []
        for msg in messages:
            
            processed_msg = process_chat_message(msg)
            processed_messages.append(processed_msg)
        
        return processed_messages
    except Exception as e:
        logger.error(f"获取聊天历史失败: {e}")
        return []


class ChatSummarizer:
    """聊天记录摘要器"""
    
    def __init__(self):
        self.llm = get_default_llm()
        self.system_prompt = """
            # Role
            你是一个数据清洗专家。你的任务是精简聊天记录，为后续的“人物画像分析”做准备。

            # Rules
            1. **保留原话**：凡是涉及“价格评价”、“个人情况（家里有孩子/要出差）”、“具体参数询问”、“强烈情绪”的句子，必须**逐字保留**。
            2. **删除废话**：删除所有的寒暄（你好）、无意义的确认（嗯嗯）、重复的啰嗦。
            3. **合并碎片**：将用户连续的短句合并为一段完整的表述。
            4. **客服精简**：客服的回复如果只是客套，直接删掉；如果包含产品关键信息（如价格/型号），精简为 [客服: 推荐了X型号，报价500元]。

            # Output Format
            用户: <保留的原话>
            客服: <精简后的关键信息>
            """
        
        self.agent = create_agent(
            model=self.llm,
            name="chat_summarizer",
            system_prompt=self.system_prompt,
        )
    
    def summarize_session(self, session: Session) -> str:
        """对会话进行摘要
        
        Args:
            session: 会话对象
            
        Returns:
            会话摘要
        """
        # 构建对话历史
        dialogue_history = []
        for msg in session.messages:
            role = "用户" if msg['chat_role'] == 'user' else "客服"
            dialogue_history.append(f"{role}: {msg['content']}")
        
        dialogue_text = "\n".join(dialogue_history)
        prompt = f"请对以下对话进行摘要：\n{dialogue_text}"
        
        try:
            response = self.agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
            )['messages'][-1].content
            
            # 提取摘要内容
            if "用户需求摘要：" in response:
                summary = response.split("用户需求摘要：")[1].strip()
            else:
                summary = response.strip()
            
            return summary
        except Exception as e:
            logger.error(f"会话摘要失败: {e}")
            return ""
    
    def summarize_chat_history(self, sessions: List[Session]) -> List[Dict[str, Any]]:
        """对聊天历史进行摘要
        
        Args:
            sessions: 会话列表
            
        Returns:
            摘要列表
        """
        summaries = []
        for session in sessions:
            if len(session.messages) > 0:
                summary = self.summarize_session(session)
                if summary:
                    summaries.append({
                        'session_id': session.session_id,
                        'start_time': format_datetime(session.start_time),
                        'end_time': format_datetime(session.end_time),
                        'summary': summary
                    })
        
        return summaries


def process_user_chat_data(user_id: str, conversation_id: int, start_time: datetime.datetime, end_time: datetime.datetime) -> Dict[str, Any]:
    """处理用户聊天数据
    
    Args:
        user_id: 用户 ID
        start_time: 开始时间
        end_time: 结束时间
        
    Returns:
        处理结果
    """
    # 获取聊天历史
    messages = get_chat_history(user_id,0, start_time, end_time)
   
    
    # 切分为会话
    sessions = split_into_sessions(messages)
    
    # 生成摘要
    summarizer = ChatSummarizer()
    summaries = summarizer.summarize_chat_history(sessions)
    
    return {
        'user_id': user_id,
        'conversation_id': conversation_id,
        'session_count': len(sessions),
        'total_messages': len(messages),
        'summaries': summaries
    }
