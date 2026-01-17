

from conf import settings
from elasticsearch_dsl import Document, Date, Integer, Keyword, Text, connections,DenseVector

connections.create_connection(hosts=settings.es_host, http_auth=(settings.es_user, settings.es_password),
                              verify_certs=False, ssl_assert_hostname=False)

from rag.items import YinyutlItem
from elasticsearch_dsl.query import Script


import urllib3
import warnings
import tqdm
from bs4 import BeautifulSoup
from pathlib import Path
import os

from rag.embedding import get_embedding



# 屏蔽 InsecureRequestWarning 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", message="Connecting to 'https://localhost:9200' using TLS")

from pydantic import BaseModel
from typing import List
from datetime import datetime


# 数据模型类定义
class CoreMemoryItem(BaseModel):
    id: str
    thread_id: str
    content: str
    last_update: datetime

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, CoreMemoryItem):
            return self.id == other.id
        return False

class ArchivalMemoryItem(BaseModel):
    id: str
    thread_id: str
    content: str
    tags: str
    created_at: datetime

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, ArchivalMemoryItem):
            return self.id == other.id
        return False

class ChatLogItem(BaseModel):
    id: str
    thread_id: str
    role: str
    content: str
    created_at: datetime

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, ChatLogItem):
            return self.id == other.id
        return False


# 向量数据库表定义

# A. 核心记忆 (Core Memory) - 存储用户画像
class CoreMemory(Document):
    thread_id = Keyword()  # 会话ID，作为索引
    content = Text(analyzer="smartcn")  # 用户画像内容
    vector = DenseVector(dims=1024, index=True, similarity="cosine")  # 内容向量
    last_update = Date()  # 最后更新时间

    class Index:
        name = 'core_memory'
        settings = {
            "number_of_shards": 2,
        }

    def query_by_thread_id(self, thread_id):
        """根据会话ID查询核心记忆"""
        s = self.search()
        s = s.query("term", thread_id=thread_id)
        items = s.execute()
        return [CoreMemoryItem(
            id=item.meta.id,
            thread_id=item.thread_id,
            content=item.content,
            last_update=item.last_update
        ) for item in items]

    def update_by_thread_id(self, thread_id, content, vector):
        """更新指定会话的核心记忆"""
        s = self.search()
        s = s.query("term", thread_id=thread_id)
        items = s.execute()
        
        if items:
            # 如果存在，更新
            item = items[0]
            item.content = content
            item.vector = vector
            item.last_update = datetime.now()
            item.save()
        else:
            # 如果不存在，创建新记录
            new_item = CoreMemory(
                thread_id=thread_id,
                content=content,
                vector=vector,
                last_update=datetime.now()
            )
            new_item.save()

# B. 归档记忆 (Archival Memory) - 存储具体事件和知识
class ArchivalMemory(Document):
    thread_id = Keyword()  # 会话ID
    content = Text(analyzer="smartcn")  # 事件或知识内容
    vector = DenseVector(dims=1024, index=True, similarity="cosine")  # 内容向量
    tags = Keyword(multi=True)  # 标签，支持多值
    created_at = Date()  # 创建时间

    class Index:
        name = 'archival_memory'
        settings = {
            "number_of_shards": 2,
        }

    def add(self, thread_id, content, vector, tags=None):
        """添加归档记忆"""
        new_item = ArchivalMemory(
            thread_id=thread_id,
            content=content,
            vector=vector,
            tags=tags or [],
            created_at=datetime.now()
        )
        new_item.save()

    def search_by_vector(self, thread_id, query_vector, k=10):
        """通过向量检索归档记忆"""
        s = self.search()
        s = s.query("match", thread_id=thread_id)
        s = s.knn(field='vector', k=k, num_candidates=100, query_vector=query_vector)
        items = s.execute()
        return [ArchivalMemoryItem(
            id=item.meta.id,
            thread_id=item.thread_id,
            content=item.content,
            tags=",".join(item.tags) if item.tags else "",
            created_at=item.created_at
        ) for item in items]

    def search_by_keyword(self, thread_id, keyword, k=10):
        """通过关键词检索归档记忆"""
        s = self.search()
        s = s.query("match", thread_id=thread_id)
        s = s.query("match", content=keyword)
        s = s[:k]
        items = s.execute()
        return [ArchivalMemoryItem(
            id=item.meta.id,
            thread_id=item.thread_id,
            content=item.content,
            tags=",".join(item.tags) if item.tags else "",
            created_at=item.created_at
        ) for item in items]

    def delete_by_id(self, id):
        """根据ID删除归档记忆"""
        s = self.search()
        s = s.query("match", id=id)
        s.delete()
        

    

# C. 对话日志 (Chat Log) - 存储完整对话历史
class ChatLog(Document):
    thread_id = Keyword()  # 会话ID
    role = Keyword()  # 角色：user, ai, system
    content = Text(analyzer="smartcn")  # 对话内容
    vector = DenseVector(dims=1024, index=True, similarity="cosine")  # 内容向量
    created_at = Date()  # 创建时间

    class Index:
        name = 'chat_log'
        settings = {
            "number_of_shards": 2,
        }

    def add(self, thread_id, role, content, vector):
        """添加对话日志"""
        new_item = ChatLog(
            thread_id=thread_id,
            role=role,
            content=content,
            vector=vector,
            created_at=datetime.now()
        )
        new_item.save()

    def get_by_thread_id(self, thread_id, limit=100):
        """获取指定会话的对话历史"""
        s = self.search()
        s = s.query("match", thread_id=thread_id)
        s = s.sort("created_at")  # 按时间排序
        s = s[:limit]  # 限制返回数量
        items = s.execute()
        return [ChatLogItem(
            id=item.meta.id,
            thread_id=item.thread_id,
            role=item.role,
            content=item.content,
            created_at=item.created_at
        ) for item in items]

    def search_by_vector(self, thread_id, query_vector, k=10):
        """通过向量检索对话历史"""
        s = self.search()
        s = s.query("match", thread_id=thread_id)
        s = s.knn(field='vector', k=k, num_candidates=100, query_vector=query_vector)
        items = s.execute()
        return [ChatLogItem(
            id=item.meta.id,
            thread_id=item.thread_id,
            role=item.role,
            content=item.content,
            created_at=item.created_at
        ) for item in items]








if __name__ == '__main__':
    # 初始化所有索引
    print("初始化向量索引...")
    CoreMemory.init()
    ArchivalMemory.init()
    ChatLog.init()
    print("所有向量索引已初始化")
    
    # 测试核心记忆功能
    print("\n测试核心记忆功能...")
    core_mem = CoreMemory()
    test_vector = get_embedding("测试用户画像").tolist()
    core_mem.update_by_thread_id("test_thread_1", "用户叫张三，程序员，喜欢吃辣", test_vector)
    
    # 查询核心记忆
    result = core_mem.query_by_thread_id("test_thread_1")
    if result:
        print(f"核心记忆查询结果: {result[0].content}")
    
    # 测试归档记忆功能
    print("\n测试归档记忆功能...")
    archival_mem = ArchivalMemory()
    archival_vector = get_embedding("下周一去北京出差").tolist()
    archival_mem.add("test_thread_1", "下周一去北京出差", archival_vector, tags=["出差", "北京"])
    
    # 测试向量检索
    print("\n测试向量检索功能...")
    query_vector = get_embedding("最近的出差计划").tolist()
    search_results = archival_mem.search_by_vector("test_thread_1", query_vector)
    print(f"向量检索结果: {[item.content for item in search_results]}")
    
    # 测试对话日志功能
    print("\n测试对话日志功能...")
    chat_log = ChatLog()
    user_vector = get_embedding("你好，我叫张三").tolist()
    chat_log.add("test_thread_1", "user", "你好，我叫张三", user_vector)
    
    # 查询对话历史
    chat_history = chat_log.get_by_thread_id("test_thread_1")
    print(f"对话历史: {[item.content for item in chat_history]}")
    
    print("\n测试完成")



              
        





