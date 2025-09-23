from typing import List, Dict, Optional
from datetime import datetime
import json

class ConversationManager:
    """对话管理器"""
    
    def __init__(self, max_history: int = 10, max_tokens: int = 3000):
        self.messages: List[Dict[str, str]] = []
        self.max_history = max_history
        self.max_tokens = max_tokens
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "total_exchanges": 0
        }
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """添加消息到对话历史"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            message["metadata"] = metadata
        
        self.messages.append(message)
        
        if role == "user":
            self.metadata["total_exchanges"] += 1
        
        # 自动清理历史
        self._cleanup_history()
    
    def _cleanup_history(self):
        """清理过长的对话历史"""
        # 保留系统消息
        system_messages = [msg for msg in self.messages if msg["role"] == "system"]
        other_messages = [msg for msg in self.messages if msg["role"] != "system"]
        
        # 限制历史长度
        if len(other_messages) > self.max_history:
            other_messages = other_messages[-self.max_history:]
        
        # 限制 token 数量
        while self._estimate_tokens(system_messages + other_messages) > self.max_tokens and other_messages:
            other_messages.pop(0)
        
        self.messages = system_messages + other_messages
    
    def _estimate_tokens(self, messages: List[Dict]) -> int:
        """估算消息的 token 数量"""
        total_text = " ".join([msg["content"] for msg in messages])
        chinese_chars = len([c for c in total_text if '\u4e00' <= c <= '\u9fff'])
        other_chars = len(total_text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)
    
    def get_messages(self) -> List[Dict[str, str]]:
        """获取用于 API 调用的消息格式"""
        return [{"role": msg["role"], "content": msg["content"]} for msg in self.messages]
    
    def save_to_file(self, filename: str):
        """保存对话到文件"""
        data = {
            "messages": self.messages,
            "metadata": self.metadata
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_from_file(self, filename: str):
        """从文件加载对话"""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.messages = data["messages"]
        self.metadata = data["metadata"]

# 使用示例
def chat_with_memory():
    conversation = ConversationManager()
    
    # 设置系统消息
    conversation.add_message("system", "你是一个有用的数学助手，擅长解释复杂的数学概念。")
    
    # 模拟对话
    conversation.add_message("user", "什么是导数？")
    conversation.add_message("assistant", "导数是函数在某一点的瞬时变化率...")
    
    conversation.add_message("user", "能给我一个具体的例子吗？")
    conversation.add_message("assistant", "当然！比如函数 f(x) = x²...")
    
    # 获取用于 API 的消息
    api_messages = conversation.get_messages()
    print("API 消息格式:")
    for msg in api_messages:
        print(f"{msg['role']}: {msg['content'][:50]}...")