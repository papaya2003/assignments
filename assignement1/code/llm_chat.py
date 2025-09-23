from openai import OpenAI, RateLimitError, APIError
import os
from dotenv import load_dotenv
import time
import random
from typing import List, Dict, Optional
from datetime import datetime

# 加载环境变量
load_dotenv('./code/.env')

# 1. 准备工作：初始化客户端
# 建议通过环境变量配置API Key，避免硬编码。
try:
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
except KeyError:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

# 2. 对话管理器
class ConversationManager:
    """对话管理器，支持记忆长度和token限制"""
    def __init__(self, max_history: int = 10, max_tokens: int = 3000):
        self.messages: List[Dict[str, str]] = []
        self.max_history = max_history
        self.max_tokens = max_tokens
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "total_exchanges": 0
        }
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
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
        self._cleanup_history()
    
    def _cleanup_history(self):
        system_messages = [msg for msg in self.messages if msg["role"] == "system"]
        other_messages = [msg for msg in self.messages if msg["role"] != "system"]
        if len(other_messages) > self.max_history:
            other_messages = other_messages[-self.max_history:]
        while self._estimate_tokens(system_messages + other_messages) > self.max_tokens and other_messages:
            other_messages.pop(0)
        self.messages = system_messages + other_messages
    
    def _estimate_tokens(self, messages: List[Dict]) -> int:
        total_text = " ".join([msg["content"] for msg in messages])
        chinese_chars = len([c for c in total_text if '\u4e00' <= c <= '\u9fff'])
        other_chars = len(total_text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)
    
    def get_messages(self) -> List[Dict[str, str]]:
        return [{"role": msg["role"], "content": msg["content"]} for msg in self.messages]

# 选择模型
def select_model():
    models = [
        "qwen-plus",
        "deepseek-v3.1",
        "Moonshot-Kimi-K2-Instruct",
        "glm-4.5"
    ]
    print("请选择模型：")
    for idx, m in enumerate(models, 1):
        print(f"{idx}. {m}")
    while True:
        choice = input("输入模型编号（默认1）: ").strip()
        if choice == "":
            return models[0]
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice)-1]
        print("输入有误，请重新选择。")

# 3. 带重试机制的API调用
def call_openai_with_retry(
        client: OpenAI, 
        model: str, 
        messages: List[Dict[str, str]], 
        max_retries: int = 3, 
        backoff_factor: float = 1.0) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True, 
                stream_options={"include_usage": True}
            )
            # 处理流式响应
            # 使用列表推导式和join()是处理大量文本片段时最高效的方式。
            content_parts = []
            print("助手: ", end="", flush=True)
            for chunk in completion:
                # 最后一个chunk不包含choices，但包含usage信息。
                if chunk.choices:
                    # 关键：delta.content可能为None，使用`or ""`避免拼接时出错。
                    content = chunk.choices[0].delta.content or ""
                    print(content, end="", flush=True)
                    content_parts.append(content)
                elif chunk.usage:
                    # 请求结束，打印Token用量。
                    print("\n--- 请求用量 ---")
                    print(f"输入 Tokens: {chunk.usage.prompt_tokens}")
                    print(f"输出 Tokens: {chunk.usage.completion_tokens}")
                    print(f"总计 Tokens: {chunk.usage.total_tokens}")
            full_response = "".join(content_parts)
            return full_response
        except RateLimitError as e:
            if attempt == max_retries - 1:
                print(f"请求次数过多，已达最大重试次数。调用失败：{e}")
                return None
            wait_time = backoff_factor * (2 ** attempt) + random.uniform(0, 1)
            print(f"请求失败: {e}. {wait_time:.2f}秒后重试...")
            time.sleep(wait_time)
        except APIError as e:
            print(f"API 请求失败: {e}")
            return None
        except Exception as e:
            print(f"发生错误: {e}. 调用失败。")
            return None
    return None

# 4. 多轮对话主循环
def chat_loop(max_history: int = 10, max_tokens: int = 3000):
    model = select_model()
    conversation = ConversationManager(max_history=max_history, max_tokens=max_tokens)
    # 设置系统消息
    conversation.add_message('system', 'You are a helpful assistant.')
    print(f"已选择模型：{model}")
    print("欢迎进入多轮对话，输入 'exit' 退出。")
    while True:
        user_input = input("你: ")
        if user_input.strip().lower() == 'exit':
            print("已退出对话。")
            break
        conversation.add_message('user', user_input)
        api_messages = conversation.get_messages()
        reply = call_openai_with_retry(client, model, api_messages)
        if reply is None:
            print("对话调用失败，结束对话。")
            break
        else:
            conversation.add_message('assistant', reply)

if __name__ == "__main__":
    chat_loop()