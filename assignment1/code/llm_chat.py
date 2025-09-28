import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Generator, Tuple

from dotenv import load_dotenv
from openai import APIError, OpenAI, RateLimitError

# 1. 配置中心
# 使用 dataclass 将所有配置项集中管理。

@dataclass
class AppConfig:

    # API 凭证设置
    api_key: Optional[str] = None
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    # 对话设置
    system_prompt: str = "You are a helpful assistant."
    max_history: int = 10
    max_tokens: int = 4000
    
    # API 调用设置
    available_models: List[str] = field(default_factory=lambda: [
        "qwen-plus",
        "qwen-max",
        "deepseek-v2",
        "glm-4"
    ])
    default_model: str = "qwen-plus"
    
    # 重试设置
    max_retries: int = 3
    backoff_factor: float = 1.5

    # 程序会先加载AppConfig，而load_dotenv在后面的主程序中，所以不能在上面直接调用os.getenv
    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("未能获取到 DASHSCOPE_API_KEY。请检查 .env 文件或环境变量设置。")


# 2. 对话历史管理

class ConversationManager:

    def __init__(self, config: AppConfig):
        self.config = config
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.config.system_prompt}
        ]

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._cleanup_history()

    def _cleanup_history(self):
        # 保留系统消息
        system_message = self.messages[0]
        user_assistant_messages = self.messages[1:]

        # 按对话轮数截断
        if len(user_assistant_messages) > self.config.max_history * 2:
            user_assistant_messages = user_assistant_messages[-self.config.max_history * 2:]
        
        # 按token数截断 (从最旧的开始移除)
        while self._estimate_tokens(user_assistant_messages) > self.config.max_tokens and user_assistant_messages:
            user_assistant_messages.pop(0)
            user_assistant_messages.pop(0) # 一次移除一对 user-assistant

        self.messages = [system_message] + user_assistant_messages

    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        # 粗略估算token数
        text = " ".join(msg["content"] for msg in messages)
        return int(len(text) / 2.5)

    def get_api_messages(self) -> List[Dict[str, str]]:
        # 获取用于API调用的消息列表
        return self.messages


# 3. API 通信
# LLMClient 负责与 API 交互。

class LLMClient:

    def __init__(self, config: AppConfig):
        if not config.api_key:
            raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.config = config

    def stream_chat_with_retry(
        self, messages: List[Dict[str, str]], model: str
    ) -> Generator[Tuple[str, Optional[Dict]], None, None]:
        """
        带重试机制的流式API调用。使用生成器(yield)逐块返回内容，将UI处理与API逻辑分离。
        """
        for attempt in range(self.config.max_retries):
            try:
                stream = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                for chunk in stream:
                    content = chunk.choices[0].delta.content if chunk.choices else ""
                    usage = chunk.usage.model_dump() if chunk.usage else None
                    yield content or "", usage
                return
            except RateLimitError as e:
                if attempt == self.config.max_retries - 1:
                    print(f"\n错误：达到最大重试次数，请求失败。{e}")
                    break
                wait_time = self.config.backoff_factor * (2 ** attempt) + random.uniform(0, 1)
                print(f"请求速率受限。将在 {wait_time:.2f} 秒后重试...")
                time.sleep(wait_time)
            except APIError as e:
                print(f"\n错误：API 请求失败。{e}")
                break
            except Exception as e:
                print(f"\n未知错误：{e}")
                break


# 4. UI
# TerminalUI 负责在终端上显示内容和接收用户输入。

class TerminalUI:

    @staticmethod
    def select_model(config: AppConfig) -> str:
        # 让用户从可用模型列表中选择一个
        print("请选择要使用的模型：")
        for i, model_name in enumerate(config.available_models, 1):
            print(f"{i}. {model_name}")
        
        while True:
            choice = input(f"输入编号 (回车默认: {config.default_model}): ").strip()
            if not choice:
                return config.default_model
            if choice.isdigit() and 1 <= int(choice) <= len(config.available_models):
                return config.available_models[int(choice) - 1]
            print("无效输入，请重新选择。")

    @staticmethod
    def display_streamed_response(stream: Generator[Tuple[str, Optional[Dict]], None, None]) -> Tuple[str, Optional[Dict]]:
        # 处理并显示流式响应
        full_response = []
        final_usage = None
        print("助手: ", end="", flush=True)
        for content, usage in stream:
            print(content, end="", flush=True)
            full_response.append(content)
            if usage:
                final_usage = usage
        print() # 换行
        return "".join(full_response), final_usage

    @staticmethod
    def display_usage(usage: Optional[Dict]):
        # 显示token用量
        if usage:
            print("---")
            print(f"Token 用量: "
                  f"输入={usage['prompt_tokens']}, "
                  f"输出={usage['completion_tokens']}, "
                  f"总计={usage['total_tokens']}")
            print("---")


# 5. 应用主程序
# ChatApplication 将所有组件串联起来，驱动整个应用。

class ChatApplication:

    def __init__(self, config: AppConfig):
        self.config = config
        self.llm_client = LLMClient(config)
        self.conversation = ConversationManager(config)
        self.ui = TerminalUI()

    def run(self):
        # 启动应用主循环
        model = self.ui.select_model(self.config)
        print(f"\n已选择模型: {model}。输入 'exit' 或 'quit' 退出对话。")
        
        while True:
            try:
                user_input = input("你: ")
                if user_input.strip().lower() in ["exit", "quit"]:
                    print("感谢使用，再见！")
                    break

                self.conversation.add_message("user", user_input)
                
                api_messages = self.conversation.get_api_messages()
                stream = self.llm_client.stream_chat_with_retry(api_messages, model)
                
                assistant_reply, usage = self.ui.display_streamed_response(stream)

                if assistant_reply:
                    self.conversation.add_message("assistant", assistant_reply)
                    self.ui.display_usage(usage)
                else:
                    print("未能获取助手回复，请检查网络或API凭证。")

            except (KeyboardInterrupt, EOFError):
                print("\n对话已中断。感谢使用，再见！")
                break


# 主程序
if __name__ == "__main__":
    load_dotenv()
    app_config = AppConfig()
    app = ChatApplication(app_config)
    app.run()