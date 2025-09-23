# -*- coding: utf-8 -*-
"""
scientific_agent_refactored.py (V2)

一个经过重构的、功能完善的科学计算 Agent。
支持流式输出、有限历史记录、自动重试和多步骤工具调用。
具有更精细的输出格式控制和更智能的行为逻辑。
"""

import os
import json
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Generator, Any

import dotenv
import numpy as np
import sympy
from openai import APIError, OpenAI, RateLimitError
from scipy import integrate

# --- 1. 配置中心 (Centralized Configuration) ---
@dataclass
class AppConfig:
    """应用程序的配置"""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    # [修改] 增强的系统提示，指导Agent的行为模式
    system_prompt: str = (
        "You are a powerful scientific computing assistant."
        "\n### INSTRUCTIONS:\n"
        "1.  **Analyze the user's query:** Determine if it's a simple task, a complex multi-step task, or a general question.\n"
        "2.  **For simple tasks:**\n"
        "    - First, briefly state which tool you will use.\n"
        "    - Then, call the tool.\n"
        "3.  **For complex, multi-step tasks:**\n"
        "    - You MUST first outline a step-by-step plan. For example: 'Here is my plan:\\nStep 1: Differentiate the function.\\nStep 2: Solve the resulting equation.'\n"
        "    - After presenting the plan, execute each step sequentially. For each step, follow the procedure for simple tasks (state the tool, then call it).\n"
        "4.  **Final Answer:** After all necessary tool calls are complete and you have the definitive answer to the user's original query, you MUST provide a final, comprehensive summary prefixed with `Final Answer:`."
    )
    max_history: int = 10
    max_tokens: int = 4000
    model: str = "deepseek-r1"
    max_retries: int = 3
    backoff_factor: float = 1.5

    def __post_init__(self):
        dotenv.load_dotenv()
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        if self.base_url is None:
            self.base_url = os.getenv("OPENAI_BASE_URL")
        if not self.api_key:
            raise ValueError("未能获取到 OPENAI_API_KEY。请检查 .env 文件或环境变量设置。")

# --- 2. 工具箱 (ToolBox) ---
# ToolBox 类保持不变，其设计已经足够解耦
class ToolBox:
    @staticmethod
    def solve_symbolic_equation(equation: str, variable: str = 'x') -> str:
        # ... (代码不变) ...
        try:
            var_symbol = sympy.symbols(variable)
            expr = sympy.sympify(equation)
            solution = sympy.solve(expr, var_symbol)
            return f"The solution to the equation is: {solution}"
        except Exception as e:
            return f"Error while solving: {e}"

    @staticmethod
    def perform_symbolic_differentiation(expression: str, variable: str = 'x') -> str:
        # ... (代码不变) ...
        try:
            var_symbol = sympy.symbols(variable)
            expr = sympy.sympify(expression)
            derivative = sympy.diff(expr, var_symbol)
            return f"The derivative of the expression is: {derivative}"
        except Exception as e:
            return f"Error while differentiating: {e}"
        
    @staticmethod
    def calculate_numerical_integral(func_str: str, lower_bound: float, upper_bound: float) -> str:
        # ... (代码不变) ...
        try:
            func = eval(func_str, {"np": np})
            lower = np.inf if str(lower_bound) == 'inf' else -np.inf if str(lower_bound) == '-inf' else lower_bound
            upper = np.inf if str(upper_bound) == 'inf' else -np.inf if str(upper_bound) == '-inf' else upper_bound
            result, error = integrate.quad(func, lower, upper)
            return f"The integral result is approximately: {result} with an error of {error}"
        except Exception as e:
            return f"Error during integration: {e}"

    def get_tools_schema(self) -> List[Dict]:
        # ... (代码不变) ...
        return [
            {"type": "function", "function": {
                "name": "solve_symbolic_equation",
                "description": "Solves a symbolic mathematical equation.",
                "parameters": {"type": "object", "properties": {
                    "equation": {"type": "string", "description": "The equation to solve, e.g., 'x**2 - 4'."},
                    "variable": {"type": "string", "description": "The variable to solve for, defaults to 'x'."}
                }, "required": ["equation"]}
            }},
            {"type": "function", "function": {
                "name": "perform_symbolic_differentiation",
                "description": "Performs symbolic differentiation on a mathematical expression.",
                "parameters": {"type": "object", "properties": {
                    "expression": {"type": "string", "description": "The expression to differentiate, e.g., 'x**3 + sin(x)'."},
                    "variable": {"type": "string", "description": "The variable to differentiate with respect to, defaults to 'x'."}
                }, "required": ["expression"]}
            }},
            {"type": "function", "function": {
                "name": "calculate_numerical_integral",
                "description": "Calculates the numerical definite integral of a function over a given interval.",
                "parameters": {"type": "object", "properties": {
                    "func_str": {"type": "string", "description": "A lambda string representing the function, e.g., 'lambda x: np.exp(-x**2)'."},
                    "lower_bound": {"type": "number", "description": "The lower bound of integration."},
                    "upper_bound": {"type": "number", "description": "The upper bound of integration."}
                }, "required": ["func_str", "lower_bound", "upper_bound"]}
            }}
        ]

    def get_function_map(self) -> Dict[str, callable]:
        # ... (代码不变) ...
        return {
            "solve_symbolic_equation": self.solve_symbolic_equation,
            "perform_symbolic_differentiation": self.perform_symbolic_differentiation,
            "calculate_numerical_integral": self.calculate_numerical_integral,
        }

# --- 3. 对话管理器 (Conversation Manager) ---
# (代码不变)
class ConversationManager:
    # ... (代码不变) ...
    def __init__(self, config: AppConfig):
        self.config = config
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.config.system_prompt}
        ]
    def add_message(self, role: str, content: Any, tool_calls: Optional[List] = None, tool_call_id: Optional[str] = None, name: Optional[str] = None):
        message = {"role": role, "content": content}
        if tool_calls: message["tool_calls"] = tool_calls
        if tool_call_id: message["tool_call_id"] = tool_call_id
        if name: message["name"] = name
        self.messages.append(message)
        self._cleanup_history()
    def _cleanup_history(self):
        system_message = self.messages[0]
        other_messages = self.messages[1:]
        if len(other_messages) > self.config.max_history * 2:
            other_messages = other_messages[-self.config.max_history * 2:]
        self.messages = [system_message] + other_messages
    def get_api_messages(self) -> List[Dict[str, Any]]:
        return self.messages

# --- 4. 大模型客户端 (LLM Client) ---
# (代码不变)
class LLMClient:
    # ... (代码不变) ...
    def __init__(self, config: AppConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def get_response_stream(
        self, messages: List[Dict[str, Any]], tools: List[Dict]
    ) -> Generator[Dict[str, Any], None, None]:
        # ... (代码不变) ...
        for attempt in range(self.config.max_retries):
            try:
                stream = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    stream=True,
                    stream_options={"include_usage": True},
                )
                for chunk in stream:
                    yield chunk.model_dump()
                return
            except RateLimitError as e:
                if attempt >= self.config.max_retries - 1:
                    yield {"type": "error", "content": f"Rate limit exceeded after multiple retries: {e}"}
                    break
                wait_time = self.config.backoff_factor * (2 ** attempt) + random.uniform(0, 1)
                yield {"type": "error", "content": f"Rate limited. Retrying in {wait_time:.2f}s..."}
                time.sleep(wait_time)
            except APIError as e:
                yield {"type": "error", "content": f"API Error: {e}"}
                break
            except Exception as e:
                yield {"type": "error", "content": f"An unexpected error occurred: {e}"}
                break
                
# --- 5. 终端界面 (Terminal UI) ---
class TerminalUI:
    @staticmethod
    def get_user_input() -> str:
        return input("你: ")
    
    # [修改] 将工具调用和结果合并显示，格式更紧凑
    @staticmethod
    def display_tool_interaction(tool_name: str, args: Dict, result: str):
        print(f"\n> 调用工具: {tool_name}({json.dumps(args, ensure_ascii=False)})")
        print(f"< 工具返回: {result}")

    @staticmethod
    def display_error(message: str):
        print(f"\n[错误] {message}")

    @staticmethod
    def display_assistant_response_start():
        print("助手: ", end="", flush=True)

    @staticmethod
    def display_assistant_response_chunk(content: str):
        print(content, end="", flush=True)

# --- 6. 应用主控 (Main Application) ---
class ScientificAgentApp:
    def __init__(self, config: AppConfig):
        self.config = config
        self.toolbox = ToolBox()
        self.conversation = ConversationManager(config)
        self.llm_client = LLMClient(config)
        self.ui = TerminalUI()
        self.function_map = self.toolbox.get_function_map()

    def run(self):
        print("科学计算 Agent 已启动。输入 'exit' 或 'quit' 退出。")
        while True:
            try:
                user_input = self.ui.get_user_input()
                if user_input.strip().lower() in ["exit", "quit"]:
                    break
                self.conversation.add_message("user", user_input)
                self._execute_agent_loop()
            except (KeyboardInterrupt, EOFError):
                break
        print("\n感谢使用，再见！")

    def _execute_agent_loop(self):
        # [修改] 引入状态标志，确保 "助手: " 每轮只打印一次
        assistant_prefix_printed = False

        while True:
            api_messages = self.conversation.get_api_messages()
            tools_schema = self.toolbox.get_tools_schema()
            
            stream = self.llm_client.get_response_stream(api_messages, tools_schema)

            full_response_content = ""
            tool_calls = []
            
            # [修改] 只有在从未打印过 "助手: " 的情况下才打印
            if not assistant_prefix_printed:
                self.ui.display_assistant_response_start()
                assistant_prefix_printed = True

            for chunk_data in stream:
                # ... (错误处理部分不变) ...
                if chunk_data.get("type") == "error":
                    self.ui.display_error(chunk_data.get("content", "未知错误"))
                    return
                choices = chunk_data.get("choices", [])
                if not choices: continue
                delta = choices[0].get("delta", {})
                
                if content_chunk := delta.get("content"):
                    full_response_content += content_chunk
                    self.ui.display_assistant_response_chunk(content_chunk)
                
                if tool_call_chunks := delta.get("tool_calls"):
                    # ... (工具调用块的拼接逻辑不变) ...
                    for tool_call_chunk in tool_call_chunks:
                        if len(tool_calls) <= tool_call_chunk["index"]:
                            tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        tc = tool_calls[tool_call_chunk["index"]]
                        if tool_call_chunk.get("id"): tc["id"] = tool_call_chunk["id"]
                        if func := tool_call_chunk.get("function"):
                            if name := func.get("name"): tc["function"]["name"] += name
                            if args := func.get("arguments"): tc["function"]["arguments"] += args
            
            # [修改] 将换行控制移到这里，在流式输出结束后执行
            # 如果后面紧跟着工具调用，这个换行可以起到分隔作用
            # 如果后面没有内容，也不会产生多余的空行
            if full_response_content:
                print()

            # 根据`Final Answer:`信号和工具调用来决定下一步
            is_final_answer = full_response_content.strip().startswith("Final Answer:")
            
            if is_final_answer or not tool_calls:
                # 如果有 "Final Answer:" 信号，或者没有任何工具调用，则认为是最终答案
                self.conversation.add_message("assistant", full_response_content)
                break
            else:
                # 有工具调用，执行它们并继续循环
                self.conversation.add_message("assistant", full_response_content, tool_calls=tool_calls)
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    try:
                        tool_args = json.loads(tool_call["function"]["arguments"])
                        function_to_call = self.function_map[tool_name]
                        result = function_to_call(**tool_args)
                        self.ui.display_tool_interaction(tool_name, tool_args, result)
                        self.conversation.add_message("tool", result, tool_call_id=tool_call["id"], name=tool_name)
                    except json.JSONDecodeError:
                        error_msg = f"Error: Invalid JSON arguments from model for tool {tool_name}."
                        self.ui.display_error(error_msg)
                        self.conversation.add_message("tool", error_msg, tool_call_id=tool_call["id"], name=tool_name)
                continue

# --- 程序入口 ---
# (代码不变)
if __name__ == "__main__":
    try:
        app_config = AppConfig()
        app = ScientificAgentApp(app_config)
        app.run()
    except ValueError as e:
        print(f"[配置错误] {e}")