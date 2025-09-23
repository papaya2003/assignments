# -*- coding: utf-8 -*-

"""
scientific_agent.py

一个简易的科学计算 Agent 实现。
该 Agent 使用 OpenAI 的 tool-calling 功能来调用外部科学计算库 (SymPy, SciPy)
从而解决用户提出的数学和科学问题。
"""

import os
import dotenv
import json
import numpy as np
import sympy
from scipy import integrate
from openai import OpenAI

dotenv.load_dotenv()  # 从 .env 文件加载环境变量

# --- 1. 工具函数定义 ---
# 我们将 sympy 和 scipy 的功能封装成独立的函数。
# 重点：每个函数的 docstring 都非常重要，它将作为 schema 提供给 LLM，
# LLM 会根据 docstring 的描述来决定调用哪个函数以及如何传递参数。

def solve_symbolic_equation(equation: str, variable: str = 'x') -> str:
    """
    求解一个符号数学方程。
    例如，要解方程 x^2 - 4 = 0，请将 'x**2 - 4' 作为 equation 参数。
    支持求解一元多次方程、超越方程等。

    :param equation: 需要求解的方程表达式字符串，例如 'x**2 - 4' 或 'sin(x) - 0.5'。
    :param variable: 方程中的变量，默认为 'x'。
    :return: 以字符串形式返回方程的解。
    """
    print(f"--- 调用工具: solve_symbolic_equation(equation='{equation}', variable='{variable}') ---")
    try:
        var_symbol = sympy.symbols(variable)
        # 使用 sympy.sympify 将字符串表达式转换为 sympy 可以理解的格式
        expr = sympy.sympify(equation)
        solution = sympy.solve(expr, var_symbol)
        return f"方程的解是: {solution}"
    except Exception as e:
        return f"求解时发生错误: {e}"

def perform_symbolic_differentiation(expression: str, variable: str = 'x') -> str:
    """
    对一个数学表达式进行符号求导。
    例如，要计算 x^3 + sin(x) 的导数。

    :param expression: 需要求导的表达式字符串，例如 'x**3 + sin(x)'。
    :param variable: 求导所针对的变量，默认为 'x'。
    :return: 以字符串形式返回求导后的表达式。
    """
    print(f"--- 调用工具: perform_symbolic_differentiation(expression='{expression}', variable='{variable}') ---")
    try:
        var_symbol = sympy.symbols(variable)
        expr = sympy.sympify(expression)
        derivative = sympy.diff(expr, var_symbol)
        return f"表达式的导数是: {derivative}"
    except Exception as e:
        return f"求导时发生错误: {e}"

def calculate_numerical_integral(
    func_str: str, 
    lower_bound: float, 
    upper_bound: float
) -> str:
    """
    计算一个函数在给定区间的定积分（数值积分）。
    函数表达式应为一个可以被 Python 'eval' 的 lambda 函数字符串。
    例如，要计算函数 f(x) = x^2 在 [0, 1] 上的定积分，func_str 应为 'lambda x: x**2'。
    无穷大和负无穷大可以用 'inf' 和 '-inf' 表示。

    :param func_str: 代表函数的 lambda 字符串，例如 'lambda x: x**2'。
    :param lower_bound: 积分下限。
    :param upper_bound: 积分上限。
    :return: 以字符串形式返回积分结果和估算的误差。
    """
    print(f"--- 调用工具: calculate_numerical_integral(func_str='{func_str}', lower_bound={lower_bound}, upper_bound={upper_bound}) ---")
    try:
        # 安全警告: 在生产环境中使用 eval 是危险的。这里仅为作业演示目的。
        func = eval(func_str)
        
        # 将字符串 'inf' 和 '-inf' 转换为 numpy 的无穷大对象
        lower = np.inf if str(lower_bound).lower() == 'inf' else -np.inf if str(lower_bound).lower() == '-inf' else lower_bound
        upper = np.inf if str(upper_bound).lower() == 'inf' else -np.inf if str(upper_bound).lower() == '-inf' else upper_bound

        result, error = integrate.quad(func, lower, upper)
        return f"积分结果约为: {result}, 估算误差为: {error}"
    except Exception as e:
        return f"计算积分时发生错误: {e}"


# --- 2. Agent 核心逻辑 ---

class ScientificAgent:
    def __init__(self, model="deepseek-r1"):
        """
        初始化 Agent。
        :param model: 使用的 OpenAI 模型，推荐使用支持 tool-calling 的新模型。
        """
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="http://162.105.151.181/v1"
        ) # 会自动从环境变量读取 OPENAI_API_KEY
        self.model = model
        self.messages = [
            {"role": "system", "content": "你是一个强大的科学计算助手。你能理解用户的数学问题，并调用合适的工具来解决它们。最后，用清晰的自然语言向用户解释结果。"}
        ]
        
        # 将我们定义的 Python 函数映射到 Agent 的工具箱中
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "solve_symbolic_equation",
                    "description": "求解一个符号数学方程，例如 'x**2 = 4'。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "equation": {"type": "string", "description": "需要求解的方程表达式字符串，例如 'x**2 - 4'"},
                            "variable": {"type": "string", "description": "方程中的变量，默认为 'x'"}
                        },
                        "required": ["equation"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "perform_symbolic_differentiation",
                    "description": "对一个数学表达式进行符号求导。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "需要求导的表达式字符串，例如 'x**3 + sin(x)'"},
                            "variable": {"type": "string", "description": "求导所针对的变量，默认为 'x'"}
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_numerical_integral",
                    "description": "计算一个函数在给定区间的定积分（数值积分）。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "func_str": {"type": "string", "description": "代表函数的 lambda 字符串, 例如 'lambda x: x**2'"},
                            "lower_bound": {"type": "number", "description": "积分下限"},
                            "upper_bound": {"type": "number", "description": "积分上限"}
                        },
                        "required": ["func_str", "lower_bound", "upper_bound"]
                    }
                }
            }
        ]
        
        # 创建一个函数名到实际函数对象的映射，方便后续调用
        self.available_functions = {
            "solve_symbolic_equation": solve_symbolic_equation,
            "perform_symbolic_differentiation": perform_symbolic_differentiation,
            "calculate_numerical_integral": calculate_numerical_integral,
        }

    def run(self, user_query: str):
        """
        运行 Agent 的主循环。
        """
        print(f"\n> 用户: {user_query}")
        self.messages.append({"role": "user", "content": user_query})

        # 循环直到模型给出最终答案而不是要求调用工具
        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
                tool_choice="auto",
            )
            response_message = response.choices[0].message
            
            # 检查模型是否要求调用工具
            if not response_message.tool_calls:
                # 模型没有要求调用工具，直接返回其回答
                final_answer = response_message.content
                self.messages.append({"role": "assistant", "content": final_answer})
                print(f"< Agent: {final_answer}")
                break
            
            # 模型要求调用工具
            self.messages.append(response_message)
            
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_to_call = self.available_functions[function_name]
                
                try:
                    function_args = json.loads(tool_call.function.arguments)
                    function_response = function_to_call(**function_args)
                except json.JSONDecodeError:
                    function_response = "错误：模型生成的参数不是有效的 JSON。"
                except Exception as e:
                    function_response = f"调用函数时发生错误: {e}"

                # 将工具的执行结果返回给模型
                self.messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )
        
        return self.messages[-1]["content"]

# --- 3. 交互式主程序 ---

def main():
    # 检查 API Key 是否设置
    if not os.getenv("OPENAI_API_KEY"):
        print("错误：请先设置环境变量 OPENAI_API_KEY。")
        return

    print("科学计算 Agent 已启动。输入 'exit' 来结束对话。")
    agent = ScientificAgent()
    
    while True:
        try:
            user_input = input("你有什么数学问题？\n> ")
            if user_input.lower() in ["exit"]:
                print("感谢使用，再见！")
                break
            agent.run(user_input)
        except (KeyboardInterrupt, EOFError):
            print("\n感谢使用，再见！")
            break

if __name__ == "__main__":
    main()