# Assignment 2

## 学生信息
- 姓名：陈嘉雷
- 学号：2200011418

## 简介
`scientific_agent.py` 是一个功能强大的科学计算 Agent，支持以下功能：
- 流式输出
- 有限历史记录
- 自动重试
- 多步骤工具调用

该程序适用于需要复杂科学计算的场景，能够通过调用工具箱中的函数完成符号方程求解、符号微分和数值积分等任务。

## 运行方法

**1. 开始对话：**

运行程序后，您就可以在`你: `提示符后输入问题或对话内容，然后按 Enter。

**2. 接收回复:**

模型会以流式的方式逐步输出回复。如果调用工具，将会按如下格式显示工具调用：

```
> 调用工具: perform_symbolic_differentiation({"expression": "x**3 + sin(x)"})
< 工具返回: The derivative of the expression is: 3*x**2 + cos(x)
```

**3. 退出程序：**

在任何时候，当您想结束对话时，只需输入 `exit` 或者 `quit` 并按 Enter。
```bash
你: exit
已退出对话。
```

## 主要功能

### 工具箱功能
1. **符号方程求解**
   - 函数：`solve_symbolic_equation`
   - 示例：输入 `x**2 - 4`，返回 `The solution to the equation is: [-2, 2]`

2. **符号微分**
   - 函数：`perform_symbolic_differentiation`
   - 示例：输入 `x**3 + sin(x)`，返回 `The derivative of the expression is: 3*x**2 + cos(x)`

3. **数值积分**
   - 函数：`calculate_numerical_integral`
   - 示例：输入 `lambda x: np.exp(-x**2)`，区间为 `-inf` 到 `inf`，返回 `The integral result is approximately: 1.7724538509055159 with an error of 1.4202636780944923e-08`
