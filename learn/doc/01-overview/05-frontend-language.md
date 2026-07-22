# 前端语言（sglang.lang）

> 参考文件：`python/sglang/lang/`

SGLang 提供了一种**前端编程语言**，用于表达 LLM 推理请求的结构。这套 DSL（领域特定语言）让用户能以声明式的方式描述复杂的多轮对话、分支逻辑和结构化输出。

## 目录结构

```
python/sglang/lang/
├── api.py              ← 用户 API（@sglang.function 等）
├── ir.py               ← 中间表示（Function, Expression, Node）
├── interpreter.py      ← 解释器（执行 IR）
├── tracer.py           ← 追踪器（记录执行轨迹）
├── chat_template.py    ← 聊天模板
├── choices.py          ← 选择结构
└── backend/            ← 后端抽象
```

## 核心概念

### 1. API 层（api.py）

这是用户直接接触的接口。核心装饰器 `@sglang.function` 将一个 Python 函数标记为 SGLang 可执行函数：

```python
@sglang.function
def chat(system_msg, user_msg):
    sglang.system(system_msg)
    sglang.user(user_msg)
    sglang.assistant(sglang.gen("response"))
```

关键 API 函数：
- `sglang.gen(name, ...)` — 生成文本，绑定到 `name` 变量
- `sglang.system(msg)` — System prompt
- `sglang.user(msg)` — User 消息
- `sglang.assistant(msg)` — Assistant 消息
- `sglang.image(path)` — 图像输入
- `sglang.video(path)` — 视频输入
- `sglang.audio(path)` — 音频输入
- `sglang.bind(variables)` — 绑定变量值
- `sglang.set_default(...)` — 设置默认参数

### 2. 中间表示（ir.py）

IR 层定义了 SGLang 程序的结构化表示：

- **`Function`** — 一个 SGLang 函数的定义
- **`Expression`** — 表达式（如 `gen("response")`）
- **`Node`** — 执行图中的节点

`Function` 对象包含了函数的完整执行计划，可以被序列化和传输。

### 3. 追踪器（tracer.py）

追踪器负责将 Python 函数的执行过程记录下来，生成 `Function` 对象：

```python
tracer = Tracer()
function = tracer.trace(chat, {"system_msg": "...", "user_msg": "..."})
```

追踪机制类似于 `torch.fx` 或 `jax` 的追踪——它不会实际执行模型推理，而是记录调用了哪些 API 函数以及它们之间的依赖关系。

### 4. 解释器（interpreter.py）

解释器读取 `Function` 对象的 IR，逐条执行图中的操作：

```python
interpreter = Interpreter(function)
result = interpreter.run(backend=backend)
```

执行过程中：
1. 遇到 `gen()` 调用时，调用后端子执行模型推理
2. 处理变量绑定和返回值传递
3. 管理对话上下文

### 5. 后端子（backend/）

后端子接口定义了 SGLang 程序如何与实际的推理引擎交互：

- `backend/` 中的抽象类定义了 `generate()` 等方法
- 可以对接不同的后端（SGLang Runtime、模拟后端等）

## 前端语言与后端的关系

```
用户代码 (Python)
    │
    ▼
@sglang.function   ← api.py
    │
    ▼
Tracer.trace()     ← tracer.py → 生成 IR (Function)
    │
    ▼
Interpreter.run()  ← interpreter.py
    │
    ▼
Backend.generate() ← backend/ → SGLang Runtime
```

## 前端语言的设计意图

1. **声明式而非命令式**：用户描述"要生成什么"，而不是"怎么做"
2. **可批处理**：IR 格式使得运行时可以优化执行顺序和批处理
3. **可流式传输**：`Function` 可以被序列化，通过网络传输到远程后端
4. **结构灵活**：支持条件分支、多轮对话、并行生成等复杂场景

## 与后端 Runtime 的分界

- **前端语言**（`lang/`）负责：程序的结构化表达、执行追踪、参数管理
- **后端 Runtime**（`srt/`）负责：实际的模型加载、推理计算、内存管理

前端语言将推理请求"编译"为中间表示，交给后端执行。这种分离使得：
- 前端可以独立迭代，支持新的编程范式
- 后端可以专注于性能优化，无需关心 API 设计
