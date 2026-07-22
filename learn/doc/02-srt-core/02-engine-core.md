# 引擎核心（Engine）

> 参考文件：`python/sglang/srt/entrypoints/engine.py`, `python/sglang/srt/entrypoints/EngineBase.py`, `python/sglang/srt/entrypoints/engine_score_mixin.py`

## Engine 的定位

`Engine` 是 SGLang 推理服务的**核心控制类**，协调 TokenizerManager、Scheduler、TP Worker 三大组件。它对外提供统一的推理接口，对内管理组件的生命周期。

## 类层次

```
EngineBase          ← 抽象基类，定义接口
    │
    ├── Engine      ← 主引擎实现（engine.py）
    │
    └── EngineScoreMixin  ← 评分功能混入（engine_score_mixin.py）
```

## EngineBase — 抽象接口

```python
# python/sglang/srt/entrypoints/EngineBase.py

class EngineBase:
    def add_request(self, request) -> None:
        """添加一个推理请求"""
        ...

    def get_request_result(self, request_id) -> RequestResult:
        """获取请求结果（阻塞等待）"""
        ...

    def generate_request(self, request) -> RequestResult:
        """同步生成：add_request + get_request_result"""
        ...

    def generate(self, prompt, sampling_params, ...) -> str:
        """高级接口：接收文本，返回文本"""
        ...

    def flush(self) -> None:
        """刷新所有待处理的请求"""
        ...

    def shutdown(self) -> None:
        """优雅关闭引擎"""
        ...
```

## Engine 的实现

`engine.py` 是 `EngineBase` 的具体实现。核心成员变量：

```python
class Engine(EngineBase):
    def __init__(self, server_args):
        self.tokenizer_manager = TokenizerManager(server_args)
        self.scheduler = Scheduler(server_args, self.tokenizer_manager)
        self.tp_worker = TpWorker(server_args)
```

### 请求处理流程

```python
def add_request(self, request):
    """添加请求到管道"""
    # 1. tokenize
    req = self.tokenizer_manager.tokenize_request(request)
    # 2. 提交到 scheduler
    self.scheduler.add_request(req)

def get_request_result(self, request_id):
    """获取请求结果（阻塞）"""
    result_queue = self.tokenizer_manager.get_result_queue(request_id)
    return result_queue.get()  # 阻塞直到结果就绪
```

### 同步生成接口

```python
def generate(self, prompt, sampling_params=None):
    """高级接口：文本 → 文本"""
    # 1. 构造请求对象
    request = self._create_request(prompt, sampling_params)
    # 2. 添加并等待
    return self.generate_request(request)
```

这个接口隐藏了所有内部细节，对外表现为简单的函数调用。这也是 SGLang Python 前端最终调用的接口。

## Engine Score Mixin

`engine_score_mixin.py` 为引擎添加了**评分（Scoring）**能力：

```python
class EngineScoreMixin:
    def score(self, prompt, completion, ...):
        """计算 prompt + completion 的 logprob/奖励"""
        ...
```

这个功能主要用于：
- RL（强化学习）训练中的奖励计算
- 重排序（reranking）任务
- 评估模型输出质量

## Engine 的生命周期

```
创建 Engine
    │
    ├── init: 初始化所有子组件
    │
    ├── 运行中:
    │   ├── add_request()  ← 从 API 入口接收请求
    │   ├── get_request_result()  → 返回结果给 API 入口
    │   └── [内部] 调度循环持续运行
    │
    ├── flush(): 清空待处理队列
    │
    └── shutdown(): 释放 GPU 内存、关闭连接
```

## Engine 与 API 入口的关系

```
HTTP Server (FastAPI)
    │
    ├── /v1/chat/completions  →  Engine.generate_request()
    ├── /v1/completions       →  Engine.generate_request()
    ├── /v1/embeddings        →  Engine.embed()
    ├── /v1/rerank            →  Engine.score()
    └── /v1/tokenize          →  Engine.tokenize()

gRPC Server
    │
    └── gRPC stubs  →  Engine 的对应方法
```

Engine 是 API 层和内部组件之间的**薄胶水层**——它不做复杂的计算，而是负责任务分发和结果收集。
