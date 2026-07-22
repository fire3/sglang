# 调度器（Scheduler）

> 参考文件：`python/sglang/srt/managers/scheduler.py`, `python/sglang/srt/managers/schedule_policy.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/managers/scheduler_pp_mixin.py`, `python/sglang/srt/managers/prefill_delayer.py`

## 角色与职责

`Scheduler` 是 SGLang 推理管道的**核心调度者**，负责：

1. **批次组织**：将待处理的请求组织为 GPU 推理批次
2. **内存管理**：分配和管理 KV Cache 内存
3. **调度决策**：决定哪些请求应该被 prefilled，哪些应该继续 decode
4. **Radix Cache 管理**：匹配共享前缀，复用 KV Cache
5. **优先级管理**：处理不同优先级请求的调度

## 核心文件分析

### scheduler.py — 主调度器

```python
class Scheduler:
    def __init__(self, server_args, tokenizer_manager):
        # 调度策略
        self.policy = SchedulePolicy(server_args)
        # 等待队列
        self.waiting_queue: List[Request] = []
        # 运行批次
        self.running_batch: Optional[Batch] = None
        # Radix Cache
        self.radix_cache = RadixCache(server_args)
        # 内存池
        self.memory_pool = MemoryPool(server_args)
    
    def add_request(self, req: Request):
        """添加请求到等待队列"""
        self.waiting_queue.append(req)
    
    def event_loop(self):
        """主调度循环（持续运行）"""
        while True:
            # 1. 收集新请求
            self._get_new_requests()
            
            # 2. 调度决策
            batch = self._schedule()
            
            # 3. 执行推理
            if batch:
                self._run_batch(batch)
            
            # 4. 处理结果
            self._process_results()
    
    def _schedule(self):
        """核心调度逻辑"""
        # 1. 确定优先级
        # 2. 匹配 Radix Cache
        # 3. 分配内存
        # 4. 构建批次
        ...
```

### 调度循环详解

调度器的核心是一个持续运行的事件循环：

```
1. _get_new_requests()
   │  从 tokenizer_manager 获取新的 tokenized 请求
   │  加入 waiting_queue
   ▼
2. _schedule()
   │  从 waiting_queue 中选取合适的请求组成批次
   │  匹配 Radix Cache（前缀复用）
   │  分配 KV Cache 内存
   ▼
3. _run_batch(batch)
   │  将批次发送到 TP Worker 执行
   │  等待 GPU 计算完成
   ▼
4. _process_results()
   │  收集推理结果
   │  决定是否需要继续 decode
   │  将已完成请求的结果发送回 TokenizerManager
   ▼
（循环）
```

### 调度策略（schedule_policy.py）

`SchedulePolicy` 定义了调度决策的具体逻辑：

```python
class SchedulePolicy:
    def decide_batch(self, waiting_queue, running_batch):
        """
        做出批处理决策：
        - 哪些请求应该被 prefilled
        - 哪些正在 decode 的请求应该继续
        - 哪些请求应该被抢占（preempted）
        """
        ...
```

调度策略考虑的因素：
- **FCFS（先来先服务）**：基础策略
- **内存足够性**：只有当前有足够 KV Cache 内存的请求才会被调度
- **前缀匹配收益**：共享前缀的请求优先一起调度
- **优先级**：高优先级请求可以插队

### 批次管理（schedule_batch.py）

`Batch` 类代表一个推理批次：

```python
class Batch:
    def __init__(self):
        self.reqs: List[Request] = []
        self.input_ids: torch.Tensor = None
        self.token_ids: List[List[int]] = []
        self.prefix_lens: List[int] = []
        self.extend_lens: List[int] = []
        self.positions: torch.Tensor = None
        ...
```

批次中包含所有需要发送到 GPU 的数据，经过精心组织以最大化计算效率。

### Pipeline Parallel 混入（scheduler_pp_mixin.py）

当启用流水线并行（PP）时，`SchedulerPPMixin` 提供：

```python
class SchedulerPPMixin:
    def _split_batch_for_pp(self, batch):
        """将批次拆分为 PP 微批次"""
        ...
    
    def _schedule_pp_stages(self):
        """调度 PP 各阶段"""
        ...
```

PP 模式下，一个推理请求会经过多个 GPU 阶段，调度器需要协调各阶段的执行。

### Prefill Delayer（prefill_delayer.py）

在某些场景下，延迟 Prefill 可以优化整体吞吐：

```python
class PrefillDelayer:
    def should_delay(self, request):
        """判断是否应该延迟 prefill"""
        ...
```

主要用于：
- **Memory 压力大**时，推迟 prefill 以让 decode 请求先完成
- **批量收益**：等待更多同前缀的请求到达，一起 prefill

## Radix Cache 调度

调度器与 Radix Cache 的协作是 SGLang 性能的关键：

```
请求到达
    │
    ▼
查询 Radix Tree
    │
    ├── 找到最长匹配前缀 → 复用缓存的 KV
    │    └── 只需计算未匹配部分的 KV
    │
    └── 无匹配 → 从头计算
    │
    ▼
分配新 KV Cache 空间
    │
    ▼
更新 Radix Tree
```

## 关键观察

1. **零开销设计**：调度决策不应该阻塞 GPU 计算。调度器在 GPU 推理时准备下一批次
2. **状态机**：每个请求经历 `waiting → running(prefill) → running(decode) → completed` 状态转换
3. **内存感知**：调度器始终跟踪 KV Cache 使用情况，避免 OOM
4. **可抢占**：低优先级请求可以被抢占，让出内存给高优先级请求
