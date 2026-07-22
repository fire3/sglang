# Scheduler 子组件

> 参考文件：`python/sglang/srt/managers/scheduler_components/`

## 概述

`scheduler_components/` 目录包含 Scheduler 运行时所需的**辅助组件**。每个组件负责一个独立的关注点，共同支撑调度器的运行。

## 组件一览

```
scheduler_components/
├── request_receiver.py       ← 请求接收
├── output_sender.py          ← 输出发送
├── output_streamer.py        ← 流式输出
├── batch_result_processor.py ← 批处理结果处理
├── flush_wrapper.py          ← 刷新包装
├── ipc_channels.py           ← 进程间通信通道
├── metrics_reporter.py       ← 指标上报
├── profiler_manager.py       ← Profiling 管理
├── weight_updater.py         ← 权重更新
├── dp_attn.py                ← 数据并行注意力
├── kv_events_publisher.py    ← KV 缓存事件发布
├── new_token_ratio_tracker.py ← 新 token 比例追踪
├── pool_stats_observer.py    ← 内存池统计观察
├── recv_skipper.py           ← 接收跳过（用于 PD 分离）
├── load_inquirer.py          ← 负载查询
├── idle_sleeper.py           ← 空闲休眠
├── invariant_checker.py      ← 不变量检查器
├── logprob_result_processor.py ← Logprob 结果处理
└── flush_wrapper.py          ← 刷新包装
```

## 核心组件详解

### request_receiver.py — 请求接收

负责从 IPC 通道接收新的推理请求：

```python
class RequestReceiver:
    def recv_requests(self):
        """接收一批新请求"""
        # 从 IPC 读取
        # 反序列化
        # 返回 Request 列表
        ...
```

### output_sender.py — 输出发送

将推理结果发送回 TokenizerManager：

```python
class OutputSender:
    def send_output(self, request_id, tokens, finished):
        """发送推理输出"""
        # 序列化结果
        # 写入 IPC
        ...
```

### output_streamer.py — 流式输出

处理流式（SSE）输出，逐 token 发送结果：

```python
class OutputStreamer:
    def stream_output(self, request_id, token):
        """逐 token 流式输出"""
        ...
```

### batch_result_processor.py — 批处理结果处理

处理 TP Worker 返回的推理结果：

```python
class BatchResultProcessor:
    def process(self, batch, model_output):
        """处理模型输出，更新批次状态"""
        # 1. 提取 logits
        # 2. 采样（选择下一个 token）
        # 3. 更新请求状态
        # 4. 检查是否完成
        ...
```

### weight_updater.py — 权重更新

支持在线更新模型权重（如 LoRA 适配器）：

```python
class WeightUpdater:
    def update_weights(self, weight_path):
        """在线更新权重"""
        ...
```

### ipc_channels.py — 进程间通信通道

管理 Scheduler 与其他进程（TokenizerManager、TP Worker）之间的通信管道：

```python
class IPCChannels:
    def __init__(self):
        # 输入通道（来自 TokenizerManager）
        self.input_channel = ...
        # 输出通道（到 DetokenizerManager）
        self.output_channel = ...
        # 控制通道
        self.control_channel = ...
```

### metrics_reporter.py — 指标上报

收集和上报运行时指标：

```python
class MetricsReporter:
    def report_throughput(self, tokens_per_second):
        ...
    def report_batch_size(self, size):
        ...
    def report_memory_usage(self, used, total):
        ...
```

### profiler_manager.py — Profiling 管理

控制性能分析的启停和数据收集。

### kv_events_publisher.py — KV 缓存事件

在 PD 分离模式下，发布 KV 缓存传输相关的事件。

## 组件之间的协作

```
请求进入 → RequestReceiver
    │
    ▼
Scheduler._schedule()  ← 使用各种组件的信息做决策
    │
    ├── IPCChannels (通信)
    ├── PoolStatsObserver (内存状态)
    ├── NewTokenRatioTracker (比例追踪)
    └── PrefillDelayer (延时决策)
    │
    ▼
Scheduler._run_batch()
    │
    ├── → TP Worker 推理
    │
    ▼
BatchResultProcessor.process()
    │
    ├── OutputSender.send() → 发送结果
    ├── OutputStreamer.stream() → 流式输出
    └── MetricsReporter.report() → 指标上报
```

## 设计模式

这些组件采用**策略模式**和**观察者模式**的组合：

- **策略模式**：不同组件封装不同的算法/行为（如调度策略、权重更新策略）
- **观察者模式**：组件订阅事件（如 `kv_events_publisher` 发布的事件）
- **混入（Mixin）**：通过 Python 的多重继承将功能混入 Scheduler 类
