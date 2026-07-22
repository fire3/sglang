# 阶段二：服务端运行时（SRT）核心

> 目标：深入理解 SGLang Runtime 的核心架构——服务器启动流程、请求处理管道、任务调度与 Tokenization 管理。

## 目录

1. [服务器启动流程](01-server-startup.md)
2. [引擎核心（Engine）](02-engine-core.md)
3. [API 入口](03-api-entrypoints.md)
4. [Tokenizer Manager](04-tokenizer-manager.md)
5. [调度器（Scheduler）](05-scheduler.md)
6. [Scheduler 子组件](06-scheduler-components.md)
7. [TP Worker（张量并行工作者）](07-tp-worker.md)
8. [Detokenizer Manager](08-detokenizer.md)
9. [运行时上下文（RuntimeContext）](09-runtime-context.md)
10. [可观测性（Observability）](10-observability.md)

## 请求处理管道概览

```
HTTP/gRPC 请求
    │
    ▼
┌──────────────────────┐
│   TokenizerManager   │  ← 1. 分词处理
│   (tokenizer_manager.py)
└──────────┬───────────┘
           │  token IDs + 元数据
           ▼
┌──────────────────────┐
│   Scheduler          │  ← 2. 调度决策
│   (scheduler.py)     │     - 批处理组织
│                      │     - Radix Cache 匹配
│                      │     - 内存分配
└──────────┬───────────┘
           │  推理批次
           ▼
┌──────────────────────┐
│   TP Worker          │  ← 3. 模型推理
│   (tp_worker.py)     │     - 前向传播
│                      │     - 采样
└──────────┬───────────┘
           │  logits / tokens
           ▼
┌──────────────────────┐
│   DetokenizerManager │  ← 4. 去分词
│   (detokenizer_manager.py)
└──────────┬───────────┘
           │  文本
           ▼
┌──────────────────────┐
│   HTTP/gRPC 响应     │
└──────────────────────┘
```

这个管道是同步的（每个阶段等待上一阶段完成），但 SGLang 通过**异步批处理**和**流水线重叠**来隐藏延迟。
