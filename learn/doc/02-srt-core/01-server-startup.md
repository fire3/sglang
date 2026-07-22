# 服务器启动流程

> 参考文件：`python/sglang/launch_server.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/server_args_config_parser.py`, `python/sglang/srt/entrypoints/engine.py`

## 启动入口

服务器的启动起点是 `python/sglang/launch_server.py`。当执行以下命令时：

```bash
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --port 30000
```

`launch_server.py` 中的 `launch_server()` 函数被调用，其核心流程如下：

```
launch_server()
    │
    ├── 1. 解析 ServerArgs
    │       └── server_args_config_parser.py → 从命令行/环境变量读取配置
    │
    ├── 2. 初始化 TokenizerManager
    │       └── managers/tokenizer_manager.py → 加载 tokenizer
    │
    ├── 3. 初始化 Scheduler
    │       └── managers/scheduler.py → 创建调度器
    │
    ├── 4. 初始化 TP Worker
    │       └── managers/tp_worker.py → 加载模型到 GPU
    │
    ├── 5. 启动 HTTP 服务器
    │       └── entrypoints/http_server.py → FastAPI 应用
    │
    └── 6. 等待请求
```

## ServerArgs — 参数配置

文件 `python/sglang/srt/server_args.py` 是项目最大的单个文件（约 380KB），因为它集中定义了**所有**服务端配置参数。参数按类别分组：

### 参数分组（ArgGroups）

`python/sglang/srt/arg_groups/` 目录将参数分组管理：

| 分组 | 用途 |
|---|---|
| 模型参数 | `--model-path`, `--model-type`, `--tokenizer-path` 等 |
| 运行时参数 | `--tp-size`, `--dp-size`, `--ep-size` 等并行度配置 |
| 内存参数 | `--mem-fraction`, `--max-num-tokens` 等 |
| 服务器参数 | `--port`, `--host`, `--ssl` 等网络配置 |
| 日志参数 | `--log-level`, `--log-file` 等 |
| 高级参数 | `--enable-mixed-penalty`, `--enable-torch-compile` 等 |

### 环境变量补充

`python/sglang/srt/environ.py` 定义了环境变量化的配置。有些配置只能通过环境变量设置，不能通过命令行参数传入。这是为了区分"启动时固定"和"运行时可变"的配置。

## 初始化顺序详解

### 1. 参数解析阶段

```python
# launch_server.py 中
server_args = ServerArgs.from_cli_args()  # 解析命令行参数
# 或
server_args = ServerArgs.from_dict()  # 从字典创建（API 方式启动时）
```

`ServerArgs` 对象会被传递到所有子组件，各组件只读取自己需要的参数。

### 2. TokenizerManager 初始化

```python
# 加载 tokenizer
tokenizer_manager = TokenizerManager(server_args)
```

这个阶段：
- 使用 HuggingFace `transformers` 加载 tokenizer
- 初始化聊天模板处理
- 准备特殊 token 的映射

### 3. Scheduler 初始化

```python
scheduler = Scheduler(server_args, tokenizer_manager)
```

这个阶段：
- 创建 Radix Cache（基于 server_args 的缓存大小设置）
- 配置调度策略
- 设置批次参数（最大批次大小、最大 tokens 数等）

### 4. TP Worker 初始化

```python
tp_worker = TpWorker(server_args)
```

这个阶段：
- 初始化分布式通信（NCCL）
- 加载模型权重到 GPU
- 预热 CUDA Graph（如果启用）
- 创建 KV Cache 池
- 编译模型（如果启用 torch.compile）

### 5. HTTP 服务器启动

```python
# 创建 FastAPI 应用
app = create_app(server_args, tokenizer_manager, scheduler, tp_worker)
```

HTTP 服务器基于 **FastAPI** 构建，使用 **uvicorn** 作为 ASGI 服务器。

### 6. 等待请求

服务器进入事件循环，等待 HTTP/gRPC 请求到达。

## gRPC 模式

除了 HTTP，SGLang 也支持 gRPC 协议。在 gRPC 模式下：
- 启动流程类似，但使用 `grpc_server.py` 替代 HTTP 服务器
- 通信通过 Protobuf 定义的协议完成
- 适用于微服务架构和高吞吐场景

## 关键观察

1. **顺序初始化**：各组件按依赖关系顺序初始化，没有循环依赖
2. **参数集中管理**：全部参数在 `server_args.py` 中统一管理，各组件只读
3. **模型加载是瓶颈**：TP Worker 的初始化（模型加载到 GPU）是最耗时的阶段，大型模型可能需要几十秒到几分钟
4. **支持热更新**：某些配置（如 LoRA 权重）可以在运行时更新，无需重启
