# API 入口

> 参考文件：`python/sglang/srt/entrypoints/http_server.py`, `python/sglang/srt/entrypoints/http_server_engine.py`, `python/sglang/srt/entrypoints/grpc_server.py`, `python/sglang/srt/entrypoints/openai/`

## HTTP 服务器

### 核心实现

`python/sglang/srt/entrypoints/http_server.py` 使用 **FastAPI** 框架构建 HTTP 服务：

```python
# 核心结构
app = FastAPI()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    result = await engine.generate_request(request)
    return result

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    result = await engine.generate_request(request)
    return result
```

`http_server_engine.py` 定义了一个特殊的远程代理 `HttpServerEngineAdapter(EngineBase)`，**非标准路径使用**。它将整个服务器启动为独立子进程，将所有 Engine 方法调用（`generate()`、`update_weights_from_tensor()` 等）转发为 HTTP POST 请求到子进程端点。它仅在需要 HTTP 级访问的场景（如 `VerlEngine`）中使用，标准 `launch_server` 路径不经过它。

⚠️ 标准路径中，真正的 HTTP ↔ Engine 桥接层是 **`TokenizerManager`**（见下文全链路分析）。

### 路由注册

在 `create_app()` 函数中注册所有路由：

```python
def create_app(engine, server_args):
    app = FastAPI()
    
    # OpenAI 兼容路由
    app.include_router(chat_router, prefix="/v1")
    app.include_router(completion_router, prefix="/v1")
    app.include_router(embedding_router, prefix="/v1")
    app.include_router(rerank_router, prefix="/v1")
    
    # 健康检查
    app.add_route("/health", health_check)
    
    # 管理接口
    app.add_route("/flush_cache", flush_cache)
    app.add_route("/get_model_info", get_model_info)
    
    return app
```

## OpenAI 兼容 API

`python/sglang/srt/entrypoints/openai/` 目录实现了完整的 OpenAI 兼容 API：

```
openai/
├── protocol.py                 ← 请求/响应数据结构定义
├── serving_chat.py             ← /v1/chat/completions
├── serving_completions.py      ← /v1/completions
├── serving_embedding.py        ← /v1/embeddings
├── serving_rerank.py           ← /v1/rerank
├── serving_score.py            ← /v1/score
├── serving_tokenize.py         ← /v1/tokenize
├── serving_responses.py        ← Responses API
├── serving_base.py             ← 基类
├── serving_classify.py         ← 分类
├── serving_transcription.py    ← 语音转录
├── chat_encoding.py            ← Chat 编码格式
├── sse_utils.py                ← SSE 流式工具
├── tool_server.py              ← 工具调用服务器
└── realtime/                   ← 实时 API（WebSocket）
```

### 核心协议（protocol.py）

`protocol.py` 定义了所有 API 的请求和响应数据结构，使用 Pydantic 模型：

```python
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    temperature: float = 1.0
    max_tokens: int = 4096
    stream: bool = False
    ...

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    choices: List[Choice]
    usage: UsageInfo
    ...
```

### Chat Completions 处理流程

```python
# serving_chat.py 中的简化逻辑
async def handle_chat_completion(request):
    # 1. 转换消息格式
    prompt = chat_template.apply_chat_template(request.messages)
    
    # 2. 构造推理请求
    gen_request = GenerationRequest(
        prompt=prompt,
        sampling_params=SamplingParams.from_request(request),
    )
    
    # 3. 提交到引擎
    if request.stream:
        return stream_generator(engine, gen_request)
    else:
        result = await engine.generate_request(gen_request)
        return format_response(result)
```

## Anthropic 兼容 API

`python/sglang/srt/entrypoints/anthropic/` 目录实现了 Anthropic Messages API 兼容层：

```
anthropic/
├── __init__.py        ← 包标记
├── protocol.py        ← Anthropic 协议数据结构（Pydantic 模型）
└── serving.py         ← Anthropic ↔ OpenAI 格式转换与处理
```

### 路由注册

Anthropic 路由直接注册在 `http_server.py` 中，与 OpenAI 路由同一层级：

```python
# http_server.py 末尾（约 L1929–1946）
@app.post("/v1/messages")
async def anthropic_v1_messages(request: AnthropicMessagesRequest):
    return await fast_api_app.state.anthropic_serving.handle_messages(request)

@app.post("/v1/messages/count_tokens")
async def anthropic_v1_count_tokens(request: AnthropicCountTokensRequest):
    return await fast_api_app.state.anthropic_serving.handle_count_tokens(request)
```

`AnthropicServing` 单例在服务器启动时初始化（`http_server.py` 约 L332–334），存储在 `fast_api_app.state.anthropic_serving` 中，依赖已初始化的 `OpenAIServingChat` 实例。

### 协议模型（protocol.py）

定义了 Anthropic Messages API 的完整请求/响应结构，包含 **内容块（Content Block）** 的判别联合体：

```python
# 内容块类型（约 L111–123）
ContentBlock = Annotated[
    Union[
        TextBlock,           # 文本
        ImageContentBlock,   # 图片
        ToolUseBlock,        # 工具调用
        ToolResultBlock,     # 工具结果
        ThinkingBlock,       # 思考过程（Claude 3.7+）
        RedactedThinkingBlock,
        ...
    ],
    Field(discriminator="type"),
]

# 工具类型（约 L223–232）
AnthropicTool = Annotated[
    Union[
        CustomTool,          # 自定义工具
        WebSearchTool,       # 联网搜索
        ComputerTool,        # 电脑操作
        BashTool,            # Bash 工具
        TextEditorTool,      # 文本编辑工具
    ],
    Field(discriminator="type"),
]
```

### 请求转换流程（serving.py）

`AnthropicServing._convert_to_chat_completion_request()`（约 L229–708）是整个转换层的核心，将 Anthropic 请求映射为 OpenAI `ChatCompletionRequest`，再复用 OpenAI 的处理管线：

```
Anthropic 请求                           OpenAI 中间表示
─────────────────                       ──────────────────────
System message     ──→  system role
TextBlock          ──→  text content part
ImageContentBlock  ──→  image_url part
ToolUseBlock       ──→  tool_calls
ToolResultBlock    ──→  tool role + content
ThinkingBlock      ──→  reasoning_history（SGLang 扩展）
Anthropic Tool     ──→  OpenAI Tool（服务端工具如 web_search 被剥离并告警）
tool_choice        ──→  tool_choice
temperature/top_p  ──→  直接映射
stop_sequences     ──→  stop
thinking           ──→  reasoning_config
```

转换后调用 `OpenAIServingChat._convert_to_internal_request()`，最终生成 `GenerationRequest` 进入调度器。

### 非流式响应

```python
# 简化流程
async def _handle_non_streaming(request):
    # 1. Anthropic → OpenAI 格式
    chat_req = _convert_to_chat_completion_request(request)

    # 2. 委托 OpenAI serving 处理
    openai_response = await openai_serving_chat._handle_non_request(chat_req)

    # 3. OpenAI → Anthropic 格式
    return _convert_response(openai_response)
```

`_convert_response()`（约 L1234–1310）的反向映射：
| OpenAI 字段 | Anthropic 字段 |
|---|---|
| `reasoning_content` | `ThinkingBlock`（`type: thinking`） |
| `content` | `TextBlock`（`type: text`） |
| `tool_calls` | `ToolUseBlock`（`type: tool_use`） |
| `finish_reason` | `stop_reason`（通过 `STOP_REASON_MAP` 映射） |

### 流式响应

`_handle_streaming()` 返回 `StreamingResponse`，其生成器 `_generate_anthropic_stream()`（约 L813–1310）包装 OpenAI 流，逐帧翻译为 Anthropic SSE 事件格式：

```
event: message_start
data: {"type": "message_start", "message": {"role": "assistant", "content": [], ...}}

event: content_block_start
data: {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", ...}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "..."}}

event: content_block_stop
data: {"type": "content_block_stop", "index": 0}

event: message_delta
data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {...}}

event: message_stop
data: {"type": "message_stop"}
```

关键实现细节：
- `content_block_start` 被延迟：直到第一个内容块或 usage 数据到达时才发送，确保 `message_start` 包含完整骨架
- 每 5 秒发送一次 `ping` 事件保持连接
- Thinking 流包含 `SignatureDelta` 事件，用于传输 Claude 思考签名

### 错误处理

`http_server.py` 为 Anthropic 路径注册了独立的异常处理器：

```python
# HTTPException 处理器（约 L484–517）
if request.url.path.startswith("/v1/messages"):
    return JSONResponse(
        {"type": "error", "error": {"type": "...", "message": "..."}},
        status_code=status_code,
    )

# RequestValidationError 处理器（约 L541–555）同理
```

OpenAI 格式的错误和 Anthropic 格式的错误在此分叉，各自返回兼容的 error envelope。

## 全链路请求生命周期

以上 API 层（OpenAI、Anthropic 等）最终都汇入同一个底层管线。理解这条管线需要先了解 SGLang 的**多进程架构**。

### 进程架构

SGLang 的 SRT 推理运行时采用**多进程隔离**设计：

```
┌─────────────────────────────────────────────────────────┐
│                    主进程（Main Process）                  │
│                                                         │
│  FastAPI 服务器（http_server.py）                          │
│    ├── OpenAI / Anthropic / Ollama Serving 层             │
│    └── TokenizerManager（主进程调度中枢）                   │
│          ├── ZMQ DEALER ──→ Scheduler 子进程              │
│          └── ZMQ DEALER ──→ DetokenizerManager 子进程      │
└─────────────────────────────────────────────────────────┘
         │                              │
         │ ZMQ IPC                      │ ZMQ IPC
         ▼                              ▼
┌────────────────────┐     ┌──────────────────────────┐
│ Scheduler 子进程     │     │ DetokenizerManager 子进程  │
│ ┌──────────────┐   │     │                          │
│ │ TpWorkerGroup │   │     │                          │
│ │    └─ Model   │   │     │  输出 token → 文本解码    │
│ └──────────────┘   │     │                          │
│  调度 + 前向推理    │     │                          │
└────────────────────┘     └──────────────────────────┘
```

**关键设计**：`Engine._launch_subprocesses()` 启动调度器和反分词器为独立子进程，在主进程返回 `TokenizerManager` 实例，这就是 API 层见到的"引擎"。

### 服务初始化链路

```
launch_server()                           http_server.py:2638
  │
  ├── Engine._launch_subprocesses()        engine.py:765
  │     ├── run_scheduler_process()        scheduler.py:4593  [子进程]
  │     ├── run_detokenizer_process()      [子进程]
  │     └── returns tokenizer_manager      [主进程]
  │
  ├── 将 tokenizer_manager 注入 Serving 类
  │     fast_api_app.state.tokenizer_manager = tokenizer_manager
  │     fast_api_app.state.openai_serving_chat =
  │         OpenAIServingChat(tokenizer_manager, template_manager)
  │
  └── FastAPI lifespan 启动 → 开始接收请求
```

### 完整请求生命周期（以 `/v1/chat/completions` 为例）

```
┌─ HTTP 层 ─────────────────────────────────────────────────────┐
│                                                                │
│ POST /v1/chat/completions                                      │
│   → http_server.py: openai_v1_chat_completions()               │
│   → app.state.openai_serving_chat.handle_request(req)          │
│     (serving_chat.py，继承自 OpenAIServingBase)                  │
│                                                                │
│   ① Parse ChatCompletionRequest（Pydantic 校验）                │
│   ② 应用 Jinja2 chat template → 生成纯文本 prompt               │
│   ③ 构造 GenerateReqInput(text=..., sampling_params=...)       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
         │
         ▼  await tokenizer_manager.generate_request(obj)
         │
┌─ TokenizerManager 层（主进程） ─────────────────────────────────┐
│                                                                │
│   ④ auto_create_handle_loop()   确保事件循环运行                 │
│   ⑤ obj.normalize_batch_and_args()  批量参数标准化              │
│   ⑥ _tokenize_one_request()    tokenizer_manager.py:828        │
│        ├── HuggingFace tokenizer → input_ids                   │
│        └── 多模态输入（图片/音频） → mm_processor 处理            │
│   ⑦ _send_one_request()    tokenizer_manager.py:1367           │
│        ├── wrap_shm_features()  共享内存序列化                   │
│        └── _dispatch_to_scheduler()  通过 ZMQ 发送              │
│              send_to_scheduler.send_multipart(bytes)            │
│                                                                │
│   ⑧ _wait_one_response()    tokenizer_manager.py:1482          │
│        ├── 等待 per-request asyncio.Event                       │
│        └── async for response:  yield {"text": ...,             │
│                                        "meta_info": {...}}      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
         │
         ▼   ZMQ IPC（进程间通信）
         │
┌─ Scheduler 子进程 ─────────────────────────────────────────────┐
│                                                                │
│   ⑨ 接收 TokenizedGenerateReqInput                              │
│   ⑩ 加入调度队列（waiting_queue）                               │
│   ⑪ 调度循环（event_loop_normal）：                              │
│        ├── 选择可运行的请求组成 batch                             │
│        ├── prepare_generation()  准备推理输入                    │
│        ├── TpWorkerGroup.forward()  GPU 前向推理                 │
│        ├── 采样生成下一个 token                                  │
│        ├── 检查 stop 条件（max_tokens / eos / stop strings）      │
│        └── 输出 token 发送至 DetokenizerManager                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
         │
         ▼   ZMQ IPC
         │
┌─ DetokenizerManager 子进程 ────────────────────────────────────┐
│                                                                │
│   ⑫ 接收增量 token                                             │
│   ⑬ 解码为文本（逐 token 或按 chunk）                            │
│   ⑭ 转发回 TokenizerManager                                    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
         │
         ▼   ZMQ IPC → asyncio.Event → async generator
         │
┌─ 响应格式层 ──────────────────────────────────────────────────┐
│                                                                │
│   ⑮ 收到 TokenizerManager yield 内容                           │
│   ⑯ 流式场景：逐帧格式化为 SSE delta，写入 StreamingResponse    │
│   ⑰ 非流式场景：聚合完整内容，格式化为 ChatCompletionResponse    │
│   ⑱ ORJSONResponse / StreamingResponse → HTTP 响应             │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 关键桥接层：TokenizerManager

`TokenizerManager` 是主进程中的核心中枢，同时面向 API 层和子进程层：

| 面向 | 接口 | 机制 |
|---|---|---|
| API 层（Serving 类） | `generate_request()` / `generate_request_stream()` | async Python 方法 |
| 子进程（Scheduler） | `_dispatch_to_scheduler()` | ZMQ DEALER 套接字 |
| 子进程（Detokenizer） | `_handle_detokenizer_response()` | ZMQ DEALER 套接字 |
| 异步等待 | `_wait_one_response()` | per-request `asyncio.Event` |

这种设计使得 API 层无需关心多进程细节——Serving 类只需对 `tokenizer_manager` 调用 `generate_request()`，即可获得一个异步生成器，背后是 ZMQ IPC、调度循环、GPU 推理的完整链路。

### 请求处理路径总结

综合以上分析，请求处理路径可按**层**归纳：

| 层 | 组件 | 职责 |
|---|---|---|
| HTTP 路由 | `http_server.py` + FastAPI | 路由匹配、异常分发 |
| 协议适配 | `serving_chat.py` / `serving.py`(Anthropic) | 外部协议 ↔ 内部 `GenerateReqInput` 转换 |
| Tokenization | `TokenizerManager._tokenize_one_request()` | 文本 → `input_ids` |
| IPC 传输 | ZMQ DEALER 套接字 | 进程间请求/响应传输 |
| 调度执行 | Scheduler + TpWorkerGroup | 批调度、GPU 推理、token 生成 |
| Detokenization | DetokenizerManager | `output_ids` → 文本 |
| 响应组装 | Serving 类 | 内部结果 → OpenAI/Anthropic 格式响应 |

## gRPC 服务器

`python/sglang/srt/entrypoints/grpc_server.py` 提供 gRPC 接口：

- 使用 Protobuf 定义的服务接口（`proto/sglang/`）
- 适用于高性能、低延迟的场景
- 支持双向流式通信

`grpc_bridge.py` 是 gRPC 和 HTTP 之间的桥接层，允许两种协议共享相同的处理逻辑。

## 其他入口

| 入口 | 文件 | 用途 |
|---|---|---|
| Anthropic API | `entrypoints/anthropic/` | Anthropic 兼容 API |
| Ollama API | `entrypoints/ollama/` | Ollama 兼容 API |
| Elastic EP | `entrypoints/elastic_ep.py` | 弹性专家并行入口 |
| Engine Info | `entrypoints/engine_info_bootstrap_server.py` | 引擎信息启动引导 |
| Warmup | `entrypoints/warmup.py` | 启动预热 |

## 快速参考

完整的全链路分析见上文**全链路请求生命周期**章节。简化的数据流：

```
HTTP 请求 → FastAPI 路由 → Serving 协议层 → TokenizerManager
    → ZMQ IPC → Scheduler（调度 + GPU 推理）
    → ZMQ IPC → DetokenizerManager（解码）
    → ZMQ IPC → TokenizerManager → Serving 格式层 → HTTP 响应

