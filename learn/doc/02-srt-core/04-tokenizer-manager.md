# Tokenizer Manager

> 参考文件：`python/sglang/srt/managers/tokenizer_manager.py`, `python/sglang/srt/managers/tokenizer_control_mixin.py`, `python/sglang/srt/managers/tokenizer_manager_score_mixin.py`, `python/sglang/srt/managers/multi_tokenizer_mixin.py`, `python/sglang/srt/tokenizer/`

## 角色与职责

`TokenizerManager` 是请求管道中的**第一站**。它负责：

1. **Tokenize**：将文本请求转换为 token ID 序列
2. **结果管理**：为每个请求创建结果队列，收集推理结果
3. **控制消息**：处理特殊的控制请求（如刷新缓存、更新 LoRA 权重）
4. **多 Tokenizer 支持**：处理需要多个不同 tokenizer 的模型

## 核心文件分析

### tokenizer_manager.py

这是 TokenizerManager 的主要实现文件。关键数据结构和方法：

```python
class TokenizerManager:
    def __init__(self, server_args):
        # 加载 tokenizer
        self.tokenizer = self._load_tokenizer(server_args.tokenizer_path)
        # 请求结果队列
        self.result_queues: Dict[str, asyncio.Queue] = {}
        # 请求元数据
        self.request_metas: Dict[str, RequestMeta] = {}
    
    def tokenize_request(self, request: GenerationRequest) -> Request:
        """将文本请求转换为内部请求对象"""
        # 1. 应用 chat template（如果需要）
        # 2. tokenize 文本
        # 3. 创建结果队列
        # 4. 返回内部 Request 对象
        ...
    
    def get_result_queue(self, request_id: str) -> asyncio.Queue:
        """获取请求的结果队列"""
        return self.result_queues[request_id]
    
    def process_response(self, request_id: str, tokens: List[int]):
        """处理推理结果（由其他组件调用）"""
        # 将 token IDs 放入结果队列
        ...
```

### tokenizer_control_mixin.py

控制混入类，处理管理类请求：

```python
class TokenizerControlMixin:
    def flush_cache(self):
        """刷新 KV 缓存"""
        ...
    
    def update_weights(self, lora_path: str):
        """在线更新 LoRA 权重"""
        ...
    
    def get_model_info(self) -> ModelInfo:
        """获取模型信息"""
        ...
```

这些操作通过特殊的**控制请求**（control request）发送到 Scheduler，由 Scheduler 在适当的时机执行。

### tokenizer_manager_score_mixin.py

评分混入，为评分任务提供支持：

```python
class TokenizerManagerScoreMixin:
    def score_request(self, prompt, completion):
        """计算 logprob 分数"""
        # 构造评分请求
        # tokenize prompt + completion
        # 提交到 scheduler
        ...
```

### multi_tokenizer_mixin.py

用于需要多个 tokenizer 的场景（如某些多模态模型）：

```python
class MultiTokenizerMixin:
    def __init__(self):
        self.tokenizers: Dict[str, PreTrainedTokenizer] = {}
    
    def add_tokenizer(self, name: str, tokenizer_path: str):
        """添加额外的 tokenizer"""
        ...
```

## Tokenizer 的工作流程

### 文本 → Token IDs

```
"Hello, world!" 
    │
    ▼
tokenizer.encode("Hello, world!")  
    │
    ▼
[9906, 11, 1917, 0]  ← token IDs
```

### Chat Template 处理

对于 chat 模型，需要将对话消息格式化为模型特定的格式：

```python
# 原始输入
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]

# 模型特定的 chat template 处理后
# Llama 3: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>..."
# Qwen: "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n..."
```

### 请求结果管理

每个请求在 TokenizerManager 中注册一个 **结果队列**：

```python
def tokenize_request(self, request):
    request_id = generate_unique_id()
    
    # 创建结果队列
    result_queue = asyncio.Queue()
    self.result_queues[request_id] = result_queue
    
    # 构造内部请求
    req = InternalRequest(
        request_id=request_id,
        token_ids=token_ids,
        result_queue=result_queue,
    )
    
    return req
```

推理完成后，Detokenizer 或 TP Worker 会将生成的 token IDs 放入结果队列，TokenizerManager 消费队列并转换为文本。

## Tokenizer 目录

`python/sglang/srt/tokenizer/` 目录包含 tokenizer 相关的辅助代码，包括特殊 token 处理、tokenizer 配置等。

## 关键观察

1. **异步队列通信**：TokenizerManager 和下游组件通过异步队列通信，实现解耦
2. **双向转换**：既负责文本 → token IDs（tokenize），也负责间接的 token IDs → 文本（通过 DetokenizerManager）
3. **控制平面与数据平面分离**：控制请求（flush、update_weights）和数据请求（推理）走不同的路径
