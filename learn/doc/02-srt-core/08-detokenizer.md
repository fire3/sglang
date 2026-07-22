# Detokenizer Manager

> 参考文件：`python/sglang/srt/managers/detokenizer_manager.py`

## 角色与职责

`DetokenizerManager` 是请求管道的**最后一站**，负责将推理生成的 token IDs 转换回人类可读的文本。它是 TokenizerManager 的"逆向操作"。

## 为什么需要独立的 Detokenizer？

将 Detokenizer 从 TokenizerManager 中分离出来有以下几个原因：

1. **解耦关注点**：TokenizerManager 专注于**入站**（请求 → tokens），DetokenizerManager 专注于**出站**（tokens → 响应）
2. **批处理效率**：Detokenizer 可以批量处理多个请求的 tokens，提高效率
3. **异步处理**：Detokenizer 可以在 TP Worker 推理的同时异步工作
4. **增量解码**：流式场景下，每次只输出几个 token，需要增量 detokenize（避免解码不一致）

## 核心实现

```python
class DetokenizerManager:
    def __init__(self, server_args):
        # 共享 TokenizerManager 的 tokenizer
        self.tokenizer_manager = ...
        
        # 结果队列
        self.result_queues: Dict[str, asyncio.Queue] = {}
    
    def process_tokens(self, request_id: str, tokens: List[int], finished: bool):
        """处理推理产生的 tokens"""
        # 1. Detokenize：tokens → 文本
        text = self.tokenizer.decode(
            tokens,
            skip_special_tokens=True,
            # 重要：增量解码时保留未完成的字节
            clean_up_tokenization_spaces=False,
        )
        
        # 2. 增量处理
        # 对于流式场景，只输出新的文本片段
        delta = self._get_text_delta(request_id, text)
        
        # 3. 放入结果队列
        if finished:
            self.result_queues[request_id].put((
                delta, 
                finish_reason="stop"  # 或 "length"
            ))
        else:
            self.result_queues[request_id].put((delta, None))
    
    def decode_streaming(self, request_id: str, tokens: List[int]):
        """流式解码 - 增量输出 token 对应的文本"""
        # 关键问题：一个 token 可能对应多个字符，也可能是一个字符的一部分
        # 需要处理跨 token 的字符边界
        
        # 安全的做法：积累一定数量的 token 后一起 decode
        # 或者使用已知 tokenizer 的转义规则
        
        decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return decoded
```

## 增量解码的挑战

流式推理中，Detokenizer 面临一个经典难题：**字节级 BPE tokenizer 的不完整性**。

### 问题场景

```
模型生成 token: ["▁Hello", "▁world", "!"]
           decode: " Hello world!"

但如果逐 token decode:
"▁Hello" → " Hello"
"▁world" → " world"  ← OK，"▁"前缀表示空格
"!"      → "!"       ← OK

问题在于：有些 token 是一个 UTF-8 字节的一部分
例如: ["Ġ", "H", "ello"]  →  " Hello"（完整）
但如果逐个输出：
"Ġ" → " "  ← 空格
"H" → "H"  ← OK
"ello" → "ello"  ← OK
```

### SGLang 的处理方式

SGLang 采用**缓存式增量解码**：

```python
class DetokenizerManager:
    def __init__(self):
        # 每个请求的文本缓存
        self._text_cache: Dict[str, str] = {}
    
    def _get_text_delta(self, request_id, full_text):
        """计算文本增量"""
        prev_text = self._text_cache.get(request_id, "")
        delta = full_text[len(prev_text):]
        self._text_cache[request_id] = full_text
        return delta
```

这样，即使某个中间 decode 结果包含不完整的字节序列，最终输出的"增量"是安全的。

## 与 TokenizerManager 的关系

```
TokenizerManager (入站)           DetokenizerManager (出站)
    │                                   │
    │  "Hello!" → [9906, 11, 1917, 0]   │  [9906, 11, 1917] → "Hello!"
    │                                   │
    └────── 同一个 tokenizer 实例 ──────┘
```

它们共享同一个 `tokenizer` 实例，但职责不重叠。

## 特殊处理

DetokenizerManager 还需要处理：

1. **Streaming (SSE)**：逐 token 输出，通过 `process_tokens` 的多次调用实现
2. **Special tokens 过滤**：跳过 `<|endoftext|>`、`<pad>` 等特殊 token
3. **Stop strings 检测**：检测是否生成了用户定义的停止字符串
4. **Partial UTF-8**：处理跨 token 的 UTF-8 字符边界

## 关键观察

1. **一个 tokenizer，两个管理器**：TokenizerManager 和 DetokenizerManager 共享同一个 tokenizer 实例，但处理不同方向的数据
2. **流式友好**：通过增量解码支持流畅的 SSE 输出
3. **批处理友好**：可以同时处理多个请求的 detokenization
4. **文本质量**：正确的 detokenization 直接影响用户体验，尤其是在流式输出和中文等 UTF-8 语言场景
