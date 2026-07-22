# TP Worker（张量并行工作者）

> 参考文件：`python/sglang/srt/managers/tp_worker.py`

## 角色与职责

`TP Worker` 是直接在 GPU 上执行模型推理的组件。在张量并行（TP）模式下，多个 TP Worker 共同处理同一个模型的不同部分。

**核心职责**：
1. **模型加载**：将模型权重从磁盘加载到 GPU
2. **前向传播**：执行模型的 `forward()` 计算
3. **采样**：从 logits 中采样生成下一个 token
4. **KV Cache 管理**：在 GPU 上管理 KV Cache
5. **CUDA Graph**：构建和运行 CUDA Graph 加速

## 核心实现

```python
class TpWorker:
    def __init__(self, server_args):
        # 分布式初始化
        self.tp_rank = get_tp_rank()
        self.tp_size = server_args.tp_size
        
        # 加载模型
        self.model = self._load_model(server_args)
        
        # 初始化 KV Cache
        self.kv_cache = self._init_kv_cache(server_args)
        
        # CUDA Graph
        self.cuda_graph_runner = ...
    
    def _load_model(self, server_args):
        """加载模型到 GPU"""
        # 1. 获取模型类
        model_cls = get_model_class(server_args.model_path)
        
        # 2. 初始化模型（只加载本 TP rank 的权重分片）
        model = model_cls(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            ...
        )
        
        # 3. 加载权重
        load_weights(model, server_args.model_path)
        
        # 4. 移动到 GPU
        model = model.cuda()
        
        return model
    
    def forward_batch(self, batch: Batch) -> ModelOutput:
        """执行一个批次的前向传播"""
        # 1. 准备输入张量
        input_ids = batch.input_ids.cuda()
        positions = batch.positions.cuda()
        
        # 2. 执行前向
        with torch.inference_mode():
            logits = self.model(
                input_ids=input_ids,
                positions=positions,
                kv_cache=self.kv_cache,
                ...
            )
        
        # 3. 采样
        next_tokens = self.sampler.sample(logits)
        
        return ModelOutput(
            next_tokens=next_tokens,
            logprobs=logprobs,
            ...
        )
```

## 模型加载流程

### 1. 获取模型类

```python
# python/sglang/srt/models/ 中注册的模型
model_cls = ModelRegistry.get_model(server_args.model_path)
```

`ModelRegistry` 根据模型名称或路径查找对应的模型实现类。

### 2. 模型分片

在 TP 模式下，每个 Worker 只加载模型权重的一部分：

```python
# 例如 TP=4 时
# TP Rank 0: 加载 0, 4, 8, ... 层的权重
# TP Rank 1: 加载 1, 5, 9, ... 层的权重
# ...
```

这种分片对用户透明——从外部看仍然是完整的模型。

### 3. 权重加载

```python
def load_weights(model, model_path):
    # 从 HuggingFace 格式加载
    state_dict = torch.load(model_path)
    
    # 按 TP 分片
    sharded_dict = shard_for_tp(state_dict, tp_rank, tp_size)
    
    # 加载到模型
    model.load_state_dict(sharded_dict)
```

## 前向传播流程

```
Batch 到达
    │
    ├── input_ids: [batch_size, seq_len]
    ├── positions: [batch_size, seq_len]
    ├── prefix_lens: [batch_size]  ← Radix Cache 前缀长度
    └── extend_lens: [batch_size]  ← 需要计算的长度
    │
    ▼
Embedding Lookup
    │
    ▼
Transformer Layers × N
    │
    ├── Self-Attention (使用 KV Cache)
    │   └── 对于前缀部分，从 KV Cache 读取
    │   └── 对于扩展部分，计算新的 KV
    │
    ├── MLP / MoE
    │
    └── LayerNorm + Residual
    │
    ▼
Final LayerNorm
    │
    ▼
LM Head (vocab projection)
    │
    ▼
Logits [batch_size, vocab_size]
```

## 采样

采样器从 logits 中生成下一个 token：

```python
class Sampler:
    def sample(self, logits, sampling_params):
        """从 logits 采样"""
        # 应用 temperature
        if sampling_params.temperature != 1.0:
            logits = logits / sampling_params.temperature
        
        # 应用 top-k / top-p
        if sampling_params.top_k:
            logits = top_k_filter(logits, sampling_params.top_k)
        if sampling_params.top_p:
            logits = top_p_filter(logits, sampling_params.top_p)
        
        # 采样
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)
        
        return next_tokens
```

## CUDA Graph 加速

TP Worker 支持使用 CUDA Graph 来加速推理。CUDA Graph 将一组 CUDA 内核调用录制为图，避免重复的内核启动开销：

```python
class CudaGraphRunner:
    def __init__(self, model, batch_size):
        # 预热并录制 CUDA Graph
        self.graph = torch.cuda.CUDAGraph()
        
        with torch.cuda.graph(self.graph):
            self.output = model(self.static_inputs)
    
    def run(self, batch):
        # 将输入复制到静态缓冲区
        self.static_inputs.copy_(batch.input_ids)
        
        # 回放 CUDA Graph
        self.graph.replay()
        
        # 读取输出
        return self.output
```

## TP Worker 的多进程模式

在分布式部署中，多个 TP Worker 运行在不同进程中：

```
主进程 (Rank 0)             辅进程 (Rank 1..N)
    │                            │
    ├── TokenizerManager         ├── TP Worker 1
    ├── Scheduler                ├── TP Worker 2
    ├── TP Worker 0              ├── TP Worker 3
    └── DetokenizerManager       └── ...
```

进程间通过 **IPC（进程间通信）** 交换数据，使用共享内存或网络通信。

## 关键观察

1. **单进程 vs 多进程**：在单 GPU 模式下，所有组件在同一个进程中；在 TP 模式下，每个 GPU 有一个独立进程
2. **模型即函数**：`TpWorker.forward_batch()` 本质上是 `model.forward()` 的无状态包装
3. **采样在 GPU 上**：采样在 GPU 上完成，避免 CPU-GPU 数据传输
4. **CUDA Graph**：对固定形状的批次会使用 CUDA Graph 加速，对变长批次则回退到 eager 模式
