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

在分布式部署中，每个 TP Worker 运行在独立进程中，通过 ZMQ IPC 与主进程的 Scheduler 通信：

```
┌─ 主进程 ──────────────────────────────────────┐
│  Scheduler                                     │
│    └── ZMQ ROUTER (port)                       │
└───────────┬─────────────────────────────────────┘
            │ ZMQ IPC（主进程 → 所有 TP Worker 广播）
            │
   ┌────────┼────────┬────────┐
   ▼        ▼        ▼        ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│TP Wkr│ │TP Wkr│ │TP Wkr│ │TP Wkr│
│Rank 0│ │Rank 1│ │Rank 2│ │Rank 3│
└──────┘ └──────┘ └──────┘ └──────┘
   │        │        │        │
   └────────┼────────┬────────┘
            │ NCCL / torch.distributed
            │ (GPU 间 all-reduce / all-gather)
            ▼
      各 Worker 独立管理自己的 KV Cache
      （按 head 数分片，互不共享）
```

**关键设计原则**：
- Scheduler 通过 ZMQ 向所有 TP Worker 广播请求
- TP Worker **不直接相互通信**进行调度协调——它们各自收到相同请求，独立执行分片计算
- 计算中的张量通信（all-reduce 等）通过 NCCL 或 torch.distributed 完成
- 每个 TP Worker 拥有完全独立的 KV Cache，按 head 数分片

## KV Cache 管理（TP 分片）

### 每个 Worker 拥有独立的 KV Cache

`TpModelWorker`（`managers/tp_worker.py:273`）中，每个 Worker 进程持有自己的 `ModelRunner`，而 `ModelRunner` 又持有独立的 KV Cache 管理对象：

```python
# tp_worker.py:102-106
def get_memory_pool(self):
    return (
        self.model_runner.req_to_token_pool,          # 页表
        self.model_runner.token_to_kv_pool_allocator,  # 空闲列表分配器
    )
```

- `self.model_runner.req_to_token_pool` — 请求 ID → token slot 的页表映射
- `self.model_runner.token_to_kv_pool` — 实际的 KV 张量缓冲区（`KVCache` 子类）
- `self.model_runner.token_to_kv_pool_allocator` — 基于空闲列表的分配器

### KV Cache 按 head 数分片

TP 模式下，每张 GPU 只存储全部 KV head 的 `1 / tp_size` 部分：

```python
# configs/model_config.py:1076-1083
def get_num_kv_heads(self, tensor_parallel_size) -> int:
    total_num_kv_heads = self.get_total_num_kv_heads()
    return max(1, total_num_kv_heads // tensor_parallel_size)
```

这个分片后的 head 数在创建 KV Cache pool 时使用：

```python
# kv_cache_configurator.py:1354
token_to_kv_pool = pool_cls(
    max_total_num_tokens=...,
    head_num=self.model_config.get_num_kv_heads(get_parallel().attn_tp_size),
    head_dim=self.model_config.head_dim,
    ...
)
```

最终每个 rank 的 KV 缓冲区形状为：

```
k_buffer[layer] = torch.zeros((num_slots, heads_per_rank, head_dim))
v_buffer[layer] = torch.zeros((num_slots, heads_per_rank, v_head_dim))
```

### 内存预算随 TP 规模线性缩放

`MemoryPoolConfigurator` 计算每个 Worker 的内存预算时使用 TP 分片后的 head 数：

```python
# pool_configurator.py:185,262
tp_size = get_parallel().attn_tp_size
cell_size = (
    model_config.get_num_kv_heads(tp_size)           # 分片后 head 数
    * (model_config.head_dim + model_config.v_head_dim)
    * effective_num_layers
    * kv_size
)
```

这意味着 **TP 规模越大，每个 Worker 的 KV Cache 占用的 GPU 显存越少**（总显存不变，但分布到更多 GPU）。

### Attention 后端只操作本地 KV Cache

在 `flashattention_backend.py:1806-1819`，attention 计算时仅读取当前 Worker 的 KV Cache 分片：

```python
key_cache, value_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
key_cache = key_cache.view(-1, self.page_size, layer.tp_k_head_num, layer.head_dim)
o = flash_attn_with_kvcache(
    q=..., k_cache=key_cache, v_cache=value_cache, ...
)
```

计算完成后，对 **输出 hidden states**（而非 KV Cache）做 all-reduce 使结果在 TP rank 间一致。这是唯一的跨 Worker 张量通信——**KV Cache 本身从未在 Worker 间传输**。

| 方面 | 结论 |
|---|---|
| 每个 Worker 独立管理自己的 KV Cache？ | **是**。每个 `TpModelWorker` 拥有独立的 `MHATokenToKVPool` 和空闲列表分配器 |
| 分片策略 | 按 KV head 分片：`total_kv_heads // tp_size` |
| Worker 间是否共享/通信 KV Cache？ | **否**。Attention 只读取本地缓冲区。只有输出 hidden states 需要 all-reduce |
| 特殊例外 | HiCache（跨 GPU KV 层迁移）、DSv4 压缩 KV 等专用功能有特殊通信模式 |

## TP Worker 间通信

### 通信层架构

TP Worker 间的通信由 `GroupCoordinator`（`parallel_state.py:220`）统一管理，它封装了 PyTorch `ProcessGroup` 并提供运行时通信器插件选择：

```
GroupCoordinator.all_reduce()
    │
    ├── HPU 通信器
    ├── XPU 通信器
    ├── NPU 通信器
    ├── PyNcclCommunicator（纯 Python ctypes NCCL 包装）
    ├── CustomAllReduce（CUDA 寄存器级 all-reduce）
    ├── QuickAllReduce（AMD ROCm all-reduce）
    ├── TorchSymmMemCommunicator（对称内存映射）
    └── torch.distributed.all_reduce()（兜底）
```

### 分布式初始化流程

```
TpModelWorker.__init__()
    │
    └── ModelRunner.init_torch_distributed()
            │
            └── bootstrap.init_torch_distributed()
                    │
                    ├── _resolve_backend()
                    │     └── platform.get_torch_distributed_backend_str()
                    │           ├── CUDA → "nccl"
                    │           ├── CPU  → "gloo"
                    │           ├── NPU  → "hccl"
                    │           └── XPU  → "xccl"
                    │
                    └── _init_parallel_groups()
                          ├── init_distributed_environment()
                          │     └── torch.distributed.init_process_group(backend)
                          └── initialize_model_parallel()
                                ├── new_group() → TP group
                                ├── new_group() → PP group
                                ├── new_group() → attn-TP group
                                └── new_group() → MoE-EP group
```

### 激活值通信（前向传播中）

模型前向传播中通过 `communication_op.py` 的包装函数调用集合通信：

| 通信操作 | 函数 | 用途 |
|---|---|---|
| all-reduce | `tensor_model_parallel_all_reduce()` | 每层输出、MoE 中间结果 |
| all-gather | `tensor_model_parallel_all_gather()` | Q/K 头收集（某些模型） |
| gather | `tensor_model_parallel_gather()` | Vocab 输出收集 |
| fused all-reduce + rmsnorm | `tensor_model_parallel_fused_allreduce_rmsnorm()` | 融合优化 |
| quant all-reduce | `tensor_model_parallel_quant_all_reduce()` | FP8 量化 all-reduce |

**典型模式**（以 Llama 模型的每一层为例）：

```python
# 前向计算
hidden_states = self.self_attn(hidden_states)
hidden_states = tensor_model_parallel_all_reduce(hidden_states)  # ← 通信在这里
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)
hidden_states = tensor_model_parallel_all_reduce(hidden_states)  # ← 通信在这里
```

### 跨 CPU 通信

当 TP Worker 运行在 CPU 而非 GPU 上时（`SGLANG_USE_CPU_ENGINE=1`）：

```
CPU 上的 all-reduce
    │
    ├── 共享内存可用（Intel AMX / ARM64）
    │     └── torch.ops.sgl_kernel.shm_allreduce()
    │          通过 Linux 共享内存（mmap）直接读写，比 Gloo 快
    │
    └── 共享内存不可用
          └── torch.distributed.all_reduce(group, backend="gloo")
```

CPU 模式下 `SGLANG_USE_CPU_ENGINE=1` 自动选择 `"gloo"` 后端（`platforms/cpu.py:119-120`），不需要 NCCL。

### 跨操作系统通信

TP Worker 目前**不支持跨操作系统**的通信。理由如下：

1. **`torch.distributed` 的 NCCL/Gloo 后端都是单机设计**——它们假设所有进程在同一台机器上，通过共享的 IPC 通道（NCCL unique ID / Gloo shared store）完成 bootstrap
2. **NCCL 本身支持多机（跨网络）**，但 SGLang 的 `GroupCoordinator` 默认使用单机 IPC bootstrap（`torch.distributed.init_process_group` 的 `store` 参数使用本地 shared memory）
3. **跨机 TP** 在工程上极少使用，因为 TP 对延迟极度敏感（每层都 all-reduce），跨网络延迟会严重拖慢性能。跨机通常使用 PP（流水线并行）或 EP（专家并行）
4. **Mooncake 传输引擎**（`mooncake_transfer_engine.py`）是一个例外——它支持跨机器 EP 通信，但不用于 TP

### 为什么 PyNccl 而非直接 torch.distributed.nccl

SGLang 使用 `PyNcclCommunicator`（`pynccl.py` + `pynccl_wrapper.py`）而非直接调用 `torch.distributed` 的 NCCL 后端，主要原因是 **CUDA Graph 兼容性**：

```python
# pynccl_wrapper.py:1-24 文档字符串
# PyTorch 的 NCCL 后端在 CUDA Graph 捕获中工作不正常，
# 因为 torch.distributed 内部包含无法被 graph 捕获的 CUDA API 调用。
# SGLang 通过 ctypes 直接调用 libnccl.so，避免了这个问题。
```

PyNccl 通过 `ctypes.CDLL` 动态加载 `libnccl.so.2`（ROCm 加载 `librccl.so.1`，MUSA 加载 `libmccl.so.2`）——`pynccl_wrapper.py:56-63`。

## 移植到其他加速器

### 现有平台抽象层

SGLang 已有较完善的平台抽象机制，位于 `platforms/` 目录：

```
platforms/
├── interface.py      ← SRTPlatform(DeviceMixin) 抽象基类
├── device_mixin.py   ← DeviceMixin 设备方法
├── cuda.py           ← CUDA 实现
├── rocm.py           ← AMD ROCm（继承 CUDA）
├── cpu.py            ← CPU 实现
├── npu.py            ← Ascend NPU 实现
├── hpu.py            ← Habana HPU 实现
├── xpu.py            ← Intel XPU 实现
├── musa.py           ← Moore Threads MUSA
└── mlx.py            ← Apple MLX
```

平台接口的关键抽象：

```python
# platforms/interface.py:26-139
class SRTPlatform(DeviceMixin):
    def get_default_attention_backend(self) -> str: ...
    def get_graph_runner_cls(self) -> type: ...
    def get_mha_kv_pool_cls(self) -> type: ...
    def get_paged_allocator_cls(self) -> type: ...
    def get_piecewise_backend_cls(self) -> type: ...

# platforms/device_mixin.py:94-260
class DeviceMixin:
    def is_cuda(self) -> bool: ...
    def is_npu(self) -> bool: ...
    def get_device(self) -> torch.device: ...
    def get_torch_distributed_backend_str(self) -> str: ...
    def synchronize(self): ...
```

### 需要注意的关键问题

将 TP Worker 移植到新的非 CUDA 加速器时，以下是最需要关注的重点：

#### 1. 通信后端

| 原平台 | 通信后端 | 替换方案 |
|---|---|---|
| CUDA | NCCL | HCLL（NPU）/ XCCL（XPU）/ Gloo（CPU） |
| PyNccl（ctypes nccl） | `libnccl.so` | 使用对应平台的 torch.distributed 后端或自包装 |
| CustomAllReduce（CUDA IPC） | 寄存器级 peer 访问 | 直接禁用（NPU 已示范：`npu/utils.py:78`） |

**关键文件**：`distributed/device_communicators/` 下已有 NPU、XPU、HPU 通信器模板可直接参考。

#### 2. 运算内核

`torch.ops.sgl_kernel.*` 是最大的 CUDA 依赖。大量算子调用分布在各 `layers/` 中：

```
torch.ops.sgl_kernel.flash_attn_varlen_func     → 注意力（最重要）
torch.ops.sgl_kernel.flash_attn_with_kvcache     → 注意力 decode
torch.ops.sgl_kernel.silu_and_mul                → 激活函数（已有 _cpu 变体）
torch.ops.sgl_kernel.rmsnorm                     → 归一化（已有 _cpu 变体）
torch.ops.sgl_kernel.rotary_embedding            → RoPE（已有 _cpu 变体）
torch.ops.sgl_kernel.fused_experts               → MoE（已有 _cpu 变体）
```

新平台有两种选择：
- **提供 `sgl_kernel_<platform>` 替代实现**（如 NPU 在 `npu/utils.py:99` 做的方式）
- **通过 `torch.library` / `torch.ops` 注册自定义算子**
- **使用纯 PyTorch / Triton fallback**

#### 3. CUDA Graph 替代

CUDA Graph（`torch.cuda.CUDAGraph`）是 NVIDIA 专属：

```python
# model_runner.py:845
def init_cuda_graphs(self):
    graph_runner_cls = current_platform.get_graph_runner_cls()
    self.graph_runner = graph_runner_cls(self)  # 平台可覆盖
```

已有替代实现：
- `compilation/npu_piecewise_backend.py` — Ascend NPU
- `compilation/xpu_piecewise_backend.py` — Intel XPU

新平台需要实现 `get_graph_runner_cls()` 返回自己的图捕获/回放类。

#### 4. 设备同步

`model_runner.py:338` 目前硬编码了 `torch.cuda.Event`：

```python
self.war_fastpath_read_done_event: Optional[torch.cuda.Event] = None
```

应改用 `torch.get_device_module(self.device).Event` 或通过平台抽象层创建。

#### 5. 动态设备分支（需复制的模式）

当前代码中有大量 `is_hip()` / `is_npu()` / `is_xpu()` / `is_cuda()` 的硬编码分支：

```python
# 散布在 attention、layernorm、activation、linear、MoE 各处
if is_npu():
    # NPU 专用路径
elif is_hip():
    # AMD 专用路径
else:
    # CUDA 默认路径
```

新平台的定义入口：`sglang.srt.platforms` 的 `entry_points` 插件系统，在 `platforms/__init__.py` 中通过 `__getattr__` 懒加载。

#### 6. DeepGEMM（CUDA 独占）

`layers/deep_gemm_wrapper/` 完全 CUDA 绑定：

```python
# entrypoint.py:45-46
_ensure_cuda()  # 强制张量在 CUDA 设备
```

新平台需：
- 禁用 DeepGEMM（`configurer.py:24` 已 gated on `is_cuda() or is_musa()`）
- 提供等效的 FP8 GEMM 替代，或降级到 BF16/FP16

#### 7. 模型加载适配

```python
# model_runner.py:934
adjust_config_with_unaligned_cpu_tp()  # CPU 特有的 TP 对齐调整
```

不同加速器可能有不同的数据排列要求和权重加载策略，需通过 `BaseModelLoader` 的子类实现。

### 现有非 CUDA 平台清单

| 平台 | 目录 | 状态 |
|---|---|---|
| AMD ROCm | `platforms/rocm.py` | 大部分通过 HIP 兼容 CUDA API |
| Ascend NPU | `hardware_backend/npu/` | 较完整：attention、MoE、graph_runner、内存管理 |
| Intel XPU | `hardware_backend/xpu/` | 部分实现 |
| Habana HPU | `hardware_backend/hpu/` | 基础通信器 |
| Apple MLX | `hardware_backend/mlx/` | tp_worker、model_runner、scheduler_mixin |
| CPU | `platforms/cpu.py` | 较完整：gloo 通信、AMX 内核、共享内存 all-reduce |
| Moore Threads MUSA | `platforms/musa.py` | 基础 attention 后端 |
