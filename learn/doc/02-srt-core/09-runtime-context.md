# 运行时上下文（RuntimeContext）

> 参考文件：`python/sglang/srt/runtime_context.py`, `python/sglang/srt/environ.py`

## 角色与职责

`RuntimeContext` 是 SGLang 进程级运行时状态的**集中管理容器**。它存储了在当前进程中所有组件共享的运行时信息。

## 核心类型

文件 `python/sglang/srt/runtime_context.py`（约 27KB）定义了以下主要类型：

### RuntimeContext 类

```python
class RuntimeContext:
    """进程级的运行时上下文"""
    
    def __init__(self):
        # 服务器参数（只读）
        self.server_args: Optional[ServerArgs] = None
        
        # 分布式状态
        self.tp_rank: Optional[int] = None
        self.tp_size: Optional[int] = None
        
        # 模型信息
        self.model_config: Optional[ModelConfig] = None
        self.model_dtype: Optional[torch.dtype] = None
        
        # 内存状态
        self.memory_pool: Optional[MemoryPool] = None
        
        # Profiling
        self.profiler: Optional[Profiler] = None
        
        # 调试
        self.debug_options: DebugOptions = DebugOptions()
        
        ...
```

### 上下文层级

RuntimeContext 分为多个层级，对应不同的作用域：

| 层级 | 作用域 | 内容 |
|---|---|---|
| **全局（Global）** | 进程级 | ServerArgs、全局开关、环境变量 |
| **运行时（Runtime）** | 引擎级 | Engine、分布式状态、模型信息 |
| **推理（Inference）** | 请求级 | 当前请求的 KV Cache 位置、采样参数 |

### 访问模式

```python
# 获取全局上下文
ctx = RuntimeContext.get_instance()

# 读取配置
tp_rank = ctx.tp_rank
model_dtype = ctx.model_dtype

# 设置值（初始化时）
RuntimeContext.set_instance(ctx)
```

## environ.py — 环境变量配置

`python/sglang/srt/environ.py`（约 60KB）是项目中**所有环境变量**的集中定义和管理文件。

### 设计模式

```python
# environ.py 中使用类变量 + 描述符来定义环境变量

class SGLangEnv:
    """SGLang 环境变量定义"""
    
    # 定义环境变量，包含默认值和文档
    SGLANG_TP_SIZE: int = Field(
        default=1,
        desc="张量并行度",
        env_var="SGLANG_TP_SIZE",
    )
    
    SGLANG_VERBOSE: bool = Field(
        default=False,
        desc="详细日志模式",
        env_var="SGLANG_VERBOSE",
    )
    
    ...
```

### 环境变量分类

| 类别 | 环境变量 | 用途 |
|---|---|---|
| 并行 | `SGLANG_TP_SIZE` | 张量并行度 |
| 并行 | `SGLANG_DP_SIZE` | 数据并行度 |
| 并行 | `SGLANG_EP_SIZE` | 专家并行度 |
| 调试 | `SGLANG_VERBOSE` | 详细日志 |
| 调试 | `SGLANG_ENABLE_LOGGING` | 启用日志 |
| 性能 | `SGLANG_USE_FLASHINFER` | 使用 FlashInfer |
| 性能 | `SGLANG_ENABLE_TORCH_COMPILE` | 启用 torch.compile |
| 内存 | `SGLANG_MEM_FRACTION` | GPU 内存使用比例 |
| 网络 | `SGLANG_MASTER_ADDR` | 分布式主节点地址 |
| 网络 | `SGLANG_MASTER_PORT` | 分布式主节点端口 |

### 访问方式

```python
# 推荐方式：通过 environ 读取
from sglang.srt.environ import SGLangEnv

verbose = SGLangEnv.SGLANG_VERBOSE
```

这种方式优于 `os.environ.get("SGLANG_VERBOSE")`，因为它提供了类型转换、默认值和文档。

## RuntimeContext 与 environ 的关系

```
RuntimeContext (运行时内存状态)
    │
    ├── server_args (来自命令行参数)
    ├── 各组件引用（Engine、Scheduler 等）
    ├── 运行时指标（当前内存使用等）
    └── ...
    
SGLangEnv (环境变量配置)
    │
    └── 从 os.environ 读取
    └── 提供默认值
    └── 类型转换
```

两者互补：
- **RuntimeContext** 存储**可变**的运行时状态
- **environ.py** 提供**不可变**的启动时配置

## 关键观察

1. **进程级单例**：RuntimeContext 是一个进程内的全局单例，同一进程内的所有组件共享
2. **配置中心化**：所有环境变量在 `environ.py` 中集中定义，便于维护和查找
3. **懒初始化**：RuntimeContext 在需要时才初始化，避免不必要的开销
4. **向后兼容**：环境变量的命名和默认值需要考虑向后兼容性（可以通过 `deprecated` 机制标记过时的变量）
