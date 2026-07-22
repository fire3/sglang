# 可观测性（Observability）

> 参考文件：`python/sglang/srt/observability/`

## 角色与职责

可观测性模块提供了 SGLang 运行时的**监控、日志和追踪**能力，是生产环境中运维调优的基础设施。

## 目录结构

```
python/sglang/srt/observability/
├── ...  ← 可观测性相关代码
```

（该目录内容会随版本更新而变化，以下为典型功能）

## 核心功能

### 1. 日志（Logging）

SGLang 使用 Python 标准库 `logging` 进行日志记录：

```python
import logging

logger = logging.getLogger(__name__)
logger.info("Scheduler: batch of %d requests", batch_size)
```

日志级别通过 `server_args.log_level` 或环境变量控制：
- `INFO`：常规运行日志
- `DEBUG`：详细调试日志
- `WARNING`：警告信息
- `ERROR`：错误信息

配置文件：`python/sglang/srt/managers/configure_logging.py`

### 2. 指标（Metrics）

SGLang 收集关键运行指标用于监控和优化：

| 指标 | 说明 |
|---|---|
| `tokens_per_second` | 生成吞吐量 |
| `batch_size` | 当前批次大小 |
| `cache_hit_rate` | Radix Cache 命中率 |
| `memory_usage` | KV Cache 内存使用 |
| `request_queue_size` | 等待队列长度 |
| `prefill_time` | Prefill 阶段耗时 |
| `decode_time` | Decode 阶段耗时 |

这些指标可以通过：
- 日志文件查看
- Prometheus 端点暴露（如果配置）
- `sglang.profiler` 程序化获取

### 3. 追踪（Tracing）

#### Kernal API 日志

`python/sglang/kernel_api_logging.py` 提供了 CUDA 内核级别的日志追踪：

```python
# 启用内核日志
kernel_api_logging.enable()

# 每次 CUDA 内核调用会被记录
# Kernel: flash_attn_v2, args: ... duration: 1.2ms
# Kernel: addmm, args: ... duration: 0.3ms
```

这对于排查 CUDA 相关问题非常有用。

#### Profiler 工具

`python/sglang/profiler.py` 提供更高层次的分析：

```python
from sglang.profiler import Profiler

profiler = Profiler()

# 获得分析摘要
profiler.summary()  
# 输出:
# - Total time: 12.3s
# - Prefill time: 8.1s (65.9%)
# - Decode time: 4.2s (34.1%)
# - Throughput: 123.4 tokens/s
```

### 4. 状态统计

`pool_stats_observer.py`（在 scheduler_components/ 中）观察内存池的使用情况：

```python
class PoolStatsObserver:
    def report_pool_stats(self):
        """报告内存池使用统计"""
        return {
            "total_pages": total_pages,
            "used_pages": used_pages,
            "free_pages": free_pages,
            "utilization": used_pages / total_pages,
        }
```

`metrics_reporter.py` 负责将各类指标统一上报。

### 5. 请求级别的追踪

每个请求可以关联追踪信息：

```python
class RequestTrace:
    def __init__(self, request_id):
        self.request_id = request_id
        self.start_time = time.time()
        self.events = []
    
    def add_event(self, event_name, details=None):
        self.events.append({
            "time": time.time() - self.start_time,
            "event": event_name,
            "details": details,
        })
    
    def summary(self):
        """返回请求生命周期的时间线"""
        return self.events
```

## 日志配置

`configure_logging.py` 负责统一的日志配置：

```python
def configure_logging(server_args):
    logging.basicConfig(
        level=server_args.log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),  # 控制台输出
            logging.FileHandler(log_file) if log_file else None,
        ].filter(None),
    )
```

## 调试工具

### Debug Utils

`python/sglang/srt/debug_utils/` 提供了额外的调试辅助：

- 内存使用打印
- 模型中间输出检查
- 分布式通信验证

### State Capturer

`python/sglang/srt/state_capturer/` 可以捕获运行时状态快照：

- 当前批次信息
- 内存池状态
- Radix Cache 内容（摘要）

## 生产环境的可观测性

在生产部署中，建议结合以下工具：

| 工具 | 用途 |
|---|---|
| Prometheus + Grafana | 指标收集和可视化 |
| ELK / Loki | 日志聚合 |
| Jaeger / Zipkin | 分布式追踪 |
| SGLang Profiler | 性能分析 |

## 关键观察

1. **多层级**：从 CUDA 内核级（kernel_api_logging）到请求级（RequestTrace），覆盖完整的可观测性栈
2. **性能影响最小化**：日志和指标收集设计为低开销，不对推理性能产生显著影响
3. **可配置**：通过 server_args 和环境变量可以灵活控制可观测性的范围和粒度
4. **与调度器集成**：指标在调度器的事件循环中自然收集，无需额外的轮询机制
