# CLI 与工具

> 参考文件：`python/sglang/cli/`, `python/sglang/utils.py`, `python/sglang/check_env.py`, `python/sglang/profiler.py`

## 命令行接口（CLI）

`python/sglang/cli/` 目录定义了命令行入口：

```text
python/sglang/cli/
├── ...  # 各子命令的实现
```

通过 `sglang` 命令启动时，常用的子命令：

| 命令 | 用途 |
|---|---|
| `sglang` / `sglang.launch_server` | 启动推理服务器 |
| `sglang.check_env` | 检查环境配置 |

启动服务器的典型用法：

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000 \
    --tp-size 1
```

## 工具函数（utils.py）

`python/sglang/utils.py`（约 23KB）包含项目中广泛使用的通用工具：

```python
# 常用工具函数类别
- 日志辅助（log 格式化、颜色输出等）
- 类型检查工具
- 配置合并工具
- PyTorch 相关辅助（张量操作、设备管理）
- 时间/性能测量
- 异常处理
```

这个文件作为项目内的"瑞士军刀"，在多个模块中被引用。

## 环境检查（check_env.py）

`python/sglang/check_env.py`（约 18KB）是一个独立的环境验证工具：

```bash
python -m sglang.check_env
```

它会检查：
- Python 版本
- PyTorch 版本和 CUDA 可用性
- 关键依赖包是否安装
- GPU 型号和数量
- 环境变量配置是否正确
- 可选：运行简短的推理测试验证功能

这对于排查部署问题非常有用。

## Profiling 工具（profiler.py）

`python/sglang/profiler.py` 提供了性能分析能力：

```python
from sglang.profiler import Profiler

profiler = Profiler()
# ... 运行推理 ...
profiler.summary()
```

主要功能：
- 记录各阶段的耗时（Prefill、Decode、调度等）
- 统计吞吐量（token/s）
- 内存使用分析
- 生成 Chrome Trace 格式的火焰图

## Kernel API 日志

`python/sglang/kernel_api_logging.py` 提供了一个调试 CUDA 内核调用的日志系统：

- 记录每个内核调用的参数、耗时
- 可用于排查 CUDA 相关的问题
- 配合 `debug-cuda-crash` 技能使用

## 全局配置（global_config.py）

```python
# python/sglang/global_config.py

# 全局开关
ENABLE_MOE = False  # 是否启用 MoE
ENABLE_MLA = False  # 是否启用 MLA 注意力
VERBOSE = False     # 详细日志模式
```

这些配置项在启动时设置，影响整个运行时的行为。

## 其他工具文件

| 文件 | 用途 |
|---|---|
| `python/sglang/_mps_stub.py` | Apple MPS 后端的桩模块 |
| `python/sglang/_triton_stub.py` | Triton JIT 编译的桩模块 |
| `python/sglang/version.py` | 版本号管理 |
| `python/sglang/compile_deep_gemm.py` | DeepGEMM 编译工具 |
