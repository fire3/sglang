# 构建系统

> 参考文件：`python/pyproject.toml`, `python/setup.py`, `python/pyproject_*.toml`

## 项目元数据（pyproject.toml）

`python/pyproject.toml` 是项目的核心构建配置文件，使用 PDM 作为包管理器。主要内容：

```toml
[project]
name = "sglang"
dynamic = ["version"]
dependencies = [
    "torch",
    "transformers",
    "numpy",
    # ... 大量依赖
]
```

SGLang 的依赖范围很广，因为需要支持多种硬件后端和模型格式。

## 多平台变体

项目提供了多个平台的 `pyproject` 变体：

| 文件 | 用途 |
|---|---|
| `pyproject.toml` | 默认配置（NVIDIA GPU） |
| `pyproject_cpu.toml` | CPU 版本 |
| `pyproject_npu.toml` | Ascend NPU 版本 |
| `pyproject_xpu.toml` | Intel XPU 版本 |
| `pyproject_other.toml` | 其他平台 |

每个变体中的依赖项根据硬件能力有所不同，例如 GPU 版本包含 CUDA 相关的包，CPU 版本则不包含。

## 安装脚本（setup.py）

```python
# python/setup.py
```

`setup.py` 中定义了：
- `ext_modules`：使用 `setuptools.Extension` 编译 C/C++ 扩展
- 可选的 AOT 编译内核（与 `sgl-kernel` 配合）
- 自定义构建命令

## sgl-kernel 构建

`sgl-kernel/` 是一个独立的子项目，有自己的 `pyproject.toml` 和 `CMakeLists.txt`：

```
sgl-kernel/
├── CMakeLists.txt       ← CMake 构建配置
├── csrc/                ← CUDA/C++ 源码
├── include/             ← 头文件
├── python/              ← Python 绑定（Pybind11）
├── tests/               ← 测试
└── benchmark/           ← 基准测试
```

这个包使用 **AOT（Ahead-of-Time）** 编译方式，预编译优化后的 CUDA/C++ 内核，安装时作为 Python 包直接引入。

支持的平台构建脚本：
- `setup_rocm.py` — AMD ROCm
- `setup_musa.py` — Moore Threads MUSA
- `setup_metal.py` — Apple Metal

## JIT Kernel 构建

`python/sglang/jit_kernel/` 中的内核使用 **JIT（Just-in-Time）** 编译，在运行时编译 CUDA 代码。这种方式的特点是：

- **优点**：可以针对具体的 GPU 型号和输入尺寸生成最优代码
- **缺点**：首次调用时有编译延迟

JIT Kernel 的构建通常通过 `triton` 或 `torch.compile` 完成。

## 构建依赖关系图

```
pip install sglang
    │
    ├── python/setup.py
    │       └── 编译 C/C++ 扩展
    │
    ├── sgl-kernel/ (AOT)
    │       └── cmake + nvcc 编译 → .so 文件
    │
    └── pip 依赖
            ├── torch (GPU 计算框架)
            ├── transformers (模型加载)
            ├── flashinfer (注意力内核)
            ├── triton (JIT 编译)
            ├── fastapi + uvicorn (HTTP 服务)
            ├── grpcio (gRPC)
            └── ...
```

## 关键观察

1. **双轨制内核**：SGLang 同时使用 AOT（高性能生产）和 JIT（灵活调试）两种编译方式
2. **平台分离**：不同硬件平台使用独立的 pyproject 文件，避免不必要的依赖
3. **依赖复杂**：由于功能全面（分布式、量化、多模态等），依赖项相当多
