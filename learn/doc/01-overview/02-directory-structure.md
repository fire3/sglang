# 顶层目录结构

SGLang 项目是一个包含 Python、CUDA/C++、Rust 等多语言的混合项目。以下是最顶层的目录和文件：

```
sglang/
├── python/                 ← 主 Python 包（核心逻辑所在）
│   ├── sglang/             ← 包根目录
│   │   ├── __init__.py
│   │   ├── launch_server.py    ← 服务启动入口
│   │   ├── global_config.py    ← 全局配置
│   │   ├── version.py          ← 版本号
│   │   ├── check_env.py        ← 环境检查
│   │   ├── utils.py            ← 通用工具
│   │   ├── profiler.py         ← Profiling 工具
│   │   ├── kernel_api_logging.py ← 内核 API 日志
│   │   ├── srt/                ← ⭐ SGLang Runtime — 核心引擎
│   │   ├── lang/               ← 前端语言
│   │   ├── jit_kernel/         ← JIT 编译的 CUDA 内核
│   │   ├── kernels/            ← Python 内核包装
│   │   ├── multimodal_gen/     ← 多模态生成
│   │   ├── cli/                ← 命令行接口
│   │   ├── benchmark/          ← Python 层基准测试
│   │   ├── eval/               ← 评估工具
│   │   └── test/               ← 包内测试
│   ├── pyproject.toml          ← 项目元数据 & 依赖
│   ├── setup.py                ← 安装脚本
│   └── pyproject_*.toml        ← 各平台的变体配置
│
├── sgl-kernel/             ← ⭐ AOT 预编译 CUDA/C++ 内核包
├── test/                   ← 顶层测试套件
├── benchmark/              ← 顶层性能基准测试（46 个子目录）
├── docs_new/               ← Mintlify 文档网站
├── examples/               ← 使用示例
├── proto/                  ← Protocol Buffers 定义
├── rust/                   ← Rust 组件
│   ├── sglang-grpc/        ← Rust gRPC 服务
│   └── sglang-mm/          ← Rust 多模态代理
├── docker/                 ← Docker 镜像构建
├── scripts/                ← 开发/部署脚本
├── 3rdparty/               ← 第三方依赖
├── assets/                 ← 静态资源（Logo 等）
├── .github/                ← GitHub Actions CI/CD
└── experimental/           ← 实验性功能
```

## SRT 核心目录详解

`python/sglang/srt/` 是 SGLang 最核心的目录，包含 42+ 个子模块。以下是按功能分类：

### 服务层
| 目录/文件 | 用途 |
|---|---|
| `entrypoints/` | 服务入口（HTTP、gRPC、Engine） |
| `managers/` | 核心管理器（Tokenizer、Scheduler、TP Worker） |
| `connector/` | 模块间连接抽象 |

### 模型层
| 目录/文件 | 用途 |
|---|---|
| `models/` | 所有支持的模型实现（约 100+ 个文件） |
| `layers/` | 神经网络层（Attention、Linear、MoE 等） |
| `model_loader/` | 模型加载器 |
| `model_executor/` | 模型执行管理 |
| `sampling/` | 采样参数和逻辑 |

### 内存管理
| 目录/文件 | 用途 |
|---|---|
| `mem_cache/` | 内存缓存系统（Radix Cache、KV Cache 等） |

### 分布式
| 目录/文件 | 用途 |
|---|---|
| `distributed/` | 分布式通信基础 |
| `disaggregation/` | PD 分离（Prefill/Decode Disaggregation） |
| `speculative/` | 推测解码 |
| `elastic_ep/` | 弹性专家并行 |
| `ray/` | Ray 集成 |

### 其他
| 目录/文件 | 用途 |
|---|---|
| `constrained/` | 约束解码（JSON Schema、CFG） |
| `multimodal/` | 多模态处理 |
| `compilation/` | torch.compile / CUDA Graph |
| `quantization/`（在 layers/ 下） | 量化（FP4/FP8/INT4/AWQ/GPTQ） |
| `lora/` | LoRA 适配 |
| `hardware_backend/` | 硬件后端抽象 |
| `platforms/` | GPU 平台适配（CUDA/ROCm/CPU） |
| `observability/` | 可观测性（日志/指标/追踪） |

## 关键架构分层图

```
┌─────────────────────────────────────────────┐
│           HTTP / gRPC / Engine API          │  ← entrypoints/
├─────────────────────────────────────────────┤
│  Tokenizer → Scheduler → TPWorker → Model   │  ← managers/ + models/
├─────────────────────────────────────────────┤
│        Layers / Attention / Sampling         │  ← layers/ + sampling/
├─────────────────────────────────────────────┤
│     Memory Pool / Radix Cache / KV Cache    │  ← mem_cache/
├─────────────────────────────────────────────┤
│       Distributed Comm / PD / Spec Dec      │  ← distributed/ + 高级特性
├─────────────────────────────────────────────┤
│         CUDA Kernels (AOT + JIT)            │  ← sgl-kernel/ + jit_kernel/
└─────────────────────────────────────────────┘
```
