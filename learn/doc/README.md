---
home: true
title: 首页
heroText: SGLang 源码分析
tagline: 逐步深入高性能 LLM 推理框架的源码世界
actions:
  - text: 开始学习 →
    link: /01-overview/
    type: primary
features:
  - title: 阶段一 · 项目概览
    details: 了解 SGLang 是什么、目录结构、构建系统、前端语言和 CLI 工具
  - title: 阶段二 · SRT 核心
    details: 深入服务端运行时，理解引擎、调度器、Tokenizer 等核心组件的协作
  - title: 阶段三 · 模型执行
    details: 分析模型加载、层实现、前向传播、采样和约束解码的完整链路
  - title: 阶段四 · 分布式高级特性
    details: 掌握张量/流水线/数据/专家并行、PD 分离和推测解码
  - title: 阶段五 · 内存与性能
    details: 理解 RadixAttention 前缀缓存、KV Cache 管理、CUDA 内核开发
  - title: 阶段六 · 工程实践
    details: 学习测试体系、CI/CD、调试方法和贡献指南
footer: MIT Licensed | Copyright © SGL Project
---

## 项目概述

[SGLang](https://github.com/sgl-project/sglang) 是一个高性能的大语言模型（LLM）和多模态模型推理框架，由 LMSYS 实验室开源。其核心创新包括：

- **RadixAttention**：基于基数树（Radix Tree）的自动前缀缓存，实现高效的 KV Cache 复用
- **零开销 CPU 调度器**：批量调度过程中不引入额外计算开销
- **PD 分离**：Prefill 和 Decode 阶段独立部署，分别扩缩容
- **结构化输出加速**：通过压缩有限状态机实现 3x JSON 解码加速

本学习项目按**六个阶段**渐进式深入源码，适合有一定 PyTorch 和 Transformer 基础的开发者。

## 如何使用

```bash
# 安装依赖后本地运行 VuePress
cd learn/doc
pnpm install
pnpm run dev
```

每个阶段的分析文档都包含：
- 要阅读的具体代码文件路径
- 关键概念解释和架构图
- 学习产出清单
