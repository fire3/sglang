# 项目简介与定位

## SGLang 是什么？

SGLang 是一个高性能的大语言模型（LLM）和多模态模型推理服务框架，由 [LMSYS 实验室](https://lmsys.org/) 开源。它在业界被广泛使用，据官方数据支撑着全球超过 40 万张 GPU 的推理负载。

核心定位：**从单卡到大规模分布式集群的高效推理引擎**。

## 核心技术创新

### RadixAttention（基数树注意力）

这是 SGLang 最具标志性的创新，发表于 2024 年 1 月。传统的 KV Cache 管理通常以 LRU（最近最少使用）策略淘汰，但无法有效识别和复用共享前缀。RadixAttention 使用**基数树（Radix Tree）**来组织 KV Cache：

- 自动检测请求之间的公共前缀（如 System Prompt、对话历史）
- 在 Radix Tree 中存储共享的 KV Cache 节点
- 新请求到达时自动匹配最长公共前缀，直接从缓存中读取
- **效果**：最高可实现 **5x 推理加速**

### 零开销 CPU 调度器

传统调度器在组织批次（batch）时会产生额外的 CPU 计算开销。SGLang 的调度器被设计为"零开销"——调度操作与 GPU 计算并行执行，不阻塞推理。

### PD 分离（Prefill/Decode Disaggregation）

Prefill（预填充）和 Decode（解码）两个阶段的计算特性完全不同：
- **Prefill**：计算密集，适合用较少的 GPU 处理大量输入
- **Decode**：内存带宽密集，适合用较多的 GPU 处理大批量请求

SGLang 支持将两个阶段部署到不同的 GPU 集群上，各自独立扩缩容。

### 结构化输出加速

使用压缩有限状态机（Compressed FSM）技术加速 JSON Schema / CFG 约束下的结构化输出生成，实现 **3x 更快的 JSON 解码**。

## 支持的模型

SGLang 支持广泛的模型架构：

- **语言模型**：Llama、Qwen、DeepSeek（含 MLA 注意力变体）、Kimi、GLM、GPT、Gemma、Mistral 等
- **多模态模型**：LLaVA、Qwen-VL、DeepSeek-VL 等
- **Embedding 模型**：e5-mistral、gte、mcdse
- **Reward 模型**：Skywork
- **扩散模型**：WAN、Qwen-Image

## 支持的硬件

```text
NVIDIA:  GB200/B300/H100/A100/Spark/5090
AMD:     MI355/MI300
Intel:   Xeon CPU
Google:  TPU
Ascend:  NPU
其他:    Apple MLX, Moore Threads MUSA
```

## 应用场景

1. **在线推理服务**：提供 OpenAI 兼容的 API，直接替代生产环境
2. **强化学习后训练**：作为 RL 训练的 rollout 后端，被 AReaL、Miles、verl、Tunix 等框架采用
3. **研究探索**：快速实验新的模型架构和推理优化技术

## 相关资源

- [官方文档](https://docs.sglang.io/)
- [GitHub 项目](https://github.com/sgl-project/sglang)
- [LMSYS 博客](https://lmsys.org/blog/) — 包含详细的技术深度文章
- [Slack 社区](https://slack.sglang.io/)
