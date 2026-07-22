# Python 包入口分析

> 参考文件：`python/sglang/__init__.py`, `python/sglang/launch_server.py`, `python/sglang/global_config.py`, `python/sglang/version.py`

## `__init__.py` — 包导出接口

```python
# python/sglang/__init__.py
```

这个文件定义了 `sglang` 包对外暴露的 API。主要导出内容包括：

- **运行时相关**：`Engine`, `ServerArgs` 等核心类
- **前端 API**：`gen`, `bind`, `image`, `video`, `audio`, `system`, `user`, `assistant` 等函数
- **工具函数**：`set_default`, `RuntimeEndpoint` 等

这是用户 `import sglang` 后能直接使用的接口定义。

## `launch_server.py` — 服务启动入口

```python
# python/sglang/launch_server.py
```

这是启动 SGLang 推理服务器的**入口文件**。核心流程：

1. 解析命令行参数，构造 `ServerArgs`
2. 调用 `srt.entrypoints.engine.Engine` 的启动逻辑
3. 根据配置启动 HTTP 服务器或 gRPC 服务器

关键函数：
- `launch_server()` — 同步启动服务器
- 内部调用 `Engine` 的 `connect()` 或 `__init__()` 完成初始化

> **注意**：实际的服务器启动逻辑在 `python/sglang/srt/entrypoints/engine.py` 中，`launch_server.py` 只是一个入口包装。

## `global_config.py` — 全局配置

```python
# python/sglang/global_config.py
```

定义了全局配置项，包括：

- `ENABLE_MOE` — 是否启用 MoE 支持
- `ENABLE_MLA` — 是否启用 MLA 注意力
- `VERBOSE` — 是否输出详细日志
- 其他开关和阈值配置

这些配置在进程启动时设置，影响整个运行时的行为。

## `version.py` — 版本信息

```python
# python/sglang/version.py
__version__ = "0.4.5"  # 示例版本号
```

定义了 `__version__` 字符串，是包版本的单一数据源（SSOT）。

## 包入口依赖关系

```
launch_server.py
    └── srt/entrypoints/engine.py
            ├── srt/managers/tokenizer_manager.py
            ├── srt/managers/scheduler.py
            ├── srt/managers/tp_worker.py
            └── srt/models/ → 模型加载
```

## 关键观察

1. **分层设计**：`sglang/__init__.py` 位于最上层，提供用户友好的接口；实际实现在 `sglang/srt/` 中深入嵌套
2. **入口明确**：`launch_server.py` 是服务端的唯一入口，参数集中管理在 `server_args.py`（约 380KB，是最大的单个文件）
3. **全局配置轻量**：`global_config.py` 只包含必要的布尔开关和阈值，不涉及运行时状态
