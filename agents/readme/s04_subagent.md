# s04_subagent.py - 子代理（Subagent）

## 核心概念

子代理机制通过**进程隔离**实现**上下文隔离**。子代理在独立的对话上下文中工作，完成任务后只返回摘要给父代理，从而保护父代理的上下文清晰度。

## 架构图

```
    父代理 (Parent Agent)                 子代理 (Subagent)
    +------------------+                  +------------------+
    | messages=[...]   |                  | messages=[]      |  <-- 全新上下文
    |                  |    dispatch      |                  |
    | 工具: task       | --------------> |  while tool_use: |
    |   prompt="..."   |                  |    call tools    |
    |   description="" |                  |    append results|
    |                  |    summary       |                  |
    |  result = "..."  | <-------------  |  return last text|
    +------------------+                  +------------------+
              |
    父代理上下文保持干净
    子代理上下文被丢弃
```

## 关键特性

### 1. 上下文隔离
- 子代理拥有全新的 `messages=[]`
- 子代理不会继承父代理的对话历史
- 只有最终文本摘要返回给父代理

### 2. 安全工具集
子代理拥有精简的工具集（不包括 task 工具，防止递归嵌套）：
- `bash`: 执行shell命令
- `read_file`: 读取文件
- `write_file`: 写入文件
- `edit_file`: 编辑文件

### 3. 安全防护
- 路径安全检查：`safe_path()` 确保文件操作不超出工作目录
- 危险命令拦截：阻止 `rm -rf /`、`sudo`、`shutdown` 等高危命令
- 执行超时限制：120秒超时

## 核心函数

### `run_subagent(prompt: str) -> str`
```python
def run_subagent(prompt: str) -> str:
    sub_messages = [{"role": "user", "content": prompt}]  # 全新上下文
    for _ in range(30):  # 安全限制
        # 调用API处理任务
        ...
    # 只有最终文本返回给父代理
    return "".join(b.text for b in response.content ...)
```

### `agent_loop(messages: list)`
主代理循环，处理父代理的工具调用。当调用 `task` 工具时，触发子代理执行。

## 使用方式

```bash
python s04_subagent.py
```

启动后可以输入任务，子代理将帮助完成并在完成后返回摘要。

## 核心洞察

> **"进程隔离为上下文隔离提供了免费方案"**
> 
> 通过为每个子代理创建独立进程，可以自然地实现上下文隔离，无需复杂的上下文管理逻辑。

## 文件结构

| 组件 | 描述 |
|------|------|
| `TOOL_HANDLERS` | 工具处理器映射表 |
| `CHILD_TOOLS` | 子代理可用的工具定义 |
| `PARENT_TOOLS` | 父代理工具集（基础工具 + task） |
| `run_subagent()` | 创建并运行子代理的核心函数 |
| `agent_loop()` | 父代理主循环 |