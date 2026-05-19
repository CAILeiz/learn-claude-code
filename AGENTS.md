# Learn Claude Code -- AGENTS 总结

## 核心概念：Agent = Model + Harness

**Agency（智能性）来自模型训练，而非外部代码编排。**

Agent 产品 = 模型 + Harness（驾驭系统）
- **模型**：决定者、推理者、驾驶员
- **Harness**：工具 + 知识 + 观察 + 操作接口 + 权限

```
Harness = Tools + Knowledge + Observation + Action Interfaces + Permissions
```

---

## Agent 模式图

```
User --> messages[] --> LLM --> response
                               |
                     stop_reason == "tool_use"?
                    /                          \
                  yes                           no
                   |                             |
             execute tools                  return text
             append results
             loop back -----------------> messages[]
```

---

## 12 阶段渐进学习路径

| Session | 主题 | 核心理念 | 新增工具数 |
|---------|------|----------|------------|
| **s01** | Agent Loop | *One loop & Bash is all you need* | 1 |
| **s02** | Tool Use | *Adding a tool means adding one handler* | +2 |
| **s03** | TodoWrite | *An agent without a plan drifts* | +3 |
| **s04** | Subagents | *Break big tasks down; each subtask gets a clean context* | +2 |
| **s05** | Skills | *Load knowledge when you need it, not upfront* | +2 |
| **s06** | Context Compact | *Context will fill up; you need a way to make room* | +1 |
| **s07** | Tasks | *Break big goals into small tasks, order them, persist to disk* | +4 |
| **s08** | Background Tasks | *Run slow operations in the background; the agent keeps thinking* | +1 |
| **s09** | Agent Teams | *When the task is too big for one, delegate to teammates* | +3 |
| **s10** | Team Protocols | *Teammates need shared communication rules* | +3 |
| **s11** | Autonomous Agents | *Teammates scan the board and claim tasks themselves* | +1 |
| **s12** | Worktree + Task Isolation | *Each works in its own directory, no interference* | +6 |

---

## 架构概览

```
learn-claude-code/
|
|-- agents/              # Python 参考实现 (s01-s12 + s_full)
|-- docs/{en,zh,ja}/     # 思维模型优先的文档 (3种语言)
|-- web/                  # 交互式学习平台 (Next.js)
|-- skills/               # Skill 文件 (s05)
+-- .github/workflows/    # CI: typecheck + build
```

---

## 关键机制详解

### s01-s06: 基础能力构建

- **s01 Agent Loop**: 核心循环，`while True` + `stop_reason` 控制
- **s02 Tool Use**: 工具注册到 dispatch map，按名称路由
- **s03 TodoWrite**: 先列步骤再执行，效率翻倍
- **s04 Subagents**: 子 agent 独立 `messages[]`，主对话保持干净
- **s05 Skills**: 通过 `tool_result` 注入知识，而非 system prompt
- **s06 Context Compact**: 三层压缩策略，支持无限会话

### s07-s12: 团队与持久化

- **s07 Tasks**: 文件持久化任务图，支持依赖关系（DAG）
- **s08 Background Tasks**: 守护线程执行慢操作，完成时注入通知
- **s09 Agent Teams**: 持久化队友 + JSONL 邮箱通信
- **s10 Team Protocols**: 共享通信规则（shutdown、计划审批 FSM）
- **s11 Autonomous Agents**: 队友自主扫描任务板并认领
- **s12 Worktree Isolation**: 任务管理目标，worktree 管理目录，绑定 ID

---

## 团队协作模式

```
.teammate/
  config.json           <- 团队名册 + 状态
  inbox/
    alice.jsonl         <- 追加式只读，读取后清空
    bob.jsonl
    lead.jsonl

生命周期:
  spawn -> WORKING -> IDLE -> WORKING -> ... -> SHUTDOWN
```

---

## 快速开始

```bash
git clone https://github.com/shareAI-lab/learn-claude-code
cd learn-claude-code
pip install -r requirements.txt
cp .env.example .env   # 填入 ANTHROPIC_API_KEY

python agents/s01_agent_loop.py       # 从这里开始
python agents/s12_worktree_task_isolation.py  # 完整进度终点
python agents/s_full.py               # 集大成者：所有机制合一
```

---

## 延伸项目

- **Kode Agent CLI**: `npm i -g @shareai-lab/kode` - 开源 Coding Agent CLI
- **Kode Agent SDK**: 嵌入式 Agent 能力库，无单用户进程开销
- **claw0**: 从"按需会话"到"常驻助手"，心跳 + 定时 + IM 通道 + 记忆 + 灵魂

---

## 核心教训

> **Agency comes from the model. The harness makes agency real.**
> 
> **Build great harnesses. The model will do the rest.**
> 
> **Bash is all you need. Real agents are all the universe needs.**