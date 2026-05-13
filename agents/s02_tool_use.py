#!/usr/bin/env python3
# Harness: tool dispatch -- expanding what the model can reach.
"""
s02_tool_use.py - Tools

The agent loop from s01 didn't change. We just added tools to the array
and a dispatch map to route calls.

    +----------+      +-------+      +------------------+
    |   User   | ---> |  LLM  | ---> | Tool Dispatch    |
    |  prompt  |      |       |      | {                |
    +----------+      +---+---+      |   bash: run_bash |
                          ^          |   read: run_read |
                          |          |   write: run_wr  |
                          +----------+   edit: run_edit |
                          tool_result| }                |
                                     +------------------+

Key insight: "The loop didn't change at all. I just added tools."
"""

import os
import subprocess
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic()
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int = None) -> str:
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# -- The dispatch map: {tool_name: handler} --
TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
]


def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                output = (
                    handler(**block.input) if handler else f"Unknown tool: {block.name}"
                )
                print(f"> {block.name}:")
                print(output[:200])
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output}
                )
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    history = []
    idx = 1
    while True:
        try:
            query = input("\033[36ms02 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        print(f"history{idx}: {history}")
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()


# 问题1: 阅读一下s02_tool_use 文件，看 是啥功能
# 回答：
# total 464
# drwxr-xr-x  17 ryanzhou  staff    544 May 12 11:37 .
# drwxr-xr-x  16 ryanzhou  staff    512 May 12 10:12 ..
# -rw-r--r--   1 ryanzhou  staff    281 May 12 11:28 .env
# -rw-r--r--   1 ryanzhou  st
# > read_file:
# #!/usr/bin/env python3
# # Harness: tool dispatch -- expanding what the model can reach.
# """
# s02_tool_use.py - Tools

# The agent loop from s01 didn't change. We just added tools to the array
# and a dispat
# history1: [{'role': 'user', 'content': '阅读一下s02_tool_us文件，是啥功能'}, {'role': 'assistant', 'content': [ThinkingBlock(signature='4fbd7110c0ca7d1c8dd52736c2584cfe2ea13a2d9e9db46e600e88395a23754a', thinking='用户想了解 s02_tool_us 文件的功能。让我先读取这个文件看看内容。', type='thinking'), ToolUseBlock(id='call_function_stu9fsyog07b_1', caller=None, input={'command': 'ls -la /Users/ryanzhou/Desktop/learn-claude-code/agents/'}, name='bash', type='tool_use')]}, {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_function_stu9fsyog07b_1', 'content': 'total 464\ndrwxr-xr-x  17 ryanzhou  staff    544 May 12 11:37 .\ndrwxr-xr-x  16 ryanzhou  staff    512 May 12 10:12 ..\n-rw-r--r--   1 ryanzhou  staff    281 May 12 11:28 .env\n-rw-r--r--   1 ryanzhou  staff    203 May 12 10:12 __init__.py\n-rw-r--r--   1 ryanzhou  staff  10468 May 12 13:52 s01_agent_loop.py\n-rw-r--r--   1 ryanzhou  staff   6151 May 12 17:50 s02_tool_use.py\n-rw-r--r--   1 ryanzhou  staff   8487 May 12 11:36 s03_todo_write.py\n-rw-r--r--   1 ryanzhou  staff   7856 May 12 11:36 s04_subagent.py\n-rw-r--r--   1 ryanzhou  staff   8484 May 12 11:36 s05_skill_loading.py\n-rw-r--r--   1 ryanzhou  staff  10105 May 12 11:36 s06_context_compact.py\n-rw-r--r--   1 ryanzhou  staff  10069 May 12 11:36 s07_task_system.py\n-rw-r--r--   1 ryanzhou  staff   9310 May 12 11:36 s08_background_tasks.py\n-rw-r--r--   1 ryanzhou  staff  16796 May 12 11:36 s09_agent_teams.py\n-rw-r--r--   1 ryanzhou  staff  20989 May 12 11:36 s10_team_protocols.py\n-rw-r--r--   1 ryanzhou  staff  24982 May 12 11:36 s11_autonomous_agents.py\n-rw-r--r--   1 ryanzhou  staff  25843 May 12 11:36 s12_worktree_task_isolation.py\n-rw-r--r--   1 ryanzhou  staff  36415 May 12 11:36 s_full.py'}]}, {'role': 'assistant', 'content': [ThinkingBlock(signature='3db21584c41fcaeef74f2d4387fc1ee89583471f8368d45c7e12d16316d5fc03', thinking='用户想让我阅读 s02_tool_us 文件，但是我看到文件名是 s02_tool_use.py。让我读取这个文件。', type='thinking'), ToolUseBlock(id='call_function_a40o6f79kbjy_1', caller=None, input={'path': '/Users/ryanzhou/Desktop/learn-claude-code/agents/s02_tool_use.py'}, name='read_file', type='tool_use')]}, {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_function_a40o6f79kbjy_1', 'content': '#!/usr/bin/env python3\n# Harness: tool dispatch -- expanding what the model can reach.\n"""\ns02_tool_use.py - Tools\n\nThe agent loop from s01 didn\'t change. We just added tools to the array\nand a dispatch map to route calls.\n\n    +----------+      +-------+      +------------------+\n    |   User   | ---> |  LLM  | ---> | Tool Dispatch    |\n    |  prompt  |      |       |      | {                |\n    +----------+      +---+---+      |   bash: run_bash |\n                          ^          |   read: run_read |\n                          |          |   write: run_wr  |\n                          +----------+   edit: run_edit |\n                          tool_result| }                |\n                                     +------------------+\n\nKey insight: "The loop didn\'t change at all. I just added tools."\n"""\n\nimport os\nimport subprocess\nfrom pathlib import Path\n\nfrom anthropic import Anthropic\nfrom dotenv import load_dotenv\n\nload_dotenv(override=True)\n\nif os.getenv("ANTHROPIC_BASE_URL"):\n    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)\n\nWORKDIR = Path.cwd()\nclient = Anthropic()\nMODEL = os.environ["MODEL_ID"]\n\nSYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don\'t explain."\n\n\ndef safe_path(p: str) -> Path:\n    path = (WORKDIR / p).resolve()\n    if not path.is_relative_to(WORKDIR):\n        raise ValueError(f"Path escapes workspace: {p}")\n    return path\n\n\ndef run_bash(command: str) -> str:\n    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]\n    if any(d in command for d in dangerous):\n        return "Error: Dangerous command blocked"\n    try:\n        r = subprocess.run(\n            command,\n            shell=True,\n            cwd=WORKDIR,\n            capture_output=True,\n            text=True,\n            timeout=120,\n        )\n        out = (r.stdout + r.stderr).strip()\n        return out[:50000] if out else "(no output)"\n    except subprocess.TimeoutExpired:\n        return "Error: Timeout (120s)"\n\n\ndef run_read(path: str, limit: int = None) -> str:\n    try:\n        text = safe_path(path).read_text()\n        lines = text.splitlines()\n        if limit and limit < len(lines):\n            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]\n        return "\\n".join(lines)[:50000]\n    except Exception as e:\n        return f"Error: {e}"\n\n\ndef run_write(path: str, content: str) -> str:\n    try:\n        fp = safe_path(path)\n        fp.parent.mkdir(parents=True, exist_ok=True)\n        fp.write_text(content)\n        return f"Wrote {len(content)} bytes to {path}"\n    except Exception as e:\n        return f"Error: {e}"\n\n\ndef run_edit(path: str, old_text: str, new_text: str) -> str:\n    try:\n        fp = safe_path(path)\n        content = fp.read_text()\n        if old_text not in content:\n            return f"Error: Text not found in {path}"\n        fp.write_text(content.replace(old_text, new_text, 1))\n        return f"Edited {path}"\n    except Exception as e:\n        return f"Error: {e}"\n\n\n# -- The dispatch map: {tool_name: handler} --\nTOOL_HANDLERS = {\n    "bash": lambda **kw: run_bash(kw["command"]),\n    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),\n    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),\n    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),\n}\n\nTOOLS = [\n    {\n        "name": "bash",\n        "description": "Run a shell command.",\n        "input_schema": {\n            "type": "object",\n            "properties": {"command": {"type": "string"}},\n            "required": ["command"],\n        },\n    },\n    {\n        "name": "read_file",\n        "description": "Read file contents.",\n        "input_schema": {\n            "type": "object",\n            "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},\n            "required": ["path"],\n        },\n    },\n    {\n        "name": "write_file",\n        "description": "Write content to file.",\n        "input_schema": {\n            "type": "object",\n            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},\n            "required": ["path", "content"],\n        },\n    },\n    {\n        "name": "edit_file",\n        "description": "Replace exact text in file.",\n        "input_schema": {\n            "type": "object",\n            "properties": {\n                "path": {"type": "string"},\n                "old_text": {"type": "string"},\n                "new_text": {"type": "string"},\n            },\n            "required": ["path", "old_text", "new_text"],\n        },\n    },\n]\n\n\ndef agent_loop(messages: list):\n    while True:\n        response = client.messages.create(\n            model=MODEL,\n            system=SYSTEM,\n            messages=messages,\n            tools=TOOLS,\n            max_tokens=8000,\n        )\n        messages.append({"role": "assistant", "content": response.content})\n        if response.stop_reason != "tool_use":\n            return\n        results = []\n        for block in response.content:\n            if block.type == "tool_use":\n                handler = TOOL_HANDLERS.get(block.name)\n                output = (\n                    handler(**block.input) if handler else f"Unknown tool: {block.name}"\n                )\n                print(f"> {block.name}:")\n                print(output[:200])\n                results.append(\n                    {"type": "tool_result", "tool_use_id": block.id, "content": output}\n                )\n        messages.append({"role": "user", "content": results})\n\n\nif __name__ == "__main__":\n    history = []\n    idx = 1\n    while True:\n        try:\n            query = input("\\033[36ms02 >> \\033[0m")\n        except (EOFError, KeyboardInterrupt):\n            break\n        if query.strip().lower() in ("q", "exit", ""):\n            break\n        history.append({"role": "user", "content": query})\n        agent_loop(history)\n        print(f"history{idx}: {history}")\n        response_content = history[-1]["content"]\n        if isinstance(response_content, list):\n            for block in response_content:\n                if hasattr(block, "text"):\n                    print(block.text)\n        print()'}]}, {'role': 'assistant', 'content': [ThinkingBlock(signature='fcb60b3e20c77441f71b0fc8426bbf4cdcf860274b59873e51977c761a8227a9', thinking='这是一个关于 Claude Code 中 Tool Use 功能的示例文件。让我总结一下它的功能。', type='thinking'), TextBlock(citations=None, text='这是一个**Agent 工具调用系统**的示例文件，展示了如何让 AI Agent 使用工具来完成任务。\n\n## 主要功能\n\n### 1. 架构概览\n```\n用户 → LLM → 工具调度 → 返回结果 → LLM\n```\n\n核心思路：**Agent 循环不变，只是增加了工具**\n\n### 2. 提供 4 个工具\n\n| 工具 | 功能 | 用途 |\n|------|------|------|\n| `bash` | 执行 shell 命令 | 运行系统命令 |\n| `read_file` | 读取文件 | 查看文件内容 |\n| `write_file` | 写入文件 | 创建/覆写文件 |\n| `edit_file` | 编辑文件 | 替换指定文本 |\n\n### 3. 安全机制\n- `safe_path()`: 防止路径穿越攻击，确保只能访问工作目录内的文件\n- `bash`: 阻止危险命令（如 `rm -rf /`、`sudo`、`shutdown` 等）\n- 命令超时限制 120 秒\n\n### 4. Agent 循环逻辑\n1. 用户输入 → 发给 LLM\n2. LLM 返回 `tool_use` 调用时 → 分发到对应 handler\n3. handler 执行并返回结果\n4. 结果发回 LLM 继续生成\n\n## 运行方式\n```bash\npython s02_tool_use.py\n```\n进入交互模式，提示符为 `s02 >>`\n\n这是 Claude Code 框架的**基础工具系统**，后续章节会基于此扩展更复杂的功能。', type='text')]}]
# 这是一个**Agent 工具调用系统**的示例文件，展示了如何让 AI Agent 使用工具来完成任务。

# ## 主要功能

# ### 1. 架构概览
# ```
# 用户 → LLM → 工具调度 → 返回结果 → LLM
# ```

# 核心思路：**Agent 循环不变，只是增加了工具**

# ### 2. 提供 4 个工具

# | 工具 | 功能 | 用途 |
# |------|------|------|
# | `bash` | 执行 shell 命令 | 运行系统命令 |
# | `read_file` | 读取文件 | 查看文件内容 |
# | `write_file` | 写入文件 | 创建/覆写文件 |
# | `edit_file` | 编辑文件 | 替换指定文本 |

# ### 3. 安全机制
# - `safe_path()`: 防止路径穿越攻击，确保只能访问工作目录内的文件
# - `bash`: 阻止危险命令（如 `rm -rf /`、`sudo`、`shutdown` 等）
# - 命令超时限制 120 秒

# ### 4. Agent 循环逻辑
# 1. 用户输入 → 发给 LLM
# 2. LLM 返回 `tool_use` 调用时 → 分发到对应 handler
# 3. handler 执行并返回结果
# 4. 结果发回 LLM 继续生成

# ## 运行方式
# ```bash
# python s02_tool_use.py
# ```
# 进入交互模式，提示符为 `s02 >>`

# 这是 Claude Code 框架的**基础工具系统**，后续章节会基于此扩展更复杂的功能。
