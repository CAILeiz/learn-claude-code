#!/usr/bin/env python3
# Harness: context isolation -- protecting the model's clarity of thought.
"""
s04_subagent.py - Subagents

Spawn a child agent with fresh messages=[]. The child works in its own
context, sharing the filesystem, then returns only a summary to the parent.

    Parent agent                     Subagent
    +------------------+             +------------------+
    | messages=[...]   |             | messages=[]      |  <-- fresh
    |                  |  dispatch   |                  |
    | tool: task       | ---------->| while tool_use:  |
    |   prompt="..."   |            |   call tools     |
    |   description="" |            |   append results |
    |                  |  summary   |                  |
    |   result = "..." | <--------- | return last text |
    +------------------+             +------------------+
              |
    Parent context stays clean.
    Subagent context is discarded.

Key insight: "Process isolation gives context isolation for free."
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

SYSTEM = f"You are a coding agent at {WORKDIR}. Use the task tool to delegate exploration or subtasks."
SUBAGENT_SYSTEM = f"You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."


# -- Tool implementations shared by parent and child --
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
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
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


TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

# Child gets all base tools except task (no recursive spawning)
CHILD_TOOLS = [
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


# -- Subagent: fresh context, filtered tools, summary-only return --
def run_subagent(prompt: str) -> str:
    sub_messages = [{"role": "user", "content": prompt}]  # fresh context
    for _ in range(30):  # safety limit
        response = client.messages.create(
            model=MODEL,
            system=SUBAGENT_SYSTEM,
            messages=sub_messages,
            tools=CHILD_TOOLS,
            max_tokens=8000,
        )
        sub_messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                output = (
                    handler(**block.input) if handler else f"Unknown tool: {block.name}"
                )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(output)[:50000],
                    }
                )
        sub_messages.append({"role": "user", "content": results})
    # Only the final text returns to the parent -- child context is discarded
    return (
        "".join(b.text for b in response.content if hasattr(b, "text"))
        or "(no summary)"
    )


# -- Parent tools: base tools + task dispatcher --
PARENT_TOOLS = CHILD_TOOLS + [
    {
        "name": "task",
        "description": "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "description": {
                    "type": "string",
                    "description": "Short description of the task",
                },
            },
            "required": ["prompt"],
        },
    },
]


def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=PARENT_TOOLS,
            max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "task":
                    desc = block.input.get("description", "subtask")
                    prompt = block.input.get("prompt", "")
                    print(f"> task ({desc}): {prompt[:80]}")
                    output = run_subagent(prompt)
                else:
                    handler = TOOL_HANDLERS.get(block.name)
                    output = (
                        handler(**block.input)
                        if handler
                        else f"Unknown tool: {block.name}"
                    )
                print(f"  {str(output)[:200]}")
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(output),
                    }
                )
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    history = []
    idx = 0
    while True:
        try:
            query = input("\033[36ms04 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        print(f"history{idx}: {history}")
        idx += 1
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()


# 问题：阅读s05_skill_loading.py，要求：必须 使用子 agent
# 回答：
# > task (阅读s05_skill_ly文件): 请阅读 /Users/ryanzhou/Desktop/learn-claude-code/agents/s05_skill_ly 文件的内容，并返回该文件的完
#   文件 `/Users/ryanzhou/Desktop/learn-claude-code/agents/s05_skill_ly` 不存在，但我找到了一个类似的文件 `s05_skill_loading.py`。以下是它的完整内容：

# ```python
# #!/usr/bin/env python3
# # Harness: on-demand knowledge -- domain experti
# history0: [{'role': 'user', 'content': '阅读s05_skill_ly，要求：必须使用子 agent '}, {'role': 'assistant', 'content': [ThinkingBlock(signature='c5086d69415f5acfb7218102b816b821c1e47f01f02824f5e1832227ad942153', thinking='用户要求我阅读s05_skill_ly文件，并明确要求必须使用子agent来完成这个任务。让我先使用task工具来创建一个子agent来阅读这个文件。', type='thinking'), ToolUseBlock(id='call_function_xq7g2ava9lz6_1', caller=None, input={'description': '阅读s05_skill_ly文件', 'prompt': '请阅读 /Users/ryanzhou/Desktop/learn-claude-code/agents/s05_skill_ly 文件的内容，并返回该文件的完整内容。'}, name='task', type='tool_use')]}, {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_function_xq7g2ava9lz6_1', 'content': '文件 `/Users/ryanzhou/Desktop/learn-claude-code/agents/s05_skill_ly` 不存在，但我找到了一个类似的文件 `s05_skill_loading.py`。以下是它的完整内容：\n\n```python\n#!/usr/bin/env python3\n# Harness: on-demand knowledge -- domain expertise, loaded when the model asks.\n"""\ns05_skill_loading.py - Skills\n\nTwo-layer skill injection that avoids bloating the system prompt:\n\n    Layer 1 (cheap): skill names in system prompt (~100 tokens/skill)\n    Layer 2 (on demand): full skill body in tool_result\n\n    skills/\n      pdf/\n        SKILL.md          <-- frontmatter (name, description) + body\n      code-review/\n        SKILL.md\n\n    System prompt:\n    +--------------------------------------+\n    | You are a coding agent.              |\n    | Skills available:                    |\n    |   - pdf: Process PDF files...        |  <-- Layer 1: metadata only\n    |   - code-review: Review code...      |\n    +--------------------------------------+\n\n    When model calls load_skill("pdf"):\n    +--------------------------------------+\n    | tool_result:                         |\n    | <skill>                              |\n    |   Full PDF processing instructions   |  <-- Layer 2: full body\n    |   Step 1: ...                        |\n    |   Step 2: ...                        |\n    | </skill>                             |\n    +--------------------------------------+\n\nKey insight: "Don\'t put everything in the system prompt. Load on demand."\n"""\n\nimport os\nimport re\nimport subprocess\nimport yaml\nfrom pathlib import Path\n\nfrom anthropic import Anthropic\nfrom dotenv import load_dotenv\n\nload_dotenv(override=True)\n\nif os.getenv("ANTHROPIC_BASE_URL"):\n    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)\n\nWORKDIR = Path.cwd()\nclient = Anthropic()\nMODEL = os.environ["MODEL_ID"]\nSKILLS_DIR = WORKDIR / "skills"\n\n\n# -- SkillLoader: scan skills/<name>/SKILL.md with YAML frontmatter --\nclass SkillLoader:\n    def __init__(self, skills_dir: Path):\n        self.skills_dir = skills_dir\n        self.skills = {}\n        self._load_all()\n\n    def _load_all(self):\n        if not self.skills_dir.exists():\n            return\n        for f in sorted(self.skills_dir.rglob("SKILL.md")):\n            text = f.read_text()\n            meta, body = self._parse_frontmatter(text)\n            name = meta.get("name", f.parent.name)\n            self.skills[name] = {"meta": meta, "body": body, "path": str(f)}\n\n    def _parse_frontmatter(self, text: str) -> tuple:\n        """Parse YAML frontmatter between --- delimiters."""\n        match = re.match(r"^---\\n(.*?)\\n---\\n(.*)", text, re.DOTALL)\n        if not match:\n            return {}, text\n        try:\n            meta = yaml.safe_load(match.group(1)) or {}\n        except yaml.YAMLError:\n            meta = {}\n        return meta, match.group(2).strip()\n\n    def get_descriptions(self) -> str:\n        """Layer 1: short descriptions for the system prompt."""\n        if not self.skills:\n            return "(no skills available)"\n        lines = []\n        for name, skill in self.skills.items():\n            desc = skill["meta"].get("description", "No description")\n            tags = skill["meta"].get("tags", "")\n            line = f"  - {name}: {desc}"\n            if tags:\n                line += f" [{tags}]"\n            lines.append(line)\n        return "\\n".join(lines)\n\n    def get_content(self, name: str) -> str:\n        """Layer 2: full skill body returned in tool_result."""\n        skill = self.skills.get(name)\n        if not skill:\n            return f"Error: Unknown skill \'{name}\'. Available: {\', \'.join(self.skills.keys())}"\n        return f"<skill name=\\"{name}\\">\\n{skill[\'body\']}\\n</skill>"\n\n\nSKILL_LOADER = SkillLoader(SKILLS_DIR)\n\n# Layer 1: skill metadata injected into system prompt\nSYSTEM = f"""You are a coding agent at {WORKDIR}.\nUse load_skill to access specialized knowledge before tackling unfamiliar topics.\n\nSkills available:\n{SKILL_LOADER.get_descriptions()}"""\n\n\n# -- Tool implementations --\ndef safe_path(p: str) -> Path:\n    path = (WORKDIR / p).resolve()\n    if not path.is_relative_to(WORKDIR):\n        raise ValueError(f"Path escapes workspace: {p}")\n    return path\n\n\ndef run_bash(command: str) -> str:\n    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]\n    if any(d in command for d in dangerous):\n        return "Error: Dangerous command blocked"\n    try:\n        r = subprocess.run(\n            command,\n            shell=True,\n            cwd=WORKDIR,\n            capture_output=True,\n            text=True,\n            timeout=120,\n        )\n        out = (r.stdout + r.stderr).strip()\n        return out[:50000] if out else "(no output)"\n    except subprocess.TimeoutExpired:\n        return "Error: Timeout (120s)"\n\n\ndef run_read(path: str, limit: int = None) -> str:\n    try:\n        lines = safe_path(path).read_text().splitlines()\n        if limit and limit < len(lines):\n            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]\n        return "\\n".join(lines)[:50000]\n    except Exception as e:\n        return f"Error: {e}"\n\n\ndef run_write(path: str, content: str) -> str:\n    try:\n        fp = safe_path(path)\n        fp.parent.mkdir(parents=True, exist_ok=True)\n        fp.write_text(content)\n        return f"Wrote {len(content)} bytes"\n    except Exception as e:\n        return f"Error: {e}"\n\n\ndef run_edit(path: str, old_text: str, new_text: str) -> str:\n    try:\n        fp = safe_path(path)\n        content = fp.read_text()\n        if old_text not in content:\n            return f"Error: Text not found in {path}"\n        fp.write_text(content.replace(old_text, new_text, 1))\n        return f"Edited {path}"\n    except Exception as e:\n        return f"Error: {e}"\n\n\nTOOL_HANDLERS = {\n    "bash": lambda **kw: run_bash(kw["command"]),\n    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),\n    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),\n    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),\n    "load_skill": lambda **kw: SKILL_LOADER.get_content(kw["name"]),\n}\n\nTOOLS = [\n    {\n        "name": "bash",\n        "description": "Run a shell command.",\n        "input_schema": {\n            "type": "object",\n            "properties": {"command": {"type": "string"}},\n            "required": ["command"],\n        },\n    },\n    {\n        "name": "read_file",\n        "description": "Read file contents.",\n        "input_schema": {\n            "type": "object",\n            "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},\n            "required": ["path"],\n        },\n    },\n    {\n        "name": "write_file",\n        "description": "Write content to file.",\n        "input_schema": {\n            "type": "object",\n            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},\n            "required": ["path", "content"],\n        },\n    },\n    {\n        "name": "edit_file",\n        "description": "Replace exact text in file.",\n        "input_schema": {\n            "type": "object",\n            "properties": {\n                "path": {"type": "string"},\n                "old_text": {"type": "string"},\n                "new_text": {"type": "string"},\n            },\n            "required": ["path", "old_text", "new_text"],\n        },\n    },\n    {\n        "name": "load_skill",\n        "description": "Load specialized knowledge by name.",\n        "input_schema": {\n            "type": "object",\n            "properties": {\n                "name": {"type": "string", "description": "Skill name to load"}\n            },\n            "required": ["name"],\n        },\n    },\n]\n\n\ndef agent_loop(messages: list):\n    while True:\n        response = client.messages.create(\n            model=MODEL,\n            system=SYSTEM,\n            messages=messages,\n            tools=TOOLS,\n            max_tokens=8000,\n        )\n        messages.append({"role": "assistant", "content": response.content})\n        if response.stop_reason != "tool_use":\n            return\n        results = []\n        for block in response.content:\n            if block.type == "tool_use":\n                handler = TOOL_HANDLERS.get(block.name)\n                try:\n                    output = (\n                        handler(**block.input)\n                        if handler\n                        else f"Unknown tool: {block.name}"\n                    )\n                except Exception as e:\n                    output = f"Error: {e}"\n                print(f"> {block.name}:")\n                print(str(output)[:200])\n                results.append(\n                    {\n                        "type": "tool_result",\n                        "tool_use_id": block.id,\n                        "content": str(output),\n                    }\n                )\n        messages.append({"role": "user", "content": results})\n\n\nif __name__ == "__main__":\n    history = []\n    while True:\n        try:\n            query = input("\\033[36ms05 >> \\033[0m")\n        except (EOFError, KeyboardInterrupt):\n            break\n        if query.strip().lower() in ("q", "exit", ""):\n            break\n        history.append({"role": "user", "content": query})\n        agent_loop(history)\n        response_content = history[-1]["content"]\n        if isinstance(response_content, list):\n            for block in response_content:\n                if hasattr(block, "text"):\n                    print(block.text)\n        print()\n```\n\n---\n\n**摘要**: 这是一个名为 `s05_skill_loading.py` 的 Claude Agent 代码样例，展示了**双层技能加载机制**：\n\n1. **Layer 1 (便宜)**: 系统提示中只包含技能名称和描述（每个技能约 100 tokens）\n2. **Layer 2 (按需)**: 通过 `load_skill` 工具调用时才加载完整的技能内容\n\n该代码实现了：\n- `SkillLoader` 类：解析 `skills/<name>/SKILL.md` 文件（包含 YAML frontmatter）\n- 工具函数：`bash`, `read_file`, `write_file`, `edit_file`, `load_skill`\n- 交互式 Agent 循环：持续对话并处理工具调用'}]}, {'role': 'assistant', 'content': [ThinkingBlock(signature='8963ebe247ba863598eb9391bedcd7104fcadbcda5d83480b6c3fd7918359895', thinking='用户要求使用子agent来阅读s05_skill_ly文件，我已经成功调用了一个子agent来读取该文件。子agent返回了结果，显示该文件不存在，但找到了一个类似的文件s05_skill_loading.py。子agent返回了完整的文件内容并做了总结。\n\n让我总结一下返回的内容给用户。', type='thinking'), TextBlock(citations=None, text='子 agent 已完成任务！以下是读取结果：\n\n## s05_skill_loading.py 内容\n\n这是 Claude Agent 的技能加载机制示例代码，文件位于 `/Users/ryanzhou/Desktop/learn-claude-code/agents/` 目录下。\n\n### 核心概念：双层技能加载\n\n| 层 | 用途 | 位置 |\n|---|---|---|\n| **Layer 1** | 元数据（名称+描述）放入系统提示 | 每个技能约 100 tokens |\n| **Layer 2** | 完整技能内容通过 `load_skill` 工具调用 | 按需加载 |\n\n### 目录结构\n```\nskills/\n  pdf/SKILL.md          # YAML frontmatter + 内容\n  code-review/SKILL.md\n```\n\n### 关键组件\n1. **SkillLoader** 类 - 解析 YAML frontmatter 并管理技能\n2. **工具函数** - `bash`, `read_file`, `write_file`, `edit_file`, `load_skill`\n3. **交互式 Agent 循环** - 持续对话处理工具调用\n\n### 核心优势\n> "Don\'t put everything in the system prompt. Load on demand."\n> (不要把所有东西都放在系统提示里，按需加载)\n\n这个设计避免了在系统提示中塞入大量技能文档，而是将技能信息分为两层，只有在需要时才加载完整内容，从而保持系统提示的精简。', type='text')]}]
# 子 agent 已完成任务！以下是读取结果：

# ## s05_skill_loading.py 内容

# 这是 Claude Agent 的技能加载机制示例代码，文件位于 `/Users/ryanzhou/Desktop/learn-claude-code/agents/` 目录下。

# ### 核心概念：双层技能加载

# | 层 | 用途 | 位置 |
# |---|---|---|
# | **Layer 1** | 元数据（名称+描述）放入系统提示 | 每个技能约 100 tokens |
# | **Layer 2** | 完整技能内容通过 `load_skill` 工具调用 | 按需加载 |

# ### 目录结构
# ```
# skills/
#   pdf/SKILL.md          # YAML frontmatter + 内容
#   code-review/SKILL.md
# ```

# ### 关键组件
# 1. **SkillLoader** 类 - 解析 YAML frontmatter 并管理技能
# 2. **工具函数** - `bash`, `read_file`, `write_file`, `edit_file`, `load_skill`
# 3. **交互式 Agent 循环** - 持续对话处理工具调用

# ### 核心优势
# > "Don't put everything in the system prompt. Load on demand."
# > (不要把所有东西都放在系统提示里，按需加载)

# 这个设计避免了在系统提示中塞入大量技能文档，而是将技能信息分为两层，只有在需要时才加载完整内容，从而保持系统提示的精简。
