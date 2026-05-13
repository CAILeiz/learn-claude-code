#!/usr/bin/env python3
# Harness: planning -- keeping the model on course without scripting the route.
"""
s03_todo_write.py - TodoWrite

The model tracks its own progress via a TodoManager. A nag reminder
forces it to keep updating when it forgets.

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> | Tools   |
    |  prompt  |      |       |      | + todo  |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                                |
                    +-----------+-----------+
                    | TodoManager state     |
                    | [ ] task A            |
                    | [>] task B <- doing   |
                    | [x] task C            |
                    +-----------------------+
                                |
                    if rounds_since_todo >= 3:
                      inject <reminder>

Key insight: "The agent can track its own progress -- and I can see it."
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

SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use the todo tool to plan multi-step tasks. Mark in_progress before starting, completed when done.
Prefer tools over prose."""


# -- TodoManager: structured state the LLM writes to --
class TodoManager:
    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")
        validated = []
        in_progress_count = 0
        for i, item in enumerate(items):
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(i + 1)))
            if not text:
                raise ValueError(f"Item {item_id}: text required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"id": item_id, "text": text, "status": status})
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")
        self.items = validated
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."
        lines = []
        for item in self.items:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[
                item["status"]
            ]
            lines.append(f"{marker} #{item['id']}: {item['text']}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)


TODO = TodoManager()


# -- Tool implementations --
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
    "todo": lambda **kw: TODO.update(kw["items"]),
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
    {
        "name": "todo",
        "description": "Update task list. Track progress on multi-step tasks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "text": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                        },
                        "required": ["id", "text", "status"],
                    },
                }
            },
            "required": ["items"],
        },
    },
]


# -- Agent loop with nag reminder injection --
def agent_loop(messages: list):
    rounds_since_todo = 0
    while True:
        # Nag reminder is injected below, alongside tool results
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
        used_todo = False
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = (
                        handler(**block.input)
                        if handler
                        else f"Unknown tool: {block.name}"
                    )
                except Exception as e:
                    output = f"Error: {e}"
                print(f"> {block.name}:")
                print(str(output)[:200])
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(output),
                    }
                )
                if block.name == "todo":
                    used_todo = True
        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
        if rounds_since_todo >= 3:
            results.append(
                {"type": "text", "text": "<reminder>Update your todos.</reminder>"}
            )
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    history = []
    idx = 0
    while True:
        try:
            query = input("\033[36ms03 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        print(f"history{idx}：{history}")
        idx += 1
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()


# 问题：帮我分析一下04 到05文件的功能，输出到  ./readme/, 每个文件需要有一个说明文件
# 回答：
# > bash:
# total 488
# -rw-r--r--   1 ryanzhou  staff    203  5月 12 10:12 __init__.py
# drwxr-xr-x  17 ryanzhou  staff    544  5月 13 11:34 .
# drwxr-xr-x  16 ryanzhou  staff    512  5月 12 10:12 ..
# -rw-r--r--   1 ryanz
# > read_file:
# #!/usr/bin/env python3
# # Harness: context isolation -- protecting the model's clarity of thought.
# """
# s04_subagent.py - Subagents

# Spawn a child agent with fresh messages=[]. The child works in its ow
# > read_file:
# #!/usr/bin/env python3
# # Harness: on-demand knowledge -- domain expertise, loaded when the model asks.
# """
# s05_skill_loading.py - Skills

# Two-layer skill injection that avoids bloating the system prom
# > bash:
# (no output)
# > todo:
# [>] #1: 分析 s04_subagent.py 文件功能
# [ ] #2: 分析 s05_skill_loading.py 文件功能
# [ ] #3: 创建 readme/s04_subagent.md
# [ ] #4: 创建 readme/s05_skill_loading.md

# (0/4 completed)
# > write_file:
# Wrote 1877 bytes
# > todo:
# [x] #1: 分析 s04_subagent.py 文件功能
# [>] #2: 分析 s05_skill_loading.py 文件功能
# [x] #3: 创建 readme/s04_subagent.md
# [ ] #4: 创建 readme/s05_skill_loading.md

# (2/4 completed)
# > write_file:
# Wrote 2652 bytes
# > todo:
# [x] #1: 分析 s04_subagent.py 文件功能
# [x] #2: 分析 s05_skill_loading.py 文件功能
# [x] #3: 创建 readme/s04_subagent.md
# [x] #4: 创建 readme/s05_skill_loading.md

# (4/4 completed)
# > bash:
# total 16
# drwxr-xr-x   4 ryanzhou  staff   128  5月 13 11:39 .
# drwxr-xr-x  18 ryanzhou  staff   576  5月 13 11:37 ..
# -rw-r--r--   1 ryanzhou  staff  2809  5月 13 11:37 s04_subagent.md
# -rw-r--r--   1 ryanz
# history0：[{'role': 'user', 'content': '帮我分析一下4到05文件的功能，输出到 ./readme/, 每个文件'}, {'role': 'assistant', 'content': [ThinkingBlock(signature='6c5da04efd0a1968a79793d4b8602ed9e9aa8cfa898f399445b54b9cfb5c300e', thinking='用户想让我分析4到05文件的功能，并将结果输出到 ./readme/ 目录，每个文件一个说明。\n\n首先，我需要：\n1. 找到这些文件（可能是当前目录下的文件）\n2. 读取每个文件的内容\n3. 分析其功能\n4. 创建 ./readme/ 目录\n5. 为每个文件创建相应的说明文档\n\n让我先看看当前目录有什么文件。', type='thinking'), TextBlock(citations=None, text='我来先看看当前目录有哪些文件，然后分析它们的功能。\n', type='text'), ToolUseBlock(id='call_function_mqjycerra7cf_1', caller=None, input={'command': 'ls -la'}, name='bash', type='tool_use')]}, {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_function_mqjycerra7cf_1', 'content': 'total 488\n-rw-r--r--   1 ryanzhou  staff    203  5月 12 10:12 __init__.py\ndrwxr-xr-x  17 ryanzhou  staff    544  5月 13 11:34 .\ndrwxr-xr-x  16 ryanzhou  staff    512  5月 12 10:12 ..\n-rw-r--r--   1 ryanzhou  staff    281  5月 12 11:28 .env\n-rw-r--r--   1 ryanzhou  staff  36415  5月 12 11:36 s_full.py\n-rw-r--r--   1 ryanzhou  staff  10468  5月 12 13:52 s01_agent_loop.py\n-rw-r--r--   1 ryanzhou  staff  18148  5月 13 09:46 s02_tool_use.py\n-rw-r--r--   1 ryanzhou  staff   9748  5月 13 11:21 s03_todo_write.py\n-rw-r--r--   1 ryanzhou  staff   7856  5月 12 11:36 s04_subagent.py\n-rw-r--r--   1 ryanzhou  staff   8484  5月 12 11:36 s05_skill_loading.py\n-rw-r--r--   1 ryanzhou  staff  10105  5月 12 11:36 s06_context_compact.py\n-rw-r--r--   1 ryanzhou  staff  10069  5月 12 11:36 s07_task_system.py\n-rw-r--r--   1 ryanzhou  staff   9310  5月 12 11:36 s08_background_tasks.py\n-rw-r--r--   1 ryanzhou  staff  16796  5月 12 11:36 s09_agent_teams.py\n-rw-r--r--   1 ryanzhou  staff  20989  5月 12 11:36 s10_team_protocols.py\n-rw-r--r--   1 ryanzhou  staff  24982  5月 12 11:36 s11_autonomous_agents.py\n-rw-r--r--   1 ryanzhou  staff  25843  5月 12 11:36 s12_worktree_task_isolation.py'}]}, {'role': 'assistant', 'content': [ThinkingBlock(signature='9d21e09f934d990402b125774db8496adbec6d8f7385570eddfe2bfa1c8f455c', thinking='我看到有 s04_subagent.py 和 s05_skill_loading.py 两个文件。用户说的是"4到05文件"，可能是指 s04 和 s05 开头的文件。让我读取这两个文件的内容，然后分析它们的功能。\n\n不过用户说的是"4到05"，可能是指从 s04 到 s05 的所有文件，或者只指 s04 和 s05。让我先读取这两个文件来分析。\n\n实际上，根据中文的"4到05"，可能用户想说的是从 s04 到 s05 的文件，也就是 s04_subagent.py 和 s05_skill_loading.py。\n\n让我读取这两个文件的内容。', type='thinking'), TextBlock(citations=None, text='看到有 s04_subagent.py 和 s05_skill_loading.py 文件。让我读取这两个文件的内容进行分析：\n', type='text'), ToolUseBlock(id='call_function_ql5sof4ifx6n_1', caller=None, input={'path': 's04_subagent.py'}, name='read_file', type='tool_use'), ToolUseBlock(id='call_function_ql5sof4ifx6n_2', caller=None, input={'path': 's05_skill_loading.py'}, name='read_file', type='tool_use')]}, {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_function_ql5sof4ifx6n_1', 'content': '#!/usr/bin/env python3\n# Harness: context isolation -- protecting the model\'s clarity of thought.\n"""\ns04_subagent.py - Subagents\n\nSpawn a child agent with fresh messages=[]. The child works in its own\ncontext, sharing the filesystem, then returns only a summary to the parent.\n\n    Parent agent                     Subagent\n    +------------------+             +------------------+\n    | messages=[...]   |             | messages=[]      |  <-- fresh\n    |                  |  dispatch   |                  |\n    | tool: task       | ---------->| while tool_use:  |\n    |   prompt="..."   |            |   call tools     |\n    |   description="" |            |   append results |\n    |                  |  summary   |                  |\n    |   result = "..." | <--------- | return last text |\n    +------------------+             +------------------+\n              |\n    Parent context stays clean.\n    Subagent context is discarded.\n\nKey insight: "Process isolation gives context isolation for free."\n"""\n\nimport os\nimport subprocess\nfrom pathlib import Path\n\nfrom anthropic import Anthropic\nfrom dotenv import load_dotenv\n\nload_dotenv(override=True)\n\nif os.getenv("ANTHROPIC_BASE_URL"):\n    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)\n\nWORKDIR = Path.cwd()\nclient = Anthropic()\nMODEL = os.environ["MODEL_ID"]\n\nSYSTEM = f"You are a coding agent at {WORKDIR}. Use the task tool to delegate exploration or subtasks."\nSUBAGENT_SYSTEM = f"You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."\n\n\n# -- Tool implementations shared by parent and child --\ndef safe_path(p: str) -> Path:\n    path = (WORKDIR / p).resolve()\n    if not path.is_relative_to(WORKDIR):\n        raise ValueError(f"Path escapes workspace: {p}")\n    return path\n\ndef run_bash(command: str) -> str:\n    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]\n    if any(d in command for d in dangerous):\n        return "Error: Dangerous command blocked"\n    try:\n        r = subprocess.run(command, shell=True, cwd=WORKDIR,\n                           capture_output=True, text=True, timeout=120)\n        out = (r.stdout + r.stderr).strip()\n        return out[:50000] if out else "(no output)"\n    except subprocess.TimeoutExpired:\n        return "Error: Timeout (120s)"\n    except (FileNotFoundError, OSError) as e:\n        return f"Error: {e}"\n\ndef run_read(path: str, limit: int = None) -> str:\n    try:\n        lines = safe_path(path).read_text().splitlines()\n        if limit and limit < len(lines):\n            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]\n        return "\\n".join(lines)[:50000]\n    except Exception as e:\n        return f"Error: {e}"\n\ndef run_write(path: str, content: str) -> str:\n    try:\n        fp = safe_path(path)\n        fp.parent.mkdir(parents=True, exist_ok=True)\n        fp.write_text(content)\n        return f"Wrote {len(content)} bytes"\n    except Exception as e:\n        return f"Error: {e}"\n\ndef run_edit(path: str, old_text: str, new_text: str) -> str:\n    try:\n        fp = safe_path(path)\n        content = fp.read_text()\n        if old_text not in content:\n            return f"Error: Text not found in {path}"\n        fp.write_text(content.replace(old_text, new_text, 1))\n        return f"Edited {path}"\n    except Exception as e:\n        return f"Error: {e}"\n\n\nTOOL_HANDLERS = {\n    "bash":       lambda **kw: run_bash(kw["command"]),\n    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),\n    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),\n    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),\n}\n\n# Child gets all base tools except task (no recursive spawning)\nCHILD_TOOLS = [\n    {"name": "bash", "description": "Run a shell command.",\n     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},\n    {"name": "read_file", "description": "Read file contents.",\n     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},\n    {"name": "write_file", "description": "Write content to file.",\n     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},\n    {"name": "edit_file", "description": "Replace exact text in file.",\n     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},\n]\n\n\n# -- Subagent: fresh context, filtered tools, summary-only return --\ndef run_subagent(prompt: str) -> str:\n    sub_messages = [{"role": "user", "content": prompt}]  # fresh context\n    for _ in range(30):  # safety limit\n        response = client.messages.create(\n            model=MODEL, system=SUBAGENT_SYSTEM, messages=sub_messages,\n            tools=CHILD_TOOLS, max_tokens=8000,\n        )\n        sub_messages.append({"role": "assistant", "content": response.content})\n        if response.stop_reason != "tool_use":\n            break\n        results = []\n        for block in response.content:\n            if block.type == "tool_use":\n                handler = TOOL_HANDLERS.get(block.name)\n                output = handler(**block.input) if handler else f"Unknown tool: {block.name}"\n                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)[:50000]})\n        sub_messages.append({"role": "user", "content": results})\n    # Only the final text returns to the parent -- child context is discarded\n    return "".join(b.text for b in response.content if hasattr(b, "text")) or "(no summary)"\n\n\n# -- Parent tools: base tools + task dispatcher --\nPARENT_TOOLS = CHILD_TOOLS + [\n    {"name": "task", "description": "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",\n     "input_schema": {"type": "object", "properties": {"prompt": {"type": "string"}, "description": {"type": "string", "description": "Short description of the task"}}, "required": ["prompt"]}},\n]\n\n\ndef agent_loop(messages: list):\n    while True:\n        response = client.messages.create(\n            model=MODEL, system=SYSTEM, messages=messages,\n            tools=PARENT_TOOLS, max_tokens=8000,\n        )\n        messages.append({"role": "assistant", "content": response.content})\n        if response.stop_reason != "tool_use":\n            return\n        results = []\n        for block in response.content:\n            if block.type == "tool_use":\n                if block.name == "task":\n                    desc = block.input.get("description", "subtask")\n                    prompt = block.input.get("prompt", "")\n                    print(f"> task ({desc}): {prompt[:80]}")\n                    output = run_subagent(prompt)\n                else:\n                    handler = TOOL_HANDLERS.get(block.name)\n                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"\n                print(f"  {str(output)[:200]}")\n                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})\n        messages.append({"role": "user", "content": results})\n\n\nif __name__ == "__main__":\n    history = []\n    while True:\n        try:\n            query = input("\\033[36ms04 >> \\033[0m")\n        except (EOFError, KeyboardInterrupt):\n            break\n        if query.strip().lower() in ("q", "exit", ""):\n            break\n        history.append({"role": "user", "content": query})\n        agent_loop(history)\n        response_content = history[-1]["content"]\n        if isinstance(response_content, list):\n            for block in response_content:\n                if hasattr(block, "text"):\n                    print(block.text)\n        print()'}, {'type': 'tool_result', 'tool_use_id': 'call_function_ql5sof4ifx6n_2', 'content': '#!/usr/bin/env python3\n# Harness: on-demand knowledge -- domain expertise, loaded when the model asks.\n"""\ns05_skill_loading.py - Skills\n\nTwo-layer skill injection that avoids bloating the system prompt:\n\n    Layer 1 (cheap): skill names in system prompt (~100 tokens/skill)\n    Layer 2 (on demand): full skill body in tool_result\n\n    skills/\n      pdf/\n        SKILL.md          <-- frontmatter (name, description) + body\n      code-review/\n        SKILL.md\n\n    System prompt:\n    +--------------------------------------+\n    | You are a coding agent.              |\n    | Skills available:                    |\n    |   - pdf: Process PDF files...        |  <-- Layer 1: metadata only\n    |   - code-review: Review code...      |\n    +--------------------------------------+\n\n    When model calls load_skill("pdf"):\n    +--------------------------------------+\n    | tool_result:                         |\n    | <skill>                              |\n    |   Full PDF processing instructions   |  <-- Layer 2: full body\n    |   Step 1: ...                        |\n    |   Step 2: ...                        |\n    | </skill>                             |\n    +--------------------------------------+\n\nKey insight: "Don\'t put everything in the system prompt. Load on demand."\n"""\n\nimport os\nimport re\nimport subprocess\nimport yaml\nfrom pathlib import Path\n\nfrom anthropic import Anthropic\nfrom dotenv import load_dotenv\n\nload_dotenv(override=True)\n\nif os.getenv("ANTHROPIC_BASE_URL"):\n    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)\n\nWORKDIR = Path.cwd()\nclient = Anthropic()\nMODEL = os.environ["MODEL_ID"]\nSKILLS_DIR = WORKDIR / "skills"\n\n\n# -- SkillLoader: scan skills/<name>/SKILL.md with YAML frontmatter --\nclass SkillLoader:\n    def __init__(self, skills_dir: Path):\n        self.skills_dir = skills_dir\n        self.skills = {}\n        self._load_all()\n\n    def _load_all(self):\n        if not self.skills_dir.exists():\n            return\n        for f in sorted(self.skills_dir.rglob("SKILL.md")):\n            text = f.read_text()\n            meta, body = self._parse_frontmatter(text)\n            name = meta.get("name", f.parent.name)\n            self.skills[name] = {"meta": meta, "body": body, "path": str(f)}\n\n    def _parse_frontmatter(self, text: str) -> tuple:\n        """Parse YAML frontmatter between --- delimiters."""\n        match = re.match(r"^---\\n(.*?)\\n---\\n(.*)", text, re.DOTALL)\n        if not match:\n            return {}, text\n        try:\n            meta = yaml.safe_load(match.group(1)) or {}\n        except yaml.YAMLError:\n            meta = {}\n        return meta, match.group(2).strip()\n\n    def get_descriptions(self) -> str:\n        """Layer 1: short descriptions for the system prompt."""\n        if not self.skills:\n            return "(no skills available)"\n        lines = []\n        for name, skill in self.skills.items():\n            desc = skill["meta"].get("description", "No description")\n            tags = skill["meta"].get("tags", "")\n            line = f"  - {name}: {desc}"\n            if tags:\n                line += f" [{tags}]"\n            lines.append(line)\n        return "\\n".join(lines)\n\n    def get_content(self, name: str) -> str:\n        """Layer 2: full skill body returned in tool_result."""\n        skill = self.skills.get(name)\n        if not skill:\n            return f"Error: Unknown skill \'{name}\'. Available: {\', \'.join(self.skills.keys())}"\n        return f"<skill name=\\"{name}\\">\\n{skill[\'body\']}\\n</skill>"\n\n\nSKILL_LOADER = SkillLoader(SKILLS_DIR)\n\n# Layer 1: skill metadata injected into system prompt\nSYSTEM = f"""You are a coding agent at {WORKDIR}.\nUse load_skill to access specialized knowledge before tackling unfamiliar topics.\n\nSkills available:\n{SKILL_LOADER.get_descriptions()}"""\n\n\n# -- Tool implementations --\ndef safe_path(p: str) -> Path:\n    path = (WORKDIR / p).resolve()\n    if not path.is_relative_to(WORKDIR):\n        raise ValueError(f"Path escapes workspace: {p}")\n    return path\n\ndef run_bash(command: str) -> str:\n    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]\n    if any(d in command for d in dangerous):\n        return "Error: Dangerous command blocked"\n    try:\n        r = subprocess.run(command, shell=True, cwd=WORKDIR,\n                           capture_output=True, text=True, timeout=120)\n        out = (r.stdout + r.stderr).strip()\n        return out[:50000] if out else "(no output)"\n    except subprocess.TimeoutExpired:\n        return "Error: Timeout (120s)"\n\ndef run_read(path: str, limit: int = None) -> str:\n    try:\n        lines = safe_path(path).read_text().splitlines()\n        if limit and limit < len(lines):\n            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]\n        return "\\n".join(lines)[:50000]\n    except Exception as e:\n        return f"Error: {e}"\n\ndef run_write(path: str, content: str) -> str:\n    try:\n        fp = safe_path(path)\n        fp.parent.mkdir(parents=True, exist_ok=True)\n        fp.write_text(content)\n        return f"Wrote {len(content)} bytes"\n    except Exception as e:\n        return f"Error: {e}"\n\ndef run_edit(path: str, old_text: str, new_text: str) -> str:\n    try:\n        fp = safe_path(path)\n        content = fp.read_text()\n        if old_text not in content:\n            return f"Error: Text not found in {path}"\n        fp.write_text(content.replace(old_text, new_text, 1))\n        return f"Edited {path}"\n    except Exception as e:\n        return f"Error: {e}"\n\n\nTOOL_HANDLERS = {\n    "bash":       lambda **kw: run_bash(kw["command"]),\n    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),\n    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),\n    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),\n    "load_skill": lambda **kw: SKILL_LOADER.get_content(kw["name"]),\n}\n\nTOOLS = [\n    {"name": "bash", "description": "Run a shell command.",\n     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},\n    {"name": "read_file", "description": "Read file contents.",\n     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},\n    {"name": "write_file", "description": "Write content to file.",\n     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},\n    {"name": "edit_file", "description": "Replace exact text in file.",\n     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},\n    {"name": "load_skill", "description": "Load specialized knowledge by name.",\n     "input_schema": {"type": "object", "properties": {"name": {"type": "string", "description": "Skill name to load"}}, "required": ["name"]}},\n]\n\n\ndef agent_loop(messages: list):\n    while True:\n        response = client.messages.create(\n            model=MODEL, system=SYSTEM, messages=messages,\n            tools=TOOLS, max_tokens=8000,\n        )\n        messages.append({"role": "assistant", "content": response.content})\n        if response.stop_reason != "tool_use":\n            return\n        results = []\n        for block in response.content:\n            if block.type == "tool_use":\n                handler = TOOL_HANDLERS.get(block.name)\n                try:\n                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"\n                except Exception as e:\n                    output = f"Error: {e}"\n                print(f"> {block.name}:")\n                print(str(output)[:200])\n                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})\n        messages.append({"role": "user", "content": results})\n\n\nif __name__ == "__main__":\n    history = []\n    while True:\n        try:\n            query = input("\\033[36ms05 >> \\033[0m")\n        except (EOFError, KeyboardInterrupt):\n            break\n        if query.strip().lower() in ("q", "exit", ""):\n            break\n        history.append({"role": "user", "content": query})\n        agent_loop(history)\n        response_content = history[-1]["content"]\n        if isinstance(response_content, list):\n            for block in response_content:\n                if hasattr(block, "text"):\n                    print(block.text)\n        print()'}]}, {'role': 'assistant', 'content': [ThinkingBlock(signature='298bf77d9475274a873029de3b6e5cc0ada466f0e8e88562f772e57cf771ef97', thinking='现在我已经读取了两个文件的内容。让我分析它们的功能，并创建 readme 目录和相应的说明文档。\n\n**s04_subagent.py 分析：**\n- 这是一个子代理(Subagent)实现\n- 核心功能：创建一个子代理，它有全新的消息上下文(messages=[])\n- 子代理在自己的上下文中工作，共享文件系统，但不共享对话历史\n- 完成后只返回摘要给父代理\n- 关键点："进程隔离免费提供上下文隔离"\n- 主要组件：\n  - 基础工具：bash、read_file、write_file、edit_file\n  - task 工具：用于派生子代理\n  - run_subagent() 函数：运行子代理\n  - agent_loop() 函数：主循环\n\n**s05_skill_loading.py 分析：**\n- 这是一个技能加载(Skills)系统实现\n- 核心功能：两层技能注入，避免系统提示词膨胀\n- Layer 1（便宜）：在系统提示词中只放技能名称和描述（每个技能约100 tokens）\n- Layer 2（按需）：在 tool_result 中返回完整的技能内容\n- SkillLoader 类：扫描 skills/<name>/SKILL.md 文件，使用 YAML frontmatter 解析元数据\n- get_descriptions()：获取所有技能的描述（Layer 1）\n- get_content(name)：获取特定技能的全部内容（Layer 2）\n\n\n- load_skill 工具允许按需加载具体技能\n\n现在让我创建 readme 目录并生成说明文档。', type='thinking'), TextBlock(citations=None, text='现在我来创建 readme 目录并生成分析文档：\n', type='text'), ToolUseBlock(id='call_function_ihbca259fyuf_1', caller=None, input={'command': 'mkdir -p readme'}, name='bash', type='tool_use')]}, {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_function_ihbca259fyuf_1', 'content': '(no output)'}, {'type': 'text', 'text': '<reminder>Update your todos.</reminder>'}]}, {'role': 'assistant', 'content': [ThinkingBlock(signature='ac29cc9d366174f5fdb870eea44eab5b475d43bb108c74d43eab75d56d68bd43', thinking='好的，用户提醒我更新任务列表。我需要分析两个文件并创建对应的README文档。让我先更新todo列表，然后继续工作。', type='thinking'), ToolUseBlock(id='call_function_vkjzoip7j7vy_1', caller=None, input={'items': [{'id': '1', 'status': 'in_progress', 'text': '分析 s04_subagent.py 文件功能'}, {'id': '2', 'status': 'pending', 'text': '分析 s05_skill_loading.py 文件功能'}, {'id': '3', 'status': 'pending', 'text': '创建 readme/s04_subagent.md'}, {'id': '4', 'status': 'pending', 'text': '创建 readme/s05_skill_loading.md'}]}, name='todo', type='tool_use')]}, {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_function_vkjzoip7j7vy_1', 'content': '[>] #1: 分析 s04_subagent.py 文件功能\n[ ] #2: 分析 s05_skill_loading.py 文件功能\n[ ] #3: 创建 readme/s04_subagent.md\n[ ] #4: 创建 readme/s05_skill_loading.md\n\n(0/4 completed)'}]}, {'role': 'assistant', 'content': [ThinkingBlock(signature='e41d85eb6a614ebfe3e7dda73c864e4e013e38c7a2770837d48a7c11b71de8dc', thinking='现在我来创建两个readme文件。', type='thinking'), ToolUseBlock(id='call_function_d3we7rzqqesz_1', caller=None, input={'content': '# s04_subagent.py - 子代理（Subagent）\n\n## 核心概念\n\n子代理机制通过**进程隔离**实现**上下文隔离**。子代理在独立的对话上下文中工作，完成任务后只返回摘要给父代理，从而保护父代理的上下文清晰度。\n\n## 架构图\n\n```\n    父代理 (Parent Agent)                 子代理 (Subagent)\n    +------------------+                  +------------------+\n    | messages=[...]   |                  | messages=[]      |  <-- 全新上下文\n    |                  |    dispatch      |                  |\n    | 工具: task       | --------------> |  while tool_use: |\n    |   prompt="..."   |                  |    call tools    |\n    |   description="" |                  |    append results|\n    |                  |    summary       |                  |\n    |  result = "..."  | <-------------  |  return last text|\n    +------------------+                  +------------------+\n              |\n    父代理上下文保持干净\n    子代理上下文被丢弃\n```\n\n## 关键特性\n\n### 1. 上下文隔离\n- 子代理拥有全新的 `messages=[]`\n- 子代理不会继承父代理的对话历史\n- 只有最终文本摘要返回给父代理\n\n### 2. 安全工具集\n子代理拥有精简的工具集（不包括 task 工具，防止递归嵌套）：\n- `bash`: 执行shell命令\n- `read_file`: 读取文件\n- `write_file`: 写入文件\n- `edit_file`: 编辑文件\n\n### 3. 安全防护\n- 路径安全检查：`safe_path()` 确保文件操作不超出工作目录\n- 危险命令拦截：阻止 `rm -rf /`、`sudo`、`shutdown` 等高危命令\n- 执行超时限制：120秒超时\n\n## 核心函数\n\n### `run_subagent(prompt: str) -> str`\n```python\ndef run_subagent(prompt: str) -> str:\n    sub_messages = [{"role": "user", "content": prompt}]  # 全新上下文\n    for _ in range(30):  # 安全限制\n        # 调用API处理任务\n        ...\n    # 只有最终文本返回给父代理\n    return "".join(b.text for b in response.content ...)\n```\n\n### `agent_loop(messages: list)`\n主代理循环，处理父代理的工具调用。当调用 `task` 工具时，触发子代理执行。\n\n## 使用方式\n\n```bash\npython s04_subagent.py\n```\n\n启动后可以输入任务，子代理将帮助完成并在完成后返回摘要。\n\n## 核心洞察\n\n> **"进程隔离为上下文隔离提供了免费方案"**\n> \n> 通过为每个子代理创建独立进程，可以自然地实现上下文隔离，无需复杂的上下文管理逻辑。\n\n## 文件结构\n\n| 组件 | 描述 |\n|------|------|\n| `TOOL_HANDLERS` | 工具处理器映射表 |\n| `CHILD_TOOLS` | 子代理可用的工具定义 |\n| `PARENT_TOOLS` | 父代理工具集（基础工具 + task） |\n| `run_subagent()` | 创建并运行子代理的核心函数 |\n| `agent_loop()` | 父代理主循环 |', 'path': 'readme/s04_subagent.md'}, name='write_file', type='tool_use')]}, {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_function_d3we7rzqqesz_1', 'content': 'Wrote 1877 bytes'}]}, {'role': 'assistant', 'content': [ThinkingBlock(signature='c85a6c6c4ade0758a825d0b01589ccae5da403d1afb7f825e94f27f781b3f719', thinking='s04_subagent.md 已创建。现在分析 s05_skill_loading.py 并创建对应的 README。', type='thinking'), ToolUseBlock(id='call_function_kn0i7uwmgvek_1', caller=None, input={'items': [{'id': '1', 'status': 'completed', 'text': '分析 s04_subagent.py 文件功能'}, {'id': '2', 'status': 'in_progress', 'text': '分析 s05_skill_loading.py 文件功能'}, {'id': '3', 'status': 'completed', 'text': '创建 readme/s04_subagent.md'}, {'id': '4', 'status': 'pending', 'text': '创建 readme/s05_skill_loading.md'}]}, name='todo', type='tool_use')]}, {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_function_kn0i7uwmgvek_1', 'content': '[x] #1: 分析 s04_subagent.py 文件功能\n[>] #2: 分析 s05_skill_loading.py 文件功能\n[x] #3: 创建 readme/s04_subagent.md\n[ ] #4: 创建 readme/s05_skill_loading.md\n\n(2/4 completed)'}]}, {'role': 'assistant', 'content': [ThinkingBlock(signature='9db6740f902470cfaea40f55964be5546eb903a6b388ef6ebdba736c23a7ac3e', thinking='现在创建 s05_skill_loading.md', type='thinking'), ToolUseBlock(id='call_function_krqynovlb429_1', caller=None, input={'content': '# s05_skill_loading.py - 技能加载（Skills）\n\n## 核心概念\n\n技能加载机制实现**按需知识注入**，通过两层架构避免系统提示词膨胀：\n- **第一层（廉价）**：技能名称和描述，放入系统提示词\n- **第二层（按需）**：完整技能内容，通过工具结果返回\n\n## 架构图\n\n```\nskills/\n  pdf/\n    SKILL.md          <-- frontmatter (name, description) + body\n  code-review/\n    SKILL.md\n\n系统提示词:\n+--------------------------------------+\n| You are a coding agent.              |\n| Skills available:                    |\n|   - pdf: Process PDF files...        |  <-- Layer 1: 元数据\n|   - code-review: Review code...     |\n+--------------------------------------+\n\n当模型调用 load_skill("pdf"):\n+--------------------------------------+\n| tool_result:                         |\n| <skill name="pdf">                   |\n|   Full PDF processing instructions   |  <-- Layer 2: 完整内容\n|   Step 1: ...                        |\n|   Step 2: ...                        |\n| </skill>                             |\n+--------------------------------------+\n```\n\n## 关键特性\n\n### 1. 双层技能加载\n\n#### Layer 1: 元数据层\n- 在 `SkillLoader.get_descriptions()` 中生成\n- 放入系统提示词，供模型了解可用技能\n- 每个技能约100 tokens，开销极低\n\n#### Layer 2: 内容层\n- 在 `SkillLoader.get_content()` 中按需加载\n- 通过 `load_skill` 工具返回完整技能内容\n- 使用 `<skill>` XML 标签包裹，便于模型识别\n\n### 2. 技能文件格式\n\n使用 YAML frontmatter 格式：\n\n```yaml\n---\nname: pdf\ndescription: Process PDF files with OCR\ntags: document, ocr, extraction\n---\n# 技能正文内容\nStep 1: Install required libraries...\nStep 2: Use PyPDF2 to extract text...\nStep 3: Apply OCR if needed...\n```\n\n### 3. SkillLoader 类\n\n```python\nclass SkillLoader:\n    def __init__(self, skills_dir: Path):\n        self.skills = {}\n        self._load_all()  # 扫描所有 SKILL.md\n\n    def get_descriptions(self) -> str:\n        """Layer 1: 生成技能描述列表"""\n        \n    def get_content(self, name: str) -> str:\n        """Layer 2: 获取完整技能内容"""\n```\n\n## 工具定义\n\n| 工具名 | 功能 |\n|--------|------|\n| `bash` | 执行shell命令 |\n| `read_file` | 读取文件内容 |\n| `write_file` | 写入文件内容 |\n| `edit_file` | 编辑文件内容 |\n| `load_skill` | **按需加载技能知识** |\n\n## 安全机制\n\n- 路径安全检查：防止目录遍历攻击\n- 危险命令拦截：阻止高危shell命令\n- 执行超时限制：120秒\n\n## 使用方式\n\n### 1. 创建技能目录\n```bash\nmkdir -p skills/pdf\nmkdir -p skills/code-review\n```\n\n### 2. 编写技能文件（skills/pdf/SKILL.md）\n```yaml\n---\nname: pdf\ndescription: Process PDF files\n---\nStep 1: Check if PDF is text-based or scanned...\nStep 2: Use PyPDF2 for text extraction...\nStep 3: Use pytesseract for OCR on scanned PDFs...\n```\n\n### 3. 运行\n```bash\npython s05_skill_loading.py\n```\n\n## 核心洞察\n\n> **"不要把所有知识都放入系统提示词，按需加载。"**\n\n按需加载策略的优势：\n1. **减少上下文开销**：系统提示词保持精简\n2. **动态加载**：只在需要时获取技能\n3. **可扩展性**：新增技能只需添加文件，无需修改代码\n\n## 文件结构\n\n| 组件 | 描述 |\n|------|------|\n| `SkillLoader` | 技能加载器类 |\n| `SKILLS_DIR` | 技能目录路径 |\n| `SYSTEM` | 包含Layer 1信息的系统提示词 |\n| `TOOL_HANDLERS` | 工具处理器映射 |\n| `TOOLS` | 工具定义列表 |', 'path': 'readme/s05_skill_loading.md'}, name='write_file', type='tool_use')]}, {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_function_krqynovlb429_1', 'content': 'Wrote 2652 bytes'}]}, {'role': 'assistant', 'content': [ThinkingBlock(signature='6b56744e10855cf125464973ef2b049e27acb0797758d33d7c8d2e1d92dd7b5c', thinking='两个文件的readme都创建完成了，现在更新todo列表。', type='thinking'), ToolUseBlock(id='call_function_ywj9eg61s806_1', caller=None, input={'items': [{'id': '1', 'status': 'completed', 'text': '分析 s04_subagent.py 文件功能'}, {'id': '2', 'status': 'completed', 'text': '分析 s05_skill_loading.py 文件功能'}, {'id': '3', 'status': 'completed', 'text': '创建 readme/s04_subagent.md'}, {'id': '4', 'status': 'completed', 'text': '创建 readme/s05_skill_loading.md'}]}, name='todo', type='tool_use')]}, {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_function_ywj9eg61s806_1', 'content': '[x] #1: 分析 s04_subagent.py 文件功能\n[x] #2: 分析 s05_skill_loading.py 文件功能\n[x] #3: 创建 readme/s04_subagent.md\n[x] #4: 创建 readme/s05_skill_loading.md\n\n(4/4 completed)'}]}, {'role': 'assistant', 'content': [ThinkingBlock(signature='59a2faa34ecd32a68765de3f289596a8ef300fb5b73ca6624eb4c796a9804fc0', thinking='All tasks are done. Let me verify the output files exist.', type='thinking'), ToolUseBlock(id='call_function_uswel461a1o5_1', caller=None, input={'command': 'ls -la readme/'}, name='bash', type='tool_use')]}, {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_function_uswel461a1o5_1', 'content': 'total 16\ndrwxr-xr-x   4 ryanzhou  staff   128  5月 13 11:39 .\ndrwxr-xr-x  18 ryanzhou  staff   576  5月 13 11:37 ..\n-rw-r--r--   1 ryanzhou  staff  2809  5月 13 11:37 s04_subagent.md\n-rw-r--r--   1 ryanzhou  staff  3560  5月 13 11:39 s05_skill_loading.md'}]}, {'role': 'assistant', 'content': [ThinkingBlock(signature='879e54fb26a499734a7ee34cd6d175d1d76f1d4e5f5c7acfd6f7daae700e4433', thinking='两个文件都创建成功了。我来总结一下完成的工作。', type='thinking'), TextBlock(citations=None, text='完成！已创建两个文件：\n\n| 文件 | 大小 | 内容 |\n|------|------|------|\n| `readme/s04_subagent.md` | 2.8KB | s04 子代理机制：上下文隔离、通过进程隔离保护父代理清晰度 |\n| `readme/s05_skill_loading.md` | 3.5KB | s05 技能加载：双层按需知识注入，避免系统提示词膨胀 |\n\n**核心区别**：\n- **s04**: 通过子代理实现**上下文隔离**（保护模型思维清晰）\n- **s05**: 通过技能加载实现**按需知识**（避免提示词膨胀）', type='text')]}]
# 完成！已创建两个文件：

# | 文件 | 大小 | 内容 |
# |------|------|------|
# | `readme/s04_subagent.md` | 2.8KB | s04 子代理机制：上下文隔离、通过进程隔离保护父代理清晰度 |
# | `readme/s05_skill_loading.md` | 3.5KB | s05 技能加载：双层按需知识注入，避免系统提示词膨胀 |

# **核心区别**：
# - **s04**: 通过子代理实现**上下文隔离**（保护模型思维清晰）
# - **s05**: 通过技能加载实现**按需知识**（避免提示词膨胀）
