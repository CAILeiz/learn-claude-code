#!/usr/bin/env python3
# Harness: the loop -- the model's first connection to the real world.
"""
s01_agent_loop.py - The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
"""

import os
import subprocess

try:
    import readline

    # #143 UTF-8 backspace fix for macOS libedit
    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
    readline.parse_and_bind("set enable-meta-keybindings on")
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

print(os.getenv("ANTHROPIC_BASE_URL"))
print(os.getenv("ANTHROPIC_API_KEY"))
print(os.getcwd())

client = Anthropic()
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }
]


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
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


# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )
        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})
        # If the model didn't call a tool, we're done
        if response.stop_reason != "tool_use":
            return
        # Execute each tool call, collect results
        results = []
        print(f"response.content {response.content}")
        for block in response.content:
            if block.type == "tool_use":
                print(f"\033[33m$ {block.input['command']}\033[0m")
                output = run_bash(block.input["command"])
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
            query = input("\033[36ms01 >> \033[0m")
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


# 提问1: ls
# 回答：
# response.content [ThinkingBlock(signature='9c5da38d788ecd8da13f48a665fb10e054e68097606090af35b5d31121c7a72a', thinking="The user wants to list the contents of the current directory. I'll use the bash command to do this.\n", type='thinking'), ToolUseBlock(id='call_function_c21iywsco1dy_1', caller=None, input={'command': 'ls'}, name='bash', type='tool_use')]
# $ ls
# __init__.py
# s01_agent_loop.py
# s02_tool_use.py
# s03_todo_write.py
# s04_subagent.py
# s05_skill_loading.py
# s06_context_compact.py
# s07_task_system.py
# s08_background_tasks.py
# s09_agent_teams.py
# s10_team_proto
# history1: [{'role': 'user', 'content': 'ls'}, {'role': 'assistant', 'content': [ThinkingBlock(signature='9c5da38d788ecd8da13f48a665fb10e054e68097606090af35b5d31121c7a72a', thinking="The user wants to list the contents of the current directory. I'll use the bash command to do this.\n", type='thinking'), ToolUseBlock(id='call_function_c21iywsco1dy_1', caller=None, input={'command': 'ls'}, name='bash', type='tool_use')]}, {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_function_c21iywsco1dy_1', 'content': '__init__.py\ns01_agent_loop.py\ns02_tool_use.py\ns03_todo_write.py\ns04_subagent.py\ns05_skill_loading.py\ns06_context_compact.py\ns07_task_system.py\ns08_background_tasks.py\ns09_agent_teams.py\ns10_team_protocols.py\ns11_autonomous_agents.py\ns12_worktree_task_isolation.py\ns_full.py'}]}, {'role': 'assistant', 'content': [ThinkingBlock(signature='50b5d2b314fd9172084bce5c826873ca187845152d7ad0e78ba30a0ca54f1e85', thinking="The user wants me to list files in the current directory. I've already done that - this appears to be a directory with several Python files related to learning Claude Code concepts (s01 through s12, plus some other files).\n", type='thinking'), TextBlock(citations=None, text='\nHere are the files in the current directory:\n\n| File | Description |\n|------|-------------|\n| `s01_agent_loop.py` | Agent loop examples |\n| `s02_tool_use.py` | Tool use examples |\n| `s03_todo_write.py` | Todo writing examples |\n| `s04_subagent.py` | Subagent examples |\n| `s05_skill_loading.py` | Skill loading examples |\n| `s06_context_compact.py` | Context compaction examples |\n| `s07_task_system.py` | Task system examples |\n| `s08_background_tasks.py` | Background task examples |\n| `s09_agent_teams.py` | Agent teams examples |\n| `s10_team_protocols.py` | Team protocols examples |\n| `s11_autonomous_agents.py` | Autonomous agent examples |\n| `s12_worktree_task_isolation.py` | Worktree/task isolation examples |\n| `s_full.py` | Full examples |\n| `__init__.py` | Package init file |', type='text')]}]

# Here are the files in the current directory:

# | File | Description |
# |------|-------------|
# | `s01_agent_loop.py` | Agent loop examples |
# | `s02_tool_use.py` | Tool use examples |
# | `s03_todo_write.py` | Todo writing examples |
# | `s04_subagent.py` | Subagent examples |
# | `s05_skill_loading.py` | Skill loading examples |
# | `s06_context_compact.py` | Context compaction examples |
# | `s07_task_system.py` | Task system examples |
# | `s08_background_tasks.py` | Background task examples |
# | `s09_agent_teams.py` | Agent teams examples |
# | `s10_team_protocols.py` | Team protocols examples |
# | `s11_autonomous_agents.py` | Autonomous agent examples |
# | `s12_worktree_task_isolation.py` | Worktree/task isolation examples |
# | `s_full.py` | Full examples |
# | `__init__.py` | Package init file |


# 提问2：你好
# 回答：
# history2: [{'role': 'user', 'content': 'ls'}, {'role': 'assistant', 'content': [ThinkingBlock(signature='9c5da38d788ecd8da13f48a665fb10e054e68097606090af35b5d31121c7a72a', thinking="The user wants to list the contents of the current directory. I'll use the bash command to do this.\n", type='thinking'), ToolUseBlock(id='call_function_c21iywsco1dy_1', caller=None, input={'command': 'ls'}, name='bash', type='tool_use')]}, {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_function_c21iywsco1dy_1', 'content': '__init__.py\ns01_agent_loop.py\ns02_tool_use.py\ns03_todo_write.py\ns04_subagent.py\ns05_skill_loading.py\ns06_context_compact.py\ns07_task_system.py\ns08_background_tasks.py\ns09_agent_teams.py\ns10_team_protocols.py\ns11_autonomous_agents.py\ns12_worktree_task_isolation.py\ns_full.py'}]}, {'role': 'assistant', 'content': [ThinkingBlock(signature='50b5d2b314fd9172084bce5c826873ca187845152d7ad0e78ba30a0ca54f1e85', thinking="The user wants me to list files in the current directory. I've already done that - this appears to be a directory with several Python files related to learning Claude Code concepts (s01 through s12, plus some other files).\n", type='thinking'), TextBlock(citations=None, text='\nHere are the files in the current directory:\n\n| File | Description |\n|------|-------------|\n| `s01_agent_loop.py` | Agent loop examples |\n| `s02_tool_use.py` | Tool use examples |\n| `s03_todo_write.py` | Todo writing examples |\n| `s04_subagent.py` | Subagent examples |\n| `s05_skill_loading.py` | Skill loading examples |\n| `s06_context_compact.py` | Context compaction examples |\n| `s07_task_system.py` | Task system examples |\n| `s08_background_tasks.py` | Background task examples |\n| `s09_agent_teams.py` | Agent teams examples |\n| `s10_team_protocols.py` | Team protocols examples |\n| `s11_autonomous_agents.py` | Autonomous agent examples |\n| `s12_worktree_task_isolation.py` | Worktree/task isolation examples |\n| `s_full.py` | Full examples |\n| `__init__.py` | Package init file |', type='text')]}, {'role': 'user', 'content': '你好'}, {'role': 'assistant', 'content': [ThinkingBlock(signature='0627ee10e4af2227f544ddad20ffa00e87bc1003d5cb625f45f2962e8506d637', thinking='The user is greeting me in Chinese with "你好" (nǐ hǎo), which means "hello" or "hi". I should respond in a friendly manner in Chinese.\n', type='thinking'), TextBlock(citations=None, text='\n你好！有什么我可以帮助你的吗？', type='text')]}]

# 你好！有什么我可以帮助你的吗？
