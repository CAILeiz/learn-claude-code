#!/usr/bin/env python3
# Harness: compression -- clean memory for infinite sessions.
"""
s06_context_compact.py - Compact

Three-layer compression pipeline so the agent can work forever:

    Every turn:
    +------------------+
    | Tool call result |
    +------------------+
            |
            v
    [Layer 1: micro_compact]        (silent, every turn)
      Replace non-read_file tool_result content older than last 3
      with "[Previous: used {tool_name}]"
            |
            v
    [Check: tokens > 50000?]
       |               |
       no              yes
       |               |
       v               v
    continue    [Layer 2: auto_compact]
                  Save full transcript to .transcripts/
                  Ask LLM to summarize conversation.
                  Replace all messages with [summary].
                        |
                        v
                [Layer 3: compact tool]
                  Model calls compact -> immediate summarization.
                  Same as auto, triggered manually.

Key insight: "The agent can forget strategically and keep working forever."
"""

import json
import os
import subprocess
import time
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic()
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."

THRESHOLD = 50000
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
KEEP_RECENT = 3
PRESERVE_RESULT_TOOLS = {"read_file"}


def estimate_tokens(messages: list) -> int:
    """Rough token count: ~4 chars per token."""
    return len(str(messages)) // 4


# -- Layer 1: micro_compact - replace old tool results with placeholders --
def micro_compact(messages: list) -> list:
    # Collect (msg_index, part_index, tool_result_dict) for all tool_result entries
    tool_results = []
    for msg_idx, msg in enumerate(messages):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for part_idx, part in enumerate(msg["content"]):
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_results.append((msg_idx, part_idx, part))
    if len(tool_results) <= KEEP_RECENT:
        return messages
    # Find tool_name for each result by matching tool_use_id in prior assistant messages
    tool_name_map = {}
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if hasattr(block, "type") and block.type == "tool_use":
                        tool_name_map[block.id] = block.name
    # Clear old results (keep last KEEP_RECENT). Preserve read_file outputs because
    # they are reference material; compacting them forces the agent to re-read files.
    to_clear = tool_results[:-KEEP_RECENT]
    for _, _, result in to_clear:
        if not isinstance(result.get("content"), str) or len(result["content"]) <= 100:
            continue
        tool_id = result.get("tool_use_id", "")
        tool_name = tool_name_map.get(tool_id, "unknown")
        if tool_name in PRESERVE_RESULT_TOOLS:
            continue
        # content 只说用了什么工具，不展示具体 content
        result["content"] = f"[Previous: used {tool_name}]"
    return messages


# -- Layer 2: auto_compact - save transcript, summarize, replace messages --
def auto_compact(messages: list) -> list:
    # Save full transcript to disk
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    print(f"[transcript saved: {transcript_path}]")
    # Ask LLM to summarize
    conversation_text = json.dumps(messages, default=str)[-80000:]
    response = client.messages.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "Summarize this conversation for continuity. Include: "
                "1) What was accomplished, 2) Current state, 3) Key decisions made. "
                "Be concise but preserve critical details.\n\n" + conversation_text,
            }
        ],
        max_tokens=2000,
    )
    summary = next(
        (block.text for block in response.content if hasattr(block, "text")), ""
    )
    if not summary:
        summary = "No summary generated."
    # Replace all messages with compressed summary
    return [
        {
            "role": "user",
            "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}",
        },
    ]


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
    "compact": lambda **kw: "Manual compression requested.",
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
        "name": "compact",
        "description": "Trigger manual conversation compression.",
        "input_schema": {
            "type": "object",
            "properties": {
                "focus": {
                    "type": "string",
                    "description": "What to preserve in the summary",
                }
            },
        },
    },
]


def agent_loop(messages: list):
    while True:
        # Layer 1: micro_compact before each LLM call
        micro_compact(messages)
        # Layer 2: auto_compact if token estimate exceeds threshold
        if estimate_tokens(messages) > THRESHOLD:
            print("[auto_compact triggered]")
            messages[:] = auto_compact(messages)
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
        manual_compact = False
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "compact":
                    manual_compact = True
                    output = "Compressing..."
                else:
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
        messages.append({"role": "user", "content": results})
        # Layer 3: manual compact triggered by the compact tool
        if manual_compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages)
            return


if __name__ == "__main__":
    history = []
    idx = 0
    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
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


# 问题：读取根目录，总结出一篇 AGENTS.md
# 回答：
# > bash:
# total 224
# drwxr-xr-x  19 ryanzhou  staff    608  5月 18 14:23 .
# drwx------@ 14 ryanzhou  staff    448  5月 18 19:03 ..
# -rw-r--r--   1 ryanzhou  staff   2118  5月 12 10:12 .env.example
# drwxr-xr-x  15 ryan
# > read_file:
# [English](./README.md) | [中文](./README-zh.md) | [日本語](./README-ja.md)
# # Learn Claude Code -- Harness Engineering for Real Agents

# ## Agency Comes from the Model. An Agent Product = Model + Harness.

# B
# > bash:
# total 632
# -rw-r--r--   1 ryanzhou  staff    203  5月 12 10:12 __init__.py
# drwxr-xr-x  18 ryanzhou  staff    576  5月 13 11:37 .
# drwxr-xr-x  19 ryanzhou  staff    608  5月 18 14:23 ..
# -rw-r--r--   1 ryanz
# > bash:
# total 0
# drwxr-xr-x   5 ryanzhou  staff  160  5月 12 10:12 .
# drwxr-xr-x  19 ryanzhou  staff  608  5月 18 14:23 ..
# drwxr-xr-x  14 ryanzhou  staff  448  5月 12 10:12 en
# drwxr-xr-x  14 ryanzhou  staff  448
# > bash:
# total 0
# drwxr-xr-x   6 ryanzhou  staff  192  5月 12 10:12 .
# drwxr-xr-x  19 ryanzhou  staff  608  5月 18 14:23 ..
# drwxr-xr-x   5 ryanzhou  staff  160  5月 12 10:12 agent-builder
# drwxr-xr-x   3 ryanzhou  s
# > read_file:
# [English](./README.md) | [中文](./README-zh.md) | [日本語](./README-ja.md)
# # Learn Claude Code -- Harness Engineering for Real Agents

# ## Agency Comes from the Model. An Agent Product = Model + Harness.

# B
# > bash:
# total 136
# drwxr-xr-x  14 ryanzhou  staff   448  5月 12 10:12 .
# drwxr-xr-x   5 ryanzhou  staff   160  5月 12 10:12 ..
# -rw-r--r--   1 ryanzhou  staff  3664  5月 12 10:12 s01-the-agent-loop.md
# -rw-r--r--
# > read_file:
# # s01: The Agent Loop (Agent 循环)

# `[ s01 ] s02 > s03 > s04 > s05 > s06 | s07 > s08 > s09 > s10 > s11 > s12`

# > *"One loop & Bash is all you need"* -- 一个工具 + 一个循环 = 一个 Agent。
# >
# > **Harness 层**: 循环 -- 模
# > read_file:
# # s07: Task System (任务系统)

# `s01 > s02 > s03 > s04 > s05 > s06 | [ s07 ] s08 > s09 > s10 > s11 > s12`

# > *"大目标要拆成小任务, 排好序, 记在磁盘上"* -- 文件持久化的任务图, 为多 agent 协作打基础。
# >
# > **Harness 层**: 持久化任务 -- 比任何一次对话都长命的目
# > read_file:
# # s09: Agent Teams (Agent 团队)

# `s01 > s02 > s03 > s04 > s05 > s06 | s07 > s08 > [ s09 ] s10 > s11 > s12`

# > *"任务太大一个人干不完, 要能分给队友"* -- 持久化队友 + JSONL 邮箱。
# >
# > **Harness 层**: 团队邮箱 -- 多个模型, 通过文件协调。

# ## 问题

# > write_file:
# Wrote 3682 bytes
# history0: [{'role': 'user', 'content': '读取根目录，总结出一篇 AGENTS.md'}, {'role': 'assistant', 'content': [ThinkingBlock(signature='36a3a7d766bec1dd1c3bcb03256fa203ca92b24647cdf94e702437557349a66f', thinking='用户要求读取根目录并总结出一篇 AGENTS.md 文件。我需要先查看根目录下有哪些文件和文件夹，然后根据内容创建一个文档。\n', type='thinking'), ToolUseBlock(id='call_function_rj7gc6fmebry_1', caller=None, input={'command': 'ls -la /Users/ryanzhou/Desktop/learn-claude-code'}, name='bash', type='tool_use')]}, {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_function_rj7gc6fmebry_1', 'content': '[Previous: used bash]'}]}, {'role': 'assistant', 'content': [ThinkingBlock(signature='739e02453090504f3101fada13d5438a0c93e7a4bd394efc976e01b77bb31607', thinking='让我先读取README.md来了解这个项目的主要内容，然后再查看agents目录的结构。\n', type='thinking'), ToolUseBlock(id='call_function_fsf81jd63sw4_1', caller=None, input={'limit': 100, 'path': '/Users/ryanzhou/Desktop/learn-claude-code/README.md'}, name='read_file', type='tool_use'), ToolUseBlock(id='call_function_fsf81jd63sw4_2', caller=None, input={'command': 'ls -la /Users/ryanzhou/Desktop/learn-claude-code/agents'}, name='bash', type='tool_use'), ToolUseBlock(id='call_function_fsf81jd63sw4_3', caller=None, input={'command': 'ls -la /Users/ryanzhou/Desktop/learn-claude-code/docs'}, name='bash', type='tool_use'), ToolUseBlock(id='call_function_fsf81jd63sw4_4', caller=None, input={'command': 'ls -la /Users/ryanzhou/Desktop/learn-claude-code/skills'}, name='bash', type='tool_use')]}, {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_function_fsf81jd63sw4_1', 'content': '[English](./README.md) | [中文](./README-zh.md) | [日本語](./README-ja.md)\n# Learn Claude Code -- Harness Engineering for Real Agents\n\n## Agency Comes from the Model. An Agent Product = Model + Harness.\n\nBefore we talk about code, let\'s get one thing straight.\n\n**Agency -- the ability to perceive, reason, and act -- comes from model training, not from external code orchestration.** But a working agent product needs both the model and the harness. The model is the driver, the harness is the vehicle. This repo teaches you how to build the vehicle.\n\n### Where Agency Comes From\n\nAt the core of every agent is a neural network -- a Transformer, an RNN, a learned function -- that has been trained, through billions of gradient updates on action-sequence data, to perceive an environment, reason about goals, and take actions. Agency is never granted by the surrounding code. It is learned by the model during training.\n\nHumans are the best example. A biological neural network shaped by millions of years of evolutionary training, perceiving the world through senses, reasoning through a brain, acting through a body. When DeepMind, OpenAI, or Anthropic say "agent," the core of what they mean is always the same thing: **a model that has learned to act, plus the infrastructure that lets it operate in a specific environment.**\n\nThe proof is written in history:\n\n- **2013 -- DeepMind DQN plays Atari.** A single neural network, receiving only raw pixels and game scores, learned to play 7 Atari 2600 games -- surpassing all prior algorithms and beating human experts on 3 of them. By 2015, the same architecture scaled to [49 games and matched professional human testers](https://www.nature.com/articles/nature14236), published in *Nature*. No game-specific rules. No decision trees. One model, learning from experience. That model was the agent.\n\n- **2019 -- OpenAI Five conquers Dota 2.** Five neural networks, having played [45,000 years of Dota 2](https://openai.com/index/openai-five-defeats-dota-2-world-champions/) against themselves in 10 months, defeated **OG** -- the reigning TI8 world champions -- 2-0 on a San Francisco livestream. In a subsequent public arena, the AI won 99.4% of 42,729 games against all comers. No scripted strategies. No meta-programmed team coordination. The models learned teamwork, tactics, and real-time adaptation entirely through self-play.\n\n- **2019 -- DeepMind AlphaStar masters StarCraft II.** AlphaStar [beat professional players 10-1](https://deepmind.google/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii/) in a closed-door match, and later achieved [Grandmaster status](https://www.nature.com/articles/d41586-019-03298-6) on European servers -- top 0.15% of 90,000 players. A game with imperfect information, real-time decisions, and a combinatorial action space that dwarfs chess and Go. The agent? A model. Trained. Not scripted.\n\n- **2019 -- Tencent Jueyu dominates Honor of Kings.** Tencent AI Lab\'s "Jueyu" [defeated KPL professional players](https://www.jiemian.com/article/3371171.html) in a full 5v5 match at the World Champion Cup. In 1v1 mode, pros won only [1 out of 15 games and never survived past 8 minutes](https://developer.aliyun.com/article/851058). Training intensity: one day equaled 440 human years. By 2021, Jueyu surpassed KPL pros across the full hero pool. No handcrafted matchup tables. No scripted compositions. A model that learned the entire game from scratch through self-play.\n\n- **2024-2025 -- LLM agents reshape software engineering.** Claude, GPT, Gemini -- large language models trained on the entirety of human code and reasoning -- are deployed as coding agents. They read codebases, write implementations, debug failures, coordinate in teams. The architecture is identical to every agent before them: a trained model, placed in an environment, given tools to perceive and act. The only difference is the scale of what they\'ve learned and the generality of the tasks they solve.\n\nEvery one of these milestones points to the same fact: **agency -- the ability to perceive, reason, and act -- is trained, not coded.** But every agent also needed an environment to operate in: the Atari emulator, the Dota 2 client, the StarCraft II engine, the IDE and terminal. The model provides intelligence. The environment provides the action space. Together they form a complete agent.\n\n### What an Agent Is NOT\n\nThe word "agent" has been hijacked by an entire cottage industry of prompt plumbing.\n\nDrag-and-drop workflow builders. No-code "AI agent" platforms. Prompt-chain orchestration libraries. They all share the same delusion: that wiring together LLM API calls with if-else branches, node graphs, and hardcoded routing logic constitutes "building an agent."\n\nIt doesn\'t. What they build is a Rube Goldberg machine -- an over-engineered, brittle pipeline of procedural rules, with an LLM wedged in as a glorified text-completion node. That is not an agent. That is a shell script with delusions of grandeur.\n\n**Prompt plumbing "agents" are the fantasy of programmers who don\'t train models.** They attempt to brute-force intelligence by stacking procedural logic -- massive rule trees, node graphs, chain-of-prompt waterfalls -- and praying that enough glue code will somehow emergently produce autonomous behavior. It won\'t. You cannot engineer your way to agency. Agency is learned, not programmed.\n\nThose systems are dead on arrival: fragile, unscalable, fundamentally incapable of generalization. They are the modern resurrection of GOFAI (Good Old-Fashioned AI) -- the symbolic rule systems the field abandoned decades ago, now spray-painted with an LLM veneer. Different packaging, same dead end.\n\n### The Mind Shift: From "Developing Agents" to Developing Harness\n\nWhen someone says "I\'m developing an agent," they can only mean one of two things:\n\n**1. Training the model.** Adjusting weights through reinforcement learning, fine-tuning, RLHF, or other gradient-based methods. Collecting task-process data -- the actual sequences of perception, reasoning, and action in real domains -- and using it to shape the model\'s behavior. This is what DeepMind, OpenAI, Tencent AI Lab, and Anthropic do. This is agent development in the truest sense.\n\n**2. Building the harness.** Writing the code that gives the model an environment to operate in. This is what most of us do, and it is the focus of this repository.\n\nA harness is everything the agent needs to function in a specific domain:\n\n```\nHarness = Tools + Knowledge + Observation + Action Interfaces + Permissions\n\n    Tools:          file I/O, shell, network, database, browser\n    Knowledge:      product docs, domain references, API specs, style guides\n    Observation:    git diff, error logs, browser state, sensor data\n    Action:         CLI commands, API calls, UI interactions\n    Permissions:    sandboxing, approval workflows, trust boundaries\n```\n\nThe model decides. The harness executes. The model reasons. The harness provides context. The model is the driver. The harness is the vehicle.\n\n**A coding agent\'s harness is its IDE, terminal, and filesystem access.** A farm agent\'s harness is its sensor array, irrigation controls, and weather data feeds. A hotel agent\'s harness is its booking system, guest communication channels, and facility management APIs. The agent -- the intelligence, the decision-maker -- is always the model. The harness changes per domain. The agent generalizes across them.\n\nThis repo teaches you to build vehicles. Vehicles for coding. But the design patterns generalize to any domain: farm management, hotel operations, manufacturing, logistics, healthcare, education, scientific research. Anywhere a task needs to be perceived, reasoned about, and acted upon -- an agent needs a harness.\n\n### What Harness Engineers Actually Do\n\nIf you are reading this repository, you are likely a harness engineer -- and that is a powerful thing to be. Here is your real job:\n\n- **Implement tools.** Give the agent hands. File read/write, shell execution, API calls, browser control, database queries. Each tool is an action the agent can take in its environment. Design them to be atomic, composable, and well-described.\n\n- **Curate knowledge.** Give the agent domain expertise. Product documentation, architectural decision records, style guides, regulatory requirements. Load them on-demand (s05), not upfront. The agent should know what\'s available and pull what it needs.\n\n- **Manage context.** Give the agent clean memory. Subagent isolation (s04) prevents noise from leaking. Context compression (s06) prevents history from overwhelming. Task systems (s07) persist goals beyond any single conversation.\n\n- **Control permissions.** Give the agent boundaries. Sandbox file access. Require approval for destructive operations. Enforce trust boundaries between the agent and external systems. This is where safety engineering meets harness engineering.\n\n- **Collect task-process data.** Every action sequence the agent executes in your harness is training signal. The perception-reasoning-action traces from real deployments are the raw material for fine-tuning the next generation of agent models. Your harness doesn\'t just serve the agent -- it can help improve the agent.\n\nYou are not writing the intelligence. You are building the world the intelligence inhabits. The quality of that world -- how clearly the agent can perceive, how precisely it can act, how rich its available knowledge is -- directly determines how effectively the intelligence can express itself.\n\n**Build great harnesses. The agent will do the rest.**\n\n### Why Claude Code -- A Masterclass in Harness Engineering\n\nWhy does this repository dissect Claude Code specifically?\n\nBecause Claude Code is the most elegant and fully-realized agent harness we have seen. Not because of any single clever trick, but because of what it *doesn\'t* do: it doesn\'t try to be the agent. It doesn\'t impose rigid workflows. It doesn\'t second-guess the model with elaborate decision trees. It provides the model with tools, knowledge, context management, and permission boundaries -- then gets out of the way.\n\nLook at what Claude Code actually is, stripped to its essence:\n\n```\nClaude Code = one agent loop\n            + tools (bash, read, write, edit, glob, grep, browser...)\n            + on-demand skill loading\n            + context compression\n            + subagent spawning\n            + task system with dependency graph\n... (277 more)'}, {'type': 'tool_result', 'tool_use_id': 'call_function_fsf81jd63sw4_2', 'content': '[Previous: used bash]'}, {'type': 'tool_result', 'tool_use_id': 'call_function_fsf81jd63sw4_3', 'content': '[Previous: used bash]'}, {'type': 'tool_result', 'tool_use_id': 'call_function_fsf81jd63sw4_4', 'content': '[Previous: used bash]'}]}, {'role': 'assistant', 'content': [ThinkingBlock(signature='89df28afba515f35253050c8a69f5d2cc284d924dfe994b4921e518265cbee49', thinking='让我继续阅读更多文件来了解整个项目结构。\n', type='thinking'), ToolUseBlock(id='call_function_7ut9zy4wcfhq_1', caller=None, input={'path': '/Users/ryanzhou/Desktop/learn-claude-code/README.md'}, name='read_file', type='tool_use'), ToolUseBlock(id='call_function_7ut9zy4wcfhq_2', caller=None, input={'command': 'ls -la /Users/ryanzhou/Desktop/learn-claude-code/docs/zh'}, name='bash', type='tool_use')]}, {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_function_7ut9zy4wcfhq_1', 'content': '[English](./README.md) | [中文](./README-zh.md) | [日本語](./README-ja.md)\n# Learn Claude Code -- Harness Engineering for Real Agents\n\n## Agency Comes from the Model. An Agent Product = Model + Harness.\n\nBefore we talk about code, let\'s get one thing straight.\n\n**Agency -- the ability to perceive, reason, and act -- comes from model training, not from external code orchestration.** But a working agent product needs both the model and the harness. The model is the driver, the harness is the vehicle. This repo teaches you how to build the vehicle.\n\n### Where Agency Comes From\n\nAt the core of every agent is a neural network -- a Transformer, an RNN, a learned function -- that has been trained, through billions of gradient updates on action-sequence data, to perceive an environment, reason about goals, and take actions. Agency is never granted by the surrounding code. It is learned by the model during training.\n\nHumans are the best example. A biological neural network shaped by millions of years of evolutionary training, perceiving the world through senses, reasoning through a brain, acting through a body. When DeepMind, OpenAI, or Anthropic say "agent," the core of what they mean is always the same thing: **a model that has learned to act, plus the infrastructure that lets it operate in a specific environment.**\n\nThe proof is written in history:\n\n- **2013 -- DeepMind DQN plays Atari.** A single neural network, receiving only raw pixels and game scores, learned to play 7 Atari 2600 games -- surpassing all prior algorithms and beating human experts on 3 of them. By 2015, the same architecture scaled to [49 games and matched professional human testers](https://www.nature.com/articles/nature14236), published in *Nature*. No game-specific rules. No decision trees. One model, learning from experience. That model was the agent.\n\n- **2019 -- OpenAI Five conquers Dota 2.** Five neural networks, having played [45,000 years of Dota 2](https://openai.com/index/openai-five-defeats-dota-2-world-champions/) against themselves in 10 months, defeated **OG** -- the reigning TI8 world champions -- 2-0 on a San Francisco livestream. In a subsequent public arena, the AI won 99.4% of 42,729 games against all comers. No scripted strategies. No meta-programmed team coordination. The models learned teamwork, tactics, and real-time adaptation entirely through self-play.\n\n- **2019 -- DeepMind AlphaStar masters StarCraft II.** AlphaStar [beat professional players 10-1](https://deepmind.google/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii/) in a closed-door match, and later achieved [Grandmaster status](https://www.nature.com/articles/d41586-019-03298-6) on European servers -- top 0.15% of 90,000 players. A game with imperfect information, real-time decisions, and a combinatorial action space that dwarfs chess and Go. The agent? A model. Trained. Not scripted.\n\n- **2019 -- Tencent Jueyu dominates Honor of Kings.** Tencent AI Lab\'s "Jueyu" [defeated KPL professional players](https://www.jiemian.com/article/3371171.html) in a full 5v5 match at the World Champion Cup. In 1v1 mode, pros won only [1 out of 15 games and never survived past 8 minutes](https://developer.aliyun.com/article/851058). Training intensity: one day equaled 440 human years. By 2021, Jueyu surpassed KPL pros across the full hero pool. No handcrafted matchup tables. No scripted compositions. A model that learned the entire game from scratch through self-play.\n\n- **2024-2025 -- LLM agents reshape software engineering.** Claude, GPT, Gemini -- large language models trained on the entirety of human code and reasoning -- are deployed as coding agents. They read codebases, write implementations, debug failures, coordinate in teams. The architecture is identical to every agent before them: a trained model, placed in an environment, given tools to perceive and act. The only difference is the scale of what they\'ve learned and the generality of the tasks they solve.\n\nEvery one of these milestones points to the same fact: **agency -- the ability to perceive, reason, and act -- is trained, not coded.** But every agent also needed an environment to operate in: the Atari emulator, the Dota 2 client, the StarCraft II engine, the IDE and terminal. The model provides intelligence. The environment provides the action space. Together they form a complete agent.\n\n### What an Agent Is NOT\n\nThe word "agent" has been hijacked by an entire cottage industry of prompt plumbing.\n\nDrag-and-drop workflow builders. No-code "AI agent" platforms. Prompt-chain orchestration libraries. They all share the same delusion: that wiring together LLM API calls with if-else branches, node graphs, and hardcoded routing logic constitutes "building an agent."\n\nIt doesn\'t. What they build is a Rube Goldberg machine -- an over-engineered, brittle pipeline of procedural rules, with an LLM wedged in as a glorified text-completion node. That is not an agent. That is a shell script with delusions of grandeur.\n\n**Prompt plumbing "agents" are the fantasy of programmers who don\'t train models.** They attempt to brute-force intelligence by stacking procedural logic -- massive rule trees, node graphs, chain-of-prompt waterfalls -- and praying that enough glue code will somehow emergently produce autonomous behavior. It won\'t. You cannot engineer your way to agency. Agency is learned, not programmed.\n\nThose systems are dead on arrival: fragile, unscalable, fundamentally incapable of generalization. They are the modern resurrection of GOFAI (Good Old-Fashioned AI) -- the symbolic rule systems the field abandoned decades ago, now spray-painted with an LLM veneer. Different packaging, same dead end.\n\n### The Mind Shift: From "Developing Agents" to Developing Harness\n\nWhen someone says "I\'m developing an agent," they can only mean one of two things:\n\n**1. Training the model.** Adjusting weights through reinforcement learning, fine-tuning, RLHF, or other gradient-based methods. Collecting task-process data -- the actual sequences of perception, reasoning, and action in real domains -- and using it to shape the model\'s behavior. This is what DeepMind, OpenAI, Tencent AI Lab, and Anthropic do. This is agent development in the truest sense.\n\n**2. Building the harness.** Writing the code that gives the model an environment to operate in. This is what most of us do, and it is the focus of this repository.\n\nA harness is everything the agent needs to function in a specific domain:\n\n```\nHarness = Tools + Knowledge + Observation + Action Interfaces + Permissions\n\n    Tools:          file I/O, shell, network, database, browser\n    Knowledge:      product docs, domain references, API specs, style guides\n    Observation:    git diff, error logs, browser state, sensor data\n    Action:         CLI commands, API calls, UI interactions\n    Permissions:    sandboxing, approval workflows, trust boundaries\n```\n\nThe model decides. The harness executes. The model reasons. The harness provides context. The model is the driver. The harness is the vehicle.\n\n**A coding agent\'s harness is its IDE, terminal, and filesystem access.** A farm agent\'s harness is its sensor array, irrigation controls, and weather data feeds. A hotel agent\'s harness is its booking system, guest communication channels, and facility management APIs. The agent -- the intelligence, the decision-maker -- is always the model. The harness changes per domain. The agent generalizes across them.\n\nThis repo teaches you to build vehicles. Vehicles for coding. But the design patterns generalize to any domain: farm management, hotel operations, manufacturing, logistics, healthcare, education, scientific research. Anywhere a task needs to be perceived, reasoned about, and acted upon -- an agent needs a harness.\n\n### What Harness Engineers Actually Do\n\nIf you are reading this repository, you are likely a harness engineer -- and that is a powerful thing to be. Here is your real job:\n\n- **Implement tools.** Give the agent hands. File read/write, shell execution, API calls, browser control, database queries. Each tool is an action the agent can take in its environment. Design them to be atomic, composable, and well-described.\n\n- **Curate knowledge.** Give the agent domain expertise. Product documentation, architectural decision records, style guides, regulatory requirements. Load them on-demand (s05), not upfront. The agent should know what\'s available and pull what it needs.\n\n- **Manage context.** Give the agent clean memory. Subagent isolation (s04) prevents noise from leaking. Context compression (s06) prevents history from overwhelming. Task systems (s07) persist goals beyond any single conversation.\n\n- **Control permissions.** Give the agent boundaries. Sandbox file access. Require approval for destructive operations. Enforce trust boundaries between the agent and external systems. This is where safety engineering meets harness engineering.\n\n- **Collect task-process data.** Every action sequence the agent executes in your harness is training signal. The perception-reasoning-action traces from real deployments are the raw material for fine-tuning the next generation of agent models. Your harness doesn\'t just serve the agent -- it can help improve the agent.\n\nYou are not writing the intelligence. You are building the world the intelligence inhabits. The quality of that world -- how clearly the agent can perceive, how precisely it can act, how rich its available knowledge is -- directly determines how effectively the intelligence can express itself.\n\n**Build great harnesses. The agent will do the rest.**\n\n### Why Claude Code -- A Masterclass in Harness Engineering\n\nWhy does this repository dissect Claude Code specifically?\n\nBecause Claude Code is the most elegant and fully-realized agent harness we have seen. Not because of any single clever trick, but because of what it *doesn\'t* do: it doesn\'t try to be the agent. It doesn\'t impose rigid workflows. It doesn\'t second-guess the model with elaborate decision trees. It provides the model with tools, knowledge, context management, and permission boundaries -- then gets out of the way.\n\nLook at what Claude Code actually is, stripped to its essence:\n\n```\nClaude Code = one agent loop\n            + tools (bash, read, write, edit, glob, grep, browser...)\n            + on-demand skill loading\n            + context compression\n            + subagent spawning\n            + task system with dependency graph\n            + team coordination with async mailboxes\n            + worktree isolation for parallel execution\n            + permission governance\n```\n\nThat\'s it. That\'s the entire architecture. Every component is a harness mechanism -- a piece of the world built for the agent to inhabit. The agent itself? It\'s Claude. A model. Trained by Anthropic on the full breadth of human reasoning and code. The harness doesn\'t make Claude smart. Claude is already smart. The harness gives Claude hands, eyes, and a workspace.\n\nThis is why Claude Code is the ideal teaching subject: **it demonstrates what happens when you trust the model and focus your engineering on the harness.** Every session in this repository (s01-s12) reverse-engineers one harness mechanism from Claude Code\'s architecture. By the end, you understand not just how Claude Code works, but the universal principles of harness engineering that apply to any agent in any domain.\n\nThe lesson is not "copy Claude Code." The lesson is: **the best agent products are built by engineers who understand that their job is harness, not intelligence.**\n\n---\n\n## The Vision: Fill the Universe with Real Agents\n\nThis is not just about coding agents.\n\nEvery domain where humans perform complex, multi-step, judgment-intensive work is a domain where agents can operate -- given the right harness. The patterns in this repository are universal:\n\n```\nEstate management agent    = model + property sensors + maintenance tools + tenant comms\nAgricultural agent         = model + soil/weather data + irrigation controls + crop knowledge\nHotel operations agent     = model + booking system + guest channels + facility APIs\nMedical research agent     = model + literature search + lab instruments + protocol docs\nManufacturing agent        = model + production line sensors + quality controls + logistics\nEducation agent            = model + curriculum knowledge + student progress + assessment tools\n```\n\nThe loop is always the same. The tools change. The knowledge changes. The permissions change. The agent -- the model -- generalizes.\n\nEvery harness engineer reading this repository is learning patterns that apply far beyond software engineering. You are learning to build the infrastructure for an intelligent, automated future. Every well-designed harness deployed in a real domain is one more place where an agent can perceive, reason, and act.\n\nFirst we fill the workshops. Then the farms, the hospitals, the factories. Then the cities. Then the planet.\n\n**Bash is all you need. Real agents are all the universe needs.**\n\n---\n\n```\n                    THE AGENT PATTERN\n                    =================\n\n    User --> messages[] --> LLM --> response\n                                      |\n                            stop_reason == "tool_use"?\n                           /                          \\\n                         yes                           no\n                          |                             |\n                    execute tools                    return text\n                    append results\n                    loop back -----------------> messages[]\n\n\n    That\'s the minimal loop. Every AI agent needs this loop.\n    The MODEL decides when to call tools and when to stop.\n    The CODE just executes what the model asks for.\n    This repo teaches you to build what surrounds this loop --\n    the harness that makes the agent effective in a specific domain.\n```\n\n**12 progressive sessions, from a simple loop to isolated autonomous execution.**\n**Each session adds one harness mechanism. Each mechanism has one motto.**\n\n> **s01** &nbsp; *"One loop & Bash is all you need"* &mdash; one tool + one loop = an agent\n>\n> **s02** &nbsp; *"Adding a tool means adding one handler"* &mdash; the loop stays the same; new tools register into the dispatch map\n>\n> **s03** &nbsp; *"An agent without a plan drifts"* &mdash; list the steps first, then execute; completion doubles\n>\n> **s04** &nbsp; *"Break big tasks down; each subtask gets a clean context"* &mdash; subagents use independent messages[], keeping the main conversation clean\n>\n> **s05** &nbsp; *"Load knowledge when you need it, not upfront"* &mdash; inject via tool_result, not the system prompt\n>\n> **s06** &nbsp; *"Context will fill up; you need a way to make room"* &mdash; three-layer compression strategy for infinite sessions\n>\n> **s07** &nbsp; *"Break big goals into small tasks, order them, persist to disk"* &mdash; a file-based task graph with dependencies, laying the foundation for multi-agent collaboration\n>\n> **s08** &nbsp; *"Run slow operations in the background; the agent keeps thinking"* &mdash; daemon threads run commands, inject notifications on completion\n>\n> **s09** &nbsp; *"When the task is too big for one, delegate to teammates"* &mdash; persistent teammates + async mailboxes\n>\n> **s10** &nbsp; *"Teammates need shared communication rules"* &mdash; one request-response pattern drives all negotiation\n>\n> **s11** &nbsp; *"Teammates scan the board and claim tasks themselves"* &mdash; no need for the lead to assign each one\n>\n> **s12** &nbsp; *"Each works in its own directory, no interference"* &mdash; tasks manage goals, worktrees manage directories, bound by ID\n\n---\n\n## The Core Pattern\n\n```python\ndef agent_loop(messages):\n    while True:\n        response = client.messages.create(\n            model=MODEL, system=SYSTEM,\n            messages=messages, tools=TOOLS,\n        )\n        messages.append({"role": "assistant",\n                         "content": response.content})\n\n        if response.stop_reason != "tool_use":\n            return\n\n        results = []\n        for block in response.content:\n            if block.type == "tool_use":\n                output = TOOL_HANDLERS[block.name](**block.input)\n                results.append({\n                    "type": "tool_result",\n                    "tool_use_id": block.id,\n                    "content": output,\n                })\n        messages.append({"role": "user", "content": results})\n```\n\nEvery session layers one harness mechanism on top of this loop -- without changing the loop itself. The loop belongs to the agent. The mechanisms belong to the harness.\n\n## Scope (Important)\n\nThis repository is a 0->1 learning project for harness engineering -- building the environment that surrounds an agent model.\nIt intentionally simplifies or omits several production mechanisms:\n\n- Full event/hook buses (for example PreToolUse, SessionStart/End, ConfigChange).\n  s12 includes only a minimal append-only lifecycle event stream for teaching.\n- Rule-based permission governance and trust workflows\n- Session lifecycle controls (resume/fork) and advanced worktree lifecycle controls\n- Full MCP runtime details (transport/OAuth/resource subscribe/polling)\n\nTreat the team JSONL mailbox protocol in this repo as a teaching implementation, not a claim about any specific production internals.\n\n## Quick Start\n\n```sh\ngit clone https://github.com/shareAI-lab/learn-claude-code\ncd learn-claude-code\npip install -r requirements.txt\ncp .env.example .env   # Edit .env with your ANTHROPIC_API_KEY\n\npython agents/s01_agent_loop.py       # Start here\npython agents/s12_worktree_task_isolation.py  # Full progression endpoint\npython agents/s_full.py               # Capstone: all mechanisms combined\n```\n\n### Web Platform\n\nInteractive visualizations, step-through diagrams, source viewer, and documentation.\n\n```sh\ncd web && npm install && npm run dev   # http://localhost:3000\n```\n\n## Learning Path\n\n```\nPhase 1: THE LOOP                    Phase 2: PLANNING & KNOWLEDGE\n==================                   ==============================\ns01  The Agent Loop          [1]     s03  TodoWrite               [5]\n     while + stop_reason                  TodoManager + nag reminder\n     |                                    |\n     +-> s02  Tool Use            [4]     s04  Subagents            [5]\n              dispatch map: name->handler     fresh messages[] per child\n                                              |\n                                         s05  Skills               [5]\n                                              SKILL.md via tool_result\n                                              |\n                                         s06  Context Compact      [5]\n                                              3-layer compression\n\nPhase 3: PERSISTENCE                 Phase 4: TEAMS\n==================                   =====================\ns07  Tasks                   [8]     s09  Agent Teams             [9]\n     file-based CRUD + deps graph         teammates + JSONL mailboxes\n     |                                    |\ns08  Background Tasks        [6]     s10  Team Protocols          [12]\n     daemon threads + notify queue        shutdown + plan approval FSM\n                                          |\n                                     s11  Autonomous Agents       [14]\n                                          idle cycle + auto-claim\n                                     |\n                                     s12  Worktree Isolation      [16]\n                                          task coordination + optional isolated execution lanes\n\n                                     [N] = number of tools\n```\n\n## Architecture\n\n```\nlearn-claude-code/\n|\n|-- agents/                        # Python reference implementations (s01-s12 + s_full capstone)\n|-- docs/{en,zh,ja}/               # Mental-model-first documentation (3 languages)\n|-- web/                           # Interactive learning platform (Next.js)\n|-- skills/                        # Skill files for s05\n+-- .github/workflows/ci.yml      # CI: typecheck + build\n```\n\n## Documentation\n\nMental-model-first: problem, solution, ASCII diagram, minimal code.\nAvailable in [English](./docs/en/) | [中文](./docs/zh/) | [日本語](./docs/ja/).\n\n| Session | Topic | Motto |\n|---------|-------|-------|\n| [s01](./docs/en/s01-the-agent-loop.md) | The Agent Loop | *One loop & Bash is all you need* |\n| [s02](./docs/en/s02-tool-use.md) | Tool Use | *Adding a tool means adding one handler* |\n| [s03](./docs/en/s03-todo-write.md) | TodoWrite | *An agent without a plan drifts* |\n| [s04](./docs/en/s04-subagent.md) | Subagents | *Break big tasks down; each subtask gets a clean context* |\n| [s05](./docs/en/s05-skill-loading.md) | Skills | *Load knowledge when you need it, not upfront* |\n| [s06](./docs/en/s06-context-compact.md) | Context Compact | *Context will fill up; you need a way to make room* |\n| [s07](./docs/en/s07-task-system.md) | Tasks | *Break big goals into small tasks, order them, persist to disk* |\n| [s08](./docs/en/s08-background-tasks.md) | Background Tasks | *Run slow operations in the background; the agent keeps thinking* |\n| [s09](./docs/en/s09-agent-teams.md) | Agent Teams | *When the task is too big for one, delegate to teammates* |\n| [s10](./docs/en/s10-team-protocols.md) | Team Protocols | *Teammates need shared communication rules* |\n| [s11](./docs/en/s11-autonomous-agents.md) | Autonomous Agents | *Teammates scan the board and claim tasks themselves* |\n| [s12](./docs/en/s12-worktree-task-isolation.md) | Worktree + Task Isolation | *Each works in its own directory, no interference* |\n\n## What\'s Next -- from understanding to shipping\n\nAfter the 12 sessions you understand how harness engineering works inside out. Two ways to put that knowledge to work:\n\n### Kode Agent CLI -- Open-Source Coding Agent CLI\n\n> `npm i -g @shareai-lab/kode`\n\nSkill & LSP support, Windows-ready, pluggable with GLM / MiniMax / DeepSeek and other open models. Install and go.\n\nGitHub: **[shareAI-lab/Kode-cli](https://github.com/shareAI-lab/Kode-cli)**\n\n### Kode Agent SDK -- Embed Agent Capabilities in Your App\n\nThe official Claude Code Agent SDK communicates with a full CLI process under the hood -- each concurrent user means a separate terminal process. Kode SDK is a standalone library with no per-user process overhead, embeddable in backends, browser extensions, embedded devices, or any runtime.\n\nGitHub: **[shareAI-lab/Kode-agent-sdk](https://github.com/shareAI-lab/Kode-agent-sdk)**\n\n---\n\n## Sister Repo: from *on-demand sessions* to *always-on assistant*\n\nThe harness this repo teaches is **use-and-discard** -- open a terminal, give the agent a task, close when done, next session starts blank. That is the Claude Code model.\n\n[OpenClaw](https://github.com/openclaw/openclaw) proved another possibility: on top of the same agent core, two harness mechanisms turn the agent from "poke it to make it move" into "it wakes up every 30 seconds to look for work":\n\n- **Heartbeat** -- every 30s the harness sends the agent a message to check if there is anything to do. Nothing? Go back to sleep. Something? Act immediately.\n- **Cron** -- the agent can schedule its own future tasks, executed automatically when the time comes.\n\nAdd multi-channel IM routing (WhatsApp / Telegram / Slack / Discord, 13+ platforms), persistent context memory, and a Soul personality system, and the agent goes from a disposable tool to an always-on personal AI assistant.\n\n**[claw0](https://github.com/shareAI-lab/claw0)** is our companion teaching repo that deconstructs these harness mechanisms from scratch:\n\n```\nclaw agent = agent core + heartbeat + cron + IM chat + memory + soul\n```\n\n```\nlearn-claude-code                   claw0\n(agent harness core:                (proactive always-on harness:\n loop, tools, planning,              heartbeat, cron, IM channels,\n teams, worktree isolation)          memory, soul personality)\n```\n\n## About\n<img width="260" src="https://github.com/user-attachments/assets/fe8b852b-97da-4061-a467-9694906b5edf" /><br>\n\nScan with WeChat to follow us,\nor follow on X: [shareAI-Lab](https://x.com/baicai003)\n\n## License\n\nMIT\n\n---\n\n**Agency comes from the model. The harness makes agency real. Build great harnesses. The model will do the rest.**\n\n**Bash is all you need. Real agents are all the universe needs.**'}, {'type': 'tool_result', 'tool_use_id': 'call_function_7ut9zy4wcfhq_2', 'content': '[Previous: used bash]'}]}, {'role': 'assistant', 'content': [ThinkingBlock(signature='bc4b69d9cf7cb522e89a2f246c0076425a63e774e129c10fecf0a6afc2ef3eff', thinking='Let me read a few of the documentation files to get a better understanding of the content and structure, so I can create a comprehensive AGENTS.md summary.\n', type='thinking'), ToolUseBlock(id='call_function_a0gmsd5d31u3_1', caller=None, input={'path': '/Users/ryanzhou/Desktop/learn-claude-code/docs/zh/s01-the-agent-loop.md'}, name='read_file', type='tool_use'), ToolUseBlock(id='call_function_a0gmsd5d31u3_2', caller=None, input={'path': '/Users/ryanzhou/Desktop/learn-claude-code/docs/zh/s07-task-system.md'}, name='read_file', type='tool_use'), ToolUseBlock(id='call_function_a0gmsd5d31u3_3', caller=None, input={'path': '/Users/ryanzhou/Desktop/learn-claude-code/docs/zh/s09-agent-teams.md'}, name='read_file', type='tool_use')]}, {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_function_a0gmsd5d31u3_1', 'content': '# s01: The Agent Loop (Agent 循环)\n\n`[ s01 ] s02 > s03 > s04 > s05 > s06 | s07 > s08 > s09 > s10 > s11 > s12`\n\n> *"One loop & Bash is all you need"* -- 一个工具 + 一个循环 = 一个 Agent。\n>\n> **Harness 层**: 循环 -- 模型与真实世界的第一道连接。\n\n## 问题\n\n语言模型能推理代码, 但碰不到真实世界 -- 不能读文件、跑测试、看报错。没有循环, 每次工具调用你都得手动把结果粘回去。你自己就是那个循环。\n\n## 解决方案\n\n```\n+--------+      +-------+      +---------+\n|  User  | ---> |  LLM  | ---> |  Tool   |\n| prompt |      |       |      | execute |\n+--------+      +---+---+      +----+----+\n                    ^                |\n                    |   tool_result  |\n                    +----------------+\n                    (loop until stop_reason != "tool_use")\n```\n\n一个退出条件控制整个流程。循环持续运行, 直到模型不再调用工具。\n\n## 工作原理\n\n1. 用户 prompt 作为第一条消息。\n\n```python\nmessages.append({"role": "user", "content": query})\n```\n\n2. 将消息和工具定义一起发给 LLM。\n\n```python\nresponse = client.messages.create(\n    model=MODEL, system=SYSTEM, messages=messages,\n    tools=TOOLS, max_tokens=8000,\n)\n```\n\n3. 追加助手响应。检查 `stop_reason` -- 如果模型没有调用工具, 结束。\n\n```python\nmessages.append({"role": "assistant", "content": response.content})\nif response.stop_reason != "tool_use":\n    return\n```\n\n4. 执行每个工具调用, 收集结果, 作为 user 消息追加。回到第 2 步。\n\n```python\nresults = []\nfor block in response.content:\n    if block.type == "tool_use":\n        output = run_bash(block.input["command"])\n        results.append({\n            "type": "tool_result",\n            "tool_use_id": block.id,\n            "content": output,\n        })\nmessages.append({"role": "user", "content": results})\n```\n\n组装为一个完整函数:\n\n```python\ndef agent_loop(query):\n    messages = [{"role": "user", "content": query}]\n    while True:\n        response = client.messages.create(\n            model=MODEL, system=SYSTEM, messages=messages,\n            tools=TOOLS, max_tokens=8000,\n        )\n        messages.append({"role": "assistant", "content": response.content})\n\n        if response.stop_reason != "tool_use":\n            return\n\n        results = []\n        for block in response.content:\n            if block.type == "tool_use":\n                output = run_bash(block.input["command"])\n                results.append({\n                    "type": "tool_result",\n                    "tool_use_id": block.id,\n                    "content": output,\n                })\n        messages.append({"role": "user", "content": results})\n```\n\n不到 30 行, 这就是整个 Agent。后面 11 个章节都在这个循环上叠加机制 -- 循环本身始终不变。\n\n## 变更内容\n\n| 组件          | 之前       | 之后                           |\n|---------------|------------|--------------------------------|\n| Agent loop    | (无)       | `while True` + stop_reason     |\n| Tools         | (无)       | `bash` (单一工具)              |\n| Messages      | (无)       | 累积式消息列表                 |\n| Control flow  | (无)       | `stop_reason != "tool_use"`    |\n\n## 试一试\n\n```sh\ncd learn-claude-code\npython agents/s01_agent_loop.py\n```\n\n试试这些 prompt (英文 prompt 对 LLM 效果更好, 也可以用中文):\n\n1. `Create a file called hello.py that prints "Hello, World!"`\n2. `List all Python files in this directory`\n3. `What is the current git branch?`\n4. `Create a directory called test_output and write 3 files in it`'}, {'type': 'tool_result', 'tool_use_id': 'call_function_a0gmsd5d31u3_2', 'content': '# s07: Task System (任务系统)\n\n`s01 > s02 > s03 > s04 > s05 > s06 | [ s07 ] s08 > s09 > s10 > s11 > s12`\n\n> *"大目标要拆成小任务, 排好序, 记在磁盘上"* -- 文件持久化的任务图, 为多 agent 协作打基础。\n>\n> **Harness 层**: 持久化任务 -- 比任何一次对话都长命的目标。\n\n## 问题\n\ns03 的 TodoManager 只是内存中的扁平清单: 没有顺序、没有依赖、状态只有做完没做完。真实目标是有结构的 -- 任务 B 依赖任务 A, 任务 C 和 D 可以并行, 任务 E 要等 C 和 D 都完成。\n\n没有显式的关系, Agent 分不清什么能做、什么被卡住、什么能同时跑。而且清单只活在内存里, 上下文压缩 (s06) 一跑就没了。\n\n## 解决方案\n\n把扁平清单升级为持久化到磁盘的**任务图**。每个任务是一个 JSON 文件, 有状态、前置依赖 (`blockedBy`)。任务图随时回答三个问题:\n\n- **什么可以做?** -- 状态为 `pending` 且 `blockedBy` 为空的任务。\n- **什么被卡住?** -- 等待前置任务完成的任务。\n- **什么做完了?** -- 状态为 `completed` 的任务, 完成时自动解锁后续任务。\n\n```\n.tasks/\n  task_1.json  {"id":1, "status":"completed"}\n  task_2.json  {"id":2, "blockedBy":[1], "status":"pending"}\n  task_3.json  {"id":3, "blockedBy":[1], "status":"pending"}\n  task_4.json  {"id":4, "blockedBy":[2,3], "status":"pending"}\n\n任务图 (DAG):\n                 +----------+\n            +--> | task 2   | --+\n            |    | pending  |   |\n+----------+     +----------+    +--> +----------+\n| task 1   |                          | task 4   |\n| completed| --> +----------+    +--> | blocked  |\n+----------+     | task 3   | --+     +----------+\n                 | pending  |\n                 +----------+\n\n顺序:   task 1 必须先完成, 才能开始 2 和 3\n并行:   task 2 和 3 可以同时执行\n依赖:   task 4 要等 2 和 3 都完成\n状态:   pending -> in_progress -> completed\n```\n\n这个任务图是 s07 之后所有机制的协调骨架: 后台执行 (s08)、多 agent 团队 (s09+)、worktree 隔离 (s12) 都读写这同一个结构。\n\n## 工作原理\n\n1. **TaskManager**: 每个任务一个 JSON 文件, CRUD + 依赖图。\n\n```python\nclass TaskManager:\n    def __init__(self, tasks_dir: Path):\n        self.dir = tasks_dir\n        self.dir.mkdir(exist_ok=True)\n        self._next_id = self._max_id() + 1\n\n    def create(self, subject, description=""):\n        task = {"id": self._next_id, "subject": subject,\n                "status": "pending", "blockedBy": [],\n                "owner": ""}\n        self._save(task)\n        self._next_id += 1\n        return json.dumps(task, indent=2)\n```\n\n2. **依赖解除**: 完成任务时, 自动将其 ID 从其他任务的 `blockedBy` 中移除, 解锁后续任务。\n\n```python\ndef _clear_dependency(self, completed_id):\n    for f in self.dir.glob("task_*.json"):\n        task = json.loads(f.read_text())\n        if completed_id in task.get("blockedBy", []):\n            task["blockedBy"].remove(completed_id)\n            self._save(task)\n```\n\n3. **状态变更 + 依赖关联**: `update` 处理状态转换和依赖边。\n\n```python\ndef update(self, task_id, status=None,\n           add_blocked_by=None, remove_blocked_by=None):\n    task = self._load(task_id)\n    if status:\n        task["status"] = status\n        if status == "completed":\n            self._clear_dependency(task_id)\n    if add_blocked_by:\n        task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by))\n    if remove_blocked_by:\n        task["blockedBy"] = [x for x in task["blockedBy"] if x not in remove_blocked_by]\n    self._save(task)\n```\n\n4. 四个任务工具加入 dispatch map。\n\n```python\nTOOL_HANDLERS = {\n    # ...base tools...\n    "task_create": lambda **kw: TASKS.create(kw["subject"]),\n    "task_update": lambda **kw: TASKS.update(kw["task_id"], kw.get("status")),\n    "task_list":   lambda **kw: TASKS.list_all(),\n    "task_get":    lambda **kw: TASKS.get(kw["task_id"]),\n}\n```\n\n从 s07 起, 任务图是多步工作的默认选择。s03 的 Todo 仍可用于单次会话内的快速清单。\n\n## 相对 s06 的变更\n\n| 组件 | 之前 (s06) | 之后 (s07) |\n|---|---|---|\n| Tools | 5 | 8 (`task_create/update/list/get`) |\n| 规划模型 | 扁平清单 (仅内存) | 带依赖关系的任务图 (磁盘) |\n| 关系 | 无 | `blockedBy` 边 |\n| 状态追踪 | 做完没做完 | `pending` -> `in_progress` -> `completed` |\n| 持久化 | 压缩后丢失 | 压缩和重启后存活 |\n\n## 试一试\n\n```sh\ncd learn-claude-code\npython agents/s07_task_system.py\n```\n\n试试这些 prompt (英文 prompt 对 LLM 效果更好, 也可以用中文):\n\n1. `Create 3 tasks: "Setup project", "Write code", "Write tests". Make them depend on each other in order.`\n2. `List all tasks and show the dependency graph`\n3. `Complete task 1 and then list tasks to see task 2 unblocked`\n4. `Create a task board for refactoring: parse -> transform -> emit -> test, where transform and emit can run in parallel after parse`'}, {'type': 'tool_result', 'tool_use_id': 'call_function_a0gmsd5d31u3_3', 'content': '# s09: Agent Teams (Agent 团队)\n\n`s01 > s02 > s03 > s04 > s05 > s06 | s07 > s08 > [ s09 ] s10 > s11 > s12`\n\n> *"任务太大一个人干不完, 要能分给队友"* -- 持久化队友 + JSONL 邮箱。\n>\n> **Harness 层**: 团队邮箱 -- 多个模型, 通过文件协调。\n\n## 问题\n\nSubagent (s04) 是一次性的: 生成、干活、返回摘要、消亡。没有身份, 没有跨调用的记忆。Background Tasks (s08) 能跑 shell 命令, 但做不了 LLM 引导的决策。\n\n真正的团队协作需要三样东西: (1) 能跨多轮对话存活的持久 Agent, (2) 身份和生命周期管理, (3) Agent 之间的通信通道。\n\n## 解决方案\n\n```\nTeammate lifecycle:\n  spawn -> WORKING -> IDLE -> WORKING -> ... -> SHUTDOWN\n\nCommunication:\n  .team/\n    config.json           <- team roster + statuses\n    inbox/\n      alice.jsonl         <- append-only, drain-on-read\n      bob.jsonl\n      lead.jsonl\n\n              +--------+    send("alice","bob","...")    +--------+\n              | alice  | -----------------------------> |  bob   |\n              | loop   |    bob.jsonl << {json_line}    |  loop  |\n              +--------+                                +--------+\n                   ^                                         |\n                   |        BUS.read_inbox("alice")          |\n                   +---- alice.jsonl -> read + drain ---------+\n```\n\n## 工作原理\n\n1. TeammateManager 通过 config.json 维护团队名册。\n\n```python\nclass TeammateManager:\n    def __init__(self, team_dir: Path):\n        self.dir = team_dir\n        self.dir.mkdir(exist_ok=True)\n        self.config_path = self.dir / "config.json"\n        self.config = self._load_config()\n        self.threads = {}\n```\n\n2. `spawn()` 创建队友并在线程中启动 agent loop。\n\n```python\ndef spawn(self, name: str, role: str, prompt: str) -> str:\n    member = {"name": name, "role": role, "status": "working"}\n    self.config["members"].append(member)\n    self._save_config()\n    thread = threading.Thread(\n        target=self._teammate_loop,\n        args=(name, role, prompt), daemon=True)\n    thread.start()\n    return f"Spawned teammate \'{name}\' (role: {role})"\n```\n\n3. MessageBus: append-only 的 JSONL 收件箱。`send()` 追加一行; `read_inbox()` 读取全部并清空。\n\n```python\nclass MessageBus:\n    def send(self, sender, to, content, msg_type="message", extra=None):\n        msg = {"type": msg_type, "from": sender,\n               "content": content, "timestamp": time.time()}\n        if extra:\n            msg.update(extra)\n        with open(self.dir / f"{to}.jsonl", "a") as f:\n            f.write(json.dumps(msg) + "\\n")\n\n    def read_inbox(self, name):\n        path = self.dir / f"{name}.jsonl"\n        if not path.exists(): return "[]"\n        msgs = [json.loads(l) for l in path.read_text().strip().splitlines() if l]\n        path.write_text("")  # drain\n        return json.dumps(msgs, indent=2)\n```\n\n4. 每个队友在每次 LLM 调用前检查收件箱, 将消息注入上下文。\n\n```python\ndef _teammate_loop(self, name, role, prompt):\n    messages = [{"role": "user", "content": prompt}]\n    for _ in range(50):\n        inbox = BUS.read_inbox(name)\n        if inbox != "[]":\n            messages.append({"role": "user",\n                "content": f"<inbox>{inbox}</inbox>"})\n        response = client.messages.create(...)\n        if response.stop_reason != "tool_use":\n            break\n        # execute tools, append results...\n    self._find_member(name)["status"] = "idle"\n```\n\n## 相对 s08 的变更\n\n| 组件           | 之前 (s08)       | 之后 (s09)                         |\n|----------------|------------------|------------------------------------|\n| Tools          | 6                | 9 (+spawn/send/read_inbox)         |\n| Agent 数量     | 单一             | 领导 + N 个队友                    |\n| 持久化         | 无               | config.json + JSONL 收件箱         |\n| 线程           | 后台命令         | 每线程完整 agent loop              |\n| 生命周期       | 一次性           | idle -> working -> idle            |\n| 通信           | 无               | message + broadcast                |\n\n## 试一试\n\n```sh\ncd learn-claude-code\npython agents/s09_agent_teams.py\n```\n\n试试这些 prompt (英文 prompt 对 LLM 效果更好, 也可以用中文):\n\n1. `Spawn alice (coder) and bob (tester). Have alice send bob a message.`\n2. `Broadcast "status update: phase 1 complete" to all teammates`\n3. `Check the lead inbox for any messages`\n4. 输入 `/team` 查看团队名册和状态\n5. 输入 `/inbox` 手动检查领导的收件箱'}]}, {'role': 'assistant', 'content': [ThinkingBlock(signature='96aca11dcec19b664e56c1ec6648932e636ebbbc8d4c252e84f0e8ee06f6506c', thinking='Now I have a good understanding of the repository. Let me create a comprehensive AGENTS.md file that summarizes the key concepts and structure of this learn-claude-code repository.\n', type='thinking'), ToolUseBlock(id='call_function_nnwrr1tv8kaq_1', caller=None, input={'content': '# Learn Claude Code -- AGENTS 总结\n\n## 核心概念：Agent = Model + Harness\n\n**Agency（智能性）来自模型训练，而非外部代码编排。**\n\nAgent 产品 = 模型 + Harness（驾驭系统）\n- **模型**：决定者、推理者、驾驶员\n- **Harness**：工具 + 知识 + 观察 + 操作接口 + 权限\n\n```\nHarness = Tools + Knowledge + Observation + Action Interfaces + Permissions\n```\n\n---\n\n## Agent 模式图\n\n```\nUser --> messages[] --> LLM --> response\n                               |\n                     stop_reason == "tool_use"?\n                    /                          \\\n                  yes                           no\n                   |                             |\n             execute tools                  return text\n             append results\n             loop back -----------------> messages[]\n```\n\n---\n\n## 12 阶段渐进学习路径\n\n| Session | 主题 | 核心理念 | 新增工具数 |\n|---------|------|----------|------------|\n| **s01** | Agent Loop | *One loop & Bash is all you need* | 1 |\n| **s02** | Tool Use | *Adding a tool means adding one handler* | +2 |\n| **s03** | TodoWrite | *An agent without a plan drifts* | +3 |\n| **s04** | Subagents | *Break big tasks down; each subtask gets a clean context* | +2 |\n| **s05** | Skills | *Load knowledge when you need it, not upfront* | +2 |\n| **s06** | Context Compact | *Context will fill up; you need a way to make room* | +1 |\n| **s07** | Tasks | *Break big goals into small tasks, order them, persist to disk* | +4 |\n| **s08** | Background Tasks | *Run slow operations in the background; the agent keeps thinking* | +1 |\n| **s09** | Agent Teams | *When the task is too big for one, delegate to teammates* | +3 |\n| **s10** | Team Protocols | *Teammates need shared communication rules* | +3 |\n| **s11** | Autonomous Agents | *Teammates scan the board and claim tasks themselves* | +1 |\n| **s12** | Worktree + Task Isolation | *Each works in its own directory, no interference* | +6 |\n\n---\n\n## 架构概览\n\n```\nlearn-claude-code/\n|\n|-- agents/              # Python 参考实现 (s01-s12 + s_full)\n|-- docs/{en,zh,ja}/     # 思维模型优先的文档 (3种语言)\n|-- web/                  # 交互式学习平台 (Next.js)\n|-- skills/               # Skill 文件 (s05)\n+-- .github/workflows/    # CI: typecheck + build\n```\n\n---\n\n## 关键机制详解\n\n### s01-s06: 基础能力构建\n\n- **s01 Agent Loop**: 核心循环，`while True` + `stop_reason` 控制\n- **s02 Tool Use**: 工具注册到 dispatch map，按名称路由\n- **s03 TodoWrite**: 先列步骤再执行，效率翻倍\n- **s04 Subagents**: 子 agent 独立 `messages[]`，主对话保持干净\n- **s05 Skills**: 通过 `tool_result` 注入知识，而非 system prompt\n- **s06 Context Compact**: 三层压缩策略，支持无限会话\n\n### s07-s12: 团队与持久化\n\n- **s07 Tasks**: 文件持久化任务图，支持依赖关系（DAG）\n- **s08 Background Tasks**: 守护线程执行慢操作，完成时注入通知\n- **s09 Agent Teams**: 持久化队友 + JSONL 邮箱通信\n- **s10 Team Protocols**: 共享通信规则（shutdown、计划审批 FSM）\n- **s11 Autonomous Agents**: 队友自主扫描任务板并认领\n- **s12 Worktree Isolation**: 任务管理目标，worktree 管理目录，绑定 ID\n\n---\n\n## 团队协作模式\n\n```\n.teammate/\n  config.json           <- 团队名册 + 状态\n  inbox/\n    alice.jsonl         <- 追加式只读，读取后清空\n    bob.jsonl\n    lead.jsonl\n\n生命周期:\n  spawn -> WORKING -> IDLE -> WORKING -> ... -> SHUTDOWN\n```\n\n---\n\n## 快速开始\n\n```bash\ngit clone https://github.com/shareAI-lab/learn-claude-code\ncd learn-claude-code\npip install -r requirements.txt\ncp .env.example .env   # 填入 ANTHROPIC_API_KEY\n\npython agents/s01_agent_loop.py       # 从这里开始\npython agents/s12_worktree_task_isolation.py  # 完整进度终点\npython agents/s_full.py               # 集大成者：所有机制合一\n```\n\n---\n\n## 延伸项目\n\n- **Kode Agent CLI**: `npm i -g @shareai-lab/kode` - 开源 Coding Agent CLI\n- **Kode Agent SDK**: 嵌入式 Agent 能力库，无单用户进程开销\n- **claw0**: 从"按需会话"到"常驻助手"，心跳 + 定时 + IM 通道 + 记忆 + 灵魂\n\n---\n\n## 核心教训\n\n> **Agency comes from the model. The harness makes agency real.**\n> \n> **Build great harnesses. The model will do the rest.**\n> \n> **Bash is all you need. Real agents are all the universe needs.**', 'path': '/Users/ryanzhou/Desktop/learn-claude-code/AGENTS.md'}, name='write_file', type='tool_use')]}, {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_function_nnwrr1tv8kaq_1', 'content': 'Wrote 3682 bytes'}]}, {'role': 'assistant', 'content': [ThinkingBlock(signature='070ccca8f154b299110f945f03eed2fa20f256d699fe9b38f2f0a7ae81e1173d', thinking='AGENTS.md 已经创建完成，包含了以下内容：\n\n1. 核心概念：Agent = Model + Harness\n2. Agent 模式图\n3. 12 阶段渐进学习路径（表格形式）\n4. 架构概览\n5. 关键机制详解（分为两阶段：s01-s06 基础能力构建，s07-s12 团队与持久化）\n6. 团队协作模式\n7. 快速开始\n8. 延伸项目\n9. 核心教训\n\n现在让我验证一下文件是否正确创建。\n', type='thinking'), TextBlock(citations=None, text='已完成！创建了 `/Users/ryanzhou/Desktop/learn-claude-code/AGENTS.md`，内容包括：\n\n## AGENTS.md 结构\n\n| 章节 | 内容 |\n|------|------|\n| **核心概念** | Agent = Model + Harness 的核心理念 |\n| **模式图** | ASCII 图展示 Agent 循环架构 |\n| **12 阶段路径** | 表格形式展示 s01-s12 的主题、理念、工具数 |\n| **架构概览** | 项目目录结构 |\n| **关键机制** | s01-s06 基础能力 + s07-s12 团队持久化 |\n| **团队协作** | JSONL 邮箱 + 生命周期状态机 |\n| **快速开始** | 命令行快速启动指南 |\n| **延伸项目** | Kode CLI/SDK、claw0 |\n| **核心教训** | 三句金句总结 |\n\n文件位于：`/Users/ryanzhou/Desktop/learn-claude-code/AGENTS.md`', type='text')]}]
# 已完成！创建了 `/Users/ryanzhou/Desktop/learn-claude-code/AGENTS.md`，内容包括：

# ## AGENTS.md 结构

# | 章节 | 内容 |
# |------|------|
# | **核心概念** | Agent = Model + Harness 的核心理念 |
# | **模式图** | ASCII 图展示 Agent 循环架构 |
# | **12 阶段路径** | 表格形式展示 s01-s12 的主题、理念、工具数 |
# | **架构概览** | 项目目录结构 |
# | **关键机制** | s01-s06 基础能力 + s07-s12 团队持久化 |
# | **团队协作** | JSONL 邮箱 + 生命周期状态机 |
# | **快速开始** | 命令行快速启动指南 |
# | **延伸项目** | Kode CLI/SDK、claw0 |
# | **核心教训** | 三句金句总结 |

# 文件位于：`/Users/ryanzhou/Desktop/learn-claude-code/AGENTS.md`
