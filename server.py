# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mcp[cli]",
#     "httpx",
# ]
# ///
"""Generic LLM MCP Server — multi-instance parallel tasks, debate, code review, analysis.

A generic MCP server for Claude Code that drives any OpenAI-compatible LLM API.
Supports parallel tasks, cross-model debate, multi-perspective analysis,
and file-aware code review / patch generation.

Usage (local):
    uv run server.py

Usage (remote, no clone needed):
    uv run https://raw.githubusercontent.com/kindomLee/cc-mm-llm/main/server.py

Environment:
    LLM_API_KEY:            API key / Bearer token (required)
    LLM_API_URL:            Chat completions endpoint
                            (default: https://api.minimax.io/v1/text/chatcompletion_v2)
    LLM_MODEL:              Default model ID (default: MiniMax-M2.5)
    LLM_MAX_CONCURRENT:     Max concurrent API calls (default: 5)
    LLM_MAX_TOKENS_FIELD:   JSON field name for max tokens in the request payload
                            (default: max_completion_tokens). Set to "max_tokens" for
                            standard OpenAI-compatible APIs.
    LLM_PROJECT_ROOT:       Project root for context injection (optional, opt-in)
"""

import asyncio
import glob as globmod
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

LLM_API_URL = os.environ.get(
    "LLM_API_URL",
    "https://api.minimax.io/v1/text/chatcompletion_v2",
)
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "MiniMax-M2.5")
DEFAULT_MAX_TOKENS = 8192
LLM_MAX_CONCURRENT = int(os.environ.get("LLM_MAX_CONCURRENT", "5"))
LLM_MAX_TOKENS_FIELD = os.environ.get("LLM_MAX_TOKENS_FIELD", "max_completion_tokens")

# ── Module-level singletons ───────────────────────────────────────

_http_client: httpx.AsyncClient | None = None
_semaphore: asyncio.Semaphore | None = None
_project_context_cache: str | None = None


def _get_http_client() -> httpx.AsyncClient:
    """Return a lazy singleton httpx.AsyncClient for connection reuse."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=120)
    return _http_client


def _get_semaphore() -> asyncio.Semaphore:
    """Return a lazy singleton semaphore for concurrency limiting."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(LLM_MAX_CONCURRENT)
    return _semaphore


def _reset_caches() -> None:
    """Reset all module-level singletons and caches. For testing only."""
    global _http_client, _semaphore, _project_context_cache
    if _http_client is not None and not _http_client.is_closed:
        # Don't await close in sync context — just discard
        pass
    _http_client = None
    _semaphore = None
    _project_context_cache = None

mcp = FastMCP(
    "llm-mcp",
    instructions=(
        "Generic LLM multi-instance integration: "
        "parallel tasks, cross-model debate, code review, second opinions"
    ),
)


def _get_api_key() -> str:
    key = os.environ.get("LLM_API_KEY", "")
    if not key:
        raise ValueError(
            "LLM_API_KEY environment variable is not set. "
            "Set it in your shell profile or Claude Code MCP config."
        )
    return key


# ── Project context injection (opt-in) ────────────────────────────

_CONTEXT_TOTAL_BUDGET = 8000
_CONTEXT_MAX_CHARS = 4000
_SKILL_MAX_CHARS = 2000
_TREE_MAX_LINES = 200
_TREE_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".egg-info", ".eggs",
}


def _collect_project_context() -> str:
    """Collect project context from LLM_PROJECT_ROOT.

    Returns an empty string if LLM_PROJECT_ROOT is not set.
    Results are cached for the server lifetime.
    """
    global _project_context_cache
    if _project_context_cache is not None:
        return _project_context_cache

    root = os.environ.get("LLM_PROJECT_ROOT", "")
    if not root:
        _project_context_cache = ""
        return ""

    root_path = Path(root).expanduser().resolve()
    parts: list[str] = []

    # Read CLAUDE.md
    claude_md = root_path / "CLAUDE.md"
    if claude_md.is_file():
        try:
            text = claude_md.read_text(encoding="utf-8", errors="replace")
            if len(text) > _CONTEXT_MAX_CHARS:
                text = text[:_CONTEXT_MAX_CHARS] + "\n... [truncated]"
            parts.append(f"## CLAUDE.md\n{text}")
        except OSError:
            pass

    # Read .claude/skills/*.md
    skills_dir = root_path / ".claude" / "skills"
    if skills_dir.is_dir():
        for f in sorted(skills_dir.glob("*.md")):
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
                if len(text) > _SKILL_MAX_CHARS:
                    text = text[:_SKILL_MAX_CHARS] + "\n... [truncated]"
                parts.append(f"## Skill: {f.name}\n{text}")
            except OSError:
                pass

    # Generate directory tree (depth <= 3)
    try:
        result = subprocess.run(
            ["find", str(root_path), "-maxdepth", "3", "-not", "-name", ".*"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            lines: list[str] = []
            for line in result.stdout.strip().split("\n"):
                rel = line.replace(str(root_path), ".", 1)
                # Skip hidden and excluded dirs
                skip = False
                for seg in Path(rel).parts:
                    if seg in _TREE_SKIP_DIRS or seg.startswith("."):
                        skip = True
                        break
                if not skip:
                    lines.append(rel)
                if len(lines) >= _TREE_MAX_LINES:
                    lines.append(f"... [truncated at {_TREE_MAX_LINES} lines]")
                    break
            if lines:
                parts.append(f"## Project tree\n```\n" + "\n".join(lines) + "\n```")
    except (OSError, subprocess.TimeoutExpired):
        pass

    combined = "\n\n".join(parts)
    if len(combined) > _CONTEXT_TOTAL_BUDGET:
        combined = combined[:_CONTEXT_TOTAL_BUDGET] + "\n... [context truncated at budget]"
    _project_context_cache = combined
    return _project_context_cache


def _enhance_system_prompt(system: str) -> str:
    """Prepend project context to a system prompt if available."""
    ctx = _collect_project_context()
    if not ctx:
        return system
    return (
        "# Project Context (auto-injected)\n\n"
        f"{ctx}\n\n"
        "---\n\n"
        f"{system}"
    )


async def _call_llm(
    messages: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """Send a chat completion request to the configured LLM API.

    Features:
    - Temperature clamped to [0.01, 1.0]
    - Reuses a module-level httpx.AsyncClient
    - Semaphore-based concurrency limiting (LLM_MAX_CONCURRENT)
    - Single retry on 429/5xx with 2s backoff
    - Auto-injects project context when LLM_PROJECT_ROOT is set
    """
    temperature = max(0.01, min(temperature, 1.0))

    # Inject project context into system prompt if available
    enhanced_messages = list(messages)
    for i, msg in enumerate(enhanced_messages):
        if msg.get("role") == "system":
            enhanced_messages[i] = {
                **msg,
                "content": _enhance_system_prompt(msg["content"]),
            }
            break

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_get_api_key()}",
    }
    payload = {
        "model": model,
        "messages": enhanced_messages,
        "temperature": temperature,
        LLM_MAX_TOKENS_FIELD: max_tokens,
    }

    client = _get_http_client()
    sem = _get_semaphore()

    async with sem:
        for attempt in range(2):
            resp = await client.post(LLM_API_URL, headers=headers, json=payload)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt == 0:
                await asyncio.sleep(2)
                continue
            resp.raise_for_status()
            return resp.json()

    # Unreachable, but satisfies type checker
    raise RuntimeError("Unexpected: retry loop exited without return")


def _format_reply(response: dict[str, Any], label: str = "") -> str:
    """Extract and format the text reply from the LLM API response."""
    choices = response.get("choices", [])
    if not choices:
        return f"[LLM returned no choices]\n{json.dumps(response, indent=2)}"
    content = choices[0].get("message", {}).get("content", "")
    usage = response.get("usage", {})
    model = response.get("model", "unknown")
    prefix = f"**[{label}]**\n\n" if label else ""
    footer = (
        f"\n\n---\n_Model: {model} | "
        f"Tokens: {usage.get('prompt_tokens', '?')} in / "
        f"{usage.get('completion_tokens', '?')} out_"
    )
    return prefix + content + footer


# ── Single-call tools ──────────────────────────────────────────────


@mcp.tool()
async def ask(
    prompt: str,
    system: str = "You are a helpful assistant. Respond concisely and precisely.",
    temperature: float = 0.7,
    model: str = DEFAULT_MODEL,
) -> str:
    """Ask the LLM a single question.

    Args:
        prompt: The question or task
        system: System prompt (optional)
        temperature: 0-1 (default 0.7)
        model: Model ID
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    resp = await _call_llm(messages, model=model, temperature=temperature)
    return _format_reply(resp)


@mcp.tool()
async def review_code(
    code: str,
    instruction: str = "Review this code for bugs, security issues, and improvements. Be concise.",
    temperature: float = 0.3,
    model: str = DEFAULT_MODEL,
) -> str:
    """Ask the LLM to review code.

    Args:
        code: The code to review
        instruction: Review focus (optional)
        temperature: 0-1 (default 0.3)
        model: Model ID
    """
    messages = [
        {
            "role": "system",
            "content": "You are an expert code reviewer. Provide actionable, concise feedback.",
        },
        {"role": "user", "content": f"{instruction}\n\n```\n{code}\n```"},
    ]
    resp = await _call_llm(messages, model=model, temperature=temperature)
    return _format_reply(resp)


# ── File helpers ───────────────────────────────────────────────────

MAX_FILE_CHARS = 60_000  # ~15K tokens


def _check_path_allowed(p: Path) -> None:
    """Raise ValueError if *p* is outside the allowed project root."""
    root = os.environ.get("LLM_PROJECT_ROOT", "")
    if not root:
        return  # no restriction when project root is unset
    root_path = Path(root).expanduser().resolve()
    if not p.is_relative_to(root_path):
        raise ValueError(
            f"Access denied: {p} is outside project root {root_path}"
        )


def _read_file(path: str) -> str:
    """Read a file, truncate if too large."""
    p = Path(path).expanduser().resolve()
    _check_path_allowed(p)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    text = p.read_text(encoding="utf-8", errors="replace")
    if len(text) > MAX_FILE_CHARS:
        text = text[:MAX_FILE_CHARS] + f"\n\n... [truncated at {MAX_FILE_CHARS} chars]"
    return text


def _read_files(paths: list[str]) -> str:
    """Read multiple files and concatenate with headers."""
    parts = []
    for path in paths:
        try:
            content = _read_file(path)
            parts.append(f"### {path}\n```\n{content}\n```")
        except (FileNotFoundError, ValueError) as e:
            parts.append(f"### {path}\n[ERROR: {e}]")
    return "\n\n".join(parts)


def _glob_files(pattern: str) -> list[str]:
    """Expand a glob pattern to file paths."""
    matches = sorted(globmod.glob(pattern, recursive=True))
    return [m for m in matches if os.path.isfile(m)]


# ── File-aware tools ──────────────────────────────────────────────


@mcp.tool()
async def review_file(
    path: str,
    focus: str = "Review for bugs, security issues, and improvements.",
    temperature: float = 0.3,
    model: str = DEFAULT_MODEL,
) -> str:
    """Ask the LLM to review a file. Server reads the file automatically.

    Args:
        path: Absolute path to the file to review
        focus: What to focus on in the review (optional)
        temperature: 0-1 (default 0.3)
        model: Model ID
    """
    code = _read_file(path)
    filename = Path(path).name
    messages = [
        {
            "role": "system",
            "content": "You are an expert code reviewer. Provide actionable, concise feedback.",
        },
        {
            "role": "user",
            "content": f"Review `{filename}` with focus: {focus}\n\n```\n{code}\n```",
        },
    ]
    resp = await _call_llm(messages, model=model, temperature=temperature)
    return _format_reply(resp, label=f"Review: {filename}")


@mcp.tool()
async def analyze_file(
    path: str,
    question: str,
    temperature: float = 0.5,
    model: str = DEFAULT_MODEL,
) -> str:
    """Ask the LLM to analyze a file and answer a question about it.

    Args:
        path: Absolute path to the file
        question: The question to answer about the file
        temperature: 0-1 (default 0.5)
        model: Model ID
    """
    code = _read_file(path)
    filename = Path(path).name
    messages = [
        {
            "role": "system",
            "content": "You are an expert software engineer. Analyze the code and answer precisely.",
        },
        {
            "role": "user",
            "content": f"File: `{filename}`\n\n```\n{code}\n```\n\nQuestion: {question}",
        },
    ]
    resp = await _call_llm(messages, model=model, temperature=temperature)
    return _format_reply(resp, label=f"Analysis: {filename}")


@mcp.tool()
async def generate_patch(
    path: str,
    instruction: str,
    temperature: float = 0.3,
    model: str = DEFAULT_MODEL,
) -> str:
    """Ask the LLM to generate a unified diff patch for a file.

    The host agent reviews the patch before applying. Server reads the file automatically.

    Args:
        path: Absolute path to the file to modify
        instruction: What changes to make
        temperature: 0-1 (default 0.3)
        model: Model ID
    """
    code = _read_file(path)
    filename = Path(path).name
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert software engineer. Generate a unified diff patch "
                "(--- a/file, +++ b/file format) for the requested changes. "
                "Output ONLY the diff, no explanation before or after."
            ),
        },
        {
            "role": "user",
            "content": f"File: `{filename}`\n\n```\n{code}\n```\n\nChanges requested: {instruction}",
        },
    ]
    resp = await _call_llm(messages, model=model, temperature=temperature)
    return _format_reply(resp, label=f"Patch: {filename}")


@mcp.tool()
async def analyze_files(
    paths: list[str],
    question: str,
    temperature: float = 0.5,
    model: str = DEFAULT_MODEL,
) -> str:
    """Ask the LLM to analyze multiple files together and answer a question.

    Args:
        paths: List of absolute file paths (or glob patterns)
        question: The question to answer about the files
        temperature: 0-1 (default 0.5)
        model: Model ID
    """
    # Expand globs
    resolved: list[str] = []
    for p in paths:
        if "*" in p or "?" in p:
            resolved.extend(_glob_files(p))
        else:
            resolved.append(p)

    if not resolved:
        return "[ERROR] No files matched the given paths/patterns."

    content = _read_files(resolved)
    messages = [
        {
            "role": "system",
            "content": "You are an expert software engineer. Analyze the provided files and answer precisely.",
        },
        {
            "role": "user",
            "content": f"{content}\n\n---\nQuestion: {question}",
        },
    ]
    resp = await _call_llm(messages, model=model, temperature=temperature)
    file_list = ", ".join(Path(p).name for p in resolved[:5])
    if len(resolved) > 5:
        file_list += f" +{len(resolved) - 5} more"
    return _format_reply(resp, label=f"Analysis: {file_list}")


@mcp.tool()
async def parallel_review(
    paths: list[str],
    focus: str = "Review for bugs, security issues, and improvements.",
    temperature: float = 0.3,
    model: str = DEFAULT_MODEL,
) -> str:
    """Review multiple files IN PARALLEL — each file gets its own LLM instance.

    Args:
        paths: List of absolute file paths to review
        focus: Review focus shared across all files
        temperature: 0-1 (default 0.3)
        model: Model ID
    """
    async def _review_one(path: str) -> str:
        try:
            code = _read_file(path)
        except FileNotFoundError as e:
            return f"**[{Path(path).name} — ERROR]**\n\n{e}"
        filename = Path(path).name
        messages = [
            {
                "role": "system",
                "content": "You are an expert code reviewer. Provide actionable, concise feedback.",
            },
            {
                "role": "user",
                "content": f"Review `{filename}` with focus: {focus}\n\n```\n{code}\n```",
            },
        ]
        resp = await _call_llm(messages, model=model, temperature=temperature)
        return _format_reply(resp, label=f"Review: {filename}")

    tasks = [_review_one(p) for p in paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    parts = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            parts.append(f"**[{Path(paths[i]).name} — ERROR]**\n\n{r}")
        else:
            parts.append(r)

    return "\n\n====\n\n".join(parts)


# ── Parallel multi-instance tools ──────────────────────────────────


@mcp.tool()
async def parallel_ask(
    prompts: list[str],
    system: str = "You are a helpful assistant. Respond concisely and precisely.",
    temperature: float = 0.7,
    model: str = DEFAULT_MODEL,
) -> str:
    """Send multiple prompts to the LLM IN PARALLEL and collect all results.

    Use this when the host agent splits a main task into sub-tasks that LLM
    instances can work on simultaneously.

    Args:
        prompts: List of prompts to send concurrently (each becomes a separate API call)
        system: Shared system prompt for all instances
        temperature: 0-1
        model: Model ID
    """
    async def _run_one(idx: int, prompt: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        resp = await _call_llm(messages, model=model, temperature=temperature)
        return _format_reply(resp, label=f"Task {idx + 1}")

    tasks = [_run_one(i, p) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    parts = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            parts.append(f"**[Task {i + 1} — ERROR]**\n\n{r}")
        else:
            parts.append(r)

    return "\n\n====\n\n".join(parts)


@mcp.tool()
async def debate(
    topic: str,
    rounds: int = 2,
    perspective_a: str = "Argue IN FAVOR. Be specific with evidence and reasoning.",
    perspective_b: str = "Argue AGAINST. Be specific with evidence and reasoning.",
    model: str = DEFAULT_MODEL,
) -> str:
    """Run a multi-round debate between two LLM instances on a topic.

    Two model instances take opposing perspectives and respond to each other
    across multiple rounds. The host agent can then synthesize the strongest arguments.

    Args:
        topic: The topic or question to debate
        rounds: Number of debate rounds (default 2, max 4)
        perspective_a: Instructions for side A (default: argue in favor)
        perspective_b: Instructions for side B (default: argue against)
        model: Model ID
    """
    rounds = min(rounds, 4)
    history_a: list[dict[str, str]] = [
        {"role": "system", "content": f"You are Debater A. {perspective_a}"},
        {"role": "user", "content": f"Debate topic: {topic}\n\nPresent your opening argument."},
    ]
    history_b: list[dict[str, str]] = [
        {"role": "system", "content": f"You are Debater B. {perspective_b}"},
    ]

    transcript: list[str] = []

    for round_num in range(rounds):
        # Side A argues
        resp_a = await _call_llm(history_a, model=model, temperature=0.7)
        reply_a = _format_reply(resp_a, label=f"Round {round_num + 1} — Side A")
        text_a = resp_a.get("choices", [{}])[0].get("message", {}).get("content", "")
        transcript.append(reply_a)

        # Feed A's argument to B
        if round_num == 0:
            history_b.append({
                "role": "user",
                "content": (
                    f"Debate topic: {topic}\n\n"
                    f"Your opponent (Side A) argues:\n{text_a}\n\n"
                    "Present your counter-argument."
                ),
            })
        else:
            history_b.append({
                "role": "user",
                "content": f"Side A responds:\n{text_a}\n\nCounter this argument.",
            })

        # Side B argues
        resp_b = await _call_llm(history_b, model=model, temperature=0.7)
        reply_b = _format_reply(resp_b, label=f"Round {round_num + 1} — Side B")
        text_b = resp_b.get("choices", [{}])[0].get("message", {}).get("content", "")
        transcript.append(reply_b)

        # Feed B's argument back to A for next round
        history_a.append({"role": "assistant", "content": text_a})
        history_a.append({
            "role": "user",
            "content": f"Side B responds:\n{text_b}\n\nCounter this argument.",
        })
        history_b.append({"role": "assistant", "content": text_b})

    return "\n\n====\n\n".join(transcript)


@mcp.tool()
async def multi_perspective(
    question: str,
    perspectives: list[str],
    model: str = DEFAULT_MODEL,
) -> str:
    """Get PARALLEL responses from multiple LLM instances, each with a different role/perspective.

    All perspectives run concurrently. The host agent can then compare and synthesize.

    Args:
        question: The question to analyze
        perspectives: List of role descriptions (e.g. ["security expert", "performance engineer", "UX designer"])
        model: Model ID
    """
    async def _run_perspective(role: str) -> str:
        messages = [
            {
                "role": "system",
                "content": f"You are a {role}. Analyze from your professional perspective. Be concise and specific.",
            },
            {"role": "user", "content": question},
        ]
        resp = await _call_llm(messages, model=model, temperature=0.6)
        return _format_reply(resp, label=role)

    tasks = [_run_perspective(p) for p in perspectives]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    parts = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            parts.append(f"**[{perspectives[i]} — ERROR]**\n\n{r}")
        else:
            parts.append(r)

    return "\n\n====\n\n".join(parts)


if __name__ == "__main__":
    mcp.run(transport="stdio")
