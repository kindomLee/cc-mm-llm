# llm-mcp

A generic LLM MCP server for [Claude Code](https://docs.anthropic.com/en/docs/claude-code) that drives any **OpenAI-compatible** chat completions API.

Claude (or any MCP host) acts as the orchestrator, dispatching tasks — parallel queries, cross-model debates, multi-perspective analysis, file-aware code review — to your chosen LLM provider via this single server.

## Supported Providers

Any provider exposing an OpenAI-compatible `/chat/completions`-style endpoint works. Examples:

| Provider | `LLM_API_URL` | `LLM_MODEL` | Notes |
|----------|---------------|-------------|-------|
| **MiniMax** (default) | `https://api.minimax.io/v1/text/chatcompletion_v2` | `MiniMax-M2.5` | Uses `max_completion_tokens` |
| **OpenAI** | `https://api.openai.com/v1/chat/completions` | `gpt-4o` | Set `LLM_MAX_TOKENS_FIELD=max_tokens` |
| **DeepSeek** | `https://api.deepseek.com/chat/completions` | `deepseek-chat` | |
| **OpenRouter** | `https://openrouter.ai/api/v1/chat/completions` | `anthropic/claude-3.5-sonnet` | |
| **Groq** | `https://api.groq.com/openai/v1/chat/completions` | `llama-3.3-70b-versatile` | |
| **Together AI** | `https://api.together.xyz/v1/chat/completions` | `meta-llama/Llama-3-70b-chat-hf` | |
| **Local (Ollama)** | `http://localhost:11434/v1/chat/completions` | `llama3` | No API key needed |

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A Claude Code installation

### Add to Claude Code

```bash
claude mcp add llm-mcp \
  -e LLM_API_KEY=your-api-key \
  -e LLM_API_URL=https://api.minimax.io/v1/text/chatcompletion_v2 \
  -e LLM_MODEL=MiniMax-M2.5 \
  -- uv run --with 'mcp[cli]' --with httpx python /path/to/server.py
```

Or add manually to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "llm-mcp": {
      "type": "stdio",
      "command": "uv",
      "args": [
        "run", "--with", "mcp[cli]", "--with", "httpx",
        "python", "/path/to/server.py"
      ],
      "env": {
        "LLM_API_KEY": "your-api-key",
        "LLM_API_URL": "https://api.minimax.io/v1/text/chatcompletion_v2",
        "LLM_MODEL": "MiniMax-M2.5"
      }
    }
  }
}
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_API_KEY` | Yes | — | API key / Bearer token |
| `LLM_API_URL` | No | `https://api.minimax.io/v1/text/chatcompletion_v2` | Chat completions endpoint |
| `LLM_MODEL` | No | `MiniMax-M2.5` | Default model ID |
| `LLM_MAX_CONCURRENT` | No | `5` | Max concurrent API calls |
| `LLM_MAX_TOKENS_FIELD` | No | `max_completion_tokens` | JSON field name for max tokens. Set to `max_tokens` for standard OpenAI API. |
| `LLM_PROJECT_ROOT` | No | — | Project root path for automatic context injection (reads `CLAUDE.md` and directory tree) |

## Tools

### Single-call

| Tool | Description |
|------|-------------|
| `ask` | Ask the LLM a single question |
| `review_code` | Review a code snippet |

### File-aware

| Tool | Description |
|------|-------------|
| `review_file` | Review a file (server reads it automatically) |
| `analyze_file` | Analyze a file and answer a question about it |
| `generate_patch` | Generate a unified diff patch for a file |
| `analyze_files` | Analyze multiple files together (supports glob patterns) |
| `parallel_review` | Review multiple files in parallel |

### Multi-instance

| Tool | Description |
|------|-------------|
| `parallel_ask` | Send multiple prompts in parallel |
| `debate` | Multi-round debate between two LLM instances |
| `multi_perspective` | Get parallel responses from different role perspectives |

## Project Context Injection

When `LLM_PROJECT_ROOT` is set, the server automatically:

1. Reads `CLAUDE.md` from the project root (truncated at 4000 chars)
2. Generates a directory tree (depth 3, max 200 lines)
3. Prepends this context to every system prompt

This gives the LLM awareness of your project structure and conventions without manual prompting.

## Running Tests

```bash
cd /path/to/cc-mm-llm
uv run --with 'mcp[cli]' --with httpx --with pytest --with pytest-asyncio \
    pytest test_server.py -v
```

## License

MIT — see [LICENSE](./LICENSE).
