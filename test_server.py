"""Tests for Generic LLM MCP Server.

Run with:
    uv run --with 'mcp[cli]' --with httpx --with pytest --with pytest-asyncio \
        pytest test_server.py -v
"""

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import server


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_all_caches():
    """Reset module-level singletons before each test."""
    server._reset_caches()
    yield
    server._reset_caches()


@pytest.fixture
def fake_response() -> dict:
    """A minimal valid LLM API response."""
    return {
        "choices": [
            {"message": {"content": "Hello from LLM"}}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "model": "test-model",
    }


def _make_httpx_response(
    status_code: int = 200,
    json_data: dict | None = None,
) -> httpx.Response:
    """Create a fake httpx.Response."""
    resp = httpx.Response(
        status_code=status_code,
        json=json_data or {},
        request=httpx.Request("POST", server.LLM_API_URL),
    )
    return resp


# ── TestTemperatureGuard ──────────────────────────────────────────


class TestTemperatureGuard:
    """Temperature is clamped to [0.01, 1.0] inside _call_llm."""

    @pytest.mark.asyncio
    async def test_zero_clamped_to_001(self, fake_response: dict):
        mock_post = AsyncMock(return_value=_make_httpx_response(200, fake_response))
        with patch.object(server, "_get_http_client") as mock_client_fn:
            client = MagicMock()
            client.post = mock_post
            mock_client_fn.return_value = client

            with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}):
                await server._call_llm(
                    [{"role": "system", "content": "hi"},
                     {"role": "user", "content": "hello"}],
                    temperature=0.0,
                )
            payload = mock_post.call_args[1]["json"]
            assert payload["temperature"] == 0.01

    @pytest.mark.asyncio
    async def test_negative_clamped_to_001(self, fake_response: dict):
        mock_post = AsyncMock(return_value=_make_httpx_response(200, fake_response))
        with patch.object(server, "_get_http_client") as mock_client_fn:
            client = MagicMock()
            client.post = mock_post
            mock_client_fn.return_value = client

            with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}):
                await server._call_llm(
                    [{"role": "system", "content": "hi"},
                     {"role": "user", "content": "hello"}],
                    temperature=-0.5,
                )
            payload = mock_post.call_args[1]["json"]
            assert payload["temperature"] == 0.01

    @pytest.mark.asyncio
    async def test_above_1_clamped_to_1(self, fake_response: dict):
        mock_post = AsyncMock(return_value=_make_httpx_response(200, fake_response))
        with patch.object(server, "_get_http_client") as mock_client_fn:
            client = MagicMock()
            client.post = mock_post
            mock_client_fn.return_value = client

            with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}):
                await server._call_llm(
                    [{"role": "system", "content": "hi"},
                     {"role": "user", "content": "hello"}],
                    temperature=1.5,
                )
            payload = mock_post.call_args[1]["json"]
            assert payload["temperature"] == 1.0

    @pytest.mark.asyncio
    async def test_normal_value_unchanged(self, fake_response: dict):
        mock_post = AsyncMock(return_value=_make_httpx_response(200, fake_response))
        with patch.object(server, "_get_http_client") as mock_client_fn:
            client = MagicMock()
            client.post = mock_post
            mock_client_fn.return_value = client

            with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}):
                await server._call_llm(
                    [{"role": "system", "content": "hi"},
                     {"role": "user", "content": "hello"}],
                    temperature=0.5,
                )
            payload = mock_post.call_args[1]["json"]
            assert payload["temperature"] == 0.5


# ── TestReadFile ──────────────────────────────────────────────────


class TestReadFile:
    def test_normal_read(self, tmp_path: Path):
        f = tmp_path / "hello.txt"
        f.write_text("Hello World")
        result = server._read_file(str(f))
        assert result == "Hello World"

    def test_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            server._read_file("/nonexistent/path/file.txt")

    def test_truncation(self, tmp_path: Path):
        f = tmp_path / "big.txt"
        f.write_text("x" * (server.MAX_FILE_CHARS + 100))
        result = server._read_file(str(f))
        assert f"truncated at {server.MAX_FILE_CHARS}" in result
        assert len(result) < server.MAX_FILE_CHARS + 200


# ── TestFormatReply ───────────────────────────────────────────────


class TestFormatReply:
    def test_normal_response(self, fake_response: dict):
        result = server._format_reply(fake_response)
        assert "Hello from LLM" in result
        assert "10 in" in result
        assert "5 out" in result

    def test_empty_choices(self):
        resp = {"choices": [], "usage": {}}
        result = server._format_reply(resp)
        assert "no choices" in result

    def test_with_label(self, fake_response: dict):
        result = server._format_reply(fake_response, label="Test Label")
        assert "**[Test Label]**" in result


# ── TestProjectContext ────────────────────────────────────────────


class TestProjectContext:
    def test_no_env_returns_empty(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LLM_PROJECT_ROOT", None)
            result = server._collect_project_context()
            assert result == ""

    def test_with_env_collects_claude_md(self, tmp_path: Path):
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("# My Project\nSome instructions here.")

        with patch.dict(os.environ, {"LLM_PROJECT_ROOT": str(tmp_path)}):
            result = server._collect_project_context()
            assert "My Project" in result
            assert "Some instructions here" in result

    def test_cache_is_used(self, tmp_path: Path):
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("Original content")

        with patch.dict(os.environ, {"LLM_PROJECT_ROOT": str(tmp_path)}):
            first = server._collect_project_context()
            # Change file — cache should return old value
            claude_md.write_text("Changed content")
            second = server._collect_project_context()
            assert first == second
            assert "Original content" in second

    def test_enhance_passthrough_without_context(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LLM_PROJECT_ROOT", None)
            result = server._enhance_system_prompt("Be helpful.")
            assert result == "Be helpful."

    def test_enhance_prepends_context(self, tmp_path: Path):
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("# Rules\nDo X.")

        with patch.dict(os.environ, {"LLM_PROJECT_ROOT": str(tmp_path)}):
            result = server._enhance_system_prompt("Be helpful.")
            assert result.startswith("# Project Context")
            assert "Rules" in result
            assert "Be helpful." in result

    def test_claude_md_truncation(self, tmp_path: Path):
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("x" * (server._CONTEXT_MAX_CHARS + 500))

        with patch.dict(os.environ, {"LLM_PROJECT_ROOT": str(tmp_path)}):
            result = server._collect_project_context()
            assert "truncated" in result

    def test_skills_collected(self, tmp_path: Path):
        skills_dir = tmp_path / ".claude" / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "lint.md").write_text("Always use ruff.")
        (skills_dir / "test.md").write_text("Use pytest with AAA pattern.")

        with patch.dict(os.environ, {"LLM_PROJECT_ROOT": str(tmp_path)}):
            result = server._collect_project_context()
            assert "Skill: lint.md" in result
            assert "Always use ruff" in result
            assert "Skill: test.md" in result

    def test_total_budget_truncation(self, tmp_path: Path):
        # CLAUDE.md is capped at _CONTEXT_MAX_CHARS, so use skills to exceed budget
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("C" * server._CONTEXT_MAX_CHARS)
        skills_dir = tmp_path / ".claude" / "skills"
        skills_dir.mkdir(parents=True)
        # Each skill up to _SKILL_MAX_CHARS; create enough to blow past budget
        for i in range(5):
            (skills_dir / f"s{i}.md").write_text("S" * server._SKILL_MAX_CHARS)

        with patch.dict(os.environ, {"LLM_PROJECT_ROOT": str(tmp_path)}):
            result = server._collect_project_context()
            assert "context truncated at budget" in result
            assert len(result) <= server._CONTEXT_TOTAL_BUDGET + 100


# ── TestFileAccessRestriction ────────────────────────────────────


class TestFileAccessRestriction:
    def test_no_root_allows_any_path(self, tmp_path: Path):
        f = tmp_path / "allowed.txt"
        f.write_text("ok")
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LLM_PROJECT_ROOT", None)
            result = server._read_file(str(f))
            assert result == "ok"

    def test_within_root_allowed(self, tmp_path: Path):
        f = tmp_path / "project" / "src.py"
        f.parent.mkdir()
        f.write_text("code")
        with patch.dict(os.environ, {"LLM_PROJECT_ROOT": str(tmp_path)}):
            result = server._read_file(str(f))
            assert result == "code"

    def test_outside_root_denied(self, tmp_path: Path):
        outside = tmp_path / "outside"
        outside.mkdir()
        secret = outside / "secret.txt"
        secret.write_text("sensitive")

        project = tmp_path / "project"
        project.mkdir()

        with patch.dict(os.environ, {"LLM_PROJECT_ROOT": str(project)}):
            with pytest.raises(ValueError, match="Access denied"):
                server._read_file(str(secret))

    def test_read_files_catches_access_denied(self, tmp_path: Path):
        project = tmp_path / "project"
        project.mkdir()
        (project / "a.py").write_text("a")

        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "b.py").write_text("b")

        with patch.dict(os.environ, {"LLM_PROJECT_ROOT": str(project)}):
            # _read_files should catch ValueError and include error in output
            result = server._read_files([str(outside / "b.py")])
            assert "Access denied" in result


# ── TestCallLlm ──────────────────────────────────────────────────


class TestCallLlm:
    @pytest.mark.asyncio
    async def test_success(self, fake_response: dict):
        mock_post = AsyncMock(return_value=_make_httpx_response(200, fake_response))
        with patch.object(server, "_get_http_client") as mock_client_fn:
            client = MagicMock()
            client.post = mock_post
            mock_client_fn.return_value = client

            with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}):
                result = await server._call_llm(
                    [{"role": "system", "content": "hi"},
                     {"role": "user", "content": "hello"}],
                )
            assert result["choices"][0]["message"]["content"] == "Hello from LLM"
            assert mock_post.call_count == 1

    @pytest.mark.asyncio
    async def test_429_retries_once(self, fake_response: dict):
        resp_429 = _make_httpx_response(429)
        resp_ok = _make_httpx_response(200, fake_response)
        mock_post = AsyncMock(side_effect=[resp_429, resp_ok])

        with patch.object(server, "_get_http_client") as mock_client_fn:
            client = MagicMock()
            client.post = mock_post
            mock_client_fn.return_value = client

            with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}):
                with patch("server.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                    result = await server._call_llm(
                        [{"role": "system", "content": "hi"},
                         {"role": "user", "content": "hello"}],
                    )
                    mock_sleep.assert_awaited_once_with(2)

            assert result["choices"][0]["message"]["content"] == "Hello from LLM"
            assert mock_post.call_count == 2

    @pytest.mark.asyncio
    async def test_400_no_retry(self):
        resp_400 = _make_httpx_response(400)
        mock_post = AsyncMock(return_value=resp_400)

        with patch.object(server, "_get_http_client") as mock_client_fn:
            client = MagicMock()
            client.post = mock_post
            mock_client_fn.return_value = client

            with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}):
                with pytest.raises(httpx.HTTPStatusError):
                    await server._call_llm(
                        [{"role": "system", "content": "hi"},
                         {"role": "user", "content": "hello"}],
                    )
            assert mock_post.call_count == 1

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self, fake_response: dict):
        """Verify semaphore limits concurrent calls."""
        max_concurrent_seen = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def _slow_post(*args, **kwargs):
            nonlocal max_concurrent_seen, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent_seen:
                    max_concurrent_seen = current_concurrent
            await asyncio.sleep(0.05)
            async with lock:
                current_concurrent -= 1
            return _make_httpx_response(200, fake_response)

        # Set semaphore to 2 for testing
        server._semaphore = asyncio.Semaphore(2)

        with patch.object(server, "_get_http_client") as mock_client_fn:
            client = MagicMock()
            client.post = AsyncMock(side_effect=_slow_post)
            mock_client_fn.return_value = client

            with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}):
                tasks = [
                    server._call_llm(
                        [{"role": "system", "content": "hi"},
                         {"role": "user", "content": f"task {i}"}],
                    )
                    for i in range(5)
                ]
                await asyncio.gather(*tasks)

        assert max_concurrent_seen <= 2

    @pytest.mark.asyncio
    async def test_max_tokens_default(self, fake_response: dict):
        mock_post = AsyncMock(return_value=_make_httpx_response(200, fake_response))
        with patch.object(server, "_get_http_client") as mock_client_fn:
            client = MagicMock()
            client.post = mock_post
            mock_client_fn.return_value = client

            with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}):
                await server._call_llm(
                    [{"role": "system", "content": "hi"},
                     {"role": "user", "content": "hello"}],
                )
            payload = mock_post.call_args[1]["json"]
            assert payload["max_completion_tokens"] == 8192


# ── TestToolIntegration ───────────────────────────────────────────


class TestToolIntegration:
    """Test that tool functions build correct message structures."""

    @pytest.mark.asyncio
    async def test_ask_messages(self, fake_response: dict):
        with patch.object(server, "_call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = fake_response
            result = await server.ask(
                prompt="What is Python?",
                system="Be concise.",
                temperature=0.5,
                model="test-model",
            )

            mock_call.assert_awaited_once()
            messages = mock_call.call_args[0][0]
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "Be concise."
            assert messages[1]["role"] == "user"
            assert "What is Python?" in messages[1]["content"]
            assert "Hello from LLM" in result

    @pytest.mark.asyncio
    async def test_review_file_messages(self, fake_response: dict, tmp_path: Path):
        test_file = tmp_path / "example.py"
        test_file.write_text("def hello():\n    return 'hi'\n")

        with patch.object(server, "_call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = fake_response
            result = await server.review_file(
                path=str(test_file),
                focus="Check for bugs",
                temperature=0.3,
            )

            mock_call.assert_awaited_once()
            messages = mock_call.call_args[0][0]
            assert messages[0]["role"] == "system"
            assert "code reviewer" in messages[0]["content"].lower()
            assert messages[1]["role"] == "user"
            assert "example.py" in messages[1]["content"]
            assert "def hello():" in messages[1]["content"]
            assert "Review: example.py" in result
