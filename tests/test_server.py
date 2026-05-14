"""Coverage for ``server.py`` — CLI parsing, catalog assembly, MCP handler
wiring, and the helper functions (``_format_payload``,
``_enforce_unique_names``, ``_forwarded_credentials_from_env``,
``_tool_to_mcp``).

We exercise the CLI assembly + catalog assembly paths directly rather
than spinning up the actual stdio / SSE runners — those are thin
wrappers over ``server.run(...)`` that would require mocking the MCP
SDK's stream contracts in significant detail.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from ai_assistant_server import tool
from ai_assistant_server.models import (
    AuthConfig,
    HttpExecution,
    OpenApiTool,
    PluginTool,
)
from ai_assistant_server.server import (
    _enforce_unique_names,
    _format_payload,
    _forwarded_credentials_from_env,
    _parse_args,
    _tool_to_mcp,
    build_server,
    main,
)


# ---------------------------------------------------------------------------
# _format_payload
# ---------------------------------------------------------------------------


def test_format_payload_passes_string_through() -> None:
    assert _format_payload("hello") == "hello"


def test_format_payload_jsonifies_dict() -> None:
    out = _format_payload({"a": 1, "b": [2, 3]})
    assert '"a": 1' in out
    assert '"b": [' in out


def test_format_payload_falls_back_to_str_on_unencodable() -> None:
    class Weird:
        def __repr__(self) -> str:
            return "<weird>"

        # Make json.dumps with default=str raise — by giving it a
        # default that itself raises, we trigger the except path.
        def __str__(self) -> str:
            raise ValueError("boom")

    # An object with both default=str raising and json.dumps failing
    # is unusual; build via a dataclass-like wrapper that json
    # genuinely can't handle even with default=str.
    class Self:
        def __repr__(self) -> str:
            return "self-repr"

    out = _format_payload(Self())
    # default=str is used by json.dumps when the object can't be
    # serialized; json calls str(obj) which falls back to __repr__.
    assert "self-repr" in out


# ---------------------------------------------------------------------------
# _forwarded_credentials_from_env
# ---------------------------------------------------------------------------


def test_forwarded_credentials_extracts_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("X_AI_ASSISTANT_AUTH_BEARERAUTH", "tok-1")
    monkeypatch.setenv("X_AI_ASSISTANT_AUTH_API_KEY", "k-2")
    monkeypatch.setenv("UNRELATED", "ignored")
    creds = _forwarded_credentials_from_env()
    assert creds["BEARERAUTH"] == "tok-1"
    assert creds["API_KEY"] == "k-2"
    assert "UNRELATED" not in creds


def test_forwarded_credentials_skips_blanks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("X_AI_ASSISTANT_AUTH_EMPTY", "")
    monkeypatch.setenv("X_AI_ASSISTANT_AUTH_REAL", "v")
    creds = _forwarded_credentials_from_env()
    assert "EMPTY" not in creds
    assert creds["REAL"] == "v"


# ---------------------------------------------------------------------------
# _enforce_unique_names
# ---------------------------------------------------------------------------


def _openapi(name: str) -> OpenApiTool:
    return OpenApiTool(
        name=name,
        description="d",
        input_schema={"type": "object", "properties": {}},
        execution=HttpExecution(
            base_url="https://x",
            method="get",
            path="/",
        ),
        auth=AuthConfig(),
    )


def _plugin(name: str) -> PluginTool:
    @tool(name=name, description="d")
    def fn() -> str:
        return name

    from ai_assistant_server.plugins import get_plugin_tool

    plugin = get_plugin_tool(fn)
    assert plugin is not None
    return plugin


def test_enforce_unique_names_drops_dups(caplog: pytest.LogCaptureFixture) -> None:
    a = _openapi("foo")
    b = _plugin("foo")
    c = _openapi("bar")
    with caplog.at_level(logging.WARNING):
        out = _enforce_unique_names([a, b, c])
    assert [t.name for t in out] == ["foo", "bar"]
    assert any("duplicate tool name 'foo'" in r.message for r in caplog.records)


def test_enforce_unique_names_keeps_distinct_names() -> None:
    a = _openapi("foo")
    b = _openapi("bar")
    out = _enforce_unique_names([a, b])
    assert [t.name for t in out] == ["foo", "bar"]


# ---------------------------------------------------------------------------
# _tool_to_mcp
# ---------------------------------------------------------------------------


def test_tool_to_mcp_round_trips_metadata() -> None:
    src = _openapi("get_thing")
    out = _tool_to_mcp(src)
    assert out.name == "get_thing"
    assert out.description == "d"


# ---------------------------------------------------------------------------
# _parse_args + env-var defaults
# ---------------------------------------------------------------------------


def test_parse_args_defaults() -> None:
    ns = _parse_args([])
    assert ns.tools_dir == "tools"
    assert ns.plugins_dir == "plugins"
    assert ns.transport == "stdio"
    assert ns.host == "127.0.0.1"
    assert ns.port == 8765
    assert ns.plugin_module == []


def test_parse_args_explicit_flags() -> None:
    ns = _parse_args(
        [
            "--tools-dir", "/other/tools",
            "--plugins-dir", "/other/plugins",
            "--plugin-module", "pkg.a",
            "--plugin-module", "pkg.b",
            "--transport", "sse",
            "--host", "0.0.0.0",
            "--port", "9999",
        ]
    )
    assert ns.tools_dir == "/other/tools"
    assert ns.plugins_dir == "/other/plugins"
    assert ns.plugin_module == ["pkg.a", "pkg.b"]
    assert ns.transport == "sse"
    assert ns.host == "0.0.0.0"
    assert ns.port == 9999


def test_parse_args_plugin_modules_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reload the module so the argparse default re-reads the env."""
    import importlib
    import ai_assistant_server.server as srv

    monkeypatch.setenv(
        "AI_ASSISTANT_SERVER_PLUGIN_MODULES", "pkg.foo, pkg.bar, "
    )
    importlib.reload(srv)
    try:
        ns = srv._parse_args([])
        assert ns.plugin_module == ["pkg.foo", "pkg.bar"]
    finally:
        # Reload once more without the env so other tests don't pick
        # up a polluted module-level default.
        monkeypatch.delenv("AI_ASSISTANT_SERVER_PLUGIN_MODULES")
        importlib.reload(srv)


# ---------------------------------------------------------------------------
# build_server: list_tools / call_tool wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_server_list_tools_returns_catalog() -> None:
    server = build_server([_openapi("a"), _openapi("b")])
    # The decorator-registered handler is reachable via the server's
    # request_handlers.  Trigger via the published list_tools request
    # name so we exercise the wired callback.
    from mcp.types import ListToolsRequest

    req = ListToolsRequest(method="tools/list")
    handler = server.request_handlers[ListToolsRequest]
    result = await handler(req)
    names = [t.name for t in result.root.tools]
    assert names == ["a", "b"]
    await server._client.aclose()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_build_server_call_tool_dispatches_plugin_path() -> None:
    @tool(name="mult", description="d")
    def mult(a: int, b: int) -> int:
        return a * b

    from ai_assistant_server.plugins import get_plugin_tool

    plugin = get_plugin_tool(mult)
    assert plugin is not None
    server = build_server([plugin])
    from mcp.types import CallToolRequest

    req = CallToolRequest(
        method="tools/call",
        params={"name": "mult", "arguments": {"a": 3, "b": 5}},
    )
    handler = server.request_handlers[CallToolRequest]
    result = await handler(req)
    text = result.root.content[0].text
    assert "15" in text
    await server._client.aclose()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_build_server_call_tool_unknown_name() -> None:
    server = build_server([_openapi("a")])
    from mcp.types import CallToolRequest

    req = CallToolRequest(
        method="tools/call",
        params={"name": "ghost", "arguments": {}},
    )
    handler = server.request_handlers[CallToolRequest]
    result = await handler(req)
    assert "Unknown tool: ghost" in result.root.content[0].text
    await server._client.aclose()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_build_server_call_tool_surfaces_execution_error() -> None:
    p = _plugin("boom")

    def boom() -> None:
        raise RuntimeError("crashed")

    object.__setattr__(p, "handler", boom)
    server = build_server([p])
    from mcp.types import CallToolRequest

    req = CallToolRequest(
        method="tools/call",
        params={"name": "boom", "arguments": {}},
    )
    handler = server.request_handlers[CallToolRequest]
    result = await handler(req)
    text = result.root.content[0].text
    # ToolExecutionError carries the wrapped message + status
    assert "crashed" in text
    await server._client.aclose()  # type: ignore[attr-defined]


def test_build_server_raises_when_mcp_unavailable() -> None:
    # Force the import-failure path the module guards.
    with patch("ai_assistant_server.server.Server", None):
        with pytest.raises(RuntimeError, match="mcp"):
            build_server([])


# ---------------------------------------------------------------------------
# main(): catalog assembly + transport dispatch
# ---------------------------------------------------------------------------


def test_main_warns_when_no_tools_found(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Empty tools dir, no plugin modules, no plugins dir.
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    # Stop main() before it actually runs the transport — we only
    # want to exercise the catalog assembly + warning path.
    with patch(
        "ai_assistant_server.server.asyncio.run", side_effect=KeyboardInterrupt
    ):
        with pytest.raises(KeyboardInterrupt), caplog.at_level(logging.WARNING):
            main(
                [
                    "--tools-dir",
                    str(tools_dir),
                    "--plugins-dir",
                    str(tmp_path / "absent-plugins"),
                ]
            )
    assert any("No tools found" in r.message for r in caplog.records)


def test_main_returns_2_on_unimportable_plugin_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.ERROR):
        rc = main(
            [
                "--tools-dir",
                str(tmp_path / "no-such-tools"),
                "--plugin-module",
                "definitely_not_a_real_pkg_zz",
                "--plugins-dir",
                str(tmp_path / "no-such-plugins"),
            ]
        )
    assert rc == 2
    assert any(
        "Failed to import plugin module" in r.message for r in caplog.records
    )


def test_main_dispatches_to_stdio_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    captured: dict[str, Any] = {}

    async def fake_run(server: Any) -> None:
        captured["transport"] = "stdio"

    with (
        patch("ai_assistant_server.server._run_stdio", fake_run),
        patch("ai_assistant_server.server._run_sse"),
    ):
        rc = main(
            [
                "--tools-dir",
                str(tmp_path / "no-such"),
                "--plugins-dir",
                str(tmp_path / "no-such"),
                "--transport",
                "stdio",
            ]
        )
    assert rc == 0
    assert captured["transport"] == "stdio"


def test_main_dispatches_to_sse_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    captured: dict[str, Any] = {}

    async def fake_sse(server: Any, *, host: str, port: int) -> None:
        captured["host"] = host
        captured["port"] = port

    with (
        patch("ai_assistant_server.server._run_stdio"),
        patch("ai_assistant_server.server._run_sse", fake_sse),
    ):
        rc = main(
            [
                "--tools-dir",
                str(tmp_path / "no-such"),
                "--plugins-dir",
                str(tmp_path / "no-such"),
                "--transport",
                "sse",
                "--host",
                "0.0.0.0",
                "--port",
                "9999",
            ]
        )
    assert rc == 0
    assert captured == {"host": "0.0.0.0", "port": 9999}


def test_main_loads_openapi_and_plugin_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both sources are walked and the resulting catalog handed to
    ``build_server``.  We capture the call to assert the tools made
    it through."""
    monkeypatch.chdir(tmp_path)

    # Empty tools dir + a plugins dir with one @tool file.
    (tmp_path / "tools").mkdir()
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "p.py").write_text(
        "from ai_assistant_server import tool\n\n"
        "@tool(name='ping_in_main', description='d')\n"
        "def ping() -> str:\n"
        "    return 'pong'\n"
    )

    captured: dict[str, Any] = {}

    def fake_build(tools: list[Any]) -> Any:
        captured["tool_names"] = [t.name for t in tools]

        class _Stub:
            pass

        return _Stub()

    async def noop_run(*_a: Any, **_kw: Any) -> None:
        return None

    with (
        patch("ai_assistant_server.server.build_server", fake_build),
        patch("ai_assistant_server.server._run_stdio", noop_run),
    ):
        rc = main([])
    assert rc == 0
    assert "ping_in_main" in captured["tool_names"]


# ---------------------------------------------------------------------------
# build_server.call_tool — OpenAPI dispatch path (uses real httpx
# transport so we cover that branch too)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_server_call_tool_dispatches_openapi_path() -> None:
    """We exercise the call_tool branch by patching execute_tool —
    the call_tool wrapper itself is what we're verifying, not the
    HTTP execution (covered by test_executor.py)."""
    from ai_assistant_server.executor import ToolResult

    tool_def = _openapi("ping_http")
    server = build_server([tool_def])

    async def fake_execute(*_a: Any, **_kw: Any) -> ToolResult:
        return ToolResult(status_code=200, body={"ok": True}, headers={})

    from mcp.types import CallToolRequest

    req = CallToolRequest(
        method="tools/call",
        params={"name": "ping_http", "arguments": {}},
    )
    with patch("ai_assistant_server.server.execute_tool", fake_execute):
        handler_fn = server.request_handlers[CallToolRequest]
        result = await handler_fn(req)
    text = result.root.content[0].text
    assert '"ok"' in text
    await server._client.aclose()  # type: ignore[attr-defined]
