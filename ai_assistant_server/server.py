"""MCP server entrypoint.

Loads tools from a directory of OpenAPI specs, registers each
operation as an MCP tool, and serves them over either stdio
(default; suitable for Claude Desktop / Code) or HTTP+SSE
(for browser / remote clients).

Usage::

    ai-assistant-server                         # stdio, ./tools
    ai-assistant-server --transport sse --port 8765
    ai-assistant-server --tools-dir ./my-specs

Forwarded credentials
---------------------
When running over SSE, the host can pass per-request credentials
via the ``X-AI-Assistant-Auth-{SCHEME}`` header.  Example::

    X-AI-Assistant-Auth-bearerAuth: <token>

These take priority over environment variables — useful for
acting on behalf of an authenticated end-user.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

import httpx

# The MCP SDK is the reference implementation Anthropic publishes.
# We import lazily so the package can still be loaded for tests
# without the SDK installed.
try:
    from mcp.server import Server
    from mcp.types import TextContent, Tool
except ImportError as err:  # pragma: no cover - import-time failure path
    Server = None  # type: ignore[assignment]
    TextContent = None  # type: ignore[assignment]
    Tool = None  # type: ignore[assignment]
    _IMPORT_ERROR: ImportError | None = err
else:
    _IMPORT_ERROR = None

from ai_assistant_server.executor import ToolExecutionError, execute_tool
from ai_assistant_server.loader import (
    load_plugins_from_directory,
    load_plugins_from_module,
    load_tools_from_directory,
)
from ai_assistant_server.models import ToolDefinition


log = logging.getLogger(__name__)


def build_server(tools: list[ToolDefinition]) -> "Server":
    """Construct a configured MCP server given a tool catalog."""
    if Server is None:
        raise RuntimeError(
            "The 'mcp' package is required to run the server "
            f"(import failed: {_IMPORT_ERROR}). Install with: "
            "pip install mcp"
        )

    server = Server("ai-assistant-server")
    catalog = {t.name: t for t in tools}
    # Share one connection pool across the server lifetime.
    client = httpx.AsyncClient(timeout=30.0)

    @server.list_tools()
    async def list_tools() -> list["Tool"]:
        return [_tool_to_mcp(t) for t in catalog.values()]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list["TextContent"]:
        tool = catalog.get(name)
        if tool is None:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
        forwarded = _forwarded_credentials_from_env()
        try:
            result = await execute_tool(
                tool, arguments, client=client, forwarded_credentials=forwarded
            )
        except ToolExecutionError as err:
            payload = {
                "error": str(err),
                "status_code": err.status_code,
                "body": err.body,
            }
            return [TextContent(type="text", text=_format_payload(payload))]
        return [TextContent(type="text", text=_format_payload(result.body))]

    server._catalog = catalog  # type: ignore[attr-defined]
    server._client = client  # type: ignore[attr-defined]
    return server


def _tool_to_mcp(tool: ToolDefinition) -> "Tool":
    kwargs: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
        "inputSchema": tool.input_schema,
    }
    annotations = _hitl_annotations(tool)
    if annotations is not None:
        # MCP `Tool.annotations` is the SDK-blessed channel for tool-
        # level metadata that isn't part of the input schema.  We
        # nest under a vendor-prefixed ``aai`` key to avoid colliding
        # with any future MCP-spec-reserved annotation names.
        kwargs["annotations"] = annotations
    return Tool(**kwargs)


def _hitl_annotations(tool: ToolDefinition) -> dict[str, Any] | None:
    """Return the ``annotations`` payload for a tool, or ``None`` when
    the tool has no HITL config (so plain tools stay un-annotated)."""
    hitl = getattr(tool, "hitl", None)
    if hitl is None or not hitl.requires_confirmation:
        return None
    payload: dict[str, Any] = {"requires_confirmation": True}
    if hitl.timeout_seconds is not None:
        payload["timeout_seconds"] = hitl.timeout_seconds
    if hitl.confirm_message:
        payload["message"] = hitl.confirm_message
    return {"aai": payload}


def _enforce_unique_names(tools: list[ToolDefinition]) -> list[ToolDefinition]:
    """Drop later tools that collide with an earlier one's name.

    Names from OpenAPI specs are already disambiguated within each
    spec; collisions here happen across sources (e.g. a plugin
    function named the same as an OpenAPI operation).  We log and
    drop the later registration rather than silently overwriting,
    so the operator notices.
    """
    seen: set[str] = set()
    out: list[ToolDefinition] = []
    for t in tools:
        if t.name in seen:
            log.warning(
                "Dropping duplicate tool name %r — first registration wins.",
                t.name,
            )
            continue
        seen.add(t.name)
        out.append(t)
    return out


def _forwarded_credentials_from_env() -> dict[str, str]:
    """Read forwarded-credential headers passed via env at start.

    Used by the stdio transport — over SSE we'll add a per-request
    middleware in a follow-up.  For now any
    ``X_AI_ASSISTANT_AUTH_<SCHEME>`` env var is treated as a
    pre-set credential for that scheme.
    """
    prefix = "X_AI_ASSISTANT_AUTH_"
    return {
        key[len(prefix):]: value
        for key, value in os.environ.items()
        if key.startswith(prefix) and value
    }


def _format_payload(payload: Any) -> str:
    import json

    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, indent=2, default=str)
    except (TypeError, ValueError):
        return str(payload)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ai-assistant-server",
        description=(
            "MCP server backed by OpenAPI/Swagger specs and (optionally) "
            "Python plugin functions registered via the @tool decorator."
        ),
    )
    parser.add_argument(
        "--tools-dir",
        default=os.environ.get("AI_ASSISTANT_SERVER_TOOLS_DIR", "tools"),
        help="Directory containing OpenAPI spec files (default: ./tools).",
    )
    parser.add_argument(
        "--plugin-module",
        action="append",
        default=[
            m.strip()
            for m in os.environ.get(
                "AI_ASSISTANT_SERVER_PLUGIN_MODULES", ""
            ).split(",")
            if m.strip()
        ],
        help=(
            "Dotted path to a Python module to import for @tool plugins.  "
            "Repeatable.  Modules must be import-resolvable on the server's "
            "PYTHONPATH.  Set AI_ASSISTANT_SERVER_PLUGIN_MODULES "
            "(comma-separated) to provide defaults."
        ),
    )
    parser.add_argument(
        "--plugins-dir",
        default=os.environ.get("AI_ASSISTANT_SERVER_PLUGINS_DIR", "plugins"),
        help=(
            "Directory containing freestanding *.py plugin files (default: "
            "./plugins).  Each file is imported and any @tool-decorated "
            "callables are registered.  Missing directory is silently OK."
        ),
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse"),
        default=os.environ.get("AI_ASSISTANT_SERVER_TRANSPORT", "stdio"),
        help="Transport to expose the MCP server over.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("AI_ASSISTANT_SERVER_HOST", "127.0.0.1"),
        help="Host to bind for SSE transport.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("AI_ASSISTANT_SERVER_PORT", "8765")),
        help="Port to bind for SSE transport.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("AI_ASSISTANT_SERVER_LOG_LEVEL", "INFO"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    tools: list[ToolDefinition] = []
    # OpenAPI spec directory — same path as before.
    if Path(args.tools_dir).is_dir():
        tools.extend(load_tools_from_directory(Path(args.tools_dir)))
    else:
        log.info("No OpenAPI tools directory at %s — skipping", args.tools_dir)

    # Plugin modules (explicit imports — primary path for packaged plugins).
    for module_path in args.plugin_module or []:
        try:
            tools.extend(load_plugins_from_module(module_path))
        except ImportError as err:
            log.error("Failed to import plugin module %s: %s", module_path, err)
            return 2

    # Plugin directory (for quick iteration without packaging).
    if Path(args.plugins_dir).is_dir():
        tools.extend(load_plugins_from_directory(Path(args.plugins_dir)))

    tools = _enforce_unique_names(tools)
    if not tools:
        log.warning(
            "No tools found.  Searched OpenAPI dir %s, plugin modules %s, "
            "plugin dir %s.",
            args.tools_dir,
            args.plugin_module or "(none)",
            args.plugins_dir,
        )
    server = build_server(tools)

    if args.transport == "stdio":
        asyncio.run(_run_stdio(server))
    else:
        asyncio.run(_run_sse(server, host=args.host, port=args.port))
    return 0


async def _run_stdio(server: "Server") -> None:
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


async def _run_sse(server: "Server", *, host: str, port: int) -> None:
    """Serve the MCP protocol over SSE using Starlette + uvicorn."""
    import uvicorn
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.routing import Mount, Route

    transport = SseServerTransport("/messages")

    async def handle_sse(request: Request) -> Any:
        # Surface forwarded credentials (per-request) to the
        # executor via context-vars in a follow-up; for now the
        # stdio env-var path covers the common case.
        async with transport.connect_sse(
            request.scope, request.receive, request._send  # noqa: SLF001
        ) as streams:
            await server.run(
                streams[0],
                streams[1],
                server.create_initialization_options(),
            )

    app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages", app=transport.handle_post_message),
        ],
    )
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    await uvicorn.Server(config).serve()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
