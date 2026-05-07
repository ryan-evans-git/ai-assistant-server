"""Executor for tool calls — dispatches by tool kind.

For an :class:`OpenApiTool`, this builds the upstream HTTP request
(path templates, query / header / cookie parameters, JSON body, and
resolved auth) and returns the upstream response body.

For a :class:`PluginTool`, this calls the registered Python handler
with the parsed argument dict, awaits it if it's a coroutine, and
wraps the return value in a :class:`ToolResult`.

Error semantics:
    * Network errors → :class:`ToolExecutionError`.
    * 4xx / 5xx responses → :class:`ToolExecutionError` with the
      status code in the message + body included so the agent
      can recover (e.g. "argument was missing — retry").
    * Plugin handler exceptions are wrapped in a
      :class:`ToolExecutionError` with ``status_code=None``.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import httpx

from ai_assistant_server.auth import AppliedAuth, resolve_auth
from ai_assistant_server.models import (
    OpenApiTool,
    PluginTool,
    ToolDefinition,
)


class ToolExecutionError(RuntimeError):
    """Raised when a tool call fails to dispatch or returns non-2xx."""

    def __init__(self, message: str, *, status_code: int | None = None, body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


@dataclass(frozen=True)
class ToolResult:
    status_code: int
    body: Any  # parsed JSON or raw text
    headers: dict[str, str]


async def execute_tool(
    tool: ToolDefinition,
    arguments: dict[str, Any] | None,
    *,
    client: httpx.AsyncClient | None = None,
    forwarded_credentials: dict[str, str] | None = None,
) -> ToolResult:
    """Run a tool call.

    For OpenAPI tools, ``client`` (an ``httpx.AsyncClient``) shares a
    connection pool across calls; one is created per-call when not
    provided.  Plugin tools ignore the HTTP-only kwargs.
    """
    if isinstance(tool, PluginTool):
        return await _execute_plugin(tool, arguments or {})
    return await _execute_openapi(
        tool, arguments or {}, client=client, forwarded_credentials=forwarded_credentials
    )


# ---------------------------------------------------------------------------
# Plugin dispatch
# ---------------------------------------------------------------------------


async def _execute_plugin(
    tool: PluginTool,
    arguments: dict[str, Any],
) -> ToolResult:
    """Call the Python handler.  Sync handlers run inline; coroutines
    are awaited.  Any exception escapes as a :class:`ToolExecutionError`
    so the MCP server's call_tool wrapper can surface it as text the
    agent can react to."""
    try:
        result = tool.handler(**arguments)
        if inspect.isawaitable(result):
            result = await result
    except TypeError as err:
        # Most likely an argument-shape mismatch the agent supplied.
        raise ToolExecutionError(
            f"plugin '{tool.name}' rejected arguments: {err}",
            status_code=None,
        ) from err
    except Exception as err:  # noqa: BLE001
        raise ToolExecutionError(
            f"plugin '{tool.name}' raised: {err}",
            status_code=None,
        ) from err

    return ToolResult(status_code=200, body=result, headers={})


# ---------------------------------------------------------------------------
# OpenAPI dispatch
# ---------------------------------------------------------------------------


async def _execute_openapi(
    tool: OpenApiTool,
    arguments: dict[str, Any],
    *,
    client: httpx.AsyncClient | None,
    forwarded_credentials: dict[str, str] | None,
) -> ToolResult:
    auth = resolve_auth(tool.auth, forwarded_credentials=forwarded_credentials)

    url, query, headers, body = _materialize_request(tool, arguments, auth)
    timeout = tool.execution.timeout_seconds
    method = tool.execution.method.upper()

    # Some specs declare body-bearing methods (POST/PUT/PATCH/DELETE)
    # without a requestBody.  If we then call httpx with `json=None`,
    # it sends no body and *no* Content-Type/Content-Length headers —
    # which a non-trivial number of upstreams reject as a protocol
    # error ("411 Length Required" or similar).  For those methods,
    # default to an empty JSON object so httpx attaches the proper
    # Content-Type: application/json + Content-Length: 2 headers.
    json_body: Any
    if body is not None:
        json_body = body
    elif method in ("POST", "PUT", "PATCH", "DELETE"):
        json_body = {}
    else:
        json_body = None

    own_client = client is None
    client = client or httpx.AsyncClient(timeout=timeout)
    try:
        try:
            response = await client.request(
                method,
                url,
                params=query or None,
                headers=headers or None,
                json=json_body,
                timeout=timeout,
            )
        except httpx.HTTPError as err:
            raise ToolExecutionError(
                f"upstream request failed: {err}",
                status_code=None,
            ) from err
    finally:
        if own_client:
            await client.aclose()

    parsed = _parse_response_body(response)
    if response.status_code >= 400:
        raise ToolExecutionError(
            f"upstream returned {response.status_code}",
            status_code=response.status_code,
            body=parsed,
        )
    return ToolResult(
        status_code=response.status_code,
        body=parsed,
        headers=dict(response.headers),
    )


def _materialize_request(
    tool: OpenApiTool,
    arguments: dict[str, Any],
    auth: AppliedAuth,
) -> tuple[str, dict[str, str], dict[str, str], Any]:
    base_url = tool.execution.base_url
    if not base_url:
        raise ToolExecutionError(
            f"tool '{tool.name}' has no base URL — set "
            "AI_ASSISTANT_SERVER_BASE_URL_OVERRIDE or add a "
            "servers[].url entry to its OpenAPI spec",
        )

    path = tool.execution.path
    query: dict[str, str] = {}
    headers: dict[str, str] = {}
    cookie_parts: list[str] = []
    body: Any = None

    locations = tool.execution.parameter_locations

    for name, value in arguments.items():
        location = locations.get(name)
        if location == "path":
            path = path.replace("{" + name + "}", _stringify(value))
        elif location == "query":
            query[name] = _stringify(value)
        elif location == "header":
            headers[name] = _stringify(value)
        elif location == "cookie":
            cookie_parts.append(f"{name}={_stringify(value)}")
        elif location == "body":
            body = value
        # Unknown locations are silently ignored — agents may
        # over-specify; the upstream will reject if it matters.

    if cookie_parts:
        existing = headers.get("Cookie")
        joined = "; ".join(cookie_parts)
        headers["Cookie"] = f"{existing}; {joined}" if existing else joined

    headers.update(auth.headers)
    query.update(auth.query)
    url = f"{base_url.rstrip('/')}{path}"
    return url, query, headers, body


def _parse_response_body(response: httpx.Response) -> Any:
    content_type = response.headers.get("content-type", "")
    if "json" in content_type.lower():
        try:
            return response.json()
        except ValueError:
            return response.text
    return response.text


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(_stringify(v) for v in value)
    return str(value)
