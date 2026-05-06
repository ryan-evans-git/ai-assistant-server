"""HTTP executor for tool calls derived from OpenAPI specs.

Given a :class:`ToolDefinition` and a JSON arguments dict, build
the upstream HTTP request: substitute path templates, attach
query / header / cookie parameters, JSON-encode the body, and
apply the resolved auth.  Returns the upstream response body
(JSON-decoded when possible) plus status code.

Error semantics:
    * Network errors → :class:`ToolExecutionError`.
    * 4xx / 5xx responses → :class:`ToolExecutionError` with the
      status code in the message + body included so the agent
      can recover (e.g. "argument was missing — retry").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from ai_assistant_server.auth import AppliedAuth, resolve_auth
from ai_assistant_server.models import ToolDefinition


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
    """Issue the HTTP request behind a tool call.

    Pass an ``httpx.AsyncClient`` to share a connection pool
    across calls; one is created per-call otherwise.
    """
    args = arguments or {}
    auth = resolve_auth(tool.auth, forwarded_credentials=forwarded_credentials)

    url, query, headers, body = _materialize_request(tool, args, auth)
    timeout = tool.execution.timeout_seconds

    own_client = client is None
    client = client or httpx.AsyncClient(timeout=timeout)
    try:
        try:
            response = await client.request(
                tool.execution.method.upper(),
                url,
                params=query or None,
                headers=headers or None,
                json=body if body is not None else None,
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
    tool: ToolDefinition,
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
