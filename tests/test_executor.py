"""Executor tests — given a ToolDefinition + args, dispatch the HTTP call."""

from __future__ import annotations

import httpx
import pytest

from ai_assistant_server.executor import (
    ToolExecutionError,
    _materialize_request,
    execute_tool,
)
from ai_assistant_server.models import (
    AuthConfig,
    AuthScheme,
    HttpExecution,
    ToolDefinition,
)
from ai_assistant_server.auth import AppliedAuth


def _tool(
    *,
    method: str = "get",
    path: str = "/things/{id}",
    locations: dict[str, str] | None = None,
    base_url: str = "https://api.example.com",
    auth: AuthConfig | None = None,
    request_body_required: bool = False,
) -> ToolDefinition:
    return ToolDefinition(
        name="get_thing",
        description="Get a thing.",
        input_schema={"type": "object", "properties": {}},
        execution=HttpExecution(
            base_url=base_url,
            method=method,
            path=path,
            parameter_locations=locations or {},
            request_body_required=request_body_required,
            request_body_property="body" if request_body_required else None,
        ),
        auth=auth or AuthConfig(),
    )


def test_path_substitution() -> None:
    tool = _tool(locations={"id": "path"})
    url, query, headers, body = _materialize_request(
        tool,
        {"id": 42},
        AppliedAuth(headers={}, query={}),
    )
    assert url == "https://api.example.com/things/42"
    assert query == {}
    assert body is None


def test_query_and_header_split() -> None:
    tool = _tool(
        path="/search",
        locations={"q": "query", "X-Trace": "header"},
    )
    url, query, headers, body = _materialize_request(
        tool,
        {"q": "hello world", "X-Trace": "abc"},
        AppliedAuth(headers={}, query={}),
    )
    assert url == "https://api.example.com/search"
    assert query == {"q": "hello world"}
    assert headers == {"X-Trace": "abc"}


def test_body_passthrough() -> None:
    tool = _tool(
        method="post",
        path="/widgets",
        locations={"body": "body"},
        request_body_required=True,
    )
    url, query, headers, body = _materialize_request(
        tool,
        {"body": {"name": "wedge", "size": 3}},
        AppliedAuth(headers={}, query={}),
    )
    assert body == {"name": "wedge", "size": 3}


def test_auth_headers_applied() -> None:
    tool = _tool(locations={"id": "path"})
    url, query, headers, body = _materialize_request(
        tool,
        {"id": 1},
        AppliedAuth(headers={"Authorization": "Bearer x"}, query={}),
    )
    assert headers["Authorization"] == "Bearer x"


def test_auth_query_applied() -> None:
    tool = _tool(path="/search", locations={"q": "query"})
    url, query, headers, body = _materialize_request(
        tool,
        {"q": "hi"},
        AppliedAuth(headers={}, query={"api_key": "abc"}),
    )
    assert query == {"q": "hi", "api_key": "abc"}


def test_missing_base_url_raises() -> None:
    tool = _tool(base_url="", locations={"id": "path"})
    with pytest.raises(ToolExecutionError):
        _materialize_request(
            tool,
            {"id": 1},
            AppliedAuth(headers={}, query={}),
        )


def test_unknown_param_silently_ignored() -> None:
    """Agents may over-specify args; upstream rejects if it matters."""
    tool = _tool(locations={"id": "path"})
    url, query, headers, body = _materialize_request(
        tool,
        {"id": 1, "extra": "unused"},
        AppliedAuth(headers={}, query={}),
    )
    assert url == "https://api.example.com/things/1"
    assert query == {}


def test_cookie_header_built() -> None:
    tool = _tool(path="/me", locations={"session": "cookie"})
    url, query, headers, body = _materialize_request(
        tool,
        {"session": "abc"},
        AppliedAuth(headers={}, query={}),
    )
    assert headers["Cookie"] == "session=abc"


# ---------------------------------------------------------------------------
# End-to-end execute_tool with a mocked transport.
# ---------------------------------------------------------------------------


def _mock_transport(handler):
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_execute_tool_happy_path() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        return httpx.Response(200, json={"ok": True})

    tool = _tool(locations={"id": "path"})
    async with httpx.AsyncClient(transport=_mock_transport(handler)) as client:
        result = await execute_tool(tool, {"id": 7}, client=client)

    assert result.status_code == 200
    assert result.body == {"ok": True}
    assert captured["url"] == "https://api.example.com/things/7"
    assert captured["method"] == "GET"


@pytest.mark.asyncio
async def test_execute_tool_4xx_raises_with_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    tool = _tool(locations={"id": "path"})
    async with httpx.AsyncClient(transport=_mock_transport(handler)) as client:
        with pytest.raises(ToolExecutionError) as excinfo:
            await execute_tool(tool, {"id": 7}, client=client)
    assert excinfo.value.status_code == 404
    assert excinfo.value.body == {"error": "not found"}


@pytest.mark.asyncio
async def test_execute_tool_with_bearer_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_ASSISTANT_SERVER_AUTH_BEARERAUTH", "tok-xyz")
    seen_auth: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("authorization"))
        return httpx.Response(200, json={})

    tool = _tool(
        locations={"id": "path"},
        auth=AuthConfig(
            scheme=AuthScheme.BEARER,
            secret_env="AI_ASSISTANT_SERVER_AUTH_BEARERAUTH",
            scheme_name="bearerAuth",
        ),
    )
    async with httpx.AsyncClient(transport=_mock_transport(handler)) as client:
        await execute_tool(tool, {"id": 1}, client=client)
    assert seen_auth == ["Bearer tok-xyz"]


@pytest.mark.asyncio
async def test_execute_tool_forwarded_credentials() -> None:
    seen_auth: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("authorization"))
        return httpx.Response(200, json={})

    tool = _tool(
        locations={"id": "path"},
        auth=AuthConfig(
            scheme=AuthScheme.BEARER,
            secret_env="UNSET",
            scheme_name="bearerAuth",
        ),
    )
    async with httpx.AsyncClient(transport=_mock_transport(handler)) as client:
        await execute_tool(
            tool,
            {"id": 1},
            client=client,
            forwarded_credentials={"bearerAuth": "tok-from-host"},
        )
    assert seen_auth == ["Bearer tok-from-host"]
