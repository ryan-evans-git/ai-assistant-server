"""Executor tests — given a OpenApiTool + args, dispatch the HTTP call."""

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
    OpenApiTool,
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
    spec_url: str | None = None,
) -> OpenApiTool:
    return OpenApiTool(
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
        spec_url=spec_url,
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
async def test_post_without_body_sends_empty_object() -> None:
    """Specs that declare POST/PUT/PATCH/DELETE without a requestBody
    used to send no body and no Content-Type header — some upstreams
    reject that as a protocol error.  We default to {} so httpx
    attaches Content-Type: application/json + a non-zero
    Content-Length."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["content_type"] = request.headers.get("content-type")
        captured["content_length"] = request.headers.get("content-length")
        captured["body"] = request.content
        return httpx.Response(200, json={"ok": True})

    tool = _tool(method="post", path="/things", locations={})
    async with httpx.AsyncClient(transport=_mock_transport(handler)) as client:
        result = await execute_tool(tool, {}, client=client)

    assert result.status_code == 200
    assert captured["method"] == "POST"
    assert captured["content_type"] == "application/json"
    # `{}` serializes to 2 bytes — Content-Length must reflect that,
    # not be missing.
    assert captured["content_length"] == "2"
    assert captured["body"] == b"{}"


@pytest.mark.asyncio
async def test_get_without_body_does_not_inject_empty_object() -> None:
    """The empty-body fallback applies only to body-bearing methods.
    GET/HEAD/OPTIONS must remain bodyless or a few servers will 400."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["content_type"] = request.headers.get("content-type")
        captured["body"] = request.content
        return httpx.Response(200, json={"ok": True})

    tool = _tool(method="get", path="/things", locations={})
    async with httpx.AsyncClient(transport=_mock_transport(handler)) as client:
        await execute_tool(tool, {}, client=client)

    assert captured["method"] == "GET"
    assert captured["content_type"] is None
    assert captured["body"] == b""


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


# ---------------------------------------------------------------------------
# Spec-drift refresh-and-retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drift_404_triggers_refresh_and_retries_with_new_tool() -> None:
    """A 404 on a tool that declares a spec URL invokes the refresh
    callback; if it returns a tool with a different execution path,
    the executor retries exactly once against the new path."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path == "/things/7":
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(200, json={"ok": True})

    old = _tool(
        path="/things/{id}",
        locations={"id": "path"},
        spec_url="https://example.com/openapi.json",
    )
    new = OpenApiTool(
        name=old.name,
        description=old.description,
        input_schema=old.input_schema,
        execution=HttpExecution(
            base_url=old.execution.base_url,
            method="get",
            path="/v2/things/{id}",
            parameter_locations={"id": "path"},
        ),
        spec_url=old.spec_url,
    )

    refresh_calls: list[OpenApiTool] = []

    async def refresh(t: OpenApiTool, err: ToolExecutionError) -> OpenApiTool:
        refresh_calls.append(t)
        return new

    async with httpx.AsyncClient(transport=_mock_transport(handler)) as client:
        result = await execute_tool(
            old, {"id": 7}, client=client, refresh=refresh
        )

    assert result.status_code == 200
    assert result.body == {"ok": True}
    assert refresh_calls == [old]
    assert calls == [
        "https://api.example.com/things/7",
        "https://api.example.com/v2/things/7",
    ]


@pytest.mark.asyncio
async def test_drift_410_also_triggers_refresh() -> None:
    """410 Gone is the other drift-shaped code we recognize."""
    served: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if not served:
            served.append(410)
            return httpx.Response(410, json={"error": "gone"})
        return httpx.Response(200, json={"ok": True})

    old = _tool(
        path="/old",
        locations={},
        spec_url="https://example.com/openapi.json",
    )
    new = OpenApiTool(
        name=old.name,
        description=old.description,
        input_schema=old.input_schema,
        execution=HttpExecution(
            base_url=old.execution.base_url,
            method="get",
            path="/new",
            parameter_locations={},
        ),
        spec_url=old.spec_url,
    )

    async def refresh(t: OpenApiTool, err: ToolExecutionError) -> OpenApiTool:
        return new

    async with httpx.AsyncClient(transport=_mock_transport(handler)) as client:
        result = await execute_tool(old, {}, client=client, refresh=refresh)
    assert result.status_code == 200


@pytest.mark.asyncio
async def test_non_drift_error_does_not_call_refresh() -> None:
    """500s / 401s / 400s are not drift-shaped — refresh stays
    untouched and the original error surfaces immediately."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    tool = _tool(
        path="/things",
        locations={},
        spec_url="https://example.com/openapi.json",
    )

    refresh_called = False

    async def refresh(t: OpenApiTool, err: ToolExecutionError) -> OpenApiTool | None:
        nonlocal refresh_called
        refresh_called = True
        return None

    async with httpx.AsyncClient(transport=_mock_transport(handler)) as client:
        with pytest.raises(ToolExecutionError) as excinfo:
            await execute_tool(tool, {}, client=client, refresh=refresh)
    assert excinfo.value.status_code == 500
    assert refresh_called is False


@pytest.mark.asyncio
async def test_refresh_skipped_when_tool_has_no_spec_url() -> None:
    """A drift-shaped error on a tool without ``spec_url`` should
    propagate immediately without consulting the callback."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    tool = _tool(path="/things", locations={}, spec_url=None)
    refresh_called = False

    async def refresh(t: OpenApiTool, err: ToolExecutionError) -> OpenApiTool | None:
        nonlocal refresh_called
        refresh_called = True
        return None

    async with httpx.AsyncClient(transport=_mock_transport(handler)) as client:
        with pytest.raises(ToolExecutionError) as excinfo:
            await execute_tool(tool, {}, client=client, refresh=refresh)
    assert excinfo.value.status_code == 404
    assert refresh_called is False


@pytest.mark.asyncio
async def test_refresh_returning_none_propagates_original_error() -> None:
    """The refresher signals 'no usable refresh' by returning None;
    the executor must surface the original 404 in that case."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    tool = _tool(
        path="/things",
        locations={},
        spec_url="https://example.com/openapi.json",
    )

    async def refresh(t: OpenApiTool, err: ToolExecutionError) -> OpenApiTool | None:
        return None

    async with httpx.AsyncClient(transport=_mock_transport(handler)) as client:
        with pytest.raises(ToolExecutionError) as excinfo:
            await execute_tool(tool, {}, client=client, refresh=refresh)
    assert excinfo.value.status_code == 404


@pytest.mark.asyncio
async def test_refresh_capped_at_one_retry_even_if_new_tool_also_404s() -> None:
    """If the refreshed tool ALSO 404s, we surface the second error
    instead of looping — the refresh callback is consumed after
    the first drift event."""
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        return httpx.Response(404, json={"error": "not found"})

    old = _tool(
        path="/v1/things",
        locations={},
        spec_url="https://example.com/openapi.json",
    )
    new = OpenApiTool(
        name=old.name,
        description=old.description,
        input_schema=old.input_schema,
        execution=HttpExecution(
            base_url=old.execution.base_url,
            method="get",
            path="/v2/things",
            parameter_locations={},
        ),
        spec_url=old.spec_url,
    )

    refresh_calls = 0

    async def refresh(t: OpenApiTool, err: ToolExecutionError) -> OpenApiTool:
        nonlocal refresh_calls
        refresh_calls += 1
        return new

    async with httpx.AsyncClient(transport=_mock_transport(handler)) as client:
        with pytest.raises(ToolExecutionError):
            await execute_tool(old, {}, client=client, refresh=refresh)

    # Exactly two upstream calls (original + one retry), and exactly
    # one refresh.
    assert seen_paths == ["/v1/things", "/v2/things"]
    assert refresh_calls == 1


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
