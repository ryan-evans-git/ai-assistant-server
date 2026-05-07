"""Targeted tests to fill remaining coverage gaps across modules.

Each test below is named for the exact uncovered branch / line it
exercises so that — if a line later becomes covered by a more
natural test — the redundancy is easy to spot and prune.
"""

from __future__ import annotations

import inspect
import logging
import textwrap
from pathlib import Path
from typing import Any

import httpx
import pytest

from ai_assistant_server import tool
from ai_assistant_server.auth import AuthResolutionError, resolve_auth
from ai_assistant_server.executor import (
    ToolExecutionError,
    _parse_response_body,
    _stringify,
    execute_tool,
)
from ai_assistant_server.loader import (
    _annotate_schema,
    _request_body_schema,
    _resolve_base_url,
    _scheme_to_auth_config,
    load_plugins_from_directory,
    load_tools_from_directory,
)
from ai_assistant_server.models import (
    AuthConfig,
    AuthScheme,
    HttpExecution,
    OpenApiTool,
)
from ai_assistant_server.plugins import derive_input_schema
from ai_assistant_server.server import _format_payload


# ---------------------------------------------------------------------------
# auth.py — apiKey schemes missing parameter_name (lines 73-76, 79-82)
# and _read_secret with no secret_env (line 98)
# ---------------------------------------------------------------------------


def test_api_key_header_without_parameter_name_raises() -> None:
    cfg = AuthConfig(
        scheme=AuthScheme.API_KEY_HEADER,
        scheme_name="apiKey",
        parameter_name=None,
    )
    with pytest.raises(AuthResolutionError, match="missing parameter name"):
        resolve_auth(cfg, forwarded_credentials={"apiKey": "abc"})


def test_api_key_query_without_parameter_name_raises() -> None:
    cfg = AuthConfig(
        scheme=AuthScheme.API_KEY_QUERY,
        scheme_name="apiKey",
        parameter_name=None,
    )
    with pytest.raises(AuthResolutionError, match="missing parameter name"):
        resolve_auth(cfg, forwarded_credentials={"apiKey": "abc"})


def test_read_secret_returns_none_when_no_env_or_forward() -> None:
    """Bearer scheme with empty secret_env and no forwarded creds —
    _read_secret hits the trailing ``return None`` branch."""
    cfg = AuthConfig(
        scheme=AuthScheme.BEARER,
        secret_env="",
        scheme_name="bearerAuth",
    )
    with pytest.raises(AuthResolutionError, match="Missing credential"):
        resolve_auth(cfg)


# ---------------------------------------------------------------------------
# executor.py — body passthrough (line 131), httpx network error (149-150),
# own_client cleanup (156), invalid-JSON content-type (224-225),
# _stringify bool / list (231, 233)
# ---------------------------------------------------------------------------


def _post_tool() -> OpenApiTool:
    return OpenApiTool(
        name="create_widget",
        description="d",
        input_schema={"type": "object", "properties": {}},
        execution=HttpExecution(
            base_url="https://api.example.com",
            method="post",
            path="/widgets",
            parameter_locations={"body": "body"},
            request_body_required=True,
            request_body_property="body",
        ),
        auth=AuthConfig(),
    )


@pytest.mark.asyncio
async def test_execute_tool_post_with_explicit_body_passthrough() -> None:
    """Body provided in arguments routes through ``json_body = body``."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(201, json={"id": 1})

    tool_def = _post_tool()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await execute_tool(
            tool_def, {"body": {"name": "wedge"}}, client=client
        )
    assert result.status_code == 201
    assert captured["body"] == b'{"name":"wedge"}'


@pytest.mark.asyncio
async def test_execute_tool_wraps_network_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("conn refused")

    tool_def = _post_tool()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ToolExecutionError, match="upstream request failed"):
            await execute_tool(tool_def, {"body": {}}, client=client)


@pytest.mark.asyncio
async def test_execute_tool_creates_and_closes_own_client_when_none_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When called without ``client=``, the executor builds + closes its
    own httpx client.  We swap in a transport that records the close()."""
    closed = {"flag": False}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    real_init = httpx.AsyncClient.__init__

    def patched_init(self, *args: Any, **kwargs: Any) -> None:
        kwargs["transport"] = httpx.MockTransport(handler)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)

    real_aclose = httpx.AsyncClient.aclose

    async def patched_aclose(self) -> None:
        closed["flag"] = True
        await real_aclose(self)

    monkeypatch.setattr(httpx.AsyncClient, "aclose", patched_aclose)

    tool_def = OpenApiTool(
        name="get_thing",
        description="d",
        input_schema={"type": "object", "properties": {}},
        execution=HttpExecution(
            base_url="https://api.example.com",
            method="get",
            path="/x",
        ),
        auth=AuthConfig(),
    )
    result = await execute_tool(tool_def, {})
    assert result.status_code == 200
    assert closed["flag"] is True


def test_parse_response_body_invalid_json_falls_back_to_text() -> None:
    """``content-type: application/json`` but body isn't valid JSON —
    return raw text instead of raising."""
    response = httpx.Response(
        200,
        headers={"content-type": "application/json"},
        content=b"not actually json",
    )
    assert _parse_response_body(response) == "not actually json"


def test_parse_response_body_text_content_type_returns_text() -> None:
    response = httpx.Response(
        200,
        headers={"content-type": "text/plain"},
        content=b"hello",
    )
    assert _parse_response_body(response) == "hello"


def test_stringify_bool_lowers() -> None:
    assert _stringify(True) == "true"
    assert _stringify(False) == "false"


def test_stringify_list_joins_with_commas() -> None:
    assert _stringify([1, 2, 3]) == "1,2,3"
    assert _stringify((True, False)) == "true,false"


# ---------------------------------------------------------------------------
# plugins.py — get_type_hints fallback (126-130) and self/cls skip (135)
# ---------------------------------------------------------------------------


def test_derive_input_schema_falls_back_when_type_hints_unresolvable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``get_type_hints`` raises (e.g. unresolvable forward refs),
    the loader falls back to bare ``signature`` annotations so the
    schema still gets built."""

    def boom(*_a: Any, **_kw: Any) -> dict[str, Any]:
        raise NameError("forward ref")

    monkeypatch.setattr(
        "ai_assistant_server.plugins.get_type_hints", boom
    )

    def fn(x: int = 5) -> int:
        return x

    schema = derive_input_schema(fn)
    assert schema["type"] == "object"
    assert "x" in schema["properties"]


def test_derive_input_schema_skips_self_and_cls() -> None:
    """Bound-method introspection — ``self``/``cls`` should not appear
    in the schema."""

    class Holder:
        def method(self, value: int) -> int:
            return value

        @classmethod
        def cmethod(cls, value: int) -> int:
            return value

    schema = derive_input_schema(Holder.method)
    assert "self" not in schema["properties"]
    assert "value" in schema["properties"]
    schema_c = derive_input_schema(Holder.cmethod.__func__)  # type: ignore[attr-defined]
    assert "cls" not in schema_c["properties"]


# ---------------------------------------------------------------------------
# server.py — _format_payload TypeError/ValueError fallback (159-160)
# ---------------------------------------------------------------------------


def test_format_payload_circular_ref_falls_back_to_str() -> None:
    """``json.dumps`` raises ``ValueError: Circular reference detected``
    for self-referencing dicts — the helper must fall back to ``str()``."""
    d: dict[str, Any] = {}
    d["self"] = d
    out = _format_payload(d)
    # Whatever str(d) produces (Python's recursion-aware repr) should
    # be returned unchanged.
    assert isinstance(out, str)
    assert "self" in out


# ---------------------------------------------------------------------------
# loader.py — many small skipped paths (118, 124, 126, 216, 233, 260, 264,
# request-body fallbacks, swagger2 host paths, security branches,
# plugin-directory load failures)
# ---------------------------------------------------------------------------


def _write(tmp: Path, name: str, body: str) -> Path:
    p = tmp / name
    p.write_text(body, encoding="utf-8")
    return p


def test_path_item_non_dict_skipped(tmp_path: Path) -> None:
    """Top-level ``paths`` entry that isn't a dict (e.g. the spec
    accidentally maps a path to a string) should be quietly ignored."""
    spec = textwrap.dedent(
        """
        openapi: 3.0.3
        info:
          title: x
          version: 1
        servers: [{url: "https://x"}]
        paths:
          /weird: "this is not a path-item"
          /good:
            get:
              operationId: ok
              responses: {"200": {description: ok}}
        """
    )
    _write(tmp_path, "spec.yaml", spec)
    tools = load_tools_from_directory(tmp_path)
    assert {t.name for t in tools} == {"ok"}


def test_non_http_method_keys_skipped(tmp_path: Path) -> None:
    """Path items often carry ``parameters`` / ``summary`` / ``x-...``
    siblings — those must not be treated as operations."""
    spec = textwrap.dedent(
        """
        openapi: 3.0.3
        info:
          title: x
          version: 1
        servers: [{url: "https://x"}]
        paths:
          /thing:
            summary: a thing
            x-internal: yes
            parameters: []
            get:
              operationId: getThing
              responses: {"200": {description: ok}}
        """
    )
    _write(tmp_path, "spec.yaml", spec)
    tools = load_tools_from_directory(tmp_path)
    assert {t.name for t in tools} == {"getthing"}


def test_operation_value_non_dict_skipped(tmp_path: Path) -> None:
    """A method key whose value is malformed (string instead of an
    operation object) is skipped, not fatal."""
    spec = textwrap.dedent(
        """
        openapi: 3.0.3
        info:
          title: x
          version: 1
        servers: [{url: "https://x"}]
        paths:
          /a:
            get: "broken"
            post:
              operationId: ok
              responses: {"200": {description: ok}}
        """
    )
    _write(tmp_path, "spec.yaml", spec)
    tools = load_tools_from_directory(tmp_path)
    assert {t.name for t in tools} == {"ok"}


def test_disambiguate_three_collisions(tmp_path: Path) -> None:
    """Three operations with the same operationId — must produce ``_2``
    and ``_3`` suffixes (covers the increment-loop branch)."""
    spec = textwrap.dedent(
        """
        openapi: 3.0.3
        info:
          title: x
          version: 1
        servers: [{url: "https://x"}]
        paths:
          /a: {get: {operationId: dup, responses: {"200": {description: ok}}}}
          /b: {get: {operationId: dup, responses: {"200": {description: ok}}}}
          /c: {get: {operationId: dup, responses: {"200": {description: ok}}}}
        """
    )
    _write(tmp_path, "spec.yaml", spec)
    tools = load_tools_from_directory(tmp_path)
    assert {t.name for t in tools} == {"dup", "dup_2", "dup_3"}


def test_summary_and_distinct_description_concatenate(tmp_path: Path) -> None:
    """When summary and description differ, both go into the tool desc."""
    spec = textwrap.dedent(
        """
        openapi: 3.0.3
        info: {title: x, version: 1}
        servers: [{url: "https://x"}]
        paths:
          /a:
            get:
              operationId: getA
              summary: short
              description: long detail
              responses: {"200": {description: ok}}
        """
    )
    _write(tmp_path, "spec.yaml", spec)
    [tool_def] = load_tools_from_directory(tmp_path)
    assert "short" in tool_def.description
    assert "long detail" in tool_def.description


def test_parameters_non_dict_and_missing_name_skipped(tmp_path: Path) -> None:
    """Malformed parameter list entries should be quietly skipped."""
    spec = textwrap.dedent(
        """
        openapi: 3.0.3
        info: {title: x, version: 1}
        servers: [{url: "https://x"}]
        paths:
          /a:
            get:
              operationId: getA
              parameters:
                - "not a dict"
                - {in: query}            # missing name
                - {name: ok, in: query, schema: {type: string}}
              responses: {"200": {description: ok}}
        """
    )
    _write(tmp_path, "spec.yaml", spec)
    [tool_def] = load_tools_from_directory(tmp_path)
    assert "ok" in tool_def.input_schema["properties"]
    # "not a dict" + the {in: query} entry are silently dropped.
    assert len(tool_def.input_schema["properties"]) == 1


def test_request_body_with_no_content_returns_none() -> None:
    """A requestBody declaration without any ``content`` map yields
    ``None`` — the executor then has no body schema to attach."""
    assert _request_body_schema({}) is None


def test_request_body_falls_back_to_first_content_type() -> None:
    """When ``application/json`` isn't offered, the loader picks the
    first content type it finds."""
    rb = {
        "description": "form data",
        "content": {"application/xml": {"schema": {"type": "object"}}},
    }
    schema = _request_body_schema(rb)
    assert schema is not None
    assert schema["type"] == "object"


def test_request_body_with_non_dict_schema_is_skipped() -> None:
    rb = {"content": {"application/json": {"schema": "broken"}}}
    assert _request_body_schema(rb) is None


def test_annotate_schema_keeps_existing_description() -> None:
    """If the schema already has a description, the parameter-level
    description is *not* overwritten."""
    schema = {"type": "string", "description": "from schema"}
    out = _annotate_schema(schema, "from parameter")
    assert out["description"] == "from schema"


def test_annotate_schema_no_description_returns_unchanged() -> None:
    schema = {"type": "string"}
    out = _annotate_schema(schema, None)
    assert out == schema


def test_resolve_base_url_returns_empty_when_no_servers_or_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AI_ASSISTANT_SERVER_BASE_URL_OVERRIDE", raising=False)
    assert _resolve_base_url({}) == ""


def test_resolve_base_url_skips_non_dict_server_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``servers[]`` list with junk entries should keep walking until
    a usable URL appears."""
    monkeypatch.delenv("AI_ASSISTANT_SERVER_BASE_URL_OVERRIDE", raising=False)
    spec = {
        "servers": [
            "not a dict",
            {"url": ""},               # empty url
            {"url": "https://api.example.com"},
        ]
    }
    assert _resolve_base_url(spec) == "https://api.example.com"


def test_resolve_base_url_swagger2_without_schemes_defaults_https(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AI_ASSISTANT_SERVER_BASE_URL_OVERRIDE", raising=False)
    spec = {"host": "api.example.com"}
    assert _resolve_base_url(spec) == "https://api.example.com"


def test_security_schemes_legacy_swagger2_path() -> None:
    """Swagger 2.0 stores security schemes under top-level
    ``securityDefinitions``; the loader should fall back to those when
    ``components.securitySchemes`` is malformed (non-dict)."""
    from ai_assistant_server.loader import _security_schemes

    spec = {
        # Force the components-path to fail isinstance(dict) so the
        # legacy fallback runs.
        "components": {"securitySchemes": ["malformed-not-a-dict"]},
        "securityDefinitions": {
            "legacyKey": {"type": "apiKey", "in": "header", "name": "X-Api-Key"}
        },
    }
    schemes = _security_schemes(spec)
    assert "legacyKey" in schemes


def test_security_schemes_legacy_returns_empty_when_malformed() -> None:
    """Both components and securityDefinitions malformed → empty dict."""
    from ai_assistant_server.loader import _security_schemes

    spec = {
        "components": {"securitySchemes": ["bad"]},
        "securityDefinitions": "not a dict either",
    }
    assert _security_schemes(spec) == {}


def test_resolve_auth_unsupported_when_security_unmatched(
    tmp_path: Path,
) -> None:
    """Security requirement references an OAuth2 flow we don't execute —
    returns UNSUPPORTED so the call site fails loudly."""
    spec = textwrap.dedent(
        """
        openapi: 3.0.3
        info: {title: x, version: 1}
        servers: [{url: "https://x"}]
        security:
          - oauth: []
        paths:
          /me:
            get:
              operationId: getMe
              responses: {"200": {description: ok}}
        components:
          securitySchemes:
            oauth:
              type: oauth2
              flows:
                clientCredentials:
                  tokenUrl: https://x/token
                  scopes: {}
        """
    )
    _write(tmp_path, "spec.yaml", spec)
    [tool_def] = load_tools_from_directory(tmp_path)
    assert tool_def.auth.scheme is AuthScheme.UNSUPPORTED


def test_resolve_auth_skips_non_dict_requirements(tmp_path: Path) -> None:
    """An entry inside ``security`` that isn't a dict is silently
    skipped — the loader keeps walking the remaining requirements."""
    spec = textwrap.dedent(
        """
        openapi: 3.0.3
        info: {title: x, version: 1}
        servers: [{url: "https://x"}]
        security:
          - "not a dict"
          - bearerAuth: []
        paths:
          /me:
            get:
              operationId: getMe
              responses: {"200": {description: ok}}
        components:
          securitySchemes:
            bearerAuth:
              type: http
              scheme: bearer
        """
    )
    _write(tmp_path, "spec.yaml", spec)
    [tool_def] = load_tools_from_directory(tmp_path)
    assert tool_def.auth.scheme is AuthScheme.BEARER


def test_resolve_auth_skips_unknown_scheme_name(tmp_path: Path) -> None:
    """``security`` references a scheme not declared in
    ``components.securitySchemes`` — that requirement is skipped, but
    the loader should still see and use the next one."""
    spec = textwrap.dedent(
        """
        openapi: 3.0.3
        info: {title: x, version: 1}
        servers: [{url: "https://x"}]
        security:
          - phantomScheme: []
          - bearerAuth: []
        paths:
          /me:
            get:
              operationId: getMe
              responses: {"200": {description: ok}}
        components:
          securitySchemes:
            bearerAuth:
              type: http
              scheme: bearer
        """
    )
    _write(tmp_path, "spec.yaml", spec)
    [tool_def] = load_tools_from_directory(tmp_path)
    assert tool_def.auth.scheme is AuthScheme.BEARER


def test_scheme_to_auth_config_basic() -> None:
    cfg = _scheme_to_auth_config("basicAuth", {"type": "http", "scheme": "basic"})
    assert cfg.scheme is AuthScheme.BASIC
    assert cfg.scheme_name == "basicAuth"


def test_scheme_to_auth_config_apikey_header() -> None:
    cfg = _scheme_to_auth_config(
        "apiKey", {"type": "apiKey", "in": "header", "name": "X-Api-Key"}
    )
    assert cfg.scheme is AuthScheme.API_KEY_HEADER
    assert cfg.parameter_name == "X-Api-Key"


def test_scheme_to_auth_config_unsupported_oauth2() -> None:
    cfg = _scheme_to_auth_config("oauth", {"type": "oauth2", "flows": {}})
    assert cfg.scheme is AuthScheme.UNSUPPORTED


def test_scheme_to_auth_config_apikey_unknown_location() -> None:
    """``apiKey`` with ``in: cookie`` (not header/query) → UNSUPPORTED."""
    cfg = _scheme_to_auth_config(
        "apiKey", {"type": "apiKey", "in": "cookie", "name": "session"}
    )
    assert cfg.scheme is AuthScheme.UNSUPPORTED


def test_load_plugins_from_directory_skips_unparseable_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A *.py file with a syntax error is logged + skipped rather than
    taking down the whole catalog."""
    (tmp_path / "broken.py").write_text("def def def\n", encoding="utf-8")
    (tmp_path / "ok.py").write_text(
        "from ai_assistant_server import tool\n\n"
        "@tool(name='good', description='d')\n"
        "def good() -> str:\n"
        "    return 'ok'\n"
    )
    with caplog.at_level(logging.WARNING):
        plugins = load_plugins_from_directory(tmp_path)
    assert [p.name for p in plugins] == ["good"]
    assert any("Skipping plugin file" in r.message for r in caplog.records)
