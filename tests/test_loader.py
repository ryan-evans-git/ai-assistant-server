"""Loader tests — OpenAPI specs in, ToolDefinitions out."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

import httpx

from ai_assistant_server.loader import (
    _extract_spec_url,
    _resolve_base_url,
    _slugify,
    _tool_name,
    fetch_spec_from_url,
    load_tools_from_directory,
    spec_hash,
    tools_from_spec_doc,
)
from ai_assistant_server.models import AuthScheme


PETSTORE_SPEC = textwrap.dedent(
    """
    openapi: 3.0.3
    info:
      title: Petstore
      version: 1.0.0
    servers:
      - url: https://petstore3.swagger.io/api/v3
    paths:
      /pet/{petId}:
        get:
          operationId: getPetById
          summary: Find pet by ID
          parameters:
            - name: petId
              in: path
              required: true
              schema:
                type: integer
          responses:
            "200":
              description: ok
      /pet/findByStatus:
        get:
          operationId: findPetsByStatus
          summary: Find pets by status
          parameters:
            - name: status
              in: query
              required: false
              schema:
                type: string
                enum: [available, pending, sold]
          responses:
            "200":
              description: ok
    """
)


BEARER_SPEC = textwrap.dedent(
    """
    openapi: 3.0.3
    info:
      title: Bearer Sample
      version: 1.0.0
    servers:
      - url: https://api.example.com
    security:
      - bearerAuth: []
    paths:
      /me:
        get:
          operationId: getMe
          summary: Current user
          responses:
            "200":
              description: ok
    components:
      securitySchemes:
        bearerAuth:
          type: http
          scheme: bearer
    """
)


API_KEY_QUERY_SPEC = textwrap.dedent(
    """
    openapi: 3.0.3
    info:
      title: Weather
      version: 1.0.0
    servers:
      - url: https://api.weather.example
    security:
      - apiKey: []
    paths:
      /forecast:
        get:
          operationId: getForecast
          summary: Forecast
          parameters:
            - name: city
              in: query
              required: true
              schema:
                type: string
          responses:
            "200":
              description: ok
    components:
      securitySchemes:
        apiKey:
          type: apiKey
          in: query
          name: api_key
    """
)


def _write_spec(tmp_path: Path, filename: str, content: str) -> Path:
    path = tmp_path / filename
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Slugify / naming
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("getPetById", "getpetbyid"),
        ("listPosts", "listposts"),
        ("get_/pet/{petId}", "get_pet_petid"),
        ("123abc", "op_123abc"),
        ("", "tool"),
    ],
)
def test_slugify(raw: str, expected: str) -> None:
    assert _slugify(raw) == expected


def test_tool_name_prefers_operation_id() -> None:
    assert _tool_name({"operationId": "getMe"}, "get", "/me") == "getme"


def test_tool_name_falls_back_to_method_path() -> None:
    assert _tool_name({}, "get", "/users/{id}") == "get_users_id"


# ---------------------------------------------------------------------------
# Base URL
# ---------------------------------------------------------------------------


def test_resolve_base_url_uses_first_server() -> None:
    spec = {"servers": [{"url": "https://api.example.com/"}]}
    assert _resolve_base_url(spec) == "https://api.example.com"


def test_resolve_base_url_swagger2_fallback() -> None:
    spec = {
        "host": "api.example.com",
        "basePath": "/v1",
        "schemes": ["https"],
    }
    assert _resolve_base_url(spec) == "https://api.example.com/v1"


def test_resolve_base_url_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "AI_ASSISTANT_SERVER_BASE_URL_OVERRIDE",
        "http://localhost:9000/proxy",
    )
    spec = {"servers": [{"url": "https://api.example.com"}]}
    assert _resolve_base_url(spec) == "http://localhost:9000/proxy"


# ---------------------------------------------------------------------------
# Full directory load
# ---------------------------------------------------------------------------


def test_load_yaml_petstore_yields_two_tools(tmp_path: Path) -> None:
    _write_spec(tmp_path, "petstore.yaml", PETSTORE_SPEC)
    tools = load_tools_from_directory(tmp_path)
    by_name = {t.name: t for t in tools}

    assert "getpetbyid" in by_name
    assert "findpetsbystatus" in by_name

    pet = by_name["getpetbyid"]
    assert pet.execution.method == "get"
    assert pet.execution.path == "/pet/{petId}"
    assert pet.execution.parameter_locations["petId"] == "path"
    assert pet.input_schema["properties"]["petId"]["type"] == "integer"
    assert "petId" in pet.input_schema["required"]
    assert pet.execution.base_url == "https://petstore3.swagger.io/api/v3"

    by_status = by_name["findpetsbystatus"]
    assert by_status.execution.parameter_locations["status"] == "query"
    assert "status" not in by_status.input_schema.get("required", [])


def test_load_json_format_works(tmp_path: Path) -> None:
    spec_dict = {
        "openapi": "3.0.0",
        "info": {"title": "x", "version": "1"},
        "servers": [{"url": "https://api.example.com"}],
        "paths": {
            "/ping": {
                "get": {
                    "operationId": "ping",
                    "summary": "Ping",
                    "responses": {"200": {"description": "ok"}},
                }
            }
        },
    }
    (tmp_path / "ping.json").write_text(json.dumps(spec_dict), encoding="utf-8")
    tools = load_tools_from_directory(tmp_path)
    assert [t.name for t in tools] == ["ping"]


def test_unparseable_file_is_skipped_not_fatal(tmp_path: Path) -> None:
    _write_spec(tmp_path, "bad.yaml", "this is :: not :: valid : yaml :")
    _write_spec(tmp_path, "good.yaml", PETSTORE_SPEC)
    tools = load_tools_from_directory(tmp_path)
    # Bad spec was skipped; good one still loaded.
    assert len(tools) == 2


def test_unsupported_extensions_skipped(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("hello", encoding="utf-8")
    _write_spec(tmp_path, "petstore.yaml", PETSTORE_SPEC)
    tools = load_tools_from_directory(tmp_path)
    assert len(tools) == 2  # only the YAML's two operations


def test_duplicate_operation_ids_disambiguate(tmp_path: Path) -> None:
    spec = textwrap.dedent(
        """
        openapi: 3.0.3
        info:
          title: dup
          version: 1.0.0
        servers:
          - url: https://api.example.com
        paths:
          /a:
            get:
              operationId: getThing
              responses:
                "200":
                  description: ok
          /b:
            get:
              operationId: getThing
              responses:
                "200":
                  description: ok
        """
    )
    _write_spec(tmp_path, "dup.yaml", spec)
    tools = load_tools_from_directory(tmp_path)
    names = {t.name for t in tools}
    assert names == {"getthing", "getthing_2"}


def test_missing_directory_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(ValueError):
        load_tools_from_directory(missing)


# ---------------------------------------------------------------------------
# Auth resolution at load time
# ---------------------------------------------------------------------------


def test_bearer_auth_recognized(tmp_path: Path) -> None:
    _write_spec(tmp_path, "bearer.yaml", BEARER_SPEC)
    tools = load_tools_from_directory(tmp_path)
    assert len(tools) == 1
    me = tools[0]
    assert me.auth.scheme is AuthScheme.BEARER
    assert me.auth.scheme_name == "bearerAuth"
    assert me.auth.secret_env == "AI_ASSISTANT_SERVER_AUTH_BEARERAUTH"


def test_api_key_query_recognized(tmp_path: Path) -> None:
    _write_spec(tmp_path, "weather.yaml", API_KEY_QUERY_SPEC)
    tools = load_tools_from_directory(tmp_path)
    assert len(tools) == 1
    fc = tools[0]
    assert fc.auth.scheme is AuthScheme.API_KEY_QUERY
    assert fc.auth.parameter_name == "api_key"


def test_no_security_means_no_auth(tmp_path: Path) -> None:
    _write_spec(tmp_path, "petstore.yaml", PETSTORE_SPEC)
    tools = load_tools_from_directory(tmp_path)
    assert all(t.auth.scheme is AuthScheme.NONE for t in tools)


# ---------------------------------------------------------------------------
# Request body
# ---------------------------------------------------------------------------


def test_request_body_lifted_into_schema(tmp_path: Path) -> None:
    spec = textwrap.dedent(
        """
        openapi: 3.0.3
        info:
          title: x
          version: 1.0.0
        servers:
          - url: https://api.example.com
        paths:
          /widgets:
            post:
              operationId: createWidget
              requestBody:
                required: true
                content:
                  application/json:
                    schema:
                      type: object
                      required: [name]
                      properties:
                        name:
                          type: string
                        size:
                          type: integer
              responses:
                "201":
                  description: ok
        """
    )
    _write_spec(tmp_path, "widgets.yaml", spec)
    tools = load_tools_from_directory(tmp_path)
    assert len(tools) == 1
    create = tools[0]
    assert create.execution.parameter_locations["body"] == "body"
    assert create.execution.request_body_required is True
    assert "body" in create.input_schema["properties"]
    body_schema = create.input_schema["properties"]["body"]
    assert body_schema["type"] == "object"
    assert body_schema["required"] == ["name"]


# ---------------------------------------------------------------------------
# x-aai-spec-url + runtime spec helpers
# ---------------------------------------------------------------------------


def test_extract_spec_url_root_extension() -> None:
    spec = {"x-aai-spec-url": "https://api.example.com/openapi.json"}
    assert _extract_spec_url(spec) == "https://api.example.com/openapi.json"


def test_extract_spec_url_info_extension() -> None:
    spec = {"info": {"x-aai-spec-url": "https://api.example.com/v3/api-docs"}}
    assert _extract_spec_url(spec) == "https://api.example.com/v3/api-docs"


def test_extract_spec_url_absent_returns_none() -> None:
    assert _extract_spec_url({"info": {}}) is None


def test_extract_spec_url_root_wins_over_info() -> None:
    spec = {
        "x-aai-spec-url": "https://root.example/spec",
        "info": {"x-aai-spec-url": "https://info.example/spec"},
    }
    assert _extract_spec_url(spec) == "https://root.example/spec"


def test_extract_spec_url_blank_string_is_ignored() -> None:
    spec = {"x-aai-spec-url": "   "}
    assert _extract_spec_url(spec) is None


def test_extract_spec_url_non_dict_info_is_ignored() -> None:
    # Malformed specs shouldn't crash the loader.
    assert _extract_spec_url({"info": "not-a-dict"}) is None


def test_load_propagates_spec_url(tmp_path: Path) -> None:
    spec_with_url = textwrap.dedent(
        """
        openapi: 3.0.3
        x-aai-spec-url: https://api.example.com/openapi.json
        info:
          title: x
          version: 1.0.0
        servers:
          - url: https://api.example.com
        paths:
          /ping:
            get:
              operationId: ping
              responses:
                "200":
                  description: ok
        """
    )
    _write_spec(tmp_path, "ping.yaml", spec_with_url)
    tools = load_tools_from_directory(tmp_path)
    assert tools[0].spec_url == "https://api.example.com/openapi.json"


def test_spec_hash_is_stable_and_ordering_invariant() -> None:
    a = {"a": 1, "b": [2, 3], "nested": {"x": True, "y": None}}
    b = {"nested": {"y": None, "x": True}, "b": [2, 3], "a": 1}
    assert spec_hash(a) == spec_hash(b)


def test_spec_hash_changes_with_content() -> None:
    a = {"paths": {"/x": {"get": {}}}}
    b = {"paths": {"/y": {"get": {}}}}
    assert spec_hash(a) != spec_hash(b)


def test_tools_from_spec_doc_disambiguates_collisions() -> None:
    spec = {
        "paths": {
            "/a": {"get": {"operationId": "getThing", "responses": {}}},
            "/b": {"get": {"operationId": "getThing", "responses": {}}},
        }
    }
    tools = tools_from_spec_doc(spec, source="live")
    names = sorted(t.name for t in tools)
    assert names == ["getthing", "getthing_2"]


async def test_fetch_spec_from_url_json_content_type() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            text='{"openapi": "3.0.0", "paths": {}}',
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        spec = await fetch_spec_from_url(
            "https://api.example.com/openapi.json", client=client
        )
    assert spec == {"openapi": "3.0.0", "paths": {}}


async def test_fetch_spec_from_url_yaml_fallback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/yaml"},
            text="openapi: 3.0.0\npaths: {}\n",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        spec = await fetch_spec_from_url(
            "https://api.example.com/openapi.yaml", client=client
        )
    assert spec == {"openapi": "3.0.0", "paths": {}}


async def test_fetch_spec_from_url_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="bad day")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await fetch_spec_from_url(
                "https://api.example.com/openapi.json", client=client
            )
