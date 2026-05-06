"""Loader tests — OpenAPI specs in, ToolDefinitions out."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from ai_assistant_server.loader import (
    _resolve_base_url,
    _slugify,
    _tool_name,
    load_tools_from_directory,
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
