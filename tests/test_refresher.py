"""Refresher tests — hash, cooldown, and catalog-update behavior."""

from __future__ import annotations

import textwrap

import httpx
import pytest

from ai_assistant_server.executor import ToolExecutionError
from ai_assistant_server.models import HttpExecution, OpenApiTool, ToolDefinition
from ai_assistant_server.refresher import SpecRefresher


SPEC_V1 = textwrap.dedent(
    """
    openapi: 3.0.3
    x-aai-spec-url: https://api.example.com/openapi.yaml
    info:
      title: Sample
      version: 1.0.0
    servers:
      - url: https://api.example.com
    paths:
      /things/{id}:
        get:
          operationId: getThing
          parameters:
            - name: id
              in: path
              required: true
              schema:
                type: integer
          responses:
            "200":
              description: ok
    """
)


SPEC_V2 = textwrap.dedent(
    """
    openapi: 3.0.3
    x-aai-spec-url: https://api.example.com/openapi.yaml
    info:
      title: Sample
      version: 2.0.0
    servers:
      - url: https://api.example.com
    paths:
      /v2/things/{id}:
        get:
          operationId: getThing
          parameters:
            - name: id
              in: path
              required: true
              schema:
                type: integer
          responses:
            "200":
              description: ok
    """
)


def _starting_tool() -> OpenApiTool:
    return OpenApiTool(
        name="getthing",
        description="x",
        input_schema={
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "required": ["id"],
            "additionalProperties": False,
        },
        execution=HttpExecution(
            base_url="https://api.example.com",
            method="get",
            path="/things/{id}",
            parameter_locations={"id": "path"},
        ),
        spec_url="https://api.example.com/openapi.yaml",
    )


def _make_handler(specs: list[str]) -> object:
    """Return an httpx mock handler that serves each spec in turn."""
    state = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        i = min(state["i"], len(specs) - 1)
        state["i"] += 1
        return httpx.Response(
            200, headers={"content-type": "text/yaml"}, text=specs[i]
        )

    return handler


@pytest.mark.asyncio
async def test_refresh_returns_none_when_tool_has_no_spec_url() -> None:
    catalog: dict[str, ToolDefinition] = {}
    async with httpx.AsyncClient() as client:
        refresher = SpecRefresher(catalog, client=client)
        tool = _starting_tool()
        tool = OpenApiTool(  # strip spec_url
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
            execution=tool.execution,
            spec_url=None,
        )
        result = await refresher.refresh(
            tool, ToolExecutionError("404", status_code=404)
        )
    assert result is None


@pytest.mark.asyncio
async def test_refresh_updates_catalog_when_spec_changes() -> None:
    tool = _starting_tool()
    catalog: dict[str, ToolDefinition] = {tool.name: tool}

    transport = httpx.MockTransport(_make_handler([SPEC_V2]))
    async with httpx.AsyncClient(transport=transport) as client:
        refresher = SpecRefresher(catalog, client=client, min_interval_seconds=0)
        new = await refresher.refresh(
            tool, ToolExecutionError("404", status_code=404)
        )
    assert new is not None
    assert new.execution.path == "/v2/things/{id}"
    assert catalog[tool.name].execution.path == "/v2/things/{id}"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_refresh_returns_none_when_spec_unchanged() -> None:
    """A fetch that yields a byte-equivalent spec is a no-op — the
    request would just fail the same way."""
    tool = _starting_tool()
    catalog: dict[str, ToolDefinition] = {tool.name: tool}

    transport = httpx.MockTransport(_make_handler([SPEC_V1, SPEC_V1]))
    async with httpx.AsyncClient(transport=transport) as client:
        refresher = SpecRefresher(catalog, client=client, min_interval_seconds=0)
        # First call seeds the hash AND replaces the catalog entry
        # (the v1 spec puts the path at ``/things/{id}`` — same as
        # the starting tool — so we expect a shape-equality skip).
        first = await refresher.refresh(
            tool, ToolExecutionError("404", status_code=404)
        )
        # Second call: same spec, hash matches → None without any
        # catalog mutation.
        second = await refresher.refresh(
            tool, ToolExecutionError("404", status_code=404)
        )

    assert first is None  # shape unchanged → no retry signal
    assert second is None  # hash matches → no retry signal


@pytest.mark.asyncio
async def test_refresh_returns_none_within_cooldown() -> None:
    """Two failures in quick succession share one fetch attempt."""
    tool = _starting_tool()
    catalog: dict[str, ToolDefinition] = {tool.name: tool}

    fetch_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal fetch_count
        fetch_count += 1
        return httpx.Response(
            200, headers={"content-type": "text/yaml"}, text=SPEC_V2
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        # min_interval is in seconds; a generous window keeps the
        # second call inside the cooldown deterministically.
        refresher = SpecRefresher(
            catalog, client=client, min_interval_seconds=3600
        )
        first = await refresher.refresh(
            tool, ToolExecutionError("404", status_code=404)
        )
        second = await refresher.refresh(
            tool, ToolExecutionError("404", status_code=404)
        )

    assert first is not None
    assert second is None  # cooldown — no second fetch
    assert fetch_count == 1


@pytest.mark.asyncio
async def test_refresh_returns_none_when_fetch_fails() -> None:
    """Network/HTTP errors on the spec URL are swallowed and reported
    as 'no usable refresh' — we don't blow up the original call."""
    tool = _starting_tool()
    catalog: dict[str, ToolDefinition] = {tool.name: tool}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="bad day")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        refresher = SpecRefresher(catalog, client=client, min_interval_seconds=0)
        result = await refresher.refresh(
            tool, ToolExecutionError("404", status_code=404)
        )
    assert result is None


@pytest.mark.asyncio
async def test_refresh_returns_none_when_shape_unchanged() -> None:
    """If the live spec changed but THIS tool's path/method/schema
    didn't, retrying would just fail again — skip it."""
    tool = _starting_tool()
    catalog: dict[str, ToolDefinition] = {tool.name: tool}

    # v1 spec: same path as the starting tool but with an updated
    # info.title — hashes differ, but tool shape is identical.
    shape_only_diff = SPEC_V1.replace("Sample", "Sample Renamed")
    transport = httpx.MockTransport(_make_handler([shape_only_diff]))
    async with httpx.AsyncClient(transport=transport) as client:
        refresher = SpecRefresher(catalog, client=client, min_interval_seconds=0)
        result = await refresher.refresh(
            tool, ToolExecutionError("404", status_code=404)
        )
    assert result is None
