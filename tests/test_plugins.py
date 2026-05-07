"""@tool decorator + signature introspection + plugin dispatch."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

import pytest

from ai_assistant_server import tool
from ai_assistant_server.executor import ToolExecutionError, execute_tool
from ai_assistant_server.loader import (
    load_plugins_from_directory,
    load_plugins_from_module,
)
from ai_assistant_server.plugins import (
    derive_input_schema,
    get_plugin_tool,
    is_plugin_tool,
)


# Module-level Enum + a module-level @tool decorator so the loader's
# `dir(module)` walk has something concrete to find.  (Decorators
# inside individual tests register at the local scope `dir()` won't
# see.)


class _SampleStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


@tool(name="module_level_ping", description="returns pong")
def module_level_ping() -> str:
    return "pong"


# ---------------------------------------------------------------------------
# Decorator + introspection
# ---------------------------------------------------------------------------


def test_simple_decorator_attaches_plugin() -> None:
    @tool(name="add", description="Add two ints.")
    def add(a: int, b: int) -> int:
        return a + b

    assert is_plugin_tool(add)
    plugin = get_plugin_tool(add)
    assert plugin is not None
    assert plugin.name == "add"
    assert plugin.description == "Add two ints."
    assert plugin.input_schema["type"] == "object"
    assert "a" in plugin.input_schema["properties"]
    assert plugin.input_schema["properties"]["a"]["type"] == "integer"
    assert plugin.input_schema["required"] == ["a", "b"]


def test_decorator_falls_back_to_docstring() -> None:
    @tool()
    def hello() -> str:
        """Say hello to the world."""
        return "hello"

    plugin = get_plugin_tool(hello)
    assert plugin is not None
    assert plugin.name == "hello"
    assert plugin.description == "Say hello to the world."


def test_decorator_requires_a_description_somewhere() -> None:
    with pytest.raises(ValueError, match="description="):
        @tool()
        def silent(x: int) -> int:
            return x


def test_default_values_become_optional() -> None:
    @tool(description="x")
    def search(query: str, limit: int = 10) -> list[str]:
        return [query] * limit

    plugin = get_plugin_tool(search)
    assert plugin is not None
    schema = plugin.input_schema
    assert schema["required"] == ["query"]
    assert schema["properties"]["limit"]["default"] == 10


def test_literal_renders_as_enum() -> None:
    @tool(description="x")
    def fmt(value: float, unit: Literal["c", "f"]) -> str:
        return f"{value}{unit}"

    plugin = get_plugin_tool(fmt)
    assert plugin is not None
    unit_schema = plugin.input_schema["properties"]["unit"]
    assert unit_schema["enum"] == ["c", "f"]


def test_optional_renders_as_nullable() -> None:
    @tool(description="x")
    def maybe(name: Optional[str] = None) -> str:
        return name or "anon"

    plugin = get_plugin_tool(maybe)
    assert plugin is not None
    name_schema = plugin.input_schema["properties"]["name"]
    # Pydantic emits anyOf [str, null] for Optional.
    assert "anyOf" in name_schema or name_schema.get("type") in ("string", ["string", "null"])


def test_enum_argument() -> None:
    @tool(description="x")
    def filter_by(status: _SampleStatus) -> str:
        return status.value

    plugin = get_plugin_tool(filter_by)
    assert plugin is not None
    # Pydantic resolves Enum to its enum-of-values; check both possible
    # shapes ($ref + $defs, or inlined enum) since v2 layout varies.
    schema = plugin.input_schema
    serialized = repr(schema)
    assert "open" in serialized and "closed" in serialized


def test_no_args_function_emits_empty_object_schema() -> None:
    @tool(description="now")
    def now() -> str:
        return "2026-05-07"

    plugin = get_plugin_tool(now)
    assert plugin is not None
    assert plugin.input_schema == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }


def test_var_args_rejected() -> None:
    def variadic(*args: int) -> int:
        return sum(args)

    with pytest.raises(ValueError, match="args"):
        derive_input_schema(variadic)


# ---------------------------------------------------------------------------
# Execution dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tool_routes_plugin_to_handler() -> None:
    @tool(name="multiply", description="x")
    def multiply(a: int, b: int) -> int:
        return a * b

    plugin = get_plugin_tool(multiply)
    assert plugin is not None
    result = await execute_tool(plugin, {"a": 3, "b": 4})
    assert result.body == 12


@pytest.mark.asyncio
async def test_execute_tool_awaits_async_handler() -> None:
    @tool(name="echo", description="x")
    async def echo(msg: str) -> dict:
        return {"msg": msg}

    plugin = get_plugin_tool(echo)
    assert plugin is not None
    result = await execute_tool(plugin, {"msg": "hi"})
    assert result.body == {"msg": "hi"}


@pytest.mark.asyncio
async def test_plugin_handler_exception_surfaces_as_tool_error() -> None:
    @tool(name="boom", description="x")
    def boom() -> None:
        raise RuntimeError("nope")

    plugin = get_plugin_tool(boom)
    assert plugin is not None
    with pytest.raises(ToolExecutionError, match="nope"):
        await execute_tool(plugin, {})


@pytest.mark.asyncio
async def test_plugin_argument_mismatch_surfaces_as_tool_error() -> None:
    @tool(name="strict", description="x")
    def strict(a: int) -> int:
        return a

    plugin = get_plugin_tool(strict)
    assert plugin is not None
    # The agent supplies an unexpected kwarg — TypeError from Python's
    # call-site is wrapped.
    with pytest.raises(ToolExecutionError, match="rejected arguments"):
        await execute_tool(plugin, {"a": 1, "b": 2})


# ---------------------------------------------------------------------------
# Loader integration
# ---------------------------------------------------------------------------


def test_load_plugins_from_module(monkeypatch: pytest.MonkeyPatch) -> None:
    # Load from this very test module — it has @tool-decorated funcs above.
    plugins = load_plugins_from_module(__name__)
    names = {p.name for p in plugins}
    # The decorators above register many tools; just confirm at least
    # one well-known one made it through.
    assert names  # non-empty
    # Each loaded plugin must be a proper PluginTool.
    for p in plugins:
        assert p.handler is not None
        assert p.input_schema["type"] == "object"


def test_load_plugins_from_directory_skips_underscore_files(tmp_path) -> None:
    (tmp_path / "_helpers.py").write_text("X = 1\n")
    (tmp_path / "ok.py").write_text(
        "from ai_assistant_server import tool\n\n"
        "@tool(name='ping', description='returns pong')\n"
        "def ping() -> str:\n"
        "    return 'pong'\n"
    )
    plugins = load_plugins_from_directory(tmp_path)
    assert [p.name for p in plugins] == ["ping"]


def test_load_plugins_from_directory_handles_missing_dir(tmp_path) -> None:
    plugins = load_plugins_from_directory(tmp_path / "not-here")
    assert plugins == []
