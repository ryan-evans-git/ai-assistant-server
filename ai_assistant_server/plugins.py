"""``@tool`` decorator + Pydantic-driven signature introspection.

Lets developers register Python callables as MCP tools alongside the
OpenAPI-derived catalog::

    from ai_assistant_server import tool

    @tool(
        name="add",
        description="Add two integers and return the sum.",
        tags=("math",),
    )
    def add(a: int, b: int) -> int:
        return a + b

The JSON Schema MCP needs is derived from the function's type hints
via Pydantic — ``str``, ``int``, ``float``, ``bool``, ``Literal[...]``,
``Optional[T]``, ``list[T]``, ``dict[K, V]``, ``Enum``, and Pydantic
``BaseModel`` subclasses are all supported out of the box.

Default values become JSON Schema ``default`` entries; parameters
without a default land in the schema's ``required`` array.

The decorator attaches a sentinel attribute to the function; the
:func:`load_plugins_from_module` discovery walks any imported module's
namespace looking for it.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import Field, create_model

from ai_assistant_server.models import PluginTool


# Attribute marker the decorator stamps onto the wrapped function.
# The plugin loader scans module namespaces for this name.
_TOOL_ATTR = "_aai_plugin_tool"


def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    tags: tuple[str, ...] | list[str] = (),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a Python callable so the server registers it as a tool.

    Parameters
    ----------
    name:
        Tool name surfaced to the LLM.  Defaults to the function's
        ``__name__``.
    description:
        Human-readable description.  Defaults to the function's
        docstring (first paragraph), stripped of leading whitespace.
    tags:
        Optional tags for progressive-discovery ranking on the
        client side.
    """

    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or fn.__name__
        doc = description if description is not None else (fn.__doc__ or "")
        tool_desc = doc.strip()
        if not tool_desc:
            raise ValueError(
                f"@tool({tool_name!r}) requires either an explicit description= "
                "or a non-empty docstring."
            )

        input_schema = derive_input_schema(fn)
        plugin = PluginTool(
            name=tool_name,
            description=tool_desc,
            input_schema=input_schema,
            handler=fn,
            tags=tuple(tags),
            source_module=fn.__module__,
        )
        # Attach the resolved PluginTool to the function so the plugin
        # loader can find it without re-introspecting.  The function
        # itself stays directly callable (tests can import + call it).
        setattr(fn, _TOOL_ATTR, plugin)
        return fn

    return wrap


def is_plugin_tool(obj: Any) -> bool:
    """Return ``True`` when ``obj`` was decorated with ``@tool``."""
    return callable(obj) and hasattr(obj, _TOOL_ATTR)


def get_plugin_tool(obj: Any) -> PluginTool | None:
    """Return the :class:`PluginTool` attached to ``obj`` by ``@tool``,
    or ``None`` when the marker isn't present."""
    plugin = getattr(obj, _TOOL_ATTR, None)
    return plugin if isinstance(plugin, PluginTool) else None


# ---------------------------------------------------------------------------
# Schema derivation
# ---------------------------------------------------------------------------


def derive_input_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Build a JSON Schema (``type: object``) from a Python signature.

    Uses Pydantic's runtime type system — every annotation it accepts
    is supported here too.  Parameters without a default are added to
    ``required``.  Returns the schema with Pydantic-internal noise
    stripped (the ``title`` field).
    """
    sig = inspect.signature(fn)
    # Resolve string annotations (Python 3.10+ ``from __future__ import
    # annotations`` defers evaluation by default).  ``include_extras``
    # preserves ``Annotated[...]`` wrappers so Pydantic can read
    # validators / Field metadata authors attach.
    try:
        resolved_hints = get_type_hints(fn, include_extras=True)
    except Exception:  # noqa: BLE001
        # If resolution fails (e.g. forward refs to non-imported names),
        # fall back to whatever the signature object reports — bare
        # types still work even if exotic annotations don't.
        resolved_hints = {}

    field_definitions: dict[str, tuple[Any, Any]] = {}
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            # *args / **kwargs can't be expressed in JSON Schema as
            # named parameters; skip with a clear error.
            raise ValueError(
                f"@tool target {fn.__qualname__!r} cannot use *args / **kwargs; "
                "declare each parameter explicitly so the schema can describe it."
            )

        annotation = resolved_hints.get(
            param_name,
            param.annotation
            if param.annotation is not inspect.Parameter.empty
            else str,
        )
        if param.default is inspect.Parameter.empty:
            field_definitions[param_name] = (annotation, Field(...))
        else:
            field_definitions[param_name] = (annotation, Field(default=param.default))

    if not field_definitions:
        # Pydantic create_model rejects an empty model; emit a static
        # "no inputs" schema directly.
        return {"type": "object", "properties": {}, "additionalProperties": False}

    Model = create_model(f"{fn.__name__}__InputModel", **field_definitions)
    schema = Model.model_json_schema()

    # Pydantic injects a ``title`` field at the top level (the model
    # name).  MCP clients don't need it; strip for cleanliness.
    schema.pop("title", None)
    # Pydantic v2 emits ``additionalProperties`` only when explicitly
    # requested.  Force-deny so the agent can't sneak extra params past
    # the server's argument validation.
    schema.setdefault("additionalProperties", False)
    return schema
