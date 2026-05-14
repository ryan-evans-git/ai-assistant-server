"""Tests for the human-in-the-loop tool metadata pipeline.

Exercises:
- @tool decorator's requires_confirmation/timeout/message kwargs.
- OpenAPI loader's x-aai-* vendor extension parsing (flat + nested).
- _tool_to_mcp's annotations surface.
- _hitl_from_operation's defensive parsing of malformed values.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from ai_assistant_server import tool
from ai_assistant_server.loader import _hitl_from_operation, load_tools_from_directory
from ai_assistant_server.models import (
    AuthConfig,
    HitlConfig,
    HttpExecution,
    OpenApiTool,
)
from ai_assistant_server.plugins import get_plugin_tool
from ai_assistant_server.server import _hitl_annotations, _tool_to_mcp


# ---------------------------------------------------------------------------
# @tool decorator → PluginTool.hitl
# ---------------------------------------------------------------------------


def test_tool_decorator_default_no_hitl() -> None:
    @tool(description="x")
    def plain() -> str:
        return "ok"

    plugin = get_plugin_tool(plain)
    assert plugin is not None
    assert plugin.hitl == HitlConfig()
    assert plugin.hitl.requires_confirmation is False


def test_tool_decorator_passes_hitl_through() -> None:
    @tool(
        description="d",
        requires_confirmation=True,
        confirm_timeout_seconds=120,
        confirm_message="Send?",
    )
    def gated() -> str:
        return "ok"

    plugin = get_plugin_tool(gated)
    assert plugin is not None
    assert plugin.hitl.requires_confirmation is True
    assert plugin.hitl.timeout_seconds == 120
    assert plugin.hitl.confirm_message == "Send?"


# ---------------------------------------------------------------------------
# Loader: x-aai-* extensions on OpenAPI operations
# ---------------------------------------------------------------------------


def _write_spec(tmp: Path, name: str, body: str) -> Path:
    p = tmp / name
    p.write_text(body, encoding="utf-8")
    return p


def test_loader_reads_flat_x_aai_extensions(tmp_path: Path) -> None:
    spec = textwrap.dedent(
        """
        openapi: 3.0.3
        info: {title: x, version: 1}
        servers: [{url: "https://x"}]
        paths:
          /charges:
            post:
              operationId: createCharge
              x-aai-requires-confirmation: true
              x-aai-confirm-timeout-seconds: 45
              x-aai-confirm-message: "Charge card?"
              responses: {"200": {description: ok}}
        """
    )
    _write_spec(tmp_path, "billing.yaml", spec)
    [t] = load_tools_from_directory(tmp_path)
    assert t.hitl.requires_confirmation is True
    assert t.hitl.timeout_seconds == 45
    assert t.hitl.confirm_message == "Charge card?"


def test_loader_reads_nested_x_aai_hitl(tmp_path: Path) -> None:
    """Authors may prefer the grouped form."""
    spec = textwrap.dedent(
        """
        openapi: 3.0.3
        info: {title: x, version: 1}
        servers: [{url: "https://x"}]
        paths:
          /charges:
            post:
              operationId: createCharge
              x-aai-hitl:
                requires_confirmation: true
                timeout_seconds: 60
                confirm_message: "Confirm"
              responses: {"200": {description: ok}}
        """
    )
    _write_spec(tmp_path, "billing.yaml", spec)
    [t] = load_tools_from_directory(tmp_path)
    assert t.hitl.requires_confirmation is True
    assert t.hitl.timeout_seconds == 60


def test_loader_no_extensions_means_default_off(tmp_path: Path) -> None:
    spec = textwrap.dedent(
        """
        openapi: 3.0.3
        info: {title: x, version: 1}
        servers: [{url: "https://x"}]
        paths:
          /ping:
            get:
              operationId: ping
              responses: {"200": {description: ok}}
        """
    )
    _write_spec(tmp_path, "ping.yaml", spec)
    [t] = load_tools_from_directory(tmp_path)
    assert t.hitl == HitlConfig()


def test_hitl_from_operation_handles_malformed_timeout() -> None:
    """A non-numeric timeout should fall back to None, not raise."""
    op = {
        "x-aai-requires-confirmation": True,
        "x-aai-confirm-timeout-seconds": "thirty",
    }
    cfg = _hitl_from_operation(op)
    assert cfg.requires_confirmation is True
    assert cfg.timeout_seconds is None


def test_hitl_from_operation_ignores_non_dict_nested() -> None:
    """``x-aai-hitl`` set to something that isn't a dict shouldn't blow up."""
    op = {"x-aai-hitl": "garbage"}
    cfg = _hitl_from_operation(op)
    assert cfg == HitlConfig()


# ---------------------------------------------------------------------------
# server._tool_to_mcp annotations surface
# ---------------------------------------------------------------------------


def _plain_openapi_tool() -> OpenApiTool:
    return OpenApiTool(
        name="ping",
        description="d",
        input_schema={"type": "object", "properties": {}},
        execution=HttpExecution(base_url="https://x", method="get", path="/"),
        auth=AuthConfig(),
    )


def _hitl_openapi_tool() -> OpenApiTool:
    return OpenApiTool(
        name="charge",
        description="d",
        input_schema={"type": "object", "properties": {}},
        execution=HttpExecution(base_url="https://x", method="post", path="/charges"),
        auth=AuthConfig(),
        hitl=HitlConfig(
            requires_confirmation=True,
            timeout_seconds=45,
            confirm_message="Charge?",
        ),
    )


def test_hitl_annotations_none_for_plain_tool() -> None:
    assert _hitl_annotations(_plain_openapi_tool()) is None


def test_hitl_annotations_payload_for_gated_tool() -> None:
    payload = _hitl_annotations(_hitl_openapi_tool())
    assert payload == {
        "aai": {
            "requires_confirmation": True,
            "timeout_seconds": 45,
            "message": "Charge?",
        }
    }


def test_hitl_annotations_omits_optional_fields() -> None:
    """Tool with confirmation but no timeout/message → minimal payload."""
    tool_def = OpenApiTool(
        name="x",
        description="d",
        input_schema={"type": "object", "properties": {}},
        execution=HttpExecution(base_url="https://x", method="get", path="/"),
        auth=AuthConfig(),
        hitl=HitlConfig(requires_confirmation=True),
    )
    payload = _hitl_annotations(tool_def)
    assert payload == {"aai": {"requires_confirmation": True}}


def test_tool_to_mcp_includes_annotations_for_hitl_tool() -> None:
    mcp_tool = _tool_to_mcp(_hitl_openapi_tool())
    # MCP's ToolAnnotations is a typed model that accepts extra kwargs.
    # The vendor-prefixed `aai` key survives and reaches the wire.
    assert mcp_tool.annotations is not None
    dumped = mcp_tool.model_dump(exclude_none=True)
    assert dumped["annotations"]["aai"]["requires_confirmation"] is True


def test_tool_to_mcp_omits_annotations_for_plain_tool() -> None:
    mcp_tool = _tool_to_mcp(_plain_openapi_tool())
    assert mcp_tool.annotations is None


def test_tool_to_mcp_works_for_plugin_tool_with_hitl() -> None:
    """The annotations pipeline applies to PluginTool too."""

    @tool(description="d", requires_confirmation=True)
    def gated() -> str:
        return "ok"

    plugin = get_plugin_tool(gated)
    assert plugin is not None
    mcp_tool = _tool_to_mcp(plugin)
    dumped = mcp_tool.model_dump(exclude_none=True)
    assert dumped["annotations"]["aai"]["requires_confirmation"] is True


def test_tool_to_mcp_handles_tool_without_hitl_attribute() -> None:
    """Defensive: a hand-rolled ToolDefinition that doesn't carry .hitl
    must not crash _tool_to_mcp."""

    class _BareTool:
        name = "bare"
        description = "d"
        input_schema = {"type": "object", "properties": {}}

    mcp_tool = _tool_to_mcp(_BareTool())  # type: ignore[arg-type]
    assert mcp_tool.annotations is None


# ---------------------------------------------------------------------------
# Sample HITL plugin + spec smoke test (load and verify metadata)
# ---------------------------------------------------------------------------


def test_sample_hitl_plugin_loads_with_confirmation_flag() -> None:
    """Smoke: the example plugin under plugins/sample_hitl.py registers
    with requires_confirmation=True so a developer copying it gets
    HITL out of the box."""
    from ai_assistant_server.loader import load_plugins_from_directory

    plugins_dir = (
        Path(__file__).parent.parent / "plugins"
    )
    plugins = load_plugins_from_directory(plugins_dir)
    by_name = {p.name: p for p in plugins}
    # Sample includes send_email + delete_records + transfer_funds.
    for expected in ("send_email", "delete_records", "transfer_funds"):
        assert expected in by_name, f"sample plugin {expected!r} missing"
        assert by_name[expected].hitl.requires_confirmation is True


def test_sample_hitl_yaml_loads_with_confirmation_flag() -> None:
    """Smoke: the example tools/billing-hitl.yaml has the extension
    parsed into the OpenApiTool's hitl block."""
    tools_dir = Path(__file__).parent.parent / "tools"
    tools = load_tools_from_directory(tools_dir)
    by_name = {t.name: t for t in tools}
    assert "createcharge" in by_name
    assert by_name["createcharge"].hitl.requires_confirmation is True
    assert by_name["createcharge"].hitl.timeout_seconds == 45
