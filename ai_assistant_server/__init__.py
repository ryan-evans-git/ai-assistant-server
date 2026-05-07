"""ai-assistant-server: spec-driven MCP server with Python plugin support."""

from ai_assistant_server.loader import (
    load_plugins_from_directory,
    load_plugins_from_module,
    load_tools_from_directory,
)
from ai_assistant_server.models import (
    AuthConfig,
    AuthScheme,
    HttpExecution,
    OpenApiTool,
    PluginTool,
    ToolDefinition,
)
from ai_assistant_server.plugins import tool

__all__ = [
    "AuthConfig",
    "AuthScheme",
    "HttpExecution",
    "OpenApiTool",
    "PluginTool",
    "ToolDefinition",
    "load_plugins_from_directory",
    "load_plugins_from_module",
    "load_tools_from_directory",
    "tool",
]
__version__ = "0.1.0"
