"""ai-assistant-server: generic MCP server backed by OpenAPI specs."""

from ai_assistant_server.loader import load_tools_from_directory
from ai_assistant_server.models import (
    AuthConfig,
    AuthScheme,
    HttpExecution,
    ToolDefinition,
)

__all__ = [
    "AuthConfig",
    "AuthScheme",
    "HttpExecution",
    "ToolDefinition",
    "load_tools_from_directory",
]
__version__ = "0.1.0"
