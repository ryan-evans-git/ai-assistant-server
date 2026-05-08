"""Data models shared by the loader, executor, plugins, and server.

A :class:`ToolDefinition` is the base shape every tool the server
surfaces conforms to: ``name`` + ``description`` + ``input_schema``
(JSON Schema).  Beyond that we have two concrete subclasses:

  * :class:`OpenApiTool` — derived from an OpenAPI/Swagger operation
    and dispatched via an HTTP request.
  * :class:`PluginTool` — a Python callable registered via the
    ``@tool`` decorator (see :mod:`ai_assistant_server.plugins`).

The executor dispatches on type at call time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Union


class AuthScheme(str, Enum):
    """Subset of OpenAPI security schemes we support out of the box.

    Bearer / API key / Basic cover the vast majority of public APIs.
    OAuth2 flows aren't auto-resolvable from a spec alone (they
    require a client-secret + token endpoint dance), so we
    surface them as ``unsupported`` and let the host configure a
    pre-minted bearer manually.
    """

    NONE = "none"
    BEARER = "bearer"
    API_KEY_HEADER = "apiKeyHeader"
    API_KEY_QUERY = "apiKeyQuery"
    BASIC = "basic"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class AuthConfig:
    """Resolved auth requirement for a single OpenAPI-derived tool.

    ``scheme`` is the *kind* of auth.  ``secret_env`` is the
    environment variable the executor will read at call time —
    we never embed secrets in the spec.  For ``API_KEY_*``,
    ``parameter_name`` is the header / query-param the upstream
    expects.
    """

    scheme: AuthScheme = AuthScheme.NONE
    secret_env: str | None = None
    parameter_name: str | None = None
    # The OpenAPI security scheme name (e.g. "bearerAuth") so
    # callers can override per-request via that key.
    scheme_name: str | None = None


@dataclass(frozen=True)
class HttpExecution:
    """How to issue the HTTP request behind an OpenAPI tool call.

    ``base_url`` comes from the spec's first ``servers[]`` entry
    (or an env override).  ``path`` and ``method`` are the
    OpenAPI operation's identity.  ``parameter_locations`` tells
    the executor where each named parameter lives — path / query
    / header / cookie / requestBody.
    """

    base_url: str
    method: str
    path: str
    parameter_locations: dict[str, str] = field(default_factory=dict)
    request_body_required: bool = False
    request_body_property: str | None = None  # "body" by default
    timeout_seconds: float = 30.0


# ---------------------------------------------------------------------------
# Human-in-the-loop config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HitlConfig:
    """Per-tool human-in-the-loop policy.

    Default-off: a tool that doesn't set any of these fields runs
    exactly as before — no confirmation, no pause.  When
    ``requires_confirmation`` is ``True`` the metadata travels to
    the client (via MCP ``Tool.annotations``) and the agent loop
    pauses before dispatch to surface a confirm/decline modal.

    ``confirm_message`` is an optional one-line UI hint shown above
    the JSON-formatted tool input.  ``timeout_seconds`` lets the
    tool author override the client-side default timeout (e.g. a
    long-running review can ask for 5 minutes).
    """

    requires_confirmation: bool = False
    timeout_seconds: int | None = None
    confirm_message: str | None = None


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenApiTool:
    """A tool derived from an OpenAPI operation.

    Carries both the wire surface (``name``/``description``/
    ``input_schema``) and the dispatch metadata the executor needs
    to issue the upstream HTTP request.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    execution: HttpExecution
    auth: AuthConfig = field(default_factory=AuthConfig)
    # Free-form tags from the OpenAPI ``tags`` field — useful
    # for progressive discovery on the client side.
    tags: tuple[str, ...] = field(default_factory=tuple)
    source_spec: str = ""
    hitl: HitlConfig = field(default_factory=HitlConfig)


# A plugin handler is an async- or sync-callable that takes the
# parsed argument dict and returns the result body.  The executor
# awaits awaitable returns; sync callables run inline.
PluginHandler = Callable[..., Union[Any, Awaitable[Any]]]


@dataclass(frozen=True)
class PluginTool:
    """A tool registered via the ``@tool`` decorator from a Python
    callable.  The input schema is derived from the callable's
    signature at registration time."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: PluginHandler
    tags: tuple[str, ...] = field(default_factory=tuple)
    source_module: str = ""
    hitl: HitlConfig = field(default_factory=HitlConfig)


# Public alias so callers don't have to spell out the union every
# time.  Either subclass satisfies "the wire surface MCP needs."
ToolDefinition = Union[OpenApiTool, PluginTool]
