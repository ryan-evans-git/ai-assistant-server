"""Data models shared by the loader, executor, and server.

These are deliberately minimal — they describe what an MCP tool
*derived from* an OpenAPI operation looks like, not the full
OpenAPI surface.  The loader is the only place that knows about
the OpenAPI document structure; downstream code only sees these
types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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
    """Resolved auth requirement for a single tool.

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
    """How to issue the HTTP request behind a tool call.

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


@dataclass(frozen=True)
class ToolDefinition:
    """A single tool surfaced to the MCP client.

    The shape mirrors what MCP wants on the wire: ``name`` +
    ``description`` + ``input_schema`` (JSON Schema).  Beyond
    that we carry an ``execution`` block + ``auth`` block so the
    executor can dispatch the call without re-parsing the spec.
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
