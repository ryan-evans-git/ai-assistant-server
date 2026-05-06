"""Resolve auth config + ambient env into HTTP-ready credentials.

Secrets are *only* read at tool-call time, never cached.  The
loader records the env var name the secret should come from
(see :func:`loader._env_for`); we read it here on every call.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass

from ai_assistant_server.models import AuthConfig, AuthScheme


@dataclass(frozen=True)
class AppliedAuth:
    """Resolved auth ready for the HTTP executor to apply."""

    headers: dict[str, str]
    query: dict[str, str]


class AuthResolutionError(RuntimeError):
    """Raised when a tool requires auth we can't supply at call time."""


def resolve_auth(
    auth: AuthConfig,
    *,
    forwarded_credentials: dict[str, str] | None = None,
) -> AppliedAuth:
    """Map an :class:`AuthConfig` + ambient env to headers + query.

    ``forwarded_credentials`` lets the MCP host pass per-request
    credentials keyed by the OpenAPI security-scheme name, e.g.
    ``{"bearerAuth": "<token>"}``.  Forwarded values take
    priority over environment variables — useful when the host
    is acting on behalf of an authenticated end-user.

    Raises :class:`AuthResolutionError` when:
        * The scheme is :data:`AuthScheme.UNSUPPORTED`.
        * No secret is available for a scheme that requires one.
    """
    if auth.scheme is AuthScheme.NONE:
        return AppliedAuth(headers={}, query={})

    if auth.scheme is AuthScheme.UNSUPPORTED:
        raise AuthResolutionError(
            f"OpenAPI security scheme '{auth.scheme_name}' is not supported "
            "(only http-bearer, http-basic, and apiKey are auto-resolved)."
        )

    secret = _read_secret(auth, forwarded_credentials)
    if not secret:
        raise AuthResolutionError(
            f"Missing credential for scheme '{auth.scheme_name}'. "
            f"Set ${auth.secret_env} or forward credentials at call time."
        )

    if auth.scheme is AuthScheme.BEARER:
        return AppliedAuth(headers={"Authorization": f"Bearer {secret}"}, query={})
    if auth.scheme is AuthScheme.BASIC:
        # Spec: secret may be either ``user:pass`` already, or a
        # pre-encoded base64 blob.  Detect by looking for ``:``.
        if ":" in secret:
            encoded = base64.b64encode(secret.encode("utf-8")).decode("ascii")
        else:
            encoded = secret
        return AppliedAuth(headers={"Authorization": f"Basic {encoded}"}, query={})
    if auth.scheme is AuthScheme.API_KEY_HEADER:
        if not auth.parameter_name:
            raise AuthResolutionError(
                f"apiKey scheme '{auth.scheme_name}' missing parameter name"
            )
        return AppliedAuth(headers={auth.parameter_name: secret}, query={})
    if auth.scheme is AuthScheme.API_KEY_QUERY:
        if not auth.parameter_name:
            raise AuthResolutionError(
                f"apiKey scheme '{auth.scheme_name}' missing parameter name"
            )
        return AppliedAuth(headers={}, query={auth.parameter_name: secret})

    raise AuthResolutionError(f"Unhandled auth scheme: {auth.scheme}")


def _read_secret(
    auth: AuthConfig,
    forwarded_credentials: dict[str, str] | None,
) -> str | None:
    if forwarded_credentials and auth.scheme_name:
        forwarded = forwarded_credentials.get(auth.scheme_name)
        if forwarded:
            return forwarded
    if auth.secret_env:
        return os.environ.get(auth.secret_env)
    return None
