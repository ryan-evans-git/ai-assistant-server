"""Auth resolution tests."""

from __future__ import annotations

import base64

import pytest

from ai_assistant_server.auth import AuthResolutionError, resolve_auth
from ai_assistant_server.models import AuthConfig, AuthScheme


def test_none_scheme_yields_empty() -> None:
    applied = resolve_auth(AuthConfig(scheme=AuthScheme.NONE))
    assert applied.headers == {}
    assert applied.query == {}


def test_bearer_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_ASSISTANT_SERVER_AUTH_BEARERAUTH", "tok-abc")
    cfg = AuthConfig(
        scheme=AuthScheme.BEARER,
        secret_env="AI_ASSISTANT_SERVER_AUTH_BEARERAUTH",
        scheme_name="bearerAuth",
    )
    applied = resolve_auth(cfg)
    assert applied.headers == {"Authorization": "Bearer tok-abc"}


def test_bearer_forwarded_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_ASSISTANT_SERVER_AUTH_BEARERAUTH", "tok-from-env")
    cfg = AuthConfig(
        scheme=AuthScheme.BEARER,
        secret_env="AI_ASSISTANT_SERVER_AUTH_BEARERAUTH",
        scheme_name="bearerAuth",
    )
    applied = resolve_auth(cfg, forwarded_credentials={"bearerAuth": "tok-forwarded"})
    assert applied.headers == {"Authorization": "Bearer tok-forwarded"}


def test_basic_pair_encoded() -> None:
    cfg = AuthConfig(scheme=AuthScheme.BASIC, scheme_name="basicAuth")
    applied = resolve_auth(cfg, forwarded_credentials={"basicAuth": "alice:secret"})
    expected = base64.b64encode(b"alice:secret").decode()
    assert applied.headers == {"Authorization": f"Basic {expected}"}


def test_basic_already_encoded_passthrough() -> None:
    cfg = AuthConfig(scheme=AuthScheme.BASIC, scheme_name="basicAuth")
    pre = base64.b64encode(b"alice:secret").decode()
    applied = resolve_auth(cfg, forwarded_credentials={"basicAuth": pre})
    assert applied.headers == {"Authorization": f"Basic {pre}"}


def test_api_key_header() -> None:
    cfg = AuthConfig(
        scheme=AuthScheme.API_KEY_HEADER,
        scheme_name="apiKey",
        parameter_name="X-API-Key",
    )
    applied = resolve_auth(cfg, forwarded_credentials={"apiKey": "abc-123"})
    assert applied.headers == {"X-API-Key": "abc-123"}
    assert applied.query == {}


def test_api_key_query() -> None:
    cfg = AuthConfig(
        scheme=AuthScheme.API_KEY_QUERY,
        scheme_name="apiKey",
        parameter_name="api_key",
    )
    applied = resolve_auth(cfg, forwarded_credentials={"apiKey": "abc-123"})
    assert applied.headers == {}
    assert applied.query == {"api_key": "abc-123"}


def test_missing_secret_raises() -> None:
    cfg = AuthConfig(
        scheme=AuthScheme.BEARER,
        secret_env="DOES_NOT_EXIST_IN_ENV_42",
        scheme_name="bearerAuth",
    )
    with pytest.raises(AuthResolutionError):
        resolve_auth(cfg)


def test_unsupported_scheme_raises() -> None:
    cfg = AuthConfig(scheme=AuthScheme.UNSUPPORTED, scheme_name="oauth2")
    with pytest.raises(AuthResolutionError):
        resolve_auth(cfg)
