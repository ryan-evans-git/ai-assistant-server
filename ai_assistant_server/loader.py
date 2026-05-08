"""OpenAPI spec → tool definition loader.

Walks a directory of .yaml / .yml / .json files, parses each as
an OpenAPI document, and emits one :class:`OpenApiTool` per
HTTP operation.

Supports OpenAPI 3.0.x and 3.1.x.  Swagger 2.0 documents are
auto-converted at load time (we read the major fields directly —
no external converter dependency).

Tool naming
-----------
Preference order for the MCP tool name:

1. ``operationId`` (snake_cased if it isn't already).
2. ``{method}_{path}`` with non-alphanumerics collapsed to ``_``.

If two operations resolve to the same name we suffix ``_2``,
``_3``, etc.

Auth resolution
---------------
Each operation's ``security`` requirement (or the document's
top-level default) is matched against ``components.securitySchemes``
to pick the first scheme we know how to execute.  See
:class:`AuthScheme` for the supported set.  The actual secret
value is *never* read from the spec — the executor reads it
from the env var the user maps via ``AI_ASSISTANT_SERVER_AUTH_*``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

from ai_assistant_server.models import (
    AuthConfig,
    AuthScheme,
    HitlConfig,
    HttpExecution,
    OpenApiTool,
    PluginTool,
)


log = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS = (".yaml", ".yml", ".json")
HTTP_METHODS = ("get", "put", "post", "delete", "options", "head", "patch", "trace")


def load_tools_from_directory(directory: str | Path) -> list[OpenApiTool]:
    """Load every spec under ``directory`` and return all derived tools.

    Files with unsupported extensions are skipped silently.
    Files that fail to parse are logged at WARNING and skipped —
    one bad spec shouldn't take down the rest of the catalog.
    """
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"tools directory does not exist: {root}")

    tools: list[OpenApiTool] = []
    seen_names: set[str] = set()

    for spec_path in sorted(root.iterdir()):
        if spec_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            spec = _read_spec(spec_path)
        except Exception as err:  # noqa: BLE001
            log.warning("Skipping %s: %s", spec_path.name, err)
            continue
        try:
            for tool in _tools_from_spec(spec, source=spec_path.name):
                tool = _disambiguate_name(tool, seen_names)
                tools.append(tool)
                seen_names.add(tool.name)
        except Exception as err:  # noqa: BLE001
            log.warning("Failed to parse %s: %s", spec_path.name, err)
    log.info("Loaded %d tool(s) from %s", len(tools), root)
    return tools


# ---------------------------------------------------------------------------
# Spec I/O
# ---------------------------------------------------------------------------


def _read_spec(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


# ---------------------------------------------------------------------------
# Spec → tools
# ---------------------------------------------------------------------------


def _tools_from_spec(
    spec: dict[str, Any], *, source: str
) -> Iterable[OpenApiTool]:
    base_url = _resolve_base_url(spec)
    security_schemes = _security_schemes(spec)
    document_security = spec.get("security", []) or []

    paths = spec.get("paths") or {}
    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue
        # Path-level params apply to every operation under this path.
        path_level_parameters = path_item.get("parameters", []) or []
        for method, operation in path_item.items():
            method_lower = method.lower()
            if method_lower not in HTTP_METHODS:
                continue
            if not isinstance(operation, dict):
                continue
            yield _build_tool(
                method=method_lower,
                path=path,
                operation=operation,
                path_level_parameters=path_level_parameters,
                base_url=base_url,
                security_schemes=security_schemes,
                document_security=document_security,
                source=source,
            )


def _build_tool(
    *,
    method: str,
    path: str,
    operation: dict[str, Any],
    path_level_parameters: list[Any],
    base_url: str,
    security_schemes: dict[str, dict[str, Any]],
    document_security: list[Any],
    source: str,
) -> OpenApiTool:
    name = _tool_name(operation, method, path)
    description = _tool_description(operation)
    parameters = list(path_level_parameters) + list(
        operation.get("parameters", []) or []
    )
    request_body = operation.get("requestBody")
    input_schema, locations, body_required = _build_input_schema(
        parameters, request_body
    )
    operation_security = operation.get("security", document_security)
    auth = _resolve_auth(operation_security, security_schemes)

    execution = HttpExecution(
        base_url=base_url,
        method=method,
        path=path,
        parameter_locations=locations,
        request_body_required=body_required,
        request_body_property="body" if request_body else None,
    )
    tags_raw = operation.get("tags") or []
    tags: tuple[str, ...] = tuple(str(t) for t in tags_raw if isinstance(t, str))
    hitl = _hitl_from_operation(operation)

    return OpenApiTool(
        name=name,
        description=description,
        input_schema=input_schema,
        execution=execution,
        auth=auth,
        tags=tags,
        source_spec=source,
        hitl=hitl,
    )


def _hitl_from_operation(operation: dict[str, Any]) -> HitlConfig:
    """Read ``x-aai-*`` HITL vendor extensions off an OpenAPI op.

    Two surface forms supported:

    * Flat: ``x-aai-requires-confirmation: true`` /
      ``x-aai-confirm-timeout-seconds: 30`` /
      ``x-aai-confirm-message: "Send email?"``
    * Nested: ``x-aai-hitl: {requires_confirmation: true, ...}`` for
      authors who prefer to group the keys.

    Unrecognized values fall back to defaults; we never raise from
    a malformed extension — the OpenAPI spec is otherwise valid.
    """
    nested = operation.get("x-aai-hitl") or {}
    if not isinstance(nested, dict):
        nested = {}

    def _flag(flat_key: str, nested_key: str, default: Any) -> Any:
        if flat_key in operation:
            return operation[flat_key]
        if nested_key in nested:
            return nested[nested_key]
        return default

    requires = bool(
        _flag("x-aai-requires-confirmation", "requires_confirmation", False)
    )
    timeout = _flag("x-aai-confirm-timeout-seconds", "timeout_seconds", None)
    message = _flag("x-aai-confirm-message", "confirm_message", None)

    timeout_int: int | None
    try:
        timeout_int = int(timeout) if timeout is not None else None
    except (TypeError, ValueError):
        timeout_int = None
    message_str = str(message) if message is not None else None

    return HitlConfig(
        requires_confirmation=requires,
        timeout_seconds=timeout_int,
        confirm_message=message_str,
    )


# ---------------------------------------------------------------------------
# Naming / description
# ---------------------------------------------------------------------------


_NON_ALPHANUM = re.compile(r"[^a-zA-Z0-9]+")


def _tool_name(operation: dict[str, Any], method: str, path: str) -> str:
    raw = operation.get("operationId")
    if isinstance(raw, str) and raw.strip():
        return _slugify(raw)
    composed = f"{method}_{path}"
    return _slugify(composed)


def _slugify(value: str) -> str:
    cleaned = _NON_ALPHANUM.sub("_", value).strip("_")
    if not cleaned:
        cleaned = "tool"
    # MCP tool names should start with a letter for parity with
    # most validators that downstream clients run.
    if not cleaned[0].isalpha():
        cleaned = f"op_{cleaned}"
    return cleaned.lower()


def _disambiguate_name(tool: OpenApiTool, seen: set[str]) -> OpenApiTool:
    if tool.name not in seen:
        return tool
    n = 2
    while f"{tool.name}_{n}" in seen:
        n += 1
    new_name = f"{tool.name}_{n}"
    return OpenApiTool(
        name=new_name,
        description=tool.description,
        input_schema=tool.input_schema,
        execution=tool.execution,
        auth=tool.auth,
        tags=tool.tags,
        source_spec=tool.source_spec,
    )


def _tool_description(operation: dict[str, Any]) -> str:
    summary = operation.get("summary") or ""
    description = operation.get("description") or ""
    if summary and description and summary.strip() != description.strip():
        return f"{summary.strip()}\n\n{description.strip()}"
    return (description or summary or "").strip() or "(no description)"


# ---------------------------------------------------------------------------
# Input schema synthesis
# ---------------------------------------------------------------------------


def _build_input_schema(
    parameters: list[Any], request_body: dict[str, Any] | None
) -> tuple[dict[str, Any], dict[str, str], bool]:
    """Combine OpenAPI parameters + requestBody into one JSON Schema.

    Returns (schema, parameter_locations, body_required).

    ``parameter_locations`` maps each top-level property name to
    where it goes in the HTTP request (``"path"`` / ``"query"`` /
    ``"header"`` / ``"cookie"`` / ``"body"``).  The MCP tool
    consumer doesn't need to know — but the executor does.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []
    locations: dict[str, str] = {}

    for raw_param in parameters:
        if not isinstance(raw_param, dict):
            continue
        pname = raw_param.get("name")
        loc = raw_param.get("in")
        if not pname or not loc:
            continue
        schema = raw_param.get("schema") or {"type": "string"}
        properties[pname] = _annotate_schema(schema, raw_param.get("description"))
        locations[pname] = loc
        if raw_param.get("required") or loc == "path":
            required.append(pname)

    body_required = False
    if request_body and isinstance(request_body, dict):
        body_required = bool(request_body.get("required"))
        body_schema = _request_body_schema(request_body)
        if body_schema is not None:
            properties["body"] = body_schema
            locations["body"] = "body"
            if body_required:
                required.append("body")

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = sorted(set(required))
    schema["additionalProperties"] = False
    return schema, locations, body_required


def _annotate_schema(schema: dict[str, Any], description: str | None) -> dict[str, Any]:
    if not description:
        return schema
    if "description" in schema:
        return schema
    return {**schema, "description": description}


def _request_body_schema(request_body: dict[str, Any]) -> dict[str, Any] | None:
    content = request_body.get("content") or {}
    # Prefer JSON.  Fall back to the first content type we find.
    if "application/json" in content:
        media = content["application/json"]
    elif content:
        media = next(iter(content.values()))
    else:
        return None
    schema = media.get("schema")
    if not isinstance(schema, dict):
        return None
    desc = request_body.get("description")
    return _annotate_schema(schema, desc)


# ---------------------------------------------------------------------------
# Servers / base URL
# ---------------------------------------------------------------------------


def _resolve_base_url(spec: dict[str, Any]) -> str:
    """Pick the upstream base URL for tools in this spec.

    Resolution order:
        1. ``AI_ASSISTANT_SERVER_BASE_URL_OVERRIDE`` env (single
           override for *all* tools — useful for routing through
           a local proxy).
        2. The first non-empty entry in ``servers[]``.
        3. Swagger 2.0 ``host`` + ``basePath`` + ``schemes``.
        4. Empty string — the executor will then require the host
           to provide it via its own config.
    """
    override = os.environ.get("AI_ASSISTANT_SERVER_BASE_URL_OVERRIDE")
    if override:
        return override.rstrip("/")

    servers = spec.get("servers")
    if isinstance(servers, list):
        for entry in servers:
            if isinstance(entry, dict):
                url = entry.get("url")
                if isinstance(url, str) and url.strip():
                    return url.rstrip("/")

    # Swagger 2.0 fallback.
    host = spec.get("host")
    if isinstance(host, str) and host:
        scheme = "https"
        schemes = spec.get("schemes")
        if isinstance(schemes, list) and schemes:
            scheme = str(schemes[0])
        base_path = str(spec.get("basePath") or "").rstrip("/")
        return f"{scheme}://{host}{base_path}".rstrip("/")
    return ""


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


def _security_schemes(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    components = spec.get("components") or {}
    schemes = components.get("securitySchemes") or {}
    if isinstance(schemes, dict):
        return schemes
    # Swagger 2.0 puts these under top-level ``securityDefinitions``.
    legacy = spec.get("securityDefinitions") or {}
    return legacy if isinstance(legacy, dict) else {}


def _resolve_auth(
    security: list[Any] | None,
    schemes: dict[str, dict[str, Any]],
) -> AuthConfig:
    """Pick the first OpenAPI security requirement we can execute.

    ``security`` is a list of requirements where each entry maps
    scheme name → list of scopes (we ignore scopes — those are
    OAuth2-specific).  An empty list means "no auth."
    """
    if not security:
        return AuthConfig()

    for requirement in security:
        if not isinstance(requirement, dict):
            continue
        # Each requirement entry can list multiple AND'ed schemes.
        # We take the first one we know how to handle.
        for scheme_name in requirement.keys():
            scheme = schemes.get(scheme_name)
            if not isinstance(scheme, dict):
                continue
            resolved = _scheme_to_auth_config(scheme_name, scheme)
            if resolved.scheme is not AuthScheme.UNSUPPORTED:
                return resolved
    # No supported scheme matched — surface as unsupported so the
    # executor can fail loudly instead of issuing an unauth'd call.
    return AuthConfig(scheme=AuthScheme.UNSUPPORTED)


def _scheme_to_auth_config(
    scheme_name: str, scheme: dict[str, Any]
) -> AuthConfig:
    type_ = (scheme.get("type") or "").lower()
    if type_ == "http":
        sub = (scheme.get("scheme") or "").lower()
        if sub == "bearer":
            return AuthConfig(
                scheme=AuthScheme.BEARER,
                secret_env=_env_for(scheme_name),
                scheme_name=scheme_name,
            )
        if sub == "basic":
            return AuthConfig(
                scheme=AuthScheme.BASIC,
                secret_env=_env_for(scheme_name),
                scheme_name=scheme_name,
            )
    if type_ == "apikey":
        location = (scheme.get("in") or "").lower()
        if location == "header":
            return AuthConfig(
                scheme=AuthScheme.API_KEY_HEADER,
                secret_env=_env_for(scheme_name),
                parameter_name=scheme.get("name"),
                scheme_name=scheme_name,
            )
        if location == "query":
            return AuthConfig(
                scheme=AuthScheme.API_KEY_QUERY,
                secret_env=_env_for(scheme_name),
                parameter_name=scheme.get("name"),
                scheme_name=scheme_name,
            )
    return AuthConfig(scheme=AuthScheme.UNSUPPORTED, scheme_name=scheme_name)


def _env_for(scheme_name: str) -> str:
    """Conventional env-var name for a security scheme.

    ``bearerAuth`` → ``AI_ASSISTANT_SERVER_AUTH_BEARERAUTH``.
    Hosts can override per scheme with this variable.
    """
    cleaned = re.sub(r"[^A-Z0-9]", "_", scheme_name.upper())
    return f"AI_ASSISTANT_SERVER_AUTH_{cleaned}"


# ---------------------------------------------------------------------------
# Python plugin loading
# ---------------------------------------------------------------------------


def load_plugins_from_module(module_path: str) -> list[PluginTool]:
    """Import ``module_path`` (dotted form, e.g. ``my_pkg.tools``) and
    return every :class:`PluginTool` registered in it via the ``@tool``
    decorator.

    Raises :class:`ImportError` propagated from ``importlib`` when the
    target module can't be imported — the server's startup logs that
    and exits with a clear error rather than silently skipping.
    """
    import importlib

    from ai_assistant_server.plugins import get_plugin_tool

    module = importlib.import_module(module_path)
    plugins: list[PluginTool] = []
    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        plugin = get_plugin_tool(obj)
        if plugin is not None:
            plugins.append(plugin)
    log.info(
        "Loaded %d plugin tool(s) from module %s", len(plugins), module_path
    )
    return plugins


def load_plugins_from_directory(directory: str | Path) -> list[PluginTool]:
    """Import every ``*.py`` file under ``directory`` and return all
    decorated :class:`PluginTool` instances found across them.

    Files starting with ``_`` are skipped (so ``__init__.py`` and
    ``_helpers.py`` style modules don't pull in side-effect imports
    twice).  Each file is loaded as a freestanding module — the
    directory does *not* need an ``__init__.py``.  Tests that need
    importable plugin modules should still use
    :func:`load_plugins_from_module`.
    """
    import importlib.util
    import sys

    from ai_assistant_server.plugins import get_plugin_tool

    root = Path(directory)
    if not root.is_dir():
        return []

    plugins: list[PluginTool] = []
    for path in sorted(root.glob("*.py")):
        if path.name.startswith("_"):
            continue
        # Use a synthetic module name namespaced under the directory
        # so multiple plugin dirs don't collide in sys.modules.
        mod_name = f"_aai_plugin_{root.name}_{path.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            log.warning("Could not load plugin file %s", path)
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as err:  # noqa: BLE001
            log.warning("Skipping plugin file %s: %s", path.name, err)
            del sys.modules[mod_name]
            continue
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            plugin = get_plugin_tool(obj)
            if plugin is not None:
                plugins.append(plugin)
    log.info("Loaded %d plugin tool(s) from %s", len(plugins), root)
    return plugins
