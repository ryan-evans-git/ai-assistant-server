# ai-assistant-server

A spec-driven [Model Context Protocol](https://modelcontextprotocol.io) (MCP)
server. Drop OpenAPI 3.x / Swagger 2.0 spec files into `tools/` and every
operation becomes an MCP tool — names, descriptions, JSON-Schema inputs,
base URLs, and auth requirements are all derived from the spec.
**Or** drop a Python file into `plugins/` with `@tool`-decorated functions
and the server registers those alongside the spec-derived tools.

Built for **multi-tenant** deployments: credentials can be supplied
per-request and forwarded to the upstream API on the caller's behalf,
keyed by the spec's `securitySchemes`. No shared service-account leaks
between tenants. Speaks both stdio and SSE MCP transports.

No code changes to add a new HTTP tool. Just drop in the spec.

```
tools/
  petstore.yaml          ← https://petstore3.swagger.io/api/v3
  github.yaml            ← bearer-auth, env-mapped
  internal-api.yaml      ← your private API

plugins/
  sample.py              ← @tool-decorated Python functions
```

## Why

Most MCP servers couple a server's plumbing to a particular API's shape.
That makes them hard to reuse across projects. This server inverts the
relationship: the **spec** is the source of truth. The Python code is a
thin adapter that knows how to:

1. **Load** OpenAPI 3.x (and Swagger 2.0) documents.
2. **Project** each operation into an MCP `Tool` (name, description,
   `inputSchema`).
3. **Execute** tool calls as upstream HTTP requests, with auth supplied
   from environment variables or per-request forwarded credentials.

Add an API by writing — or downloading — its OpenAPI spec.

## Adding a Python plugin (non-API tools)

Some tools don't have an API behind them — local file access, math,
date computations, an in-process database query, an inference call,
etc. For these, use the `@tool` decorator:

```python
# plugins/my_tools.py
from typing import Literal
from ai_assistant_server import tool


@tool(
    name="convert_temperature",
    description="Convert between Celsius, Fahrenheit, and Kelvin.",
    tags=("math", "units"),
)
def convert_temperature(
    value: float,
    from_unit: Literal["c", "f", "k"],
    to_unit: Literal["c", "f", "k"],
) -> dict:
    ...
```

The JSON Schema MCP needs is **derived from the function's signature**
via Pydantic — every annotation Pydantic understands is supported
(`str` / `int` / `float` / `bool` / `Literal[...]` / `Optional[T]` /
`list[T]` / `dict[K, V]` / `Enum` / Pydantic `BaseModel` subclasses).
Defaults become the schema's `default`; parameters without a default
land in `required`.

Two ways to register your plugins:

| Source | When to use |
|---|---|
| `plugins/*.py` directory (default `./plugins`) | Quick iteration, no packaging required. Each `*.py` file is loaded as a freestanding module; `_*.py` is skipped. |
| `--plugin-module pkg.module` (repeatable) | When your plugins are an installed Python package. Module must be import-resolvable on the server's `PYTHONPATH`. |

Set `AI_ASSISTANT_SERVER_PLUGINS_DIR` or `AI_ASSISTANT_SERVER_PLUGIN_MODULES`
(comma-separated list) to drive these from the environment instead of CLI
flags.

Async handlers (`async def`) are supported — they're awaited; sync
handlers run inline. Handler exceptions are wrapped and returned to
the agent as a tool error, so the assistant can react and retry.

## Human-in-the-loop tools

Mark a tool as needing the user's approval before it runs. The
agent loop on the client side pauses, fires a confirmation modal in
the chat UI, and only dispatches the tool once the user clicks
**Confirm** (or aborts on **Decline** / timeout).

For Python plugins, three new kwargs on `@tool(...)`:

```python
@tool(
    name="send_email",
    description="Send an email on the user's behalf.",
    requires_confirmation=True,
    confirm_message="Send this email?",         # optional one-line UI hint
    confirm_timeout_seconds=60,                 # optional override
)
def send_email(*, to: str, subject: str, body: str) -> dict:
    ...
```

For OpenAPI specs, three vendor extensions on the operation (or a
nested `x-aai-hitl: { ... }` block):

```yaml
paths:
  /charges:
    post:
      operationId: createCharge
      x-aai-requires-confirmation: true
      x-aai-confirm-timeout-seconds: 45
      x-aai-confirm-message: "Charge customer card?"
```

The flag rides through MCP via `Tool.annotations.aai` and is
honored by any HITL-aware client. See `plugins/sample_hitl.py` and
`tools/billing-hitl.yaml` for runnable examples.

## Install

```bash
pip install git+https://github.com/ryan-evans-git/ai-assistant-server.git
```

Or clone and install in editable mode for local development:

```bash
git clone https://github.com/ryan-evans-git/ai-assistant-server.git
cd ai-assistant-server
pip install -e ".[dev]"
```

Python 3.11+ required.

## Quickstart

```bash
# Use the bundled sample specs (Petstore + JSONPlaceholder + GitHub).
ai-assistant-server --tools-dir ./tools

# Or run via Docker:
docker build -t ai-assistant-server .
docker run -p 8765:8765 -v $(pwd)/tools:/app/tools ai-assistant-server
```

By default the server speaks MCP over **stdio** — the right transport for
Claude Desktop, Claude Code, and any subprocess-based MCP host. To expose
it over HTTP+SSE for remote hosts:

```bash
ai-assistant-server --transport sse --port 8765
```

## Adding a new tool

1. Save the API's OpenAPI document as `tools/<name>.yaml` (or `.json`).
2. Restart the server.

That's it. The loader will:

- Read every `paths.{path}.{method}` entry as a tool.
- Use `operationId` as the tool name (or fall back to `{method}_{path}`).
- Combine `summary` and `description` into the tool's description.
- Synthesize a JSON-Schema `inputSchema` from `parameters` + `requestBody`.
- Pick the first server URL from `servers[]` as the base URL.
- Resolve security requirements against `components.securitySchemes`.

## Auth

The server supports the OpenAPI security schemes that can be auto-resolved
from credentials at request time:

| OpenAPI spec | What you set |
|---|---|
| `type: http` + `scheme: bearer` | `AI_ASSISTANT_SERVER_AUTH_<schemeName>=<token>` |
| `type: http` + `scheme: basic` | `AI_ASSISTANT_SERVER_AUTH_<schemeName>=user:pass` (or pre-encoded base64) |
| `type: apiKey` + `in: header` | `AI_ASSISTANT_SERVER_AUTH_<schemeName>=<key>` |
| `type: apiKey` + `in: query` | `AI_ASSISTANT_SERVER_AUTH_<schemeName>=<key>` |

`<schemeName>` is the upper-cased name from `components.securitySchemes`
(e.g. `bearerAuth` → `AI_ASSISTANT_SERVER_AUTH_BEARERAUTH`).

OAuth2 flows are surfaced as **unsupported** — they require a token-endpoint
dance the server can't do from a spec alone. To use an OAuth2-protected
API, mint a bearer token externally and supply it via the bearer scheme.

### Forwarded credentials (per-request)

When the MCP host is acting on behalf of an authenticated end-user, it
can forward credentials per-request rather than baking them into the
server's environment:

```bash
# stdio: env-var prefix is read at startup
X_AI_ASSISTANT_AUTH_bearerAuth=<user-token> ai-assistant-server
```

Forwarded credentials always take priority over `AI_ASSISTANT_SERVER_AUTH_*`.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `AI_ASSISTANT_SERVER_TOOLS_DIR` | `./tools` | Where to scan for specs. |
| `AI_ASSISTANT_SERVER_TRANSPORT` | `stdio` | `stdio` or `sse`. |
| `AI_ASSISTANT_SERVER_HOST` | `127.0.0.1` | SSE bind host. |
| `AI_ASSISTANT_SERVER_PORT` | `8765` | SSE port. |
| `AI_ASSISTANT_SERVER_LOG_LEVEL` | `INFO` | Logger level. |
| `AI_ASSISTANT_SERVER_BASE_URL_OVERRIDE` | _(unset)_ | Force every tool through this URL. Useful for routing through a local proxy. |
| `AI_ASSISTANT_SERVER_AUTH_<NAME>` | _(unset)_ | Per-scheme credential. |

CLI flags override these where applicable; see `ai-assistant-server --help`.

## Tool-naming rules

Resolution order:

1. `operationId`, slugified (lowercase, non-alphanumerics → `_`).
2. `{method}_{path}`, same slugify.

Duplicate names get a `_2`, `_3`, ... suffix in registration order so the
server never silently shadows a tool.

## Project layout

```
ai_assistant_server/
  __init__.py        # Public API exports
  loader.py          # OpenAPI doc → ToolDefinition list
  executor.py        # ToolDefinition + args → upstream HTTP call
  auth.py            # AuthConfig + ambient env → auth headers/query
  models.py          # ToolDefinition / AuthConfig / HttpExecution
  server.py          # MCP server entrypoint (stdio + SSE)

tools/               # Drop your OpenAPI specs here
tests/               # pytest suite
```

The packages above are deliberately small (under ~300 lines each). Every
new feature should fit one of those slots — and if it doesn't, the right
move is usually to extend the loader's `ToolDefinition` rather than adding
a parallel pipeline.

## Companion projects

- [ai-assistant-client](https://github.com/ryan-evans-git/ai-assistant-client)
  — streaming chat client for Claude with progressive tool discovery.
- [ai-assistant-ui](https://github.com/ryan-evans-git/ai-assistant-ui)
  — drop-in React chat panel.

The three together compose into a complete assistant stack: the UI talks
to the client, the client orchestrates Claude + tool calls, and tool calls
land here.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
