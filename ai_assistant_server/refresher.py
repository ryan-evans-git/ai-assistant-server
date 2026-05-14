"""Runtime OpenAPI spec refresher.

When an upstream tool call fails in a way that suggests the on-disk
spec is out of date (a 404 / 410 on a known operation — see
:data:`ai_assistant_server.executor.DRIFT_STATUS_CODES`), the
executor consults a :class:`SpecRefresher` to ask whether a usable
re-fetched spec is available.  When one is, the executor retries
the call exactly once against the updated tool.

Conservative on purpose:

* **Pointer required.**  The tool's spec must declare
  ``x-aai-spec-url`` (root or under ``info``); without it the
  refresher returns ``None`` immediately.
* **Cooldown-gated.**  At most one fetch per spec URL per
  ``min_interval_seconds`` to protect the upstream when a flood
  of failures arrives concurrently.
* **Hash-gated.**  A fetch that returns a byte-equivalent spec
  doesn't mutate the catalog and signals "no retry" — the request
  would just fail the same way.
* **Shape-gated.**  Even when the spec changed, we only signal a
  retry when *this* tool's execution shape (or input schema)
  actually moved.  A description-only diff isn't worth a retry.
* **Never writes to disk.**  The pinned spec file is source of
  truth; the in-memory catalog is patched, and a structured log
  line records the drift event.  Reconciling the file is a
  separate, deliberate operation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from ai_assistant_server.executor import ToolExecutionError
from ai_assistant_server.loader import (
    fetch_spec_from_url,
    spec_hash,
    tools_from_spec_doc,
)
from ai_assistant_server.models import OpenApiTool, ToolDefinition


log = logging.getLogger(__name__)


class SpecRefresher:
    """Owns the catalog-replace + cooldown + hash-guard machinery."""

    def __init__(
        self,
        catalog: dict[str, ToolDefinition],
        *,
        client: httpx.AsyncClient,
        min_interval_seconds: float = 60.0,
    ) -> None:
        self._catalog = catalog
        self._client = client
        self._min_interval = min_interval_seconds
        self._last_hash: dict[str, str] = {}
        self._last_attempt: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def refresh(
        self, tool: OpenApiTool, error: ToolExecutionError
    ) -> OpenApiTool | None:
        """Refresh ``tool``'s spec from its declared URL, if useful.

        Returns the updated catalog entry for ``tool.name`` when the
        live spec changed in a way that affects this tool's
        execution.  Returns ``None`` in every "skip the retry" case:
        no spec URL, cooldown active, fetch failed, spec unchanged,
        or the new tool's shape matches the one we just tried.
        """
        if tool.spec_url is None:
            return None
        url = tool.spec_url
        lock = self._locks.setdefault(url, asyncio.Lock())
        async with lock:
            now = time.monotonic()
            last = self._last_attempt.get(url)
            if last is not None and now - last < self._min_interval:
                return None
            self._last_attempt[url] = now

            try:
                spec = await fetch_spec_from_url(url, client=self._client)
            except Exception as err:  # noqa: BLE001
                log.warning("spec refresh failed for %s: %s", url, err)
                return None

            new_hash = spec_hash(spec)
            if self._last_hash.get(url) == new_hash:
                return None
            self._last_hash[url] = new_hash

            replaced = self._apply_spec(spec, source=url)
            log.info(
                "spec drift detected at %s after %s; updated %d tool(s) "
                "in catalog",
                url,
                error,
                replaced,
            )

            new_tool = self._catalog.get(tool.name)
            if not isinstance(new_tool, OpenApiTool):
                return None
            if (
                new_tool.execution == tool.execution
                and new_tool.input_schema == tool.input_schema
            ):
                # Spec changed elsewhere but this tool's shape is the
                # same — retrying would just fail again.
                return None
            return new_tool

    def _apply_spec(self, spec: dict[str, Any], *, source: str) -> int:
        """Replace catalog entries for tools that came from ``source``.

        We only overwrite OpenAPI-backed entries; plugin tools with
        the same name are left untouched (the original
        first-registration-wins rule from
        :func:`ai_assistant_server.server._enforce_unique_names`
        still applies).  Returns the number of catalog entries
        replaced.
        """
        new_tools = tools_from_spec_doc(spec, source=source)
        count = 0
        for t in new_tools:
            existing = self._catalog.get(t.name)
            if existing is None or isinstance(existing, OpenApiTool):
                self._catalog[t.name] = t
                count += 1
        return count
