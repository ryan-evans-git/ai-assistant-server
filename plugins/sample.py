"""Sample Python plugin module.

Drop additional ``*.py`` files into this directory (or import a
module via ``--plugin-module pkg.mod``) to register your own
non-API tools.  Every callable decorated with ``@tool`` becomes
an MCP tool the assistant can invoke.

The function's signature drives the JSON Schema sent to the LLM —
parameter names, types, defaults, and Literal/Enum constraints
are all picked up automatically by Pydantic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from ai_assistant_server import tool


@tool(
    name="now",
    description="Return the current UTC timestamp in ISO-8601 format.",
    tags=("time",),
)
def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@tool(
    name="convert_temperature",
    description=(
        "Convert a temperature between Celsius, Fahrenheit, and Kelvin.  "
        "Returns a dict with the converted value and the unit symbol."
    ),
    tags=("math", "units"),
)
def convert_temperature(
    value: float,
    from_unit: Literal["c", "f", "k"],
    to_unit: Literal["c", "f", "k"],
) -> dict[str, float | str]:
    if from_unit == to_unit:
        return {"value": value, "unit": to_unit}
    # Normalize to Celsius first, then to the target.
    celsius = {
        "c": value,
        "f": (value - 32.0) * 5.0 / 9.0,
        "k": value - 273.15,
    }[from_unit]
    converted = {
        "c": celsius,
        "f": celsius * 9.0 / 5.0 + 32.0,
        "k": celsius + 273.15,
    }[to_unit]
    return {"value": round(converted, 4), "unit": to_unit}
