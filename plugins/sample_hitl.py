"""Sample human-in-the-loop plugin tools.

Demonstrates the ``requires_confirmation=True`` flag on the
``@tool`` decorator.  Drop this file into the server's plugins
directory (``./plugins`` by default) and the assistant will pause
to ask the user before executing any of these.

The handlers themselves are stubs — replace the bodies with real
side effects (an SMTP send, a SQL DELETE, a Stripe charge, …)
when wiring against your own backend.
"""

from __future__ import annotations

from typing import Literal

from ai_assistant_server import tool


@tool(
    name="send_email",
    description=(
        "Send an email on the user's behalf.  Pauses for confirmation "
        "before dispatch — the user sees the recipient, subject, and "
        "body in the modal and must approve before send."
    ),
    tags=("email", "communications"),
    requires_confirmation=True,
    confirm_message="Send this email?",
    confirm_timeout_seconds=60,
)
def send_email(*, to: str, subject: str, body: str) -> dict:
    # Stub.  In a real deployment this would call your SMTP / API
    # gateway.  The confirmation gate is enforced by the agent loop
    # *before* this handler runs, so by the time we're here the
    # user has already approved.
    return {"sent": True, "to": to, "subject": subject}


@tool(
    name="delete_records",
    description=(
        "Delete rows from the named table matching the given ids.  "
        "Destructive — pauses for confirmation, defaults to a longer "
        "timeout because the user may need to spot-check the ids."
    ),
    tags=("database", "destructive"),
    requires_confirmation=True,
    confirm_message="About to delete records — confirm?",
    confirm_timeout_seconds=120,
)
def delete_records(*, table: str, ids: list[str]) -> dict:
    # Stub.  Production code would dispatch a DELETE against your
    # database with the supplied ids.
    return {"deleted": len(ids), "table": table}


@tool(
    name="transfer_funds",
    description=(
        "Move funds between two internal accounts.  Pauses for "
        "confirmation; the modal shows the source, destination, "
        "and amount so the user can sanity-check before approving."
    ),
    tags=("payments", "destructive"),
    requires_confirmation=True,
    confirm_message="Confirm transfer?",
    confirm_timeout_seconds=180,
)
def transfer_funds(
    *,
    from_account: str,
    to_account: str,
    amount_cents: int,
    currency: Literal["USD", "EUR", "GBP"] = "USD",
) -> dict:
    # Stub.  Real impl would call your payments backend.
    return {
        "transferred": True,
        "from": from_account,
        "to": to_account,
        "amount_cents": amount_cents,
        "currency": currency,
    }
