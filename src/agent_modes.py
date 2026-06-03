# src/agent_modes.py
"""Agent execution modes + per-action approval gate.

Three modes, mirroring Claude Code's behaviour but adapted to this app's
always-autonomous agent loop:

  "agent"        — full autonomy. Every tool runs immediately (legacy behaviour).
  "accept_edits" — read-only and local-edit tools run immediately; DESTRUCTIVE
                   tools (shell/python, send/delete email, external API calls…)
                   pause and ask the user to approve or deny.
  "plan"         — read-only tools run; ANY mutating tool is blocked (not
                   executed) and the agent is told to propose a plan instead.

The approval gate is a small registry of asyncio Futures keyed by an opaque
id. The streaming agent loop emits a ``tool_approval_request`` SSE event and
awaits the matching Future; a separate HTTP route resolves it when the user
clicks Approve/Deny. Non-interactive callers (background monitor, scheduler,
teacher, skills) never pass a mode, so they default to "agent" and never gate.
"""

import asyncio
import logging
import uuid
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# ── Modes ──
MODE_AGENT = "agent"
MODE_ACCEPT_EDITS = "accept_edits"
MODE_PLAN = "plan"
VALID_MODES = {MODE_AGENT, MODE_ACCEPT_EDITS, MODE_PLAN}


def normalize_mode(mode: str) -> str:
    """Coerce arbitrary input to a valid mode; unknown -> 'agent' (safe default)."""
    m = (mode or "").strip().lower()
    return m if m in VALID_MODES else MODE_AGENT


# ── Risk classification ──
READ_ONLY = "read_only"
EDIT = "edit"
DESTRUCTIVE = "destructive"

# Code-execution blocks (and their fenced-language aliases) are always
# DESTRUCTIVE — they can do anything the host can.
_CODE_ALIASES = {
    "bash", "sh", "shell", "zsh", "conda", "run",
    "python", "python3", "py",
    "js", "javascript", "node", "ts", "typescript",
}

# Tools that only read state — never gated in any mode.
_READ_ONLY_TOOLS = {
    "web_search", "read_file", "list_emails", "read_email",
    "list_email_accounts", "resolve_contact", "list_served_models",
    "list_sessions", "suggest_document",
}

# Tools that reach outside the app or are hard to undo — gated in plan AND
# accept_edits.
_DESTRUCTIVE_TOOLS = {
    "send_email", "delete_email", "bulk_email", "reply_to_email",
    "stop_served_model", "api_call",
}


def classify_tool(tool_type: str) -> str:
    """Bucket a tool_type into READ_ONLY / EDIT / DESTRUCTIVE.

    Unknown tools default to EDIT — gated in plan mode, auto-run in
    accept_edits, never blocked in agent mode. That's the conservative
    middle: a new tool won't silently run in plan mode, but also won't
    nag with an approval prompt for routine edits.
    """
    t = (tool_type or "").strip().lower()
    if t in _CODE_ALIASES:
        return DESTRUCTIVE
    if t in _READ_ONLY_TOOLS:
        return READ_ONLY
    if t in _DESTRUCTIVE_TOOLS:
        return DESTRUCTIVE
    return EDIT


def gate_decision(mode: str, tool_type: str) -> str:
    """Decide what to do with a tool call under a given mode.

    Returns one of:
      "run"     — execute immediately
      "approve" — pause and ask the user (accept_edits + destructive)
      "block"   — do not execute (plan mode + any mutation)
    """
    mode = normalize_mode(mode)
    risk = classify_tool(tool_type)

    if mode == MODE_AGENT:
        return "run"
    if risk == READ_ONLY:
        return "run"
    if mode == MODE_PLAN:
        return "block"
    # accept_edits:
    if risk == EDIT:
        return "run"
    return "approve"  # destructive


# Injected as an extra system message when mode == plan.
PLAN_MODE_PROMPT = (
    "PLAN MODE IS ACTIVE. You are in read-only planning mode. You may use "
    "read-only tools (web_search, read_file, listing/reading email, etc.) to "
    "investigate, but you must NOT create, edit, delete, send, run, or "
    "otherwise change anything — those actions are blocked and will return a "
    "notice instead of running. Your job this turn is to produce a clear, "
    "concrete, step-by-step plan of what you WOULD do, then STOP and let the "
    "user review it. Do not attempt mutating actions to 'just do it'. If the "
    "request is trivial, a one-line plan is fine."
)


# ── Approval registry ──
# approval_id -> Future[str]  (resolved with "approve" or "deny")
_pending: Dict[str, "asyncio.Future[str]"] = {}

# How long the agent waits on a human before auto-denying (seconds).
APPROVAL_TIMEOUT = 300.0


def create_approval() -> Tuple[str, "asyncio.Future[str]"]:
    """Create a pending-approval Future and return (id, future)."""
    approval_id = uuid.uuid4().hex
    fut: "asyncio.Future[str]" = asyncio.get_event_loop().create_future()
    _pending[approval_id] = fut
    return approval_id, fut


def resolve_approval(approval_id: str, decision: str) -> bool:
    """Resolve a pending approval from the HTTP route. Returns True if it
    matched an outstanding request."""
    fut = _pending.get(approval_id)
    if fut is None or fut.done():
        return False
    fut.set_result("approve" if str(decision).lower() in ("approve", "approved", "yes", "true", "1") else "deny")
    return True


async def wait_for_approval(approval_id: str, fut: "asyncio.Future[str]",
                            timeout: float = APPROVAL_TIMEOUT) -> str:
    """Await the user's decision; auto-deny on timeout. Always cleans up."""
    try:
        return await asyncio.wait_for(fut, timeout=timeout)
    except asyncio.TimeoutError:
        logger.info("Tool approval %s timed out after %.0fs — auto-denying", approval_id, timeout)
        return "deny"
    except asyncio.CancelledError:
        # Client disconnected mid-wait; nothing to approve anymore.
        raise
    finally:
        _pending.pop(approval_id, None)
