# routes/approval_routes.py
"""Tool-approval API — resolves a pending approval the agent loop is awaiting.

When the agent is in 'accept_edits' mode and wants to run a destructive tool,
stream_agent_loop emits a `tool_approval_request` SSE event carrying an
approval_id and then awaits an asyncio Future. The browser shows Approve/Deny
buttons; clicking one POSTs here, which resolves that Future so the stream
resumes (running or skipping the tool).
"""

import logging
from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)


def setup_approval_routes():
    router = APIRouter(prefix="/api/tool_approval", tags=["approval"])

    @router.post("/{approval_id}")
    async def resolve_tool_approval(approval_id: str, request: Request):
        """Resolve a pending tool approval. Body: {"decision": "approve"|"deny"}."""
        from src.agent_modes import resolve_approval

        decision = "deny"
        try:
            body = await request.json()
            decision = str(body.get("decision", "deny"))
        except Exception:
            pass

        matched = resolve_approval(approval_id, decision)
        if not matched:
            # Already resolved, timed out, or unknown id — not an error worth
            # 500ing over; tell the client it's no longer actionable.
            raise HTTPException(status_code=404, detail={"message": "No pending approval for that id (it may have timed out)."})
        return {"ok": True, "decision": "approve" if decision.lower() in ("approve", "approved", "yes", "true", "1") else "deny"}

    return router
