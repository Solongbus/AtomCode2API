"""
Async HTTP client for the atomcode-daemon process.

Reverse-engineered from the AtomCode VS Code extension (v0.0.9),
specifically ``daemon/client.js`` and ``daemon/process.js``.

The daemon listens on ``127.0.0.1:<port>`` (default 13456) and exposes
a REST + SSE API for chat, sessions, providers, auth, and project management.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Optional, Union

import httpx

logger = logging.getLogger("atomcode2api.daemon_client")

DEFAULT_DAEMON_PORT = 13456
REST_TIMEOUT = 10.0  # seconds
STREAM_TIMEOUT = 300.0  # 5 minutes for long-running chat


# ── SSE event types (mirrors daemon/client.js handleSSEData) ─────────────

@dataclass
class SSEText:
    content: str


@dataclass
class SSEToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class SSEToolBatch:
    calls: list[SSEToolCall]


@dataclass
class SSEToolStart:
    id: str
    name: str
    arguments: dict


@dataclass
class SSEToolResult:
    id: str
    name: str
    output: str
    success: bool
    duration_ms: int


@dataclass
class SSETokens:
    prompt: int
    completion: int
    total: int


@dataclass
class SSEArtifactStart:
    id: str
    artifact_type: str
    language: str
    title: str


@dataclass
class SSEArtifactContent:
    id: str
    content: str


@dataclass
class SSEArtifactEnd:
    id: str


@dataclass
class SSEDone:
    tokens: dict
    tool_calls: int
    session_id: str


@dataclass
class SSEStopped:
    pass


@dataclass
class SSEError:
    message: str


SSEEvent = Union[
    SSEText,
    SSEToolBatch,
    SSEToolStart,
    SSEToolResult,
    SSETokens,
    SSEArtifactStart,
    SSEArtifactContent,
    SSEArtifactEnd,
    SSEDone,
    SSEStopped,
    SSEError,
]


# ── Chat request payload ────────────────────────────────────────────────

@dataclass
class ChatRequest:
    """Mirrors what the VS Code extension sends to POST /chat.

    The daemon expects the **single-string** ``message`` + ``working_dir``
    format (the same format the VS Code extension always uses).  When
    ``working_dir`` is set, the daemon knows which project to operate on
    and will run the full coding-agent workflow (tool calls, file
    read/write, etc.).

    The ``messages`` field is kept for backward compatibility (used by the
    legacy CLI-mode path), but the daemon-mode path should always prefer
    ``message`` + ``working_dir``.
    """
    message: str = ""              # Single user prompt (preferred)
    working_dir: str = ""          # Project workspace directory
    messages: list[dict] = field(default_factory=list)  # legacy array format
    model: Optional[str] = None
    provider: Optional[str] = None
    stream: bool = True
    session_id: Optional[str] = None


# ── DaemonClient ────────────────────────────────────────────────────────

class DaemonClient:
    """
    Async HTTP client for atomcode-daemon.

    Usage::

        client = DaemonClient(port=13456)
        await client.health()
        models = await client.list_models()
        async for event in client.stream_chat(ChatRequest(...)):
            if isinstance(event, SSEText):
                print(event.content)
    """

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_DAEMON_PORT) -> None:
        self.base_url = f"http://{host}:{port}"
        self._client = httpx.AsyncClient(timeout=REST_TIMEOUT)

    async def close(self) -> None:
        await self._client.aclose()

    # ── generic request helpers ──────────────────────────────────────────

    async def _get(self, path: str) -> Any:
        resp = await self._client.get(f"{self.base_url}{path}")
        resp.raise_for_status()
        return resp.json()

    async def _post(self, path: str, body: Optional[dict] = None) -> Any:
        resp = await self._client.post(f"{self.base_url}{path}", json=body or {})
        resp.raise_for_status()
        return resp.json()

    async def _patch(self, path: str, body: Optional[dict] = None) -> Any:
        resp = await self._client.patch(f"{self.base_url}{path}", json=body or {})
        resp.raise_for_status()
        return resp.json()

    async def _delete(self, path: str) -> Any:
        resp = await self._client.delete(f"{self.base_url}{path}")
        resp.raise_for_status()
        return resp.json()

    # ── Health ───────────────────────────────────────────────────────────

    async def health(self) -> dict:
        """GET /health — returns daemon version and status."""
        return await self._get("/health")

    async def is_running(self) -> bool:
        try:
            await self.health()
            return True
        except Exception:
            return False

    # ── Shutdown ─────────────────────────────────────────────────────────

    async def shutdown(self) -> dict:
        """POST /shutdown — gracefully stop the daemon."""
        return await self._post("/shutdown")

    # ── Project ──────────────────────────────────────────────────────────

    async def get_project(self) -> dict:
        """GET /project — current working directory info."""
        return await self._get("/project")

    async def change_dir(self, path: str) -> dict:
        """POST /cd — change daemon working directory."""
        return await self._post("/cd", {"path": path})

    # ── Models ───────────────────────────────────────────────────────────

    async def list_models(self) -> list[dict]:
        """GET /models — returns available models (name, provider, is_default)."""
        return await self._get("/models")

    async def set_reasoning_effort(self, provider: str, effort: str) -> dict:
        """POST /live/reasoning_effort — set reasoning effort for a provider."""
        return await self._post("/live/reasoning_effort", {
            "provider": provider,
            "reasoning_effort": effort,
        })

    # ── Skills ───────────────────────────────────────────────────────────

    async def list_skills(self) -> list[dict]:
        """GET /skills — list available agent skills."""
        return await self._get("/skills")

    # ── Config ───────────────────────────────────────────────────────────

    async def get_config(self) -> dict:
        """GET /config — daemon configuration."""
        return await self._get("/config")

    async def reload_config(self) -> dict:
        """POST /config/reload — reload configuration from disk."""
        return await self._post("/config/reload")

    # ── Providers ────────────────────────────────────────────────────────

    async def list_providers(self) -> list[dict]:
        """GET /providers — list LLM providers."""
        return await self._get("/providers")

    async def create_provider(self, req: dict) -> dict:
        """POST /providers — register a new LLM provider."""
        return await self._post("/providers", req)

    async def patch_provider(self, name: str, req: dict) -> dict:
        """PATCH /providers/{name} — update a provider."""
        return await self._patch(f"/providers/{_e(name)}", req)

    async def delete_provider(self, name: str) -> dict:
        """DELETE /providers/{name} — remove a provider."""
        return await self._delete(f"/providers/{_e(name)}")

    async def set_default_provider(self, name: str) -> dict:
        """POST /providers/{name}/default — set as default."""
        return await self._post(f"/providers/{_e(name)}/default")

    async def patch_thinking(self, name: str, req: dict) -> dict:
        """PATCH /providers/{name}/thinking — configure thinking/reasoning."""
        return await self._patch(f"/providers/{_e(name)}/thinking", req)

    # ── Auth / CodingPlan ────────────────────────────────────────────────

    async def auth_status(self) -> dict:
        """GET /auth/status — current authentication state."""
        return await self._get("/auth/status")

    async def start_login(self, open_browser: bool = True) -> dict:
        """POST /auth/login/start — begin OAuth login flow."""
        return await self._post("/auth/login/start", {"open_browser": open_browser})

    async def poll_login(self, login_id: str) -> dict:
        """POST /auth/login/{id}/poll — poll for login completion."""
        return await self._post(f"/auth/login/{_e(login_id)}/poll")

    async def cancel_login(self, login_id: str) -> dict:
        """DELETE /auth/login/{id} — cancel an in-progress login."""
        return await self._delete(f"/auth/login/{_e(login_id)}")

    async def logout(self) -> dict:
        """POST /auth/logout — log out current user."""
        return await self._post("/auth/logout")

    async def setup_coding_plan(self, login_id: str) -> dict:
        """POST /codingplan/setup — activate CodingPlan after login."""
        return await self._post("/codingplan/setup", {"login_id": login_id})

    # ── Sessions ─────────────────────────────────────────────────────────

    async def list_sessions(self) -> list[dict]:
        """GET /sessions — list all chat sessions."""
        return await self._get("/sessions")

    async def get_session(self, project_hash: str, session_id: str) -> dict:
        """GET /projects/{hash}/sessions/{id} — get session details."""
        return await self._get(f"/projects/{_e(project_hash)}/sessions/{_e(session_id)}")

    async def create_session(self, title: str, working_dir: str) -> dict:
        """POST /sessions — create a new chat session."""
        return await self._post("/sessions", {
            "title": title,
            "working_dir": working_dir,
        })

    async def append_session_messages(self, session_id: str, req: dict) -> dict:
        """POST /sessions/{id}/messages — append messages to a session."""
        return await self._post(f"/sessions/{_e(session_id)}/messages", req)

    async def rename_session(self, project_hash: str, session_id: str, name: str) -> dict:
        """PATCH /projects/{hash}/sessions/{id}/rename — rename a session."""
        return await self._patch(
            f"/projects/{_e(project_hash)}/sessions/{_e(session_id)}/rename",
            {"name": name},
        )

    async def delete_session(self, project_hash: str, session_id: str) -> dict:
        """DELETE /projects/{hash}/sessions/{id} — delete a session."""
        return await self._delete(
            f"/projects/{_e(project_hash)}/sessions/{_e(session_id)}"
        )

    async def search_sessions(self, query: str) -> list[dict]:
        """GET /sessions/search?q=... — search sessions by keyword."""
        return await self._get(f"/sessions/search?q={_e(query)}")

    # ── Chat (SSE streaming) ─────────────────────────────────────────────

    async def stream_chat(
        self,
        req: ChatRequest,
    ) -> AsyncGenerator[SSEEvent, None]:
        """
        POST /chat — SSE streaming chat.

        Yields typed SSEEvent objects that mirror the daemon's event types.

        The daemon's ``/chat`` endpoint recognises **two** request shapes:

        **Agent mode** (preferred – full coding-agent workflow)
        -------------------------------------------------------
        .. code:: json

            {
              "message": "Create a login page …",
              "working_dir": "C:/Projects/myapp",
              "session_id": "abc-123",
              "stream": true
            }

        **Chat mode** (legacy – simple Q&A, no tool execution)
        ------------------------------------------------------
        .. code:: json

            {
              "messages": [{"role": "user", "content": "…"}],
              "session_id": "abc-123",
              "stream": true
            }

        This method sends **agent mode** when ``req.message`` is non-empty
        (the daemon-mode path) and falls back to **chat mode** when only
        ``req.messages`` is populated (the legacy path).
        """
        # ── Build payload ────────────────────────────────────────────────
        # Use the VS Code extension format (message + working_dir) when
        # available – this triggers the full coding-agent workflow.
        payload: dict[str, Any] = {"stream": req.stream}
        if req.message:
            payload["message"] = req.message
            if req.working_dir:
                payload["working_dir"] = req.working_dir
        else:
            payload["messages"] = req.messages
        if req.model:
            payload["model"] = req.model
        if req.provider:
            payload["provider"] = req.provider
        if req.session_id:
            payload["session_id"] = req.session_id

        async with self._client.stream(
            "POST",
            f"{self.base_url}/chat",
            json=payload,
            timeout=httpx.Timeout(STREAM_TIMEOUT, connect=REST_TIMEOUT),
            headers={"Accept": "text/event-stream"},
        ) as resp:
            resp.raise_for_status()
            buffer = ""
            async for chunk in resp.aiter_text():
                buffer += chunk
                lines = buffer.split("\n")
                buffer = lines.pop() or ""
                for line in lines:
                    trimmed = line.strip()
                    if not trimmed or trimmed.startswith(":"):
                        continue
                    if trimmed.startswith("data: "):
                        data_str = trimmed[6:]
                        event = _parse_sse(data_str)
                        if event is not None:
                            yield event

    async def stop_generation(self, session_id: str) -> dict:
        """POST /chat/stop — stop an ongoing generation."""
        return await self._post("/chat/stop", {"session_id": session_id})

    async def active_sessions(self) -> list[dict]:
        """GET /chat/active — list currently active chat sessions."""
        return await self._get("/chat/active")


# ── Internals ────────────────────────────────────────────────────────────

def _e(s: str) -> str:
    """URL-encode a path segment."""
    import urllib.parse
    return urllib.parse.quote(s, safe="")


def _parse_sse(data: str) -> Optional[SSEEvent]:
    """Parse a single SSE data JSON into a typed event."""
    try:
        event = json.loads(data)
    except json.JSONDecodeError:
        return None

    t = event.get("type")
    if t == "text":
        return SSEText(content=event.get("content", ""))
    elif t == "tool_batch":
        raw_calls = event.get("calls", [])
        calls = [SSEToolCall(id=c["id"], name=c["name"], arguments=c.get("arguments", {}))
                 for c in raw_calls]
        return SSEToolBatch(calls=calls)
    elif t == "tool_start":
        return SSEToolStart(id=event["id"], name=event["name"], arguments=event.get("arguments", {}))
    elif t == "tool_result":
        return SSEToolResult(
            id=event["id"], name=event["name"], output=event.get("output", ""),
            success=event.get("success", True), duration_ms=event.get("duration_ms", 0),
        )
    elif t == "tokens":
        return SSETokens(
            prompt=event.get("prompt", 0), completion=event.get("completion", 0),
            total=event.get("total", 0),
        )
    elif t == "artifact_start":
        return SSEArtifactStart(
            id=event["id"], artifact_type=event.get("artifact_type", ""),
            language=event.get("language", ""), title=event.get("title", ""),
        )
    elif t == "artifact_content":
        return SSEArtifactContent(id=event["id"], content=event.get("content", ""))
    elif t == "artifact_end":
        return SSEArtifactEnd(id=event["id"])
    elif t == "done":
        return SSEDone(
            tokens=event.get("tokens", {}), tool_calls=event.get("tool_calls", 0),
            session_id=event.get("session_id", ""),
        )
    elif t == "stopped":
        return SSEStopped()
    elif t == "error":
        return SSEError(message=event.get("message", "Unknown error"))
    return None
