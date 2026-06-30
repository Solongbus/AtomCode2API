"""
FastAPI application entry point.

Supports two modes (configurable via ``ATOMCODE_MODE`` env var):

1. **daemon mode** (default) — Connects to (or auto-starts) the persistent
   ``atomcode-daemon`` process over its HTTP API on port 13456.  Rich SSE
   events (tool calls, artifacts, token counts) are translated to the
   OpenAI SSE format.

2. **cli mode** (legacy) — Spawns ``atomcode --execute <prompt>`` as an
   isolated subprocess per request.  Simpler but less powerful.

Endpoints
---------
POST /api/v1/agent/coding           → Trigger an atomcode task (both modes).
GET  /api/v1/agent/task/{id}        → Query task status & real-time logs.
POST /v1/chat/completions           → OpenAI-compatible endpoint.
GET  /v1/models                     → List available models (daemon mode).
GET  /v1/providers                  → List LLM providers (daemon mode).
POST /v1/providers                  → Add a provider (daemon mode).
GET  /v1/sessions                   → List chat sessions (daemon mode).
POST /v1/sessions                   → Create a session (daemon mode).
DELETE /v1/sessions/{id}            → Delete a session (daemon mode).
GET  /v1/skills                     → List available skills (daemon mode).
GET  /health                        → Health check (always public).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager

# PyInstaller --windowed mode strips the console, making sys.std{out,err} == None.
# uvicorn's logging formatter calls .isatty() on sys.stderr, which crashes on None.
# Must be at module level, before any uvicorn import.
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
from pathlib import Path
from typing import Any, AsyncGenerator, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

try:
    from atomcode2api.config import settings
except ModuleNotFoundError:
    from config import settings

# ── Mode-specific imports & globals ─────────────────────────────────────

# Lazy-init singletons populated in lifespan()
_daemon_client: Any = None  # daemon_client.DaemonClient
_daemon_manager: Any = None  # daemon_manager.DaemonManager

if settings.mode == "daemon":
    try:
        from atomcode2api.daemon_client import (
            ChatRequest,
            DaemonClient,
            SSEArtifactContent,
            SSEArtifactEnd,
            SSEArtifactStart,
            SSEDone,
            SSEError,
            SSEStopped,
            SSEText,
            SSEToolBatch,
            SSEToolCall,
            SSEToolResult,
            SSEToolStart,
            SSETokens,
        )
        from atomcode2api.daemon_manager import DaemonManager
    except ModuleNotFoundError:
        from daemon_client import (
            ChatRequest,
            DaemonClient,
            SSEArtifactContent,
            SSEArtifactEnd,
            SSEArtifactStart,
            SSEDone,
            SSEError,
            SSEStopped,
            SSEText,
            SSEToolBatch,
            SSEToolCall,
            SSEToolResult,
            SSEToolStart,
            SSETokens,
        )
        from daemon_manager import DaemonManager
else:
    try:
        from atomcode2api.utils.executor import (
            get_task,
            init_task,
            run_atomcode,
            STATUS_COMPLETED,
            STATUS_FAILED,
            STATUS_QUEUED,
            STATUS_RUNNING,
        )
        from atomcode2api.utils.locker import clear_all_locks
    except ModuleNotFoundError:
        from utils.executor import (
            get_task,
            init_task,
            run_atomcode,
            STATUS_COMPLETED,
            STATUS_FAILED,
            STATUS_QUEUED,
            STATUS_RUNNING,
        )
        from utils.locker import clear_all_locks

# ── Python 3.9 compatibility: fix pydantic json_schema ──────────────────
if sys.version_info < (3, 10):

    def _patch_pydantic_json_schema():
        from pydantic.json_schema import GenerateJsonSchema
        from typing_inspection.introspection import get_literal_values
        from pydantic.json_schema import CoreSchemaOrFieldType

        _flat = []
        for item in get_literal_values(CoreSchemaOrFieldType):
            if isinstance(item, str):
                _flat.append(item)
            else:
                try:
                    for sub in get_literal_values(item):
                        if isinstance(sub, str):
                            _flat.append(sub)
                except Exception:
                    continue
        _all_keys = tuple(_flat)

        orig_build = GenerateJsonSchema.build_schema_type_to_method

        def _build_schema_type_to_method(self):
            mapping: dict = {}
            for key in _all_keys:
                method_name = f'{key.replace("-", "_")}_schema'
                try:
                    mapping[key] = getattr(self, method_name)
                except AttributeError:
                    continue
            return mapping

        GenerateJsonSchema.build_schema_type_to_method = _build_schema_type_to_method

    _patch_pydantic_json_schema()

# ── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s %(levelname)-8s %(message)s",
)
logger = logging.getLogger("atomcode2api")


# ── Lifecycle ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "atomcode2api starting  mode=%s  host=%s  port=%d",
        settings.mode, settings.host, settings.port,
    )

    # Clear stale lock files from the temp directory so that leftover
    # locks from a previous crash never block subsequent conversations.
    clear_all_locks()

    if settings.mode == "daemon":
        global _daemon_client, _daemon_manager
        _daemon_client = DaemonClient(
            host=settings.daemon_host,
            port=settings.daemon_port,
        )
        _daemon_manager = DaemonManager(
            client=_daemon_client,
            binary_path=settings.daemon_binary_path,
            auto_start=settings.daemon_auto_start,
            daemon_port=settings.daemon_port,
        )
        if settings.daemon_auto_start:
            running = await _daemon_manager.ensure_running()
            logger.info("Daemon ensure_running: %s", running)

    yield

    if settings.mode == "daemon" and _daemon_client:
        await _daemon_client.close()
    if settings.mode == "daemon" and _daemon_manager:
        _daemon_manager.dispose()


app = FastAPI(
    title="atomcode2api",
    version="0.2.0",
    lifespan=lifespan,
)


# ── Pydantic models ─────────────────────────────────────────────────────

class CodingRequest(BaseModel):
    prompt: str = Field(..., description="Natural-language coding task description")
    project_path: str = Field(..., description="Absolute path to the target workspace")


class CodingResponse(BaseModel):
    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    logs: str
    updated_at: str


class OpenAIChatMessage(BaseModel):
    model_config = {"extra": "ignore"}  # tolerate extra fields from Cursor/Trae

    role: str
    content: Optional[str] = ""
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[dict]] = None

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, v):
        """Normalize multimodal array content to plain string."""
        if isinstance(v, list):
            parts = []
            for item in v:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "\n".join(parts) or ""
        return v or ""


class OpenAIChatRequest(BaseModel):
    model_config = {"extra": "ignore"}  # tolerate Cursor/Trae extra fields

    model: str = "atomcode"
    messages: List[OpenAIChatMessage]
    stream: bool = False
    project_path: Optional[str] = None
    session_id: Optional[str] = None
    temperature: Optional[float] = None


class OpenAIChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: Optional[dict] = None


# ── Auth middleware ─────────────────────────────────────────────────────

async def verify_api_key(authorization: Optional[str] = Header(None)):
    if not settings.api_key:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ── CLI-mode helpers (kept for backward compatibility) ──────────────────

if settings.mode != "daemon":
    # These are only used in cli mode; in daemon mode the daemon itself
    # manages locking.
    _cli_background_tasks: set[asyncio.Task] = set()

    def _validate_project_path(path: str) -> Path:
        p = Path(path).resolve()
        if not p.is_dir():
            raise HTTPException(status_code=400, detail=f"Project path is not a directory: {p}")
        # Check allowed roots if configured
        if settings.allowed_project_roots:
            if not any(str(p).startswith(root) for root in settings.allowed_project_roots):
                raise HTTPException(
                    status_code=403,
                    detail=f"Project path not in allowed roots: {p}",
                )
        return p

    def _resolve_project_path(
        project_path_field: Optional[str], request: Request
    ) -> str:
        path = (
            project_path_field
            or request.headers.get("X-Project-Path")
            or settings.default_workspace
            or os.getcwd()
        )
        return str(_validate_project_path(path))

    def _extract_prompt(messages: List[OpenAIChatMessage]) -> str:
        parts = []
        for msg in messages:
            prefix = f"[{msg.role.upper()}]"
            parts.append(f"{prefix}\n{msg.content}")
        return "\n\n".join(parts)

    def _log_stream_task_result(task_id: str, task: asyncio.Task) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            logger.warning("[openai][cli] background task cancelled task=%s", task_id)
        except Exception:
            logger.exception("[openai][cli] background task crashed task=%s", task_id)

    async def _run_cli_background_task(
        task_id: str,
        prompt: str,
        project_path: str,
    ) -> None:
        logger.info("[openai][cli] background task entered task=%s", task_id)
        try:
            await run_atomcode(task_id, prompt, project_path)
        finally:
            record = get_task(task_id) or {}
            logger.info(
                "[openai][cli] background task finished task=%s status=%s logs_len=%d",
                task_id,
                record.get("status"),
                len(record.get("logs", "")),
            )

    def _spawn_cli_background_task(
        task_id: str,
        prompt: str,
        project_path: str,
    ) -> asyncio.Task:
        task = asyncio.create_task(_run_cli_background_task(task_id, prompt, project_path))
        _cli_background_tasks.add(task)
        task.add_done_callback(lambda done: _cli_background_tasks.discard(done))
        task.add_done_callback(lambda done: _log_stream_task_result(task_id, done))
        return task

    async def _stream_cli_chunks(
        task_id: str,
        project_path: str,
        model: str,
        request: Request,
    ):
        """Async generator that streams atomcode CLI logs as OpenAI SSE chunks.

        Sends periodic SSE keep-alive comments (every 5 s) while waiting
        for the subprocess to produce output, so the IDE connection never
        times out during the AI's "thinking" phase.
        """
        logger.info("[openai][cli] stream enter task=%s project=%s", task_id, project_path)
        last_log_len = 0
        yielded_any_chunk = False
        last_keepalive = time.monotonic()
        KEEPALIVE_INTERVAL = 5.0  # seconds
        accumulated_content = ""

        while True:
            record = get_task(task_id)
            if record is None:
                logger.warning("[openai][cli] stream missing task record task=%s", task_id)
                break

            # ── Read new logs ──────────────────────────────────────────
            new_logs = record["logs"][last_log_len:]
            if new_logs:
                for line in new_logs.splitlines(keepends=True):
                    yielded_any_chunk = True
                    chunk_str = _sse_chunk(task_id, model, {"content": line})
                    accumulated_content += line
                    yield chunk_str
                last_log_len = len(record["logs"])
                last_keepalive = time.monotonic()  # reset keepalive on real data

            # ── Check terminal status ──────────────────────────────────
            if record["status"] in (STATUS_COMPLETED, STATUS_FAILED):
                logger.info(
                    "[openai][cli] stream exit task=%s final_status=%s logs_len=%d",
                    task_id,
                    record["status"],
                    len(record["logs"]),
                )
                break

            # ── Client disconnect ──────────────────────────────────────
            if await request.is_disconnected():
                logger.warning("[openai][cli] stream disconnected task=%s", task_id)
                break

            # ── First chunk: set role ──────────────────────────────────
            if not yielded_any_chunk:
                yielded_any_chunk = True
                # OpenAI-compatible first chunk: role + empty content
                yield _sse_chunk(task_id, model, {"role": "assistant", "content": ""})
                last_keepalive = time.monotonic()

            # ── Keep-alive heartbeat ───────────────────────────────────
            elif time.monotonic() - last_keepalive >= KEEPALIVE_INTERVAL:
                yield ": keepalive\n\n"
                last_keepalive = time.monotonic()

            await asyncio.sleep(0.1)  # poll interval

        yield _sse_chunk(task_id, model, {}, finish_reason="stop")
        yield "data: [DONE]\n\n"
        logger.info(
            "[openai][cli] stream done task=%s total_content_len=%d",
            task_id,
            len(accumulated_content),
        )


# ── /api/v1/agent/coding ────────────────────────────────────────────────

@app.post(
    "/api/v1/agent/coding",
    status_code=202,
    response_model=CodingResponse,
    summary="Trigger an atomcode coding task",
    dependencies=[Depends(verify_api_key)],
)
async def trigger_coding(body: CodingRequest, background_tasks: BackgroundTasks):
    """Start a background atomcode task (non-streaming, agent-style)."""
    task_id = str(uuid.uuid4())

    if settings.mode == "daemon":
        # In daemon mode, we use the daemon's chat API with session support.
        # For the agent endpoint, create a session and kick off the task.
        try:
            from atomcode2api.daemon_client import ChatRequest as DCR
        except ModuleNotFoundError:
            from daemon_client import ChatRequest as DCR

        session = await _daemon_client.create_session(
            title=body.prompt[:80],
            working_dir=body.project_path,
        )
        session_id = session.get("session_id") or session.get("id")

        # Store as in-memory task record for polling
        if _daemon_client is not None:
            _init_task(task_id, body.prompt, body.project_path, session_id)

        # Launch streaming in background — use VS Code extension message format
        background_tasks.add_task(
            _run_daemon_task, task_id, DCR(
                message=body.prompt,
                working_dir=body.project_path,
                session_id=session_id,
            )
        )

        return CodingResponse(
            task_id=task_id,
            status="queued",
            message="Agent task queued. Poll GET /api/v1/agent/task/{task_id} for status.",
        )
    else:
        # CLI mode: original subprocess approach
        _validate_project_path(body.project_path)
        init_task(task_id, body.prompt, body.project_path)
        background_tasks.add_task(run_atomcode, task_id, body.prompt, body.project_path)
        return CodingResponse(
            task_id=task_id,
            status="queued",
            message="Agent triggered successfully.",
        )


@app.get(
    "/api/v1/agent/task/{task_id}",
    response_model=TaskStatusResponse,
    summary="Query task status & logs",
    dependencies=[Depends(verify_api_key)],
)
async def query_task(task_id: str):
    if settings.mode == "daemon":
        record = _get_task(task_id)
    else:
        record = get_task(task_id)

    if record is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    return TaskStatusResponse(
        task_id=record["task_id"],
        status=record["status"],
        logs=record["logs"],
        updated_at=record["updated_at"],
    )


# ── /health ─────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
async def health():
    if settings.mode == "daemon":
        try:
            daemon_health = await _daemon_client.health()
            return {
                "status": "ok",
                "mode": "daemon",
                "daemon": daemon_health,
            }
        except Exception as exc:
            return {
                "status": "degraded",
                "mode": "daemon",
                "daemon_error": str(exc),
            }
    return {"status": "ok", "mode": "cli"}


# ── Atomcode config.toml reader ─────────────────────────────────────────

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Python < 3.11


_ATOMCODE_CONFIG_PATH = Path.home() / ".atomcode" / "config.toml"
_atomcode_models_cache: Optional[list[dict]] = None
_atomcode_models_cache_ts: float = 0
_ATOMCODE_CACHE_TTL = 60.0  # seconds


def _load_atomcode_models() -> list[dict]:
    """Read ``~/.atomcode/config.toml`` and return a list of OpenAI-compatible
    model objects for every configured provider.

    Results are cached for ``_ATOMCODE_CACHE_TTL`` seconds.
    """
    global _atomcode_models_cache, _atomcode_models_cache_ts

    now = time.time()
    if _atomcode_models_cache is not None and (now - _atomcode_models_cache_ts) < _ATOMCODE_CACHE_TTL:
        return _atomcode_models_cache

    models: list[dict] = []
    if not _ATOMCODE_CONFIG_PATH.is_file():
        # Fallback: return a single default model
        models.append(_make_model_entry("atomcode"))
        _atomcode_models_cache = models
        _atomcode_models_cache_ts = now
        return models

    try:
        raw = _ATOMCODE_CONFIG_PATH.read_bytes()
        data = tomllib.loads(raw.decode("utf-8"))
    except Exception as exc:
        logging.getLogger("atomcode2api").warning(
            "Failed to parse %s: %s", _ATOMCODE_CONFIG_PATH, exc
        )
        models.append(_make_model_entry("atomcode"))
        _atomcode_models_cache = models
        _atomcode_models_cache_ts = now
        return models

    providers = data.get("providers", {})
    for provider_name, provider_cfg in providers.items():
        if not isinstance(provider_cfg, dict):
            continue
        model_name = provider_cfg.get("model", provider_name)
        models.append(_make_model_entry(model_name, provider_name))

    # If config has a default_provider, sort it first
    default_provider = data.get("default_provider", "")
    if default_provider:
        models.sort(key=lambda m: (0 if m.get("id") == default_provider else 1))

    if not models:
        models.append(_make_model_entry("atomcode"))

    _atomcode_models_cache = models
    _atomcode_models_cache_ts = now
    return models


def _make_model_entry(model_id: str, provider_name: str = "") -> dict:
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": provider_name or "atomcode",
    }


# ── /v1/models ──────────────────────────────────────────────────────────

async def _get_models_list():
    """Return OpenAI-standard model list from daemon or toml fallback."""
    if settings.mode == "daemon" and _daemon_client is not None:
        try:
            raw_models = await _daemon_client.list_models()
            models = []
            now_ts = int(time.time())
            for m in raw_models:
                if not isinstance(m, dict):
                    continue
                model_id = m.get("model") or m.get("id", "unknown")
                provider = m.get("provider") or m.get("owned_by", "")
                models.append({
                    "id": model_id,
                    "object": "model",
                    "created": now_ts,
                    "owned_by": provider,
                })
            return {"object": "list", "data": models}
        except Exception as exc:
            logger.warning("Daemon list_models failed, falling back to config: %s", exc)
    # Fallback / CLI mode
    return {"object": "list", "data": _load_atomcode_models()}


@app.get(
    "/v1/models",
    summary="List available models",
    dependencies=[Depends(verify_api_key)],
    include_in_schema=True,
)
async def list_models():
    return await _get_models_list()


@app.get(
    "/models",
    summary="List available models (no v1 prefix)",
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
async def list_models_no_prefix():
    return await _get_models_list()


# ── /v1/providers (daemon mode only) ────────────────────────────────────

@app.get(
    "/v1/providers",
    summary="List LLM providers",
    dependencies=[Depends(verify_api_key)],
)
async def list_providers():
    if settings.mode != "daemon":
        raise HTTPException(status_code=400, detail="Providers only available in daemon mode")
    try:
        return await _daemon_client.list_providers()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Daemon error: {exc}")


@app.post(
    "/v1/providers",
    summary="Add a new LLM provider",
    dependencies=[Depends(verify_api_key)],
)
async def create_provider(body: dict):
    if settings.mode != "daemon":
        raise HTTPException(status_code=400, detail="Providers only available in daemon mode")
    try:
        return await _daemon_client.create_provider(body)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Daemon error: {exc}")


# ── /v1/sessions (daemon mode only) ─────────────────────────────────────

@app.get(
    "/v1/sessions",
    summary="List chat sessions",
    dependencies=[Depends(verify_api_key)],
)
async def list_sessions(q: Optional[str] = None):
    """List all sessions, or search by keyword with ``?q=...``."""
    if settings.mode != "daemon":
        raise HTTPException(status_code=400, detail="Sessions only available in daemon mode")
    try:
        if q:
            return await _daemon_client.search_sessions(q)
        return await _daemon_client.list_sessions()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Daemon error: {exc}")


@app.post(
    "/v1/sessions",
    summary="Create a new session",
    dependencies=[Depends(verify_api_key)],
)
async def create_session(body: dict):
    if settings.mode != "daemon":
        raise HTTPException(status_code=400, detail="Sessions only available in daemon mode")
    try:
        return await _daemon_client.create_session(
            title=body.get("title", ""),
            working_dir=body.get("working_dir", ""),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Daemon error: {exc}")


@app.delete(
    "/v1/sessions/{project_hash}/{session_id}",
    summary="Delete a session",
    dependencies=[Depends(verify_api_key)],
)
async def delete_session(project_hash: str, session_id: str):
    if settings.mode != "daemon":
        raise HTTPException(status_code=400, detail="Sessions only available in daemon mode")
    try:
        return await _daemon_client.delete_session(project_hash, session_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Daemon error: {exc}")


# ── /v1/skills (daemon mode only) ───────────────────────────────────────

@app.get(
    "/v1/skills",
    summary="List available skills",
    dependencies=[Depends(verify_api_key)],
)
async def list_skills():
    if settings.mode != "daemon":
        raise HTTPException(status_code=400, detail="Skills only available in daemon mode")
    try:
        return await _daemon_client.list_skills()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Daemon error: {exc}")


# ── /v1/chat/completions (OpenAI-compatible) ────────────────────────────

@app.post(
    "/v1/chat/completions",
    summary="OpenAI-compatible chat completions",
    dependencies=[Depends(verify_api_key)],
    include_in_schema=True,
)
async def chat_completions(body: OpenAIChatRequest, request: Request):
    """
    OpenAI-compatible endpoint for Cursor / Trae / etc.

    **Daemon mode** (default):
    - Supports multi-turn conversations via ``session_id``.
    - SSE streams typed events (text, tool calls, artifacts) as delta content.
    - Returns token usage when available.

    **CLI mode** (legacy):
    - Concatenates messages into a single prompt, runs as subprocess.
    - Simpler but loses tool call / artifact granularity.
    """
    logger.info(
        "[openai] request mode=%s model=%s stream=%s messages=%d",
        settings.mode,
        body.model,
        body.stream,
        len(body.messages),
    )
    if settings.mode == "daemon":
        return await _chat_completions_daemon(body, request)
    else:
        return await _chat_completions_cli(body, request)


# ── Daemon-mode chat implementation ─────────────────────────────────────

async def _chat_completions_daemon(body: OpenAIChatRequest, request: Request):
    logger.info(
        "[openai][daemon] start model=%s stream=%s messages=%d",
        body.model,
        body.stream,
        len(body.messages),
    )

    # Resolve project path
    project_path = (
        body.project_path
        or request.headers.get("X-Project-Path")
        or settings.default_workspace
    )
    if not project_path:
        # If no project path is given, use daemon's current working directory
        try:
            proj = await _daemon_client.get_project()
            project_path = proj.get("cwd") or proj.get("path", "")
            logger.info("[openai][daemon] fallback project_path from daemon=%s", project_path)
        except Exception as exc:
            logger.warning("[openai][daemon] get_project failed: %s", exc)
            project_path = ""
    if project_path:
        try:
            await _daemon_client.change_dir(project_path)
            logger.info("[openai][daemon] changed dir to %s", project_path)
        except Exception as exc:
            logger.warning("Could not change daemon working dir to %s: %s", project_path, exc)

    # Convert OpenAI messages → daemon single-string message
    # The daemon expects the VS-Code-extension format:
    #   { "message": "…", "working_dir": "…", "session_id": "…" }
    # which triggers the full coding-agent workflow (tools, artifacts, …).
    # DO NOT send the "messages" array — that makes the daemon treat it
    # as a simple Q&A without agent capabilities.
    user_message = ""
    for msg in reversed(body.messages):
        if msg.role == "user" and msg.content:
            user_message = msg.content
            break

    logger.info(
        "[openai][daemon] message=%s  working_dir=%s  session_id=%s",
        user_message[:120],
        project_path,
        body.session_id,
    )
    chat_req = ChatRequest(
        message=user_message,
        working_dir=project_path,
        model=body.model if body.model != "atomcode" else None,
        stream=True,
        session_id=body.session_id,
    )

    if body.stream:
        logger.info("[openai][daemon] return streaming response")
        return StreamingResponse(
            _stream_daemon_sse(chat_req, body.model, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming: accumulate all events
    logger.info("[openai][daemon] start non-stream collect")
    full_content = ""
    tool_calls: list[dict] = []
    usage_info: Optional[dict] = None
    artifact_buffer: dict[str, str] = {}
    finish = "stop"

    async for event in _daemon_client.stream_chat(chat_req):
        if isinstance(event, SSEText):
            full_content += event.content
        elif isinstance(event, SSEToolBatch):
            for tc in event.calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                })
        elif isinstance(event, SSEToolResult):
            # Append tool result as content
            full_content += f"\n[Tool: {event.name}]\n{event.output}\n"
        elif isinstance(event, SSEArtifactStart):
            artifact_buffer[event.id] = f"\n```{event.language}\n"
        elif isinstance(event, SSEArtifactContent):
            if event.id in artifact_buffer:
                artifact_buffer[event.id] += event.content
        elif isinstance(event, SSEArtifactEnd):
            if event.id in artifact_buffer:
                artifact_buffer[event.id] += "\n```\n"
                full_content += artifact_buffer.pop(event.id)
        elif isinstance(event, SSETokens):
            usage_info = {
                "prompt_tokens": event.prompt,
                "completion_tokens": event.completion,
                "total_tokens": event.total,
            }
        elif isinstance(event, SSEDone):
            finish = "stop"
            if event.session_id:
                full_content += f"\n\n_Session: {event.session_id}_"
            break
        elif isinstance(event, SSEStopped):
            finish = "stop"
            break
        elif isinstance(event, SSEError):
            finish = "error"
            full_content += f"\n\n**Error:** {event.message}"
            break

    message: dict = {"role": "assistant", "content": full_content or None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    logger.info(
        "[openai][daemon] non-stream complete finish=%s content_len=%d tool_calls=%d",
        finish,
        len(full_content),
        len(tool_calls),
    )

    return OpenAIChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=body.model,
        choices=[{"index": 0, "message": message, "finish_reason": finish}],
        usage=usage_info,
    )


async def _stream_daemon_sse(
    chat_req: ChatRequest,
    model: str,
    request: Request,
) -> AsyncGenerator[str, None]:
    """Translate daemon SSE events → OpenAI chat.completion.chunk SSE.

    Includes periodic SSE keep-alive comments during long pauses
    (AI thinking / tool execution) so the IDE never times out.
    """
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    artifact_buffer: dict[str, str] = {}
    tool_calls_acc: dict[str, dict] = {}

    # ── Keep-alive task ──────────────────────────────────────────────────
    # Send a "noop" SSE comment every 10 seconds while no real event
    # arrives.  This keeps the HTTP connection alive during long AI
    # thinking or tool-execution pauses.
    KEEPALIVE_INTERVAL = 10.0

    async def _keepalive() -> AsyncGenerator[str, None]:
        while True:
            await asyncio.sleep(KEEPALIVE_INTERVAL)
            yield ": keepalive\n\n"

    keepalive_gen = _keepalive()
    # We'll interleave keep-alive via asyncio.wait / queue below.

    # ── Process events ───────────────────────────────────────────────────
    event_iter = _daemon_client.stream_chat(chat_req)
    next_event_task: Optional[asyncio.Task] = None
    keepalive_task: Optional[asyncio.Task] = None

    try:
        # Prime: start fetching the first event
        next_event_task = asyncio.ensure_future(event_iter.__anext__())
        keepalive_task = asyncio.ensure_future(keepalive_gen.__anext__())

        while True:
            if await request.is_disconnected():
                logger.info("[openai][daemon] client disconnected, stopping stream")
                return

            done_set, _ = await asyncio.wait(
                [next_event_task, keepalive_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if keepalive_task in done_set:
                # Keep-alive timer fired — send a heartbeat
                yield ": keepalive\n\n"
                keepalive_task = asyncio.ensure_future(keepalive_gen.__anext__())
                continue

            # An event arrived
            try:
                event = next_event_task.result()
            except StopAsyncIteration:
                # No more events from daemon — stream is over
                break
            except Exception as exc:
                logger.error("[openai][daemon] stream error: %s", exc)
                break
            finally:
                next_event_task = None

            # Cancel the pending keepalive and schedule next event
            if keepalive_task and not keepalive_task.done():
                keepalive_task.cancel()
            keepalive_task = asyncio.ensure_future(keepalive_gen.__anext__())
            next_event_task = asyncio.ensure_future(event_iter.__anext__())

            # ── Handle typed events ──────────────────────────────────────
            if isinstance(event, SSEText):
                logger.debug("[openai][daemon] text chunk len=%d", len(event.content))
                delta = {"content": event.content}
                yield _sse_chunk(response_id, model, delta)

            elif isinstance(event, SSEToolBatch):
                logger.info("[openai][daemon] tool_batch calls=%d", len(event.calls))
                for tc in event.calls:
                    idx = len(tool_calls_acc)
                    tool_calls_acc[tc.id] = tc
                    delta = {
                        "tool_calls": [{
                            "index": idx,
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }]
                    }
                    yield _sse_chunk(response_id, model, delta)

            elif isinstance(event, SSEToolResult):
                logger.info("[openai][daemon] tool_result name=%s len=%d success=%s",
                            event.name, len(event.output), event.success)
                delta = {"content": f"\n\n**Tool result ({event.name}):**\n{event.output}\n"}
                yield _sse_chunk(response_id, model, delta)

            elif isinstance(event, SSEArtifactStart):
                logger.info("[openai][daemon] artifact_start id=%s lang=%s", event.id, event.language)
                artifact_buffer[event.id] = ""
                delta = {"content": f"\n\n```{event.language}\n"}
                yield _sse_chunk(response_id, model, delta)

            elif isinstance(event, SSEArtifactContent):
                if event.id in artifact_buffer:
                    artifact_buffer[event.id] += event.content
                    delta = {"content": event.content}
                    yield _sse_chunk(response_id, model, delta)

            elif isinstance(event, SSEArtifactEnd):
                if event.id in artifact_buffer:
                    logger.info("[openai][daemon] artifact_end id=%s total_len=%d",
                                event.id, len(artifact_buffer[event.id]))
                    yield _sse_chunk(response_id, model, {"content": "\n```\n"})
                    artifact_buffer.pop(event.id, None)

            elif isinstance(event, SSETokens):
                logger.debug("[openai][daemon] tokens p=%d c=%d t=%d",
                             event.prompt, event.completion, event.total)

            elif isinstance(event, SSEDone):
                logger.info("[openai][daemon] done — ending stream")
                finish = "stop"
                usage = {
                    "prompt_tokens": event.tokens.get("prompt", 0),
                    "completion_tokens": event.tokens.get("completion", 0),
                    "total_tokens": event.tokens.get("total", 0),
                } if event.tokens else None
                if event.session_id:
                    logger.info("[openai][daemon] session_id=%s", event.session_id)
                yield _sse_chunk(response_id, model, {}, finish_reason=finish, usage=usage)
                yield "data: [DONE]\n\n"
                return

            elif isinstance(event, SSEStopped):
                logger.info("[openai][daemon] stopped")
                yield _sse_chunk(response_id, model, {}, finish_reason="stop")
                yield "data: [DONE]\n\n"
                return

            elif isinstance(event, SSEError):
                logger.error("[openai][daemon] error: %s", event.message)
                delta = {"content": f"\n\n**Error:** {event.message}"}
                yield _sse_chunk(response_id, model, delta, finish_reason="error")
                yield "data: [DONE]\n\n"
                return

            else:
                logger.warning("[openai][daemon] unknown event type: %s", type(event).__name__)

    finally:
        # Cleanup tasks
        if next_event_task and not next_event_task.done():
            next_event_task.cancel()
        if keepalive_task and not keepalive_task.done():
            keepalive_task.cancel()
        # Drain the generator in case the daemon still has data
        # (prevents "async generator not exhausted" warnings)
        if 'event_iter' in locals():
            await event_iter.aclose()

    # Fallback
    logger.info("[openai][daemon] stream ended naturally (fallback)")
    yield _sse_chunk(response_id, model, {}, finish_reason="stop")
    yield "data: [DONE]\n\n"


# ── CLI-mode chat implementation (unchanged from original) ──────────────

async def _chat_completions_cli(body: OpenAIChatRequest, request: Request):
    prompt = _extract_prompt(body.messages)
    project_path = _resolve_project_path(body.project_path, request)
    Path(project_path).resolve()

    task_id = str(uuid.uuid4())
    logger.info("[openai][cli] start task=%s stream=%s project=%s prompt_len=%d", task_id, body.stream, project_path, len(prompt))
    init_task(task_id, prompt, project_path)

    if body.stream:
        logger.info("[openai][cli] spawn background task=%s", task_id)
        _spawn_cli_background_task(task_id, prompt, project_path)
        return StreamingResponse(
            _stream_cli_chunks(task_id, project_path, body.model, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    await run_atomcode(task_id, prompt, project_path)
    record = get_task(task_id)
    if record is None:
        record = {"logs": "", "status": "failed"}
    logger.info("[openai][cli] finish task=%s status=%s logs_len=%d", task_id, record.get("status"), len(record.get("logs", "")))

    return OpenAIChatResponse(
        id=f"chatcmpl-{task_id}",
        created=int(time.time()),
        model=body.model,
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": record["logs"] or f"[task {record['status']}]",
                },
                "finish_reason": "stop",
            }
        ],
    )


# ── In-memory task store for daemon mode ────────────────────────────────

import threading

_daemon_task_store: dict[str, dict] = {}
_daemon_task_lock = threading.Lock()


def _init_task(task_id: str, prompt: str, project_path: str, session_id: str = "") -> dict:
    from datetime import datetime, timezone
    record = {
        "task_id": task_id,
        "prompt": prompt,
        "project_path": project_path,
        "session_id": session_id,
        "status": "queued",
        "logs": "",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with _daemon_task_lock:
        _daemon_task_store[task_id] = record
    return record


def _get_task(task_id: str) -> Optional[dict]:
    with _daemon_task_lock:
        raw = _daemon_task_store.get(task_id)
        return dict(raw) if raw else None


def _update_task(task_id: str, **kwargs):
    with _daemon_task_lock:
        record = _daemon_task_store.get(task_id)
        if record:
            record.update(kwargs)
            from datetime import datetime, timezone
            record["updated_at"] = datetime.now(timezone.utc).isoformat()


async def _run_daemon_task(task_id: str, chat_req: ChatRequest):
    """Run a daemon chat in the background and collect logs."""
    _update_task(task_id, status="running")
    logs: list[str] = []

    try:
        async for event in _daemon_client.stream_chat(chat_req):
            if isinstance(event, SSEText):
                logs.append(event.content)
            elif isinstance(event, SSEToolBatch):
                logs.append(f"\n[Tool calls: {', '.join(c.name for c in event.calls)}]\n")
            elif isinstance(event, SSEToolResult):
                logs.append(f"\n[Tool: {event.name}]\n{event.output}\n")
            elif isinstance(event, SSEArtifactStart):
                logs.append(f"\n```{event.language}\n")
            elif isinstance(event, SSEArtifactContent):
                logs.append(event.content)
            elif isinstance(event, SSEArtifactEnd):
                logs.append("\n```\n")
            elif isinstance(event, SSETokens):
                logger.debug("[task %s] tokens p=%d c=%d t=%d",
                             task_id, event.prompt, event.completion, event.total)
            elif isinstance(event, SSEDone):
                _update_task(task_id, status="completed")
                break
            elif isinstance(event, SSEStopped):
                _update_task(task_id, status="completed")
                break
            elif isinstance(event, SSEError):
                logs.append(f"\n[Error] {event.message}")
                _update_task(task_id, status="failed")
                break
    except Exception as exc:
        logs.append(f"\n[Exception] {exc}")
        _update_task(task_id, status="failed")

    _update_task(task_id, logs="".join(logs))


# ── Shared helpers ──────────────────────────────────────────────────────

def _sse_chunk(
    response_id: str,
    model: str,
    delta: dict,
    finish_reason: Optional[str] = None,
    usage: Optional[dict] = None,
) -> str:
    """Build an OpenAI-compatible SSE data line."""
    chunk: dict[str, Any] = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


# ── Entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import threading

    # Try to import GUI (PySide6); fall back to headless if unavailable
    try:
        from gui import Dashboard, STYLESHEET  # noqa: F811
        from gui import QtLogHandler, log_signal  # noqa: F811
        from gui import main as gui_main
        HAS_GUI = True
    except ImportError:
        HAS_GUI = False

    if HAS_GUI:
        import uvicorn
        from uvicorn import Config, Server

        # Run uvicorn in background thread, GUI in main thread
        server = Server(
            Config(
                app,
                host=settings.host,
                port=settings.port,
                reload=False,           # reload forks -> incompatible with GUI thread
                log_level="info",
            )
        )

        t = threading.Thread(target=server.run, daemon=True)
        t.start()

        # GUI blocks until window closed
        gui_main()
    else:
        import uvicorn

        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            reload=settings.debug,
        )
