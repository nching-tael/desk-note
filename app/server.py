"""FastAPI app: serves the single-page UI and the JSON API."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .agent import run_agent
from .analytics import Analytics, AnalyticsHolder

log = logging.getLogger(__name__)
STATIC = Path(__file__).resolve().parent / "static"
MISSING_KEY = "Add ANTHROPIC_API_KEY to .env to start chatting."


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=60)


def _error(status: int, message: str, code: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message, "code": code})


def _has_credentials() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def _api_error_message(exc: Exception) -> tuple[int, str, str]:
    import anthropic

    if isinstance(exc, anthropic.AuthenticationError):
        return 401, "Anthropic rejected the API key. Check ANTHROPIC_API_KEY in .env.", "auth"
    if isinstance(exc, anthropic.RateLimitError):
        return 429, "Rate limited by the Anthropic API. Wait a moment and try again.", "rate_limit"
    if isinstance(exc, anthropic.NotFoundError):
        return 400, "The model in ANTHROPIC_MODEL wasn't found. Check the model name.", "model"
    if isinstance(exc, anthropic.APIConnectionError):
        return 502, "Couldn't reach the Anthropic API. Check your connection.", "connection"
    if isinstance(exc, anthropic.APIStatusError):
        return 502, f"The Anthropic API returned an error ({exc.status_code}). Try again.", "api"
    return 500, "Something went wrong while answering. Try again.", "internal"


def create_app(mock: bool = False, analytics: Analytics | None = None,
               client_factory: Callable[[], Any] | None = None) -> FastAPI:
    """``client_factory`` lets tests inject a fake Anthropic client."""
    app = FastAPI(title="Desk Note", docs_url=None, redoc_url=None)
    holder = AnalyticsHolder(mock=mock, analytics=analytics)

    def get_analytics() -> Analytics:
        return holder.get().load()

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/overview")
    def overview():
        try:
            a = get_analytics()
            data = a.overview()
            data["chart"] = a.chart_portfolio_vs_market("1m")
        except Exception as exc:
            log.exception("overview failed")
            return _error(503, f"Couldn't load portfolio data: {exc}", "data")
        data["mock"] = a.provider.is_mock
        data["chat_enabled"] = client_factory is not None or _has_credentials()
        return data

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        if client_factory is None and not _has_credentials():
            return _error(503, MISSING_KEY, "missing_api_key")
        try:
            a = get_analytics()
        except Exception as exc:
            log.exception("analytics failed")
            return _error(503, f"Couldn't load portfolio data: {exc}", "data")
        messages = [m.model_dump() for m in req.messages]
        try:
            client = client_factory() if client_factory else None
            return run_agent(messages, a, client=client)
        except ValueError as exc:
            return _error(400, str(exc), "bad_request")
        except Exception as exc:
            log.exception("chat failed")
            status, message, code = _api_error_message(exc)
            return _error(status, message, code)

    @app.get("/api/health")
    def health():
        return {"ok": True, "mock": mock}

    return app
