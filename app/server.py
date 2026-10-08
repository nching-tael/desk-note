"""Web app: serves the chat page and a small JSON API."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import anthropic
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .agent import run_agent
from .analytics import AnalyticsHolder

log = logging.getLogger(__name__)

INDEX = Path(__file__).parent / "static" / "index.html"
MISSING_KEY = "Add ANTHROPIC_API_KEY to .env to start chatting."


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20_000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=60)


def error(status, message, code):
    return JSONResponse(status_code=status, content={"error": message, "code": code})


def has_api_key():
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def explain_api_error(e):
    """Map Anthropic SDK errors to (status, message for the user, code)."""
    if isinstance(e, anthropic.AuthenticationError):
        return 401, "Anthropic rejected the API key. Check ANTHROPIC_API_KEY in .env.", "auth"
    if isinstance(e, anthropic.RateLimitError):
        return 429, "Rate limited by the Anthropic API. Wait a moment and try again.", "rate_limit"
    if isinstance(e, anthropic.NotFoundError):
        return 400, "The model in ANTHROPIC_MODEL wasn't found. Check the model name.", "model"
    if isinstance(e, anthropic.APIConnectionError):
        return 502, "Couldn't reach the Anthropic API. Check your connection.", "connection"
    if isinstance(e, anthropic.APIStatusError):
        return 502, f"The Anthropic API returned an error ({e.status_code}). Try again.", "api"
    return 500, "Something went wrong while answering. Try again.", "internal"


def create_app(mock=False, analytics=None, client_factory=None):
    """client_factory lets tests swap in a fake Anthropic client."""
    app = FastAPI(title="Desk Note", docs_url=None, redoc_url=None)
    holder = AnalyticsHolder(mock=mock, analytics=analytics)
    chat_enabled = client_factory is not None or has_api_key()

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(INDEX, headers={"Cache-Control": "no-cache"})

    @app.get("/api/overview")
    def overview():
        try:
            analytics = holder.get()
            data = analytics.overview()
            data["chart"] = analytics.chart_portfolio_vs_market("1m")
        except Exception as e:
            log.exception("overview failed")
            return error(503, f"Couldn't load portfolio data: {e}", "data")
        data["mock"] = analytics.provider.is_mock
        data["chat_enabled"] = chat_enabled
        return data

    @app.post("/api/chat")
    def chat(request: ChatRequest):
        if not chat_enabled:
            return error(503, MISSING_KEY, "missing_api_key")
        try:
            analytics = holder.get()
        except Exception as e:
            log.exception("loading analytics failed")
            return error(503, f"Couldn't load portfolio data: {e}", "data")

        messages = [m.model_dump() for m in request.messages]
        try:
            client = client_factory() if client_factory else None
            return run_agent(messages, analytics, client=client)
        except ValueError as e:
            return error(400, str(e), "bad_request")
        except Exception as e:
            log.exception("chat failed")
            return error(*explain_api_error(e))

    @app.get("/api/health")
    def health():
        return {"ok": True, "mock": mock}

    return app
