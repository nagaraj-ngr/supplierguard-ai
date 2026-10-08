"""Chat model boundary. Approaches depend on this protocol, never on a vendor SDK directly."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import List, Protocol

from .config import Settings


class LLMError(RuntimeError):
    """The model call failed (network, auth, quota). Approaches turn this into a flagged partial report."""


class MissingApiKey(LLMError):
    pass


@dataclass(frozen=True)
class ChatResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class ChatModel(Protocol):
    name: str

    def complete(self, system: str, user: str) -> ChatResponse: ...


class OpenAIChatModel:
    def __init__(self, model: str, api_key: str):
        from openai import OpenAI

        self.name = f"openai-{model}"
        self._model = model
        self._client = OpenAI(api_key=api_key)

    def complete(self, system: str, user: str) -> ChatResponse:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
        except Exception as exc:  # SDK raises many types; all become one LLMError for the approaches
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        usage = resp.usage
        return ChatResponse(
            text=resp.choices[0].message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


def get_chat_model(settings: Settings) -> ChatModel:
    if not settings.has_openai_key:
        raise MissingApiKey("OPENAI_API_KEY is not set. Add it to .env to run the LLM-based approaches.")
    return OpenAIChatModel(settings.llm_model, settings.openai_api_key)


@dataclass
class RunStats:
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    errors: List[str] = field(default_factory=list)


class Meter:
    """Wraps a ChatModel to count calls, tokens and time. Safe to share across parallel agents."""

    def __init__(self, model: ChatModel):
        self._model = model
        self._lock = threading.Lock()
        self.stats = RunStats()

    def complete(self, system: str, user: str) -> str:
        start = time.perf_counter()
        try:
            resp = self._model.complete(system, user)
        finally:
            elapsed = time.perf_counter() - start
            with self._lock:
                self.stats.llm_calls += 1
                self.stats.seconds += elapsed
        with self._lock:
            self.stats.input_tokens += resp.input_tokens
            self.stats.output_tokens += resp.output_tokens
        return resp.text

    def error(self, message: str) -> None:
        with self._lock:
            self.stats.errors.append(message)
