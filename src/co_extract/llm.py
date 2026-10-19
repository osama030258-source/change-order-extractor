"""LLM providers behind one tiny interface. Switch with LLM_PROVIDER=groq|anthropic."""
from __future__ import annotations

import os
import time
from typing import Callable, Optional, Union

import requests


class LLMError(RuntimeError):
    pass


class BaseLLM:
    name = "base"

    def complete_json(self, system: str, user: str) -> str:  # pragma: no cover
        raise NotImplementedError


def _post_with_retry(url: str, headers: dict, payload: dict, retries: int = 4, timeout: int = 120) -> dict:
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            if attempt == retries:
                raise LLMError(f"network error: {exc.__class__.__name__}") from exc
            time.sleep(delay)
            delay *= 2
            continue
        if resp.status_code in (429, 500, 502, 503, 529) and attempt < retries:
            try:
                wait = float(resp.headers.get("retry-after", delay))
            except ValueError:
                wait = delay
            time.sleep(min(wait, 60))
            delay *= 2
            continue
        if resp.status_code >= 400:
            raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()
    raise LLMError("retries exhausted")  # pragma: no cover


class GroqLLM(BaseLLM):
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        if not self.api_key:
            raise LLMError("GROQ_API_KEY is not set. Put it in .env (never commit it).")
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.name = f"groq:{self.model}"

    def complete_json(self, system: str, user: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        # Seen on a real 13-page PDF: Groq's JSON mode sometimes returns HTTP 400 json_validate_failed
        # (the model's output was not valid JSON). Retry, then drop JSON mode: our own parser handles the rest.
        data = None
        for attempt in range(3):
            try:
                data = _post_with_retry(self.URL, headers, payload)
                break
            except LLMError as exc:
                if "json_validate_failed" not in str(exc):
                    raise
                if attempt == 1:
                    payload = {k: v for k, v in payload.items() if k != "response_format"}
        if data is None:
            raise LLMError("Groq kept failing JSON validation")
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("unexpected Groq response shape") from exc


class AnthropicLLM(BaseLLM):
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set. Put it in .env (never commit it).")
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        self.name = f"anthropic:{self.model}"

    def complete_json(self, system: str, user: str) -> str:
        data = _post_with_retry(
            self.URL,
            {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            {
                "model": self.model,
                "max_tokens": 4096,
                "temperature": 0,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        try:
            return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        except (KeyError, TypeError) as exc:
            raise LLMError("unexpected Anthropic response shape") from exc


class StubLLM(BaseLLM):
    """Deterministic stand-in for tests. NOT a way to measure accuracy."""

    name = "stub"

    def __init__(self, responses: Union[list[str], Callable[[str, str], str]]):
        self._responses = responses
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if callable(self._responses):
            return self._responses(system, user)
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[idx]


def get_llm(provider: Optional[str] = None) -> BaseLLM:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover
        pass
    provider = (provider or os.getenv("LLM_PROVIDER") or "groq").lower()
    if provider == "groq":
        return GroqLLM()
    if provider == "anthropic":
        return AnthropicLLM()
    raise LLMError(f"unknown provider: {provider!r} (use groq or anthropic)")