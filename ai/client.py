"""Streaming LLM client with cancellation.

Two things this file exists to guarantee:

1. **Tokens reach the UI as they arrive.** The whole perceived-latency argument
   collapses if we buffer the response. The callback fires per chunk.
2. **A request can be abandoned instantly.** When the speaker corrects
   themselves mid-question we must stop rendering the stale answer *now*. The
   cancel flag is checked on every chunk, so cancellation takes effect within
   one token rather than waiting for the HTTP response to finish.

Providers are behind one tiny interface. OpenAI is the default because of its
low time-to-first-token on the mini models; Anthropic is supported because
Claude Haiku is comparably fast and some people already have that key.
"""

from __future__ import annotations

import logging
import threading
import time

from config.settings import Settings

from .prompts import build_messages

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised with a message that is safe and useful to show in the UI."""


class StreamHandle:
    """Returned by `LLMClient.stream()`. Cancel it or wait on it."""

    def __init__(self, request_id: int) -> None:
        self.request_id = request_id
        self._cancelled = threading.Event()
        self._done = threading.Event()
        self.text = ""
        self.error = ""

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    @property
    def finished(self) -> bool:
        return self._done.is_set()

    def cancel(self) -> None:
        self._cancelled.set()

    def wait(self, timeout: float = None) -> bool:
        return self._done.wait(timeout)

    def _finish(self) -> None:
        self._done.set()


class LLMClient:
    """Thread-per-request. Requests are short-lived and rare (one per question),
    so a pool would be complexity without benefit."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None
        self._client_provider = ""
        self._client_lock = threading.Lock()
        self._request_counter = 0

    # ------------------------------------------------------------------
    def describe(self) -> str:
        s = self.settings
        if s.llm_base_url:
            host = s.llm_base_url.split("//")[-1].split("/")[0]
            return "%s @ %s" % (s.llm_model, host)
        return "%s / %s" % (s.llm_provider, s.llm_model)

    def set_model(self, model: str) -> None:
        self.settings.llm_model = model

    def preflight(self) -> str:
        """Cheap configuration check for `--check`. Returns a human message."""
        s = self.settings
        if s.llm_provider == "openai" and not s.openai_api_key:
            raise LLMError(
                "OPENAI_API_KEY is missing. Copy .env.example to .env and put "
                "your key in it."
            )
        if s.llm_provider == "anthropic" and not s.anthropic_api_key:
            raise LLMError(
                "ANTHROPIC_API_KEY is missing. Copy .env.example to .env and put "
                "your key in it."
            )
        self._ensure_client()
        return "LLM configured: %s" % self.describe()

    def _ensure_client(self):
        with self._client_lock:
            if self._client is not None and self._client_provider == self.settings.llm_provider:
                return self._client
            s = self.settings
            # Check the key here rather than letting the SDK raise: this path
            # runs on the LLM thread, where a clean LLMError becomes a one-line
            # message in the UI instead of a traceback in the console.
            if s.llm_provider == "anthropic" and not s.anthropic_api_key:
                raise LLMError(
                    "ANTHROPIC_API_KEY is missing. Copy .env.example to .env and "
                    "put your key in it."
                )
            if s.llm_provider == "openai" and not s.openai_api_key:
                raise LLMError(
                    "OPENAI_API_KEY is missing. Copy .env.example to .env and "
                    "put your key in it."
                )
            if s.llm_provider == "anthropic":
                try:
                    from anthropic import Anthropic
                except Exception as exc:
                    raise LLMError(
                        "The 'anthropic' package is not installed (%s). "
                        "Run: pip install anthropic" % exc
                    ) from exc
                self._client = Anthropic(api_key=s.anthropic_api_key, timeout=s.llm_timeout)
            else:
                try:
                    from openai import OpenAI
                except Exception as exc:
                    raise LLMError(
                        "The 'openai' package is not installed (%s). "
                        "Run: pip install openai" % exc
                    ) from exc
                kwargs = {"api_key": s.openai_api_key, "timeout": s.llm_timeout}
                if s.llm_base_url:
                    # Groq, Cerebras, Together, a local Ollama - anything that
                    # speaks the OpenAI chat-completions protocol.
                    kwargs["base_url"] = s.llm_base_url
                self._client = OpenAI(**kwargs)
            self._client_provider = s.llm_provider
            return self._client

    # ------------------------------------------------------------------
    def stream(self, question: str, context_messages: list = None, mode: str = None,
               on_start=None, on_chunk=None, on_done=None, on_error=None) -> StreamHandle:
        """Start a streaming completion on a background thread.

        Callbacks run on that thread - keep them cheap and thread-safe.
        `on_done(full_text, cancelled)` always fires exactly once, including on
        error and on cancellation, so the caller can always clean up.
        """
        self._request_counter += 1
        handle = StreamHandle(self._request_counter)
        mode = mode or self.settings.answer_mode
        messages = build_messages(question, context_messages, mode)

        def run() -> None:
            try:
                if on_start:
                    on_start(handle.request_id)
                client = self._ensure_client()
                if self.settings.llm_provider == "anthropic":
                    self._stream_anthropic(client, messages, handle, on_chunk)
                else:
                    self._stream_openai(client, messages, handle, on_chunk)
            except LLMError as exc:
                handle.error = str(exc)
                log.error("LLM request failed: %s", exc)
                if on_error:
                    on_error(str(exc))
            except Exception as exc:
                handle.error = friendly_error(exc)
                log.error("LLM request failed: %s", handle.error)
                log.debug("LLM traceback", exc_info=True)
                if on_error:
                    on_error(handle.error)
            finally:
                handle._finish()
                if on_done:
                    on_done(handle.text, handle.cancelled)

        threading.Thread(
            target=run, name="llm-%d" % handle.request_id, daemon=True
        ).start()
        return handle

    # ------------------------------------------------------------------
    def _stream_openai(self, client, messages, handle: StreamHandle, on_chunk) -> None:
        s = self.settings
        started = time.perf_counter()
        stream = client.chat.completions.create(
            model=s.llm_model,
            messages=messages,
            stream=True,
            max_tokens=s.llm_max_tokens,
            temperature=0.3,          # low: we want accuracy, not creativity
        )
        try:
            for event in stream:
                if handle.cancelled:
                    log.info("Request %d cancelled mid-stream", handle.request_id)
                    break
                if not event.choices:
                    continue
                piece = event.choices[0].delta.content or ""
                if piece:
                    handle.text += piece
                    if on_chunk:
                        on_chunk(piece, handle.request_id)
        finally:
            # Closing the stream releases the connection immediately, which is
            # what makes cancellation actually free up the socket.
            try:
                stream.close()
            except Exception:
                pass
        log.debug("OpenAI stream finished in %.0f ms (%d chars)",
                  (time.perf_counter() - started) * 1000, len(handle.text))

    def _stream_anthropic(self, client, messages, handle: StreamHandle, on_chunk) -> None:
        s = self.settings
        # Anthropic takes the system prompt as a separate argument.
        system = ""
        chat = []
        for message in messages:
            if message["role"] == "system":
                system = message["content"]
            else:
                chat.append(message)

        with client.messages.stream(
            model=s.llm_model,
            system=system,
            messages=chat,
            max_tokens=s.llm_max_tokens,
            temperature=0.3,
        ) as stream:
            for piece in stream.text_stream:
                if handle.cancelled:
                    log.info("Request %d cancelled mid-stream", handle.request_id)
                    break
                if piece:
                    handle.text += piece
                    if on_chunk:
                        on_chunk(piece, handle.request_id)


def friendly_error(exc: Exception) -> str:
    """Turn an SDK exception into something worth showing in a 400px window."""
    name = type(exc).__name__
    text = str(exc)
    lowered = text.lower()

    if "missing credentials" in lowered or "api_key" in lowered:
        return "API key missing. Put OPENAI_API_KEY in the .env file next to app.py."
    if "api key" in lowered or "authentication" in lowered or name == "AuthenticationError":
        return "API key rejected. Check OPENAI_API_KEY / ANTHROPIC_API_KEY in .env."
    if "rate limit" in lowered or name == "RateLimitError":
        return "Rate limited by the API. Wait a moment or switch to a smaller model."
    if "insufficient_quota" in lowered or "quota" in lowered:
        return "Your API account is out of quota. Check billing on the provider's dashboard."
    if "model" in lowered and ("not found" in lowered or "does not exist" in lowered):
        return "Model not available on this account. Set a different LLM_MODEL in .env."
    if name in ("APITimeoutError", "Timeout") or "timeout" in lowered or "timed out" in lowered:
        return "The model took too long to respond. Try again or use a faster model."
    if name == "APIConnectionError" or "connection" in lowered or "getaddrinfo" in lowered:
        return "Cannot reach the API. Check your internet connection."
    if "overloaded" in lowered or name == "InternalServerError":
        return "The provider is overloaded right now. Try again in a few seconds."
    # Last resort: show something, but keep it to one line.
    return "%s: %s" % (name, text.splitlines()[0][:160] if text else "unknown error")
