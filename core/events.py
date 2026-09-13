"""Event types and a tiny thread-safe event bus.

The pipeline runs on background threads; the UI lives on the Qt main thread.
Rather than letting those threads touch each other's objects, every stage
publishes plain dataclasses here and the UI subscribes once and re-emits them
as Qt signals. That keeps `core/` completely free of Qt imports, which is also
what makes the headless test mode (`python app.py --test`) possible.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Event payloads
# --------------------------------------------------------------------------

@dataclass
class StatusEvent:
    """Top-level state shown in the title bar pill."""
    status: str          # idle | listening | speech | transcribing | thinking | answering | error
    detail: str = ""


@dataclass
class PartialTranscript:
    text: str
    utterance_id: int


@dataclass
class FinalTranscript:
    text: str
    utterance_id: int
    duration: float = 0.0


@dataclass
class QuestionDetected:
    text: str
    utterance_id: int
    speculative: bool = False
    confidence: float = 0.0


@dataclass
class AnswerStarted:
    question: str
    request_id: int


@dataclass
class AnswerChunk:
    text: str
    request_id: int


@dataclass
class AnswerCompleted:
    request_id: int
    full_text: str
    cancelled: bool = False


@dataclass
class LatencyReport:
    stt_ms: float = 0.0
    llm_first_token_ms: float = 0.0
    total_ms: float = 0.0
    marks: dict = field(default_factory=dict)

    def as_line(self) -> str:
        return "STT: %d ms   LLM first token: %d ms   Total: %.2f s" % (
            round(self.stt_ms),
            round(self.llm_first_token_ms),
            self.total_ms / 1000.0,
        )


@dataclass
class ErrorEvent:
    """A recoverable problem. The UI shows `message`; `detail` goes to the log."""
    message: str
    detail: str = ""
    fatal: bool = False


@dataclass
class AudioLevel:
    """Emitted ~10x/second so the UI can draw a level meter."""
    rms_db: float
    noise_floor_db: float
    is_speech: bool


# --------------------------------------------------------------------------
# Bus
# --------------------------------------------------------------------------

class EventBus:
    """Synchronous publish/subscribe.

    Handlers run on whichever thread emitted the event, so handlers must be
    cheap and thread-safe. The UI's handler does nothing but emit a Qt signal,
    which Qt queues onto the main thread for us.
    """

    def __init__(self) -> None:
        self._subscribers: list = []
        self._lock = threading.Lock()

    def subscribe(self, handler: Callable[[object], None]) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(handler)

        def unsubscribe() -> None:
            with self._lock:
                if handler in self._subscribers:
                    self._subscribers.remove(handler)

        return unsubscribe

    def emit(self, event: object) -> None:
        with self._lock:
            handlers = list(self._subscribers)
        for handler in handlers:
            try:
                handler(event)
            except Exception:  # never let a bad subscriber kill the pipeline
                log.exception("Event handler failed for %s", type(event).__name__)
