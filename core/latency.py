"""Latency instrumentation.

Every question gets one LatencyTracker. Stages call `mark()` as they happen and
the UI reads a summary when the answer completes, so the numbers you see are
measured, not estimated.

`time.perf_counter()` is used rather than `time.time()` because it is monotonic
and has much better resolution - wall-clock adjustments cannot make a stage
appear to take negative time.
"""

from __future__ import annotations

import time

from .events import LatencyReport

# Canonical mark names (the spec's list in section 17).
AUDIO_DETECTED = "audio_detected"
SPEECH_STARTED = "speech_started"
PARTIAL_TRANSCRIPT = "partial_transcript"
FINAL_TRANSCRIPT = "final_transcript"
QUESTION_DETECTED = "question_detected"
LLM_REQUEST_STARTED = "llm_request_started"
LLM_FIRST_TOKEN = "llm_first_token"
LLM_COMPLETED = "llm_completed"
ANSWER_RENDERED = "answer_rendered"

ORDER = [
    AUDIO_DETECTED,
    SPEECH_STARTED,
    PARTIAL_TRANSCRIPT,
    FINAL_TRANSCRIPT,
    QUESTION_DETECTED,
    LLM_REQUEST_STARTED,
    LLM_FIRST_TOKEN,
    LLM_COMPLETED,
    ANSWER_RENDERED,
]


class LatencyTracker:
    def __init__(self, name: str = "") -> None:
        self.name = name
        self.marks: dict = {}
        self.created = time.perf_counter()

    # -- recording ---------------------------------------------------------
    def mark(self, name: str, overwrite: bool = False) -> float:
        """Record a timestamp. First write wins unless `overwrite=True`.

        First-write-wins matters for PARTIAL_TRANSCRIPT: we want the time of the
        *first* partial, not the last one.
        """
        now = time.perf_counter()
        if overwrite or name not in self.marks:
            self.marks[name] = now
        return self.marks[name]

    def has(self, name: str) -> bool:
        return name in self.marks

    # -- reading -----------------------------------------------------------
    def delta_ms(self, start: str, end: str) -> float:
        """Milliseconds between two marks, or 0.0 if either is missing."""
        if start not in self.marks or end not in self.marks:
            return 0.0
        return (self.marks[end] - self.marks[start]) * 1000.0

    def since_ms(self, start: str) -> float:
        if start not in self.marks:
            return 0.0
        return (time.perf_counter() - self.marks[start]) * 1000.0

    def report(self) -> LatencyReport:
        """The three headline numbers.

        STT           = end of speech -> final transcript in hand
        LLM 1st token = request sent -> first streamed character
        Total         = end of speech -> first streamed character
                        (this is the number you actually feel)
        """
        end_of_speech = self.marks.get(FINAL_TRANSCRIPT)
        stt_ms = self.delta_ms(SPEECH_STARTED, FINAL_TRANSCRIPT)
        if self.has(QUESTION_DETECTED) and self.has(FINAL_TRANSCRIPT):
            # Speculative starts fire before the final transcript exists; in that
            # case measure STT up to the partial that triggered us instead.
            if self.marks[QUESTION_DETECTED] < self.marks[FINAL_TRANSCRIPT]:
                stt_ms = self.delta_ms(SPEECH_STARTED, QUESTION_DETECTED)

        first_token_ms = self.delta_ms(LLM_REQUEST_STARTED, LLM_FIRST_TOKEN)

        total_ms = 0.0
        if self.has(LLM_FIRST_TOKEN):
            anchor = self.marks.get(QUESTION_DETECTED) or end_of_speech or self.created
            total_ms = (self.marks[LLM_FIRST_TOKEN] - anchor) * 1000.0

        return LatencyReport(
            stt_ms=max(0.0, stt_ms),
            llm_first_token_ms=max(0.0, first_token_ms),
            total_ms=max(0.0, total_ms),
            marks=self.timeline(),
        )

    def timeline(self) -> dict:
        """All marks as milliseconds relative to the first one - for the log."""
        if not self.marks:
            return {}
        origin = min(self.marks.values())
        return {
            name: round((self.marks[name] - origin) * 1000.0, 1)
            for name in ORDER
            if name in self.marks
        }

    def log_line(self) -> str:
        parts = ["%s=%.0fms" % (k, v) for k, v in self.timeline().items()]
        return " ".join(parts) if parts else "(no marks)"
