"""End-to-end pipeline behaviour with a fake LLM - no network, no audio.

This covers the wiring that the unit tests above cannot: that a question
reaches the model, that tokens stream out as events, that latency is measured
on the real path, and that an API failure is reported instead of crashing.
"""

import threading
import time

from ai.client import StreamHandle
from core.events import (
    AnswerChunk,
    AnswerCompleted,
    ErrorEvent,
    LatencyReport,
    QuestionDetected,
)
from core.pipeline import Pipeline


class FakeLLM:
    """Stands in for LLMClient. Streams a canned answer one word at a time."""

    def __init__(self, answer="Overfitting is when a model memorises the training set.",
                 fail_with=None):
        self.answer = answer
        self.fail_with = fail_with
        self.calls = []
        self.model = "fake"

    def describe(self):
        return "fake"

    def set_model(self, model):
        self.model = model

    def stream(self, question, context_messages=None, mode=None,
               on_start=None, on_chunk=None, on_done=None, on_error=None):
        self.calls.append({"question": question, "context": context_messages or [],
                           "mode": mode})
        handle = StreamHandle(len(self.calls))

        def run():
            try:
                if on_start:
                    on_start(handle.request_id)
                if self.fail_with:
                    if on_error:
                        on_error(self.fail_with)
                    return
                for word in self.answer.split():
                    if handle.cancelled:
                        break
                    piece = word + " "
                    handle.text += piece
                    if on_chunk:
                        on_chunk(piece, handle.request_id)
                    time.sleep(0.002)
            finally:
                handle._finish()
                if on_done:
                    on_done(handle.text, handle.cancelled)

        threading.Thread(target=run, daemon=True).start()
        return handle


def _collect(pipeline):
    received = []
    pipeline.bus.subscribe(received.append)
    return received


def _wait_for_completion(received, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(isinstance(e, AnswerCompleted) for e in received):
            return True
        time.sleep(0.01)
    return False


def test_question_streams_an_answer_and_reports_latency(settings):
    pipeline = Pipeline(settings)
    fake = FakeLLM()
    pipeline.llm = fake
    received = _collect(pipeline)

    assert pipeline.submit_text("What is overfitting?")
    assert _wait_for_completion(received)

    assert [e for e in received if isinstance(e, QuestionDetected)]
    chunks = [e for e in received if isinstance(e, AnswerChunk)]
    assert len(chunks) > 3, "answer should arrive in pieces, not all at once"

    completed = [e for e in received if isinstance(e, AnswerCompleted)][0]
    assert "Overfitting" in completed.full_text

    report = [e for e in received if isinstance(e, LatencyReport)][0]
    assert report.llm_first_token_ms > 0
    assert lat_marks(report) >= 4
    pipeline.shutdown()


def lat_marks(report):
    return len(report.marks)


def test_follow_up_receives_previous_exchange_as_context(settings):
    pipeline = Pipeline(settings)
    fake = FakeLLM()
    pipeline.llm = fake
    received = _collect(pipeline)

    pipeline.submit_text("What is a random forest?")
    assert _wait_for_completion(received)
    received.clear()

    pipeline.submit_text("How is that different from a decision tree?")
    assert _wait_for_completion(received)

    context = fake.calls[1]["context"]
    assert any("random forest" in m["content"].lower() for m in context)
    pipeline.shutdown()


def test_clearing_the_conversation_drops_context(settings):
    pipeline = Pipeline(settings)
    fake = FakeLLM()
    pipeline.llm = fake
    received = _collect(pipeline)

    pipeline.submit_text("What is a random forest?")
    assert _wait_for_completion(received)
    pipeline.clear_conversation()
    received.clear()

    pipeline.submit_text("What is bagging?")
    assert _wait_for_completion(received)
    assert fake.calls[1]["context"] == []
    pipeline.shutdown()


def test_api_failure_surfaces_as_an_error_not_a_crash(settings):
    pipeline = Pipeline(settings)
    pipeline.llm = FakeLLM(fail_with="Rate limited by the API.")
    received = _collect(pipeline)

    pipeline.submit_text("What is overfitting?")
    assert _wait_for_completion(received)

    errors = [e for e in received if isinstance(e, ErrorEvent)]
    assert errors and "Rate limited" in errors[0].message
    # The pipeline must still accept the next question.
    assert pipeline.submit_text("What is bagging?")
    pipeline.shutdown()


def test_a_new_question_cancels_the_one_in_flight(settings):
    pipeline = Pipeline(settings)
    # A long answer gives us time to interrupt it.
    pipeline.llm = FakeLLM(answer=" ".join(["token"] * 400))
    received = _collect(pipeline)

    pipeline.submit_text("What is overfitting?")
    time.sleep(0.05)
    pipeline.submit_text("Actually, what is underfitting?")

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        completed = [e for e in received if isinstance(e, AnswerCompleted)]
        if len(completed) >= 2:
            break
        time.sleep(0.01)

    completed = [e for e in received if isinstance(e, AnswerCompleted)]
    assert any(e.cancelled for e in completed), "the first answer should be cancelled"
    pipeline.shutdown()


def test_empty_input_is_ignored(settings):
    pipeline = Pipeline(settings)
    pipeline.llm = FakeLLM()
    assert not pipeline.submit_text("   ")
    assert not pipeline.submit_text("")
    pipeline.shutdown()


def test_gated_submission_rejects_small_talk(settings):
    pipeline = Pipeline(settings)
    pipeline.llm = FakeLLM()
    assert not pipeline.submit_text("okay that makes sense", force=False)
    assert pipeline.submit_text("What is overfitting?", force=False)
    pipeline.shutdown()
