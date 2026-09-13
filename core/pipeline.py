"""The real-time pipeline: audio in, streamed answer out.

Thread map (four threads, no asyncio except inside the Deepgram client):

    audio callback  -> pushes 16 kHz frames onto a bounded queue
    pipeline worker -> drains the queue, runs VAD, drives the transcriber,
                       evaluates the question gate
    stt worker      -> decodes audio, calls back with partials/finals
    llm thread      -> streams tokens, calls back per chunk

Nothing here imports Qt. The UI subscribes to the EventBus and turns events
into Qt signals; that separation is what lets `python app.py --test` run the
identical pipeline with no window at all.

Shared mutable state lives behind `self._lock` and is deliberately small: the
current utterance id, the in-flight request handle, and the partial-stability
timestamps. Everything else is thread-confined.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from ai.client import LLMClient, friendly_error
from ai.context import ConversationContext
from audio.capture import AudioCapture
from audio.devices import AudioDevice, AudioDeviceError, resolve_device
from config.settings import SAMPLE_RATE, Settings
from speech.transcriber import create_transcriber
from speech.vad import Level, SpeechChunk, SpeechEnd, SpeechStart, VoiceActivityDetector

from . import latency as lat
from .events import (
    AnswerChunk,
    AnswerCompleted,
    AnswerStarted,
    AudioLevel,
    ErrorEvent,
    EventBus,
    FinalTranscript,
    LatencyReport,
    PartialTranscript,
    QuestionDetected,
    StatusEvent,
)
from .latency import LatencyTracker
from .question import QuestionGate

log = logging.getLogger(__name__)

STATUS_IDLE = "idle"
STATUS_LISTENING = "listening"
STATUS_SPEECH = "speech"
STATUS_THINKING = "thinking"
STATUS_ANSWERING = "answering"
STATUS_READY = "ready"
STATUS_ERROR = "error"


class Pipeline:
    def __init__(self, settings: Settings, bus: EventBus = None) -> None:
        self.settings = settings
        self.bus = bus or EventBus()
        self.context = ConversationContext(max_turns=settings.context_turns)
        self.gate = QuestionGate(settings)
        self.llm = LLMClient(settings)

        self._lock = threading.RLock()
        self._running = threading.Event()
        self._worker = None
        self._capture = None
        self._transcriber = None
        self._vad = VoiceActivityDetector(settings)
        self.device = None

        # Per-utterance state (written by the worker, read by callbacks).
        self._utterance_id = 0
        self._utterance_audio = []
        self._utterance_samples = 0
        self._last_partial_request = 0.0
        self._partial_text = ""
        self._partial_changed_at = 0.0
        self._tracker = None

        # In-flight answer.
        self._handle = None
        self._answer_tracker = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._running.is_set()

    def start(self, device: AudioDevice = None) -> None:
        """Open the audio device, load the STT model, begin listening."""
        if self._running.is_set():
            return

        try:
            self.device = device or resolve_device(
                self.settings.audio_device, self.settings.audio_mode
            )
        except AudioDeviceError as exc:
            self._fail("No usable audio device: %s" % exc, fatal=True)
            return

        log.info("Selected audio device: %s", self.device)

        self._transcriber = create_transcriber(
            self.settings,
            on_partial=self._on_partial,
            on_final=self._on_final,
            on_error=lambda message: self._fail(message),
            on_ready=lambda message: self._emit(StatusEvent(STATUS_LISTENING, message)),
        )

        self._capture = AudioCapture(self.device, on_error=lambda m: self._fail(m))
        try:
            self._capture.start()
        except AudioDeviceError as exc:
            self._fail(str(exc), fatal=True)
            self._capture = None
            return

        self._vad.reset()
        self._transcriber.start()
        self._running.set()
        self._worker = threading.Thread(target=self._run, name="pipeline", daemon=True)
        self._worker.start()

        detail = "%s (%s)" % (self.device.label, self._capture.backend)
        if self.settings.stt_provider == "local":
            detail += " - loading speech model..."
        self._emit(StatusEvent(STATUS_LISTENING, detail))

    def stop(self) -> None:
        if not self._running.is_set():
            self._emit(StatusEvent(STATUS_IDLE, "Stopped"))
            return
        self._running.clear()

        self.cancel_answer(reason="stopping")

        worker, self._worker = self._worker, None
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)

        capture, self._capture = self._capture, None
        if capture is not None:
            capture.stop()

        transcriber, self._transcriber = self._transcriber, None
        if transcriber is not None:
            transcriber.stop()

        self._vad.reset()
        self._reset_utterance()
        self._emit(StatusEvent(STATUS_IDLE, "Stopped"))
        log.info("Pipeline stopped")

    def shutdown(self) -> None:
        try:
            self.stop()
        except Exception:
            log.exception("Error during shutdown")

    # ------------------------------------------------------------------
    # Settings the UI can change while running
    # ------------------------------------------------------------------
    def set_answer_mode(self, mode: str) -> None:
        self.settings.answer_mode = (mode or "NORMAL").upper()
        log.info("Answer mode -> %s", self.settings.answer_mode)

    def set_model(self, model: str) -> None:
        self.llm.set_model(model)
        log.info("LLM model -> %s", model)

    def clear_conversation(self) -> None:
        self.cancel_answer(reason="conversation cleared")
        self.context.clear()
        self.gate.reset()
        log.info("Conversation cleared")

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------
    def _run(self) -> None:
        log.info("Pipeline worker started")
        while self._running.is_set():
            try:
                block = self._capture.read(timeout=0.25) if self._capture else None
                if block is not None:
                    for event in self._vad.process(block):
                        self._handle_vad_event(event)
                self._maybe_speculate()
            except Exception:
                log.exception("Pipeline worker error")
                time.sleep(0.2)
        log.info("Pipeline worker exiting")

    def _handle_vad_event(self, event) -> None:
        if isinstance(event, Level):
            self._emit(AudioLevel(event.rms_db, event.noise_floor_db, event.is_speech))
        elif isinstance(event, SpeechStart):
            self._on_speech_start(event)
        elif isinstance(event, SpeechChunk):
            self._on_speech_chunk(event)
        elif isinstance(event, SpeechEnd):
            self._on_speech_end(event)

    # -- utterance lifecycle ----------------------------------------------
    def _on_speech_start(self, event: SpeechStart) -> None:
        with self._lock:
            self._utterance_id += 1
            uid = self._utterance_id
            self._utterance_audio = [event.audio]
            self._utterance_samples = event.audio.size
            self._partial_text = ""
            self._partial_changed_at = time.monotonic()
            self._last_partial_request = time.monotonic()
            self._tracker = LatencyTracker("utterance-%d" % uid)
            self._tracker.mark(lat.AUDIO_DETECTED)
            self._tracker.mark(lat.SPEECH_STARTED)

        self._transcriber.begin_utterance(uid)
        self._transcriber.feed(event.audio, uid)
        self._emit(StatusEvent(STATUS_SPEECH, "Listening..."))
        log.debug("Utterance %d started", uid)

    def _on_speech_chunk(self, event: SpeechChunk) -> None:
        with self._lock:
            uid = self._utterance_id
            self._utterance_audio.append(event.audio)
            self._utterance_samples += event.audio.size
            due = (
                time.monotonic() - self._last_partial_request
                >= self.settings.partial_interval
            )
            if due:
                self._last_partial_request = time.monotonic()
                audio = np.concatenate(self._utterance_audio)
            else:
                audio = None

        # Deepgram wants every frame; Whisper wants periodic whole-utterance
        # re-decodes. `feed` is a no-op for Whisper, `transcribe_partial` is a
        # no-op for Deepgram.
        self._transcriber.feed(event.audio, uid)
        if audio is not None and audio.size >= SAMPLE_RATE * 0.6:
            self._transcriber.transcribe_partial(audio, uid)

    def _on_speech_end(self, event: SpeechEnd) -> None:
        with self._lock:
            uid = self._utterance_id
            self._utterance_audio = []
            self._utterance_samples = 0
        log.debug("Utterance %d ended (%.1fs%s)", uid, event.duration,
                  ", truncated" if event.truncated else "")
        self._emit(StatusEvent(STATUS_THINKING, "Transcribing..."))
        self._transcriber.transcribe_final(event.audio, uid)

    def _reset_utterance(self) -> None:
        with self._lock:
            self._utterance_audio = []
            self._utterance_samples = 0
            self._partial_text = ""
            self._tracker = None

    # ------------------------------------------------------------------
    # Transcription callbacks (run on the STT thread)
    # ------------------------------------------------------------------
    def _on_partial(self, text: str, utterance_id: int) -> None:
        with self._lock:
            if utterance_id != self._utterance_id or not text:
                return
            if text.strip() != self._partial_text.strip():
                self._partial_text = text
                self._partial_changed_at = time.monotonic()
            if self._tracker is not None:
                self._tracker.mark(lat.PARTIAL_TRANSCRIPT)
        self._emit(PartialTranscript(text, utterance_id))

    def _on_final(self, text: str, utterance_id: int) -> None:
        with self._lock:
            tracker = self._tracker
            if tracker is not None:
                tracker.mark(lat.FINAL_TRANSCRIPT)
            current = self._utterance_id

        if not text:
            # Noise, or a hallucination we filtered out.
            self._emit(StatusEvent(STATUS_LISTENING, "Waiting for speech..."))
            return

        self._emit(FinalTranscript(text, utterance_id))
        log.info("Final transcript [%d]: %s", utterance_id, text)

        decision = self.gate.consider_final(text, has_context=self.context.has_context())
        if not decision.should_ask:
            log.info("Not asking: %s", decision.reason)
            status = (
                "Answering..." if self._has_live_request() else "Waiting for speech..."
            )
            self._emit(StatusEvent(
                STATUS_ANSWERING if self._has_live_request() else STATUS_LISTENING,
                status,
            ))
            return

        if utterance_id != current:
            log.debug("Dropping stale final for utterance %d", utterance_id)
            return

        self._ask(decision.question, tracker=tracker, speculative=False,
                  supersedes=decision.supersedes, confidence=decision.confidence,
                  utterance_id=utterance_id)

    # ------------------------------------------------------------------
    # Speculative start
    # ------------------------------------------------------------------
    def _maybe_speculate(self) -> None:
        """Called every worker tick. Fires the LLM before end-of-utterance when
        the partial transcript already reads as a finished question."""
        with self._lock:
            text = self._partial_text
            changed_at = self._partial_changed_at
            uid = self._utterance_id
            tracker = self._tracker
            in_speech = self._vad.state.in_speech
        if not text or not in_speech:
            return

        stable_for = time.monotonic() - changed_at
        decision = self.gate.consider_partial(
            text, stable_for, has_context=self.context.has_context()
        )
        if not decision.should_ask:
            return
        log.info("Speculative start (%s): %s", decision.reason, text)
        self._ask(decision.question, tracker=tracker, speculative=True,
                  supersedes=False, confidence=decision.confidence, utterance_id=uid)

    # ------------------------------------------------------------------
    # Asking the LLM
    # ------------------------------------------------------------------
    def _has_live_request(self) -> bool:
        with self._lock:
            return self._handle is not None and not self._handle.finished

    def cancel_answer(self, reason: str = "") -> None:
        with self._lock:
            handle, self._handle = self._handle, None
        if handle is not None and not handle.finished:
            handle.cancel()
            log.info("Cancelled answer %d (%s)", handle.request_id, reason or "no reason")
            self.context.abandon_exchange()
            self.gate.note_answered()

    def _ask(self, question: str, tracker: LatencyTracker = None,
             speculative: bool = False, supersedes: bool = False,
             confidence: float = 0.0, utterance_id: int = 0) -> None:
        if supersedes and self._has_live_request():
            self.cancel_answer(reason="superseded by a new question")

        tracker = tracker or LatencyTracker("manual")
        tracker.mark(lat.QUESTION_DETECTED, overwrite=True)

        self.gate.note_asked(question, speculative=speculative)
        self.context.start_exchange(question)
        self._emit(QuestionDetected(question, utterance_id, speculative, confidence))
        self._emit(StatusEvent(STATUS_THINKING, "Thinking..."))

        context_messages = self.context.as_messages()
        # Shared between the callbacks, which run on the LLM thread before
        # `self._handle` has necessarily been assigned below.
        state = {"request_id": 0}

        def on_start(request_id: int) -> None:
            state["request_id"] = request_id
            tracker.mark(lat.LLM_REQUEST_STARTED, overwrite=True)
            self._emit(AnswerStarted(question, request_id))

        def on_chunk(piece: str, request_id: int) -> None:
            if not tracker.has(lat.LLM_FIRST_TOKEN):
                tracker.mark(lat.LLM_FIRST_TOKEN)
                self._emit(StatusEvent(STATUS_ANSWERING, "Answering..."))
            self._emit(AnswerChunk(piece, request_id))

        def on_error(message: str) -> None:
            self._emit(ErrorEvent(message))
            self._emit(StatusEvent(STATUS_ERROR, message))

        def on_done(full_text: str, cancelled: bool) -> None:
            tracker.mark(lat.LLM_COMPLETED)
            tracker.mark(lat.ANSWER_RENDERED)
            request_id = state["request_id"]
            with self._lock:
                handle = self._handle
                if handle is not None and handle.request_id == request_id:
                    self._handle = None
            if cancelled:
                self.context.abandon_exchange()
            else:
                self.context.complete_exchange(full_text)
                self.gate.note_answered()
            self._emit(AnswerCompleted(request_id, full_text, cancelled))
            if not cancelled and tracker.has(lat.LLM_FIRST_TOKEN):
                # No first token means the request failed - an all-zero latency
                # report would be noise on top of the error message.
                report = tracker.report()
                log.info("Latency: %s | %s", report.as_line(), tracker.log_line())
                self._emit(report)
                self._emit(StatusEvent(
                    STATUS_READY if self._running.is_set() else STATUS_IDLE,
                    "Ready" if self._running.is_set() else "Stopped",
                ))

        try:
            handle = self.llm.stream(
                question,
                context_messages=context_messages,
                mode=self.settings.answer_mode,
                on_start=on_start,
                on_chunk=on_chunk,
                on_done=on_done,
                on_error=on_error,
            )
        except Exception as exc:
            message = friendly_error(exc)
            self._emit(ErrorEvent(message, detail=str(exc)))
            self._emit(StatusEvent(STATUS_ERROR, message))
            return

        with self._lock:
            self._handle = handle
            self._answer_tracker = tracker

    # ------------------------------------------------------------------
    # Manual entry - used by --test and by typing into the UI
    # ------------------------------------------------------------------
    def submit_text(self, text: str, force: bool = True) -> bool:
        """Push a question straight into the LLM stage, bypassing audio.

        `force=False` runs it through the question gate instead, which is how
        you check the detection logic against real phrasings.
        """
        # Strip the BOM and zero-width characters that arrive when text is
        # pasted or piped in on Windows.
        text = (text or "").replace("﻿", "").replace("​", "").strip()
        if not text:
            return False

        tracker = LatencyTracker("manual")
        tracker.mark(lat.AUDIO_DETECTED)
        tracker.mark(lat.SPEECH_STARTED)
        tracker.mark(lat.FINAL_TRANSCRIPT)
        self._emit(FinalTranscript(text, -1))

        if not force:
            decision = self.gate.consider_final(
                text, has_context=self.context.has_context()
            )
            if not decision.should_ask:
                log.info("Manual text rejected by gate: %s", decision.reason)
                self._emit(StatusEvent(STATUS_READY, "Not a question: %s" % decision.reason))
                return False
            text = decision.question

        self._ask(text, tracker=tracker, supersedes=True, confidence=1.0, utterance_id=-1)
        return True

    # ------------------------------------------------------------------
    def _emit(self, event) -> None:
        self.bus.emit(event)

    def _fail(self, message: str, fatal: bool = False) -> None:
        log.error("%s", message)
        self._emit(ErrorEvent(message, fatal=fatal))
        self._emit(StatusEvent(STATUS_ERROR, message.splitlines()[0]))
