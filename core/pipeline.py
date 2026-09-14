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
from speech.vad import (
    Level,
    SpeechChunk,
    SpeechEnd,
    SpeechPause,
    SpeechStart,
    VoiceActivityDetector,
)

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

# How long end-of-utterance will defer to an in-flight pause decode before
# giving up and decoding itself.
_PREFINAL_TIMEOUT = 6.0


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
        # Transcript decoded during a pause; reusable as the final one if the
        # speaker never resumed. See _on_prefinal / _on_speech_end.
        self._prefinal = None            # (uid, speech_samples, text) - arrived
        self._prefinal_pending = None    # (uid, speech_samples) - still decoding
        self._awaiting_prefinal = None   # (uid, speech_samples, audio, deadline)
        self._eager_fired = -1           # utterance we already answered early

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
            on_prefinal=self._on_prefinal,
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

        # Open the HTTPS connection to the model now, on a throwaway thread,
        # so the first real question does not pay for DNS + TLS. Guarded
        # because `llm` is swappable - tests substitute their own client.
        warmup = getattr(self.llm, "warmup", None)
        if callable(warmup):
            threading.Thread(target=warmup, name="llm-warmup", daemon=True).start()
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
                self._check_prefinal_timeout()
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
        elif isinstance(event, SpeechPause):
            self._on_speech_pause(event)
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
            self._prefinal = None
            self._prefinal_pending = None
            self._awaiting_prefinal = None
            self._eager_fired = -1
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
            # No periodic partial while the speaker is pausing: the pause decode
            # covers exactly the same audio at final quality, and a competing
            # partial would only delay it.
            in_pause = self._vad.state.silence_frames > 0
            due = (
                not in_pause
                and time.monotonic() - self._last_partial_request
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

    def _on_speech_pause(self, event: SpeechPause) -> None:
        """The speaker went quiet. Start decoding now instead of waiting.

        Everything from here to end-of-utterance is silence, so this decode
        produces the same text the final one would - it just runs during the
        END_OF_UTTERANCE_SILENCE window rather than after it. On this machine
        that takes ~390 ms of Whisper off the critical path entirely.
        """
        if not getattr(self._transcriber, "supports_prefinal", False):
            # Streaming providers do their own endpointing and never run a
            # pause decode. Recording one as pending here made end-of-utterance
            # wait the full timeout for a result that was never coming - the
            # "Pause decode did not return in time" warnings in the log.
            return
        with self._lock:
            uid = self._utterance_id
            self._prefinal = None
            self._prefinal_pending = (uid, event.speech_samples)
        self._transcriber.transcribe_prefinal(event.audio, uid, event.speech_samples)

    def _on_prefinal(self, text: str, utterance_id: int, speech_samples: int) -> None:
        """A pause decode came back. Three jobs: show it, maybe answer now, and
        satisfy end-of-utterance if it is already waiting on us."""
        with self._lock:
            if utterance_id != self._utterance_id:
                return
            self._prefinal = (utterance_id, speech_samples, text)
            self._prefinal_pending = None
            if text and text.strip() != self._partial_text.strip():
                self._partial_text = text
                self._partial_changed_at = time.monotonic()
            if self._tracker is not None:
                self._tracker.mark(lat.PARTIAL_TRANSCRIPT)
            waiting = self._awaiting_prefinal
            if waiting is not None and waiting[0] == utterance_id \
                    and waiting[1] == speech_samples:
                self._awaiting_prefinal = None
            else:
                waiting = None

        if waiting is not None:
            # End-of-utterance already happened and deferred to this decode.
            log.info("Pause transcript satisfied end-of-utterance for %d", utterance_id)
            self._on_final(text, utterance_id)
            return

        if text:
            self._emit(PartialTranscript(text, utterance_id))
            self._maybe_speculate(text, utterance_id, speech_samples)

    def _on_speech_end(self, event: SpeechEnd) -> None:
        with self._lock:
            uid = self._utterance_id
            self._utterance_audio = []
            self._utterance_samples = 0
            prefinal = self._prefinal
            pending = self._prefinal_pending
            self._prefinal = None
        log.debug("Utterance %d ended (%.1fs%s)", uid, event.duration,
                  ", truncated" if event.truncated else "")

        usable = (not event.truncated) and uid == self._utterance_id

        # Case 1: the pause decode already finished and covers the same speech.
        # Matching sample counts is an exact test, not a heuristic - the VAD
        # reports the same number only if the speaker never resumed.
        if (usable and prefinal is not None
                and prefinal[0] == uid
                and prefinal[1] == event.speech_samples):
            log.info("Reusing pause transcript for utterance %d (no re-decode)", uid)
            self._on_final(prefinal[2], uid)
            return

        # Case 2: it is still decoding the very same audio. Waiting for it
        # beats starting a second identical decode, which on a slow machine
        # would double the work at exactly the wrong moment.
        if (usable and pending is not None
                and pending[0] == uid
                and pending[1] == event.speech_samples):
            log.debug("Waiting on in-flight pause decode for utterance %d", uid)
            with self._lock:
                self._awaiting_prefinal = (uid, event.speech_samples, event.audio,
                                           time.monotonic() + _PREFINAL_TIMEOUT)
            self._emit(StatusEvent(STATUS_THINKING, "Transcribing..."))
            return

        self._emit(StatusEvent(STATUS_THINKING, "Transcribing..."))
        self._transcriber.transcribe_final(event.audio, uid)

    def _check_prefinal_timeout(self) -> None:
        """Safety net: never hang forever on a pause decode that never returns."""
        with self._lock:
            waiting = self._awaiting_prefinal
            if waiting is None or time.monotonic() < waiting[3]:
                return
            self._awaiting_prefinal = None
        log.warning("Pause decode did not return in time; decoding normally")
        self._transcriber.transcribe_final(waiting[2], waiting[0])

    def _reset_utterance(self) -> None:
        with self._lock:
            self._utterance_audio = []
            self._utterance_samples = 0
            self._partial_text = ""
            self._tracker = None
            self._prefinal = None
            self._prefinal_pending = None
            self._awaiting_prefinal = None
            self._eager_fired = -1

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
        self._maybe_answer_early(text, utterance_id)

    def _maybe_answer_early(self, text: str, utterance_id: int) -> None:
        """Answer from a mid-speech partial, before the speaker has stopped.

        Off by default (EAGER_ANSWER). When on, this is what puts text on
        screen while the question is still being asked - at the cost of
        sometimes answering a question that turns out to have a second half.
        Limited to one attempt per utterance so a long question cannot fire a
        stream of requests.
        """
        if not self.settings.eager_answer:
            return
        with self._lock:
            if utterance_id != self._utterance_id or self._eager_fired == utterance_id:
                return
            tracker = self._tracker

        decision = self.gate.consider_eager(text, has_context=self.context.has_context())
        if not decision.should_ask:
            return

        with self._lock:
            if self._eager_fired == utterance_id:
                return          # another thread won the race
            self._eager_fired = utterance_id

        log.info("Answering early (%s): %s", decision.reason, decision.question)
        self._ask(decision.question, tracker=tracker, speculative=True,
                  supersedes=False, confidence=decision.confidence,
                  utterance_id=utterance_id)

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
    def _maybe_speculate(self, text: str, uid: int, speech_samples: int) -> None:
        """Fire the LLM before end-of-utterance, from a pause transcript only.

        This is deliberately *not* driven by mid-speech partials. An earlier
        version triggered whenever the partial transcript stopped changing, on
        the theory that a settled transcript means a finished sentence. It does
        not: a partial also stops changing simply because no new decode has
        completed yet. Measured against real audio, that fired on
        "What is overfitting in machine" while the speaker was still saying
        "learning", and answered the wrong question about a third of the time.

        A pause transcript carries the signal the partial lacked - the VAD has
        confirmed the speaker actually went quiet, and the decode covers every
        speech frame up to that point. So the text is already settled by
        construction, and the gate is told so.

        One more check before firing: the decode itself took a few hundred
        milliseconds, so by now the speaker has been silent for roughly
        PAUSE_DECODE_AFTER plus the decode time - most of the way to
        END_OF_UTTERANCE_SILENCE. If they had merely drawn breath mid-sentence
        they would almost certainly have resumed by now, and if they did, the
        speech extent no longer matches and we let end-of-utterance handle it.
        That is what makes firing early nearly as safe as waiting.
        """
        with self._lock:
            tracker = self._tracker
            current = self._utterance_id
        if not text or uid != current:
            return
        if speech_samples >= 0:
            # Local path: require that not one new frame of speech arrived.
            if not self._vad.is_quiet():
                log.debug("Speculation skipped: speaker resumed during the decode")
                return
            if self._vad.current_speech_samples() != speech_samples:
                log.debug("Speculation skipped: more speech arrived during the decode")
                return
        elif not self._vad.state.in_speech:
            # Deepgram path: no local extent to compare, so just require that
            # the utterance is still open.
            return

        decision = self.gate.consider_partial(
            text, stable_for=float("inf"), has_context=self.context.has_context()
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
