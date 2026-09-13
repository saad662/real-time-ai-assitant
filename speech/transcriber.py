"""Speech-to-text with incremental (streaming-style) output.

Two providers, one interface:

* LocalWhisperTranscriber (default) - faster-whisper on CPU. It is not a
  streaming model, so "incremental" is achieved by *re-decoding the current
  utterance from its start* every PARTIAL_INTERVAL seconds. Re-decoding is
  wasteful, but it is far more accurate than stitching independently decoded
  fragments together, because Whisper sees the whole utterance every time.
  Measured warm on a Ryzen 7 PRO 4750U: `tiny.en` + int8 decodes a 2.8 s
  utterance in ~390 ms, `base.en` in ~745 ms. That cost no longer sits on the
  critical path - core/pipeline.py starts the decode during the speaker's
  pause - but it is still real CPU work, which is why `tiny.en` is the default.

* DeepgramTranscriber - a genuine streaming model over a WebSocket. Interims
  in ~150-300 ms and no local CPU at all, but needs a key and a network. Its
  `speech_final` marker drives the same early-answer path that the local
  pause decode does. Verified against a mock server in tests/test_deepgram.py.

Both run their work on their own thread and report through callbacks. Partial
jobs are *coalescing*: if a decode is still running when the next partial is
due, the older request is discarded rather than queued, so the transcriber can
never fall behind real time.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import re
import threading
import time

import numpy as np

from config.settings import SAMPLE_RATE, Settings

log = logging.getLogger(__name__)

# Whisper reliably emits these when handed near-silence or noise. They are
# artefacts of its training data (subtitle corpora), not speech.
_HALLUCINATIONS = {
    "", ".", "..", "...", "you", "thank you", "thank you.", "thanks",
    "thanks for watching", "thanks for watching!", "thank you for watching",
    "bye", "bye.", "bye bye", "okay", "ok", "yeah", "uh", "um", "hmm",
    "please subscribe", "subscribe", "[music]", "(music)", "[silence]",
    "amara.org", "subtitles by the amara.org community",
}

_PUNCT_RE = re.compile(r"[^\w\s']+")


def _decode_threads() -> int:
    """How many CPU threads to give the decoder.

    Measured on a 16-logical-core Ryzen 7 PRO 4750U with tiny.en/int8 on a
    2.8 s utterance: 2 threads 564 ms, 4 -> 422 ms, 6 -> 380 ms, 8 -> 370 ms,
    12 -> 427 ms, 16 -> 437 ms. Past the physical core count the SMT siblings
    fight each other and it gets slower, so cap at 8 and leave two logical
    cores for audio capture and the UI.
    """
    logical = os.cpu_count() or 4
    return max(1, min(8, logical - 2))


def is_probable_hallucination(text: str) -> bool:
    """True when a transcript looks like Whisper filling in silence."""
    cleaned = text.strip().lower()
    if cleaned in _HALLUCINATIONS:
        return True
    stripped = _PUNCT_RE.sub("", cleaned).strip()
    if not stripped:
        return True
    if stripped in _HALLUCINATIONS:
        return True
    # "you you you you" style loops.
    words = stripped.split()
    if len(words) >= 4 and len(set(words)) == 1:
        return True
    return False


class TranscriptionError(RuntimeError):
    pass


class BaseTranscriber:
    """Common callback plumbing.

    on_partial(text, utterance_id)   - best guess so far, may change
    on_prefinal(text, utterance_id, speech_samples)
                                     - final-quality transcript produced during
                                       a pause, before end-of-utterance
    on_final(text, utterance_id)     - the transcript we act on
    on_error(message)                - recoverable; the app keeps running
    on_ready(message)                - model loaded / socket connected
    """

    name = "base"

    def __init__(self, settings: Settings, on_partial=None, on_final=None,
                 on_error=None, on_ready=None, on_prefinal=None) -> None:
        self.settings = settings
        self.on_partial = on_partial or (lambda text, uid: None)
        self.on_prefinal = on_prefinal or (lambda text, uid, samples: None)
        self.on_final = on_final or (lambda text, uid: None)
        self.on_error = on_error or (lambda message: None)
        self.on_ready = on_ready or (lambda message: None)

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def begin_utterance(self, utterance_id: int) -> None: ...
    def feed(self, audio: np.ndarray, utterance_id: int) -> None: ...
    def transcribe_partial(self, audio: np.ndarray, utterance_id: int) -> None: ...
    def transcribe_prefinal(self, audio: np.ndarray, utterance_id: int,
                            speech_samples: int) -> None: ...
    def transcribe_final(self, audio: np.ndarray, utterance_id: int) -> None: ...


# ---------------------------------------------------------------------------
# Local faster-whisper
# ---------------------------------------------------------------------------

class LocalWhisperTranscriber(BaseTranscriber):
    name = "faster-whisper"

    def __init__(self, settings: Settings, **callbacks) -> None:
        super().__init__(settings, **callbacks)
        self._model = None
        self._thread = None
        self._running = threading.Event()
        self._final_queue: queue.Queue = queue.Queue()
        self._pending_partial = None          # (audio, utterance_id) - newest wins
        self._partial_lock = threading.Lock()
        self._wake = threading.Event()
        self._active_utterance = -1
        self.model_ready = threading.Event()

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._worker, name="stt-whisper", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        self._wake.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)

    def begin_utterance(self, utterance_id: int) -> None:
        self._active_utterance = utterance_id
        with self._partial_lock:
            self._pending_partial = None

    # -- job submission ----------------------------------------------------
    def transcribe_partial(self, audio: np.ndarray, utterance_id: int) -> None:
        if not self._running.is_set() or audio.size == 0:
            return
        with self._partial_lock:
            self._pending_partial = (audio, utterance_id)  # replaces any older one
        self._wake.set()

    def transcribe_prefinal(self, audio: np.ndarray, utterance_id: int,
                            speech_samples: int) -> None:
        """Decode during a pause. Same quality as a final, just earlier."""
        if not self._running.is_set() or audio.size == 0:
            return
        self._final_queue.put(("prefinal", audio, utterance_id,
                               speech_samples, time.perf_counter()))
        self._wake.set()

    def transcribe_final(self, audio: np.ndarray, utterance_id: int) -> None:
        if not self._running.is_set():
            return
        self._final_queue.put(("final", audio, utterance_id, 0, time.perf_counter()))
        self._wake.set()

    # -- worker ------------------------------------------------------------
    def _worker(self) -> None:
        if not self._load_model():
            return
        while self._running.is_set():
            job = None
            try:
                job = self._final_queue.get_nowait()
            except queue.Empty:
                pass

            if job is not None:
                kind, audio, uid, speech_samples, queued_at = job
                if kind == "prefinal":
                    self._run_prefinal(audio, uid, speech_samples, queued_at)
                else:
                    self._run_final(audio, uid, queued_at)
                continue

            with self._partial_lock:
                pending, self._pending_partial = self._pending_partial, None
            if pending is not None:
                self._run_partial(*pending)
                continue

            self._wake.wait(timeout=0.1)
            self._wake.clear()

    def _load_model(self) -> bool:
        s = self.settings
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            self.on_error(
                "faster-whisper is not installed (%s). Run: "
                "pip install faster-whisper" % exc
            )
            self._running.clear()
            return False

        started = time.perf_counter()
        log.info("Loading Whisper model '%s' (%s)...", s.stt_model, s.stt_compute)
        try:
            self._model = WhisperModel(
                s.stt_model,
                device="cpu",
                compute_type=s.stt_compute,
                cpu_threads=_decode_threads(),
            )
        except Exception as exc:
            self.on_error(
                "Could not load the speech model '%s': %s\n"
                "The first run downloads it, so check your internet connection, "
                "or set STT_MODEL=tiny.en in .env for a smaller download."
                % (s.stt_model, exc)
            )
            self._running.clear()
            return False

        # The first decode after loading is roughly twice as slow as a warm one
        # (measured: 850 ms vs 390 ms) because CTranslate2 allocates its
        # workspace lazily. Burning one decode on silence here means the user's
        # first real question does not pay for it.
        if s.warmup_model:
            try:
                warm_started = time.perf_counter()
                self._decode(np.zeros(SAMPLE_RATE, dtype=np.float32), quick=False)
                log.debug("Warm-up decode took %.0f ms",
                          (time.perf_counter() - warm_started) * 1000)
            except Exception:
                log.debug("Warm-up decode failed (harmless)", exc_info=True)

        elapsed = time.perf_counter() - started
        self.model_ready.set()
        log.info("Whisper model ready in %.1fs (%d decode threads)",
                 elapsed, _decode_threads())
        self.on_ready("Speech model '%s' ready (%.1fs)" % (s.stt_model, elapsed))
        return True

    def _decode(self, audio: np.ndarray, *, quick: bool) -> str:
        """One Whisper pass. `quick` trades a little accuracy for latency."""
        s = self.settings
        language = s.stt_language or None
        segments, _info = self._model.transcribe(
            audio,
            language=language,
            task="transcribe",
            beam_size=1 if quick else 3,
            best_of=1,
            temperature=0.0,
            # We already gate on VAD, and Whisper's own filter costs time.
            vad_filter=False,
            # Critical for correctness here: without this, a re-decode can be
            # primed by the previous utterance and invent continuations.
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def _run_partial(self, audio: np.ndarray, uid: int) -> None:
        if uid != self._active_utterance:
            return  # utterance already finished; this partial is worthless
        try:
            started = time.perf_counter()
            text = self._decode(audio, quick=True)
            took = (time.perf_counter() - started) * 1000.0
            if text and not is_probable_hallucination(text):
                log.debug("partial[%d] %.0fms: %s", uid, took, text)
                self.on_partial(text, uid)
            if took > self.settings.partial_interval * 1000 * 1.5:
                log.debug(
                    "Partial decode (%.0f ms) is slower than the partial interval; "
                    "consider STT_MODEL=tiny.en", took
                )
        except Exception:
            log.exception("Partial transcription failed")

    def _run_prefinal(self, audio: np.ndarray, uid: int, speech_samples: int,
                      queued_at: float) -> None:
        """Decode the utterance during the end-of-utterance wait.

        If the speaker stays quiet this result becomes the final transcript
        with no further work, so the whole decode disappears from the critical
        path. If they resume, the pipeline discards it - it compares
        `speech_samples` against the value SpeechEnd reports.
        """
        if uid != self._active_utterance:
            return
        try:
            text = self._decode(audio, quick=False)
            took = (time.perf_counter() - queued_at) * 1000.0
            if not text or is_probable_hallucination(text):
                self.on_prefinal("", uid, speech_samples)
                return
            log.info("prefinal[%d] %.0fms (%.1fs audio): %s",
                     uid, took, audio.size / SAMPLE_RATE, text)
            self.on_prefinal(text, uid, speech_samples)
        except Exception:
            # Deliberately silent: with no callback the pipeline has nothing to
            # reuse and falls back to a normal decode at end-of-utterance.
            log.exception("Pause transcription failed; falling back to final decode")

    def _run_final(self, audio: np.ndarray, uid: int, queued_at: float) -> None:
        try:
            text = self._decode(audio, quick=False)
            took = (time.perf_counter() - queued_at) * 1000.0
            if not text or is_probable_hallucination(text):
                log.debug("final[%d] discarded as noise: %r", uid, text)
                self.on_final("", uid)
                return
            log.info("final[%d] %.0fms (%.1fs audio): %s",
                     uid, took, audio.size / SAMPLE_RATE, text)
            self.on_final(text, uid)
        except Exception as exc:
            log.exception("Final transcription failed")
            self.on_error("Transcription failed: %s" % exc)
            self.on_final("", uid)


# ---------------------------------------------------------------------------
# Deepgram streaming
# ---------------------------------------------------------------------------

class DeepgramTranscriber(BaseTranscriber):
    """True streaming STT over a WebSocket.

    Implemented against the raw API with the `websockets` library rather than
    the SDK, because the SDK's client class has moved between major versions
    and this endpoint is four lines of query string.
    """

    name = "deepgram"
    _QUERY = (
        "?model={model}&encoding=linear16&sample_rate={rate}&channels=1"
        "&interim_results=true&punctuate=true&smart_format=true&endpointing={endpointing}"
    )

    def __init__(self, settings: Settings, **callbacks) -> None:
        super().__init__(settings, **callbacks)
        self._loop = None
        self._thread = None
        self._ws = None
        self._running = threading.Event()
        self._active_utterance = -1
        self._buffer = []           # confirmed pieces of the current utterance
        self._last_interim = ""
        self._connected = threading.Event()

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run_loop, name="stt-deepgram", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Close the socket first, then the loop.

        Stopping the loop outright leaves the send/finalize coroutines pending
        and asyncio complains loudly during interpreter teardown, so we ask the
        connection to close and give it a moment to unwind first.
        """
        self._running.clear()
        loop, self._loop = self._loop, None
        if loop is None:
            return

        # Always run this, even with no live socket: sends scheduled while the
        # connection was dropping are still queued on the loop.
        try:
            asyncio.run_coroutine_threadsafe(self._shutdown(), loop).result(timeout=2.0)
        except Exception:
            log.debug("Deepgram did not shut down cleanly", exc_info=True)

        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        self._connected.clear()

    def begin_utterance(self, utterance_id: int) -> None:
        self._active_utterance = utterance_id
        self._buffer = []
        self._last_interim = ""

    def feed(self, audio: np.ndarray, utterance_id: int) -> None:
        """Stream audio as it arrives - this is what makes it low latency."""
        if not self._running.is_set() or audio.size == 0:
            return
        pcm16 = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype("<i2").tobytes()
        self._submit(self._send(pcm16))

    def transcribe_partial(self, audio: np.ndarray, utterance_id: int) -> None:
        return  # Deepgram produces its own interims; nothing to poll.

    def transcribe_final(self, audio: np.ndarray, utterance_id: int) -> None:
        """Ask Deepgram to flush, then emit whatever we have."""
        self._submit(self._finalize(utterance_id))

    # -- asyncio plumbing --------------------------------------------------
    def _submit(self, coro) -> None:
        loop = self._loop
        if loop is None or not self._running.is_set():
            coro.close()
            return
        try:
            asyncio.run_coroutine_threadsafe(coro, loop)
        except Exception:
            coro.close()

    def _run_loop(self) -> None:
        try:
            import websockets  # noqa: F401
        except Exception as exc:
            self.on_error(
                "The 'websockets' package is required for STT_PROVIDER=deepgram "
                "(%s). Run: pip install websockets" % exc
            )
            self._running.clear()
            return

        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._connect_and_listen())
        except asyncio.CancelledError:
            # Expected: stop() cancels the listener. CancelledError derives from
            # BaseException, not Exception, so without this it escapes the
            # thread and is reported as an unhandled thread exception.
            log.debug("Deepgram listener cancelled during shutdown")
        except Exception:
            log.exception("Deepgram loop ended")
        finally:
            try:
                loop.close()
            except Exception:
                pass
            self._loop = None

    async def _connect_and_listen(self) -> None:
        import json
        import websockets

        s = self.settings
        url = s.deepgram_url.rstrip("/") + self._QUERY.format(
            model=s.deepgram_model, rate=SAMPLE_RATE,
            endpointing=s.deepgram_endpointing,
        )
        headers = {"Authorization": "Token %s" % s.deepgram_api_key}
        log.info("Connecting to %s", s.deepgram_url)

        while self._running.is_set():
            try:
                # websockets renamed this parameter in v13.
                try:
                    connect = websockets.connect(url, additional_headers=headers)
                except TypeError:
                    connect = websockets.connect(url, extra_headers=headers)

                async with connect as ws:
                    self._ws = ws
                    self._connected.set()
                    self.on_ready("Deepgram connected (%s)" % self.settings.deepgram_model)
                    log.info("Deepgram WebSocket connected")
                    async for raw in ws:
                        if not self._running.is_set():
                            break
                        try:
                            self._handle_message(json.loads(raw))
                        except Exception:
                            log.debug("Bad Deepgram message", exc_info=True)
            except Exception as exc:
                self._connected.clear()
                self._ws = None
                if not self._running.is_set():
                    return
                log.warning("Deepgram connection lost: %s; reconnecting in 2s", exc)
                self.on_error("Deepgram connection lost, reconnecting...")
                await asyncio.sleep(2.0)

    def _handle_message(self, message: dict) -> None:
        channel = message.get("channel") or {}
        alternatives = channel.get("alternatives") or []
        if not alternatives:
            return
        text = (alternatives[0].get("transcript") or "").strip()
        if not text:
            return
        uid = self._active_utterance
        if message.get("is_final"):
            self._buffer.append(text)
            self._last_interim = ""
        else:
            self._last_interim = text
        combined = " ".join(
            self._buffer + ([self._last_interim] if self._last_interim else [])
        ).strip()
        self.on_partial(combined, uid)

        # Deepgram's own endpointing says the speaker has stopped. That is the
        # same signal the local path gets from the VAD pause, so it drives
        # speculation the same way. speech_samples is -1 because there is no
        # local audio extent to compare against, which also means the pipeline
        # will never reuse this as the final transcript - Finalize still does
        # that, from the server's flushed result.
        if message.get("speech_final") and combined:
            self.on_prefinal(combined, uid, -1)

    async def _shutdown(self) -> None:
        """Close the socket and drop any audio still queued for sending.

        Without this, sends scheduled just before Stop outlive the loop and
        asyncio reports them as destroyed-while-pending during teardown.
        """
        ws, self._ws = self._ws, None
        current = asyncio.current_task()
        for task in asyncio.all_tasks():
            if task is not current:
                task.cancel()
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                log.debug("Deepgram close failed", exc_info=True)

    async def _send(self, payload: bytes) -> None:
        ws = self._ws
        if ws is None or not self._running.is_set():
            return
        try:
            await ws.send(payload)
        except Exception:
            log.debug("Deepgram send failed", exc_info=True)

    async def _finalize(self, uid: int) -> None:
        import json
        ws = self._ws
        if ws is not None:
            try:
                await ws.send(json.dumps({"type": "Finalize"}))
            except Exception:
                log.debug("Deepgram finalize failed", exc_info=True)
        # Give the server a moment to return the flushed result.
        await asyncio.sleep(0.25)
        text = " ".join(self._buffer + ([self._last_interim] if self._last_interim else []))
        self._buffer = []
        self._last_interim = ""
        self.on_final(text.strip(), uid)


def create_transcriber(settings: Settings, **callbacks) -> BaseTranscriber:
    if settings.stt_provider == "deepgram":
        return DeepgramTranscriber(settings, **callbacks)
    return LocalWhisperTranscriber(settings, **callbacks)
