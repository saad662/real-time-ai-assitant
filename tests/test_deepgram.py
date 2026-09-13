"""Deepgram streaming path, verified against a mock server.

The real endpoint needs a paid key, so these tests stand up a local WebSocket
server that speaks Deepgram's JSON protocol: interim results, `is_final`
segments, a `speech_final` end-of-speech marker, and a response to `Finalize`.

That is enough to verify the parts that are actually ours - URL construction,
auth header, PCM encoding, interim accumulation, the speech_final -> prefinal
hop that drives speculation, and Finalize -> final - without pretending we have
tested Deepgram's own transcription accuracy.
"""

import asyncio
import json
import threading
import time

import numpy as np
import pytest

from config.settings import SAMPLE_RATE, Settings
from speech.transcriber import DeepgramTranscriber

websockets = pytest.importorskip("websockets", reason="websockets not installed")


class MockDeepgram:
    """Minimal stand-in for api.deepgram.com/v1/listen."""

    def __init__(self, script):
        # script: list of (transcript, is_final, speech_final)
        self.script = list(script)
        self.received_audio = bytearray()
        self.received_control = []
        self.path = ""
        self.auth = ""
        self.port = None
        self._loop = None
        self._server = None
        self._thread = None
        self.connected = threading.Event()

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        ready = threading.Event()

        def run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._serve(ready))
            self._loop.run_forever()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        assert ready.wait(10), "mock Deepgram server did not start"
        return self

    async def _serve(self, ready):
        async def handler(ws, path=None):
            # websockets >= 13 passes only the connection and exposes the
            # request on it; older versions passed the path as a second arg.
            request = getattr(ws, "request", None)
            if path is not None:
                self.path = path
            elif request is not None:
                self.path = request.path
            else:
                self.path = getattr(ws, "path", "")

            headers = getattr(ws, "request_headers", None)
            if headers is None and request is not None:
                headers = request.headers
            try:
                self.auth = headers.get("Authorization", "")
            except Exception:
                self.auth = ""
            self.connected.set()

            # Deliver the scripted results as the audio arrives.
            async for message in ws:
                if isinstance(message, (bytes, bytearray)):
                    self.received_audio.extend(message)
                    if self.script:
                        await ws.send(json.dumps(self._result(*self.script.pop(0))))
                else:
                    self.received_control.append(message)
                    payload = json.loads(message)
                    if payload.get("type") == "Finalize":
                        # Deepgram flushes whatever it is holding.
                        await ws.send(json.dumps(
                            self._result("flushed tail", True, True)))

        self._server = await websockets.serve(handler, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        ready.set()

    def stop(self):
        if self._loop is not None and self._server is not None:
            # Close the listening socket before stopping the loop, otherwise
            # the accept coroutine is left pending and asyncio complains.
            def shut():
                self._server.close()
                self._loop.call_later(0.05, self._loop.stop)
            try:
                self._loop.call_soon_threadsafe(shut)
            except Exception:
                pass
        elif self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=3)

    @property
    def url(self):
        return "ws://127.0.0.1:%d/v1/listen" % self.port

    @staticmethod
    def _result(transcript, is_final, speech_final):
        return {
            "type": "Results",
            "is_final": is_final,
            "speech_final": speech_final,
            "channel": {"alternatives": [{"transcript": transcript,
                                          "confidence": 0.98}]},
        }


def _settings(url):
    s = Settings()
    s.stt_provider = "deepgram"
    s.deepgram_api_key = "test-key-123"
    s.deepgram_url = url
    s.validate()
    return s


def _tone(seconds=0.1):
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


class _Collector:
    def __init__(self):
        self.partials, self.prefinals, self.finals, self.errors = [], [], [], []
        self.ready = threading.Event()

    def kwargs(self):
        return dict(
            on_partial=lambda text, uid: self.partials.append(text),
            on_prefinal=lambda text, uid, n: self.prefinals.append((text, n)),
            on_final=lambda text, uid: self.finals.append(text),
            on_error=self.errors.append,
            on_ready=lambda message: self.ready.set(),
        )


def _wait(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def server():
    mock = MockDeepgram([
        ("What is", False, False),
        ("What is overfitting", False, False),
        ("What is overfitting in machine learning", True, True),
    ]).start()
    yield mock
    mock.stop()


def test_connects_and_sends_the_api_key(server):
    collector = _Collector()
    t = DeepgramTranscriber(_settings(server.url), **collector.kwargs())
    t.start()
    try:
        assert collector.ready.wait(10), "never connected: %s" % collector.errors
        assert server.auth == "Token test-key-123"
    finally:
        t.stop()


def test_request_url_carries_the_streaming_options(server):
    collector = _Collector()
    settings = _settings(server.url)
    settings.deepgram_model = "nova-3"
    settings.deepgram_endpointing = 250
    t = DeepgramTranscriber(settings, **collector.kwargs())
    t.start()
    try:
        assert collector.ready.wait(10)
        assert "model=nova-3" in server.path
        assert "interim_results=true" in server.path
        assert "sample_rate=16000" in server.path
        assert "encoding=linear16" in server.path
        assert "endpointing=250" in server.path
    finally:
        t.stop()


def test_audio_is_sent_as_16_bit_pcm(server):
    collector = _Collector()
    t = DeepgramTranscriber(_settings(server.url), **collector.kwargs())
    t.start()
    try:
        assert collector.ready.wait(10)
        t.begin_utterance(1)
        audio = _tone(0.1)
        t.feed(audio, 1)
        assert _wait(lambda: len(server.received_audio) > 0), "no audio reached the server"
        assert _wait(lambda: len(server.received_audio) >= audio.size * 2)
        # Round-trip the bytes and check they still describe the same waveform.
        decoded = np.frombuffer(bytes(server.received_audio[:audio.size * 2]),
                                dtype="<i2").astype(np.float32) / 32767.0
        assert np.allclose(decoded, audio, atol=1e-3)
    finally:
        t.stop()


def test_interims_accumulate_into_a_growing_partial(server):
    collector = _Collector()
    t = DeepgramTranscriber(_settings(server.url), **collector.kwargs())
    t.start()
    try:
        assert collector.ready.wait(10)
        t.begin_utterance(1)
        for _ in range(3):
            t.feed(_tone(0.05), 1)
            time.sleep(0.15)
        assert _wait(lambda: len(collector.partials) >= 3), collector.partials
        assert collector.partials[0] == "What is"
        assert "overfitting" in collector.partials[-1]
    finally:
        t.stop()


def test_speech_final_produces_a_prefinal_for_speculation(server):
    """This is what lets Deepgram trigger the same early-answer path as the
    local pause decode."""
    collector = _Collector()
    t = DeepgramTranscriber(_settings(server.url), **collector.kwargs())
    t.start()
    try:
        assert collector.ready.wait(10)
        t.begin_utterance(7)
        for _ in range(3):
            t.feed(_tone(0.05), 7)
            time.sleep(0.15)
        assert _wait(lambda: collector.prefinals), "speech_final did not yield a prefinal"
        text, samples = collector.prefinals[-1]
        assert "overfitting in machine learning" in text
        # -1 means "no local audio extent", so the pipeline will not reuse it
        # as the final transcript - Finalize still does that.
        assert samples == -1
    finally:
        t.stop()


def test_finalize_flushes_and_yields_a_final(server):
    collector = _Collector()
    t = DeepgramTranscriber(_settings(server.url), **collector.kwargs())
    t.start()
    try:
        assert collector.ready.wait(10)
        t.begin_utterance(1)
        t.feed(_tone(0.05), 1)
        time.sleep(0.2)
        t.transcribe_final(np.zeros(0, dtype=np.float32), 1)
        assert _wait(lambda: collector.finals), "Finalize produced no final transcript"
        assert any(json.loads(m).get("type") == "Finalize"
                   for m in server.received_control)
        assert collector.finals[-1].strip()
    finally:
        t.stop()


def test_transcribe_partial_is_a_no_op(server):
    """Deepgram produces its own interims; polling it would be wrong."""
    collector = _Collector()
    t = DeepgramTranscriber(_settings(server.url), **collector.kwargs())
    t.start()
    try:
        assert collector.ready.wait(10)
        t.begin_utterance(1)
        t.transcribe_partial(_tone(0.5), 1)
        time.sleep(0.3)
        assert not server.received_audio, "transcribe_partial should not send audio"
    finally:
        t.stop()


def test_unreachable_server_reports_an_error_and_keeps_running():
    collector = _Collector()
    settings = _settings("ws://127.0.0.1:1/v1/listen")   # nothing listens here
    t = DeepgramTranscriber(settings, **collector.kwargs())
    t.start()
    try:
        assert _wait(lambda: collector.errors, timeout=8), "no error was reported"
        assert "deepgram" in collector.errors[0].lower()
        # Feeding it must stay harmless while it retries.
        t.feed(_tone(0.05), 1)
    finally:
        t.stop()


def test_begin_utterance_clears_the_previous_transcript(server):
    collector = _Collector()
    t = DeepgramTranscriber(_settings(server.url), **collector.kwargs())
    t.start()
    try:
        assert collector.ready.wait(10)
        t.begin_utterance(1)
        t.feed(_tone(0.05), 1)
        assert _wait(lambda: collector.partials)
        t.begin_utterance(2)
        assert t._buffer == [] and t._last_interim == ""
    finally:
        t.stop()
