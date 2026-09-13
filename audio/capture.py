"""Real-time audio capture.

Design notes
------------
* The audio callback does *one* thing: push a numpy array onto a queue. Any
  work done in that callback (resampling, VAD, logging) risks buffer underruns
  and audible glitches in the call itself, so all of it happens downstream.
* Downmix + resample to 16 kHz mono happens in the consumer thread. Loopback
  endpoints are almost always 48 kHz stereo, which is an exact 3:1 decimation.
* The queue is bounded. If the consumer stalls we drop the *oldest* frames,
  because for a live assistant stale audio is worthless - better a gap than a
  growing backlog that makes every answer later than the last.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import warnings

import numpy as np

from config.settings import SAMPLE_RATE
from .devices import LOOPBACK, AudioDevice, AudioDeviceError, resolve_device

log = logging.getLogger(__name__)

# ~8 seconds of 20 ms frames. Plenty of slack, still bounded.
_MAX_QUEUED_FRAMES = 400

_RPC_E_CHANGED_MODE = -2147417850  # 0x80010106


def _co_initialize() -> bool:
    """Initialise COM on the current thread. Returns True if we own it.

    WASAPI is COM-based, so any thread that touches `soundcard` must do this
    first. Non-Windows platforms and already-initialised threads are no-ops.
    """
    try:
        import ctypes
        ole32 = ctypes.windll.ole32
    except Exception:
        return False
    result = ole32.CoInitializeEx(None, 0x0)      # COINIT_MULTITHREADED
    if result == _RPC_E_CHANGED_MODE:
        # Another library already picked apartment threading on this thread.
        result = ole32.CoInitializeEx(None, 0x2)  # COINIT_APARTMENTTHREADED
    return result >= 0




def to_mono(block: np.ndarray) -> np.ndarray:
    """(frames, channels) float32 -> (frames,) float32."""
    if block.ndim == 1:
        return block.astype(np.float32, copy=False)
    if block.shape[1] == 1:
        return block[:, 0].astype(np.float32, copy=False)
    return block.mean(axis=1, dtype=np.float32)


def resample_to_16k(mono: np.ndarray, source_rate: int) -> np.ndarray:
    """Resample mono float32 audio to 16 kHz.

    Integer ratios (48k, 32k, 96k) use mean-pooling, which is a cheap but real
    anti-aliasing filter. Everything else falls back to linear interpolation,
    which is good enough for speech features at these rates.
    """
    if source_rate == SAMPLE_RATE or mono.size == 0:
        return mono.astype(np.float32, copy=False)

    if source_rate > SAMPLE_RATE and source_rate % SAMPLE_RATE == 0:
        factor = source_rate // SAMPLE_RATE
        usable = (mono.size // factor) * factor
        if usable == 0:
            return np.zeros(0, dtype=np.float32)
        return mono[:usable].reshape(-1, factor).mean(axis=1).astype(np.float32)

    duration = mono.size / float(source_rate)
    target_len = int(round(duration * SAMPLE_RATE))
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    source_x = np.linspace(0.0, duration, num=mono.size, endpoint=False)
    target_x = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(target_x, source_x, mono).astype(np.float32)


class AudioCapture:
    """Streams 16 kHz mono float32 audio from a device into a queue.

    Usage:
        cap = AudioCapture(device)
        cap.start()
        block = cap.read(timeout=0.5)   # np.ndarray or None
        cap.stop()
    """

    def __init__(self, device: AudioDevice, on_error=None) -> None:
        self.device = device
        self.on_error = on_error
        self._queue: queue.Queue = queue.Queue(maxsize=_MAX_QUEUED_FRAMES)
        self._running = threading.Event()
        self._stream = None            # sounddevice stream
        self._thread = None            # soundcard polling thread
        self._backend = ""
        self._dropped = 0
        self._last_drop_log = 0.0

    # -- lifecycle ---------------------------------------------------------
    @property
    def backend(self) -> str:
        return self._backend

    @property
    def running(self) -> bool:
        return self._running.is_set()

    @property
    def dropped_frames(self) -> int:
        return self._dropped

    def start(self) -> None:
        """Open the device, trying the backend most likely to work first.

        For loopback the order is soundcard-then-sounddevice, because the
        PortAudio builds shipped in sounddevice wheels usually do *not* expose
        WASAPI loopback (`WasapiSettings` has no `loopback` argument) - verified
        on sounddevice 0.5.6 / PortAudio 19.7. For microphones sounddevice is
        preferred: it is callback-driven, so there is no polling thread.
        """
        if self._running.is_set():
            return
        self._running.set()

        if self.device.kind == LOOPBACK:
            order = [("soundcard", self._start_soundcard),
                     ("sounddevice", self._start_sounddevice)]
        else:
            order = [("sounddevice", self._start_sounddevice),
                     ("soundcard", self._start_soundcard)]

        failures = []
        for name, starter in order:
            try:
                starter()
                self._backend = name
                break
            except Exception as exc:
                failures.append((name, exc))
                log.warning("%s capture failed: %s", name, exc)
        else:
            self._running.clear()
            raise AudioDeviceError(self._capture_help(failures))
        log.info(
            "Audio capture started on '%s' (%s, %d ch @ %d Hz) via %s",
            self.device.name, self.device.kind, self.device.channels,
            self.device.samplerate, self._backend,
        )

    def stop(self) -> None:
        self._running.clear()
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                log.debug("Error closing audio stream", exc_info=True)
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.5)
        self._drain()
        log.info("Audio capture stopped (%d frames dropped)", self._dropped)

    # -- consumer API ------------------------------------------------------
    def read(self, timeout: float = 0.5):
        """Next block of 16 kHz mono audio, or None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _publish(self, mono16k: np.ndarray) -> None:
        if mono16k.size == 0:
            return
        try:
            self._queue.put_nowait(mono16k)
        except queue.Full:
            # Drop the oldest frame to keep latency bounded.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(mono16k)
            except queue.Empty:
                pass
            except queue.Full:
                pass
            self._dropped += 1
            now = time.monotonic()
            if now - self._last_drop_log > 5.0:
                self._last_drop_log = now
                log.warning(
                    "Audio consumer is behind; dropped %d frames so far. "
                    "Try a smaller STT model.", self._dropped
                )

    # -- backends ----------------------------------------------------------
    def _start_sounddevice(self) -> None:
        import sounddevice as sd

        extra = None
        if self.device.kind == LOOPBACK:
            # Opening an *output* endpoint as an input is what loopback means.
            extra = sd.WasapiSettings(loopback=True)

        source_rate = self.device.samplerate
        channels = max(1, self.device.channels)
        blocksize = max(160, int(source_rate * 0.02))  # 20 ms at the native rate

        def callback(indata, frames, time_info, status):  # noqa: ARG001
            if status:
                log.debug("Audio stream status: %s", status)
            if not self._running.is_set():
                return
            mono = to_mono(np.asarray(indata, dtype=np.float32))
            self._publish(resample_to_16k(mono, source_rate))

        self._stream = sd.InputStream(
            device=self.device.index,
            channels=channels,
            samplerate=source_rate,
            dtype="float32",
            blocksize=blocksize,
            latency="low",
            extra_settings=extra,
            callback=callback,
        )
        self._stream.start()

    def _start_soundcard(self) -> None:
        """Primary loopback path: soundcard drives WASAPI directly via ctypes.

        It has no callback API, so we poll from a dedicated thread. Every
        soundcard call - enumeration included - has to happen on that thread,
        because WASAPI is COM and COM is per-thread: calling it from a thread
        that never ran CoInitializeEx fails with 0x800401F0
        (CO_E_NOTINITIALIZED).

        Startup is therefore asynchronous, and we block briefly on `ready` so
        that a failure surfaces here as an exception rather than as silence.

        Order matters: soundcard initialises COM once, at *import* time, and
        treats CoInitializeEx returning S_FALSE ("already initialised on this
        thread") as a hard error. So the import and the device lookup happen
        here, on the calling thread, and only the recorder - which creates its
        COM objects lazily - runs on the pump thread after our own
        CoInitializeEx.
        """
        import soundcard as sc

        # soundcard warns on every buffer gap, and a loopback stream is full of
        # them whenever nothing is playing - which is most of a call. The
        # warning is on by default (it calls simplefilter('always')), so it
        # would flood the console. We already count real drops ourselves.
        warnings.filterwarnings("ignore", category=sc.SoundcardRuntimeWarning)

        if self.device.kind == LOOPBACK:
            speaker = self._match_soundcard_speaker(sc, self.device.name)
            mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
        else:
            mic = self._match_soundcard_mic(sc, self.device.name)

        source_rate = self.device.samplerate
        chunk = max(160, int(source_rate * 0.02))
        ready = threading.Event()
        failure = {}

        def pump() -> None:
            com_initialized = _co_initialize()
            try:
                with mic.recorder(samplerate=source_rate, blocksize=chunk) as rec:
                    ready.set()
                    while self._running.is_set():
                        data = rec.record(numframes=chunk)
                        mono = to_mono(np.asarray(data, dtype=np.float32))
                        self._publish(resample_to_16k(mono, source_rate))
            except Exception as exc:
                failure["error"] = exc
                if not ready.is_set():
                    ready.set()          # unblock start(), which will raise
                    return
                if self._running.is_set():
                    log.exception("soundcard capture thread died")
                    self._running.clear()
                    if self.on_error:
                        self.on_error(
                            "Audio capture stopped: %s. The device may have been "
                            "disconnected." % exc
                        )
            finally:
                # Deliberately no CoUninitialize: soundcard caches COM
                # interface pointers, and tearing COM down under it produces
                # noisy errors during interpreter shutdown. The thread is
                # daemonised, so the process reclaims everything anyway.
                del com_initialized

        self._thread = threading.Thread(target=pump, name="audio-soundcard", daemon=True)
        self._thread.start()

        if not ready.wait(timeout=5.0):
            self._thread = None
            raise AudioDeviceError("soundcard did not open the device within 5 seconds")
        if "error" in failure:
            self._thread = None
            raise AudioDeviceError("soundcard could not open the device: %s"
                                   % failure["error"])

    @staticmethod
    def _match_by_name(candidates: list, name: str, fallback):
        """Match a sounddevice name against soundcard's device list.

        In practice the two libraries report identical strings, so an exact
        match almost always wins. The looser passes exist for the cases where
        PortAudio truncates a long endpoint name to 31 characters.
        """
        if not candidates:
            raise AudioDeviceError("soundcard found no matching devices")
        target = name.strip().lower()
        names = [(item, str(item.name).strip().lower()) for item in candidates]

        for item, item_name in names:
            if item_name == target:
                return item
        for item, item_name in names:
            if item_name.startswith(target[:28]) or target.startswith(item_name[:28]):
                return item
        for item, item_name in names:
            if item_name in target or target in item_name:
                return item
        log.warning("No soundcard device matches %r; using the system default", name)
        return fallback()

    @classmethod
    def _match_soundcard_speaker(cls, sc, name: str):
        return cls._match_by_name(sc.all_speakers(), name, sc.default_speaker)

    @classmethod
    def _match_soundcard_mic(cls, sc, name: str):
        return cls._match_by_name(
            sc.all_microphones(include_loopback=False), name, sc.default_microphone
        )

    def _capture_help(self, failures: list) -> str:
        detail = "\n".join("  %-12s %s" % (name + ":", exc) for name, exc in failures)
        return (
            "Could not open '%s' for capture.\n"
            "%s\n"
            "Things to try:\n"
            "  1. Run `python app.py --list-devices` and set AUDIO_DEVICE to a "
            "listed index.\n"
            "  2. Make sure the device is the one Windows is actually playing to "
            "(check the speaker icon in the taskbar).\n"
            "  3. Install the loopback backend: pip install soundcard\n"
            "  4. Some apps grab the device in exclusive mode - turn that off in "
            "Sound Control Panel > Properties > Advanced.\n"
            "  5. As a last resort, capture your microphone: python app.py --mic"
            % (self.device.name, detail)
        )


def open_default_capture(spec: str = "", mode: str = LOOPBACK, on_error=None) -> AudioCapture:
    """Convenience helper used by the pipeline and the --check command."""
    device = resolve_device(spec, mode)
    return AudioCapture(device, on_error=on_error)
