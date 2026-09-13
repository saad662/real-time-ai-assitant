"""Voice activity detection and utterance segmentation.

Why energy VAD and not webrtcvad / Silero
-----------------------------------------
webrtcvad needs a C compiler on Windows and Silero pulls in onnxruntime; both
are real installation hazards for the "no dev tools installed" case. An RMS
detector with an *adaptive noise floor* is about thirty lines, has no
dependencies, and is entirely predictable - and because call audio arrives
already noise-suppressed by Meet/Zoom, it performs well here.

The adaptive floor is the important part: a fixed dB threshold breaks the
moment someone's fan, air conditioner, or headset hiss changes. We track the
quietest recent level and call anything `vad_margin_db` above it speech.

Segmentation rules (all configurable, see config/settings.py):
  * speech must last MIN_SPEECH_DURATION before an utterance opens - this
    rejects keyboard clicks and mouth noise;
  * an utterance only closes after END_OF_UTTERANCE_SILENCE of continuous
    silence, so a 200 ms thinking pause does not chop a question in half;
  * MAX_UTTERANCE_SECONDS is a hard stop so a monologue still gets processed.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from config.settings import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, Settings

log = logging.getLogger(__name__)

_EPS = 1e-10
# How much trailing silence to keep on a finished utterance. A little tail
# helps Whisper decode the final word; too much invites hallucination.
_KEEP_TAIL_SECONDS = 0.25


@dataclass
class SpeechStart:
    audio: np.ndarray          # pre-roll + the frames that opened the utterance


@dataclass
class SpeechChunk:
    audio: np.ndarray          # newly captured audio while speech continues


@dataclass
class SpeechEnd:
    audio: np.ndarray          # the complete utterance
    duration: float
    truncated: bool = False    # True when MAX_UTTERANCE_SECONDS forced the cut


@dataclass
class Level:
    rms_db: float
    noise_floor_db: float
    is_speech: bool


@dataclass
class VadState:
    in_speech: bool = False
    noise_floor_db: float = -60.0
    speech_run: float = 0.0     # seconds of consecutive speech frames
    silence_run: float = 0.0    # seconds of consecutive silence frames
    utterance: list = field(default_factory=list)


def rms_dbfs(frame: np.ndarray) -> float:
    """RMS of a float32 frame in dBFS. Silence is about -90, loud speech -20."""
    if frame.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)) + _EPS))
    return 20.0 * float(np.log10(max(rms, _EPS)))


class VoiceActivityDetector:
    """Feed it audio, get back utterance events.

    Stateful and *not* thread-safe - one instance lives on the pipeline's
    worker thread and nothing else touches it.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.state = VadState()
        self._pending = np.zeros(0, dtype=np.float32)  # partial frame carry-over
        preroll_frames = max(1, int(settings.preroll_seconds * 1000 / FRAME_MS))
        self._preroll = deque(maxlen=preroll_frames)
        self._frame_seconds = FRAME_MS / 1000.0
        self._level_counter = 0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.state = VadState(noise_floor_db=self.state.noise_floor_db)
        self._pending = np.zeros(0, dtype=np.float32)
        self._preroll.clear()

    def process(self, block: np.ndarray) -> list:
        """Consume a block of 16 kHz mono audio and return a list of events."""
        events: list = []
        if block is None or block.size == 0:
            return events

        if self._pending.size:
            block = np.concatenate([self._pending, block])
        frame_count = block.size // FRAME_SAMPLES
        self._pending = block[frame_count * FRAME_SAMPLES:].copy()

        for i in range(frame_count):
            frame = block[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES]
            events.extend(self._process_frame(frame))
        return events

    def flush(self) -> list:
        """Close any open utterance - used when the user presses Stop."""
        if not self.state.in_speech:
            self.reset()
            return []
        event = self._close_utterance(truncated=False)
        self.reset()
        return [event] if event else []

    # ------------------------------------------------------------------
    def _process_frame(self, frame: np.ndarray) -> list:
        s = self.settings
        st = self.state
        events: list = []

        level_db = rms_dbfs(frame)
        threshold = max(s.vad_threshold_db, st.noise_floor_db + s.vad_margin_db)
        is_speech = level_db > threshold

        # Track the noise floor only from non-speech frames. Rise slowly (the
        # room got louder) and fall quickly (we over-estimated), so a single
        # loud burst cannot deafen the detector for the rest of the call.
        if not is_speech:
            if level_db < st.noise_floor_db:
                st.noise_floor_db += 0.25 * (level_db - st.noise_floor_db)
            else:
                st.noise_floor_db += 0.02 * (level_db - st.noise_floor_db)
            st.noise_floor_db = max(-90.0, min(-20.0, st.noise_floor_db))

        self._level_counter += 1
        if self._level_counter % 5 == 0:  # ~10 Hz, enough for a UI meter
            events.append(Level(level_db, st.noise_floor_db, is_speech))

        if not st.in_speech:
            self._preroll.append(frame.copy())
            if is_speech:
                st.speech_run += self._frame_seconds
                if st.speech_run >= s.min_speech_duration:
                    opening = np.concatenate(list(self._preroll))
                    self._preroll.clear()
                    st.in_speech = True
                    st.silence_run = 0.0
                    st.utterance = [opening]
                    events.append(SpeechStart(opening))
            else:
                st.speech_run = 0.0
            return events

        # --- inside an utterance -----------------------------------------
        st.utterance.append(frame.copy())
        events.append(SpeechChunk(frame))

        if is_speech:
            st.silence_run = 0.0
        else:
            st.silence_run += self._frame_seconds

        duration = self._utterance_seconds()
        if st.silence_run >= s.end_of_utterance_silence:
            end = self._close_utterance(truncated=False)
            if end:
                events.append(end)
        elif duration >= s.max_utterance_seconds:
            log.info("Utterance hit MAX_UTTERANCE_SECONDS (%.1fs); cutting", duration)
            end = self._close_utterance(truncated=True)
            if end:
                events.append(end)
        return events

    # ------------------------------------------------------------------
    def _utterance_seconds(self) -> float:
        return sum(chunk.size for chunk in self.state.utterance) / float(SAMPLE_RATE)

    def _close_utterance(self, truncated: bool):
        st = self.state
        if not st.utterance:
            st.in_speech = False
            return None

        audio = np.concatenate(st.utterance)
        if not truncated:
            # Trim the silence we waited through, keeping a short tail.
            trim = int((st.silence_run - _KEEP_TAIL_SECONDS) * SAMPLE_RATE)
            if trim > 0 and trim < audio.size:
                audio = audio[:audio.size - trim]

        duration = audio.size / float(SAMPLE_RATE)
        st.in_speech = False
        st.speech_run = 0.0
        st.silence_run = 0.0
        st.utterance = []
        self._preroll.clear()

        if duration < self.settings.min_speech_duration:
            return None
        return SpeechEnd(audio=audio, duration=duration, truncated=truncated)
