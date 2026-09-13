"""Configuration, context memory, latency measurement, VAD and error handling."""

import time

import numpy as np
import pytest

from ai.client import ReasoningFilter, friendly_error
from ai.context import ConversationContext
from ai.prompts import build_messages, system_prompt
from audio.capture import resample_to_16k, to_mono
from config.settings import SAMPLE_RATE, Settings
from core import latency as lat
from core.events import EventBus, StatusEvent
from core.latency import LatencyTracker
from speech.transcriber import is_probable_hallucination
from speech.vad import (
    SpeechEnd,
    SpeechPause,
    SpeechStart,
    VoiceActivityDetector,
    rms_dbfs,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_bad_values_are_clamped_not_fatal():
    s = Settings(font_size=999, context_turns=-4, end_of_utterance_silence=99,
                 answer_mode="LOUD", llm_provider="hal9000", stt_provider="magic")
    s.validate()
    assert s.font_size == 32
    assert s.context_turns == 0
    assert s.end_of_utterance_silence == 5.0
    assert s.answer_mode == "NORMAL"
    assert s.llm_provider == "openai"
    assert s.stt_provider == "local"
    assert len(s.warnings) >= 4


def test_missing_key_warns_but_does_not_raise():
    s = Settings(openai_api_key="")
    s.validate()
    assert any("OPENAI_API_KEY" in w for w in s.warnings)


def test_deepgram_without_key_falls_back_to_local():
    s = Settings(stt_provider="deepgram", deepgram_api_key="")
    s.validate()
    assert s.stt_provider == "local"


def test_secrets_are_redacted_before_logging():
    s = Settings(openai_api_key="sk-super-secret-value", deepgram_api_key="")
    dump = s.redacted()
    assert "sk-super-secret-value" not in str(dump)
    assert dump["openai_api_key"].startswith("<set:")
    assert dump["deepgram_api_key"] == "<missing>"


# ---------------------------------------------------------------------------
# Conversation context
# ---------------------------------------------------------------------------

def test_context_keeps_only_the_configured_number_of_turns():
    ctx = ConversationContext(max_turns=3)
    for i in range(6):
        ctx.start_exchange("question %d" % i)
        ctx.complete_exchange("answer %d" % i)
    turns = ctx.turns()
    assert len(turns) == 3
    assert turns[0].question == "question 3"


def test_context_truncates_long_answers():
    ctx = ConversationContext(max_turns=2)
    ctx.start_exchange("q")
    ctx.complete_exchange("word " * 500)
    messages = ctx.as_messages()
    assert len(messages[1]["content"]) < 400


def test_abandoned_exchange_never_becomes_context():
    ctx = ConversationContext(max_turns=3)
    ctx.start_exchange("cancelled question")
    ctx.abandon_exchange()
    assert len(ctx) == 0
    assert ctx.as_messages() == []


def test_context_disabled_when_turns_is_zero():
    ctx = ConversationContext(max_turns=0)
    ctx.start_exchange("q")
    ctx.complete_exchange("a")
    assert ctx.as_messages() == []


def test_messages_are_ordered_system_context_question():
    ctx = ConversationContext(max_turns=2)
    ctx.start_exchange("What is a random forest")
    ctx.complete_exchange("An ensemble of decision trees.")
    messages = build_messages("How is that different from a decision tree",
                              ctx.as_messages(), "SHORT")
    assert messages[0]["role"] == "system"
    assert messages[1]["content"] == "What is a random forest"
    assert messages[-1]["role"] == "user"
    assert "decision tree" in messages[-1]["content"]


@pytest.mark.parametrize("mode", ["SHORT", "NORMAL", "DETAILED"])
def test_every_answer_mode_has_a_prompt(mode):
    prompt = system_prompt(mode)
    assert "LENGTH: %s" % mode in prompt
    assert "No preamble" in prompt


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

def test_latency_marks_and_report():
    t = LatencyTracker("test")
    t.mark(lat.SPEECH_STARTED)
    time.sleep(0.02)
    t.mark(lat.FINAL_TRANSCRIPT)
    t.mark(lat.QUESTION_DETECTED)
    t.mark(lat.LLM_REQUEST_STARTED)
    time.sleep(0.02)
    t.mark(lat.LLM_FIRST_TOKEN)

    report = t.report()
    assert report.stt_ms >= 15
    assert report.llm_first_token_ms >= 15
    assert report.total_ms > 0
    assert "STT:" in report.as_line() and "Total:" in report.as_line()


def test_first_mark_wins_unless_overwritten():
    t = LatencyTracker()
    first = t.mark(lat.PARTIAL_TRANSCRIPT)
    time.sleep(0.01)
    assert t.mark(lat.PARTIAL_TRANSCRIPT) == first
    assert t.mark(lat.PARTIAL_TRANSCRIPT, overwrite=True) > first


def test_missing_marks_report_zero_instead_of_crashing():
    t = LatencyTracker()
    assert t.delta_ms(lat.SPEECH_STARTED, lat.LLM_FIRST_TOKEN) == 0.0
    report = t.report()
    assert report.stt_ms == 0.0 and report.total_ms == 0.0


# ---------------------------------------------------------------------------
# Event bus error handling
# ---------------------------------------------------------------------------

def test_a_broken_subscriber_cannot_break_the_pipeline():
    bus = EventBus()
    received = []
    bus.subscribe(lambda e: (_ for _ in ()).throw(ValueError("boom")))
    bus.subscribe(received.append)
    bus.emit(StatusEvent("ready"))          # must not raise
    assert len(received) == 1


def test_unsubscribe_stops_delivery():
    bus = EventBus()
    received = []
    stop = bus.subscribe(received.append)
    bus.emit(StatusEvent("ready"))
    stop()
    bus.emit(StatusEvent("idle"))
    assert len(received) == 1


# ---------------------------------------------------------------------------
# Error message mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("message,expected", [
    ("Incorrect API key provided", "API key"),
    ("Rate limit reached for requests", "Rate limited"),
    ("You exceeded your current quota", "quota"),
    ("Connection error while contacting host", "internet"),
    ("Request timed out", "too long"),
])
def test_errors_become_readable(message, expected):
    assert expected.lower() in friendly_error(RuntimeError(message)).lower()


# ---------------------------------------------------------------------------
# Reasoning-token filtering
# ---------------------------------------------------------------------------

def _stream(filt, pieces):
    return "".join(filt.feed(p) for p in pieces) + filt.flush()


def test_plain_text_passes_through_untouched():
    assert _stream(ReasoningFilter(), ["Over", "fitting ", "is bad."]) == \
        "Overfitting is bad."


def test_think_block_is_removed():
    out = _stream(ReasoningFilter(),
                  ["<think>let me reason about this</think>", "Overfitting is bad."])
    assert out == "Overfitting is bad."


def test_think_block_split_across_chunks():
    """Tokens arrive a few characters at a time, so tags straddle chunks."""
    out = _stream(ReasoningFilter(),
                  ["<th", "ink>", "hmm", " maybe", "</th", "ink>", "Answer."])
    assert out == "Answer."


def test_text_before_and_after_a_think_block_is_kept():
    out = _stream(ReasoningFilter(), ["A", "<think>x</think>", "B"])
    assert out == "AB"


def test_unterminated_think_block_emits_nothing():
    """A truncated reasoning stream must not dump raw thoughts into the UI."""
    assert _stream(ReasoningFilter(), ["<think>", "still thinking and then cut"]) == ""


def test_angle_brackets_that_are_not_tags_survive():
    out = _stream(ReasoningFilter(), ["if a < b and c > d then"])
    assert out == "if a < b and c > d then"


def test_unknown_errors_stay_on_one_line():
    text = friendly_error(RuntimeError("something\nwith\nnewlines"))
    assert "\n" not in text


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def test_stereo_is_downmixed():
    stereo = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    assert np.allclose(to_mono(stereo), [0.5, 0.5])


def test_resampling_preserves_duration():
    one_second_48k = np.zeros(48000, dtype=np.float32)
    assert resample_to_16k(one_second_48k, 48000).size == SAMPLE_RATE
    one_second_44k = np.zeros(44100, dtype=np.float32)
    assert abs(resample_to_16k(one_second_44k, 44100).size - SAMPLE_RATE) <= 1
    already = np.zeros(SAMPLE_RATE, dtype=np.float32)
    assert resample_to_16k(already, SAMPLE_RATE).size == SAMPLE_RATE


# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------

def _tone(seconds: float, amplitude: float = 0.25) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


def test_rms_dbfs_ranges():
    assert rms_dbfs(_silence(0.1)) < -80
    assert -20 < rms_dbfs(_tone(0.1, amplitude=0.5)) < 0


def test_vad_segments_one_utterance(settings):
    vad = VoiceActivityDetector(settings)
    events = []
    for block in (_silence(0.5), _tone(1.2), _silence(1.2)):
        events.extend(vad.process(block))

    starts = [e for e in events if isinstance(e, SpeechStart)]
    ends = [e for e in events if isinstance(e, SpeechEnd)]
    assert len(starts) == 1
    assert len(ends) == 1
    # Pre-roll included, waited-through silence trimmed.
    assert 1.0 < ends[0].duration < 2.0


def test_short_pause_does_not_split_an_utterance(settings):
    vad = VoiceActivityDetector(settings)
    events = []
    for block in (_silence(0.4), _tone(0.8), _silence(0.3), _tone(0.8), _silence(1.2)):
        events.extend(vad.process(block))
    assert len([e for e in events if isinstance(e, SpeechEnd)]) == 1


def test_blips_shorter_than_min_duration_are_ignored(settings):
    vad = VoiceActivityDetector(settings)
    events = []
    for block in (_silence(0.4), _tone(0.05), _silence(1.2)):
        events.extend(vad.process(block))
    assert not [e for e in events if isinstance(e, SpeechStart)]


def test_long_monologue_is_cut_at_the_limit(settings):
    settings.max_utterance_seconds = 5.0
    vad = VoiceActivityDetector(settings)
    events = vad.process(_silence(0.3))
    events += vad.process(_tone(8.0))
    ends = [e for e in events if isinstance(e, SpeechEnd)]
    assert ends and ends[0].truncated


# --- pause decode: the main latency optimisation ---------------------------

def test_pause_fires_before_end_of_utterance(settings):
    """The pause event must arrive early enough to be worth anything."""
    vad = VoiceActivityDetector(settings)
    events = []
    for block in (_silence(0.4), _tone(1.0), _silence(1.2)):
        events.extend(vad.process(block))

    kinds = [type(e).__name__ for e in events]
    pauses = [e for e in events if isinstance(e, SpeechPause)]
    ends = [e for e in events if isinstance(e, SpeechEnd)]
    assert len(pauses) == 1, kinds
    assert len(ends) == 1
    # The pause must come first, otherwise there is no window to decode in.
    assert events.index(pauses[0]) < events.index(ends[0])


def test_pause_and_end_agree_when_speech_does_not_resume(settings):
    """Equal speech_samples is the signal to reuse the pause transcript."""
    vad = VoiceActivityDetector(settings)
    events = []
    for block in (_silence(0.4), _tone(1.0), _silence(1.2)):
        events.extend(vad.process(block))
    pause = [e for e in events if isinstance(e, SpeechPause)][0]
    end = [e for e in events if isinstance(e, SpeechEnd)][0]
    assert pause.speech_samples == end.speech_samples


def test_resumed_speech_invalidates_the_pause_transcript(settings):
    """A mid-sentence pause must not be mistaken for the end of a question."""
    vad = VoiceActivityDetector(settings)
    events = []
    for block in (_silence(0.4), _tone(0.8), _silence(0.35), _tone(0.8), _silence(1.2)):
        events.extend(vad.process(block))
    pauses = [e for e in events if isinstance(e, SpeechPause)]
    end = [e for e in events if isinstance(e, SpeechEnd)][0]
    assert len(pauses) == 2, "one pause per quiet stretch"
    # The first pause is stale - more speech followed it.
    assert pauses[0].speech_samples < end.speech_samples
    # The last one is still valid.
    assert pauses[-1].speech_samples == end.speech_samples


def test_pause_audio_excludes_the_silence(settings):
    vad = VoiceActivityDetector(settings)
    events = []
    for block in (_silence(0.4), _tone(1.0), _silence(1.2)):
        events.extend(vad.process(block))
    pause = [e for e in events if isinstance(e, SpeechPause)][0]
    assert pause.audio.size == pause.speech_samples
    assert pause.audio.size > 0


def test_pause_decode_cannot_be_configured_after_end_of_utterance(settings):
    """A pause decode at or after end-of-utterance would buy nothing."""
    settings.pause_decode_after = 5.0
    settings.end_of_utterance_silence = 0.7
    settings.validate()
    assert settings.pause_decode_after < settings.end_of_utterance_silence


def test_flush_closes_an_open_utterance(settings):
    vad = VoiceActivityDetector(settings)
    vad.process(_silence(0.3))
    vad.process(_tone(1.0))
    assert len(vad.flush()) == 1
    assert not vad.state.in_speech


# ---------------------------------------------------------------------------
# Transcript hygiene
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["", ".", "you", "Thank you.", "Thanks for watching!",
                                  "  ...  ", "you you you you"])
def test_whisper_silence_artefacts_are_dropped(text):
    assert is_probable_hallucination(text)


@pytest.mark.parametrize("text", ["What is overfitting?", "Explain gradient descent",
                                  "Thank you for explaining, but what about recall?"])
def test_real_speech_is_kept(text):
    assert not is_probable_hallucination(text)
