"""Central configuration.

Everything is read from environment variables (loaded from .env) exactly once
at startup. Nothing else in the codebase touches os.environ, so there is a
single place to look when you want to know what a knob does.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"

# Whisper works on 16 kHz mono audio. Everything upstream is resampled to this.
SAMPLE_RATE = 16_000
FRAME_MS = 20  # VAD frame size
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 320 samples

# Suffixes, not substrings: "llm_max_tokens" is not a secret.
_SECRET_SUFFIXES = ("_api_key", "_token", "_secret", "_password")


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(_env(name) or default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # ---- LLM -------------------------------------------------------------
    llm_provider: str = "openai"          # openai | anthropic
    llm_model: str = "gpt-4o-mini"
    # Any OpenAI-compatible endpoint. This is how you reach the genuinely fast
    # inference providers (Groq, Cerebras) without new client code - they speak
    # the same wire protocol, they just run on hardware built for low
    # time-to-first-token. Empty means api.openai.com.
    llm_base_url: str = ""
    # Tried if the primary model fails before producing any output. Providers
    # have brief outages where one model returns "not available" while the
    # rest of the catalogue is fine - seen on Groq for ~3 minutes, which in a
    # live call would have meant three questions with no answer at all.
    llm_fallback_model: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    llm_timeout: float = 30.0
    llm_max_tokens: int = 700

    # ---- Speech to text --------------------------------------------------
    stt_provider: str = "local"           # local | deepgram
    # tiny.en, not base.en: measured on a Ryzen 7 PRO 4750U, base.en takes
    # ~1.7 s to decode a 2.8 s utterance versus ~0.85 s for tiny.en, and that
    # time lands directly on the critical path. See the README's latency table.
    stt_model: str = "tiny.en"            # faster-whisper model name
    stt_compute: str = "int8"             # int8 is the fastest CPU option
    stt_language: str = "en"
    deepgram_api_key: str = ""
    deepgram_model: str = "nova-3"
    # Endpoint override. Exists so the streaming path can be tested against a
    # local mock server, and so a self-hosted Deepgram can be pointed at.
    deepgram_url: str = "wss://api.deepgram.com/v1/listen"
    # Deepgram's own end-of-speech detection, milliseconds. Its equivalent of
    # END_OF_UTTERANCE_SILENCE, running server-side on a real streaming model.
    deepgram_endpointing: int = 300
    # Terms to boost in the language model. Without these, nova-3 hears "SQL"
    # as "sequel" and "NumPy" as "numpy pie" - which then reach the LLM as the
    # question, so the answer is wrong for a reason that has nothing to do with
    # the LLM. Comma-separated; keep it to jargon that is actually misheard.
    deepgram_keyterms: str = (
        "SQL,NoSQL,PostgreSQL,MySQL,Redis,Kafka,NumPy,pandas,scikit-learn,"
        "PyTorch,TensorFlow,Keras,XGBoost,Kubernetes,Docker,Terraform,gRPC,"
        "GraphQL,REST API,OAuth,JWT,CI/CD,DevOps,Airflow,Spark,Hadoop,"
        "PySpark,Jupyter,CUDA,GPU,API,JSON,YAML,regex,async,await,"
        "middleware,microservice,idempotent,ORM,CRUD,schema,index,"
        "overfitting,underfitting,regularization,gradient descent,"
        "backpropagation,hyperparameter,cross-validation,precision,recall,"
        "F1 score,ROC AUC,embedding,transformer,LLM,RAG,inference,latency,"
        "throughput,big O,time complexity,linked list,binary tree,hash map"
    )
    partial_interval: float = 0.7         # seconds between partial re-decodes
    # Start decoding the finished-looking utterance this far into a pause,
    # rather than waiting out end_of_utterance_silence first. The decode then
    # runs *during* the silence instead of after it, which takes the whole STT
    # cost off the critical path. See README "Latency".
    pause_decode_after: float = 0.20
    warmup_model: bool = True             # first decode is ~2x slower otherwise

    # ---- Audio -----------------------------------------------------------
    audio_mode: str = "loopback"          # loopback | microphone
    audio_device: str = ""                # name substring or index; "" = default

    # ---- Voice activity detection ---------------------------------------
    vad_threshold_db: float = -45.0       # absolute floor, dBFS
    vad_margin_db: float = 7.0            # how far above the noise floor is speech
    min_speech_duration: float = 0.20     # ignore blips shorter than this
    end_of_utterance_silence: float = 0.70
    max_utterance_seconds: float = 30.0
    preroll_seconds: float = 0.35

    # ---- Question handling ----------------------------------------------
    context_turns: int = 5                # rolling exchanges sent to the LLM
    speculative_start: bool = True
    speculative_min_words: int = 5
    # Answer *while the speaker is still talking*, as soon as a mid-speech
    # transcript already reads as a complete question. This is the only way to
    # have text on screen before they finish. It is a deliberate gamble: if
    # they keep going and change the question, the answer is cancelled and
    # replaced, which costs extra tokens and is visible to you.
    eager_answer: bool = False
    eager_min_words: int = 6
    # 0.65, not 0.70: a mid-speech transcript rarely has a question mark yet,
    # and without one even a textbook "What is X in Y" tops out at 0.67. A
    # higher bar would mean eager answering almost never fires.
    eager_confidence: float = 0.65
    duplicate_similarity: float = 0.90
    duplicate_window: float = 45.0        # seconds

    # ---- UI --------------------------------------------------------------
    answer_mode: str = "NORMAL"           # SHORT | NORMAL | DETAILED
    font_size: int = 13
    always_on_top: bool = True
    # The latency readout is useful while tuning and clutter once you trust it.
    # Turning it off only hides the label - the numbers still go to app.log.
    show_latency: bool = True
    # The "heard: ..." line under the question, showing the raw transcript.
    show_transcript: bool = True

    # ---- Misc ------------------------------------------------------------
    save_audio: bool = False              # privacy: off by default
    log_level: str = "INFO"
    hotkeys_enabled: bool = True

    warnings: list = field(default_factory=list)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, env_file=None) -> "Settings":
        load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
        s = cls(
            llm_provider=_env("LLM_PROVIDER", "openai").lower(),
            llm_model=_env("LLM_MODEL", "gpt-4o-mini"),
            llm_base_url=_env("LLM_BASE_URL"),
            llm_fallback_model=_env("LLM_FALLBACK_MODEL"),
            openai_api_key=_env("OPENAI_API_KEY"),
            anthropic_api_key=_env("ANTHROPIC_API_KEY"),
            llm_timeout=_env_float("LLM_TIMEOUT", 30.0),
            llm_max_tokens=_env_int("LLM_MAX_TOKENS", 700),
            stt_provider=_env("STT_PROVIDER", "local").lower(),
            stt_model=_env("STT_MODEL", "tiny.en"),
            stt_compute=_env("STT_COMPUTE", "int8"),
            stt_language=_env("STT_LANGUAGE", "en"),
            deepgram_api_key=_env("DEEPGRAM_API_KEY"),
            deepgram_model=_env("DEEPGRAM_MODEL", "nova-3"),
            deepgram_url=_env("DEEPGRAM_URL", "wss://api.deepgram.com/v1/listen"),
            deepgram_endpointing=_env_int("DEEPGRAM_ENDPOINTING", 300),
            deepgram_keyterms=_env("DEEPGRAM_KEYTERMS", cls.deepgram_keyterms),
            partial_interval=_env_float("PARTIAL_INTERVAL", 0.7),
            pause_decode_after=_env_float("PAUSE_DECODE_AFTER", 0.20),
            warmup_model=_env_bool("WARMUP_MODEL", True),
            audio_mode=_env("AUDIO_MODE", "loopback").lower(),
            audio_device=_env("AUDIO_DEVICE"),
            vad_threshold_db=_env_float("VAD_THRESHOLD_DB", -45.0),
            vad_margin_db=_env_float("VAD_MARGIN_DB", 7.0),
            min_speech_duration=_env_float("MIN_SPEECH_DURATION", 0.20),
            end_of_utterance_silence=_env_float("END_OF_UTTERANCE_SILENCE", 0.70),
            max_utterance_seconds=_env_float("MAX_UTTERANCE_SECONDS", 30.0),
            preroll_seconds=_env_float("PREROLL_SECONDS", 0.35),
            context_turns=_env_int("CONTEXT_TURNS", 5),
            speculative_start=_env_bool("SPECULATIVE_START", True),
            speculative_min_words=_env_int("SPECULATIVE_MIN_WORDS", 5),
            eager_answer=_env_bool("EAGER_ANSWER", False),
            eager_min_words=_env_int("EAGER_MIN_WORDS", 6),
            eager_confidence=_env_float("EAGER_CONFIDENCE", 0.65),
            duplicate_similarity=_env_float("DUPLICATE_SIMILARITY", 0.90),
            duplicate_window=_env_float("DUPLICATE_WINDOW", 45.0),
            answer_mode=_env("ANSWER_MODE", "NORMAL").upper(),
            font_size=_env_int("FONT_SIZE", 13),
            always_on_top=_env_bool("ALWAYS_ON_TOP", True),
            show_latency=_env_bool("SHOW_LATENCY", True),
            show_transcript=_env_bool("SHOW_TRANSCRIPT", True),
            save_audio=_env_bool("SAVE_AUDIO", False),
            log_level=_env("LOG_LEVEL", "INFO").upper(),
            hotkeys_enabled=_env_bool("HOTKEYS_ENABLED", True),
        )
        s.validate()
        return s

    def validate(self) -> None:
        """Clamp nonsense values and collect human-readable warnings.

        This never raises: a bad .env should degrade, not crash the app.
        """
        self.warnings = []

        if self.llm_provider not in ("openai", "anthropic"):
            self.warnings.append(
                "Unknown LLM_PROVIDER '%s', falling back to openai." % self.llm_provider
            )
            self.llm_provider = "openai"

        if self.llm_provider == "openai" and not self.openai_api_key:
            self.warnings.append("OPENAI_API_KEY is not set - answers will fail.")
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            self.warnings.append("ANTHROPIC_API_KEY is not set - answers will fail.")

        if self.stt_provider not in ("local", "deepgram"):
            self.warnings.append(
                "Unknown STT_PROVIDER '%s', falling back to local." % self.stt_provider
            )
            self.stt_provider = "local"
        if self.stt_provider == "deepgram" and not self.deepgram_api_key:
            self.warnings.append(
                "DEEPGRAM_API_KEY is not set - falling back to local transcription."
            )
            self.stt_provider = "local"

        if self.answer_mode not in ("SHORT", "NORMAL", "DETAILED"):
            self.warnings.append(
                "Unknown ANSWER_MODE '%s', using NORMAL." % self.answer_mode
            )
            self.answer_mode = "NORMAL"

        if self.audio_mode not in ("loopback", "microphone"):
            self.warnings.append(
                "Unknown AUDIO_MODE '%s', using loopback." % self.audio_mode
            )
            self.audio_mode = "loopback"

        # Guard rails so a typo cannot make the pipeline unusable.
        self.font_size = max(8, min(32, self.font_size))
        self.context_turns = max(0, min(20, self.context_turns))
        self.partial_interval = max(0.25, min(5.0, self.partial_interval))
        self.end_of_utterance_silence = max(0.2, min(5.0, self.end_of_utterance_silence))
        self.min_speech_duration = max(0.05, min(2.0, self.min_speech_duration))
        self.max_utterance_seconds = max(5.0, min(120.0, self.max_utterance_seconds))
        self.preroll_seconds = max(0.0, min(2.0, self.preroll_seconds))
        # The pause decode must fire strictly before end-of-utterance, or it
        # buys nothing.
        self.pause_decode_after = max(
            0.05, min(self.pause_decode_after, self.end_of_utterance_silence - 0.05)
        )
        self.duplicate_similarity = max(0.5, min(1.0, self.duplicate_similarity))
        self.eager_confidence = max(0.5, min(1.0, self.eager_confidence))
        self.eager_min_words = max(3, min(30, self.eager_min_words))

    def redacted(self) -> dict:
        """asdict() with secrets masked - safe to write to the log file."""
        out = {}
        for key, value in asdict(self).items():
            if key.endswith(_SECRET_SUFFIXES):
                out[key] = ("<set:%d chars>" % len(value)) if value else "<missing>"
            else:
                out[key] = value
        return out


def setup_logging(level: str = "INFO") -> logging.Logger:
    """File + console logging. API keys never reach here (see redacted())."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    # These libraries are extremely chatty at DEBUG level.
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "faster_whisper", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger("assistant")
