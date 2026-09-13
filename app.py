"""Real-Time AI Interview Assistant - entry point.

    python app.py                  launch the assistant window
    python app.py --test           type questions, no audio, no window
    python app.py --list-devices   show every capturable audio device
    python app.py --check          verify config, audio and API access
    python app.py --audio-test     10 seconds of live level + VAD readout
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow `python app.py` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import LOG_DIR, Settings, setup_logging  # noqa: E402


def _banner(settings: Settings) -> str:
    return (
        "Real-Time AI Assistant\n"
        "  LLM:    %s / %s\n"
        "  Speech: %s / %s\n"
        "  Audio:  %s%s\n"
        "  Logs:   %s"
        % (
            settings.llm_provider, settings.llm_model,
            settings.stt_provider,
            settings.stt_model if settings.stt_provider == "local" else settings.deepgram_model,
            settings.audio_mode,
            (" (%s)" % settings.audio_device) if settings.audio_device else " (default device)",
            LOG_DIR / "app.log",
        )
    )


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_list_devices() -> int:
    from audio.devices import describe_devices, loopback_supported, soundcard_available

    sd_loopback = loopback_supported()
    sc_loopback = soundcard_available()

    print(describe_devices())
    print("Loopback backends")
    print("  soundcard (primary):        %s"
          % ("available" if sc_loopback else "NOT installed"))
    print("  sounddevice WASAPI loopback: %s"
          % ("available" if sd_loopback else "not supported by this PortAudio build"))

    if not (sd_loopback or sc_loopback):
        print(
            "\nNeither loopback backend is available, so system audio cannot be\n"
            "captured. Install the fallback:\n"
            "    pip install soundcard\n"
            "Or capture your microphone instead:\n"
            "    python app.py --mic\n"
        )
    elif not sc_loopback:
        print("\nInstall `soundcard` for the most reliable loopback: pip install soundcard\n")
    return 0


def cmd_check(settings: Settings) -> int:
    """Fail fast, with a specific reason, before you debug in a real call."""
    from ai.client import LLMClient, LLMError
    from audio.capture import AudioCapture
    from audio.devices import AudioDeviceError, resolve_device

    ok = True
    print(_banner(settings))
    print()

    for warning in settings.warnings:
        print("  [warn] %s" % warning)

    # 1. audio ------------------------------------------------------------
    try:
        device = resolve_device(settings.audio_device, settings.audio_mode)
        print("  [ ok ] audio device resolved: %s" % device)
        capture = AudioCapture(device)
        capture.start()
        heard = 0
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            block = capture.read(timeout=0.3)
            if block is not None:
                heard += block.size
        capture.stop()
        if heard:
            print("  [ ok ] captured %.1fs of audio via %s"
                  % (heard / 16000.0, capture.backend))
        else:
            ok = False
            print("  [FAIL] device opened but produced no audio. Is anything playing?")
    except AudioDeviceError as exc:
        ok = False
        print("  [FAIL] audio: %s" % exc)
    except Exception as exc:
        ok = False
        print("  [FAIL] audio: %s" % exc)

    # 2. speech-to-text ---------------------------------------------------
    if settings.stt_provider == "local":
        try:
            import faster_whisper  # noqa: F401
            print("  [ ok ] faster-whisper installed (model downloads on first run)")
        except Exception as exc:
            ok = False
            print("  [FAIL] faster-whisper missing: %s" % exc)
    else:
        print("  [ ok ] using Deepgram streaming (%s)" % settings.deepgram_model)

    # 3. LLM ---------------------------------------------------------------
    try:
        print("  [ ok ] %s" % LLMClient(settings).preflight())
    except LLMError as exc:
        ok = False
        print("  [FAIL] %s" % exc)
    except Exception as exc:
        ok = False
        print("  [FAIL] LLM: %s" % exc)

    print("\n%s" % ("All checks passed." if ok else "Some checks failed - see above."))
    return 0 if ok else 1


def cmd_audio_test(settings: Settings, seconds: float = 10.0) -> int:
    """Live VAD readout. The fastest way to tune the thresholds."""
    import numpy as np

    from audio.capture import AudioCapture
    from audio.devices import resolve_device
    from speech.vad import Level, SpeechEnd, SpeechStart, VoiceActivityDetector

    device = resolve_device(settings.audio_device, settings.audio_mode)
    print("Listening on: %s" % device)
    print("Play some audio (a video, a call). Ctrl+C to stop early.\n")

    capture = AudioCapture(device)
    capture.start()
    vad = VoiceActivityDetector(settings)
    deadline = time.monotonic() + seconds
    utterances = 0
    try:
        while time.monotonic() < deadline:
            block = capture.read(timeout=0.25)
            if block is None:
                continue
            for event in vad.process(block):
                if isinstance(event, Level):
                    bars = int(max(0, min(30, (event.rms_db + 70) / 2)))
                    sys.stdout.write(
                        "\r  %-30s %6.1f dB  floor %6.1f dB  %s   "
                        % ("#" * bars, event.rms_db, event.noise_floor_db,
                           "SPEECH" if event.is_speech else "      ")
                    )
                    sys.stdout.flush()
                elif isinstance(event, SpeechStart):
                    print("\n  -> speech started")
                elif isinstance(event, SpeechEnd):
                    utterances += 1
                    rms = float(np.sqrt(np.mean(event.audio ** 2)))
                    print("\n  -> utterance %d: %.1fs (rms %.4f)"
                          % (utterances, event.duration, rms))
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()

    print("\n\nDetected %d utterance(s)." % utterances)
    if utterances == 0:
        print(
            "Nothing detected. Either the wrong device is selected "
            "(`python app.py --list-devices`), or the level never crossed the "
            "threshold - try lowering VAD_THRESHOLD_DB (e.g. -55) in .env."
        )
    return 0


def cmd_test(settings: Settings) -> int:
    """Headless REPL: exercises gate -> context -> LLM -> streaming, no audio."""
    from core.events import (
        AnswerChunk,
        AnswerCompleted,
        ErrorEvent,
        LatencyReport,
        QuestionDetected,
    )
    from core.pipeline import Pipeline

    pipeline = Pipeline(settings)
    done = {"flag": False}

    def on_event(event) -> None:
        if isinstance(event, QuestionDetected):
            print("\nQ: %s\n" % event.text)
        elif isinstance(event, AnswerChunk):
            sys.stdout.write(event.text)
            sys.stdout.flush()
        elif isinstance(event, AnswerCompleted):
            print()
            done["flag"] = True
        elif isinstance(event, LatencyReport):
            print("\n[%s]" % event.as_line())
        elif isinstance(event, ErrorEvent):
            print("\n[error] %s" % event.message)
            done["flag"] = True

    pipeline.bus.subscribe(on_event)

    print(_banner(settings))
    print(
        "\nTest mode - no audio, no window.\n"
        "Type a question and press Enter. Commands:\n"
        "  :mode short|normal|detailed   change answer length\n"
        "  :gate <text>                  run it through question detection only\n"
        "  :clear                        clear conversation memory\n"
        "  :quit                         exit\n"
    )

    while True:
        try:
            # Strip the BOM Windows prepends when input is piped rather than
            # typed - otherwise ":quit" arrives as "﻿:quit".
            line = input("\n> ").replace("﻿", "").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in (":quit", ":q", "quit", "exit"):
            break
        if line == ":clear":
            pipeline.clear_conversation()
            print("Conversation cleared.")
            continue
        if line.startswith(":mode"):
            parts = line.split()
            if len(parts) > 1:
                pipeline.set_answer_mode(parts[1])
                print("Answer mode: %s" % settings.answer_mode)
            continue
        if line.startswith(":gate"):
            from core.question import classify
            candidate = line[len(":gate"):].strip()
            result = classify(candidate, has_context=pipeline.context.has_context())
            print("  question=%s  confidence=%.2f  complete=%s  (%s)"
                  % (result.is_question, result.confidence, result.complete, result.reason))
            continue

        done["flag"] = False
        pipeline.submit_text(line, force=True)
        # Wait for the stream to finish before showing the next prompt.
        deadline = time.monotonic() + settings.llm_timeout + 5
        while not done["flag"] and time.monotonic() < deadline:
            time.sleep(0.05)
        if not done["flag"]:
            print("\n[timed out waiting for the model]")

    pipeline.shutdown()
    return 0


def cmd_gui(settings: Settings, autostart: bool) -> int:
    try:
        from ui.main_window import run_gui
    except Exception as exc:
        print(
            "Could not load the user interface: %s\n"
            "Install the GUI dependency with:  pip install PySide6" % exc
        )
        return 1

    from core.pipeline import Pipeline

    pipeline = Pipeline(settings)
    try:
        return run_gui(settings, pipeline, autostart=autostart)
    finally:
        pipeline.shutdown()


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="Real-time speech -> transcription -> AI answer assistant.",
    )
    parser.add_argument("--test", action="store_true",
                        help="headless text mode: type questions, no audio")
    parser.add_argument("--list-devices", action="store_true",
                        help="list audio devices and exit")
    parser.add_argument("--check", action="store_true",
                        help="verify configuration, audio capture and API access")
    parser.add_argument("--audio-test", action="store_true",
                        help="show live audio levels and VAD decisions")
    parser.add_argument("--autostart", action="store_true",
                        help="start listening as soon as the window opens")
    parser.add_argument("--device", default=None,
                        help="override AUDIO_DEVICE (index or name substring)")
    parser.add_argument("--mic", action="store_true",
                        help="capture the microphone instead of system audio")
    parser.add_argument("--env", default=None, help="path to an alternative .env file")
    args = parser.parse_args(argv)

    settings = Settings.load(args.env)
    if args.device is not None:
        settings.audio_device = args.device
    if args.mic:
        settings.audio_mode = "microphone"

    log = setup_logging(settings.log_level)
    log.info("Starting: %s", settings.redacted())
    for warning in settings.warnings:
        log.warning("Config: %s", warning)

    try:
        if args.list_devices:
            return cmd_list_devices()
        if args.check:
            return cmd_check(settings)
        if args.audio_test:
            return cmd_audio_test(settings)
        if args.test:
            return cmd_test(settings)
        return cmd_gui(settings, autostart=args.autostart)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        log.exception("Fatal error")
        print("\nUnexpected error: %s\nSee %s for the traceback."
              % (exc, LOG_DIR / "app.log"))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
