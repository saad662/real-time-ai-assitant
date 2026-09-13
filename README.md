# Real-Time AI Assistant

Listens to what your computer is **playing** (the other person in a Google Meet,
Teams, Zoom or Discord call), transcribes it as they speak, works out when
they have asked a question, and streams a concise answer into a small
always-on-top window.

```
System audio ─▶ VAD ─▶ streaming transcription ─▶ question detection ─▶ LLM ─▶ streamed answer
```

---

## Contents

1. [What it does](#1-what-it-does)
2. [Requirements](#2-requirements)
3. [Python version](#3-python-version)
4. [Create a virtual environment](#4-create-a-virtual-environment)
5. [Install dependencies](#5-install-dependencies)
6. [Configure your API key](#6-configure-your-api-key)
7. [Select the audio input](#7-select-the-audio-input)
8. [Run it](#8-run-it)
9. [Test it](#9-test-it)
10. [Troubleshooting](#10-troubleshooting)
11. [Latency: measured numbers and how to improve them](#11-latency-measured-numbers-and-how-to-improve-them)
12. [Privacy](#12-privacy)
13. [Known limitations](#13-known-limitations)
14. [How it works](#14-how-it-works)

---

## 1. What it does

- **Captures system audio** on Windows via WASAPI loopback, so it hears the call
  without a virtual audio cable and without recording anything manually.
- **Detects speech** with an adaptive-threshold VAD that tolerates thinking
  pauses instead of chopping questions in half.
- **Transcribes incrementally** — a partial transcript appears while the person
  is still talking.
- **Decides whether it was a question**, including follow-ups like *"and why
  would you use that?"* that only make sense in context.
- **Streams the answer** token by token, so the first words appear long before
  the model has finished.
- **Measures its own latency** and shows the numbers in the window.

### The window

```
┌────────────────────────────────────────────┐
│ REAL-TIME AI ASSISTANT           ● READY   │
│ ▂▄▆█▆▄▂▁                                   │
│ [Stop] [System Audio - Speakers  ▼] [⟳]    │
│ [gpt-4o-mini ▼] [NORMAL ▼]      [13 px]    │
├────────────────────────────────────────────┤
│ QUESTION                                   │
│ What is overfitting and how do you         │
│ prevent it?                                │
├────────────────────────────────────────────┤
│ ANSWER                                     │
│ Overfitting is when a model learns the     │
│ training data closely enough that it       │
│ stops generalising to new data.            │
│                                            │
│ - Regularisation (L1/L2, dropout)          │
│ - Cross-validation to detect it early      │
│ - More or better-augmented data            │
├────────────────────────────────────────────┤
│ Type a question and press Enter...         │
│ Latency: STT 0 ms  LLM 640 ms  Total 0.64 s│
└────────────────────────────────────────────┘
```

### Hotkeys

| Shortcut | Action |
|---|---|
| `Ctrl+Shift+Space` | Start / stop listening |
| `Ctrl+Shift+C` | Clear the answer and conversation memory |
| `Ctrl+Shift+H` | Hide / show the window |
| `Ctrl+Shift+S` | Cycle SHORT → NORMAL → DETAILED |
| `Ctrl+Q` | Quit (window focused) |

These are registered system-wide, so they work while the call has focus.

---

## 2. Requirements

- **Windows 10 or 11.** The loopback capture is Windows-specific. The rest of
  the code is portable; the audio layer is not.
- **An OpenAI API key** (or an Anthropic one). This is the only thing that costs
  money — roughly $0.001 per question on `gpt-4o-mini`.
- **Internet** for the LLM. Transcription runs locally by default.
- **No GPU needed.** Nothing here assumes CUDA.
- About **400 MB of disk** for the dependencies plus ~75 MB for the speech
  model, downloaded automatically on first run.

---

## 3. Python version

**Use Python 3.11, 3.12 or 3.13.** Those have prebuilt wheels for every
dependency.

> Verified working on Python 3.14.4 as well — all dependencies installed
> cleanly. If you already have 3.14, you do not need to downgrade.

Check what you have:

```
python --version
```

If that says "Python was not found" or shows 3.9 or older, install Python from
<https://www.python.org/downloads/windows/> and **tick "Add python.exe to
PATH"** on the first screen of the installer.

---

## 4. Create a virtual environment

A virtual environment keeps these packages out of your system Python. From a
regular (non-admin) PowerShell or Command Prompt:

```
cd C:\path\to\real_time_ai_assistant
python -m venv .venv
```

Activate it:

```
.venv\Scripts\activate
```

Your prompt now starts with `(.venv)`. **You need to run this activate command
every time you open a new terminal.**

> If PowerShell says *"running scripts is disabled on this system"*, run this
> once and then try activating again:
> ```
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```
> Or just use `cmd.exe` instead of PowerShell, where the same command works
> without any policy change.

---

## 5. Install dependencies

```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This takes a few minutes — PySide6 alone is about 250 MB. It installs:

| Package | Why |
|---|---|
| `soundcard`, `sounddevice` | System-audio capture (two backends, see below) |
| `faster-whisper` | Local speech-to-text |
| `openai` | The LLM |
| `PySide6` | The desktop window |
| `pynput` | System-wide hotkeys |
| `python-dotenv`, `numpy` | Configuration and audio maths |
| `websockets` | Only used if you switch to Deepgram |
| `pytest` | Tests |

---

## 6. Configure your API key

Copy the example file:

```
copy .env.example .env
```

Open `.env` in Notepad and set your key:

```
OPENAI_API_KEY=sk-...your key here...
```

Get a key at <https://platform.openai.com/api-keys>. You need a small amount of
credit on the account.

**`.env` is in `.gitignore` and must never be committed or shared.** No key is
ever written to the log file — `logs/app.log` records only `<set:51 chars>`.

Everything else in `.env` has a working default. The settings worth knowing:

| Setting | Default | What it does |
|---|---|---|
| `LLM_MODEL` | `gpt-4o-mini` | Fast and cheap. `gpt-4o` is smarter and slower. |
| `STT_MODEL` | `tiny.en` | Speech model. `base.en` is more accurate and now affordable. |
| `PAUSE_DECODE_AFTER` | `0.20` | Start decoding this far into a pause. The main latency lever. |
| `END_OF_UTTERANCE_SILENCE` | `0.70` | Silence before a question counts as finished. |
| `ANSWER_MODE` | `NORMAL` | `SHORT`, `NORMAL` or `DETAILED`. |
| `CONTEXT_TURNS` | `5` | How many past exchanges the model can see. |
| `SPECULATIVE_START` | `true` | Start answering before the speaker stops. |

---

## 7. Select the audio input

List what is available:

```
python app.py --list-devices
```

```
Available audio devices (* = Windows default)
----------------------------------------------------------------
  [11] System Audio - Speakers (Realtek(R) Audio) * (2 ch @ 48000 Hz)
  [10] System Audio - 2 - VY249HGR (AMD High Definition Audio) (2 ch @ 48000 Hz)
  [1]  Microphone - Microphone (Realtek(R) Audio) * (2 ch @ 44100 Hz)
  ...
```

- **"System Audio"** entries capture what the computer is *playing* — this is
  what hears the other person. Pick the one marked `*`; it is the device
  Windows is currently playing through.
- **"Microphone"** entries capture *you*.

By default the app picks the default System Audio device, so usually you do not
need to set anything. To pin a specific one, put it in `.env`:

```
AUDIO_DEVICE=11
```
or
```
AUDIO_DEVICE=Speakers
```

You can also change it in the dropdown at the top of the window at any time.

### A note on the two backends

Loopback capture is the most fragile part of this app, so it has two
implementations and falls back automatically:

1. **`soundcard`** — talks to WASAPI directly. This is the one that actually
   works on current versions. Used first.
2. **`sounddevice`** — used if `soundcard` fails, and for microphone input.
   Its WASAPI loopback support depends on the bundled PortAudio build; on
   sounddevice 0.5.6 / PortAudio 19.7 it is **not** available, which is why it
   is second in line.

`--list-devices` tells you which backends are usable, and the window's title
bar tells you which one it ended up using.

---

## 8. Run it

```
python app.py
```

Then:

1. Pick a **System Audio** device in the dropdown.
2. Press **Start** (or `Ctrl+Shift+Space`).
3. The first run pauses for a few seconds while the speech model downloads.
4. Play something with speech in it. The level bar at the top should move.

Useful flags:

```
python app.py --autostart          start listening immediately
python app.py --mic                capture your microphone instead
python app.py --device 11          override the audio device
python app.py --env other.env      use a different config file
```

---

## 9. Test it

### Without audio

Check the LLM and the prompt without touching any audio plumbing:

```
python app.py --test
```

```
> What is overfitting?

Q: What is overfitting?

Overfitting is when a model fits the training data so closely that it
captures noise rather than signal, so accuracy collapses on unseen data.
...
[STT: 0 ms   LLM first token: 612 ms   Total: 0.61 s]
```

Commands inside test mode:

| Command | Effect |
|---|---|
| `:mode short` | Switch answer length |
| `:gate <text>` | Show what the question detector thinks, without calling the LLM |
| `:clear` | Clear conversation memory |
| `:quit` | Exit |

`:gate` is the quickest way to understand why something did or did not trigger:

```
> :gate and why would you use that instead
  question=True  confidence=0.92  complete=True  (opens with 'why', follow-up 'and why')

> :gate yeah that makes sense
  question=False  confidence=0.00  complete=False  (no question signals)
```

### Check the whole setup

```
python app.py --check
```

```
  [ ok ] audio device resolved: [11] System Audio - Speakers (Realtek(R) Audio) *
  [ ok ] captured 2.0s of audio via soundcard
  [ ok ] faster-whisper installed
  [ ok ] LLM configured: openai / gpt-4o-mini
All checks passed.
```

### Check audio and VAD tuning

```
python app.py --audio-test
```

Ten seconds of live level readout. The bar should move when audio plays, and
you should see `-> utterance 1: 2.4s` lines when someone speaks. If the bar
moves but no utterances are detected, lower `VAD_THRESHOLD_DB`.

### Unit tests

```
pytest -q
```

81 tests covering duplicate suppression, question detection, conversation
context, configuration, latency measurement, VAD segmentation, resampling and
error handling. They need no API key, no audio device and no network.

---

## 10. Troubleshooting

**"No audio devices were found" / the level bar never moves**

Run `python app.py --list-devices`. If no "System Audio" entry appears, Windows
has no active playback device — check Settings ▸ System ▸ Sound. If entries
appear but the bar stays flat, you probably picked a device that Windows is not
currently playing through; pick the one marked `*`.

**"Could not open ... for capture"**

The error message lists what each backend said. Most common causes:

- Another application holds the device in *exclusive mode*. Sound Control Panel
  ▸ the device ▸ Properties ▸ Advanced ▸ untick "Allow applications to take
  exclusive control".
- You unplugged or switched headphones after pressing Start. Press Stop, `⟳`,
  then Start.
- `soundcard` is not installed: `pip install soundcard`.

**Bluetooth headphones sound terrible / nothing is captured**

When a Bluetooth headset switches to its microphone profile (HFP), playback
quality drops to 8 kHz and loopback can stop working. Use wired headphones or
your laptop speakers while testing.

**It transcribes me instead of the other person**

You selected a Microphone device. Pick a "System Audio" one.

**It hears the other person *and* me**

Loopback captures everything the computer plays, which does not include your
microphone — but it *does* include the meeting app playing your own voice back
if you have monitoring enabled. Turn off mic monitoring in the call app.

**Questions get cut in half**

Increase `END_OF_UTTERANCE_SILENCE` (try `1.0`). The cost is latency.

**Statements are treated as questions**

Lower the sensitivity by setting `SPECULATIVE_START=false`, which stops the app
firing early on partial transcripts. Use `:gate <text>` in test mode to see what
scored.

**Quiet speakers are missed**

Lower `VAD_THRESHOLD_DB` to `-55`, and check the floor marker in `--audio-test`.

**"API key rejected" / "out of quota"**

Check the key in `.env` and the billing page for your provider. The app shows
these as one-line messages and keeps running.

**The first question after starting is slow**

The speech model loads on first use (1.5 s for `tiny.en`, 11 s for `base.en`,
plus a one-off download). The window says "loading speech model..." while this
happens.

**Hotkeys do not work**

`pynput` cannot see keystrokes sent to a window running as administrator unless
this app is too. The in-window shortcuts always work.

**Everything else**

`logs/app.log` has the full history, including timings for every stage. It
never contains API keys.

---

## 11. Latency: measured numbers and how to improve them

All numbers below were **measured** on this machine, not estimated:

- AMD Ryzen 7 PRO 4750U, 8 cores, no GPU used
- 2.8 second spoken question, `int8` quantisation

### The headline number

Speaker stops talking → LLM request on the wire, measured through the real
pipeline (playback → loopback capture → VAD → Whisper → question gate), best of
three runs each:

| Configuration | Latency | Transcript |
|---|---|---|
| Naive (decode after the silence window) | 1319 ms | correct |
| **+ pause decode** | **692 ms** | correct |
| **+ pause decode + speculative start** | **609 ms** | correct |

**2.2× faster, same transcript, still exactly one API call per question.**
Add the model's own time-to-first-token (400–700 ms on `gpt-4o-mini`) for the
moment text appears on screen: roughly **1.0–1.3 s** end to end.

### Where the time went, and how it was removed

**1. The silence window was dead time.** The VAD waits
`END_OF_UTTERANCE_SILENCE` (700 ms) before declaring a question finished, and
the old code then started decoding — 700 ms of waiting followed by 400 ms of
work. But everything inside that window is *silence*, so a transcript decoded
at the start of it is identical to one decoded at the end. The pipeline now
starts decoding `PAUSE_DECODE_AFTER` (200 ms) into the pause, so the decode
finishes before the window closes. Measured end-of-speech → transcript in hand:
**410 ms → 0 ms**. Speech-to-text left the critical path entirely.

The reuse is guarded by an exact test, not a guess: the VAD reports how many
samples of actual speech it has seen, and the transcript is reused only if that
number is unchanged at end-of-utterance. If the speaker resumed — even by one
20 ms frame — the numbers differ and it re-decodes.

**2. The first decode was paying a warm-up tax.** CTranslate2 allocates its
workspace lazily, making the first decode ~850 ms against ~390 ms warm. The app
now burns one throwaway decode on silence at startup (`WARMUP_MODEL=true`).

**3. `cpu_threads` was set too low.** Measured on 16 logical cores, `tiny.en`,
2.8 s utterance:

| threads | 2 | 4 | 6 | **8** | 12 | 16 |
|---|---|---|---|---|---|---|
| decode | 564 ms | 422 ms | 380 ms | **370 ms** | 427 ms | 437 ms |

Past the physical core count the SMT siblings contend and it gets *slower*. The
app now picks `min(8, logical - 2)`.

### Speech-to-text model choice

Warm decode of a 2.8 s utterance:

| Model | Download | Load | Warm decode |
|---|---|---|---|
| `tiny.en` *(default)* | ~75 MB | 1.7 s | **390 ms** |
| `base.en` | ~145 MB | 11 s | 745 ms |
| `small.en` | ~480 MB | 37 s | 4.4 s |

**Decode time barely grows with utterance length** — 2.8 s costs 390 ms and
5.3 s costs 409 ms, because Whisper pads to a 30-second window either way. So a
long question is no more expensive than a short one, and `base.en` is perfectly
usable now that the decode is hidden inside the silence window.

### Squeezing out the rest

The dominant remaining cost is the 700 ms silence window itself — that is the
price of not cutting people off mid-sentence.

1. **`STT_PROVIDER=deepgram`.** A genuine streaming model with ~300 ms
   endpointing and zero local CPU. Its `speech_final` signal drives speculation
   the same way the local pause decode does. Needs a key from
   <https://deepgram.com>:
   ```
   STT_PROVIDER=deepgram
   DEEPGRAM_API_KEY=...
   ```
2. **Lower `END_OF_UTTERANCE_SILENCE`** to `0.5`. Directly removes 200 ms. This
   is now the single biggest remaining lever, and the cost is occasionally
   cutting someone off mid-question.
3. **Lower `PAUSE_DECODE_AFTER`** to `0.12`. Starts the decode sooner so
   speculation can fire earlier. Costs some wasted decodes on mid-sentence
   pauses; correctness is unaffected.
4. **Keep `SPECULATIVE_START=true`.** Worth ~80 ms on top.
5. **Lower `CONTEXT_TURNS`** to `2–3`. Fewer prompt tokens, slightly faster
   first token.
6. **Keep `LLM_MODEL=gpt-4o-mini`.** `gpt-4o` roughly doubles time-to-first-token.
7. **`ANSWER_MODE=SHORT`** makes the answer *complete* sooner. It does not
   change time-to-first-token.

### Where the time actually goes

The window shows the real breakdown after every answer, and `logs/app.log`
records the full timeline:

```
Latency: STT: 0 ms   LLM first token: 640 ms   Total: 0.64 s |
  audio_detected=0ms speech_started=0ms partial_transcript=3050ms
  final_transcript=3060ms question_detected=3061ms llm_request_started=3063ms
  llm_first_token=3703ms llm_completed=5120ms
```

`STT: 0 ms` is the pause decode working — the transcript was already in hand
when the utterance closed. If `STT` dominates, use a smaller model or Deepgram.
If `LLM first token`
dominates, use a smaller model or reduce `CONTEXT_TURNS`.

### What is *not* claimed

Sub-second *total* latency — speaker stops to first word on screen — is not
guaranteed. Getting the request sent in 609 ms is measured and repeatable, but
the model's time-to-first-token is network- and load-dependent, so the honest
end-to-end figure is **1.0–1.3 s typical** and worse on a bad connection.

The 700 ms silence window is a deliberate floor, not an oversight: it is what
stops the app answering half a sentence. You can lower it, and the README tells
you how, but that is a trade rather than a free win.

---

## 12. Privacy

Read this before using it on a call with other people.

- **Audio is never written to disk.** It lives in memory, in a bounded queue,
  and is discarded once transcribed. There is no recording file. `SAVE_AUDIO`
  exists in the config and defaults to `false`; nothing in the current code
  writes audio even when it is true.
- **Only text leaves your machine by default.** In local mode, audio is
  transcribed on your CPU and only the resulting *question text* plus a short
  rolling context is sent to the LLM. Whole recordings are never uploaded.
- **If you set `STT_PROVIDER=deepgram`, audio does leave your machine** — it is
  streamed to Deepgram for transcription. That is the trade for the lower
  latency. Their retention policy is theirs, not this app's.
- **The log contains transcripts.** `logs/app.log` records detected questions
  and transcription events, so it contains what was said. It never contains API
  keys. Delete it when you are done, or set `LOG_LEVEL=WARNING` to record much
  less.
- **`.env` is gitignored.** So are `logs/`, `*.wav` and `recordings/`.
- **Recording or transcribing other people may require their consent**, and in
  some jurisdictions it is a legal requirement rather than a courtesy. That is
  on you, not on the software.

---

## 13. Known limitations

- **Windows only** for audio capture. The pipeline, UI and tests run anywhere;
  the loopback layer does not.
- **English by default.** `STT_LANGUAGE` and a non-`.en` model (`tiny`,
  `base`) will handle other languages, less accurately and more slowly.
- **No speaker separation.** If two people talk, everything is treated as one
  stream. There is no way to tell who asked what.
- **Overlapping speech confuses the VAD.** Crosstalk tends to produce one long
  utterance.
- **The 700 ms silence window is now the floor** in local mode. Whisper's
  decode was removed from the critical path, so what remains is mostly the
  deliberate wait that stops the app answering half a sentence.
- **Question detection is heuristic**, not a model. It handles the phrasings in
  `tests/test_question.py` well and will occasionally be wrong on unusual ones.
  It deliberately errs toward *not* firing, because a wrong answer on screen is
  worse than no answer.
- **Very long questions (>30 s)** are cut at `MAX_UTTERANCE_SECONDS` and
  processed in pieces.
- **Answers can be wrong.** It is a language model. Do not read one aloud
  without understanding it.

---

## 14. How it works

### Project layout

```
real_time_ai_assistant/
├── app.py                  entry point and CLI sub-commands
├── requirements.txt
├── .env.example            every setting, documented
├── config/settings.py      loads .env once; clamps bad values; logging setup
├── audio/
│   ├── devices.py          enumerate + resolve capture endpoints
│   └── capture.py          WASAPI loopback, resampling, bounded queue
├── speech/
│   ├── vad.py              adaptive-threshold VAD + utterance segmentation
│   └── transcriber.py      faster-whisper and Deepgram behind one interface
├── ai/
│   ├── client.py           streaming LLM calls with cancellation
│   ├── prompts.py          system prompts per answer mode
│   └── context.py          rolling, truncated conversation memory
├── core/
│   ├── pipeline.py         wires it all together
│   ├── question.py         question detection, dedup, speculative gate
│   ├── events.py           event types + thread-safe bus
│   └── latency.py          timestamps and reporting
├── ui/
│   ├── main_window.py      the window
│   └── widgets.py          status pill, level meter, styles
├── tests/                  81 tests, no network or audio required
└── logs/app.log
```

`core/question.py` is the one addition to the layout in the brief. It holds the
logic that decides whether to spend an API call, which is both the most
latency-critical decision in the system and the easiest to unit-test.

### Threads

| Thread | Does | Never does |
|---|---|---|
| Audio callback | Push frames onto a queue | Anything else — work here causes glitches |
| Pipeline worker | Drain queue, run VAD, drive STT, run the gate | Touch a widget |
| STT worker | Decode audio | Block the pipeline (partials coalesce) |
| LLM thread | Stream tokens | Survive a cancel by more than one token |
| Qt main thread | Paint | Any blocking call |

Background threads publish plain dataclasses to an `EventBus`; the window
subscribes once and re-emits them as Qt signals, which Qt marshals to the GUI
thread. `core/` contains no Qt imports at all — that is what lets `--test` run
the identical pipeline headlessly.

### Key design decisions

**Energy VAD rather than webrtcvad or Silero.** webrtcvad needs a C compiler on
Windows and Silero pulls in onnxruntime; both are real installation hazards.
An RMS detector with an adaptive noise floor is ~30 lines, has no dependencies
and is predictable. Call audio arrives already noise-suppressed by the meeting
app, which makes it work better here than it would on raw microphone input.

**Re-decoding the whole utterance for each partial.** Whisper is not a
streaming model. Stitching independently decoded fragments produces mangled
text at the seams; re-decoding from the start gives Whisper full context every
time. Partial requests coalesce — if a decode is still running when the next is
due, the older request is dropped rather than queued, so the transcriber can
never fall behind real time.

**Decode during the pause, not after it.** See the latency section — this is
the change that took 1319 ms down to 692 ms. Reuse is gated on an exact
sample-count match, so a speaker who resumes mid-sentence always gets a fresh
decode.

**Speculation fires from the pause transcript, never from a partial.** An
earlier version triggered whenever a mid-speech partial stopped changing, on
the theory that a settled transcript means a finished sentence. Measuring it
against real audio disproved that: a partial also stops changing simply because
no new decode has completed yet, and it fired on *"What is overfitting in
machine"* while the speaker was still saying *"learning"* — answering the wrong
question in roughly a third of runs. A pause transcript carries the signal a
partial lacks: the VAD has confirmed silence, and the decode covers every
speech frame. Before firing, the pipeline re-checks that not one new frame of
speech arrived while the decode was running. After that change, every measured
run produced the correct transcript in a single API call.

**Dedup on prefixes, not just equality.** `"What is"` → `"What is overfitting"`
→ `"What is overfitting in machine learning"` must cost one API call, not
three. Anything that is a prefix, a substring, or ≥90% similar to a question
answered in the last 45 seconds is suppressed.

**Buffered UI rendering.** Streaming tokens are appended as plain text on a
60 ms timer; markdown is rendered once at the end. Re-laying out a rich-text
document on every token costs more than the model spends generating it.

**Bounded audio queue that drops the oldest frames.** For a live assistant,
stale audio is worthless. A gap is better than a backlog that makes every
answer later than the last.
