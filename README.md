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
| `LLM_BASE_URL` | *(empty)* | Point at Groq/Cerebras for 100–250 ms first token. |
| `EAGER_ANSWER` | `false` | Answer while they are still talking. See below. |
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

105 tests covering duplicate suppression, question detection, conversation
context, configuration, latency measurement, VAD segmentation and pause
decoding, resampling, error handling, and the Deepgram streaming client
(against a mock server). They need no API key, no audio device and no network.

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

### End to end, measured against live APIs

Spoken question → answer text on screen. Nothing stubbed: real speakers, real
WASAPI loopback capture, real Deepgram, real Groq (`openai/gpt-oss-20b`),
`ANSWER_MODE=SHORT`, `EAGER_ANSWER=true`.

| Question | First word | Settled |
|---|---|---|
| "What is overfitting in machine learning" | 644 ms | 708 ms |
| "What is the difference between a list and a tuple in Python" | 490 ms | 538 ms |
| "How would you find the second highest salary in SQL" | **−443 ms** | 968 ms |

A negative number means exactly what it looks like: the answer began appearing
**before the speaker finished the sentence**. The third question also shows the
eager trade honestly — it fired early on "How would you find the second
highest", then re-asked once "salary in SQL" arrived, and settled on correct
SQL at 968 ms.

Three things had to be fixed to get here, each found by measuring rather than
reading:

- **The first question of a session took 3.3 s** against ~0.6 s for the rest,
  purely DNS + TLS handshake. The app now opens the connection at startup.
- **Deepgram dropped the socket between questions.** Audio only streams while
  someone is speaking, and Deepgram closes an idle connection after ~10 s, so
  the first question after a lull was lost while it reconnected. Fixed with
  KeepAlive frames.
- **"SQL" came through as "sequel"**, and that is what reached the model — so
  the answer was wrong for a reason that had nothing to do with the model.
  Fixed with `DEEPGRAM_KEYTERMS`.

### The headline number

Speaker stops talking → LLM request on the wire, measured through the real
pipeline (playback → loopback capture → VAD → Whisper → question gate), best of
three runs each:

| Configuration | Latency | Transcript |
|---|---|---|
| Local Whisper, naive (decode after the silence window) | 1319 ms | correct |
| Local Whisper + pause decode | 692 ms | correct |
| Local Whisper + pause decode + speculative start | 609 ms | correct |
| Deepgram, waiting for its endpointing | 664 ms | correct |
| Deepgram, `DEEPGRAM_ENDPOINTING=100` | 452 ms | correct |
| **Deepgram + `EAGER_ANSWER=true`** | **170–192 ms** | correct |

Those Deepgram rows are against the live API, not a mock.

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

### Knobs that turned out not to matter

Worth stating plainly, because it saves you fiddling with them. Measured across
three runs each, on the same audio:

| `END_OF_UTTERANCE_SILENCE` / `PAUSE_DECODE_AFTER` | best |
|---|---|
| 0.70 / 0.20 *(default)* | 782 ms |
| 0.50 / 0.20 | 801 ms |
| 0.40 / 0.20 | 778 ms |
| 0.70 / 0.12 | 810 ms |
| 0.50 / 0.12 | 771 ms |
| 0.70 / 0.06 | 777 ms |

Everything lands inside the noise. Once the decode runs during the pause, the
binding constraint is **the decode itself**, not either timer — so lowering the
silence window just buys truncation risk for nothing. Tracing confirmed it:

```
-279 ms  partial queued              <- a routine partial is still running
+203 ms  VAD emits SpeechPause       <- pause decode queued, but must wait
+250 ms  decode END                  <- partial finishes
+251 ms  decode START (full, 2.0s)
+852 ms  decode END                  <- ~600 ms under real load, not the 390 ms
+852 ms  LLM request sent               measured in isolation
```

An attempt to fix the contention — spacing partials adaptively at twice the
measured decode time — was **measured and reverted**: median 942 ms against
847 ms for the fixed interval. It made things worse, so it is not in the code.

### Getting answers before they stop talking

Two settings change the shape of this, not just the size.

**1. A provider built for speed (`LLM_BASE_URL`).** Groq and Cerebras serve the
OpenAI protocol from custom inference hardware and typically start replying in
100–250 ms rather than 400–700 ms. Nothing in the code changes — point the base
URL at them and put their key in `OPENAI_API_KEY`:

```
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-20b
OPENAI_API_KEY=gsk_...your groq key...
```

Measured on Groq, time to the first visible word of the answer:

| Model | First word | Full answer |
|---|---|---|
| **`openai/gpt-oss-20b`** | **491 ms** | 657 ms |
| `groq/compound-mini` | 727 ms | 986 ms |
| `openai/gpt-oss-120b` | 752 ms | 972 ms |
| `qwen/qwen3.6-27b` | *393 ms* | 1725 ms |

Ignore that Qwen number — it is fast only because its first token is `<think>`,
not an answer. The app strips reasoning blocks so you never see them, which
means a reasoning model simply looks slow here. Avoid them for this job.

Model catalogues change and differ per account; list yours with
`curl -H "Authorization: Bearer $KEY" https://api.groq.com/openai/v1/models`.
OpenAI's own fastest is `LLM_MODEL=gpt-4.1-nano` with no base URL.

**2. Eager answering (`EAGER_ANSWER=true`).** Normally the app waits for the
speaker to pause. With this on, the request goes out as soon as a *mid-speech*
transcript already reads as a complete question — so the answer can be on
screen before they have finished asking.

This is a gamble, and the README would be lying if it called it a free win.
"What is overfitting in machine learning" is a complete question; it is also
the first half of "What is overfitting in machine learning, and how would you
prevent it on edge hardware?". When that happens the first answer is discarded
and a second request goes out. You see the answer clear and rewrite itself, and
you pay for both.

What keeps it honest rather than merely wrong:

- The question is shown in amber with *"answering early — may update when they
  finish"*, so a provisional answer never reads as a settled one.
- A superseded answer is **cleared immediately**, not left on screen where it
  could be misread as the answer to what they actually asked.
- A fragment that trails off on a conjunction or a bare interrogative
  (*"...and why"*, *"...if the"*) never fires. Mid-speech, those mean more is
  coming — though note the same words *are* a complete question once someone
  has actually stopped, which is why that rule applies only to eager answering.
- One attempt per utterance, so a long rambling question cannot spray requests.

Worth turning on when questions are short and self-contained. Leave it off if
the speaker thinks out loud or habitually tacks on clauses.

**Measured against the live Deepgram API**, eager answering is the single
biggest win available — because Deepgram's interim transcripts are fast
(~150 ms) while its *endpointing* is not (~630 ms). Acting on the interim skips
that wait entirely:

| Deepgram configuration | Speaker stops → request sent |
|---|---|
| Wait for `speech_final` (default) | 664 ms |
| `DEEPGRAM_ENDPOINTING=100` | 452 ms |
| **`EAGER_ANSWER=true`** | **170–192 ms** |

And the failure case, measured rather than assumed. Playing *"What is
overfitting in machine learning, and how would you prevent it on edge
hardware?"* with eager on:

```
2 requests, 1 cancelled, 1 early
  - "What is overfitting in machine learning?"              <- early guess
  - "What is overfitting in machine learning and how        <- replaces it
     would you prevent it on edge hardware"
```

You get an answer to the first half almost immediately, and it is replaced by
the full answer when they finish. That is the trade, in full: faster start,
one extra request, and a visible swap.

**Combined**, with Groq and eager answering on, a self-contained question can
have its answer starting while the last few words are still being spoken. A
question with a twist at the end will still be *finished* after they stop —
there is no way around that, because the twist has not been said yet.

### Squeezing out the rest

1. **`STT_PROVIDER=deepgram` — the one that actually moves the needle.** It
   removes local transcription entirely: nothing is decoded on your CPU, and
   interims arrive in 150–300 ms. Its server-side `speech_final` marker drives
   the same early-answer path the local pause decode does, and its endpointing
   (`DEEPGRAM_ENDPOINTING=300`) can be far tighter than the local 700 ms
   because it runs on a real streaming model rather than an energy threshold.
   Needs a key from <https://deepgram.com>:
   ```
   STT_PROVIDER=deepgram
   DEEPGRAM_API_KEY=...
   ```
   The streaming client is verified against a mock Deepgram server in
   `tests/test_deepgram.py` — URL and auth, PCM encoding, interim accumulation,
   `speech_final` → early answer, `Finalize` → final, and reconnect-on-failure.
   Deepgram's own accuracy and real-world latency are not something those tests
   can measure; expect roughly 300–500 ms end of speech to request sent.
2. **A faster CPU, or `STT_MODEL=tiny.en`.** In local mode the decode *is* the
   latency. Everything else is already hidden behind it.
3. **Lower `CONTEXT_TURNS`** to `2–3`. Fewer prompt tokens, slightly faster
   first token.
4. **Keep `LLM_MODEL=gpt-4o-mini`.** `gpt-4o` roughly doubles time-to-first-token.
5. **`ANSWER_MODE=SHORT`** makes the answer *complete* sooner. It does not
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
├── tests/                  105 tests, no network or audio required
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
