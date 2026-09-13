"""Question detection, duplicate suppression, and the speculative-start gate.

This file is the one deviation from the file layout in the brief. It earns its
place: deciding *whether and when to spend an LLM call* is the single decision
that determines both perceived latency and cost, and it is pure logic with no
I/O, which makes it the easiest part of the system to unit-test.

Three responsibilities:

1. `classify()` - is this utterance a question? Punctuation is unreliable
   (Whisper adds it late, Deepgram sometimes not at all), so we score
   lexical structure instead: interrogative openers, inverted auxiliaries,
   interview-style imperatives ("walk me through..."), and follow-up markers.

2. `QuestionGate.consider()` - deduplication. A transcript that is a prefix or
   near-duplicate of something we already answered must not cost a second call.

3. Speculative start - if a *partial* already reads as a complete question and
   has stopped growing, fire early and be prepared to cancel.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher

log = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9']+")

# Words that open a direct question.
_INTERROGATIVES = {
    "what", "why", "how", "when", "where", "which", "who", "whom", "whose",
}

# Auxiliaries that, in first position, signal an inverted (yes/no) question.
_AUXILIARIES = {
    "can", "could", "would", "will", "should", "shall", "do", "does", "did",
    "is", "are", "was", "were", "am", "have", "has", "had", "may", "might",
}

# Interview phrasing that is an instruction, not a grammatical question, but
# needs an answer all the same.
_IMPERATIVE_OPENERS = (
    "explain", "describe", "tell me", "walk me through", "talk me through",
    "give me an example", "give an example", "compare", "contrast",
    "write a", "write an", "write me", "write some", "implement", "code up",
    "suppose", "imagine", "assume", "consider", "let's say", "lets say",
    "say you", "given that", "given a", "define", "outline", "summarize",
)

# Conversational follow-ups. These are short and would otherwise be rejected,
# but in context they are real questions.
_FOLLOWUP_OPENERS = (
    "and what", "and why", "and how", "but why", "but how", "so why", "so how",
    "what about", "how about", "and then", "okay and", "ok and", "and that",
    "why not", "why is that", "how so", "such as", "for example", "like what",
    "any others", "anything else", "and the", "what else",
)

# Phrases that are usually the speaker talking about themselves or the process,
# not asking us anything.
_NON_QUESTION_MARKERS = (
    "let me share my screen", "can you hear me", "can you see my screen",
    "are you able to hear", "am i audible", "sorry about that", "give me a second",
    "one moment", "thanks for joining", "nice to meet you", "how are you",
    "how's it going", "how are you doing",
)

_MIN_QUESTION_WORDS = 3

# Discourse markers people put in front of the actual question. Stripping them
# is what lets us recognise the self-correction case from the brief:
# "Can you explain... actually, how would you tune an XGBoost model?"
_LEADING_FILLERS = {
    "actually", "so", "okay", "ok", "well", "right", "um", "uh", "erm", "like",
    "yeah", "yes", "now", "alright", "anyway", "sorry", "hmm", "cool", "great",
    "and", "but", "then", "wait", "listen", "look", "just", "basically",
}


def _strip_leading_fillers(words: list) -> list:
    """Drop up to three leading discourse markers, never all of the words."""
    index = 0
    while index < len(words) and index < 3 and words[index] in _LEADING_FILLERS:
        index += 1
    return words[index:] if index < len(words) else words


@dataclass
class Classification:
    is_question: bool
    confidence: float          # 0.0 - 1.0
    reason: str = ""
    complete: bool = False     # safe to answer without waiting for more speech


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace - for comparison only."""
    return " ".join(_WORD_RE.findall((text or "").lower()))


def similarity(a: str, b: str) -> float:
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def classify(text: str, has_context: bool = False) -> Classification:
    """Score an utterance as a question.

    `has_context` relaxes the rules for short follow-ups: "and why?" is only a
    real question if something came before it.
    """
    raw = (text or "").strip()
    if not raw:
        return Classification(False, 0.0, "empty")

    norm = normalize(raw)
    words = norm.split()
    if not words:
        return Classification(False, 0.0, "no words")

    for marker in _NON_QUESTION_MARKERS:
        if marker in norm:
            return Classification(False, 0.05, "small talk: %r" % marker)

    # Openers are matched after discourse markers are removed, so "so, what is
    # X" scores the same as "what is X". Follow-up markers are matched on the
    # original text, because there the leading "and" is the signal.
    core_words = _strip_leading_fillers(words)
    core = " ".join(core_words)

    score = 0.0
    reasons = []

    if raw.rstrip().endswith("?"):
        score += 0.55
        reasons.append("question mark")

    first = core_words[0]
    second = core_words[1] if len(core_words) > 1 else ""

    if first in _INTERROGATIVES:
        score += 0.55
        reasons.append("opens with '%s'" % first)
    elif first in _AUXILIARIES and len(core_words) >= 3:
        # "can you explain X" is unambiguous; "is a decision tree a linear
        # model" needs the length bonus below to clear the bar.
        score += 0.55 if second in ("you", "we", "i", "it", "they", "there") else 0.45
        reasons.append("inverted auxiliary '%s'" % first)

    for opener in _IMPERATIVE_OPENERS:
        if core.startswith(opener):
            score += 0.45
            reasons.append("imperative '%s'" % opener)
            break

    for opener in _FOLLOWUP_OPENERS:
        if norm.startswith(opener):
            score += 0.5 if has_context else 0.25
            reasons.append("follow-up '%s'" % opener)
            break

    # Interrogative later in the sentence: "Suppose you had a million rows,
    # what would you do?" / "I'm curious how you'd optimise this."
    if not reasons or score < 0.5:
        for i, word in enumerate(core_words[1:8], start=1):
            if word in _INTERROGATIVES and i < len(core_words) - 1:
                score += 0.3
                reasons.append("embedded '%s'" % word)
                break

    if "difference between" in norm or "compared to" in norm or "versus" in norm or " vs " in norm:
        score += 0.3
        reasons.append("comparison")

    # Length shaping: very short utterances are usually acknowledgements.
    if len(words) < _MIN_QUESTION_WORDS:
        followup = any(norm.startswith(o) for o in _FOLLOWUP_OPENERS)
        if not (followup and has_context) and not raw.rstrip().endswith("?"):
            return Classification(False, min(score, 0.3), "too short (%d words)" % len(words))
        score += 0.1
    elif len(words) >= 5:
        score += 0.12

    score = max(0.0, min(1.0, score))
    is_question = score >= 0.5
    # "Complete" means: safe to send without waiting for more speech. A trailing
    # conjunction or filler almost always means the sentence is still coming.
    complete = is_question and not _looks_unfinished(words)

    return Classification(
        is_question=is_question,
        confidence=score,
        reason=", ".join(reasons) or "no question signals",
        complete=complete,
    )


_TRAILING_INCOMPLETE = {
    "and", "or", "but", "so", "the", "a", "an", "of", "to", "for", "with",
    "about", "like", "that", "if", "when", "because", "actually", "um", "uh",
    "is", "are", "was", "in", "on", "at", "between", "versus", "vs",
}


def _looks_unfinished(words: list) -> bool:
    """Heuristic for 'the speaker is mid-sentence'.

    Covers the self-interruption case from the brief:
    "Can you explain... actually, before that, what is..." - the fragment ends
    on a dangling word, so we hold off instead of answering half a question.
    """
    if not words:
        return True
    return words[-1] in _TRAILING_INCOMPLETE


# ---------------------------------------------------------------------------
# Dedup + trigger policy
# ---------------------------------------------------------------------------

@dataclass
class GateDecision:
    should_ask: bool
    question: str = ""
    reason: str = ""
    speculative: bool = False
    confidence: float = 0.0
    supersedes: bool = False   # cancel the in-flight answer and replace it


class QuestionGate:
    """Decides which transcripts become LLM calls.

    Holds the recent-question memory that makes duplicate protection work:

        "What is"                        -> too short, ignored
        "What is overfitting"            -> maybe speculative
        "What is overfitting in ML"      -> supersedes the speculative one
        "What is overfitting in ML"      -> duplicate, ignored

    One instance per session; all calls happen on the pipeline thread.
    """

    def __init__(self, settings) -> None:
        self.settings = settings
        self._recent = []              # list of (normalized_text, timestamp)
        self._in_flight = ""           # question currently being answered
        self._speculative = ""         # question sent before end-of-utterance

    # -- state -------------------------------------------------------------
    def reset(self) -> None:
        self._recent = []
        self._in_flight = ""
        self._speculative = ""

    def note_asked(self, text: str, speculative: bool = False) -> None:
        self._recent.append((normalize(text), time.monotonic()))
        self._in_flight = text
        self._speculative = text if speculative else ""
        self._prune()

    def note_answered(self) -> None:
        self._in_flight = ""
        self._speculative = ""

    def _prune(self) -> None:
        cutoff = time.monotonic() - self.settings.duplicate_window
        self._recent = [(t, ts) for t, ts in self._recent if ts >= cutoff][-20:]

    # -- the decision ------------------------------------------------------
    def is_duplicate(self, text: str) -> bool:
        """True if we already answered this, or a longer version of it."""
        self._prune()
        candidate = normalize(text)
        if not candidate:
            return True
        threshold = self.settings.duplicate_similarity
        for previous, _ts in self._recent:
            if candidate == previous:
                return True
            # Growing partials in either direction:
            #   "what is over"  vs  "what is overfitting"            (shorter)
            #   "what is overfitting" vs "what is overfitting in ml" (longer)
            # The length guard keeps a genuinely elaborated question askable.
            if candidate in previous:
                return True
            # The brief's case: "What is overfitting" was answered, then the
            # transcriber finalises "What is overfitting in machine learning".
            # Same question, more words - not a second call.
            if previous and candidate.startswith(previous):
                return True
            if similarity(candidate, previous) >= threshold:
                return True
        return False

    def consider_final(self, text: str, has_context: bool = False) -> GateDecision:
        """A finished utterance. This is the normal path."""
        result = classify(text, has_context=has_context)
        if not result.is_question:
            return GateDecision(False, reason="not a question (%s)" % result.reason,
                                confidence=result.confidence)

        # Did we already fire speculatively on (almost) this text?
        if self._speculative:
            score = similarity(text, self._speculative)
            if score >= 0.92:
                return GateDecision(
                    False,
                    reason="already answering the speculative version (%.2f)" % score,
                    confidence=result.confidence,
                )
            # The speaker said something materially different - replace it.
            return GateDecision(
                True, question=text.strip(),
                reason="final differs from speculative (%.2f)" % score,
                confidence=result.confidence, supersedes=True,
            )

        if self.is_duplicate(text):
            return GateDecision(False, reason="duplicate", confidence=result.confidence)

        return GateDecision(True, question=text.strip(), reason=result.reason,
                            confidence=result.confidence,
                            supersedes=bool(self._in_flight))

    def consider_partial(self, text: str, stable_for: float,
                         has_context: bool = False) -> GateDecision:
        """A mid-utterance transcript. This is the latency optimisation.

        We only fire early when everything lines up: speculation enabled, no
        request already in flight, the text is long enough to be a real
        question, it parses as a *complete* one, and it has stopped changing.
        Anything less and we wait for end-of-utterance, because a wrong
        speculative call costs money and shows the user a wrong answer.
        """
        s = self.settings
        if not s.speculative_start:
            return GateDecision(False, reason="speculation disabled")
        if self._in_flight:
            return GateDecision(False, reason="already answering")

        words = normalize(text).split()
        if len(words) < max(3, s.speculative_min_words):
            return GateDecision(False, reason="too short to speculate")

        # Require the transcript to have settled for most of the silence window.
        if stable_for < s.end_of_utterance_silence * 0.6:
            return GateDecision(False, reason="transcript still changing")

        result = classify(text, has_context=has_context)
        if not (result.is_question and result.complete and result.confidence >= 0.65):
            return GateDecision(False, reason="not confidently complete (%s)" % result.reason)

        if self.is_duplicate(text):
            return GateDecision(False, reason="duplicate")

        return GateDecision(True, question=text.strip(), reason="speculative: " + result.reason,
                            speculative=True, confidence=result.confidence)
