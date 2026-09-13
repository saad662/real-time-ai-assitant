"""Rolling conversation memory.

Follow-up questions ("and how is that different from a decision tree?") are
meaningless without the previous turn, so we must send *some* history. But
history is the cheapest way to destroy latency: every token in the prompt is
time before the first output token.

The compromise implemented here:

* keep only the last CONTEXT_TURNS exchanges (default 5);
* store questions verbatim - they are short and carry the topic;
* store answers *truncated* to a few hundred characters, because the model only
  needs to remember what it was talking about, not what it said word for word.

That keeps the context under roughly 600 tokens even at the maximum setting.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Enough to preserve the topic and the shape of the last answer, not the prose.
_ANSWER_CONTEXT_CHARS = 320
_QUESTION_CONTEXT_CHARS = 300


@dataclass
class Exchange:
    question: str
    answer: str = ""

    def truncated_answer(self, limit: int = _ANSWER_CONTEXT_CHARS) -> str:
        answer = " ".join((self.answer or "").split())
        if len(answer) <= limit:
            return answer
        return answer[:limit].rsplit(" ", 1)[0] + " ..."


class ConversationContext:
    """Thread-safe because the pipeline writes it and the UI reads it."""

    def __init__(self, max_turns: int = 5) -> None:
        self.max_turns = max(0, max_turns)
        self._turns: deque = deque(maxlen=max(1, self.max_turns))
        self._lock = threading.Lock()
        self._open: Exchange | None = None

    # -- writing -----------------------------------------------------------
    def start_exchange(self, question: str) -> None:
        """Record a question. The answer is filled in as it streams."""
        with self._lock:
            self._open = Exchange(question=question.strip())

    def complete_exchange(self, answer: str) -> None:
        with self._lock:
            if self._open is None:
                return
            self._open.answer = (answer or "").strip()
            if self.max_turns > 0:
                self._turns.append(self._open)
            self._open = None

    def abandon_exchange(self) -> None:
        """Cancelled or superseded - never becomes context."""
        with self._lock:
            self._open = None

    def clear(self) -> None:
        with self._lock:
            self._turns.clear()
            self._open = None

    # -- reading -----------------------------------------------------------
    def has_context(self) -> bool:
        with self._lock:
            return len(self._turns) > 0

    def turns(self) -> list:
        with self._lock:
            return list(self._turns)

    def as_messages(self) -> list:
        """The completed exchanges as chat messages, oldest first."""
        messages = []
        for turn in self.turns():
            question = turn.question[:_QUESTION_CONTEXT_CHARS]
            messages.append({"role": "user", "content": question})
            answer = turn.truncated_answer()
            if answer:
                messages.append({"role": "assistant", "content": answer})
        return messages

    def topic_summary(self) -> str:
        """One line for the UI: what we have been talking about."""
        turns = self.turns()
        if not turns:
            return ""
        return " | ".join(t.question[:60] for t in turns[-3:])

    def __len__(self) -> int:
        with self._lock:
            return len(self._turns)
