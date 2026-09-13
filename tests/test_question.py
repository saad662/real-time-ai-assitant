"""Question detection and duplicate suppression.

These are the rules that decide whether we spend an API call, so they get the
most test coverage in the project.
"""

import pytest

from core.question import QuestionGate, classify, normalize, similarity

QUESTIONS = [
    "Can you explain overfitting?",
    "What is the difference between supervised and unsupervised learning",
    "Why would you use random forest instead of a decision tree",
    "How would you optimize this query",
    "Suppose you had a million rows, what would you do",
    "What happens if the dataset is imbalanced",
    "Walk me through how you would design a rate limiter",
    "Explain the bias variance tradeoff",
    "Write a SQL query to find the second highest salary",
    "Tell me about inner join versus left join",
    "Is a decision tree a linear model",
    "Describe how gradient descent works",
]

NOT_QUESTIONS = [
    "So I was working on a pipeline last year",
    "That makes sense, thanks",
    "Okay",
    "Right, exactly",
    "Let me share my screen for a second",
    "Can you hear me okay?",
    "We use Postgres in production",
    "",
]


@pytest.mark.parametrize("text", QUESTIONS)
def test_detects_questions(text):
    result = classify(text)
    assert result.is_question, "%r scored %.2f (%s)" % (text, result.confidence, result.reason)


@pytest.mark.parametrize("text", NOT_QUESTIONS)
def test_rejects_non_questions(text):
    assert not classify(text).is_question, "%r was wrongly treated as a question" % text


@pytest.mark.parametrize("text", ["and why", "what about", "and then", "why not"])
def test_followups_need_context(text):
    """Short follow-ups only count once a conversation is under way."""
    assert classify(text, has_context=True).is_question
    assert not classify(text, has_context=False).is_question


def test_unfinished_questions_are_not_complete():
    """Self-interruption must not be answered mid-sentence."""
    incomplete = classify("Can you explain the difference between")
    assert not incomplete.complete
    finished = classify("Can you explain the difference between bagging and boosting")
    assert finished.complete


def test_normalize_and_similarity():
    assert normalize("  What IS overfitting?? ") == "what is overfitting"
    assert similarity("what is overfitting", "What is overfitting?") == 1.0
    assert similarity("what is a random forest", "how do i deploy to aws") < 0.5


# ---------------------------------------------------------------------------
# Duplicate protection (brief section 15)
# ---------------------------------------------------------------------------

def test_growing_partials_produce_one_call(settings):
    """The exact scenario from the brief: three transcripts, one LLM call."""
    gate = QuestionGate(settings)
    asked = []
    for text in ("What is", "What is overfitting", "What is overfitting in machine learning"):
        decision = gate.consider_final(text)
        if decision.should_ask:
            asked.append(decision.question)
            gate.note_asked(decision.question)
            gate.note_answered()
    assert len(asked) == 1, "expected one call, got %r" % (asked,)


def test_repeated_question_is_suppressed(settings):
    gate = QuestionGate(settings)
    first = gate.consider_final("What is a random forest")
    assert first.should_ask
    gate.note_asked(first.question)
    gate.note_answered()
    assert not gate.consider_final("What is a random forest?").should_ask
    assert not gate.consider_final("what is a random forest").should_ask


def test_different_question_still_gets_through(settings):
    gate = QuestionGate(settings)
    gate.note_asked("What is a random forest")
    gate.note_answered()
    decision = gate.consider_final("How is that different from a decision tree",
                                   has_context=True)
    assert decision.should_ask


def test_speculative_then_matching_final_does_not_ask_twice(settings):
    gate = QuestionGate(settings)
    spec = gate.consider_partial("What is overfitting in machine learning",
                                 stable_for=10.0)
    assert spec.should_ask and spec.speculative
    gate.note_asked(spec.question, speculative=True)

    # The final transcript is essentially the same - no second call.
    assert not gate.consider_final("What is overfitting in machine learning?").should_ask


def test_speculative_then_different_final_supersedes(settings):
    gate = QuestionGate(settings)
    spec = gate.consider_partial("Can you explain how random forests work",
                                 stable_for=10.0)
    assert spec.should_ask
    gate.note_asked(spec.question, speculative=True)

    decision = gate.consider_final("Actually, how would you tune an XGBoost model")
    assert decision.should_ask and decision.supersedes


def test_speculation_waits_for_a_stable_transcript(settings):
    gate = QuestionGate(settings)
    assert not gate.consider_partial("What is overfitting in ML", stable_for=0.05).should_ask


# ---------------------------------------------------------------------------
# Eager answering: replying before the speaker has finished
# ---------------------------------------------------------------------------

def test_eager_is_off_unless_asked_for(settings):
    gate = QuestionGate(settings)
    assert not gate.consider_eager("What is overfitting in machine learning").should_ask


def test_eager_fires_on_a_complete_mid_speech_question(settings):
    settings.eager_answer = True
    gate = QuestionGate(settings)
    decision = gate.consider_eager("What is the difference between bagging and boosting")
    assert decision.should_ask and decision.speculative


def test_eager_refuses_a_trailing_fragment(settings):
    """The whole risk of eager answering is committing mid-sentence, so a
    fragment that trails off must never fire."""
    settings.eager_answer = True
    gate = QuestionGate(settings)
    for fragment in ("What is the difference between bagging and",
                     "Can you explain how random forests work and why",
                     "How would you optimise this query if the"):
        assert not gate.consider_eager(fragment).should_ask, fragment


def test_eager_needs_more_words_than_the_pause_gate(settings):
    settings.eager_answer = True
    settings.eager_min_words = 6
    gate = QuestionGate(settings)
    assert not gate.consider_eager("What is overfitting").should_ask
    assert gate.consider_eager("What is overfitting in a deep learning model").should_ask


def test_eager_respects_the_confidence_bar(settings):
    settings.eager_answer = True
    settings.eager_confidence = 0.99
    gate = QuestionGate(settings)
    assert not gate.consider_eager("What is overfitting in machine learning").should_ask


def test_eager_does_not_stack_on_an_in_flight_answer(settings):
    settings.eager_answer = True
    gate = QuestionGate(settings)
    first = gate.consider_eager("What is overfitting in machine learning")
    assert first.should_ask
    gate.note_asked(first.question, speculative=True)
    assert not gate.consider_eager("What is overfitting in machine learning models").should_ask


def test_eager_answer_is_replaced_when_the_question_grows(settings):
    """The speaker adds a second half - the early answer must be superseded."""
    settings.eager_answer = True
    gate = QuestionGate(settings)
    early = gate.consider_eager("What is overfitting in machine learning")
    assert early.should_ask
    gate.note_asked(early.question, speculative=True)

    final = gate.consider_final(
        "What is overfitting in machine learning and how do you prevent it on edge hardware"
    )
    assert final.should_ask and final.supersedes


def test_eager_answer_is_kept_when_the_question_ends_there(settings):
    settings.eager_answer = True
    gate = QuestionGate(settings)
    early = gate.consider_eager("What is overfitting in machine learning")
    assert early.should_ask
    gate.note_asked(early.question, speculative=True)

    # Same question, just punctuated by the final decode - no second call.
    assert not gate.consider_final("What is overfitting in machine learning?").should_ask


def test_speculation_can_be_disabled(settings):
    settings.speculative_start = False
    gate = QuestionGate(settings)
    assert not gate.consider_partial("What is overfitting in ML", stable_for=10.0).should_ask
