"""Tests for the measurement instrument (evaluate)."""

import pytest

from emotisense.engine import EmotiSenseEngine
from emotisense.evaluate import EvaluationReport, evaluate, format_report
from emotisense.taxonomy import CANONICAL_EMOTIONS


@pytest.fixture
def engine():
    return EmotiSenseEngine(use_api=False)


def test_perfect_predictions_give_accuracy_one(engine):
    # Texts engineered to hit obvious keywords for their label.
    examples = [
        ("I am so happy and grateful", "joy"),
        ("I feel devastated and heartbroken", "sadness"),
        ("I am furious and livid", "anger"),
    ]
    report = evaluate(engine, examples)
    assert report.accuracy == 1.0
    assert report.n == 3
    assert report.backend == "keyword"


def test_report_structure(engine):
    report = evaluate(engine, [("I am thrilled", "joy"), ("normal routine day", "neutral")])
    assert isinstance(report, EvaluationReport)
    # Confusion matrix is square over the full taxonomy.
    assert list(report.confusion.index) == CANONICAL_EMOTIONS
    assert list(report.confusion.columns) == CANONICAL_EMOTIONS
    # Per-class metrics cover every emotion.
    assert {m.emotion for m in report.per_class} == set(CANONICAL_EMOTIONS)
    assert 0.0 <= report.macro_f1 <= 1.0


def test_confusion_counts_a_mistake(engine):
    # "qwerty" has no keywords -> predicted neutral, but labelled joy: a miss.
    report = evaluate(engine, [("qwerty asdf", "joy")])
    assert report.accuracy == 0.0
    assert int(report.confusion.loc["joy", "neutral"]) == 1


def test_true_labels_are_normalized(engine):
    # "love" as a true label should normalise to joy and count as correct.
    report = evaluate(engine, [("I adore and cherish you", "love")])
    assert report.accuracy == 1.0


def test_empty_examples_raises(engine):
    with pytest.raises(ValueError):
        evaluate(engine, [])


def test_format_report_is_readable(engine):
    text = format_report(evaluate(engine, [("I am happy", "joy")]))
    assert "Accuracy" in text
    assert "Confusion matrix" in text
    assert "observed, not a target" in text
