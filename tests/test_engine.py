"""Unit tests for the EmotiSense engine (keyword path — no network needed)."""

import math

import pytest

from emotisense.engine import EmotiSenseEngine, EmotionResult


@pytest.fixture
def engine():
    # Force the offline keyword path so tests are deterministic and hermetic.
    return EmotiSenseEngine(use_api=False)


def test_keyword_scores_sum_to_one(engine):
    scores = engine.keyword_based_detection("I am so happy and grateful and joyful")
    assert math.isclose(sum(scores.values()), 1.0, rel_tol=1e-9)


def test_detects_joy(engine):
    result = engine.analyze_emotion("I feel amazing, wonderful and thrilled today!")
    assert result.primary_emotion == "joy"
    assert result.confidence > 0
    assert result.source == "keyword"


def test_detects_anger(engine):
    result = engine.analyze_emotion("I am furious and livid, I hate this so much")
    assert result.primary_emotion == "anger"


def test_no_keywords_returns_neutral(engine):
    result = engine.analyze_emotion("The quick brown fox jumps over the lazy dog")
    assert result.primary_emotion == "neutral"
    assert math.isclose(result.confidence, 1.0)


def test_empty_or_short_text_is_neutral(engine):
    for text in ["", "  ", "hi"]:
        result = engine.analyze_emotion(text)
        assert result.primary_emotion == "neutral"
        assert result.word_count == 0


def test_word_count(engine):
    result = engine.analyze_emotion("I am very happy today")
    assert result.word_count == 5


def test_batch_preserves_order_and_length(engine):
    texts = ["I am happy", "I am sad", "I am angry"]
    results = engine.batch_analyze(texts)
    assert len(results) == 3
    assert [r.text for r in results] == texts


def test_preprocess_strips_noise(engine):
    # Keeps sentence punctuation (.,!?-), drops other symbols, collapses spaces.
    assert engine.preprocess_text("  Hello,   WORLD!!! @#$ ") == "hello, world!!!"


def test_use_api_disabled_without_token():
    # use_api should resolve to False when no token is available.
    engine = EmotiSenseEngine(use_api=True, api_key=None)
    if engine.api_key is None:
        assert engine.use_api is False


def test_emotion_result_helpers():
    result = EmotionResult(
        text="hello world", primary_emotion="joy", confidence=0.8,
        all_emotions={"joy": 0.8, "sadness": 0.15, "anger": 0.04, "fear": 0.01},
        word_count=2, source="api",
    )
    top = result.top_emotions(n=2)
    assert top[0] == ("joy", 0.8)
    assert len(top) == 2  # anger (0.04) and fear (0.01) fall below threshold
    d = result.to_dict()
    assert d["primary_emotion"] == "joy"
    assert d["source"] == "api"
