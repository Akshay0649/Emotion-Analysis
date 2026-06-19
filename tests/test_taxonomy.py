"""Tests for the canonical emotion taxonomy (the collision fix)."""

import math

from emotisense.engine import EmotiSenseEngine
from emotisense.taxonomy import (
    CANONICAL_EMOTIONS,
    EMOTION_KEYWORDS,
    normalize_label,
    to_canonical_scores,
)


def test_keyword_keys_match_canonical():
    assert set(EMOTION_KEYWORDS) == set(CANONICAL_EMOTIONS)


def test_no_love_label_leaks():
    # "love" must not be its own label anymore — it collapses into joy.
    assert "love" not in CANONICAL_EMOTIONS
    assert normalize_label("love") == "joy"


def test_disgust_is_supported_by_keywords():
    assert "disgust" in EMOTION_KEYWORDS
    assert EMOTION_KEYWORDS["disgust"]  # non-empty


def test_normalize_label_aliases_and_casing():
    assert normalize_label("Happy") == "joy"
    assert normalize_label(" ANGER ") == "anger"
    assert normalize_label("disgusted") == "disgust"
    assert normalize_label("joy") == "joy"


def test_to_canonical_scores_full_coverage_and_merge():
    raw = {"love": 0.6, "joy": 0.2, "disgust": 0.2}
    scores = to_canonical_scores(raw)
    # Every canonical emotion present.
    assert set(scores) == set(CANONICAL_EMOTIONS)
    # love (0.6) merges into joy (0.2) -> 0.8.
    assert math.isclose(scores["joy"], 0.8)
    assert math.isclose(scores["disgust"], 0.2)


def test_to_canonical_scores_drops_unknown():
    scores = to_canonical_scores({"gibberish": 0.9, "joy": 0.1})
    assert math.isclose(scores["joy"], 0.1)
    assert "gibberish" not in scores


def test_engine_results_cover_full_taxonomy():
    engine = EmotiSenseEngine(use_api=False)
    # Both a keyword-hit entry and a no-keyword entry should span all labels.
    for text in ["I am furious and livid", "qwerty asdf zxcv", "ok", ""]:
        result = engine.analyze_emotion(text)
        assert set(result.all_emotions) == set(CANONICAL_EMOTIONS)


def test_disgust_detected_by_keywords():
    engine = EmotiSenseEngine(use_api=False)
    result = engine.analyze_emotion("that was absolutely disgusting and vile")
    assert result.primary_emotion == "disgust"
