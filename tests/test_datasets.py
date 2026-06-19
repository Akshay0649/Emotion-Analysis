"""Offline tests for the Hugging Face data-bank loader (parsing logic only)."""

from emotisense.datasets import (
    DatasetSource,
    extract_examples,
    label_distribution,
)

# Mirrors the real shape returned by datasets-server for dair-ai/emotion.
EMOTION_FEATURES = [
    {"name": "text", "type": {"dtype": "string", "_type": "Value"}},
    {"name": "label", "type": {
        "names": ["sadness", "joy", "love", "anger", "fear", "surprise"],
        "_type": "ClassLabel",
    }},
]
EMOTION_ROWS = [
    {"row": {"text": "i didnt feel humiliated", "label": 0}},      # sadness
    {"row": {"text": "i feel so blessed and happy", "label": 1}},  # joy
    {"row": {"text": "i adore and cherish them", "label": 2}},     # love -> joy
]

SOURCE = DatasetSource("dair-ai/emotion", "split", "train", "text", "label")


def test_extract_maps_int_labels_to_names():
    examples, dropped = extract_examples(EMOTION_FEATURES, EMOTION_ROWS, SOURCE)
    assert dropped == 0
    assert examples[0] == ("i didnt feel humiliated", "sadness")
    assert examples[1][1] == "joy"


def test_love_label_collapses_to_joy():
    examples, _ = extract_examples(EMOTION_FEATURES, EMOTION_ROWS, SOURCE)
    # The "love" row (index 2) must normalise to canonical "joy".
    assert examples[2][1] == "joy"


def test_only_canonical_drops_unmapped_labels():
    features = [
        {"name": "text", "type": {"dtype": "string", "_type": "Value"}},
        {"name": "label", "type": {"names": ["joy", "optimism"], "_type": "ClassLabel"}},
    ]
    rows = [
        {"row": {"text": "what a great day", "label": 0}},   # joy -> kept
        {"row": {"text": "things will get better", "label": 1}},  # optimism -> dropped
    ]
    src = DatasetSource("x", "default", "train", "text", "label")
    kept, dropped = extract_examples(features, rows, src, only_canonical=True)
    assert len(kept) == 1
    assert dropped == 1
    assert kept[0][1] == "joy"


def test_string_labels_are_handled():
    features = [
        {"name": "text", "type": {"dtype": "string", "_type": "Value"}},
        {"name": "label", "type": {"dtype": "string", "_type": "Value"}},  # not ClassLabel
    ]
    rows = [{"row": {"text": "i am angry", "label": "anger"}}]
    src = DatasetSource("x", "default", "train", "text", "label")
    kept, _ = extract_examples(features, rows, src)
    assert kept == [("i am angry", "anger")]


def test_multilabel_takes_first():
    features = [
        {"name": "text", "type": {"dtype": "string", "_type": "Value"}},
        {"name": "labels", "type": {
            "names": ["joy", "anger", "fear"], "_type": "ClassLabel"}},
    ]
    rows = [{"row": {"text": "mixed feelings", "labels": [2, 0]}}]  # fear, joy -> fear
    src = DatasetSource("x", "default", "train", "text", "labels")
    kept, _ = extract_examples(features, rows, src)
    assert kept[0][1] == "fear"


def test_label_distribution_counts():
    dist = label_distribution([("a", "joy"), ("b", "joy"), ("c", "anger")])
    assert dist["joy"] == 2
    assert dist["anger"] == 1
