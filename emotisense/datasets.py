"""Stream real labelled emotion data from the Hugging Face Hub.

Uses the Hugging Face **datasets-server** REST API (``/rows``) so the engine can
explore large, real corpora over plain HTTP — no heavy local ``datasets``
dependency, just ``urllib``. Labels are mapped onto the canonical taxonomy
(:mod:`emotisense.taxonomy`), so anything you load feeds straight into
:func:`emotisense.evaluate.evaluate`.

Example
-------
>>> from emotisense.datasets import load_labeled_examples
>>> data = load_labeled_examples("emotion", limit=500)   # dair-ai/emotion
>>> data[0]
('i didnt feel humiliated', 'sadness')
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from .taxonomy import CANONICAL_EMOTIONS, normalize_label

ROWS_URL = "https://datasets-server.huggingface.co/rows"
MAX_PAGE = 100  # the API caps `length` at 100 rows per request

LabeledExample = Tuple[str, str]


@dataclass(frozen=True)
class DatasetSource:
    """Where to find text + emotion labels in a Hub dataset."""

    dataset: str
    config: str = "default"
    split: str = "train"
    text_column: str = "text"
    label_column: str = "label"


# Verified, ready-to-use emotion datasets. Pass a key here, or any raw
# "owner/name" dataset id, to ``load_labeled_examples``.
KNOWN_DATASETS: Dict[str, DatasetSource] = {
    # 6 emotions: sadness, joy, love, anger, fear, surprise (love -> joy)
    "emotion": DatasetSource("dair-ai/emotion", "split", "train", "text", "label"),
    # 4 emotions: anger, joy, optimism, sadness (optimism is dropped as unmapped)
    "tweet_eval_emotion": DatasetSource("tweet_eval", "emotion", "train", "text", "label"),
}


def _label_names(features: List[dict], label_column: str) -> Optional[List[str]]:
    """Return the ClassLabel name list for ``label_column``, if it has one."""
    for feat in features:
        if feat.get("name") == label_column and feat.get("type", {}).get("_type") == "ClassLabel":
            return feat["type"].get("names")
    return None


def _resolve_label(raw, names: Optional[List[str]]) -> Optional[str]:
    """Turn a raw label value (int index, string, or list) into a string label."""
    if isinstance(raw, list):  # multi-label datasets -> take the first label
        raw = raw[0] if raw else None
    if raw is None:
        return None
    if isinstance(raw, int) and names is not None and 0 <= raw < len(names):
        return names[raw]
    return str(raw)


def extract_examples(
    features: List[dict],
    rows: List[dict],
    source: DatasetSource,
    only_canonical: bool = True,
) -> Tuple[List[LabeledExample], int]:
    """Pure parser: rows -> [(text, canonical_label)]. Returns (examples, dropped).

    Separated from the network call so it can be unit-tested offline.
    """
    names = _label_names(features, source.label_column)
    examples: List[LabeledExample] = []
    dropped = 0
    for item in rows:
        row = item.get("row", item)
        text = row.get(source.text_column)
        label = _resolve_label(row.get(source.label_column), names)
        if not text or label is None:
            dropped += 1
            continue
        canonical = normalize_label(str(label))
        if only_canonical and canonical not in CANONICAL_EMOTIONS:
            dropped += 1
            continue
        examples.append((text, canonical))
    return examples, dropped


def _fetch_page(source: DatasetSource, offset: int, length: int, token: Optional[str]) -> dict:
    params = urllib.parse.urlencode({
        "dataset": source.dataset,
        "config": source.config,
        "split": source.split,
        "offset": offset,
        "length": length,
    })
    request = urllib.request.Request(f"{ROWS_URL}?{params}")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def load_labeled_examples(
    source: Union[str, DatasetSource],
    limit: int = 200,
    offset: int = 0,
    only_canonical: bool = True,
    token: Optional[str] = None,
) -> List[LabeledExample]:
    """Stream up to ``limit`` labelled examples from a Hub dataset.

    Parameters
    ----------
    source:
        A key in :data:`KNOWN_DATASETS`, or a raw ``"owner/name"`` dataset id
        (uses default config/split/columns), or a :class:`DatasetSource`.
    limit:
        Maximum examples to return (paged transparently in 100-row requests).
    offset:
        Row offset to start from (handy for sampling a different slice).
    only_canonical:
        Drop examples whose label doesn't map into the canonical taxonomy, so
        measurement stays meaningful.
    token:
        Optional HF token (only needed for gated/private datasets).
    """
    if isinstance(source, str):
        source = KNOWN_DATASETS.get(source, DatasetSource(source))

    collected: List[LabeledExample] = []
    position = offset
    while len(collected) < limit:
        page = _fetch_page(source, position, MAX_PAGE, token)
        rows = page.get("rows", [])
        if not rows:
            break
        examples, _ = extract_examples(page.get("features", []), rows, source, only_canonical)
        collected.extend(examples)
        position += len(rows)
        total = page.get("num_rows_total")
        if total is not None and position >= total:
            break
    return collected[:limit]


def label_distribution(examples: List[LabeledExample]) -> Counter:
    """Count examples per canonical label (for a quick look at your data bank)."""
    return Counter(label for _, label in examples)
