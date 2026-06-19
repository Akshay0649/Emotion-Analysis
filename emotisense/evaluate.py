"""A microscope for the engine, not a scoreboard.

These helpers let you *observe* what the engine actually does on text you've
labelled yourself: where it agrees with you, where it confuses one emotion for
another, which emotions it handles well and which it doesn't. The point is
insight while you learn to model — there is no target to "pass".

Everything here is pure-Python + pandas (already a dependency); no scikit-learn
needed, so the metric definitions stay visible and hackable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .engine import EmotiSenseEngine
from .taxonomy import CANONICAL_EMOTIONS, normalize_label

# A labelled example is just (text, true_emotion).
LabeledExample = Tuple[str, str]


@dataclass
class ClassMetrics:
    """Per-emotion precision / recall / F1 and support."""

    emotion: str
    precision: float
    recall: float
    f1: float
    support: int  # how many labelled examples truly had this emotion


@dataclass
class EvaluationReport:
    """Everything observed from one evaluation run."""

    accuracy: float
    macro_f1: float
    per_class: List[ClassMetrics]
    confusion: pd.DataFrame  # rows = true, cols = predicted
    n: int
    backend: str  # "api" or "keyword" — which path actually ran

    def to_frame(self) -> pd.DataFrame:
        """Per-class metrics as a tidy DataFrame."""
        return pd.DataFrame(
            [
                {
                    "emotion": m.emotion,
                    "precision": round(m.precision, 3),
                    "recall": round(m.recall, 3),
                    "f1": round(m.f1, 3),
                    "support": m.support,
                }
                for m in self.per_class
            ]
        )


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(
    engine: EmotiSenseEngine,
    examples: Sequence[LabeledExample],
) -> EvaluationReport:
    """Run ``engine`` over labelled ``examples`` and measure what it did.

    Parameters
    ----------
    engine:
        Any configured :class:`EmotiSenseEngine` (API or keyword backend).
    examples:
        Sequence of ``(text, true_emotion)``. True labels are normalised onto
        the canonical taxonomy, so "love"/"happy" etc. are accepted.
    """
    if not examples:
        raise ValueError("No labelled examples provided.")

    labels = list(CANONICAL_EMOTIONS)
    # Confusion matrix initialised to zero counts.
    confusion = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    confusion.index.name = "true"
    confusion.columns.name = "predicted"

    backends_seen: set[str] = set()
    correct = 0

    for text, true_label in examples:
        truth = normalize_label(true_label)
        result = engine.analyze_emotion(text)
        pred = result.primary_emotion
        backends_seen.add(result.source)

        if truth in labels and pred in labels:
            confusion.loc[truth, pred] += 1
        if pred == truth:
            correct += 1

    n = len(examples)
    accuracy = _safe_div(correct, n)

    per_class: List[ClassMetrics] = []
    f1_values: List[float] = []
    for emotion in labels:
        tp = int(confusion.loc[emotion, emotion])
        fp = int(confusion[emotion].sum() - tp)          # predicted emotion, wrong truth
        fn = int(confusion.loc[emotion].sum() - tp)      # truly emotion, predicted else
        support = int(confusion.loc[emotion].sum())

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        f1_values.append(f1)
        per_class.append(ClassMetrics(emotion, precision, recall, f1, support))

    macro_f1 = _safe_div(sum(f1_values), len(f1_values))
    backend = "+".join(sorted(backends_seen)) if backends_seen else "unknown"

    return EvaluationReport(
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion=confusion,
        n=n,
        backend=backend,
    )


def format_report(report: EvaluationReport) -> str:
    """Render an :class:`EvaluationReport` as readable plain text."""
    lines: List[str] = []
    lines.append(f"Examples evaluated : {report.n}")
    lines.append(f"Backend used       : {report.backend}")
    lines.append(f"Accuracy           : {report.accuracy:.3f}  (observed, not a target)")
    lines.append(f"Macro F1           : {report.macro_f1:.3f}")
    lines.append("")
    lines.append("Per-emotion behaviour:")
    lines.append(report.to_frame().to_string(index=False))
    lines.append("")
    lines.append("Confusion matrix (rows = your label, cols = engine's guess):")
    lines.append(report.confusion.to_string())
    return "\n".join(lines)
