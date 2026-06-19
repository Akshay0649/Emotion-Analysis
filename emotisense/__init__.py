"""EmotiSense: hybrid journal emotion analysis (Hugging Face API + keyword fallback)."""

from .engine import (
    DEFAULT_MODEL,
    EmotionResult,
    EmotiSenseEngine,
)
from .datasets import (
    KNOWN_DATASETS,
    DatasetSource,
    label_distribution,
    load_labeled_examples,
)
from .evaluate import EvaluationReport, evaluate, format_report
from .taxonomy import (
    CANONICAL_EMOTIONS,
    EMOTION_KEYWORDS,
    normalize_label,
    to_canonical_scores,
)
from .visualize import export_csv, results_to_dataframe, visualize_results

__all__ = [
    "EmotiSenseEngine",
    "EmotionResult",
    "CANONICAL_EMOTIONS",
    "EMOTION_KEYWORDS",
    "normalize_label",
    "to_canonical_scores",
    "DEFAULT_MODEL",
    "evaluate",
    "format_report",
    "EvaluationReport",
    "load_labeled_examples",
    "label_distribution",
    "KNOWN_DATASETS",
    "DatasetSource",
    "export_csv",
    "results_to_dataframe",
    "visualize_results",
]

__version__ = "0.1.0"
