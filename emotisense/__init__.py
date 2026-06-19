"""EmotiSense: hybrid journal emotion analysis (Hugging Face API + keyword fallback)."""

from .engine import (
    DEFAULT_MODEL,
    EMOTION_KEYWORDS,
    EmotionResult,
    EmotiSenseEngine,
)
from .visualize import export_csv, results_to_dataframe, visualize_results

__all__ = [
    "EmotiSenseEngine",
    "EmotionResult",
    "EMOTION_KEYWORDS",
    "DEFAULT_MODEL",
    "export_csv",
    "results_to_dataframe",
    "visualize_results",
]

__version__ = "0.1.0"
