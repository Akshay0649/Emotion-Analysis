"""Visualization and export helpers for EmotiSense results."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from .engine import EmotionResult


def results_to_dataframe(results: List[EmotionResult]) -> pd.DataFrame:
    """Convert a list of :class:`EmotionResult` into a tidy DataFrame."""
    return pd.DataFrame([r.to_dict() for r in results])


def export_csv(results: List[EmotionResult], path: str = "emotisense_results.csv") -> str:
    """Write results to ``path`` as CSV and return the path."""
    results_to_dataframe(results).to_csv(path, index=False)
    return path


def visualize_results(results: List[EmotionResult], save_path: Optional[str] = None):
    """Render a 2x2 summary grid of emotion analytics.

    Imports matplotlib/seaborn lazily so the core engine has no plotting
    dependency. Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    emotions = [r.primary_emotion for r in results]
    confidences = [r.confidence for r in results]
    word_counts = [r.word_count for r in results]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Emotion distribution pie chart
    emotion_counts = pd.Series(emotions).value_counts()
    colors = plt.cm.Set3(range(len(emotion_counts)))
    ax1.pie(emotion_counts.values, labels=emotion_counts.index, autopct="%1.1f%%", colors=colors)
    ax1.set_title("Distribution of Primary Emotions", fontsize=14, fontweight="bold")

    # 2. Confidence histogram
    ax2.hist(confidences, bins=10, alpha=0.7, color="skyblue", edgecolor="black")
    ax2.axvline(x=0.7, color="red", linestyle="--", label="Target Threshold (0.7)")
    ax2.set_xlabel("Confidence Score")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Confidence Score Distribution", fontsize=14, fontweight="bold")
    ax2.legend()

    # 3. Boxplot of confidence by emotion
    df = pd.DataFrame({"emotion": emotions, "confidence": confidences})
    sns.boxplot(data=df, x="emotion", y="confidence", ax=ax3)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
    ax3.set_title("Confidence by Emotion Type", fontsize=14, fontweight="bold")

    # 4. Word count vs confidence scatter
    ax4.scatter(word_counts, confidences, alpha=0.6, color="green")
    ax4.set_xlabel("Word Count")
    ax4.set_ylabel("Confidence Score")
    ax4.set_title("Word Count vs Confidence", fontsize=14, fontweight="bold")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
