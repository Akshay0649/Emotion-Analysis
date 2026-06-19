"""Core emotion-analysis engine for EmotiSense.

Provides :class:`EmotiSenseEngine`, a hybrid emotion classifier that prefers the
Hugging Face Inference API and gracefully falls back to a keyword-based detector
when the API is unavailable, unconfigured, or fails.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import requests

from .taxonomy import (
    CANONICAL_EMOTIONS,
    EMOTION_KEYWORDS,
    to_canonical_scores,
)

logger = logging.getLogger("emotisense")

DEFAULT_MODEL = "j-hartmann/emotion-english-distilroberta-base"
DEFAULT_API_URL = "https://api-inference.huggingface.co/models/{model}"


@dataclass
class EmotionResult:
    """Structured outcome of a single emotion analysis."""

    text: str
    primary_emotion: str
    confidence: float
    all_emotions: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    word_count: int = 0
    source: str = "keyword"  # "api" or "keyword"

    def top_emotions(self, n: int = 3, threshold: float = 0.05) -> List[tuple[str, float]]:
        """Return the top ``n`` emotions above ``threshold``, highest first."""
        ranked = sorted(self.all_emotions.items(), key=lambda kv: kv[1], reverse=True)
        return [(k, v) for k, v in ranked if v > threshold][:n]

    def to_dict(self) -> Dict[str, object]:
        """Flatten the result for CSV/DataFrame export."""
        return {
            "text": self.text,
            "primary_emotion": self.primary_emotion,
            "confidence": round(self.confidence, 4),
            "word_count": self.word_count,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
        }


class EmotiSenseEngine:
    """Hybrid emotion classifier (Hugging Face API + keyword fallback).

    Parameters
    ----------
    use_api:
        Whether to attempt the Hugging Face Inference API. When ``False`` the
        engine always uses the keyword fallback.
    api_key:
        Hugging Face token. If omitted, the ``HF_TOKEN`` environment variable is
        used. With no token available the engine falls back to keywords.
    model:
        Hugging Face model id to query.
    timeout:
        Per-request timeout, in seconds.
    """

    def __init__(
        self,
        use_api: bool = True,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        timeout: float = 15.0,
    ) -> None:
        self.api_key = api_key or os.getenv("HF_TOKEN")
        self.use_api = bool(use_api and self.api_key)
        self.model = model
        self.api_url = DEFAULT_API_URL.format(model=model)
        self.timeout = timeout
        self.emotion_keywords = EMOTION_KEYWORDS
        self._session = requests.Session()

    # ------------------------------------------------------------------ #
    # Text utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def preprocess_text(text: str) -> str:
        """Lowercase, collapse whitespace, and strip noise punctuation."""
        text = re.sub(r"\s+", " ", text.lower().strip())
        text = re.sub(r"[^\w\s.,!?-]", "", text)
        return text.strip()

    # ------------------------------------------------------------------ #
    # Detection strategies
    # ------------------------------------------------------------------ #
    def keyword_based_detection(self, text: str) -> Dict[str, float]:
        """Score emotions by keyword frequency, normalised to sum to 1.0."""
        words = self.preprocess_text(text).split()
        scores = {emotion: 0.0 for emotion in self.emotion_keywords}
        for word in words:
            for emotion, keywords in self.emotion_keywords.items():
                if word in keywords:
                    scores[emotion] += 1.0

        total = sum(scores.values())
        if total > 0:
            return {k: v / total for k, v in scores.items()}
        scores = {k: 0.0 for k in scores}
        scores["neutral"] = 1.0
        return scores

    def _api_detection(self, text: str) -> Optional[Dict[str, float]]:
        """Query the Hugging Face API. Returns ``None`` on any failure."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = self._session.post(
                self.api_url, headers=headers, json={"inputs": text}, timeout=self.timeout
            )
        except requests.RequestException as exc:
            logger.warning("API request failed: %s", exc)
            return None

        if response.status_code != 200:
            logger.warning("API error %s: %s", response.status_code, response.text[:200])
            return None

        try:
            data = response.json()
        except ValueError:
            logger.warning("API returned non-JSON response")
            return None

        # The model returns [[{"label": .., "score": ..}, ...]]. Project the raw
        # labels onto the canonical taxonomy so the API and keyword backends
        # always speak the same emotional vocabulary.
        if isinstance(data, list) and data and isinstance(data[0], list):
            raw = {item["label"]: item["score"] for item in data[0]}
            return to_canonical_scores(raw)
        logger.warning("Unexpected API response format: %r", data)
        return None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def analyze_emotion(self, text: str) -> EmotionResult:
        """Analyse a single piece of text and return an :class:`EmotionResult`."""
        if not text or len(text.strip()) < 3:
            neutral_scores = {emotion: 0.0 for emotion in CANONICAL_EMOTIONS}
            neutral_scores["neutral"] = 1.0
            return EmotionResult(
                text=text, primary_emotion="neutral", confidence=0.0,
                all_emotions=neutral_scores, word_count=0, source="keyword",
            )

        if self.use_api:
            emotions = self._api_detection(text)
            if emotions:
                primary = max(emotions, key=emotions.get)
                return EmotionResult(
                    text=text, primary_emotion=primary, confidence=emotions[primary],
                    all_emotions=emotions, word_count=len(text.split()), source="api",
                )

        emotions = self.keyword_based_detection(text)
        primary = max(emotions, key=emotions.get)
        return EmotionResult(
            text=text, primary_emotion=primary, confidence=emotions[primary],
            all_emotions=emotions, word_count=len(text.split()), source="keyword",
        )

    def batch_analyze(self, texts: List[str]) -> List[EmotionResult]:
        """Analyse many texts, preserving order."""
        return [self.analyze_emotion(text) for text in texts]
