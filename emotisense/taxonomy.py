"""Single source of truth for EmotiSense's emotion vocabulary.

Both detection backends (the Hugging Face transformer and the keyword fallback)
speak *exactly* this vocabulary. Any raw label a backend produces is normalised
into one of :data:`CANONICAL_EMOTIONS` before it ever reaches the rest of the
system, so aggregation and any future timeline stay internally consistent.

Want a different/richer emotional vocabulary later (e.g. Plutchik's 8, or
valence/arousal)? Change it here once and the whole engine follows.
"""

from __future__ import annotations

from typing import Dict, List

# The canonical emotion set. This matches the default transformer
# (j-hartmann/emotion-english-distilroberta-base), so the API backend maps
# 1:1 and the keyword backend is aligned to it below.
CANONICAL_EMOTIONS: List[str] = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust",
    "neutral",
]

# Maps raw labels (from any backend, any synonym, any casing) onto a canonical
# emotion. Labels not listed here are assumed already canonical. This is where
# the old collision is resolved: "love" no longer becomes its own competing
# label — it collapses into "joy" so both backends agree.
LABEL_ALIASES: Dict[str, str] = {
    "love": "joy",
    "happiness": "joy",
    "happy": "joy",
    "sad": "sadness",
    "sorrow": "sadness",
    "grief": "sadness",
    "angry": "anger",
    "rage": "anger",
    "scared": "fear",
    "afraid": "fear",
    "anxiety": "fear",
    "anxious": "fear",
    "surprised": "surprise",
    "shock": "surprise",
    "disgusted": "disgust",
    "neutral_": "neutral",
}


def normalize_label(label: str) -> str:
    """Collapse any raw label onto its canonical emotion."""
    key = label.strip().lower()
    return LABEL_ALIASES.get(key, key)


def to_canonical_scores(raw: Dict[str, float]) -> Dict[str, float]:
    """Project an arbitrary {label: score} dict onto the canonical emotion set.

    - Aliased labels are merged (their scores summed).
    - Every canonical emotion is present (missing ones default to 0.0), so each
      analysis covers the full vocabulary and downstream code never has to guess.
    - Unknown labels that don't map to a canonical emotion are dropped.
    """
    scores = {emotion: 0.0 for emotion in CANONICAL_EMOTIONS}
    for label, value in raw.items():
        canonical = normalize_label(label)
        if canonical in scores:
            scores[canonical] += float(value)
    return scores


# Keyword lexicon for the offline fallback, keyed strictly by canonical emotion.
# "love" words live under "joy"; "disgust" now has its own list so the fallback
# can express every emotion the transformer can.
EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "joy": ["happy", "excited", "thrilled", "delighted", "cheerful", "elated", "joyful",
            "glad", "pleased", "content", "satisfied", "grateful", "blessed", "amazing",
            "fantastic", "wonderful", "great", "awesome", "brilliant", "excellent",
            "perfect", "celebrate", "celebration",
            # folded-in "love" vocabulary
            "love", "adore", "cherish", "affection", "romantic", "devoted", "caring",
            "tender", "passionate", "fond", "smitten", "heart"],
    "sadness": ["sad", "depressed", "down", "blue", "melancholy", "gloomy", "dejected",
                "sorrowful", "unhappy", "disappointed", "hurt", "heartbroken", "crying",
                "tears", "lonely", "empty", "hopeless", "devastated", "grief", "mourning",
                "regret"],
    "anger": ["angry", "furious", "mad", "irritated", "frustrated", "annoyed", "outraged",
              "livid", "enraged", "pissed", "heated", "bothered", "rage", "hate",
              "infuriated", "resentful", "bitter", "hostile"],
    "fear": ["scared", "afraid", "fearful", "terrified", "anxious", "worried", "nervous",
             "panicked", "frightened", "concerned", "uneasy", "apprehensive", "stress",
             "stressed", "overwhelmed", "paranoid", "insecure", "vulnerable", "helpless"],
    "surprise": ["surprised", "shocked", "amazed", "astonished", "stunned", "bewildered",
                 "startled", "unexpected", "sudden", "wow", "incredible", "unbelievable",
                 "speechless"],
    "disgust": ["disgusted", "disgusting", "gross", "revolted", "revolting", "repulsed",
                "repulsive", "sickened", "nauseated", "nauseating", "vile", "nasty",
                "appalled", "loathe", "loathing", "yuck"],
    "neutral": ["okay", "fine", "normal", "regular", "usual", "typical", "routine",
                "ordinary", "standard", "average", "whatever", "meh", "alright", "decent"],
}

# Sanity check: keyword keys must be exactly the canonical set.
assert set(EMOTION_KEYWORDS) == set(CANONICAL_EMOTIONS), (
    "EMOTION_KEYWORDS keys are out of sync with CANONICAL_EMOTIONS"
)
