"""A tiny hand-labelled seed set for ``evaluate``.

This is deliberately small and *yours to grow*. The whole point of measurement
here is to learn what the engine does — so expand this with your own entries and
honest labels, and watch the numbers move. Labels use the canonical taxonomy
(see :mod:`emotisense.taxonomy`).
"""

from typing import List, Tuple

LabeledExample = Tuple[str, str]

LABELED_SAMPLES: List[LabeledExample] = [
    ("I had an absolutely amazing day, everything went perfectly!", "joy"),
    ("I'm so grateful and blessed for my wonderful friends.", "joy"),
    ("My heart is broken and I can't stop crying tonight.", "sadness"),
    ("I feel so lonely and empty lately, nothing helps.", "sadness"),
    ("I'm furious about how unfairly they treated me today.", "anger"),
    ("I hate this, I'm livid and completely fed up.", "anger"),
    ("I'm terrified about the results and the anxiety is overwhelming.", "fear"),
    ("So nervous about tomorrow, what if I mess everything up?", "fear"),
    ("Wow, I did not see that coming at all, I'm stunned!", "surprise"),
    ("That news caught me completely off guard, unbelievable.", "surprise"),
    ("That smell was absolutely disgusting, it made me sick.", "disgust"),
    ("Honestly that behaviour is vile and repulsive.", "disgust"),
    ("Just a normal, ordinary day, nothing much happened.", "neutral"),
    ("Everything is fine, going through the usual routine.", "neutral"),
]
