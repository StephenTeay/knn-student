"""
ml/label_utils.py
==================
Tiny integer<->string label codec for the three performance classes.

WHY THIS EXISTS: while running the grid search (see ml/train.py), a subset
of (metric, weights, PCA) combinations for KNeighborsClassifier.predict_proba
raised `ValueError: invalid literal for int() with base 10: 'at_risk'`
inside scikit-learn's internal ArgKminClassMode Cython fast-path, which in
this sklearn build (1.8.0) assumes integer class labels for that
particular optimized code path even though the public API documents
predict_proba as label-type-agnostic. GridSearchCV treated each failure as
a non-fatal NaN-scored fold (visible as the "Scoring failed" warnings),
so the grid search still completed and the best params it picked were
genuinely the best *among the combinations that scored successfully* --
but combinations that happened to hit the buggy code path were silently
removed from consideration, which understates the true search space.

The fix is to never hand sklearn a string-typed target at all: every
model in this project is fit on integer-encoded labels (0/1/2, in the
exact order of config.PERFORMANCE_CLASSES so encoded order matches the
project's canonical class order), and predictions are decoded back to the
human-readable strings ('at_risk'/'satisfactory'/'high') immediately after
each .predict()/.predict_proba() call in ml/train.py, before any
evaluation, CSV export, or dashboard code ever sees them. This keeps every
other module (ml/evaluate.py, the dashboard, the Predictions table) working
with friendly strings throughout, while sidestepping the bug at its source.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PERFORMANCE_CLASSES

_ENCODE_MAP = {label: idx for idx, label in enumerate(PERFORMANCE_CLASSES)}
_DECODE_MAP = {idx: label for label, idx in _ENCODE_MAP.items()}


def encode(y) -> np.ndarray:
    return np.array([_ENCODE_MAP[v] for v in y])


def decode(y_int) -> np.ndarray:
    return np.array([_DECODE_MAP[int(v)] for v in y_int])
