"""
ml/predict.py
==============
Scores every student with the deployment KNN model and writes the result,
including the k nearest TRAINING neighbors' research codes, into the
Predictions table -- this is what makes Section 3.6.4's "Student Detail"
dashboard view ("the profile of the three to five closest training
neighbors that support the prediction ... shown as anonymized comparison
cards") and Section 3.6.2's audit-trail design work.

Run directly: python -m ml.predict

DECISION: this script reproduces the in-sample scenario described in
Section 3.3.1's "online inference phase where the already trained model
scores incoming students" -- but since this is a synthetic-data dry run
with no genuinely held-out "incoming" cohort, every student in the
synthetic dataset is scored using the model that was (in part) fit on
their own data. That is an optimistic, in-sample inference, explicitly
NOT the held-out-test performance reported in ml/train.py's
model_comparison.csv (which is the number that should be cited as "model
performance"). The Predictions table here exists to give the dashboard
something real to display end-to-end; HANDOVER.md flags this as the first
thing to replace with genuinely out-of-sample inference once a real
"incoming cohort" exists.

DECISION (k for the neighbor display): fixed at 5, the upper end of
Section 3.6.4's stated "three to five closest training neighbors" range,
independent of whatever k the grid search chose for the classification
decision itself (which optimizes a different objective -- predictive
accuracy, not how many example cards are useful for a counselor to read).
Conflating "k used to vote on the class" with "k shown as supporting
evidence" would be a modeling/UX decision masquerading as one choice when
they're really two: the former is tuned in ml/train.py for performance,
the latter is fixed here for legibility.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ARTIFACTS_DIR, PROCESSED_DIR
from db.db_init import get_connection
from features.feature_engineering import FEATURE_COLUMNS
from ml.label_utils import decode

NEIGHBOR_DISPLAY_K = 5
CHECKPOINT_LABEL = "mid_semester_synthetic_cohort"
MODEL_VERSION = "knn_deployment_v1"


def populate_predictions(checkpoint: str = CHECKPOINT_LABEL) -> int:
    df = pd.read_parquet(PROCESSED_DIR / "feature_matrix.parquet")
    research_codes = df.index.to_numpy()
    X = df[FEATURE_COLUMNS]

    pipeline = joblib.load(ARTIFACTS_DIR / "knn_deployment.joblib")

    pred_enc = pipeline.predict(X)
    pred_labels = decode(pred_enc)
    proba = pipeline.predict_proba(X)
    confidence = proba.max(axis=1)

    # preprocess+pca only (excludes the knn step itself) so we can query
    # the fitted KNeighborsClassifier's neighbor index directly.
    transformed = pipeline[:-1].transform(X)
    knn = pipeline.named_steps["knn"]
    # query k+1 neighbors so we can drop each student's match against
    # itself (distance 0, since the deployment model is fit on all
    # students including the one being queried).
    _, neighbor_idx = knn.kneighbors(transformed, n_neighbors=NEIGHBOR_DISPLAY_K + 1)

    rows = []
    for i, code in enumerate(research_codes):
        idxs = [j for j in neighbor_idx[i] if research_codes[j] != code][:NEIGHBOR_DISPLAY_K]
        neighbor_codes = [str(research_codes[j]) for j in idxs]
        rows.append((
            str(code), checkpoint, str(pred_labels[i]), float(confidence[i]),
            json.dumps(neighbor_codes), MODEL_VERSION,
        ))

    conn = get_connection()
    try:
        conn.execute("DELETE FROM Predictions WHERE checkpoint = ?", (checkpoint,))
        conn.executemany(
            """INSERT INTO Predictions
               (research_code, checkpoint, predicted_class, confidence,
                neighbor_research_codes, model_version)
               VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
    finally:
        conn.close()

    print(f"[predict] wrote {len(rows)} predictions for checkpoint='{checkpoint}'")
    print(f"[predict] predicted class distribution:\n"
          f"{pd.Series(pred_labels).value_counts(normalize=True).round(3)}")
    return len(rows)


if __name__ == "__main__":
    populate_predictions()
