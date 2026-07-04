"""
features/feature_engineering.py
================================
Turns the normalized survey-item and LMS-log tables in SQLite into the
composite feature matrix described in Section 3.3.2:

    "The EI part has four branch scores, ... Engagement has three
    dimension scores, Behavioral Engagement Index (BEI, built from LMS
    metrics), Emotional Engagement Score (EES, from the survey) and
    Cognitive Engagement Score (CES, also from the survey). On top of
    that, demographic covariates are fed in as control features ...
    the full feature vector ends up containing eleven main predictors."

The eleven predictors produced here are:
    4 EI branches (PE, UE, UndE, ME)
  + 3 engagement dimensions (BEI, EES, CES)
  + 4 demographic covariates (age_bracket, gender, programme_type, year_of_study)
  = 11

DECISION: branch/dimension scores are computed as the mean of their
constituent Likert items, the standard MSCEIT/OSE branch-scoring
convention, using the *mid_semester* survey wave for EI and self-report
engagement (this is the snapshot actually available at prediction time,
per Section 3.4.4's "the current setup gives a prediction at mid-semester"
design decision -- using the start-of-semester wave instead would leak
information from before the behavioral data window and wouldn't match
what the deployed system actually has available at inference time).

DECISION: BEI (the one LMS-derived score) is built as a weighted
composite of the five raw LMS signals (login_count, avg_submission_lag_hrs,
forum_posts, video_completion_rate, quiz_attempts), z-scored *within this
function* before combining so no single signal with a larger numeric range
dominates the index purely due to units (e.g. login_count integers vs.
video_completion_rate in [0,1]). This within-feature standardization is
distinct from -- and happens before -- the StandardScaler step in the ML
pipeline (ml/preprocessing.py), which scales the final assembled feature
matrix as a whole; without this index-construction step, BEI itself would
just be "average login count", which is exactly the single-source
behavioral measure Section 2.8.2 says is outperformed by genuinely
multimodal indices.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EI_BRANCHES, ENGAGEMENT_SELF_REPORT_DIMENSIONS, PROCESSED_DIR
from db.db_init import get_connection

MID_SEMESTER_WAVE = "mid_semester"
FEATURE_COLUMNS = (
    list(EI_BRANCHES.keys())  # PE, UE, UndE, ME
    + ["BEI"] + list(ENGAGEMENT_SELF_REPORT_DIMENSIONS.keys())  # BEI, EES, CES
    + ["age_bracket", "gender", "programme_type", "year_of_study"]
)


def _branch_scores(surveys: pd.DataFrame, instrument: str, wave: str) -> pd.DataFrame:
    """Mean of item-level Likert responses per (student, construct)."""
    subset = surveys[(surveys["instrument"] == instrument) & (surveys["wave"] == wave)]
    pivot = subset.groupby(["research_code", "construct_code"])["response_value"].mean().unstack()
    return pivot


def _build_bei(lms: pd.DataFrame) -> pd.Series:
    """Behavioral Engagement Index: z-scored weighted blend of the five
    raw LMS signals, averaged across the semester-to-date weeks available.
    Submission lag is sign-flipped before scoring since *lower* (earlier)
    lag indicates *higher* engagement."""
    agg = lms.groupby("research_code").agg(
        login_count=("login_count", "mean"),
        avg_submission_lag_hrs=("avg_submission_lag_hrs", "mean"),
        forum_posts=("forum_posts", "mean"),
        video_completion_rate=("video_completion_rate", "mean"),
        quiz_attempts=("quiz_attempts", "mean"),
    )
    agg["avg_submission_lag_hrs"] = -agg["avg_submission_lag_hrs"]  # flip so higher = better

    z = (agg - agg.mean()) / agg.std(ddof=0).replace(0, 1)
    weights = {
        "login_count": 0.25, "avg_submission_lag_hrs": 0.20, "forum_posts": 0.15,
        "video_completion_rate": 0.25, "quiz_attempts": 0.15,
    }
    bei_z = sum(z[col] * w for col, w in weights.items())
    # Rescale the composite z-score back onto a 1-5 Likert-like range so
    # BEI sits on a comparable footing with EES/CES for human-readable
    # dashboard display (the ML pipeline will standardize everything
    # again regardless, so this rescaling is purely for interpretability,
    # not a modeling necessity).
    bei_15 = 3 + bei_z.clip(-2.5, 2.5) * (2 / 2.5)
    return bei_15.rename("BEI")


def build_feature_matrix(checkpoint_label: str = "mid_semester") -> pd.DataFrame:
    conn = get_connection()
    try:
        students = pd.read_sql("SELECT * FROM Students", conn)
        surveys = pd.read_sql("SELECT * FROM Surveys", conn)
        lms = pd.read_sql("SELECT * FROM LMS_Metrics", conn)
        outcomes = pd.read_sql("SELECT * FROM Outcomes", conn)
    finally:
        conn.close()

    ei_scores = _branch_scores(surveys, "EI", MID_SEMESTER_WAVE)
    eng_scores = _branch_scores(surveys, "ENGAGEMENT", MID_SEMESTER_WAVE)
    bei = _build_bei(lms)

    features = students.set_index("research_code")[
        ["age_bracket", "gender", "programme_type", "year_of_study"]
    ].copy()
    features = features.join(ei_scores).join(eng_scores).join(bei)

    # Missing-value handling per Section 3.3.5 step 2: median imputation
    # for continuous engineered scores, mode for categorical covariates.
    # This happens here (at feature-build time) rather than inside the ML
    # pipeline's preprocessing step because a missing branch score is a
    # *data engineering* fact (a student skipped that section of the
    # survey, or had no LMS rows for the window), whereas the
    # ml/preprocessing.py imputer exists as a defensive second layer in
    # case future data sources introduce missingness the feature layer
    # doesn't already know how to fill in a domain-appropriate way.
    numeric_cols = list(EI_BRANCHES.keys()) + ["BEI"] + list(ENGAGEMENT_SELF_REPORT_DIMENSIONS.keys())
    for col in numeric_cols:
        if col in features.columns:
            features[col] = features[col].fillna(features[col].median())
    for col in ["age_bracket", "gender", "programme_type"]:
        features[col] = features[col].fillna(features[col].mode().iloc[0])
    features["year_of_study"] = features["year_of_study"].fillna(features["year_of_study"].mode().iloc[0])

    features = features[FEATURE_COLUMNS]
    features["checkpoint"] = checkpoint_label

    target = outcomes.set_index("research_code")["performance_class"]
    final = features.join(target, how="inner")  # inner: only students with a recorded outcome

    out_path = PROCESSED_DIR / "feature_matrix.parquet"
    final.to_parquet(out_path)
    print(f"[feature_engineering] built feature matrix: {final.shape[0]} rows x {final.shape[1]} cols")
    print(f"[feature_engineering] saved to {out_path}")
    return final


if __name__ == "__main__":
    df = build_feature_matrix()
    print(df.head())
    print("\nClass balance:\n", df["performance_class"].value_counts(normalize=True).round(3))
