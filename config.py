"""
config.py
=========
Single source of truth for paths and constants used across the pipeline.

DECISION: centralising config avoids the classic problem of a threshold
(e.g. the 50%/70% performance-class cut points from Chapter 3, Section 3.3.2
of the proposal) being copy-pasted into five different scripts and drifting
out of sync when someone changes it in only one place. Every module below
imports from here.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_DIR = ROOT_DIR / "db"
DB_PATH = DB_DIR / "distance_ed.sqlite"
SCHEMA_PATH = DB_DIR / "schema.sql"
ARTIFACTS_DIR = ROOT_DIR / "ml" / "artifacts"
LOGS_DIR = ROOT_DIR / "logs"

for d in (RAW_DIR, PROCESSED_DIR, DB_DIR, ARTIFACTS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# DECISION: a single fixed seed used everywhere (synthetic data generation,
# train/test split, SMOTE, KNN tie-breaking via RandomState) so the whole
# pipeline is byte-for-byte reproducible on re-run, per the proposal's
# Section 3.6.3 requirement to "set all the random seeds ... and write down
# every software version".
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Survey instrument structure (Section 3.3.2)
# ---------------------------------------------------------------------------
# EI branches from the Mayer-Salovey four-branch ability model, adapted as a
# 5-point Likert subscale of N items each (mirrors how MSCEIT-style branch
# scores are typically composited: mean of items -> branch score).
EI_BRANCHES = {
    "PE": "Perceiving Emotions",
    "UE": "Using Emotions to Facilitate Thought",
    "UndE": "Understanding Emotions",
    "ME": "Managing Emotions",
}
EI_ITEMS_PER_BRANCH = 5
LIKERT_MIN, LIKERT_MAX = 1, 5

# Engagement dimensions: Behavioral comes from LMS logs (BEI), Emotional and
# Cognitive come from the Online Student Engagement (OSE) self-report scale.
ENGAGEMENT_SELF_REPORT_DIMENSIONS = {
    "EES": "Emotional Engagement Score",
    "CES": "Cognitive Engagement Score",
}
ENGAGEMENT_ITEMS_PER_DIMENSION = 5

# ---------------------------------------------------------------------------
# Demographic covariates (control features, Section 3.3.2 + ethics 3.5)
# ---------------------------------------------------------------------------
GENDERS = ["Male", "Female"]
AGE_BRACKETS = ["18-21", "22-25", "26-30", "31+"]
PROGRAMME_TYPES = ["Undergraduate", "Postgraduate Diploma", "Masters"]
YEARS_OF_STUDY = [1, 2, 3, 4]

# ---------------------------------------------------------------------------
# Outcome / target variable (Section 3.3.2)
# ---------------------------------------------------------------------------
# Three-class target. Encoded as ints for sklearn, with a label map for
# display. Order matters for ROC-AUC (one-vs-rest) and confusion matrices.
PERFORMANCE_CLASSES = ["at_risk", "satisfactory", "high"]
PERFORMANCE_THRESHOLDS = {  # final percentage score cut points
    "high": 70.0,        # score >= 70
    "satisfactory": 50.0,  # 50 <= score < 70
    # < 50 => at_risk
}

# ---------------------------------------------------------------------------
# LMS behavioral feature window (Section 3.3.2 / 3.6.2)
# ---------------------------------------------------------------------------
N_WEEKS = 14  # one semester, weekly granularity, per "one row per
              # student-module-week" design in Section 3.6.2

# ---------------------------------------------------------------------------
# KNN optimization search space (Section 3.3.3)
# ---------------------------------------------------------------------------
KNN_K_GRID = [3, 5, 7, 9, 11, 15, 21, 31]
KNN_METRIC_GRID = ["euclidean", "manhattan", "minkowski"]
KNN_MINKOWSKI_P = 3  # only used when metric == "minkowski"
KNN_WEIGHTS_GRID = ["uniform", "distance"]  # distance-weighted KNN (2.7.3)
PCA_VARIANCE_GRID = [None, 0.90, 0.95]  # None = no PCA, per 3.3.3
CV_FOLDS = 10  # "Stratified 10-fold cross-validation" (Section 3.3.5)

# Default synthetic cohort size. Hundreds-to-low-thousands per Section 3.4.1
# ("typical university cohort"). Kept moderate so the full pipeline
# (grid search x CV x SMOTE x PCA grid) runs in well under a minute on a
# laptop-class CPU.
DEFAULT_N_STUDENTS = 600
