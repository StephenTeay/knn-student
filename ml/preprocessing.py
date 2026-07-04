"""
ml/preprocessing.py
====================
Builds the preprocessing ColumnTransformer shared by the KNN model and all
three baseline classifiers (Section 3.3.4: "Each model uses an identical
preprocessing pipeline").

DECISION: numeric features (the 4 EI branches + BEI/EES/CES) get median
imputation (defensive second layer, see feature_engineering.py docstring)
then z-score standardization (Section 3.3.3, step 1 -- "feature scaling
with z-score standardization is used on all continuous variables before
distances are computed, so that columns with bigger numerical ranges do
not end up dominating the Euclidean distance part"). This matters far more
for KNN/SVM (distance-based) than for the tree/logistic baselines, but
applying it uniformly keeps the comparison in Section 3.3.4 fair -- nobody
can argue KNN looks better or worse than logistic regression because of an
avoidable scaling asymmetry.

DECISION: categorical demographic covariates (gender, programme_type,
age_bracket as an ordered-but-treated-as-nominal bucket, and year_of_study
as a small-cardinality integer) are one-hot encoded with
handle_unknown="ignore" rather than ordinal-encoded, because none of them
have a model-relevant ordinal relationship that a raw integer encoding
would represent honestly (e.g. "year_of_study=4" is not "twice" anything
meaningful to a distance metric), and one-hot avoids KNN distance
computations imposing a false ordinal scale on categories like
programme_type. year_of_study is included in the categorical block for the
same reason even though it happens to be stored as an int.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EI_BRANCHES, ENGAGEMENT_SELF_REPORT_DIMENSIONS

NUMERIC_FEATURES = list(EI_BRANCHES.keys()) + ["BEI"] + list(ENGAGEMENT_SELF_REPORT_DIMENSIONS.keys())
CATEGORICAL_FEATURES = ["age_bracket", "gender", "programme_type", "year_of_study"]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    return preprocessor
