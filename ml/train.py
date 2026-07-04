"""
ml/train.py
============
Orchestrates Section 3.3.3-3.3.5 end to end:
  1. stratified train/held-out-test split
  2. grid search (k, metric, weights, PCA variance) for the optimized
     KNN+SMOTE pipeline, selected via 10-fold CV on the training split only
  3. an SMOTE-ablation comparison (same KNN hyperparameters, with vs.
     without SMOTE) on the held-out test set, per Section 3.4.3
  4. the three baseline comparators (Section 3.3.4), each lightly tuned
     with the same CV protocol
  5. subgroup fairness reporting (Section 3.5) on the held-out test set
  6. persistence of every artifact a later session/dashboard needs

Run directly: python -m ml.train

DECISION: model selection (which KNN hyperparameters count as "best") uses
weighted F1 as the GridSearchCV scoring/refit criterion, not raw accuracy.
Section 3.3.4 lists "weighted F1 score, overall accuracy, AUC, and
precision-recall for the at-risk class" as the things to report, but
doesn't single out one as the *selection* criterion. Plain accuracy is
the wrong choice here specifically because the classes are imbalanced
(Section 3.4.3): a model that always predicts "satisfactory" or "high"
can post a deceptively good accuracy while having ~0 recall on the
at-risk minority class, which is the group an early-warning system exists
to serve. Weighted F1 balances precision and recall per class while still
accounting for class frequency, which is a safer single number to drive
automatic model selection on. All four metrics are still computed and
reported for every model regardless of what drove the selection.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ARTIFACTS_DIR, CV_FOLDS, PERFORMANCE_CLASSES, PROCESSED_DIR, RANDOM_SEED
from features.feature_engineering import FEATURE_COLUMNS, build_feature_matrix
from ml.evaluate import compute_metrics, confusion_matrix_df, subgroup_fairness_report
from ml.label_utils import decode, encode
from ml.model_pipeline import build_baseline_pipelines, build_knn_param_grid, build_knn_pipeline

TEST_SIZE = 0.20
SCORING = "f1_weighted"
SUBGROUP_COLUMNS = ["gender", "age_bracket", "programme_type"]

# Compact grid for Streamlit Cloud free tier (~10s on 1 vCPU).
# Targets the high-value corners of the search space identified by the full run.
FAST_KNN_PARAM_GRID = {
    "pca": ["passthrough"],
    "knn__n_neighbors": [5, 11, 21],
    "knn__metric": ["euclidean", "manhattan"],
    "knn__weights": ["uniform", "distance"],
    "knn__p": [3],
}
FAST_CV_FOLDS = 5


def _load_data() -> pd.DataFrame:
    feature_path = PROCESSED_DIR / "feature_matrix.parquet"
    if not feature_path.exists():
        return build_feature_matrix()
    return pd.read_parquet(feature_path)


def _grid_search(pipeline, param_grid, X_train, y_train, n_folds: int = CV_FOLDS) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    search = GridSearchCV(
        pipeline, param_grid, scoring=SCORING, cv=cv, n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)
    return search


def run_training(fast_mode: bool = False) -> dict:
    df = _load_data()
    X = df[FEATURE_COLUMNS]
    y = df["performance_class"]
    demographics = df[SUBGROUP_COLUMNS]

    X_train, X_test, y_train, y_test, demo_train, demo_test = train_test_split(
        X, y, demographics, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    print(f"[train] split: {len(X_train)} train / {len(X_test)} held-out test "
          f"(stratified, test_size={TEST_SIZE})")

    # See ml/label_utils.py: every model is fit on integer-encoded labels
    # to sidestep a scikit-learn 1.8.0 KNeighborsClassifier.predict_proba
    # bug with string class labels; predictions are decoded back to
    # strings immediately after each predict()/predict_proba() call below.
    y_train_enc = encode(y_train)

    # -----------------------------------------------------------------
    # 1. Optimized KNN: grid search over PCA x k x metric x weights,
    #    with SMOTE inside the pipeline (Section 3.3.3).
    # -----------------------------------------------------------------
    if fast_mode:
        print("[train] grid-searching KNN (fast mode — compact grid for cloud startup)...")
        knn_pipeline = build_knn_pipeline(use_smote=True)
        knn_grid = FAST_KNN_PARAM_GRID
        knn_search = _grid_search(knn_pipeline, knn_grid, X_train, y_train_enc, n_folds=FAST_CV_FOLDS)
    else:
        print("[train] grid-searching optimized KNN (this is the slowest step)...")
        knn_pipeline = build_knn_pipeline(use_smote=True)
        knn_grid = build_knn_param_grid()
        knn_search = _grid_search(knn_pipeline, knn_grid, X_train, y_train_enc)
    best_knn_params = knn_search.best_params_
    print(f"[train] best KNN params: {best_knn_params} "
          f"(cv {SCORING}={knn_search.best_score_:.4f})")

    knn_pred = decode(knn_search.predict(X_test))
    knn_proba = knn_search.predict_proba(X_test)  # columns already in PERFORMANCE_CLASSES order
    knn_metrics = compute_metrics(y_test, knn_pred, knn_proba)
    knn_metrics["model"] = "knn_optimized_smote"

    # -----------------------------------------------------------------
    # 2. SMOTE ablation: identical KNN hyperparameters, SMOTE removed,
    #    refit on the same training split, evaluated on the same test
    #    split (Section 3.4.3's "shown separately next to the unbalanced
    #    baseline numbers").
    # -----------------------------------------------------------------
    knn_no_smote = build_knn_pipeline(use_smote=False)
    knn_no_smote.set_params(**best_knn_params)
    knn_no_smote.fit(X_train, y_train_enc)
    nosmote_pred = decode(knn_no_smote.predict(X_test))
    nosmote_proba = knn_no_smote.predict_proba(X_test)
    nosmote_metrics = compute_metrics(y_test, nosmote_pred, nosmote_proba)
    nosmote_metrics["model"] = "knn_optimized_no_smote"

    # -----------------------------------------------------------------
    # 3. Baseline comparators (Section 3.3.4).
    # -----------------------------------------------------------------
    baseline_pipelines, baseline_grids = build_baseline_pipelines(use_smote=True)
    baseline_results = []
    fitted_baselines = {}
    for name, pipeline in baseline_pipelines.items():
        print(f"[train] grid-searching baseline: {name}")
        search = _grid_search(pipeline, baseline_grids[name], X_train, y_train_enc)
        pred = decode(search.predict(X_test))
        try:
            proba = search.predict_proba(X_test)
        except AttributeError:
            proba = None
        m = compute_metrics(y_test, pred, proba)
        m["model"] = name
        m["best_params"] = search.best_params_
        baseline_results.append(m)
        fitted_baselines[name] = search.best_estimator_

    # -----------------------------------------------------------------
    # 4. Comparison table + confusion matrix + subgroup fairness, all on
    #    the optimized KNN (the model the proposal is centrally about).
    # -----------------------------------------------------------------
    comparison_rows = [knn_metrics, nosmote_metrics] + baseline_results
    comparison_df = pd.DataFrame(comparison_rows).set_index("model")
    print("\n[train] model comparison (held-out test set):\n", comparison_df)

    cm_df = confusion_matrix_df(y_test, knn_pred)
    print("\n[train] KNN confusion matrix (held-out test set):\n", cm_df)

    fairness_reports = {}
    for col in SUBGROUP_COLUMNS:
        rep = subgroup_fairness_report(y_test, knn_pred, demo_test[col])
        fairness_reports[col] = rep
        flag = rep.attrs.get("flagged", False)
        gap = rep.attrs.get("accuracy_gap", 0.0)
        print(f"\n[train] subgroup fairness by {col} "
              f"(accuracy gap={gap:.3f}, flagged={flag}):\n", rep)

    # -----------------------------------------------------------------
    # 5. Deployment / interpretability model: SAME best KNN
    #    hyperparameters, NO SMOTE, refit on the FULL labeled dataset
    #    (train+test combined).
    #
    #    Two reasons this differs from the evaluated model:
    #    (a) SMOTE inserts synthetic interpolated rows into the fitted
    #        KNeighborsClassifier's training set (verified empirically:
    #        600 real rows became 762 after SMOTE on this dataset). The
    #        Predictions table and the Student Detail dashboard view need
    #        every neighbor to map back to a real research_code, per
    #        Section 3.6.2's audit-trail design and Section 3.6.4's
    #        "anonymized comparison cards" of actual training-set
    #        students -- a synthetic neighbor has no real student behind
    #        it to show, which would silently break that feature.
    #    (b) once model selection is finalized using the held-out test
    #        set, it is standard practice to refit the chosen
    #        configuration on all available labeled data before
    #        deployment, so the live model isn't artificially deprived of
    #        20% of the data it could have learned from.
    #    The trade-off (this deployment model forgoes SMOTE's measured
    #    at-risk recall improvement, see comparison_df above) is real and
    #    is written into the training report rather than hidden -- see
    #    HANDOVER.md item "SMOTE vs. interpretability tension" for a
    #    discussion of alternatives if recall on at-risk students is the
    #    priority in a future iteration.
    # -----------------------------------------------------------------
    deployment_pipeline = build_knn_pipeline(use_smote=False)
    deployment_pipeline.set_params(**best_knn_params)
    deployment_pipeline.fit(X, encode(y))

    # -----------------------------------------------------------------
    # 6. Persist artifacts.
    # -----------------------------------------------------------------
    joblib.dump(knn_search.best_estimator_, ARTIFACTS_DIR / "knn_evaluated_smote.joblib")
    joblib.dump(deployment_pipeline, ARTIFACTS_DIR / "knn_deployment.joblib")
    for name, est in fitted_baselines.items():
        joblib.dump(est, ARTIFACTS_DIR / f"baseline_{name}.joblib")

    comparison_df.to_csv(ARTIFACTS_DIR / "model_comparison.csv")
    cm_df.to_csv(ARTIFACTS_DIR / "knn_confusion_matrix.csv")
    for col, rep in fairness_reports.items():
        rep.to_csv(ARTIFACTS_DIR / f"fairness_{col}.csv", index=False)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "n_total_students": len(df),
        "test_size": TEST_SIZE,
        "cv_folds": FAST_CV_FOLDS if fast_mode else CV_FOLDS,
        "scoring_criterion": SCORING,
        "best_knn_params": {k: str(v) for k, v in best_knn_params.items()},
        "best_knn_cv_score": float(knn_search.best_score_),
        "deployment_model_uses_smote": False,
        "deployment_model_trained_on": "full labeled dataset (train+test)",
    }
    with open(ARTIFACTS_DIR / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[train] artifacts saved to {ARTIFACTS_DIR}")
    return {
        "comparison": comparison_df,
        "confusion_matrix": cm_df,
        "fairness": fairness_reports,
        "metadata": metadata,
    }


if __name__ == "__main__":
    run_training()
