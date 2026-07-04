"""
ml/evaluate.py
===============
Metric computation shared by every model evaluated in train.py.

DECISION: metrics implement exactly the set named in Section 3.3.4
("weighted F1 score, overall accuracy, area under the ROC curve (AUC) and
also precision-recall specifically for the at-risk minority class") plus
the subgroup fairness reporting required by Section 3.5's "disaggregated
performance metrics ... shared across demographic subgroups. If there are
systematic disparities, they get recorded, and then described." rather
than a generic sklearn classification_report dump -- the latter doesn't
single out the at-risk class precision/recall the way the proposal's own
evaluation protocol asks for, and says nothing about subgroup fairness at
all.

DECISION: subgroup fairness is reported (a measurement), not enforced (no
re-weighting or post-hoc threshold adjustment is applied to "fix" a
disparity). Section 3.5 is explicit that the obligation here is to
"record" and "describe" disparities and to keep demographic variables out
of the operational feature set as direct predictors -- it does not ask for
an automated fairness-correction step, and silently correcting for a
disparity without a documented institutional fairness policy behind the
choice would itself be a questionable unilateral decision for a research
prototype to make.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PERFORMANCE_CLASSES

AT_RISK_LABEL = "at_risk"


def compute_metrics(y_true, y_pred, y_proba=None, class_order=PERFORMANCE_CLASSES) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_at_risk": precision_score(
            y_true, y_pred, labels=[AT_RISK_LABEL], average="micro", zero_division=0
        ),
        "recall_at_risk": recall_score(
            y_true, y_pred, labels=[AT_RISK_LABEL], average="micro", zero_division=0
        ),
    }
    if y_proba is not None:
        y_true_bin = label_binarize(y_true, classes=class_order)
        try:
            metrics["roc_auc_ovr_weighted"] = roc_auc_score(
                y_true_bin, y_proba, average="weighted", multi_class="ovr"
            )
        except ValueError:
            # Can happen if a class is entirely absent from a small test
            # fold; recorded as NaN rather than crashing the whole run.
            metrics["roc_auc_ovr_weighted"] = float("nan")
    return metrics


def confusion_matrix_df(y_true, y_pred, class_order=PERFORMANCE_CLASSES) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=class_order)
    return pd.DataFrame(cm, index=[f"true_{c}" for c in class_order],
                         columns=[f"pred_{c}" for c in class_order])


def subgroup_fairness_report(y_true, y_pred, subgroup_series: pd.Series) -> pd.DataFrame:
    """Per Section 3.5: accuracy and at-risk recall, disaggregated by a
    demographic column (gender / age_bracket / programme_type)."""
    df = pd.DataFrame({"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred),
                        "group": subgroup_series.values})
    rows = []
    for group_val, sub in df.groupby("group"):
        rows.append({
            "group": group_val,
            "n": len(sub),
            "accuracy": accuracy_score(sub["y_true"], sub["y_pred"]),
            "recall_at_risk": recall_score(
                sub["y_true"], sub["y_pred"], labels=[AT_RISK_LABEL], average="micro", zero_division=0
            ),
        })
    report = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
    # Flag disparity: max-min accuracy gap across groups exceeding 10
    # percentage points is recorded as a flag for the research team to
    # describe (Section 3.5), not auto-corrected (see module docstring).
    if len(report) > 1:
        gap = report["accuracy"].max() - report["accuracy"].min()
        report.attrs["accuracy_gap"] = gap
        report.attrs["flagged"] = bool(gap > 0.10)
    return report
