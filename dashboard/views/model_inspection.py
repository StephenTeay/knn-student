"""
dashboard/views/model_inspection.py
=====================================
Researcher-only view surfacing the full evaluation outputs from ml/train.py:
- confusion matrix (held-out test set)
- model comparison table (KNN + SMOTE ablation + 3 baselines)
- subgroup fairness reports (Section 3.5)
- feature importance via permutation (Section 3.3.5's "mutual information"
  and "neighbor inspection readouts")
- training metadata + software version provenance

This view has no institutional-staff equivalent and is gated entirely by
the is_researcher check in dashboard/app.py -- Section 3.6.4 is explicit
that raw model internals are "for the researcher" and not the staff view.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.data_access import (
    load_confusion_matrix,
    load_fairness_reports,
    load_feature_matrix,
    load_model_comparison,
    load_training_metadata,
)
from features.feature_engineering import FEATURE_COLUMNS

CLASS_ORDER = ["at_risk", "satisfactory", "high"]
CLASS_COLORS = {"at_risk": "#d62728", "satisfactory": "#ff7f0e", "high": "#2ca02c"}


def _confusion_matrix_fig(cm_df: pd.DataFrame) -> go.Figure:
    """Plotly annotated heatmap of the confusion matrix."""
    labels = [c.replace("true_", "") for c in cm_df.index]
    z = cm_df.values.tolist()
    fig = go.Figure(go.Heatmap(
        z=z,
        x=[f"pred_{l}" for l in labels],
        y=[f"true_{l}" for l in labels],
        colorscale="Blues",
        showscale=True,
        text=[[str(v) for v in row] for row in z],
        texttemplate="%{text}",
        textfont_size=14,
    ))
    fig.update_layout(
        title="Confusion matrix (KNN+SMOTE, held-out test set)",
        xaxis_title="Predicted",
        yaxis_title="True",
        margin=dict(t=50, b=20, l=20, r=20),
        height=360,
    )
    return fig


def _feature_importance_fig(features: pd.DataFrame) -> go.Figure:
    """Mutual-information-based feature importance using scikit-learn's
    mutual_info_classif, computed on the full processed feature matrix
    (numeric columns only, so MI is meaningful -- one-hot columns from the
    categorical demographics are excluded here because MI on post-encoded
    dummy variables doesn't aggregate back to the original categorical
    cleanly without extra bookkeeping, and Section 3.3.5's 'filter-based
    feature importance ranking using mutual information' specifically targets
    the substantive predictors: the 4 EI branches + 3 engagement dimensions,
    not the demographic control covariates).
    """
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder

    numeric_cols = ["PE", "UE", "UndE", "ME", "BEI", "EES", "CES"]
    X = features[numeric_cols].fillna(features[numeric_cols].median())
    y = LabelEncoder().fit_transform(features["performance_class"])
    mi = mutual_info_classif(X, y, random_state=42)
    importance = pd.Series(mi, index=numeric_cols).sort_values(ascending=True)

    col_colors = {
        "PE": "#1f77b4", "UE": "#aec7e8", "UndE": "#4e79a7", "ME": "#76b7b2",
        "BEI": "#e15759", "EES": "#f28e2b", "CES": "#59a14f",
    }
    fig = px.bar(
        importance.reset_index(),
        x=0,
        y="index",
        orientation="h",
        color="index",
        color_discrete_map=col_colors,
        labels={"index": "Feature", 0: "Mutual Information"},
        title="Feature importance (Mutual Information, Section 3.3.5)",
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(t=50, b=20, l=20, r=20),
        height=360,
    )
    return fig


def render(is_researcher: bool) -> None:
    if not is_researcher:
        st.warning("Model inspection is available to the Researcher role only.")
        return

    st.subheader("Model Inspection (Researcher)")

    meta = load_training_metadata()
    if not meta:
        st.error("No training metadata found. Run `python -m ml.train` first.")
        return

    with st.expander("Training provenance", expanded=False):
        st.json(meta)

    st.divider()

    # ------------------------------------------------------------------
    # Model comparison table
    # ------------------------------------------------------------------
    st.markdown("### Model comparison (held-out test set)")
    comparison = load_model_comparison()
    if comparison.empty:
        st.warning("No model comparison CSV found.")
    else:
        numeric_fmt = {
            c: "{:.4f}"
            for c in comparison.columns
            if pd.api.types.is_numeric_dtype(comparison[c])
        }
        st.dataframe(
            comparison.style.format(numeric_fmt) if numeric_fmt else comparison,
            use_container_width=True,
        )
        st.caption(
            "**Key finding**: compare `recall_at_risk` between "
            "`knn_optimized_smote` (SMOTE on) and `knn_optimized_no_smote` "
            "(SMOTE off) to quantify SMOTE's effect on the minority class. "
            "The deployment model omits SMOTE for neighbor interpretability "
            "(see ml/train.py §5 and HANDOVER.md)."
        )

    st.divider()

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------
    st.markdown("### Confusion matrix")
    cm_df = load_confusion_matrix()
    if cm_df.empty:
        st.warning("No confusion matrix CSV found.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(_confusion_matrix_fig(cm_df), use_container_width=True)
        with col2:
            st.markdown("**Row = true class, column = predicted class.**")
            total = cm_df.values.sum()
            st.metric(
                "Correctly classified",
                int(np.diag(cm_df.values).sum()),
                help=f"Out of {total} held-out test students.",
            )
            at_risk_row = cm_df.loc[cm_df.index.str.contains("at_risk")]
            if not at_risk_row.empty:
                tp = int(at_risk_row["pred_at_risk"].values[0])
                fn = int(at_risk_row.drop(columns=["pred_at_risk"]).values[0].sum())
                st.metric(
                    "At-risk recall (SMOTE model)",
                    f"{tp / (tp + fn):.1%}" if (tp + fn) > 0 else "—",
                    help="True at-risk students correctly flagged.",
                )

    st.divider()

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    st.markdown("### Feature importance (Mutual Information)")
    features = load_feature_matrix()
    if not features.empty:
        st.plotly_chart(_feature_importance_fig(features), use_container_width=True)
        st.caption(
            "Higher MI = stronger association with performance class. "
            "EI branches (blue tones) vs. engagement dimensions (warm/green tones) "
            "ordering shows which construct family drives more predictive signal in "
            "this synthetic cohort -- Section 3.7's empirical contribution (3) "
            "is about this question on real data."
        )

    st.divider()

    # ------------------------------------------------------------------
    # Subgroup fairness
    # ------------------------------------------------------------------
    st.markdown("### Subgroup fairness (Section 3.5)")
    fairness = load_fairness_reports()
    if not fairness:
        st.warning("No fairness report CSVs found.")
    else:
        for attr, rep in fairness.items():
            st.markdown(f"**By {attr}**")
            acc_gap = rep["accuracy"].max() - rep["accuracy"].min()
            flagged = acc_gap > 0.10
            if flagged:
                st.warning(
                    f"Accuracy gap across {attr} groups = {acc_gap:.3f} > 10 pp threshold "
                    "(Section 3.5: 'systematic disparities ... recorded and described').",
                    icon="⚠️",
                )
            else:
                st.info(
                    f"Accuracy gap across {attr} groups = {acc_gap:.3f} (within 10 pp)."
                )
            fig = px.bar(
                rep,
                x="group",
                y="accuracy",
                color="group",
                labels={"group": attr, "accuracy": "Accuracy"},
                height=250,
            )
            fig.add_hline(
                y=rep["accuracy"].mean(),
                line_dash="dash",
                annotation_text=f"cohort mean: {rep['accuracy'].mean():.3f}",
            )
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(rep, use_container_width=True, hide_index=True)
