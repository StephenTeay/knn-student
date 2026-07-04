"""
dashboard/views/cohort_overview.py
====================================
Implements Section 3.6.4's "Cohort Overview view shows aggregate
performance predictions for the current student group, like pie charts
with predicted class spread, heatmaps that map feature relationships, and
time-series plots tracking how engagement metrics drift through the
semester."

DECISION: Plotly instead of the proposal's stated matplotlib/seaborn
(Section 3.6.3). matplotlib/seaborn figures are static images inside
Streamlit (st.pyplot) -- fine for a generated report, but this is an
interactive dashboard a researcher or staff member is meant to explore
(hover for exact values, zoom into a date range on the drift chart). The
task brief for this implementation explicitly names plotly as part of the
stack, and Streamlit's st.plotly_chart integrates it natively with no
extra glue code. matplotlib/seaborn remain the right call for the
*offline* model-evaluation report figures Chapter Four will eventually
need for a static thesis document, since a PDF can't host a hover
tooltip anyway -- they are not mutually exclusive, just suited to
different output targets.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.data_access import (
    load_data_quality_log, load_feature_matrix, load_latest_predictions,
    load_lms_metrics, load_model_comparison, load_students,
)

CLASS_COLORS = {"at_risk": "#d62728", "satisfactory": "#ff7f0e", "high": "#2ca02c"}
CLASS_ORDER = ["at_risk", "satisfactory", "high"]


def render(is_researcher: bool) -> None:
    st.subheader("Cohort Overview")
    st.caption(
        "Synthetic demonstration cohort (see README.md / data/_calibration_notes.md). "
        "These numbers are illustrative of the pipeline, not a real research finding."
    )

    predictions = load_latest_predictions()
    students = load_students()

    if predictions.empty:
        st.warning("No predictions found. Run `python -m ml.predict` first.")
        return

    col1, col2 = st.columns([1, 1])

    # --- Pie chart: predicted class spread -----------------------------
    with col1:
        st.markdown("**Predicted performance class distribution**")
        counts = predictions["predicted_class"].value_counts().reindex(CLASS_ORDER).fillna(0)
        fig = go.Figure(data=[go.Pie(
            labels=counts.index, values=counts.values,
            marker=dict(colors=[CLASS_COLORS[c] for c in counts.index]),
            hole=0.35,
        )])
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
        st.plotly_chart(fig, use_container_width=True)
        n_at_risk = int(counts.get("at_risk", 0))
        st.metric("Students flagged at-risk", n_at_risk,
                   help="Candidates for outreach in the Intervention Tracking tab.")

    # --- Confidence distribution ----------------------------------------
    with col2:
        st.markdown("**Prediction confidence by class**")
        fig = px.box(
            predictions, x="predicted_class", y="confidence", color="predicted_class",
            category_orders={"predicted_class": CLASS_ORDER},
            color_discrete_map=CLASS_COLORS, points="all",
        )
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=320,
                           xaxis_title=None, yaxis_title="confidence")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Feature correlation heatmap (researcher-only: uses raw scores) --
    if is_researcher:
        st.markdown("**Feature correlation heatmap** (Researcher view — raw EI/engagement scores)")
        features = load_feature_matrix()
        numeric_cols = ["PE", "UE", "UndE", "ME", "BEI", "EES", "CES"]
        corr = features[numeric_cols].corr()
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            aspect="auto",
        )
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "High inter-correlation among the four EI branches is expected (they share an "
            "underlying ability-EI construct, Section 2.9); BEI/EES/CES correlations show how "
            "closely behavioral and self-reported engagement track each other in this cohort."
        )
    else:
        st.info(
            "Feature-level correlation analysis uses raw EI/engagement scores and is available "
            "in the Researcher view."
        )

    st.divider()

    # --- Engagement drift over the semester ------------------------------
    st.markdown("**Cohort-wide engagement drift across the semester**")
    lms = load_lms_metrics()
    weekly = lms.groupby("week_number").agg(
        avg_logins=("login_count", "mean"),
        avg_forum_posts=("forum_posts", "mean"),
        avg_video_completion=("video_completion_rate", "mean"),
    ).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly["week_number"], y=weekly["avg_logins"],
                              mode="lines+markers", name="Avg. weekly logins"))
    fig.add_trace(go.Scatter(x=weekly["week_number"], y=weekly["avg_forum_posts"],
                              mode="lines+markers", name="Avg. forum posts"))
    fig.add_trace(go.Scatter(x=weekly["week_number"], y=weekly["avg_video_completion"] * 10,
                              mode="lines+markers", name="Avg. video completion (×10 for scale)"))
    fig.update_layout(xaxis_title="Semester week", yaxis_title="Average value",
                       margin=dict(t=10, b=10, l=10, r=10), height=380,
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "This is the longitudinal signal Section 3.6.2 describes the LMS_Metrics table as "
        "supporting beyond the single mid-semester snapshot used for prediction."
    )

    if is_researcher:
        st.divider()
        st.markdown("**Most recent ETL data-quality log** (Researcher view)")
        st.dataframe(load_data_quality_log(), use_container_width=True, hide_index=True)

        st.markdown("**Model comparison (held-out test set)**")
        comparison = load_model_comparison()
        if not comparison.empty:
            st.dataframe(comparison.style.format({
                c: "{:.3f}" for c in comparison.columns if comparison[c].dtype != object
            }), use_container_width=True)
            st.caption(
                "knn_optimized_smote vs. knn_optimized_no_smote isolates SMOTE's effect on "
                "at-risk recall (Section 3.4.3); the deployed dashboard model intentionally "
                "omits SMOTE for neighbor-interpretability reasons (see ml/train.py docstring)."
            )
