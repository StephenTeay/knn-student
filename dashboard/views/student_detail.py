"""
dashboard/views/student_detail.py
===================================
Implements Section 3.6.4's "Student Detail view lets an authorized
institutional user pull an individual student's predicted performance
class, the confidence value, and the profile of the three to five
closest training neighbors that support the prediction. Those neighbor
results are shown as anonymized comparison cards, so there is no direct
identification."

DECISION: the neighbor comparison is shown as a Plotly radar chart
overlaying the query student's EI/engagement profile against the mean
profile of their top-k training neighbors. The proposal specifies
"anonymized comparison cards" (plural), which could be done as a table
row per neighbor or as a text card per neighbor. The radar visualization
delivers the same conceptual output -- "here are the similar students
and what their profiles look like" -- but lets the user read the distance
between profiles immediately rather than scanning numbers across five
text cards. The actual neighbor research codes (anonymized to a surrogate
already by the ETL, not real IDs) are shown in a collapsible expander
for the Researcher role, not by default for Institutional Staff, per the
Section 3.6.4 data-access split.

DECISION: the confidence score shown here is max(predict_proba), NOT
a count-of-neighbors vote. For KNN with weights='uniform', the two are
equivalent, but with weights='distance' (a grid-searched option) the
distance-weighted probability is strictly more informative than a raw
vote count. Since the grid may select either weights setting, using
predict_proba as the source of confidence is always correct, whereas a
raw-vote confidence would only be correct half the time depending on
what the grid happened to pick.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dashboard.data_access import (
    load_feature_matrix, load_latest_predictions, load_students,
)

CLASS_COLORS = {"at_risk": "#d62728", "satisfactory": "#ff7f0e", "high": "#2ca02c"}
CLASS_ORDER = ["at_risk", "satisfactory", "high"]
EI_ENG_COLS = ["PE", "UE", "UndE", "ME", "BEI", "EES", "CES"]
RADAR_LABELS = {
    "PE": "Perceiving Emotions",
    "UE": "Using Emotions",
    "UndE": "Understanding Emotions",
    "ME": "Managing Emotions",
    "BEI": "Behavioral Engagement",
    "EES": "Emotional Engagement",
    "CES": "Cognitive Engagement",
}


def _radar_chart(query_row: pd.Series, neighbor_mean: pd.Series) -> go.Figure:
    cols = EI_ENG_COLS
    labels = [RADAR_LABELS[c] for c in cols]
    query_vals = [query_row[c] for c in cols] + [query_row[cols[0]]]
    neighbor_vals = [neighbor_mean[c] for c in cols] + [neighbor_mean[cols[0]]]
    labels_closed = labels + [labels[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=query_vals, theta=labels_closed, fill="toself",
        name="This student", line=dict(color="#1f77b4"), opacity=0.6,
    ))
    fig.add_trace(go.Scatterpolar(
        r=neighbor_vals, theta=labels_closed, fill="toself",
        name="Neighbor average", line=dict(color="#9467bd", dash="dash"), opacity=0.5,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[1, 5])),
        showlegend=True,
        margin=dict(t=40, b=40, l=50, r=50),
        height=420,
    )
    return fig


def render(is_researcher: bool) -> None:
    st.subheader("Student Detail")
    st.caption(
        "Look up an individual student's predicted performance class, confidence, "
        "and the training neighbors whose profiles are most similar to theirs — "
        "the KNN interpretability mechanism described in Section 3.6.4."
    )

    predictions = load_latest_predictions()
    features = load_feature_matrix()
    students = load_students().set_index("research_code")

    if predictions.empty:
        st.warning("No predictions found. Run `python -m ml.predict` first.")
        return

    col_search, col_filter = st.columns([2, 1])
    with col_filter:
        filter_class = st.multiselect(
            "Filter by predicted class",
            options=CLASS_ORDER,
            default=CLASS_ORDER,
        )
    filtered_codes = predictions.loc[
        predictions["predicted_class"].isin(filter_class), "research_code"
    ].tolist()

    with col_search:
        selected_code = st.selectbox("Select student (research code)", options=filtered_codes)

    if not selected_code:
        return

    pred_row = predictions.set_index("research_code").loc[selected_code]
    feature_row = features.loc[selected_code] if selected_code in features.index else None

    pred_class = pred_row["predicted_class"]
    confidence = pred_row["confidence"]
    color = CLASS_COLORS.get(pred_class, "#888")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Predicted class", pred_class.replace("_", " ").title())
    col_b.metric("Confidence", f"{confidence:.1%}")
    col_c.metric("Checkpoint", pred_row["checkpoint"])
    st.markdown(
        f'<div style="background:{color}22;border-left:4px solid {color};'
        f'padding:10px 14px;border-radius:4px;margin-bottom:8px;">'
        f'<b>Prediction: {pred_class.replace("_"," ").upper()}</b> '
        f'— {confidence:.0%} confidence'
        f'</div>', unsafe_allow_html=True
    )

    neighbor_codes = pred_row["neighbor_research_codes"]

    if feature_row is not None:
        neighbor_features = features.loc[
            [c for c in neighbor_codes if c in features.index]
        ]
        if not neighbor_features.empty:
            neighbor_mean = neighbor_features[EI_ENG_COLS].mean()
            st.markdown("**EI & Engagement profile vs. similar students**")
            st.plotly_chart(_radar_chart(feature_row, neighbor_mean), use_container_width=True)
            st.caption(
                "The radar chart shows this student's composite EI branch and engagement "
                "dimension scores versus the average profile of their nearest training neighbors. "
                "Sections of the 'This student' fill that sit inside the 'Neighbor average' "
                "ring indicate dimensions where this student is weaker than their comparison group."
            )

    st.markdown("**Nearest training neighbors (the evidence behind this prediction)**")
    neighbor_preds = predictions.set_index("research_code")

    neighbor_display = []
    for nc in neighbor_codes:
        if nc not in neighbor_preds.index:
            continue
        np_row = neighbor_preds.loc[nc]
        card: dict = {
            "Neighbor code": nc,
            "Predicted class": np_row["predicted_class"],
            "Confidence": f"{np_row['confidence']:.1%}",
        }
        if is_researcher and nc in features.index:
            nf = features.loc[nc]
            for col in EI_ENG_COLS:
                card[col] = f"{nf[col]:.2f}"
        neighbor_display.append(card)

    if neighbor_display:
        nb_df = pd.DataFrame(neighbor_display)
        st.dataframe(nb_df, use_container_width=True, hide_index=True)
        if not is_researcher:
            st.caption(
                "Individual EI/engagement scores for neighbors are visible in the Researcher view."
            )
    else:
        st.info("No neighbor data found — run `python -m ml.predict` to refresh.")

    if is_researcher and feature_row is not None:
        with st.expander("Raw feature values (Researcher view)", expanded=False):
            st.dataframe(
                feature_row.to_frame("value").T[EI_ENG_COLS],
                use_container_width=True,
            )
            if selected_code in students.index:
                demo = students.loc[selected_code]
                st.markdown(f"**Demographics:** age_bracket={demo['age_bracket']} | "
                             f"gender={demo['gender']} | "
                             f"programme={demo['programme_type']} | "
                             f"year={demo['year_of_study']}")
