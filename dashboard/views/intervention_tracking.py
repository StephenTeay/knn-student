"""
dashboard/views/intervention_tracking.py
=========================================
Section 3.6.4: "The Intervention Tracking view helps staff record outreach
actions tied to at-risk flags and then watch if later engagement metric
updates nudge students out of the at-risk cluster."
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.data_access import (
    load_interventions, load_latest_predictions, load_students, log_intervention,
)

CLASS_EMOJIS = {"at_risk": "🔴", "satisfactory": "🟡", "high": "🟢"}


def render(is_researcher: bool) -> None:
    st.subheader("Intervention Tracking")
    st.caption(
        "Record and monitor outreach actions for at-risk students. "
        "Coverage panel shows which flagged students have been actioned."
    )

    predictions = load_latest_predictions()
    students = load_students()

    if predictions.empty:
        st.warning("No predictions found. Run `python -m ml.predict` first.")
        return

    at_risk = (
        predictions.loc[predictions["predicted_class"] == "at_risk"]
        .merge(students, on="research_code", how="left")
        .sort_values("confidence", ascending=False)
        .reset_index(drop=True)
    )

    st.markdown(f"### At-risk roster ({len(at_risk)} flagged)")
    if at_risk.empty:
        st.success("No students currently predicted at-risk.")
        return

    display_cols = ["research_code", "confidence", "programme_type", "year_of_study", "age_bracket"]
    if not is_researcher:
        display_cols = ["research_code", "confidence", "programme_type"]
    roster_display = at_risk[display_cols].copy()
    roster_display["confidence"] = roster_display["confidence"].map("{:.1%}".format)
    st.dataframe(roster_display, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Log an outreach action")
    col1, col2 = st.columns([1, 1])
    with col1:
        target_code = st.selectbox(
            "Student (research code)",
            at_risk["research_code"].tolist(),
        )
    with col2:
        action_type = st.selectbox(
            "Action type",
            ["Advisor call", "Email nudge", "Peer mentor referral", "Webinar invitation", "Other"],
        )
    notes = st.text_area("Notes", placeholder="Optional: describe the outreach or student context.")
    if st.button("Submit", type="primary"):
        pred_id = int(predictions.loc[predictions["research_code"] == target_code, "prediction_id"].iloc[0])
        role_for_log = "researcher" if is_researcher else "institutional_staff"
        log_intervention(target_code, pred_id, action_type, notes, role_for_log)
        st.success(f"Logged '{action_type}' for {target_code}.")
        st.rerun()

    st.divider()
    st.markdown("### Intervention audit log")
    interventions = load_interventions()
    if interventions.empty:
        st.info("No interventions logged yet.")
    else:
        show_cols = ["research_code", "action_type", "logged_by_role", "logged_at", "notes"]
        if not is_researcher:
            show_cols = ["research_code", "action_type", "logged_at"]
        st.dataframe(interventions[show_cols], use_container_width=True, hide_index=True)

        counts = interventions["action_type"].value_counts().reset_index()
        counts.columns = ["action_type", "count"]
        fig = px.bar(
            counts, x="action_type", y="count",
            title="Actions by type", color="action_type",
            labels={"action_type": "Action", "count": "Count"},
        )
        fig.update_layout(showlegend=False, margin=dict(t=40, b=20, l=10, r=10), height=280)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### Coverage check")
    if not interventions.empty:
        actioned = set(interventions["research_code"].unique())
        all_risk = set(at_risk["research_code"].unique())
        covered = len(actioned & all_risk)
        not_covered = len(all_risk - actioned)
        c1, c2, c3 = st.columns(3)
        c1.metric("At-risk total", len(all_risk))
        c2.metric("Actioned", covered)
        c3.metric("Not yet actioned", not_covered)
        outstanding = sorted(all_risk - actioned)
        if outstanding:
            st.warning("Still to action: " + ", ".join(outstanding[:10]) + ("..." if len(outstanding) > 10 else ""))
        else:
            st.success("All at-risk students have at least one logged action.")
    else:
        st.info("Log your first intervention above to see coverage.")
