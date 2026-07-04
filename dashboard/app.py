"""
dashboard/app.py
=================
Main Streamlit entry point.

Local:           streamlit run dashboard/app.py
Streamlit Cloud: set Main file path = dashboard/app.py
                 (bootstrap.py auto-runs the pipeline on first visit)
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="EI & Engagement Academic Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Cloud / fresh-environment bootstrap.
# st.cache_resource runs this function exactly once per server instance
# (not on every user interaction or page load). If artifacts already
# exist locally it exits immediately. If they don't (e.g. first deploy
# on Streamlit Cloud) it runs the full 5-stage pipeline with a compact
# grid search and shows a progress bar to the user.
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _bootstrap():
    from dashboard.bootstrap import run_bootstrap_pipeline
    run_bootstrap_pipeline(fast_mode=True)

_bootstrap()

from dashboard.auth import is_researcher, render_role_selector
from dashboard.views import cohort_overview, intervention_tracking, model_inspection, student_detail


def main() -> None:
    st.sidebar.title("🎓 EI & Engagement\nAcademic Predictor")
    st.sidebar.markdown(
        "KNN-based prediction system  \n"
        "Mayer-Salovey EI × Fredricks Engagement  \n"
        "Distance education context  \n"
        "*(Synthetic demo — Section 3.6 prototype)*"
    )
    st.sidebar.divider()

    role = render_role_selector()
    researcher = is_researcher()

    st.sidebar.divider()
    st.sidebar.markdown("**Data snapshot**")
    try:
        from dashboard.data_access import load_latest_predictions, load_students
        n_students = len(load_students())
        n_at_risk = len(load_latest_predictions().query("predicted_class == 'at_risk'"))
        st.sidebar.metric("Students", n_students)
        st.sidebar.metric("At-risk flagged", n_at_risk)
    except Exception:
        st.sidebar.warning("DB not ready. Run the pipeline first.")

    st.sidebar.divider()
    st.sidebar.caption(
        "PhD Proposal: Designing an Improved KNN Model  \n"
        "ACETEL / NOUN — ACE24140004"
    )

    tab_labels = ["🏠 Cohort Overview", "🔍 Student Detail", "📋 Interventions"]
    if researcher:
        tab_labels.append("🔬 Model Inspection")

    tabs = st.tabs(tab_labels)

    with tabs[0]:
        cohort_overview.render(is_researcher=researcher)
    with tabs[1]:
        student_detail.render(is_researcher=researcher)
    with tabs[2]:
        intervention_tracking.render(is_researcher=researcher)
    if researcher and len(tabs) > 3:
        with tabs[3]:
            model_inspection.render(is_researcher=True)


if __name__ == "__main__":
    main()
