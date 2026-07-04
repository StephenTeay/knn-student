"""
dashboard/data_access.py
=========================
Shared, cached data-loading functions used by every dashboard view.

DECISION: all reads go through st.cache_data with a short TTL, not
st.cache_resource or an uncached query-per-view. The underlying SQLite
file does not change within a single demo session except when the
Intervention Tracking view writes a new row, so caching avoids re-hitting
disk on every Streamlit rerun (which happens on basically every widget
interaction) -- but a TTL (rather than an unbounded cache) means a newly
logged intervention shows up again within seconds rather than requiring a
manual cache-clear, which would be a confusing dead-end for a non-technical
institutional-staff user.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ARTIFACTS_DIR
from db.db_init import get_connection

CACHE_TTL_SECONDS = 15


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_students() -> pd.DataFrame:
    conn = get_connection()
    try:
        return pd.read_sql("SELECT * FROM Students", conn)
    finally:
        conn.close()


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_latest_predictions() -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql(
            """SELECT p.* FROM Predictions p
               INNER JOIN (
                   SELECT research_code, MAX(predicted_at) AS max_ts
                   FROM Predictions GROUP BY research_code
               ) latest
               ON p.research_code = latest.research_code AND p.predicted_at = latest.max_ts""",
            conn,
        )
    finally:
        conn.close()
    df["neighbor_research_codes"] = df["neighbor_research_codes"].apply(json.loads)
    return df


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_feature_matrix() -> pd.DataFrame:
    from config import PROCESSED_DIR
    return pd.read_parquet(PROCESSED_DIR / "feature_matrix.parquet")


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_lms_metrics() -> pd.DataFrame:
    conn = get_connection()
    try:
        return pd.read_sql("SELECT * FROM LMS_Metrics", conn)
    finally:
        conn.close()


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_interventions() -> pd.DataFrame:
    conn = get_connection()
    try:
        return pd.read_sql("SELECT * FROM Interventions ORDER BY logged_at DESC", conn)
    finally:
        conn.close()


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_data_quality_log() -> pd.DataFrame:
    conn = get_connection()
    try:
        return pd.read_sql("SELECT * FROM DataQualityLog ORDER BY run_id DESC", conn)
    finally:
        conn.close()


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_model_comparison() -> pd.DataFrame:
    path = ARTIFACTS_DIR / "model_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_fairness_reports() -> dict:
    reports = {}
    for col in ["gender", "age_bracket", "programme_type"]:
        path = ARTIFACTS_DIR / f"fairness_{col}.csv"
        if path.exists():
            reports[col] = pd.read_csv(path)
    return reports


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_confusion_matrix() -> pd.DataFrame:
    path = ARTIFACTS_DIR / "knn_confusion_matrix.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_training_metadata() -> dict:
    path = ARTIFACTS_DIR / "training_metadata.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def log_intervention(research_code: str, prediction_id, action_type: str, notes: str,
                      intervention_week: int, role: str) -> None:
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO Interventions
               (research_code, prediction_id, action_type, notes, intervention_week, logged_by_role)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (research_code, int(prediction_id) if prediction_id is not None else None,
             action_type, notes, int(intervention_week), role),
        )
        conn.commit()
    finally:
        conn.close()
    load_interventions.clear()  # invalidate cache so the new row shows immediately
