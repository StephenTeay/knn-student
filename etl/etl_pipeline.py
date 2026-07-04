"""
etl/etl_pipeline.py
====================
Extract-Transform-Load pipeline that takes the raw CSV exports (survey
platform export + LMS log extract + outcomes export) and loads them into
SQLite, implementing Section 3.6.2's description: "Raw survey exports ...
are reviewed for completeness, range compliance and whether responses are
consistent before they go in. LMS log extracts are cleaned, to remove
system generated events ... After that, a data quality dashboard shows
missingness rates, outlier flags and consistency validations."

DECISION: validation failures are *recorded*, not silently dropped or used
to crash the whole load. Section 3.6.2 frames this as a quality-reporting
step the research team reviews before modeling starts, not a hard gate --
in a real pipeline a human (the "data custodian" from Section 3.5) decides
whether a flagged record needs to be excluded or just understood. Every
metric this pipeline computes is written to the DataQualityLog table so
it survives the ETL run and can be inspected later (the dashboard's
Researcher view reads from this table).

Run directly: python -m etl.etl_pipeline
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LIKERT_MAX, LIKERT_MIN, RAW_DIR
from db.db_init import get_connection, init_db


def _log_quality_metric(conn, source: str, metric_name: str, metric_value: float) -> None:
    conn.execute(
        "INSERT INTO DataQualityLog (source, metric_name, metric_value) VALUES (?, ?, ?)",
        (source, metric_name, float(metric_value)),
    )


def load_students(conn, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "students_export.csv")
    before = len(df)
    df = df.drop_duplicates(subset="research_code")
    _log_quality_metric(conn, "students", "duplicate_rows_removed", before - len(df))
    _log_quality_metric(conn, "students", "missing_rate", df.isna().mean().mean())
    df.to_sql("Students", conn, if_exists="append", index=False)
    return df


def load_surveys(conn, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "surveys_export.csv")

    # Completeness check: what fraction of expected (student x wave x
    # instrument x construct x item) cells came back missing entirely
    # (rows dropped at export time, e.g. a skipped survey item).
    missing_rate = df["response_value"].isna().mean()
    _log_quality_metric(conn, "surveys", "missing_rate", missing_rate)
    df = df.dropna(subset=["response_value"])

    # Range compliance check: every Likert response must be in
    # [LIKERT_MIN, LIKERT_MAX]. Anything outside that is a data-entry or
    # export error and is excluded from the load (and counted).
    in_range = df["response_value"].between(LIKERT_MIN, LIKERT_MAX)
    range_violations = int((~in_range).sum())
    _log_quality_metric(conn, "surveys", "range_violations", range_violations)
    df = df[in_range]

    # Consistency check: each (research_code, wave, instrument,
    # construct_code, item_number) should appear at most once. Duplicates
    # suggest a re-export or double submission.
    dup_mask = df.duplicated(subset=["research_code", "wave", "instrument", "construct_code", "item_number"])
    _log_quality_metric(conn, "surveys", "duplicate_rows_removed", int(dup_mask.sum()))
    df = df[~dup_mask]

    df.to_sql("Surveys", conn, if_exists="append", index=False)
    return df


def load_lms_logs(conn, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "lms_log_export.csv")

    # "Cleaned to remove system generated events" (Section 3.6.2): a
    # student-week with zero of every behavioral signal at once almost
    # never reflects genuine non-enrollment in a long-running module --
    # more often it is an artifact (e.g. the log extract missed a week
    # entirely). We flag, count, and drop such all-zero rows rather than
    # let them masquerade as "this student did nothing this week".
    all_zero = (
        (df["login_count"] == 0) & (df["forum_posts"] == 0) &
        (df["quiz_attempts"] == 0) & (df["video_completion_rate"].fillna(0) == 0)
    )
    _log_quality_metric(conn, "lms_logs", "all_zero_rows_removed", int(all_zero.sum()))
    df = df[~all_zero]

    # Outlier flagging via IQR on login_count, consistent with Section
    # 3.6.1's stated outlier-detection method (IQR), surfaced as a count
    # rather than silently removed -- a very high login count in one week
    # could be a genuine cram session, not necessarily bad data.
    q1, q3 = df["login_count"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    outliers = int((df["login_count"] > upper_fence).sum())
    _log_quality_metric(conn, "lms_logs", "login_count_outliers_flagged", outliers)

    df.to_sql("LMS_Metrics", conn, if_exists="append", index=False)
    return df


def load_outcomes(conn, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "outcomes_export.csv")
    valid_classes = {"at_risk", "satisfactory", "high"}
    bad_class = ~df["performance_class"].isin(valid_classes)
    _log_quality_metric(conn, "outcomes", "invalid_class_label_rows", int(bad_class.sum()))
    df = df[~bad_class]
    df.to_sql("Outcomes", conn, if_exists="append", index=False)
    return df


def run_etl(reset_db: bool = True, raw_dir: Path = RAW_DIR) -> dict:
    if reset_db:
        init_db(reset=True)
    conn = get_connection()
    try:
        students = load_students(conn, raw_dir)
        surveys = load_surveys(conn, raw_dir)
        lms = load_lms_logs(conn, raw_dir)
        outcomes = load_outcomes(conn, raw_dir)
        conn.commit()
    finally:
        conn.close()

    summary = {
        "students_loaded": len(students),
        "survey_responses_loaded": len(surveys),
        "lms_rows_loaded": len(lms),
        "outcomes_loaded": len(outcomes),
    }
    print("[etl_pipeline] load complete:", summary)
    return summary


def print_quality_report() -> None:
    conn = get_connection()
    report = pd.read_sql(
        "SELECT source, metric_name, metric_value, run_at FROM DataQualityLog ORDER BY run_id", conn
    )
    conn.close()
    print("\n[etl_pipeline] data quality report (most recent run):")
    print(report.to_string(index=False))


if __name__ == "__main__":
    run_etl(reset_db=True)
    print_quality_report()
