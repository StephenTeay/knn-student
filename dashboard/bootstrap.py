"""
dashboard/bootstrap.py
=======================
Auto-initialises the full pipeline on first startup (Streamlit Cloud or any
fresh environment where no artifacts exist yet).

WHY THIS EXISTS:
Streamlit Cloud only runs `streamlit run dashboard/app.py` — it has no
pre-run hook for `python scripts/run_pipeline.py`. This module is called
from app.py via st.cache_resource (which guarantees it runs exactly ONCE
per server instance, not on every page interaction) and runs every pipeline
stage in order if the artifacts are missing. If they exist (e.g. on a local
machine where you already ran the pipeline) it exits immediately.

FAST MODE:
The full grid search (8 k-values × 3 metrics × 2 weights × 3 PCA options
= 144 combinations × 10 folds) takes ~60s on a laptop. Streamlit Community
Cloud has a 1 vCPU free tier that would stretch that to 3–5 minutes, which
is too long for a first-render spinner. `fast_mode=True` (default on cloud)
uses a compact 12-combo × 5-fold grid that completes in about 5–10s on the
free tier — still a genuine grid search, just over a smaller but
well-targeted search space (we already know from the full run that manhattan
distance and moderate k values dominate; the fast grid covers those).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from config import ARTIFACTS_DIR, DB_PATH, PROCESSED_DIR


def _artifacts_exist() -> bool:
    """Check all three required artifact groups."""
    feature_matrix = (PROCESSED_DIR / "feature_matrix.parquet").exists()
    deployment_model = (ARTIFACTS_DIR / "knn_deployment.joblib").exists()
    db_ready = DB_PATH.exists() and DB_PATH.stat().st_size > 0
    return feature_matrix and deployment_model and db_ready


def run_bootstrap_pipeline(fast_mode: bool = True, n_students: int = 600) -> None:
    """
    Run the full 5-stage pipeline if artifacts don't already exist.
    Displays a Streamlit progress bar so the user sees what's happening.

    Parameters
    ----------
    fast_mode : bool
        If True, uses a compact grid for faster cloud startup (default True).
        Set to False to run the full search (for local use or a powerful server).
    n_students : int
        Synthetic cohort size (default 600; reduce to 300 for faster demos).
    """
    if _artifacts_exist():
        return  # nothing to do

    st.info(
        "⚙️ **First-time setup** — running the 5-stage pipeline to initialise the demo. "
        "This takes about 30–60 seconds on Streamlit Cloud and then the app is ready. "
        "Subsequent visits load instantly.",
        icon="⏳",
    )
    progress = st.progress(0, text="Starting pipeline…")

    # ------------------------------------------------------------------
    # Stage 1: Generate synthetic data
    # ------------------------------------------------------------------
    progress.progress(5, text="Stage 1/5 — Generating synthetic cohort…")
    from data.generate_synthetic_data import generate
    generate(n_students=n_students)

    # ------------------------------------------------------------------
    # Stage 2: ETL
    # ------------------------------------------------------------------
    progress.progress(20, text="Stage 2/5 — ETL: validating & loading into database…")
    from etl.etl_pipeline import run_etl
    run_etl(reset_db=True)

    # ------------------------------------------------------------------
    # Stage 3: Feature engineering
    # ------------------------------------------------------------------
    progress.progress(40, text="Stage 3/5 — Building feature matrix (EI branches + BEI + engagement)…")
    from features.feature_engineering import build_feature_matrix
    build_feature_matrix()

    # ------------------------------------------------------------------
    # Stage 4: Model training
    # ------------------------------------------------------------------
    progress.progress(55, text="Stage 4/5 — Training & evaluating KNN model"
                                + (" (fast grid)" if fast_mode else " (full grid search)")
                                + "…")
    from ml.train import run_training
    run_training(fast_mode=fast_mode)

    # ------------------------------------------------------------------
    # Stage 5: Populate Predictions table
    # ------------------------------------------------------------------
    progress.progress(90, text="Stage 5/5 — Scoring students and writing predictions…")
    from ml.predict import populate_predictions
    populate_predictions()

    progress.progress(100, text="✅ Pipeline complete — loading dashboard…")
    progress.empty()
    st.success("Setup complete! The dashboard is ready.", icon="✅")
    st.rerun()
