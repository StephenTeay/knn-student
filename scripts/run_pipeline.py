#!/usr/bin/env python3
"""
scripts/run_pipeline.py
========================
One-command entrypoint to run the full pipeline from scratch:

  Stage 1: Generate synthetic raw data
  Stage 2: ETL (validate + load into SQLite)
  Stage 3: Feature engineering (build feature matrix parquet)
  Stage 4: Model training (grid search + evaluation + artifact saves)
  Stage 5: Prediction population (score all students, write to DB)

Usage:
  python scripts/run_pipeline.py                  # full clean run (600 students, seed 42)
  python scripts/run_pipeline.py --n-students 300  # smaller cohort for quick tests
  python scripts/run_pipeline.py --skip-generate   # reuse existing raw CSVs (e.g. real data dropped in)

This corresponds to Stages 1-7 of the process framework in Section 3.3.5,
run in sequence. Stage 6 (interpretation/reporting) happens interactively
in the Streamlit dashboard.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _section(label: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")


def run_stage(module: str, extra_args: list = None) -> None:
    cmd = [sys.executable, "-m", module] + (extra_args or [])
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[run_pipeline] FAILED at {module} (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"[run_pipeline] {module} done in {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-students", type=int, default=600, help="Synthetic cohort size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip Stage 1 (use existing raw CSVs in data/raw/)")
    args = parser.parse_args()

    t_total = time.time()

    if not args.skip_generate:
        _section("Stage 1: Generate synthetic data")
        run_stage("data.generate_synthetic_data",
                   ["--n-students", str(args.n_students), "--seed", str(args.seed)])
    else:
        print("[run_pipeline] --skip-generate: Stage 1 skipped, using existing data/raw/ files.")

    _section("Stage 2: ETL — validate & load into SQLite")
    run_stage("etl.etl_pipeline")

    _section("Stage 3: Feature engineering")
    run_stage("features.feature_engineering")

    _section("Stage 4: Model training (grid search + evaluation)")
    run_stage("ml.train")

    _section("Stage 5: Prediction population")
    run_stage("ml.predict")

    total_elapsed = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete in {total_elapsed:.1f}s")
    print(f"  Launch dashboard: streamlit run dashboard/app.py")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
