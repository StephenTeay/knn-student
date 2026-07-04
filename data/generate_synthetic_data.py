"""
data/generate_synthetic_data.py
================================
Generates synthetic "raw" CSV exports that stand in for the data sources
described in Section 3.3.2: (1) the EI + engagement survey export, (2) the
LMS log extract, and (3) the end-of-semester outcome export.

WHY THIS EXISTS (important framing, not a cosmetic detail):
The proposal document is a PhD *proposal* -- Chapter Three's own milestone
table (Table 3.2) places "Data Collection" at Months 3-6 and "Ethical
Approval and Institutional Access" at Months 1-2, both *before* any model
can be built. No real student survey or LMS data exists yet, and it would
be both factually wrong and an ethics violation to fabricate "real-looking"
student records and pass them off as collected data. What this script
generates is explicitly synthetic, seeded, parameterized data whose only
purpose is to let the rest of the pipeline (ETL -> features -> model ->
dashboard) be built, run, and demonstrated end-to-end *before* real data
exists -- exactly the kind of dry-run a research team would do to debdesign-validate
the pipeline ahead of fieldwork. Every file this script writes is tagged
and every downstream README/dashboard caption says "synthetic" so nobody
mistakes it for a real result.

The generator encodes the theoretical relationships the study expects to
find (Section 3.4.2 "Expected Behavior Given the Feature Set" -- the three
archetypes: high-EI/high-engagement achievers, low-EI/low-engagement
strugglers, and mixed-profile students) plus realistic noise, so that the
downstream KNN pipeline has a genuine (if synthetic) non-linear signal to
recover, and a believable ~20-30% at-risk class imbalance to correct with
SMOTE (Section 3.4.3).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    AGE_BRACKETS, EI_BRANCHES, EI_ITEMS_PER_BRANCH, ENGAGEMENT_ITEMS_PER_DIMENSION,
    ENGAGEMENT_SELF_REPORT_DIMENSIONS, GENDERS, LIKERT_MAX, LIKERT_MIN, N_WEEKS,
    PERFORMANCE_THRESHOLDS, PROGRAMME_TYPES, RANDOM_SEED, RAW_DIR, YEARS_OF_STUDY,
)


def _clip_likert(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x), LIKERT_MIN, LIKERT_MAX).astype(int)


def generate(n_students: int, seed: int = RANDOM_SEED, output_dir: Path = RAW_DIR):
    rng = np.random.default_rng(seed)

    research_codes = [f"S{idx:05d}" for idx in range(1, n_students + 1)]

    # ------------------------------------------------------------------
    # 1. Students (demographics) -- already "anonymized" at generation
    #    time since there is no underlying real identity to begin with.
    # ------------------------------------------------------------------
    students = pd.DataFrame({
        "research_code": research_codes,
        "age_bracket": rng.choice(AGE_BRACKETS, size=n_students, p=[0.45, 0.30, 0.15, 0.10]),
        "gender": rng.choice(GENDERS, size=n_students, p=[0.48, 0.52]),
        "programme_type": rng.choice(PROGRAMME_TYPES, size=n_students, p=[0.65, 0.20, 0.15]),
        "year_of_study": rng.choice(YEARS_OF_STUDY, size=n_students, p=[0.35, 0.30, 0.20, 0.15]),
    })

    # ------------------------------------------------------------------
    # 2. Latent traits per student -- these are NOT written to disk; they
    #    only drive the simulation so we can later check the model
    #    recovers a meaningful signal. Three archetypes are mixed in,
    #    per Section 3.4.2, via a latent "profile" draw plus continuous
    #    noise so the boundary between groups is fuzzy, not a clean
    #    cluster -- this is what makes KNN's local-neighborhood approach
    #    actually matter relative to a linear baseline.
    # ------------------------------------------------------------------
    archetype = rng.choice(
        ["high_high", "low_low", "mixed"], size=n_students, p=[0.30, 0.25, 0.45]
    )
    base_ei = np.where(archetype == "high_high", 4.0,
               np.where(archetype == "low_low", 2.0, 3.0))
    base_engagement = np.where(archetype == "high_high", 4.0,
                       np.where(archetype == "low_low", 2.0,
                       rng.choice([2.0, 4.0], size=n_students)))  # mixed = one high, one low
    # For the "mixed" archetype, decouple EI from engagement so the model
    # has to use *both* signals rather than one proxying for the other.
    mixed_mask = archetype == "mixed"
    base_ei[mixed_mask] = rng.choice([2.2, 3.8], size=mixed_mask.sum())

    latent_ei = {
        branch: np.clip(base_ei + rng.normal(0, 0.6, n_students), 1.0, 5.0)
        for branch in EI_BRANCHES
    }
    latent_eng = {
        dim: np.clip(base_engagement + rng.normal(0, 0.6, n_students), 1.0, 5.0)
        for dim in ENGAGEMENT_SELF_REPORT_DIMENSIONS
    }
    # Behavioral engagement latent trait (drives LMS logs) correlates with
    # but is not identical to self-reported engagement.
    latent_bei = np.clip(base_engagement + rng.normal(0, 0.7, n_students), 1.0, 5.0)

    # ------------------------------------------------------------------
    # 3. Surveys -- long format, item-level responses built by adding
    #    item-level noise around each student's latent branch/dimension
    #    score, for both administration waves (start + mid semester).
    #    Mid-semester wave has slightly more noise (regression toward
    #    the mean / measurement reliability over time) which is a mild,
    #    realistic complication for the ETL/feature code to handle.
    # ------------------------------------------------------------------
    survey_rows = []
    for wave, wave_noise in [("start_of_semester", 0.35), ("mid_semester", 0.45)]:
        for branch in EI_BRANCHES:
            item_vals = _clip_likert(
                latent_ei[branch][:, None] + rng.normal(0, wave_noise, (n_students, EI_ITEMS_PER_BRANCH))
            )
            for item_idx in range(EI_ITEMS_PER_BRANCH):
                survey_rows.append(pd.DataFrame({
                    "research_code": research_codes,
                    "wave": wave,
                    "instrument": "EI",
                    "construct_code": branch,
                    "item_number": item_idx + 1,
                    "response_value": item_vals[:, item_idx],
                }))
        for dim in ENGAGEMENT_SELF_REPORT_DIMENSIONS:
            item_vals = _clip_likert(
                latent_eng[dim][:, None] + rng.normal(0, wave_noise, (n_students, ENGAGEMENT_ITEMS_PER_DIMENSION))
            )
            for item_idx in range(ENGAGEMENT_ITEMS_PER_DIMENSION):
                survey_rows.append(pd.DataFrame({
                    "research_code": research_codes,
                    "wave": wave,
                    "instrument": "ENGAGEMENT",
                    "construct_code": dim,
                    "item_number": item_idx + 1,
                    "response_value": item_vals[:, item_idx],
                }))
    surveys_df = pd.concat(survey_rows, ignore_index=True)

    # Inject a small amount of realistic missingness (skipped items) for
    # the ETL completeness check to have something genuine to flag --
    # mirrors Section 3.6.2's "reviewed for completeness ... before they
    # go in".
    drop_idx = rng.choice(surveys_df.index, size=int(0.01 * len(surveys_df)), replace=False)
    surveys_df = surveys_df.drop(index=drop_idx).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 4. LMS_Metrics -- one row per student-module-week. Single module
    #    code "MOD1" used for simplicity; the schema already supports
    #    multiple modules per student if a real deployment needs it.
    #    Behavioral signal trends upward for high archetypes and decays
    #    for low/at-risk archetypes across the semester, which is what
    #    gives the Cohort Overview's engagement-drift time series
    #    (Section 3.6.4) something meaningful to show.
    # ------------------------------------------------------------------
    lms_rows = []
    week_trend = np.linspace(-0.3, 0.3, N_WEEKS)  # mild drift over the semester
    for week in range(1, N_WEEKS + 1):
        weekly_bei = np.clip(latent_bei + week_trend[week - 1] * (latent_bei - 3) + rng.normal(0, 0.4, n_students), 1, 5)
        login_count = np.clip(rng.poisson(lam=weekly_bei * 1.6), 0, None)
        submission_lag = rng.normal(loc=(3 - weekly_bei) * 8, scale=6, size=n_students)  # higher engagement -> earlier submission
        forum_posts = np.clip(rng.poisson(lam=np.maximum(weekly_bei - 1.5, 0.05)), 0, None)
        video_completion = np.clip(weekly_bei / 5 + rng.normal(0, 0.1, n_students), 0, 1)
        quiz_attempts = np.clip(rng.poisson(lam=np.maximum(weekly_bei - 1.0, 0.1)), 0, None)
        lms_rows.append(pd.DataFrame({
            "research_code": research_codes,
            "module_code": "MOD1",
            "week_number": week,
            "login_count": login_count,
            "avg_submission_lag_hrs": submission_lag.round(2),
            "forum_posts": forum_posts,
            "video_completion_rate": video_completion.round(3),
            "quiz_attempts": quiz_attempts,
        }))
    lms_df = pd.concat(lms_rows, ignore_index=True)

    # ------------------------------------------------------------------
    # 5. Outcomes -- final score is a noisy function of the latent traits
    #    (weighted sum + interaction term for the "synergy" pathway
    #    described in Section 2.9, plus noise), then bucketed into the
    #    three performance classes using the exact thresholds from
    #    Section 3.3.2 (>=70 high, 50-69 satisfactory, <50 at_risk).
    # ------------------------------------------------------------------
    ei_mean = np.mean([latent_ei[b] for b in EI_BRANCHES], axis=0)
    eng_mean = np.mean([latent_eng[d] for d in ENGAGEMENT_SELF_REPORT_DIMENSIONS] + [latent_bei], axis=0)
    interaction = (ei_mean - 3) * (eng_mean - 3)  # synergy term
    # Coefficients below were calibrated (see data/_calibration_notes.md) so
    # the resulting class split lands at roughly at_risk≈22%,
    # satisfactory≈37%, high≈41% -- matching Section 3.4.3's statement that
    # "students at risk tend to be a smaller class, maybe 20-30% of all
    # enrolments", rather than an arbitrary/uncalibrated split.
    raw_score = (
        15.0
        + 8.0 * ei_mean
        + 8.0 * eng_mean
        + 2.0 * interaction
        + rng.normal(0, 8.0, n_students)
    )
    final_score = np.clip(raw_score, 0, 100)

    performance_class = np.where(
        final_score >= PERFORMANCE_THRESHOLDS["high"], "high",
        np.where(final_score >= PERFORMANCE_THRESHOLDS["satisfactory"], "satisfactory", "at_risk"),
    )
    outcomes_df = pd.DataFrame({
        "research_code": research_codes,
        "final_score_pct": final_score.round(2),
        "performance_class": performance_class,
    })

    # ------------------------------------------------------------------
    # Write raw exports exactly as the ETL pipeline expects to receive
    # them: separate CSV files, as if exported from a survey platform and
    # the institutional LMS respectively (Section 3.6.2).
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    students.to_csv(output_dir / "students_export.csv", index=False)
    surveys_df.to_csv(output_dir / "surveys_export.csv", index=False)
    lms_df.to_csv(output_dir / "lms_log_export.csv", index=False)
    outcomes_df.to_csv(output_dir / "outcomes_export.csv", index=False)

    print(f"[generate_synthetic_data] wrote {n_students} students to {output_dir}")
    print(f"[generate_synthetic_data] performance class distribution:\n"
          f"{outcomes_df['performance_class'].value_counts(normalize=True).round(3)}")
    return students, surveys_df, lms_df, outcomes_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-students", type=int, default=600)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    generate(n_students=args.n_students, seed=args.seed)
