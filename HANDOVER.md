# SESSION HANDOVER — Distance Education KNN Predictor
# PhD Proposal: ACE24140004 (ACETEL / NOUN)
# Handover written: 2026-07-01

---

## 1. WHAT HAS BEEN BUILT (complete and tested)

All stages run end-to-end. A clean run takes ~61 seconds:

```bash
cd distance_ed_ml
python scripts/run_pipeline.py       # ~61 s: generates data → ETL → features → train → predict
streamlit run dashboard/app.py       # opens the dashboard on localhost:8501
```

### Stage outputs

| Stage | Module | Output |
|-------|--------|--------|
| 1 | `data/generate_synthetic_data.py` | 4 CSV files in `data/raw/` |
| 2 | `etl/etl_pipeline.py` | SQLite DB populated; DataQualityLog written |
| 3 | `features/feature_engineering.py` | `data/processed/feature_matrix.parquet` |
| 4 | `ml/train.py` | 6 joblib artifacts + CSVs in `ml/artifacts/` |
| 5 | `ml/predict.py` | 600 rows in `Predictions` table (with neighbor JSON) |
| — | `dashboard/app.py` | 4-tab Streamlit app (3 views + Researcher) |

### Last known training results (synthetic cohort, seed=42, n=600)

```
Model                     Accuracy  F1-weighted  Precision@AR  Recall@AR   AUC
knn_optimized_smote       0.6750    0.6630        0.4419        0.7917      0.8210
knn_optimized_no_smote    0.7167    0.7120        0.5000        0.5000      0.8368
logistic_regression       0.6917    0.6812        0.4865        0.7500      0.8392
decision_tree             0.6583    0.6629        0.4412        0.6250      0.8164
svm_rbf                   0.7167    0.7159        0.5333        0.6667      0.8510
```

Best KNN params: `metric=manhattan, n_neighbors=15, weights=uniform, PCA(n_components=0.95)`

SMOTE nearly doubles at-risk recall (0.50 → 0.79) at a 4 pp accuracy cost — the key
quantitative finding the proposal promised to demonstrate in Section 3.4.3.

---

## 2. FILE TREE

```
distance_ed_ml/
├── config.py                          # single source of truth for paths/constants
├── requirements.txt
├── data/
│   ├── _calibration_notes.md          # why the outcome-score coefficients were chosen
│   ├── generate_synthetic_data.py     # Stage 1 – synthetic raw CSV exports
│   ├── raw/                           # 4 CSV files (git-ignored in a real repo)
│   └── processed/                     # feature_matrix.parquet
├── db/
│   ├── schema.sql                     # 7-table SQLite schema (4 from proposal + 3 added)
│   ├── db_init.py                     # creates/resets the database
│   └── distance_ed.sqlite             # the live database (git-ignored in a real repo)
├── etl/
│   └── etl_pipeline.py                # Stage 2 – validate, clean, load
├── features/
│   └── feature_engineering.py         # Stage 3 – build 11-predictor feature matrix
├── ml/
│   ├── label_utils.py                 # int↔string class label codec (sklearn 1.8.0 fix)
│   ├── preprocessing.py               # ColumnTransformer shared by all models
│   ├── model_pipeline.py              # KNN+SMOTE pipeline + 3 baseline pipelines
│   ├── evaluate.py                    # metrics + fairness reporting
│   ├── train.py                       # Stage 4 – grid search, ablation, artifacts
│   ├── predict.py                     # Stage 5 – populate Predictions table
│   └── artifacts/                     # 6 joblib files + CSVs
├── dashboard/
│   ├── auth.py                        # role selector (Researcher vs Institutional Staff)
│   ├── data_access.py                 # cached DB / artifact loaders
│   ├── app.py                         # Streamlit entry point (4 tabs)
│   └── views/
│       ├── cohort_overview.py         # Tab 1: class dist / heatmap / LMS drift
│       ├── student_detail.py          # Tab 2: per-student + neighbor cards + radar
│       ├── intervention_tracking.py   # Tab 3: at-risk roster + action log
│       └── model_inspection.py        # Tab 4 (Researcher only): CM / fairness / MI
└── scripts/
    └── run_pipeline.py                # one-command full pipeline runner
```

---

## 3. OPEN DESIGN TENSIONS (things deferred deliberately)

### 3a. SMOTE vs. interpretability — the deployment model tension

The **evaluated** model (saved as `knn_evaluated_smote.joblib`) uses SMOTE and
achieves 79% at-risk recall. The **deployment** model (saved as `knn_deployment.joblib`,
the one the dashboard reads from) was deliberately retrained WITHOUT SMOTE so that
every nearest neighbor returned by `knn.kneighbors()` maps back to a real student's
research_code and can be displayed as an "anonymized comparison card" (Section 3.6.4).
A SMOTE-synthetic neighbor has no real student behind it and cannot be shown.

**If a next iteration prioritises at-risk recall over neighbor display fidelity:**
Option A — Use the SMOTE model for prediction and separately run the no-SMOTE model
only to retrieve the 5 neighbors for the card display. Two models, two calls per
inference, but both outputs are correct on their own terms.
Option B — Use SMOTE's training-set expanded data, mark which rows are synthetic,
and show "similar profile (synthetic)" cards for synthetic neighbors.
Option C — Switch to cost-sensitive KNN (class_weight parameter or a custom
distance-weighted voting scheme) instead of SMOTE; no synthetic rows created,
so interpretability is preserved while minority-class weighting is still applied.
Option C is closest to the proposal's Section 3.3.3 spirit (SMOTE is listed as
one of four strategies, not the only one).

### 3b. sklearn 1.8.0 KNeighborsClassifier bug with string labels

When `metric='minkowski'` and `weights='uniform'` (or certain other combinations),
`predict_proba()` raises `ValueError: invalid literal for int() with base 10: 'at_risk'`
inside a Cython fast-path that assumes integer class labels. The fix (encode/decode in
`ml/label_utils.py`) works but forces every consumer to remember to decode after
`.predict()` / decode-is-already-applied-to-.predict_proba()`.

**If upgrading sklearn:** test first with the label-encoding layer removed to see if
the bug is fixed in the new version. If it is, remove `label_utils.py` and the
`encode()`/`decode()` calls from `ml/train.py` and `ml/predict.py`. If it isn't,
leave the layer in place but consider filing a sklearn bug report with a minimal
repro (the conditions under which it fires: string y, metric='minkowski', sklearn≥1.7).

### 3c. Authentication is a role selector, not real auth

`dashboard/auth.py` lets any user pick "Researcher" or "Institutional Staff" from a
dropdown. This is appropriate for the synthetic-data prototype; it would be an ethics
violation on real student data. Before connecting real data:

1. Replace the radio button with your institution's SSO/OAuth integration
   (Streamlit supports `streamlit-oauth`, `streamlit-authenticator`, or a custom
   reverse-proxy with header-forwarded identity).
2. Add per-request audit logging to `DataQualityLog` or a new `AccessLog` table:
   who, what research_code, when.
3. Consider whether the `researcher` role should ever have direct
   individual-student access on the live system, or only aggregate/anonymised views.

### 3d. Prediction scope is in-sample for the demo

`ml/predict.py` scores the same 600 students the model was trained on. This is fine for
a dry-run but misleading if cited as "model performance" — that number comes from
`ml/artifacts/model_comparison.csv` (held-out 20% test set only). When real data exists:
- Train on cohort N-1 (Section 3.3.5 "temporal validation on a later cohort").
- Score only cohort N (the genuinely unseen incoming students).
- Prediction population then becomes a one-way write (no student in the prediction set
  ever appeared in the training set).

### 3e. PostgreSQL migration path

The schema is designed for it. To migrate:
1. Change `get_connection()` in `db/db_init.py` to return a SQLAlchemy engine or a
   psycopg2 connection to your PostgreSQL instance.
2. Replace `PRAGMA foreign_keys = ON` with nothing (PostgreSQL enforces FKs by default).
3. Change `INTEGER PRIMARY KEY AUTOINCREMENT` → `SERIAL PRIMARY KEY` or `BIGSERIAL`.
4. Change `datetime('now')` → `NOW()` in schema.sql.
5. The ETL, feature, ML, and dashboard layers are all database-agnostic (they use pandas
   `read_sql` / `to_sql` / connection objects) — they do not need to change.

---

## 4. WHAT CHAPTER 4 NEEDS FROM THIS IMPLEMENTATION

The proposal's Chapter 4 ("findings from data collection, exploratory analysis, and
the actual model training part") maps to specific outputs this codebase already produces:

| Chapter 4 element | Where to find it |
|------------------|-----------------|
| Class balance (before/after SMOTE) | `model_comparison.csv`, `train.py` stdout |
| KNN hyperparameter selection (best params, CV score) | `training_metadata.json` |
| Model comparison table (all 5 models × 5 metrics) | `model_comparison.csv` |
| Confusion matrix | `knn_confusion_matrix.csv` |
| Subgroup fairness | `fairness_gender.csv`, `fairness_age_bracket.csv`, `fairness_programme_type.csv` |
| Feature importance (MI) | Computed live in `model_inspection.py`; save with `mutual_info_classif()` |
| SMOTE ablation (recall gain) | Row comparison in `model_comparison.csv` |
| "Example-based" neighbor explanation | `Predictions.neighbor_research_codes` (JSON) |
| EI branch vs. engagement dimension importance ordering | MI plot in Model Inspection tab |

When real data replaces the synthetic cohort, every output file regenerates automatically
from the same pipeline — Chapter 4 writing can reference the exact same CSV/table names.

---

## 5. NEXT IMMEDIATE TASKS (priority order for the next session)

### HIGH PRIORITY
1. **Real data integration** (when available from institutional partner):
   - Drop real CSVs into `data/raw/` matching the column schemas in `etl/etl_pipeline.py`
   - Run `python scripts/run_pipeline.py --skip-generate`
   - Inspect the DataQualityLog output carefully before training
   - Replace the in-sample prediction scope (item 3d above)

2. **Temporal validation** (Section 3.3.5 item 6):
   - If two cohort-years of data become available, add a `run_temporal_validation()`
     function to `ml/train.py` that trains on year N-1 and tests on year N
   - This is the validation mode the proposal explicitly calls out as better than
     cross-validation for assessing real-world generalization

3. **SMOTE vs. interpretability resolution** (item 3a above):
   - Decide with the supervisory team which trade-off is the research priority
   - Implement Option A (dual-model inference) or Option C (cost-sensitive KNN)

### MEDIUM PRIORITY
4. **Feature importance as a proper artifact** (Chapter 4 §3):
   - `ml/evaluate.py` currently has no feature-importance function; the MI plot in
     `model_inspection.py` computes it inline on every render
   - Move to `ml/train.py`, persist as `ml/artifacts/feature_importance_mi.csv`,
     and load it in the dashboard view like the other artifacts

5. **Permutation importance** alongside MI:
   - Section 3.3.5 mentions "neighbor inspection readouts"; Section 2.7.3 suggests
     feature weighting. Add `sklearn.inspection.permutation_importance` on the best
     KNN model as a post-training step in `ml/train.py` — it gives a model-specific
     (not just statistical) importance measure and is a stronger result for the thesis

6. **Start-of-semester vs. mid-semester EI comparison** (Section 3.3.2):
   - The feature engineering currently uses only the mid-semester survey wave.
   - Adding a `delta_EI` feature (mid - start per branch) would operationalise the
     "temporal LMS analytics features" motivation in the proposal and might improve
     accuracy on the mixed-profile archetype specifically.

### LOWER PRIORITY
7. Extend `generate_synthetic_data.py` to simulate TWO cohort years for temporal validation dry-runs.
8. Add a `scripts/export_chapter4_tables.py` that pulls the artifact CSVs and formats them
   as LaTeX tables (useful when the Chapter 4 write-up stage begins).
9. Write a `.streamlit/config.toml` to set theme colors and `server.headless = true`
   so the dashboard launches cleanly in a headless server environment without prompts.

---

## 6. ENVIRONMENT SNAPSHOT

```
Python  3.12.3
sklearn 1.8.0
pandas  3.0.2
numpy   2.4.4
plotly  6.8.0
imbalanced-learn 0.14.2
joblib  1.5.3
streamlit 1.58.0
pyarrow (any ≥ 15.0)
SQLite  3 (built-in)
OS: Ubuntu 24
```

Reproducibility: every random operation is seeded via `config.RANDOM_SEED = 42` passed
explicitly to numpy's default_rng, sklearn's `random_state`, and SMOTE's `random_state`.
Running `python scripts/run_pipeline.py` twice from the same starting state should
produce bit-identical artifact files.

---

## 7. KEY DESIGN DECISIONS (summary reference)

| Decision | Where justified |
|----------|----------------|
| SQLite not PostgreSQL | `db/schema.sql` header, §3e above |
| imblearn Pipeline for SMOTE containment | `ml/model_pipeline.py` module docstring |
| Deployment model omits SMOTE | `ml/train.py` step 5 docstring, §3a above |
| Integer label encoding | `ml/label_utils.py` module docstring, §3b above |
| Plotly not matplotlib in dashboard | `dashboard/views/cohort_overview.py` DECISION note |
| Role selector not real auth | `dashboard/auth.py` module docstring, §3c above |
| Long-format Surveys table | `db/schema.sql` Surveys table comment |
| BEI as weighted z-score composite | `features/feature_engineering.py` DECISION note |
| Synthetic data, not fabricated "real" data | `data/generate_synthetic_data.py` WHY THIS EXISTS |
| f1_weighted as grid-search criterion | `ml/train.py` module docstring DECISION note |
| Neighbor display k=5 ≠ classification k | `ml/predict.py` module docstring |
| Fairness reported, not auto-corrected | `ml/evaluate.py` module docstring |

End of handover document.
