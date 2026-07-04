# EI & Engagement Academic Performance Predictor
### Prototype implementation — PhD Proposal ACE24140004  
**Designing an Improved KNN Model for Predicting Emotional Intelligence and Engagement Impact on Student Academic Performance in Distance Education**  
*Africa Centre of Excellence on Technology Enhanced Learning (ACETEL) / National Open University of Nigeria (NOUN)*

---

> **Synthetic demo only.** This codebase implements the full ML pipeline described in Chapters 1–3 of the PhD proposal, running on algorithmically-generated data. No real student survey or LMS records are included. Every chart and metric produced is illustrative of the *pipeline*, not a reported research finding. See `data/_calibration_notes.md` for how the synthetic data was calibrated.

---

## Quick start

```bash
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Run the complete 5-stage pipeline (generates data → ETL → features → train → predict)
python scripts/run_pipeline.py          # ~60 s on a modern laptop

# 3. Launch the interactive dashboard
streamlit run dashboard/app.py
# Opens at http://localhost:8501
```

---

## What this implements

The seven-stage process framework from **Section 3.3.5** of the proposal, end-to-end:

| Stage | Module | Proposal section |
|-------|--------|-----------------|
| 1 · Data Collection | `data/generate_synthetic_data.py` | §3.3.2 data sources |
| 2 · Preprocessing / ETL | `etl/etl_pipeline.py` | §3.3.5 step 2, §3.6.2 |
| 3 · Feature Engineering | `features/feature_engineering.py` | §3.3.5 step 3, §3.3.2 |
| 4 · Feature Scaling + Model Training | `ml/train.py` + `ml/preprocessing.py` | §3.3.3, §3.3.4, §3.3.5 steps 4–5 |
| 5 · Validation + Evaluation | `ml/train.py` (GridSearchCV, CV, baselines) | §3.3.4, §3.3.5 step 6 |
| 6 · Inference / Predictions | `ml/predict.py` | §3.3.1 online inference phase |
| 7 · Interpretation + Dashboard | `dashboard/` | §3.6.4, §3.3.5 step 7 |

---

## Architecture

```
distance_ed_ml/
├── config.py                      ← single source of truth (paths, constants, thresholds)
├── requirements.txt
├── data/
│   ├── generate_synthetic_data.py ← Stage 1: generates 4 raw CSV exports
│   ├── raw/                       ← survey_export.csv, lms_log_export.csv, …
│   └── processed/                 ← feature_matrix.parquet (built by Stage 3)
├── db/
│   ├── schema.sql                 ← 7-table SQLite schema
│   ├── db_init.py                 ← creates/resets the database
│   └── distance_ed.sqlite         ← the live database
├── etl/
│   └── etl_pipeline.py            ← Stage 2: validate, clean, load, log quality metrics
├── features/
│   └── feature_engineering.py     ← Stage 3: 11-predictor feature matrix
├── ml/
│   ├── label_utils.py             ← int↔string codec (sklearn 1.8 workaround)
│   ├── preprocessing.py           ← shared ColumnTransformer (scale + encode)
│   ├── model_pipeline.py          ← KNN+SMOTE pipeline + 3 baseline pipelines
│   ├── evaluate.py                ← metrics + subgroup fairness reporting
│   ├── train.py                   ← Stage 4+5: grid search, ablation, artifact save
│   ├── predict.py                 ← Stage 6: populate Predictions table
│   └── artifacts/                 ← 6 joblib models + model_comparison.csv + fairness CSVs
├── dashboard/
│   ├── auth.py                    ← Researcher vs. Institutional Staff role gate
│   ├── data_access.py             ← cached DB/artifact loaders
│   ├── app.py                     ← Streamlit entry point (4 tabs)
│   └── views/
│       ├── cohort_overview.py     ← class dist, correlation heatmap, LMS drift chart
│       ├── student_detail.py      ← per-student radar chart + neighbor cards
│       ├── intervention_tracking.py ← at-risk roster + outreach action log
│       └── model_inspection.py    ← confusion matrix, fairness bars, MI importance
└── scripts/
    └── run_pipeline.py            ← one-command full pipeline runner
```

---

## Key design decisions

Every decision that isn't obvious from the code has a `DECISION:` comment in the
relevant module's docstring. The most important ones:

| Decision | Where justified |
|----------|----------------|
| SQLite instead of PostgreSQL | `db/schema.sql` header |
| `imblearn.Pipeline` for SMOTE | `ml/model_pipeline.py` docstring |
| Deployment model omits SMOTE (interpretability) | `ml/train.py` §5 docstring |
| Integer label encoding (sklearn 1.8.0 bug fix) | `ml/label_utils.py` docstring |
| Plotly instead of matplotlib in dashboard | `dashboard/views/cohort_overview.py` DECISION note |
| Role selector not real auth | `dashboard/auth.py` docstring |
| Fairness measured, not auto-corrected | `ml/evaluate.py` docstring |

---

## Dashboard views

| Tab | Who sees it | What it shows |
|-----|-------------|--------------|
| 🏠 Cohort Overview | All | Predicted class distribution, LMS engagement drift, feature correlation (Researcher only) |
| 🔍 Student Detail | All | Per-student radar + KNN neighbor comparison cards |
| 📋 Interventions | All | At-risk roster, outreach action logging, coverage check |
| 🔬 Model Inspection | Researcher only | Confusion matrix, model comparison, MI feature importance, subgroup fairness |

---

## Last known results (synthetic cohort, seed=42, n=600)

```
Model                    Accuracy   F1-weighted   Recall@at_risk   AUC
knn_optimized_smote      0.675      0.663          0.792            0.821
knn_optimized_no_smote   0.717      0.712          0.500            0.837
logistic_regression      0.692      0.681          0.750            0.839
decision_tree            0.658      0.663          0.625            0.816
svm_rbf                  0.717      0.716          0.667            0.851
```

**Key finding demonstrated:** SMOTE nearly doubles at-risk recall (0.50 → 0.79) at a 4 pp accuracy cost — quantifying the minority-class correction effect described in Section 3.4.3.

Best KNN: `metric=manhattan, k=15, weights=uniform, PCA(n_components=0.95)`

---

## See also

- `HANDOVER.md` — open design tensions, next tasks, Chapter 4 mapping
- `data/_calibration_notes.md` — how synthetic outcome coefficients were calibrated
- Each module's docstring for `DECISION:` comments explaining implementation choices
