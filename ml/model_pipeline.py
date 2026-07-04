"""
ml/model_pipeline.py
=====================
Defines the optimized KNN pipeline and the three baseline classifier
pipelines from Sections 3.3.3-3.3.4.

DECISION: imblearn.pipeline.Pipeline instead of sklearn.pipeline.Pipeline.
SMOTE is a *resampling* step (it changes the number of rows, not just their
values), which scikit-learn's own Pipeline object cannot contain -- it
will raise a clear TypeError if you try ("All intermediate steps should be
transformers and implement fit and transform"). imbalanced-learn's drop-in
Pipeline replacement is built specifically to host resamplers, and it
makes the leakage-avoidance behavior from Section 3.3.3 *automatic* rather
than something we'd otherwise have to hand-roll: SMOTE only runs during
.fit() on whatever rows are in front of it in that fit call, never during
.predict()/.transform(), so when GridSearchCV/cross_validate splits the
data internally, each fold's SMOTE oversampling only ever sees that fold's
training rows. This is exactly the "applied only to the training splits
within cross validation, which helps the minority at-risk class be
properly represented without accidentally polishing the evaluation
results via test-set contamination" requirement in Section 3.3.3, achieved
by construction rather than by careful manual bookkeeping that could be
gotten wrong.

DECISION: PCA is modeled as a pipeline step that can be 'passthrough' (no
PCA) or PCA(n_components=0.90 / 0.95), searched as a hyperparameter inside
the same GridSearchCV as k/metric/weights, rather than as a separate
preliminary experiment. This directly implements "we test variance
retention levels of 90% and 95% as possible setups" (Section 3.3.3) as
one coherent search rather than three separate ad hoc runs, and it means
the reported "best" KNN configuration is genuinely the best *combination*
of scaling+PCA+k+metric+weights+SMOTE, not best-k-given-an-arbitrarily-
chosen-PCA-setting.
"""
import sys
from pathlib import Path

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    KNN_K_GRID, KNN_METRIC_GRID, KNN_MINKOWSKI_P, KNN_WEIGHTS_GRID,
    PCA_VARIANCE_GRID, RANDOM_SEED,
)
from ml.preprocessing import build_preprocessor


def _pca_grid_values():
    values = []
    for v in PCA_VARIANCE_GRID:
        values.append("passthrough" if v is None else PCA(n_components=v, random_state=RANDOM_SEED))
    return values


def build_knn_pipeline(use_smote: bool = True) -> ImbPipeline:
    """The optimized KNN pipeline: preprocess -> [PCA] -> [SMOTE] -> KNN.

    use_smote=False produces the "unbalanced baseline" variant referenced
    in Section 3.4.3 ("The impact of SMOTE is shown separately next to the
    unbalanced baseline numbers"), used by evaluate.py to quantify SMOTE's
    effect on minority (at_risk) class recall.
    """
    steps = [("preprocess", build_preprocessor()), ("pca", "passthrough")]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_SEED)))
    steps.append(("knn", KNeighborsClassifier()))
    return ImbPipeline(steps=steps)


def build_knn_param_grid() -> dict:
    """Search space implementing Section 3.3.3's four optimization levers
    that apply to KNN itself: scaling is always-on (not searched, since
    Section 3.3.3 treats it as a non-optional first step, not an ablation
    choice), PCA variance retention, k, distance metric, and (from Section
    2.7.3's "weighted KNN often gives a noticeable gain") neighbor weighting.
    """
    grid = {
        "pca": _pca_grid_values(),
        "knn__n_neighbors": KNN_K_GRID,
        "knn__metric": KNN_METRIC_GRID,
        "knn__weights": KNN_WEIGHTS_GRID,
        "knn__p": [KNN_MINKOWSKI_P],  # only takes effect when metric='minkowski'
    }
    return grid


def build_baseline_pipelines(use_smote: bool = True) -> dict:
    """The three baseline comparators from Section 3.3.4, sharing the
    identical preprocessing pipeline used by KNN (per that section's
    explicit "identical preprocessing pipeline" requirement). A light,
    fixed-budget hyperparameter grid is searched for each so the
    comparison isn't an unfairly under-tuned strawman, but the search
    is intentionally smaller than KNN's -- these are *comparators*, the
    proposal's optimization effort (Section 3.3.3) is specifically about
    KNN, not about exhaustively tuning the baselines too.
    """
    pipelines, grids = {}, {}

    def _steps(model):
        steps = [("preprocess", build_preprocessor())]
        if use_smote:
            steps.append(("smote", SMOTE(random_state=RANDOM_SEED)))
        steps.append(("model", model))
        return ImbPipeline(steps=steps)

    pipelines["logistic_regression"] = _steps(
        LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
    )
    grids["logistic_regression"] = {"model__C": [0.1, 1.0, 10.0]}

    pipelines["decision_tree"] = _steps(
        DecisionTreeClassifier(random_state=RANDOM_SEED)
    )
    grids["decision_tree"] = {"model__max_depth": [3, 5, 8, None], "model__min_samples_leaf": [1, 5, 10]}

    pipelines["svm_rbf"] = _steps(
        SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED)
    )
    grids["svm_rbf"] = {"model__C": [0.5, 1.0, 5.0], "model__gamma": ["scale", "auto"]}

    return pipelines, grids
