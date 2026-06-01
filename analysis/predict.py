"""
Multimodal EEG/fMRI fusion prediction pipeline.

Steps per condition (EEG-only, fMRI-only, multimodal):
  1. Keep only subjects present in all inputs.
  2. GLM confound correction: regress out age, gender, study_site
     (minus the target column if it is one of those).
  3. Nested cross-validation:
       outer k-fold — evaluate generalisation
       inner k-fold — tune hyperparameters
     Inside each outer fold:
       a. Fit PCA on training split (explained variance threshold).
       b. Transform train and test with that PCA.
       c. GridSearchCV (inner CV) to pick best hyperparameters.
       d. Evaluate on outer test split.
  4. Collect per-fold scores.
  5. Permutation test (shuffle y) to build a null distribution and
     estimate p-value vs chance.
  6. Paired t-test between EEG-only and fMRI-only, EEG-only and
     multimodal, fMRI-only and multimodal.

Outputs:
  output_data/results/metrics.tsv   — one row per condition with
      mean/std scores and significance values.
  output_data/results/fold_scores.tsv — raw per-fold scores.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR


# ── task detection ──────────────────────────────────────────────────────────

def detect_task(y: pd.Series) -> Literal["classification", "regression"]:
    """Return 'classification' if y is categorical or has ≤10 unique integer-like values."""
    if y.dtype.kind in ("i", "u") or set(y.unique()) <= {0, 1}:
        return "classification"
    if y.dtype.kind in ("O", "S", "U"):  # object / string dtype → categorical
        return "classification"
    try:
        if y.nunique() <= 5 and all(float(v).is_integer() for v in y.dropna()):
            return "classification"
    except (ValueError, TypeError):
        return "classification"
    return "regression"


# ── model catalogue ─────────────────────────────────────────────────────────

_CLF_CATALOGUE = {
    "logistic": (
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        {"model__C": [0.01, 0.1, 1.0, 10.0]},
    ),
    "ridge": (
        LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000, class_weight="balanced"),
        {"model__C": [0.01, 0.1, 1.0, 10.0]},
    ),
    "elasticnet": (
        LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=1000, class_weight="balanced"
        ),
        {"model__C": [0.01, 0.1, 1.0], "model__l1_ratio": [0.2, 0.5, 0.8]},
    ),
    "svm": (
        SVC(kernel="rbf", class_weight="balanced", probability=True),
        {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", "auto"]},
    ),
    "random_forest": (
        RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=0),
        {"model__max_depth": [None, 5, 10], "model__min_samples_leaf": [1, 3]},
    ),
}

_REG_CATALOGUE = {
    "logistic": (
        Ridge(),
        {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    ),
    "ridge": (
        Ridge(),
        {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    ),
    "elasticnet": (
        ElasticNet(max_iter=5000),
        {"model__alpha": [0.01, 0.1, 1.0], "model__l1_ratio": [0.2, 0.5, 0.8]},
    ),
    "svm": (
        SVR(kernel="rbf"),
        {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", "auto"]},
    ),
    "random_forest": (
        RandomForestRegressor(n_estimators=100, random_state=0),
        {"model__max_depth": [None, 5, 10], "model__min_samples_leaf": [1, 3]},
    ),
}


def _get_estimator_and_grid(model_type: str, task_type: str):
    catalogue = _CLF_CATALOGUE if task_type == "classification" else _REG_CATALOGUE
    if model_type not in catalogue:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose from: {', '.join(catalogue)}."
        )
    return catalogue[model_type]


# ── confound correction ─────────────────────────────────────────────────────

def correct_confounds(
    features_df: pd.DataFrame,
    phenotype_df: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """
    GLM confound correction: regress age, gender, study_site out of features.
    Confounds that equal the target column are skipped.

    Returns a DataFrame with the same columns as features_df but with
    confound variance removed (residuals). participant_id is preserved.
    """
    confound_cols = [c for c in ("age", "gender", "study_site") if c != target_col]
    available = [c for c in confound_cols if c in phenotype_df.columns]

    if not available:
        return features_df.copy()

    # Align on participant_id
    merged = features_df.merge(phenotype_df[["participant_id"] + available], on="participant_id", how="left")

    feature_cols = [c for c in features_df.columns if c != "participant_id"]
    confounds = pd.get_dummies(merged[available], drop_first=True).astype(float)
    confounds.insert(0, "intercept", 1.0)

    X_conf = confounds.values
    X_feat = merged[feature_cols].values.astype(float)

    # OLS: residuals = X_feat - X_conf @ pinv(X_conf) @ X_feat
    beta = np.linalg.lstsq(X_conf, X_feat, rcond=None)[0]
    residuals = X_feat - X_conf @ beta

    result = pd.DataFrame(residuals, columns=feature_cols)
    result.insert(0, "participant_id", features_df["participant_id"].values)
    return result


# ── nested cross-validation ─────────────────────────────────────────────────

def _score(y_true, y_pred, task_type: str, y_score=None) -> dict:
    if task_type == "classification":
        scores = {"balanced_accuracy": balanced_accuracy_score(y_true, y_pred)}
        if y_score is not None:
            try:
                classes = np.unique(y_true)
                if len(classes) == 2:
                    auc = roc_auc_score(y_true, y_score[:, 1] if y_score.ndim == 2 else y_score)
                else:
                    auc = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
                scores["roc_auc"] = auc
            except Exception:
                scores["roc_auc"] = float("nan")
        return scores
    else:
        r, _ = stats.pearsonr(y_true, y_pred)
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "pearson_r": r,
        }


def _extract_feature_importance(gs, pca: PCA):
    """
    Project model importance back to original feature space via PCA loadings.

    For linear models (coef_): weight vector in standardised PCA space is
    divided by the scaler's scale then dotted with |pca.components_| to obtain
    per-original-feature importance magnitudes.

    For RandomForest (feature_importances_): MDI scores in PCA space are
    distributed back to original features proportionally to loading magnitudes.

    Returns None for SVM RBF (no analytic importance available).
    """
    best_pipe = gs.best_estimator_
    model = best_pipe.named_steps["model"]
    scaler = best_pipe.named_steps["scaler"]
    loadings = np.abs(pca.components_)  # (n_components, n_features)

    if hasattr(model, "coef_"):
        coef = np.atleast_2d(model.coef_)           # (n_classes_or_1, n_components)
        coef_1d = np.mean(np.abs(coef), axis=0)     # average across classes
        coef_raw = coef_1d / scaler.scale_           # undo StandardScaler
        return coef_raw @ loadings                   # (n_features,)

    if hasattr(model, "feature_importances_"):
        return model.feature_importances_ @ loadings  # (n_features,)

    return None  # SVM RBF


def _run_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    task_type: str,
    n_outer: int = 5,
    n_inner: int = 5,
    pca_variance: float = 0.95,
    random_state: int = 0,
    return_importances: bool = False,
):
    """Return per-fold score dicts, and optionally per-fold importance arrays."""
    is_clf = task_type == "classification"
    outer_cv = (
        StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=random_state)
        if is_clf
        else KFold(n_splits=n_outer, shuffle=True, random_state=random_state)
    )
    inner_cv = (
        StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=random_state)
        if is_clf
        else KFold(n_splits=n_inner, shuffle=True, random_state=random_state)
    )

    estimator, param_grid = _get_estimator_and_grid(model_type, task_type)
    fold_scores = []
    fold_importances = [] if return_importances else None

    for train_idx, test_idx in outer_cv.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Impute residual NaN (fitted on train only — no leakage)
        imputer = SimpleImputer(strategy="median")
        X_tr = imputer.fit_transform(X_tr)
        X_te = imputer.transform(X_te)

        # PCA fitted only on training data
        n_components = min(pca_variance, X_tr.shape[0] - 1, X_tr.shape[1])
        pca = PCA(n_components=n_components, random_state=random_state)
        X_tr_pca = pca.fit_transform(X_tr)
        X_te_pca = pca.transform(X_te)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gs = GridSearchCV(
                pipe,
                param_grid,
                cv=inner_cv,
                scoring="roc_auc" if is_clf else "neg_mean_absolute_error",
                n_jobs=-1,
                refit=True,
            )
            gs.fit(X_tr_pca, y_tr)

        y_pred = gs.predict(X_te_pca)
        y_score = gs.predict_proba(X_te_pca) if is_clf and hasattr(gs, "predict_proba") else None
        fold_scores.append(_score(y_te, y_pred, task_type, y_score=y_score))

        if return_importances:
            fold_importances.append(_extract_feature_importance(gs, pca))

    if return_importances:
        return fold_scores, fold_importances
    return fold_scores


# ── permutation test ────────────────────────────────────────────────────────

def _permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    task_type: str,
    n_outer: int,
    n_inner: int,
    pca_variance: float,
    n_permutations: int,
    primary_metric: str,
    random_state: int = 0,
    label: str = "",
) -> float:
    """Return p-value vs chance.

    For metrics where higher is better (AUC, Pearson r): fraction of null scores >= observed.
    For metrics where lower is better (MAE): fraction of null scores <= observed.
    """
    lower_is_better = primary_metric == "mae"
    rng = np.random.default_rng(random_state)
    observed_folds = _run_nested_cv(X, y, model_type, task_type, n_outer, n_inner, pca_variance, random_state)
    observed = np.mean([f[primary_metric] for f in observed_folds])

    null_scores = []
    milestone = max(1, n_permutations // 10)
    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        perm_folds = _run_nested_cv(
            X, y_perm, model_type, task_type, n_outer, n_inner, pca_variance, random_state + i + 1
        )
        null_scores.append(np.mean([f[primary_metric] for f in perm_folds]))
        if (i + 1) % milestone == 0:
            print(f"[predict]   {label} permutation {i + 1}/{n_permutations}", flush=True)

    null_arr = np.array(null_scores)
    if lower_is_better:
        p_value = (np.sum(null_arr <= observed) + 1) / (n_permutations + 1)
    else:
        p_value = (np.sum(null_arr >= observed) + 1) / (n_permutations + 1)
    return p_value


# ── main entry point ────────────────────────────────────────────────────────

def run_prediction(
    eeg_df: pd.DataFrame,
    fmri_df: pd.DataFrame,
    phenotype_df: pd.DataFrame,
    target_col: str,
    model_type: str = "ridge",
    n_outer: int = 5,
    n_inner: int = 5,
    pca_variance: float = 0.95,
    n_permutations: int = 100,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run EEG-only, fMRI-only, and multimodal prediction.

    Returns:
        metrics_df    — one row per condition with mean/std scores and p-values
        fold_df       — raw per-fold scores (long format)
        importance_df — per-feature importance (mean/std across folds, tagged by modality);
                        empty DataFrame for models without analytic importance (SVM RBF)
    """
    # ── align subjects ──────────────────────────────────────────────────────
    common_ids = (
        set(eeg_df["participant_id"])
        & set(fmri_df["participant_id"])
        & set(phenotype_df["participant_id"])
    )
    common_ids = sorted(common_ids)
    if not common_ids:
        raise ValueError("No subjects in common across EEG, fMRI, and phenotype.")
    print(f"[predict] {len(common_ids)} subjects in common across all inputs.", flush=True)

    eeg_df = eeg_df[eeg_df["participant_id"].isin(common_ids)].sort_values("participant_id").reset_index(drop=True)
    fmri_df = fmri_df[fmri_df["participant_id"].isin(common_ids)].sort_values("participant_id").reset_index(drop=True)
    phenotype_df = phenotype_df[phenotype_df["participant_id"].isin(common_ids)].sort_values("participant_id").reset_index(drop=True)

    y_series = phenotype_df.set_index("participant_id").loc[common_ids, target_col]
    task_type = detect_task(y_series)

    # Keep strings as-is for classification (sklearn handles them); cast to float for regression
    y = y_series.values if task_type == "classification" else y_series.values.astype(float)
    print(f"[predict] Task type: {task_type} | Target: {target_col} | Model: {model_type}", flush=True)
    primary_metric = "roc_auc" if task_type == "classification" else "mae"

    # ── confound correction ─────────────────────────────────────────────────
    eeg_corrected = correct_confounds(eeg_df, phenotype_df, target_col)
    fmri_corrected = correct_confounds(fmri_df, phenotype_df, target_col)

    eeg_feat = [c for c in eeg_corrected.columns if c != "participant_id"]
    fmri_feat = [c for c in fmri_corrected.columns if c != "participant_id"]

    X_eeg = eeg_corrected[eeg_feat].values.astype(float)
    X_fmri = fmri_corrected[fmri_feat].values.astype(float)
    X_multi = np.hstack([X_eeg, X_fmri])

    print(f"[predict] EEG features: {X_eeg.shape[1]} | fMRI features: {X_fmri.shape[1]}", flush=True)

    # ── nested CV per condition ─────────────────────────────────────────────
    conditions = {
        "eeg_only": X_eeg,
        "fmri_only": X_fmri,
        "multimodal": X_multi,
    }

    fold_records = []
    condition_folds = {}
    condition_importances = {}

    for cond_name, X in conditions.items():
        print(f"[predict] Running nested CV: {cond_name} …", flush=True)
        folds, importances = _run_nested_cv(
            X, y, model_type, task_type, n_outer, n_inner, pca_variance, random_state,
            return_importances=True,
        )
        condition_folds[cond_name] = folds
        condition_importances[cond_name] = importances
        for fold_i, scores in enumerate(folds):
            for metric, value in scores.items():
                fold_records.append({
                    "condition": cond_name,
                    "fold": fold_i,
                    "metric": metric,
                    "value": value,
                })

    fold_df = pd.DataFrame(fold_records)

    # ── permutation tests ───────────────────────────────────────────────────
    perm_pvalues = {}
    for cond_name, X in conditions.items():
        print(f"[predict] Permutation test: {cond_name} ({n_permutations} permutations) …", flush=True)
        perm_pvalues[cond_name] = _permutation_test(
            X, y, model_type, task_type, n_outer, n_inner, pca_variance,
            n_permutations, primary_metric, random_state, label=cond_name
        )

    # ── paired t-tests between conditions ──────────────────────────────────
    def _primary_scores(cond):
        return np.array([f[primary_metric] for f in condition_folds[cond]])

    pairs = [
        ("eeg_only", "fmri_only"),
        ("eeg_only", "multimodal"),
        ("fmri_only", "multimodal"),
    ]
    paired_pvalues = {}
    for a, b in pairs:
        diffs = _primary_scores(a) - _primary_scores(b)
        if np.std(diffs) == 0:
            paired_pvalues[f"{a}_vs_{b}"] = 1.0
        else:
            _, p = stats.ttest_rel(_primary_scores(a), _primary_scores(b))
            paired_pvalues[f"{a}_vs_{b}"] = p

    # ── build metrics table ─────────────────────────────────────────────────
    metric_rows = []
    all_metrics = list(condition_folds["eeg_only"][0].keys())

    for cond_name in ("eeg_only", "fmri_only", "multimodal"):
        row = {"condition": cond_name, "model": model_type, "task": task_type, "target": target_col}
        for metric in all_metrics:
            vals = [f[metric] for f in condition_folds[cond_name]]
            row[f"{metric}_mean"] = np.mean(vals)
            row[f"{metric}_std"] = np.std(vals)
        row["p_vs_chance"] = perm_pvalues[cond_name]
        for pair_key, p_val in paired_pvalues.items():
            row[f"p_{pair_key}"] = p_val
        metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)

    # ── feature importance table ────────────────────────────────────────────
    feat_info = {
        "eeg_only":   (eeg_feat,            ["eeg"]  * len(eeg_feat)),
        "fmri_only":  (fmri_feat,           ["fmri"] * len(fmri_feat)),
        "multimodal": (eeg_feat + fmri_feat, ["eeg"] * len(eeg_feat) + ["fmri"] * len(fmri_feat)),
    }
    importance_records = []
    for cond_name, (feat_names, modality_tags) in feat_info.items():
        imp_list = condition_importances[cond_name]
        if any(imp is None for imp in imp_list):
            print(
                f"[predict] Feature importance not available for {cond_name} "
                f"(model '{model_type}' has no analytic importance — skipped)",
                flush=True,
            )
            continue
        imp_matrix = np.stack(imp_list)          # (n_folds, n_features)
        imp_mean = imp_matrix.mean(axis=0)
        imp_std  = imp_matrix.std(axis=0)
        for feat, mod, mean_val, std_val in zip(feat_names, modality_tags, imp_mean, imp_std):
            importance_records.append({
                "condition":       cond_name,
                "feature":         feat,
                "modality":        mod,
                "importance_mean": mean_val,
                "importance_std":  std_val,
            })

    importance_df = pd.DataFrame(importance_records)
    return metrics_df, fold_df, importance_df
