"""
models.py
---------
Model zoo used AFTER cleaning + preprocessing selection.

Supervised (a target column exists)
    classification : Logistic Regression, Random Forest, Extra Trees, Gradient Boosting,
                     HistGradientBoosting, Decision Tree, k-Nearest Neighbors, Naive Bayes,
                     SVM, AdaBoost, Neural Network (MLP) (+ XGBoost / LightGBM if installed)
    regression     : Linear, Ridge, Lasso, ElasticNet, Random Forest, Extra Trees,
                     Gradient Boosting, HistGradientBoosting, Decision Tree, k-NN, SVR,
                     AdaBoost, Neural Network (MLP) (+ XGBoost / LightGBM if installed)

Unsupervised (no target)
    clustering        : KMeans (with automatic k), MiniBatchKMeans, Agglomerative (Ward),
                        Gaussian Mixture, BIRCH, DBSCAN   -> silhouette / Davies-Bouldin / Calinski-Harabasz
    anomaly detection : Isolation Forest, Local Outlier Factor, One-Class SVM (+ consensus)
    dimensionality    : PCA (explained variance + 2-D projection)

Every model is evaluated with k-fold cross-validation on the same folds through the
same preprocessing pipeline (scaling is applied only to models that need it).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import Birch, DBSCAN, AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier,
                              ExtraTreesRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, IsolationForest,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (auc, calinski_harabasz_score, confusion_matrix, davies_bouldin_score,
                             roc_curve, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_predict, cross_validate,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, LocalOutlierFactor, NearestNeighbors
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR, OneClassSVM
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .transform import PrepConfig, build_preprocessor

SEED = 42
MAX_MODEL_ROWS = 20_000
SPEED_CV = {"fast": 3, "balanced": 3, "thorough": 5}
SPEED_LEVEL = {"fast": 0, "balanced": 1, "thorough": 2}


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    name: str
    family: str
    build: Callable[[], object]
    scale: bool = False
    min_speed: str = "fast"      # fast | balanced | thorough


def _optional_boosters(task: str) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    try:
        import xgboost as xgb  # type: ignore
        if task == "classification":
            specs.append(ModelSpec("XGBoost", "boosting", lambda: xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.9, n_jobs=-1,
                random_state=SEED, verbosity=0), min_speed="balanced"))
        else:
            specs.append(ModelSpec("XGBoost", "boosting", lambda: xgb.XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.9, n_jobs=-1,
                random_state=SEED, verbosity=0), min_speed="balanced"))
    except Exception:
        pass
    try:
        import lightgbm as lgb  # type: ignore
        if task == "classification":
            specs.append(ModelSpec("LightGBM", "boosting", lambda: lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.1, random_state=SEED, verbose=-1), min_speed="balanced"))
        else:
            specs.append(ModelSpec("LightGBM", "boosting", lambda: lgb.LGBMRegressor(
                n_estimators=200, learning_rate=0.1, random_state=SEED, verbose=-1), min_speed="balanced"))
    except Exception:
        pass
    return specs


def supervised_specs(task: str, n_rows: int, n_estimators: int = 150) -> List[ModelSpec]:
    if task == "classification":
        small = n_rows <= 5000
        specs = [
            ModelSpec("Logistic Regression", "linear", lambda: LogisticRegression(max_iter=2000), scale=True),
            ModelSpec("Random Forest", "tree ensemble", lambda: RandomForestClassifier(
                n_estimators=n_estimators, random_state=SEED, n_jobs=-1)),
            ModelSpec("Extra Trees", "tree ensemble", lambda: ExtraTreesClassifier(
                n_estimators=n_estimators, random_state=SEED, n_jobs=-1), min_speed="balanced"),
            ModelSpec("Gradient Boosting", "boosting", lambda: GradientBoostingClassifier(random_state=SEED),
                      min_speed="balanced"),
            ModelSpec("HistGradient Boosting", "boosting", lambda: HistGradientBoostingClassifier(random_state=SEED)),
            ModelSpec("Decision Tree", "tree", lambda: DecisionTreeClassifier(max_depth=8, random_state=SEED)),
            ModelSpec("k-Nearest Neighbors", "instance-based", lambda: KNeighborsClassifier(n_neighbors=15),
                      scale=True, min_speed="balanced"),
            ModelSpec("Naive Bayes", "probabilistic", lambda: GaussianNB()),
            ModelSpec("SVM (RBF)" if small else "SVM (linear)", "kernel / margin",
                      (lambda: SVC(kernel="rbf", C=1.0, random_state=SEED)) if small
                      else (lambda: LinearSVC(dual=False, random_state=SEED)),
                      scale=True, min_speed="balanced"),
            ModelSpec("AdaBoost", "boosting", lambda: AdaBoostClassifier(n_estimators=100, random_state=SEED),
                      min_speed="balanced"),
            ModelSpec("Neural Network (MLP)", "neural network", lambda: MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=SEED),
                      scale=True, min_speed="thorough"),
        ]
    else:
        small = n_rows <= 5000
        specs = [
            ModelSpec("Linear Regression", "linear", lambda: LinearRegression(), scale=True),
            ModelSpec("Ridge", "linear", lambda: Ridge(alpha=1.0), scale=True),
            ModelSpec("Lasso", "linear", lambda: Lasso(alpha=0.01, max_iter=5000), scale=True, min_speed="balanced"),
            ModelSpec("ElasticNet", "linear", lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
                      scale=True, min_speed="balanced"),
            ModelSpec("Random Forest", "tree ensemble", lambda: RandomForestRegressor(
                n_estimators=n_estimators, random_state=SEED, n_jobs=-1)),
            ModelSpec("Extra Trees", "tree ensemble", lambda: ExtraTreesRegressor(
                n_estimators=n_estimators, random_state=SEED, n_jobs=-1), min_speed="balanced"),
            ModelSpec("Gradient Boosting", "boosting", lambda: GradientBoostingRegressor(random_state=SEED),
                      min_speed="balanced"),
            ModelSpec("HistGradient Boosting", "boosting", lambda: HistGradientBoostingRegressor(random_state=SEED)),
            ModelSpec("Decision Tree", "tree", lambda: DecisionTreeRegressor(max_depth=8, random_state=SEED)),
            ModelSpec("k-Nearest Neighbors", "instance-based", lambda: KNeighborsRegressor(n_neighbors=15),
                      scale=True, min_speed="balanced"),
            ModelSpec("SVR (RBF)" if small else "SVR (linear)", "kernel / margin",
                      (lambda: SVR(kernel="rbf")) if small else (lambda: LinearSVR(random_state=SEED, max_iter=5000)),
                      scale=True, min_speed="balanced"),
            ModelSpec("AdaBoost", "boosting", lambda: AdaBoostRegressor(n_estimators=100, random_state=SEED),
                      min_speed="balanced"),
            ModelSpec("Neural Network (MLP)", "neural network", lambda: MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=SEED),
                      scale=True, min_speed="thorough"),
        ]
    return specs + _optional_boosters(task)


def available_model_names(task: str, speed: str = "balanced") -> List[str]:
    lvl = SPEED_LEVEL.get(speed, 1)
    return [s.name for s in supervised_specs(task, 1000) if SPEED_LEVEL[s.min_speed] <= lvl]


# ---------------------------------------------------------------------------
# Supervised comparison
# ---------------------------------------------------------------------------
@dataclass
class ModelComparison:
    task: str
    metric_name: str
    table: pd.DataFrame
    best_name: Optional[str]
    best_score: Optional[float]
    cv_folds: int
    n_rows: int
    classes: Optional[List[str]] = None
    confusion: Optional[np.ndarray] = None
    roc: Optional[Dict[str, object]] = None
    importance: Optional[pd.DataFrame] = None
    pred_vs_actual: Optional[pd.DataFrame] = None
    class_counts: Optional[Dict[str, int]] = None
    notes: List[str] = field(default_factory=list)
    best_import: str = ""        # e.g. "from sklearn.ensemble import RandomForestClassifier"
    best_repr: str = ""          # e.g. "RandomForestClassifier(n_estimators=150, random_state=42)"
    best_needs_scaling: bool = False


def _cv_splitter(task: str, y: np.ndarray, folds: int):
    if task == "classification":
        _, counts = np.unique(y, return_counts=True)
        folds = int(max(2, min(folds, counts.min())))
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED), folds
    return KFold(n_splits=folds, shuffle=True, random_state=SEED), folds


def _subsample(X: pd.DataFrame, y: np.ndarray, task: str, max_rows: int) -> Tuple[pd.DataFrame, np.ndarray]:
    if len(X) <= max_rows:
        return X, y
    if task == "classification":
        idx, _ = train_test_split(np.arange(len(X)), train_size=max_rows, stratify=y, random_state=SEED)
    else:
        idx = np.random.RandomState(SEED).choice(len(X), max_rows, replace=False)
    idx = np.sort(idx)
    return X.iloc[idx], y[idx]


def compare_supervised(X: pd.DataFrame, y_enc: np.ndarray, numeric_cols: List[str], categorical_cols: List[str],
                       task: str, cfg: PrepConfig, skewed_cols: Sequence[str] = (), speed: str = "balanced",
                       classes: Optional[List[str]] = None, n_estimators: int = 150,
                       time_budget_s: float = 240.0, progress: Optional[Callable[[float, str], None]] = None
                       ) -> ModelComparison:
    """Cross-validate every model in the zoo on the same folds and rank them."""
    warnings.filterwarnings("ignore")
    notes: List[str] = []
    metric_name = "F1 (weighted)" if task == "classification" else "R²"

    if cfg.missing == "drop":
        keep = X.notna().all(axis=1).values
        X, y_enc = X[keep], y_enc[keep]
        notes.append(f"'Drop rows' strategy selected: {int(keep.sum()):,} complete rows used.")

    if task == "classification":
        vals, counts = np.unique(y_enc, return_counts=True)
        min_needed = SPEED_CV.get(speed, 3)
        rare = vals[counts < min_needed]
        if len(rare):
            keep = ~np.isin(y_enc, rare)
            X, y_enc = X[keep], y_enc[keep]
            notes.append(f"{len(rare)} class(es) with fewer than {min_needed} rows were left out of cross-validation.")

    X, y_enc = _subsample(X, y_enc, task, MAX_MODEL_ROWS)
    n = len(X)
    if n < 30 or (task == "classification" and len(np.unique(y_enc)) < 2):
        return ModelComparison(task, metric_name, pd.DataFrame(), None, None, 0, n, classes,
                               notes=notes + ["Not enough usable rows to compare models."])
    cv, folds = _cv_splitter(task, y_enc, SPEED_CV.get(speed, 3))
    binary = task == "classification" and len(np.unique(y_enc)) == 2

    if task == "classification":
        scoring = {"f1": "f1_weighted", "accuracy": "accuracy", "balanced_acc": "balanced_accuracy",
                   "precision": "precision_weighted", "recall": "recall_weighted"}
        if binary:
            scoring["roc_auc"] = "roc_auc"
        primary = "f1"
    else:
        scoring = {"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}
        primary = "r2"

    lvl = SPEED_LEVEL.get(speed, 1)
    specs = [s for s in supervised_specs(task, n, n_estimators) if SPEED_LEVEL[s.min_speed] <= lvl]
    rows, pipelines = [], {}
    t_start = time.time()
    for i, spec in enumerate(specs):
        if progress:
            progress(i / max(len(specs), 1), f"Cross-validating {spec.name}")
        if time.time() - t_start > time_budget_s:
            rows.append({"Model": spec.name, "Family": spec.family, "error": "skipped (time budget reached)"})
            continue
        try:
            pipe = Pipeline([
                ("prep", build_preprocessor(numeric_cols, categorical_cols, cfg, skewed_cols=skewed_cols,
                                            scale=spec.scale)),
                ("model", spec.build()),
            ])
            t0 = time.time()
            res = cross_validate(pipe, X, y_enc, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")
            elapsed = time.time() - t0
            row = {"Model": spec.name, "Family": spec.family,
                   metric_name: float(np.mean(res[f"test_{primary}"])),
                   "± std": float(np.std(res[f"test_{primary}"])),
                   "seconds": round(elapsed, 1), "error": ""}
            for k in scoring:
                if k != primary:
                    v = float(np.mean(res[f"test_{k}"]))
                    row[k] = -v if k in ("rmse", "mae") else v
            rows.append(row)
            pipelines[spec.name] = pipe
        except Exception as e:
            rows.append({"Model": spec.name, "Family": spec.family, "error": str(e)[:160]})
    if progress:
        progress(1.0, "Model comparison finished")

    table = pd.DataFrame(rows)
    if metric_name not in table.columns or table[metric_name].notna().sum() == 0:
        return ModelComparison(task, metric_name, table, None, None, folds, n, classes, notes=notes + ["No model could be evaluated."])
    table = table.sort_values(metric_name, ascending=False, na_position="last").reset_index(drop=True)
    table.insert(0, "Rank", [i + 1 if pd.notna(v) else None for i, v in enumerate(table[metric_name])])
    rename = {"accuracy": "Accuracy", "balanced_acc": "Balanced accuracy", "precision": "Precision",
              "recall": "Recall", "roc_auc": "ROC-AUC", "rmse": "RMSE", "mae": "MAE"}
    table = table.rename(columns=rename)
    best_name = str(table.loc[0, "Model"])
    best_score = float(table.loc[0, metric_name])
    result = ModelComparison(task, metric_name, table, best_name, best_score, folds, n, classes, notes=notes)
    best_spec = next(sp for sp in specs if sp.name == best_name)
    est = best_spec.build()
    result.best_import = f"from {type(est).__module__.split('._')[0]} import {type(est).__name__}"
    result.best_repr = repr(est)
    result.best_needs_scaling = best_spec.scale

    # ---- diagnostics for the winning model -------------------------------------
    best_pipe = pipelines[best_name]
    try:
        if task == "classification":
            from sklearn.base import clone
            pred = cross_val_predict(clone(best_pipe), X, y_enc, cv=cv)
            result.confusion = confusion_matrix(y_enc, pred)
            result.class_counts = {str(classes[int(c)]) if classes else str(c): int((y_enc == c).sum())
                                   for c in np.unique(y_enc)}
            if binary:
                method = "predict_proba" if hasattr(best_pipe, "predict_proba") else "decision_function"
                sc = cross_val_predict(clone(best_pipe), X, y_enc, cv=cv, method=method)
                sc = sc[:, 1] if sc.ndim == 2 else sc
                fpr, tpr, _ = roc_curve(y_enc, sc)
                result.roc = {"fpr": fpr, "tpr": tpr, "auc": float(auc(fpr, tpr))}
        else:
            from sklearn.base import clone
            pred = cross_val_predict(clone(best_pipe), X, y_enc, cv=cv)
            result.pred_vs_actual = pd.DataFrame({"actual": y_enc, "predicted": pred})
    except Exception as e:
        result.notes.append(f"Diagnostics unavailable: {str(e)[:120]}")
    try:
        from sklearn.base import clone
        strat = y_enc if task == "classification" and np.bincount(y_enc.astype(int)).min() >= 2 else None
        Xtr, Xte, ytr, yte = train_test_split(X, y_enc, test_size=0.25, random_state=SEED, stratify=strat)
        if len(Xte) > 3000:
            Xte, yte = Xte.iloc[:3000], yte[:3000]
        fitted = clone(best_pipe).fit(Xtr, ytr)
        pi = permutation_importance(fitted, Xte, yte, n_repeats=5, random_state=SEED, n_jobs=1,
                                    scoring="f1_weighted" if task == "classification" else "r2")
        imp = pd.DataFrame({"feature": list(X.columns), "importance": pi.importances_mean,
                            "std": pi.importances_std}).sort_values("importance", ascending=False)
        result.importance = imp.head(15).reset_index(drop=True)
    except Exception as e:
        result.notes.append(f"Feature importance unavailable: {str(e)[:120]}")
    return result


# ---------------------------------------------------------------------------
# Unsupervised analysis
# ---------------------------------------------------------------------------
@dataclass
class UnsupervisedResult:
    n_rows: int
    n_features: int
    k_scores: pd.DataFrame
    algorithms: pd.DataFrame
    best_algorithm: Optional[str]
    best_k: Optional[int]
    pca_variance: pd.DataFrame
    projection: pd.DataFrame
    cluster_profile: pd.DataFrame
    cluster_sizes: pd.DataFrame
    anomalies: pd.DataFrame
    notes: List[str] = field(default_factory=list)


def _cluster_metrics(Z: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    mask = labels != -1
    lab = labels[mask]
    if mask.sum() < 10 or len(np.unique(lab)) < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    Zm = Z[mask]
    sil = silhouette_score(Zm, lab, sample_size=min(3000, len(Zm)), random_state=SEED)
    return {"silhouette": float(sil), "davies_bouldin": float(davies_bouldin_score(Zm, lab)),
            "calinski_harabasz": float(calinski_harabasz_score(Zm, lab))}


def analyze_unsupervised(X: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str],
                         speed: str = "balanced", max_rows: int = 4000,
                         progress: Optional[Callable[[float, str], None]] = None) -> UnsupervisedResult:
    warnings.filterwarnings("ignore")
    notes: List[str] = []
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=SEED)
        notes.append(f"Analysis ran on a {max_rows:,}-row random sample.")
    X = X.reset_index(drop=True)
    pre = build_preprocessor(numeric_cols, categorical_cols, PrepConfig(missing="median"), scale=True)
    Z = np.asarray(pre.fit_transform(X), dtype=float)
    n, d = Z.shape
    if n < 30 or d < 1:
        return UnsupervisedResult(n, d, pd.DataFrame(), pd.DataFrame(), None, None, pd.DataFrame(),
                                  pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                                  notes + ["Not enough data for unsupervised analysis."])

    def tick(f, m):
        if progress:
            progress(f, m)

    # ---- PCA ---------------------------------------------------------------------
    tick(0.05, "PCA")
    n_comp = int(max(2, min(10, d, n - 1)))
    pca = PCA(n_components=n_comp, random_state=SEED).fit(Z)
    var = pd.DataFrame({"component": [f"PC{i + 1}" for i in range(n_comp)],
                        "explained": pca.explained_variance_ratio_,
                        "cumulative": np.cumsum(pca.explained_variance_ratio_)})
    Z2 = pca.transform(Z)[:, :2] if n_comp >= 2 else np.column_stack([pca.transform(Z)[:, 0], np.zeros(n)])

    # ---- choose k ------------------------------------------------------------------
    n_init = 10 if speed == "thorough" else 5
    k_max = int(min(8, max(3, n // 40)))
    krows = []
    for k in range(2, k_max + 1):
        tick(0.10 + 0.25 * (k - 2) / max(k_max - 1, 1), f"KMeans k={k}")
        km = KMeans(n_clusters=k, n_init=n_init, random_state=SEED).fit(Z)
        m = _cluster_metrics(Z, km.labels_)
        krows.append({"k": k, "silhouette": m["silhouette"], "davies_bouldin": m["davies_bouldin"],
                      "calinski_harabasz": m["calinski_harabasz"], "inertia": float(km.inertia_)})
    k_scores = pd.DataFrame(krows)
    best_k = int(k_scores.loc[k_scores["silhouette"].idxmax(), "k"]) if k_scores["silhouette"].notna().any() else 3

    # ---- compare algorithms at best_k ---------------------------------------------------
    n_agg = min(n, 3000)
    agg_idx = np.arange(n) if n <= n_agg else np.random.RandomState(SEED).choice(n, n_agg, replace=False)
    min_samples = int(min(20, max(5, 2 * d)))
    try:
        nn = NearestNeighbors(n_neighbors=min_samples).fit(Z)
        eps = float(np.percentile(nn.kneighbors(Z)[0][:, -1], 90))
    except Exception:
        eps = 1.0

    algos: List[Tuple[str, Callable[[], np.ndarray]]] = [
        ("KMeans", lambda: KMeans(n_clusters=best_k, n_init=n_init, random_state=SEED).fit_predict(Z)),
        ("MiniBatch KMeans", lambda: MiniBatchKMeans(n_clusters=best_k, n_init=n_init, random_state=SEED).fit_predict(Z)),
        ("Gaussian Mixture", lambda: GaussianMixture(n_components=best_k, covariance_type="diag" if d > 20 else "full",
                                                     reg_covar=1e-4, random_state=SEED).fit_predict(Z)),
        ("BIRCH", lambda: Birch(n_clusters=best_k).fit_predict(Z)),
        ("DBSCAN", lambda: DBSCAN(eps=max(eps, 1e-3), min_samples=min_samples).fit_predict(Z)),
    ]
    all_labels: Dict[str, np.ndarray] = {}
    arows = []
    for i, (name, fn) in enumerate(algos):
        tick(0.40 + 0.25 * i / len(algos), name)
        try:
            lab = np.asarray(fn())
            all_labels[name] = lab
            m = _cluster_metrics(Z, lab)
            arows.append({"Algorithm": name, "clusters": int(len(set(lab)) - (1 if -1 in lab else 0)),
                          "noise %": round(100 * float((lab == -1).mean()), 1), **m})
        except Exception as e:
            arows.append({"Algorithm": name, "clusters": None, "noise %": None, "silhouette": np.nan,
                          "davies_bouldin": np.nan, "calinski_harabasz": np.nan, "error": str(e)[:100]})
    try:
        tick(0.66, "Agglomerative (Ward)")
        lab_sub = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(Z[agg_idx])
        lab = np.full(n, -1)
        lab[agg_idx] = lab_sub
        m = _cluster_metrics(Z[agg_idx], lab_sub)
        arows.append({"Algorithm": "Agglomerative (Ward)", "clusters": best_k, "noise %": 0.0, **m})
        if n <= n_agg:
            all_labels["Agglomerative (Ward)"] = lab_sub
    except Exception as e:
        arows.append({"Algorithm": "Agglomerative (Ward)", "clusters": None, "noise %": None,
                      "silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan,
                      "error": str(e)[:100]})
    algorithms = pd.DataFrame(arows).sort_values("silhouette", ascending=False, na_position="last").reset_index(drop=True)
    # DBSCAN's silhouette ignores noise points, which flatters it - only allow it when little is discarded
    valid = algorithms[algorithms["silhouette"].notna() & algorithms["Algorithm"].isin(all_labels)
                       & (algorithms["noise %"].fillna(0) <= 15)]
    best_algo = str(valid.iloc[0]["Algorithm"]) if len(valid) else "KMeans"
    best_labels = all_labels.get(best_algo, all_labels.get("KMeans", np.zeros(n, dtype=int)))

    # ---- anomaly detection ------------------------------------------------------------------
    tick(0.72, "Anomaly detection")
    flags: Dict[str, np.ndarray] = {}
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=SEED).fit(Z)
    flags["Isolation Forest"] = iso.predict(Z) == -1
    iso_score = -iso.score_samples(Z)
    try:
        flags["Local Outlier Factor"] = LocalOutlierFactor(n_neighbors=20, contamination=0.05).fit_predict(Z) == -1
    except Exception:
        pass
    try:
        sub = np.random.RandomState(SEED).choice(n, min(n, 3000), replace=False)
        flags["One-Class SVM"] = OneClassSVM(nu=0.05, gamma="scale").fit(Z[sub]).predict(Z) == -1
    except Exception:
        pass
    votes = np.sum([f.astype(int) for f in flags.values()], axis=0)
    consensus = votes >= 2
    arows2 = [{"Detector": k, "flagged rows": int(v.sum()), "flagged %": round(100 * float(v.mean()), 2)}
              for k, v in flags.items()]
    arows2.append({"Detector": "Consensus (≥2 detectors)", "flagged rows": int(consensus.sum()),
                   "flagged %": round(100 * float(consensus.mean()), 2)})
    anomalies = pd.DataFrame(arows2)

    # ---- profiles & projection ----------------------------------------------------------------
    tick(0.9, "Cluster profiles")
    labels_str = np.where(best_labels == -1, "noise", ["Cluster " + str(int(v) + 1) for v in best_labels])
    projection = pd.DataFrame({"PC1": Z2[:, 0], "PC2": Z2[:, 1], "cluster": labels_str,
                               "anomaly": np.where(consensus, "anomaly", "normal"),
                               "anomaly score": iso_score})
    num_only = X[numeric_cols].astype(float) if numeric_cols else pd.DataFrame(index=X.index)
    profile = pd.DataFrame()
    if len(num_only.columns):
        z = (num_only - num_only.mean()) / num_only.std().replace(0, np.nan)
        z["cluster"] = labels_str
        profile = z.groupby("cluster").mean(numeric_only=True).fillna(0.0)
    sizes = pd.Series(labels_str).value_counts().rename_axis("cluster").reset_index(name="rows")
    sizes["share %"] = (100 * sizes["rows"] / n).round(1)
    tick(1.0, "Done")
    return UnsupervisedResult(n, d, k_scores, algorithms, best_algo, best_k, var, projection, profile,
                              sizes, anomalies, notes)
