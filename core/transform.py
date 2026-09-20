"""
transform.py
------------
Preprocessing building blocks and the automatic preprocessing ablation study.

The ablation study tries a small, sensible set of preprocessing choices and
keeps whichever one *measurably* helps, judged by cross-validated score of a
Random Forest probe on the SAME folds:

    axis                 options tried (one change at a time from the baseline)
    ------------------   ----------------------------------------------------
    missing values       median (baseline) | mean | drop rows
    outliers             none (baseline)   | clip to 1.5 x IQR fences
    skewed numerics      none (baseline)   | log1p on skewed, non-negative columns
    categorical encoding one-hot (baseline)| frequency encoding

After the single-axis runs the winners are combined and re-tested, and a change
is only adopted when it beats the baseline by more than the noise floor.
All estimators are module-level classes so they can be pickled / cloned.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder, StandardScaler

RANDOM_SEED = 42
NOISE_FLOOR = 0.002          # a candidate must beat the baseline by more than this to be adopted


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PrepConfig:
    missing: str = "median"     # mean | median | drop
    outliers: str = "none"      # none | clip
    skew: str = "none"          # none | log
    encoding: str = "onehot"    # onehot | frequency

    def describe(self) -> str:
        parts = [f"{self.missing} imputation" if self.missing != "drop" else "drop rows with missing values"]
        parts.append("IQR outlier clipping" if self.outliers == "clip" else "no outlier clipping")
        parts.append("log1p on skewed columns" if self.skew == "log" else "no log transform")
        parts.append("one-hot encoding" if self.encoding == "onehot" else "frequency encoding")
        return " / ".join(parts)


# ---------------------------------------------------------------------------
# Transformers (module-level so they pickle/clone cleanly)
# ---------------------------------------------------------------------------
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category with its relative frequency in the training data."""

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        self.maps_ = [df[c].astype(str).value_counts(normalize=True).to_dict() for c in df.columns]
        self.n_features_in_ = df.shape[1]
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        out = np.zeros(df.shape, dtype=float)
        for i, c in enumerate(df.columns):
            out[:, i] = df[c].astype(str).map(self.maps_[i]).fillna(0.0).values
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to [Q1 - k*IQR, Q3 + k*IQR] learned on the training data."""

    def __init__(self, k: float = 1.5):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        lo, hi = q1 - self.k * iqr, q3 + self.k * iqr
        flat = iqr == 0
        lo[flat], hi[flat] = -np.inf, np.inf
        self.lo_, self.hi_ = lo, hi
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def _log1p_nonneg(X):
    return np.log1p(np.clip(np.asarray(X, dtype=float), 0, None))


# ---------------------------------------------------------------------------
# Frame preparation / preprocessor factory
# ---------------------------------------------------------------------------
def prepare_frame(df: pd.DataFrame, numeric_cols: Sequence[str], categorical_cols: Sequence[str]) -> pd.DataFrame:
    """Feature frame with clean numeric floats and object categoricals (np.nan for missing)."""
    X = pd.DataFrame(index=df.index)
    for c in numeric_cols:
        X[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in categorical_cols:
        s = df[c]
        X[c] = s.astype(object).where(s.notna(), np.nan)
    return X


def build_preprocessor(numeric_cols: Sequence[str], categorical_cols: Sequence[str], cfg: PrepConfig,
                       skewed_cols: Sequence[str] = (), scale: bool = False,
                       min_frequency: float = 0.01) -> ColumnTransformer:
    skewed = [c for c in numeric_cols if c in set(skewed_cols)] if cfg.skew == "log" else []
    plain = [c for c in numeric_cols if c not in set(skewed)]

    def num_steps(log: bool):
        steps = []
        if cfg.missing in ("mean", "median"):
            steps.append(("impute", SimpleImputer(strategy=cfg.missing, keep_empty_features=True)))
        if cfg.outliers == "clip":
            steps.append(("clip", Winsorizer()))
        if log:
            steps.append(("log", FunctionTransformer(_log1p_nonneg, feature_names_out="one-to-one")))
        if scale:
            steps.append(("scale", StandardScaler()))
        return Pipeline(steps) if steps else "passthrough"

    transformers = []
    if plain:
        transformers.append(("num", num_steps(False), list(plain)))
    if skewed:
        transformers.append(("num_log", num_steps(True), list(skewed)))
    if categorical_cols:
        steps = []
        if cfg.missing != "drop":
            steps.append(("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)))
        if cfg.encoding == "frequency":
            steps.append(("enc", FrequencyEncoder()))
        else:
            steps.append(("enc", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                               min_frequency=min_frequency, sparse_output=False)))
        transformers.append(("cat", Pipeline(steps), list(categorical_cols)))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def detect_skewed_numeric(df: pd.DataFrame, numeric_cols: Sequence[str], threshold: float = 1.0) -> List[str]:
    skewed = []
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) >= 20 and s.nunique() > 10 and s.std() > 0 and s.min() >= 0:
            try:
                if abs(s.skew()) >= threshold:
                    skewed.append(c)
            except Exception:
                pass
    return skewed


# ---------------------------------------------------------------------------
# Task / target helpers
# ---------------------------------------------------------------------------
def infer_task_type(y: pd.Series) -> str:
    y = y.dropna()
    if len(y) == 0:
        return "classification"
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification"
    n_unique = y.nunique()
    if n_unique <= 2:
        return "classification"
    integer_like = bool(np.all(np.isclose(y.values.astype(float), np.round(y.values.astype(float)))))
    if integer_like and n_unique <= max(15, int(0.02 * len(y))):
        return "classification"
    return "regression"


def encode_target(y: pd.Series, task: str) -> Tuple[np.ndarray, Optional[List[str]]]:
    if task == "classification":
        # keep integer labels (0/1) readable as "0"/"1"; text labels stay text
        if pd.api.types.is_numeric_dtype(y):
            labels = y.astype(float)
            as_str = labels.map(lambda v: str(int(v)) if float(v).is_integer() else str(v))
        else:
            as_str = y.astype(str)
        le = LabelEncoder()
        enc = le.fit_transform(as_str)
        return enc, [str(c) for c in le.classes_]
    return pd.to_numeric(y, errors="coerce").astype(float).values, None


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------
@dataclass
class CandidateResult:
    name: str
    cfg: PrepConfig
    score: float
    metric_name: str
    secondary: Dict[str, float] = field(default_factory=dict)
    delta_vs_baseline_pct: Optional[float] = None
    n_rows: int = 0
    eligible: bool = True
    error: Optional[str] = None
    note: str = ""

    # kept for backwards compatibility with the old UI / code generator
    @property
    def missing_strategy(self) -> str:
        return self.cfg.missing

    @property
    def skew_strategy(self) -> str:
        return self.cfg.skew

    @property
    def encoding_strategy(self) -> str:
        return self.cfg.encoding


@dataclass
class AblationOutcome:
    task_type: str
    metric_name: str
    baseline: CandidateResult
    candidates: List[CandidateResult] = field(default_factory=list)
    best: Optional[CandidateResult] = None
    skewed_numeric_cols: List[str] = field(default_factory=list)
    explanation: str = ""
    probe_model: str = "Random Forest"
    n_rows: int = 0
    cv_folds: int = 3


def _cv(task: str, y: np.ndarray, folds: int):
    if task == "classification":
        _, counts = np.unique(y, return_counts=True)
        folds = int(max(2, min(folds, counts.min())))
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
    return KFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)


def _probe_model(task: str, n_estimators: int):
    if task == "classification":
        return RandomForestClassifier(n_estimators=n_estimators, random_state=RANDOM_SEED, n_jobs=-1)
    return RandomForestRegressor(n_estimators=n_estimators, random_state=RANDOM_SEED, n_jobs=-1)


def run_ablation(X: pd.DataFrame, y_enc: np.ndarray, numeric_cols: List[str], categorical_cols: List[str],
                 task: str, n_estimators: int = 100, cv_folds: int = 3,
                 progress=None) -> AblationOutcome:
    """X must come from prepare_frame(); y_enc from encode_target()."""
    metric_name = "F1 (weighted)" if task == "classification" else "R²"
    skewed = detect_skewed_numeric(X, numeric_cols)
    scoring = {"f1": "f1_weighted", "acc": "accuracy"} if task == "classification" else {"r2": "r2"}
    primary = "f1" if task == "classification" else "r2"

    def evaluate(cfg: PrepConfig, label: str) -> CandidateResult:
        try:
            Xc, yc = X, y_enc
            note = ""
            eligible = True
            if cfg.missing == "drop":
                keep = X.notna().all(axis=1).values
                Xc, yc = X[keep], y_enc[keep]
                frac = keep.mean() if len(keep) else 0
                # scored on a different (smaller) set of rows, so the comparison is not apples-to-apples:
                # shown for reference, never auto-selected
                eligible = False
                note = f"reference only — keeps {frac:.0%} of rows (different row set, never auto-selected)"
            if len(Xc) < 40 or (task == "classification" and len(np.unique(yc)) < 2):
                return CandidateResult(label, cfg, float("nan"), metric_name, error="Not enough rows after handling missing values.")
            pipe = Pipeline([
                ("prep", build_preprocessor(numeric_cols, categorical_cols, cfg, skewed_cols=skewed)),
                ("model", _probe_model(task, n_estimators)),
            ])
            res = cross_validate(pipe, Xc, yc, cv=_cv(task, yc, cv_folds), scoring=scoring, n_jobs=1)
            sec = {k: float(np.mean(res[f"test_{k}"])) for k in scoring}
            return CandidateResult(label, cfg, sec[primary], metric_name,
                                   secondary={("accuracy" if k == "acc" else k): v for k, v in sec.items() if k != primary},
                                   n_rows=len(Xc), eligible=eligible, note=note)
        except Exception as e:  # keep the study going; the failure is shown in the table
            return CandidateResult(label, cfg, float("nan"), metric_name, error=str(e)[:200])

    def tick(msg, frac):
        if progress:
            progress(frac, msg)

    base_cfg = PrepConfig()
    tick("Ablation: baseline", 0.0)
    baseline = evaluate(base_cfg, "Baseline (median / no-clip / no-log / one-hot)")

    variants: List[Tuple[str, PrepConfig]] = [("Mean imputation", replace(base_cfg, missing="mean"))]
    if X[numeric_cols + categorical_cols].isna().any().any():
        variants.append(("Drop rows with missing values", replace(base_cfg, missing="drop")))
    if numeric_cols:
        variants.append(("Clip outliers (1.5×IQR)", replace(base_cfg, outliers="clip")))
    if skewed:
        variants.append(("Log transform (skewed numeric)", replace(base_cfg, skew="log")))
    if categorical_cols:
        variants.append(("Frequency encoding", replace(base_cfg, encoding="frequency")))

    candidates: List[CandidateResult] = []
    for i, (label, cfg) in enumerate(variants):
        tick(f"Ablation: {label}", (i + 1) / (len(variants) + 2))
        candidates.append(evaluate(cfg, label))

    def better(c: CandidateResult) -> bool:
        return (c.eligible and c.error is None and not np.isnan(c.score) and not np.isnan(baseline.score)
                and c.score - baseline.score > NOISE_FLOOR)

    # combine per-axis winners
    winners = [c for c in candidates if better(c)]
    if len(winners) >= 2:
        merged = base_cfg
        for c in winners:
            if c.cfg.missing != base_cfg.missing:
                merged = replace(merged, missing=c.cfg.missing)
            if c.cfg.outliers != base_cfg.outliers:
                merged = replace(merged, outliers=c.cfg.outliers)
            if c.cfg.skew != base_cfg.skew:
                merged = replace(merged, skew=c.cfg.skew)
            if c.cfg.encoding != base_cfg.encoding:
                merged = replace(merged, encoding=c.cfg.encoding)
        tick("Ablation: combined winners", (len(variants) + 1) / (len(variants) + 2))
        candidates.append(evaluate(merged, "Combined winners"))

    def pct_delta(c: CandidateResult) -> Optional[float]:
        if np.isnan(c.score) or np.isnan(baseline.score) or baseline.score == 0:
            return None
        return 100.0 * (c.score - baseline.score) / abs(baseline.score)

    baseline.delta_vs_baseline_pct = 0.0
    for c in candidates:
        c.delta_vs_baseline_pct = pct_delta(c)

    valid = [c for c in candidates if c.eligible and c.error is None and not np.isnan(c.score)]
    best = baseline
    if valid:
        top = max(valid, key=lambda c: c.score)
        if not np.isnan(baseline.score) and top.score - baseline.score > NOISE_FLOOR:
            best = top
        elif np.isnan(baseline.score):
            best = top

    lines = [f"Task detected: {task} (metric: {metric_name}, {cv_folds}-fold cross-validation, Random Forest probe)."]
    lines.append(f"Baseline scored {baseline.score:.4f}." if not np.isnan(baseline.score) else "Baseline could not be evaluated.")
    lines.append(f"Skewed non-negative numeric columns: {', '.join(skewed)}." if skewed
                 else "No strongly skewed numeric columns detected.")
    if best is not baseline:
        lines.append(f"Selected '{best.name}' — {best.cfg.describe()} — improved {metric_name} by "
                     f"{best.delta_vs_baseline_pct:+.1f}% over the baseline.")
    else:
        lines.append("No candidate beat the baseline by more than the noise floor "
                     f"({NOISE_FLOOR:.3f}), so the simple baseline is kept.")
    return AblationOutcome(
        task_type=task, metric_name=metric_name, baseline=baseline, candidates=candidates, best=best,
        skewed_numeric_cols=skewed, explanation=" ".join(lines), n_rows=len(X), cv_folds=cv_folds,
    )


def results_to_frame(outcome: AblationOutcome) -> pd.DataFrame:
    rows = []
    for r in [outcome.baseline] + outcome.candidates:
        rows.append({
            "Candidate": r.name,
            outcome.metric_name: round(r.score, 4) if not np.isnan(r.score) else np.nan,
            "Δ vs baseline": f"{r.delta_vs_baseline_pct:+.1f}%" if r.delta_vs_baseline_pct is not None else "—",
            "rows used": r.n_rows,
            "selected": "✅" if (outcome.best is not None and r is outcome.best) else "",
            "note": r.error or r.note,
        })
    return pd.DataFrame(rows)
