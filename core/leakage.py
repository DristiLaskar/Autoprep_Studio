"""
leakage.py
----------
Flags columns that are suspicious for data leakage or unlikely to be useful
as predictive features. The pipeline runs this automatically on the CLEANED
sample and excludes anything at or above a risk threshold.

Signals
  1. ID-like columns: near-unique integer/text columns or ID-style names.
  2. Near-perfect correlation with the target (|r| >= 0.99 auto-excluded, >= 0.95 flagged for review).
  3. Near-deterministic relationship with the target, measured with mutual
     information normalised by the target's entropy (classification), e.g. a
     column that is filled in only for customers who already churned.
  4. Column name contains the target's name plus a "post-outcome" word.
  5. Constant / near-constant columns.

Mutual information "dominance" is deliberately judged on an absolute scale
(share of the target's entropy explained) instead of "is this the best
feature?", so the app never throws away the genuinely strongest legitimate
predictor just because it is the strongest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

ID_NAME_PATTERN = re.compile(r"(^|_)(id|uuid|guid|index|idx|key|row[_ ]?num)($|_)", re.I)
LEAKY_NAME_PATTERN = re.compile(r"(outcome|result|label|after|post_|future|final_|target_leak|leak)", re.I)

CORR_HIGH_THRESHOLD = 0.95    # worth a look (shown, but not auto-excluded)
CORR_LEAK_THRESHOLD = 0.99    # essentially a copy of the target (auto-excluded)
UNIQUE_RATIO_ID_THRESHOLD = 0.98
NMI_DETERMINISTIC = 0.60      # single feature explains >= 60% of the target's entropy
NMI_DOMINANT = 0.30           # ... and is >= 3x stronger than the next-best feature
MAX_LEAKAGE_ROWS = 30_000


@dataclass
class LeakageFlag:
    column: str
    risk_score: float  # 0-1
    reasons: List[str]


def _is_task_classification(y: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return True
    n_unique = y.nunique(dropna=True)
    return n_unique <= max(20, int(0.05 * len(y)))


def _encode_categorical(col: pd.Series) -> np.ndarray:
    filled = col.astype(object).where(col.notna(), "__NA__").astype(str)
    return LabelEncoder().fit_transform(filled)


def detect_leakage(pdf: pd.DataFrame, target_col: str, task_type: Optional[str] = None,
                   random_state: int = 42) -> List[LeakageFlag]:
    """Scan all non-target columns for leakage / uselessness signals."""
    if target_col not in pdf.columns:
        return []
    pdf = pdf.dropna(subset=[target_col])
    if len(pdf) > MAX_LEAKAGE_ROWS:
        pdf = pdf.sample(MAX_LEAKAGE_ROWS, random_state=random_state)
    pdf = pdf.reset_index(drop=True)
    if len(pdf) < 20:
        return []

    y_raw = pdf[target_col]
    is_clf = (task_type == "classification") if task_type else _is_task_classification(y_raw)
    if is_clf:
        y = LabelEncoder().fit_transform(y_raw.astype(str))
        y_series = pd.Series(y, dtype=float)
    else:
        y_series = pd.to_numeric(y_raw, errors="coerce").astype(float)
        y = y_series.values

    feature_cols = [c for c in pdf.columns if c != target_col]
    flags: List[LeakageFlag] = []
    if not feature_cols:
        return flags

    # ---- numeric-encoded matrix for mutual information ---------------------
    mi_cols, discrete = {}, []
    for c in feature_cols:
        col = pdf[c]
        if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
            x = pd.to_numeric(col, errors="coerce").astype(float)
            med = x.median()
            mi_cols[c] = x.fillna(0.0 if pd.isna(med) else med).values
            discrete.append(False)
        else:
            try:
                mi_cols[c] = _encode_categorical(col)
            except Exception:
                mi_cols[c] = np.zeros(len(pdf))
            discrete.append(True)
    mi_scores = {}
    try:
        X = np.column_stack([mi_cols[c] for c in feature_cols]).astype(float)
        valid = ~np.isnan(y_series.values)
        if valid.sum() > 20 and X.shape[1] > 0:
            if is_clf:
                mi = mutual_info_classif(X[valid], y[valid], discrete_features=np.array(discrete),
                                         random_state=random_state)
            else:
                mi = mutual_info_regression(X[valid], y_series.values[valid],
                                            discrete_features=np.array(discrete), random_state=random_state)
            mi_scores = dict(zip(feature_cols, mi))
    except Exception:
        mi_scores = {}

    h_y = 0.0
    if is_clf:
        counts = np.bincount(y.astype(int))
        h_y = float(entropy(counts[counts > 0]))
    sorted_mi = sorted(mi_scores.values(), reverse=True)
    second_best = sorted_mi[1] if len(sorted_mi) > 1 else 0.0

    n = len(pdf)
    for c in feature_cols:
        reasons: List[str] = []
        risk = 0.0
        col = pdf[c]
        n_unique = col.nunique(dropna=True)
        unique_ratio = n_unique / n if n else 0

        # 1. ID-like ---------------------------------------------------------
        is_num = pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col)
        if is_num:
            nz = col.dropna()
            is_int_like = bool(len(nz)) and bool(np.all(np.isclose(nz.values, np.round(nz.values))))
        else:
            is_int_like = True  # text columns are always "discrete"
        if is_int_like and unique_ratio >= UNIQUE_RATIO_ID_THRESHOLD:
            reasons.append(f"Near-unique per row (unique ratio {unique_ratio:.2f}) — likely an identifier.")
            risk = max(risk, 0.7)
        if ID_NAME_PATTERN.search(c) and unique_ratio >= 0.5:
            reasons.append("Column name matches common ID/index naming patterns.")
            risk = max(risk, 0.6)

        # 2. correlation with the target -----------------------------------------
        if is_num:
            try:
                corr = pd.to_numeric(col, errors="coerce").corr(pd.Series(y_series.values, index=col.index))
                if corr is not None and not np.isnan(corr) and abs(corr) >= CORR_LEAK_THRESHOLD:
                    reasons.append(f"Essentially a copy of the target (r = {corr:.3f}).")
                    risk = max(risk, 0.9)
                elif corr is not None and not np.isnan(corr) and abs(corr) >= CORR_HIGH_THRESHOLD:
                    reasons.append(f"Very high correlation with target (r = {corr:.3f}) — review, but it may be a "
                                   "genuine strong predictor.")
                    risk = max(risk, 0.5)
            except Exception:
                pass

        # 3. near-deterministic mutual information ------------------------------------
        if is_clf and c in mi_scores and h_y > 0:
            mi = mi_scores[c]
            nmi = mi / h_y
            extra = " Name also suggests post-outcome data." if LEAKY_NAME_PATTERN.search(c) else ""
            if nmi >= NMI_DETERMINISTIC and mi > 0.02:
                reasons.append(f"Near-deterministic relationship with target — explains {min(nmi, 1):.0%} "
                               f"of its entropy (mutual information).{extra}")
                risk = max(risk, 0.75 + 0.2 * min(1.0, (nmi - NMI_DETERMINISTIC) / 0.4))
            elif nmi >= NMI_DOMINANT and mi > 0.02 and (second_best <= 0 or mi >= 3 * second_best):
                reasons.append(f"Dominant mutual information with target (explains {nmi:.0%} of its entropy, "
                               f"3x+ the next feature).{extra}")
                risk = max(risk, 0.6)
        if LEAKY_NAME_PATTERN.search(c) and target_col.lower() in c.lower():
            reasons.append("Column name contains the target column's name.")
            risk = max(risk, 0.8)

        # 4. constant -----------------------------------------------------------------
        if n_unique <= 1:
            reasons.append("Column is constant or near-constant — provides no signal.")
            risk = max(risk, 0.3)

        if reasons:
            flags.append(LeakageFlag(column=c, risk_score=round(min(risk, 1.0), 2), reasons=reasons))

    flags.sort(key=lambda f: -f.risk_score)
    return flags


def flags_to_frame(flags: List[LeakageFlag]) -> pd.DataFrame:
    return pd.DataFrame([
        {"column": f.column, "risk score": f.risk_score, "reasons": " | ".join(f.reasons)}
        for f in flags
    ])
