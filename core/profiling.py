"""
profiling.py
------------
Data-quality profile of a pandas DataFrame: type, missing %, cardinality,
IQR outliers, skewness and basic statistics per column.

Profiling now runs on the *cleaned* sample (so numbers stored as text are
recognised as numeric, hidden "NA" tokens count as missing, and duplicates are
found after case/spacing differences are normalised). The pipeline also calls
it on the raw sample so the report can show a before / after comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    kind: str
    missing_count: int
    missing_pct: float
    n_unique: Optional[int]
    unique_ratio: Optional[float]
    is_numeric: bool
    outlier_count: Optional[int] = None
    outlier_pct: Optional[float] = None
    skewness: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean_val: Optional[float] = None
    std_val: Optional[float] = None
    top_values: Optional[List] = None


@dataclass
class DataQualityReport:
    n_rows: int
    n_cols: int
    used_sampling: bool
    sample_size: Optional[int]
    duplicate_rows: int
    columns: Dict[str, ColumnProfile] = field(default_factory=dict)

    def to_summary_frame(self) -> pd.DataFrame:
        rows = []
        for c in self.columns.values():
            rows.append({
                "column": c.name,
                "type": c.kind,
                "dtype": c.dtype,
                "missing %": round(c.missing_pct, 2),
                "unique values": c.n_unique,
                "unique ratio": round(c.unique_ratio, 3) if c.unique_ratio is not None else None,
                "outlier %": round(c.outlier_pct, 2) if c.outlier_pct is not None else None,
                "skewness": round(c.skewness, 3) if c.skewness is not None else None,
                "min": c.min_val,
                "max": c.max_val,
                "mean": round(c.mean_val, 4) if c.mean_val is not None else None,
            })
        return pd.DataFrame(rows)

    @property
    def total_missing_cells(self) -> int:
        return int(sum(c.missing_count for c in self.columns.values()))


def _iqr_outlier_pct(series: pd.Series) -> Optional[float]:
    s = series.dropna()
    if len(s) < 8:
        return None
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return 100.0 * float(((s < lower) | (s > upper)).sum()) / len(s)


def count_duplicates(df: pd.DataFrame, ignore_cols: Optional[Iterable[str]] = None) -> int:
    """Rows that repeat an earlier row (ignoring row-counter / index-artifact columns)."""
    skip = set(ignore_cols or [])
    cols = [c for c in df.columns if c not in skip]
    if not cols:
        return 0
    try:
        return int(df.duplicated(subset=cols).sum())
    except Exception:
        return 0


def profile_dataframe(df: pd.DataFrame, kinds: Optional[Dict[str, str]] = None,
                      n_rows_total: Optional[int] = None, sampled: bool = False,
                      duplicate_rows: Optional[int] = None,
                      ignore_for_duplicates: Optional[Iterable[str]] = None) -> DataQualityReport:
    kinds = kinds or {}
    n = len(df)
    dup = duplicate_rows if duplicate_rows is not None else count_duplicates(df, ignore_for_duplicates)
    columns: Dict[str, ColumnProfile] = {}
    for col in df.columns:
        s = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        miss = int(s.isna().sum())
        try:
            n_unique = int(s.nunique(dropna=True))
        except Exception:
            n_unique = None
        ratio = (n_unique / max(n - miss, 1)) if n_unique is not None and n else None
        outlier_pct = skew = None
        min_v = max_v = mean_v = std_v = None
        top = None
        if is_numeric and n:
            x = pd.to_numeric(s, errors="coerce").astype(float)
            x = x.where(np.isfinite(x))
            outlier_pct = _iqr_outlier_pct(x)
            clean = x.dropna()
            if len(clean) >= 3 and clean.std() > 0:
                skew = float(scipy_stats.skew(clean))
            if len(clean):
                min_v, max_v = float(clean.min()), float(clean.max())
                mean_v, std_v = float(clean.mean()), float(clean.std())
        elif n:
            try:
                vc = s.value_counts(dropna=True).head(5)
                top = [(str(k), int(v)) for k, v in vc.items()]
            except Exception:
                top = None
        kind = kinds.get(col) or ("numeric" if is_numeric else "text")
        columns[col] = ColumnProfile(
            name=col, dtype=str(s.dtype), kind=kind, missing_count=miss,
            missing_pct=(100.0 * miss / n) if n else 0.0, n_unique=n_unique, unique_ratio=ratio,
            is_numeric=is_numeric,
            outlier_count=int(round((outlier_pct or 0) / 100 * (n - miss))) if outlier_pct is not None else None,
            outlier_pct=outlier_pct, skewness=skew, min_val=min_v, max_val=max_v,
            mean_val=mean_v, std_val=std_v, top_values=top,
        )
    return DataQualityReport(
        n_rows=n_rows_total if n_rows_total is not None else n,
        n_cols=df.shape[1], used_sampling=sampled, sample_size=n if sampled else None,
        duplicate_rows=dup, columns=columns,
    )
