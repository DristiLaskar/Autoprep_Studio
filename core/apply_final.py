"""
apply_final.py
--------------
Turns the fitted pipeline into the final deliverables and applies it to the COMPLETE dataset:

  * cleaned_dataset.csv     – tidy, human-readable, imputed (the file you download)
  * ml_ready_dataset.csv    – the same rows, fully numeric (encoded + scaled) for modelling
  * autoprep_pipeline.joblib + preprocessing_pipeline.py – re-apply everything to new data
  * cleaning_report.md      – what was changed and why

The full file is streamed through in chunks, so memory stays bounded for big inputs. Duplicate
detection keeps a running set of row hashes so duplicates are found across chunk boundaries too.
(This replaces the old export loop, which lost its results when the UI re-ran.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .cleaning import DataCleaner
from .transform import PrepConfig, build_preprocessor, prepare_frame

CHUNK_ROWS = 100_000


# ---------------------------------------------------------------------------
# Row-level de-duplication that works across chunks
# ---------------------------------------------------------------------------
class Deduper:
    def __init__(self, key_col: Optional[str] = None):
        self.key_col = key_col
        self.seen_hashes: set = set()
        self.seen_keys: set = set()
        self.removed_exact = 0
        self.removed_key = 0

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        hashes = pd.util.hash_pandas_object(df, index=False).values
        hs = pd.Series(hashes, index=df.index)
        dup_exact = hs.duplicated() | hs.isin(self.seen_hashes)
        keep = ~dup_exact
        self.removed_exact += int(dup_exact.sum())
        if self.key_col and self.key_col in df.columns:
            k = df[self.key_col]
            has_key = k.notna()
            dup_key = has_key & (k.duplicated() | k.isin(self.seen_keys))
            dup_key = dup_key & keep          # count each removed row once
            self.removed_key += int(dup_key.sum())
            keep = keep & ~dup_key
            self.seen_keys.update(k[keep & has_key].tolist())
        self.seen_hashes.update(hashes[keep.values].tolist())
        return df[keep]


# ---------------------------------------------------------------------------
# Export plan (everything needed to reproduce the result on any raw chunk)
# ---------------------------------------------------------------------------
@dataclass
class ExportPlan:
    cleaner: DataCleaner
    target_clean: Optional[str]
    task: str
    key_col: Optional[str]
    numeric_cols: List[str]
    categorical_cols: List[str]
    extra_cols: List[str]                 # identifier / contact / raw-date columns kept on request
    prep: PrepConfig
    skewed_cols: List[str] = field(default_factory=list)
    fill_num: Dict[str, float] = field(default_factory=dict)
    fill_cat: Dict[str, str] = field(default_factory=dict)
    clip_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    integer_cols: List[str] = field(default_factory=list)
    ml_pre: Optional[object] = None
    classes: Optional[List[str]] = None
    best_model_import: str = ""
    best_model_repr: str = ""
    best_model_scaled: bool = False
    metric_name: str = ""
    best_score: Optional[float] = None

    @property
    def feature_cols(self) -> List[str]:
        return self.numeric_cols + self.categorical_cols

    # -- learn imputation values / clip bounds from the cleaned sample ----------
    def learn(self, sample: pd.DataFrame) -> None:
        feats = self.numeric_cols
        for c in feats:
            s = pd.to_numeric(sample[c], errors="coerce")
            v = s.dropna()
            if v.empty:
                self.fill_num[c] = 0.0
                continue
            fill = float(v.mean() if self.prep.missing == "mean" else v.median())
            is_int = bool(np.all(np.isclose(v.values, np.round(v.values))))
            if is_int:
                self.integer_cols.append(c)
                fill = float(np.round(fill))
            self.fill_num[c] = fill
            if self.prep.outliers == "clip" and v.nunique() > 2:
                q1, q3 = float(v.quantile(0.25)), float(v.quantile(0.75))
                iqr = q3 - q1
                if iqr > 0:
                    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    if is_int:
                        lo, hi = float(np.ceil(lo)), float(np.floor(hi))
                    self.clip_bounds[c] = (lo, hi)
        for c in self.categorical_cols:
            mode = sample[c].dropna().mode()
            self.fill_cat[c] = str(mode.iloc[0]) if len(mode) else "unknown"
        if self.target_clean and self.target_clean in sample.columns and pd.api.types.is_numeric_dtype(sample[self.target_clean]):
            tv = sample[self.target_clean].dropna()
            if len(tv) and np.all(np.isclose(tv.values.astype(float), np.round(tv.values.astype(float)))):
                self.integer_cols.append(self.target_clean)

    def fit_ml_preprocessor(self, sample: pd.DataFrame) -> None:
        if not self.feature_cols:
            return
        X = prepare_frame(sample, self.numeric_cols, self.categorical_cols)
        if self.prep.missing == "drop":
            X = X[X.notna().all(axis=1)]
        pre = build_preprocessor(self.numeric_cols, self.categorical_cols, self.prep,
                                 skewed_cols=self.skewed_cols, scale=True)
        pre.fit(X)
        self.ml_pre = pre

    # -- apply to a cleaned (pre-imputation) frame ----------------------------------
    def finalize(self, clean: pd.DataFrame, keep_extras: bool = False) -> pd.DataFrame:
        df = clean.copy()
        if self.prep.missing == "drop":
            df = df.dropna(subset=[c for c in self.feature_cols if c in df.columns])
        cols = []
        if keep_extras:
            cols += [c for c in self.extra_cols if c in df.columns]
        cols += [c for c in self.numeric_cols + self.categorical_cols if c in df.columns]
        if self.target_clean and self.target_clean in df.columns:
            cols.append(self.target_clean)
        df = df[cols]
        for c in self.numeric_cols:
            if c not in df.columns:
                continue
            s = pd.to_numeric(df[c], errors="coerce").astype(float)
            if self.prep.missing != "drop":
                s = s.fillna(self.fill_num.get(c, 0.0))
            if c in self.clip_bounds:
                lo, hi = self.clip_bounds[c]
                s = s.clip(lo, hi)
            if self.prep.skew == "log" and c in self.skewed_cols:
                s = np.log1p(s.clip(lower=0))
            df[c] = s
        if self.prep.missing != "drop":
            for c in self.categorical_cols:
                if c in df.columns:
                    df[c] = df[c].astype(object).where(df[c].notna(), self.fill_cat.get(c, "unknown"))
        for c in self.integer_cols:
            if c in df.columns and not (self.prep.skew == "log" and c in self.skewed_cols):
                if df[c].notna().all():
                    df[c] = np.round(df[c].astype(float)).astype("int64")
        if self.prep.skew == "log":
            df = df.rename(columns={c: f"{c}_log1p" for c in self.skewed_cols if c in df.columns})
        return df

    def to_ml(self, clean: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.ml_pre is None:
            return None
        df = clean
        if self.prep.missing == "drop":
            df = df.dropna(subset=[c for c in self.feature_cols if c in df.columns])
        X = prepare_frame(df, self.numeric_cols, self.categorical_cols)
        arr = np.asarray(self.ml_pre.transform(X), dtype=float)
        names = [str(n) for n in self.ml_pre.get_feature_names_out()]
        out = pd.DataFrame(arr, columns=names, index=df.index)
        if self.target_clean and self.target_clean in df.columns:
            out[self.target_clean] = df[self.target_clean].values
        return out

    # -- convenience for re-using the pipeline on NEW raw data --------------------------
    def transform_raw(self, raw: pd.DataFrame, ml_ready: bool = False, keep_extras: bool = False) -> pd.DataFrame:
        clean = self.cleaner.transform(raw)
        if ml_ready:
            out = self.to_ml(clean)
            if out is None:
                raise RuntimeError("No ML-ready preprocessor was fitted for this dataset.")
            return out
        return self.finalize(clean, keep_extras=keep_extras)


@dataclass
class ExportInfo:
    cleaned_path: str
    ml_path: Optional[str]
    bundle_path: Optional[str]
    script_path: Optional[str]
    report_path: Optional[str]
    rows_in: int = 0
    rows_out: int = 0
    duplicates_removed: int = 0
    key_duplicates_removed: int = 0
    missing_target_removed: int = 0
    missing_feature_rows_removed: int = 0
    cols_out: List[str] = field(default_factory=list)
    ml_cols: List[str] = field(default_factory=list)


def export_full_dataset(ds, plan: ExportPlan, out_dir: str, keep_extras: bool = False,
                        progress: Optional[Callable[[float, str], None]] = None) -> ExportInfo:
    """Stream the COMPLETE dataset through the pipeline and write the CSV deliverables."""
    os.makedirs(out_dir, exist_ok=True)
    cleaned_path = os.path.join(out_dir, "cleaned_dataset.csv")
    ml_path = os.path.join(out_dir, "ml_ready_dataset.csv") if plan.ml_pre is not None else None
    for p in (cleaned_path, ml_path):
        if p and os.path.exists(p):
            os.remove(p)

    dedupe = Deduper(plan.key_col)
    info = ExportInfo(cleaned_path, ml_path, None, None, None)
    total = None
    try:
        total = ds.row_count_estimate()
    except Exception:
        pass
    first = True
    first_ml = True
    seen_rows = 0
    for chunk in ds.iter_chunks(CHUNK_ROWS):
        info.rows_in += len(chunk)
        seen_rows += len(chunk)
        if progress and total:
            progress(min(seen_rows / total, 1.0), f"Exporting rows {seen_rows:,} / {total:,}")
        clean = plan.cleaner.transform(chunk)
        clean = dedupe.filter(clean)
        if plan.target_clean and plan.target_clean in clean.columns:
            before = len(clean)
            clean = clean[clean[plan.target_clean].notna()]
            info.missing_target_removed += before - len(clean)
        if clean.empty:
            continue
        if plan.prep.missing == "drop":
            before = len(clean)
            clean = clean.dropna(subset=[c for c in plan.feature_cols if c in clean.columns])
            info.missing_feature_rows_removed += before - len(clean)
            if clean.empty:
                continue
        out = plan.finalize(clean, keep_extras=keep_extras)
        out.round(6).to_csv(cleaned_path, mode="w" if first else "a", header=first, index=False,
                            encoding="utf-8-sig" if first else "utf-8")
        if first:
            info.cols_out = list(out.columns)
        first = False
        info.rows_out += len(out)
        if ml_path:
            ml = plan.to_ml(clean)
            if ml is not None and len(ml):
                ml.round(6).to_csv(ml_path, mode="w" if first_ml else "a", header=first_ml, index=False,
                                   encoding="utf-8-sig" if first_ml else "utf-8")
                if first_ml:
                    info.ml_cols = list(ml.columns)
                first_ml = False
    if first:  # nothing survived filtering: still write a valid, header-only file
        cols = plan.finalize(plan.cleaner.transform(ds.sample(5)).head(0), keep_extras=keep_extras).columns
        pd.DataFrame(columns=list(cols)).to_csv(cleaned_path, index=False, encoding="utf-8-sig")
        info.cols_out = list(cols)
    if ml_path and first_ml:
        ml_path = None
        info.ml_path = None
    info.duplicates_removed = dedupe.removed_exact
    info.key_duplicates_removed = dedupe.removed_key
    return info


# ---------------------------------------------------------------------------
# Generated artefacts
# ---------------------------------------------------------------------------
def save_bundle(plan: ExportPlan, path: str) -> str:
    import joblib
    joblib.dump(plan, path)
    return path


def generate_pipeline_code(plan: ExportPlan, task: str, source_name: str = "your_dataset.csv") -> str:
    """Standalone script: re-applies the fitted cleaning + preprocessing to new data and shows how to train the winner."""
    lines = ['"""',
             "Auto-generated by AutoPrep Studio.",
             "",
             "Re-applies the SAME cleaning + preprocessing that was applied to your dataset and shows",
             "how to train the winning model. Put this file next to `autoprep_pipeline.joblib` and run it",
             "from the AutoPrep Studio project folder (it imports core.cleaning / core.transform),",
             "or set the AUTOPREP_HOME environment variable to that folder.",
             "",
             f"Preprocessing chosen by the ablation study: {plan.prep.describe()}."]
    if plan.best_model_repr:
        lines.append(f"Best model: {plan.best_model_repr.split('(')[0]} ({plan.metric_name} = {plan.best_score:.4f} in cross-validation).")
    lines += ['"""', "",
              "import os", "import sys", "",
              'sys.path.insert(0, os.environ.get("AUTOPREP_HOME", "."))', "",
              "import joblib", "import pandas as pd", "",
              f"NUMERIC_COLS = {plan.numeric_cols!r}",
              f"CATEGORICAL_COLS = {plan.categorical_cols!r}",
              f"TARGET_COL = {plan.target_clean!r}",
              f"SKEWED_COLS = {plan.skewed_cols!r}   # log1p applied here: {plan.prep.skew == 'log'}",
              f"TASK = {task!r}", "",
              'bundle = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "autoprep_pipeline.joblib"))',
              "",
              "",
              "def load_raw(path: str) -> pd.DataFrame:",
              '    """Read the raw file as TEXT - the cleaner does all the typing."""',
              '    return pd.read_csv(path, dtype=str, keep_default_na=False).replace("", pd.NA)',
              "",
              "",
              "def clean(raw: pd.DataFrame) -> pd.DataFrame:",
              '    """Tidy, imputed, human-readable table (same as cleaned_dataset.csv)."""',
              "    return bundle.transform_raw(raw)",
              "",
              "",
              "def ml_ready(raw: pd.DataFrame) -> pd.DataFrame:",
              '    """Fully numeric table (encoded + scaled) for modelling (same as ml_ready_dataset.csv)."""',
              "    return bundle.transform_raw(raw, ml_ready=True)",
              ""]
    if plan.best_model_repr and plan.target_clean:
        lines += ["def train_best_model(raw: pd.DataFrame):",
                  '    """Train the winning model on a raw dataset that contains the target column."""',
                  "    from sklearn.pipeline import Pipeline",
                  "    from core.transform import build_preprocessor, prepare_frame, PrepConfig, encode_target",
                  f"    {plan.best_model_import}", "",
                  "    cleaned = bundle.cleaner.transform(raw)",
                  "    cleaned = cleaned[cleaned[TARGET_COL].notna()]",
                  "    X = prepare_frame(cleaned, NUMERIC_COLS, CATEGORICAL_COLS)",
                  "    y, classes = encode_target(cleaned[TARGET_COL], TASK)",
                  f"    cfg = PrepConfig(missing={plan.prep.missing!r}, outliers={plan.prep.outliers!r}, "
                  f"skew={plan.prep.skew!r}, encoding={plan.prep.encoding!r})",
                  f"    prep = build_preprocessor(NUMERIC_COLS, CATEGORICAL_COLS, cfg, skewed_cols=SKEWED_COLS, "
                  f"scale={plan.best_model_scaled})",
                  f"    model = Pipeline([('prep', prep), ('model', {plan.best_model_repr})])",
                  "    return model.fit(X, y), classes",
                  ""]
    lines += ["", 'if __name__ == "__main__":',
              f'    raw = load_raw("{source_name}")   # replace with your input path',
              "    print(clean(raw).head())",
              "    print(ml_ready(raw).shape)"]
    return "\n".join(lines)


def build_report_markdown(result) -> str:
    """Human-readable summary of everything the pipeline did (duck-typed on AutoResult)."""
    L = [f"# AutoPrep Studio report — {result.source_name}", ""]
    L.append(f"- Rows: {result.n_rows_total:,} → {result.export.rows_out:,} after cleaning" if result.export else f"- Rows: {result.n_rows_total:,}")
    L.append(f"- Task: **{result.task}**" + (f" (target `{result.target_clean}`)" if result.target_clean else ""))
    L.append(f"- Preprocessing chosen: {result.prep.describe()}")
    if result.models and result.models.best_name:
        L.append(f"- Best model: **{result.models.best_name}** — {result.models.metric_name} {result.models.best_score:.4f}")
    L += ["", "## Cleaning actions", ""]
    if result.cleaning is not None and len(result.cleaning.actions):
        for _, r in result.cleaning.actions.iterrows():
            L.append(f"- {r['action']}: {int(r['cells']):,} cells")
    L += ["", "## Columns", ""]
    if result.cleaning is not None:
        for _, r in result.cleaning.column_table.iterrows():
            L.append(f"- `{r['original name']}` → `{r['clean name']}` ({r['detected type']}): {r['what was done']}")
    if result.excluded is not None and len(result.excluded):
        L += ["", "## Columns not used as features", ""]
        for _, r in result.excluded.iterrows():
            L.append(f"- `{r['column']}`: {r['reason']}")
    if result.leakage:
        L += ["", "## Leakage flags", ""]
        for f in result.leakage:
            L.append(f"- `{f.column}` (risk {f.risk_score:.2f}): {' | '.join(f.reasons)}")
    return "\n".join(L)
