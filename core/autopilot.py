"""
autopilot.py
------------
The automatic pipeline. Upload a file and this runs EVERYTHING, in order:

  1. load + sample                       (data_io)
  2. detect the target column            (name / position heuristics, or "none" -> unsupervised)
  3. clean + normalise                   (cleaning.py: currency, units, dates, categories, junk columns)
  4. profile raw vs cleaned              (profiling.py)
  5. leakage scan + automatic exclusion  (leakage.py)
  6. preprocessing ablation study        (transform.py)          [supervised]
  7. model comparison                    (models.py)             [supervised: many models]
     or clustering / anomaly / PCA       (models.py)             [no target: unsupervised]
  8. export the FULL dataset             (apply_final.py)

The only thing left for the user is to download the result.

`ds` only needs: row_count_estimate(), sample(n), iter_chunks(rows), source_name -
so it can be a polars-backed LoadedDataset or any pandas-backed stand-in.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .apply_final import (Deduper, ExportInfo, ExportPlan, build_report_markdown, export_full_dataset,
                          generate_pipeline_code, save_bundle)
from .cleaning import CleaningReport, DataCleaner
from .leakage import LeakageFlag, detect_leakage
from .models import ModelComparison, UnsupervisedResult, analyze_unsupervised, compare_supervised, _subsample
from .profiling import DataQualityReport, count_duplicates, profile_dataframe
from .transform import (AblationOutcome, PrepConfig, encode_target, infer_task_type, prepare_frame,
                        run_ablation)

CLEAN_SAMPLE_ROWS = 100_000      # rows used to LEARN the cleaning plan / run experiments
AUTO = "auto"
NONE = "__none__"

STRONG_TARGET_NAMES = [
    "target", "label", "class", "y", "outcome", "churn", "churned", "survived", "default", "fraud",
    "is_fraud", "attrition", "diagnosis", "species", "response", "converted", "exited", "will_fail",
    "failure", "readmitted", "loan_status", "approved", "saleprice", "sale_price", "price",
]
_TARGET_PATTERN = re.compile(r"(^|_)(target|label|churn|survived|fraud|default|outcome|attrition|failure)($|_)")
_NOT_TARGET_PATTERN = re.compile(r"(reason|date|time|source|note|comment|id$)")


@dataclass
class Settings:
    target: str = AUTO                 # AUTO | NONE | <raw column name>
    task: str = AUTO                   # auto | classification | regression | unsupervised
    speed: str = "balanced"            # fast | balanced | thorough
    keep_extra_columns: bool = False   # keep identifier / contact / raw-date columns in the export
    max_missing_frac: float = 0.70
    auto_exclude_risk: float = 0.60

    def key(self) -> tuple:
        return (self.target, self.task, self.speed, self.keep_extra_columns, self.max_missing_frac,
                self.auto_exclude_risk)


@dataclass
class AutoResult:
    source_name: str
    settings: Settings
    n_rows_total: int
    n_cols_raw: int
    sample_rows: int
    used_sampling: bool
    target_raw: Optional[str]
    target_clean: Optional[str]
    target_reason: str
    task: str
    classes: Optional[List[str]]
    cleaning: CleaningReport
    raw_profile: DataQualityReport
    clean_profile: DataQualityReport
    missing_compare: pd.DataFrame
    leakage: List[LeakageFlag]
    excluded: pd.DataFrame
    numeric_cols: List[str]
    categorical_cols: List[str]
    prep: PrepConfig
    ablation: Optional[AblationOutcome] = None
    models: Optional[ModelComparison] = None
    unsup: Optional[UnsupervisedResult] = None
    export: Optional[ExportInfo] = None
    plan: Optional[ExportPlan] = None
    correlation: Optional[pd.DataFrame] = None
    class_counts: Optional[Dict[str, int]] = None
    preview_clean: Optional[pd.DataFrame] = None
    final_profile: Optional[DataQualityReport] = None
    warnings: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    pipeline_code: str = ""
    report_markdown: str = ""
    dup_raw: int = 0
    dup_removed_sample: int = 0
    target_missing_removed_sample: int = 0


# ---------------------------------------------------------------------------
# Target detection
# ---------------------------------------------------------------------------
def detect_target(cleaner: DataCleaner, clean: pd.DataFrame) -> Tuple[Optional[str], str]:
    """Pick a target from the column names / position. Returns (raw column name, reason)."""
    cands = [(orig, p) for orig, p in cleaner.plans.items()
             if not p.drop and p.role == "feature" and p.kind != "datetime"]
    if not cands:
        return None, "no usable columns"
    by_name = {p.name: orig for orig, p in cands}
    for s in STRONG_TARGET_NAMES:
        if s in by_name:
            return by_name[s], f"column name '{s}' looks like a prediction target"
    for orig, p in reversed(cands):
        if _TARGET_PATTERN.search(p.name) and not _NOT_TARGET_PATTERN.search(p.name):
            return orig, f"column name '{p.name}' looks like a prediction target"
    orig, p = cands[-1]
    col = clean[p.name] if p.name in clean.columns else None
    if col is not None and col.notna().sum() > 0:
        n_unique = col.nunique(dropna=True)
        if p.kind == "boolean" or (p.kind == "categorical" and 2 <= n_unique <= 20):
            return orig, f"last column '{p.name}' is a {p.kind} column with {n_unique} values (typical label position)"
        if p.kind == "numeric" and n_unique > 2 and not re.search(r"(^|_)(id|index|year|month|day)($|_)", p.name):
            return orig, f"last column '{p.name}' is numeric (typical position for a value to predict)"
    return None, "no obvious target column — running unsupervised analysis instead"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sub_progress(progress, lo: float, hi: float):
    if progress is None:
        return None
    return lambda f, m: progress(lo + (hi - lo) * float(min(max(f, 0.0), 1.0)), m)


def _correlation(clean: pd.DataFrame, cols: List[str], max_cols: int = 25) -> Optional[pd.DataFrame]:
    cols = [c for c in cols if c in clean.columns and clean[c].nunique(dropna=True) > 1][:max_cols]
    if len(cols) < 2:
        return None
    return clean[cols].astype(float).corr()


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
def run_autopilot(ds, settings: Optional[Settings] = None,
                  progress: Optional[Callable[[float, str], None]] = None,
                  out_dir: Optional[str] = None, export: bool = True) -> AutoResult:
    settings = settings or Settings()
    timings: Dict[str, float] = {}
    warns: List[str] = []
    t_start = time.time()

    def tick(f: float, msg: str):
        if progress:
            progress(f, msg)

    # ---- 1. load ------------------------------------------------------------------
    tick(0.01, "Loading data")
    n_total = int(ds.row_count_estimate())
    sample_n = min(n_total, CLEAN_SAMPLE_ROWS)
    raw = ds.sample(sample_n).reset_index(drop=True)
    used_sampling = n_total > len(raw)
    if raw.empty:
        raise ValueError("The dataset has no rows.")
    timings["load"] = time.time() - t_start

    # ---- 2/3. target detection + cleaning -----------------------------------------------
    t0 = time.time()
    tick(0.06, "Cleaning & normalising columns")
    probe = DataCleaner().fit(raw)
    probe_clean = probe.transform(raw)
    target_raw: Optional[str]
    if settings.target == NONE or settings.task == "unsupervised":
        target_raw, reason = None, "unsupervised analysis requested"
    elif settings.target == AUTO:
        target_raw, reason = detect_target(probe, probe_clean)
    elif settings.target in raw.columns:
        target_raw, reason = settings.target, "chosen by you"
    else:
        target_raw, reason = detect_target(probe, probe_clean)
        warns.append(f"Target '{settings.target}' was not found; auto-detected instead.")
    if target_raw is not None and probe.plans[target_raw].drop:
        warns.append(f"Target '{target_raw}' is unusable ({probe.plans[target_raw].drop_reason}); running unsupervised analysis.")
        target_raw, reason = None, "target column unusable"

    cleaner = DataCleaner(target=target_raw).fit(raw)
    clean = cleaner.transform(raw, collect_stats=True)
    report = cleaner.report()
    target_clean = cleaner.target_name
    roles = cleaner.roles()
    timings["cleaning"] = time.time() - t0

    # ---- row-level cleanup on the sample: duplicates + missing target ----------------------------
    key_col = None
    for c in roles["identifier"]:
        s = clean[c]
        if s.notna().sum() and s.nunique() / s.notna().sum() >= 0.9:
            key_col = c
            break
    index_cols = [p.original for p in cleaner.plans.values() if p.kind == "index"]
    dup_raw = count_duplicates(raw, ignore_cols=index_cols)
    dd = Deduper(key_col)
    clean_s = dd.filter(clean)
    dup_removed = dd.removed_exact + dd.removed_key
    tmr = 0
    if target_clean is not None:
        before = len(clean_s)
        clean_s = clean_s[clean_s[target_clean].notna()]
        tmr = before - len(clean_s)
    clean_s = clean_s.reset_index(drop=True)
    if len(clean_s) < 30:
        raise ValueError("Fewer than 30 usable rows remain after cleaning — the data looks too small or too empty.")

    # ---- 4. profiling -------------------------------------------------------------------------------
    t0 = time.time()
    tick(0.16, "Profiling data quality")
    raw_profile = profile_dataframe(raw, n_rows_total=n_total, sampled=used_sampling, duplicate_rows=dup_raw)
    clean_profile = profile_dataframe(clean_s, kinds=cleaner.kinds(), n_rows_total=n_total, sampled=used_sampling,
                                      duplicate_rows=dup_removed)
    mc_rows = []
    for orig in cleaner.order:
        p = cleaner.plans[orig]
        if p.drop:
            continue
        mc_rows.append({"column": p.name,
                        "before cleaning %": round(100 * float(raw[orig].isna().mean()), 2),
                        "after cleaning %": round(100 * float(clean[p.name].isna().mean()), 2)})
    missing_compare = pd.DataFrame(mc_rows)
    timings["profiling"] = time.time() - t0

    # ---- task ---------------------------------------------------------------------------------------------
    task = "unsupervised"
    if target_clean is not None:
        y_clean = clean_s[target_clean]
        if settings.task in ("classification", "regression"):
            task = settings.task
            if task == "regression" and not pd.api.types.is_numeric_dtype(y_clean):
                warns.append("Target is not numeric, so regression is impossible — using classification.")
                task = "classification"
        else:
            task = infer_task_type(y_clean)
        if task == "classification" and not pd.api.types.is_numeric_dtype(y_clean) and y_clean.nunique() > 100:
            warns.append(f"Target has {y_clean.nunique()} distinct labels — too many for classification; "
                         "running unsupervised analysis instead.")
            task, target_clean, target_raw = "unsupervised", None, None
    if task == "classification" and target_clean is not None:
        vc = clean_s[target_clean].value_counts()
        if len(vc) > 2 and target_raw and cleaner.plans[target_raw].kind == "categorical" and any(
                k in ("1", "0") or str(k).lower() in ("yes", "no", "y", "n", "true", "false") for k in vc.index):
            warns.append("The target has more than two label spellings that could not be merged automatically "
                         f"({', '.join(map(str, vc.index[:6]))} ...). Check the Cleaning tab.")

    # ---- candidate features ----------------------------------------------------------------------------------
    excluded: List[dict] = []
    for _, r in report.dropped.iterrows():
        excluded.append({"column": r["column"], "reason": r["reason"], "stage": "cleaning"})
    for c in roles["identifier"]:
        excluded.append({"column": c, "reason": "identifier — not a predictive feature", "stage": "role"})
    for c in roles["contact"]:
        excluded.append({"column": c, "reason": "contact detail (email / phone) — not a predictive feature", "stage": "role"})
    for c in roles["text"]:
        excluded.append({"column": c, "reason": "free text — not used as a feature", "stage": "role"})
    for c in roles["datetime_raw"]:
        excluded.append({"column": c, "reason": "raw date replaced by year / month / weekday / epoch-day features", "stage": "role"})

    num = [c for c in roles["numeric"] if c != target_clean]
    cat = [c for c in roles["categorical"] if c != target_clean]

    # ---- 5. leakage --------------------------------------------------------------------------------------------------
    t0 = time.time()
    tick(0.20, "Scanning for leakage")
    flags: List[LeakageFlag] = []
    if task != "unsupervised" and (num or cat):
        flags = detect_leakage(clean_s[num + cat + [target_clean]], target_clean, task_type=task)
        for f in flags:
            if f.risk_score >= settings.auto_exclude_risk:
                excluded.append({"column": f.column, "reason": f"leakage / low-value risk {f.risk_score:.2f}: {f.reasons[0]}",
                                 "stage": "leakage"})
        drop_now = {f.column for f in flags if f.risk_score >= settings.auto_exclude_risk}
        num = [c for c in num if c not in drop_now]
        cat = [c for c in cat if c not in drop_now]
    # too many missing values / constants after cleaning
    for c in list(num + cat):
        miss = float(clean_s[c].isna().mean())
        if miss > settings.max_missing_frac:
            excluded.append({"column": c, "reason": f"{miss:.0%} of values are missing", "stage": "missing"})
            num, cat = [x for x in num if x != c], [x for x in cat if x != c]
        elif clean_s[c].nunique(dropna=True) <= 1:
            excluded.append({"column": c, "reason": "constant after cleaning", "stage": "cleaning"})
            num, cat = [x for x in num if x != c], [x for x in cat if x != c]
    timings["leakage"] = time.time() - t0
    excluded_df = pd.DataFrame(excluded, columns=["column", "reason", "stage"])

    if not (num or cat):
        warns.append("No usable feature columns remain after cleaning and exclusions.")

    # ---- 6/7. experiments ---------------------------------------------------------------------------------------------------
    prep = PrepConfig()
    ablation = models = unsup = None
    classes = None
    class_counts = None
    skewed: List[str] = []
    if (num or cat) and task != "unsupervised":
        X = prepare_frame(clean_s, num, cat)
        y_enc, classes = encode_target(clean_s[target_clean], task)
        if task == "classification":
            class_counts = {str(classes[int(c)]): int((y_enc == c).sum()) for c in np.unique(y_enc)}
        t0 = time.time()
        Xa, ya = _subsample(X, y_enc, task, 20_000)
        try:
            ablation = run_ablation(Xa, ya, num, cat, task, n_estimators=100, cv_folds=3,
                                    progress=_sub_progress(progress, 0.24, 0.42))
            prep = ablation.best.cfg if ablation.best is not None else PrepConfig()
            skewed = ablation.skewed_numeric_cols
        except Exception as e:  # never let an experiment kill the export
            warns.append(f"Ablation study failed: {e}")
        timings["ablation"] = time.time() - t0
        t0 = time.time()
        try:
            models = compare_supervised(X, y_enc, num, cat, task, prep, skewed_cols=skewed, speed=settings.speed,
                                        classes=classes, progress=_sub_progress(progress, 0.42, 0.80))
        except Exception as e:
            warns.append(f"Model comparison failed: {e}")
        timings["models"] = time.time() - t0
    elif (num or cat):
        t0 = time.time()
        X = prepare_frame(clean_s, num, cat)
        try:
            unsup = analyze_unsupervised(X, num, cat, speed=settings.speed, progress=_sub_progress(progress, 0.24, 0.80))
        except Exception as e:
            warns.append(f"Unsupervised analysis failed: {e}")
        timings["unsupervised"] = time.time() - t0

    # ---- 8. export -----------------------------------------------------------------------------------------------------------
    corr_cols = num + ([target_clean] if target_clean and pd.api.types.is_numeric_dtype(clean_s[target_clean]) else [])
    correlation = _correlation(clean_s, corr_cols)
    extra = roles["identifier"] + roles["contact"] + roles["text"] + roles["datetime_raw"]
    plan = ExportPlan(cleaner=cleaner, target_clean=target_clean, task=task, key_col=key_col, numeric_cols=num,
                      categorical_cols=cat, extra_cols=extra, prep=prep, skewed_cols=skewed, classes=classes)
    if models is not None and models.best_name:
        plan.best_model_import, plan.best_model_repr = models.best_import, models.best_repr
        plan.best_model_scaled = models.best_needs_scaling
        plan.metric_name, plan.best_score = models.metric_name, models.best_score
    plan.learn(clean_s)
    try:
        plan.fit_ml_preprocessor(clean_s)
    except Exception as e:
        warns.append(f"ML-ready encoding unavailable: {e}")
        plan.ml_pre = None

    preview = plan.finalize(clean_s, keep_extras=settings.keep_extra_columns)
    final_profile = profile_dataframe(preview, n_rows_total=len(preview))
    info = None
    result = AutoResult(
        source_name=getattr(ds, "source_name", "dataset"), settings=settings, n_rows_total=n_total,
        n_cols_raw=raw.shape[1], sample_rows=len(raw), used_sampling=used_sampling, target_raw=target_raw,
        target_clean=target_clean, target_reason=reason, task=task, classes=classes, cleaning=report,
        raw_profile=raw_profile, clean_profile=clean_profile, missing_compare=missing_compare, leakage=flags,
        excluded=excluded_df, numeric_cols=num, categorical_cols=cat, prep=prep, ablation=ablation,
        models=models, unsup=unsup, plan=plan, correlation=correlation, class_counts=class_counts,
        preview_clean=preview.head(300), final_profile=final_profile, warnings=warns, timings=timings,
        dup_raw=dup_raw, dup_removed_sample=dup_removed, target_missing_removed_sample=tmr,
    )
    if export:
        t0 = time.time()
        out_dir = out_dir or tempfile.mkdtemp(prefix="autoprep_")
        info = export_full_dataset(ds, plan, out_dir, keep_extras=settings.keep_extra_columns,
                                   progress=_sub_progress(progress, 0.82, 0.96))
        try:
            info.bundle_path = save_bundle(plan, os.path.join(out_dir, "autoprep_pipeline.joblib"))
            code = generate_pipeline_code(plan, task, source_name=result.source_name)
            info.script_path = os.path.join(out_dir, "preprocessing_pipeline.py")
            with open(info.script_path, "w", encoding="utf-8") as fh:
                fh.write(code)
            result.pipeline_code = code
        except Exception as e:
            warns.append(f"Could not write the reusable pipeline files: {e}")
        result.export = info
        try:
            result.report_markdown = build_report_markdown(result)
            info.report_path = os.path.join(out_dir, "cleaning_report.md")
            with open(info.report_path, "w", encoding="utf-8") as fh:
                fh.write(result.report_markdown)
        except Exception as e:
            warns.append(f"Could not write the report: {e}")
        timings["export"] = time.time() - t0
    timings["total"] = time.time() - t_start
    tick(1.0, "Done")
    return result
