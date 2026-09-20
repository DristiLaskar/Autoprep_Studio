"""
AutoPrep Studio — automatic data cleaning, leakage detection, preprocessing search and model comparison.
====================================================================================================
Run with:  streamlit run app.py

Upload a file and everything runs by itself. The only thing left to do is download the result.
See README.md for details.
"""

import hashlib
import os
import re
import tempfile
import traceback

import pandas as pd
import streamlit as st

from core import charts as ch
from core.autopilot import AUTO, NONE, AutoResult, Settings, run_autopilot
from core.data_io import LARGE_ROW_THRESHOLD, load_from_manual_table, load_from_path
from core.leakage import flags_to_frame
from core.models import available_model_names
from core.theme import CSS

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_PATH = os.path.join(APP_DIR, "sample_data", "messy_churn_dataset.csv")

st.set_page_config(page_title="AutoPrep Studio", page_icon="🧪", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Small UI helpers (work across Streamlit versions)
# ---------------------------------------------------------------------------
def _stretch(fn, *args, **kwargs):
    try:
        return fn(*args, use_container_width=True, **kwargs)
    except TypeError:
        return fn(*args, width="stretch", **kwargs)


def show_df(df: pd.DataFrame, height=None, column_config=None):
    kw = {"hide_index": True}
    if height:
        kw["height"] = height
    if column_config:
        kw["column_config"] = column_config
    _stretch(st.dataframe, df, **kw)


def show_fig(fig):
    _stretch(st.plotly_chart, fig, theme=None)


@st.cache_data(show_spinner=False)
def read_bytes(path: str, mtime: float) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def file_bytes(path: str) -> bytes:
    return read_bytes(path, os.path.getmtime(path))


def pills(items, kind=""):
    html = "".join(f'<span class="pill {kind}">{i}</span>' for i in items)
    st.markdown(html, unsafe_allow_html=True)


def progress_col(label, series, fmt="%.3f"):
    lo = float(min(0.0, series.min())) if len(series) else 0.0
    hi = float(max(1.0, series.max())) if len(series) else 1.0
    return st.column_config.ProgressColumn(label, min_value=lo, max_value=hi, format=fmt)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
ss = st.session_state
for key, default in {
    "ds": None, "data_key": None, "last_upload_key": None, "result": None, "run_key": None,
    "error": None, "settings": Settings(), "manual_df": None, "load_error": None, "preview": None,
}.items():
    if key not in ss:
        ss[key] = default


def set_dataset(ds, key):
    ss.ds, ss.data_key = ds, key
    ss.result, ss.run_key, ss.error, ss.load_error = None, None, None, None
    ss.settings = Settings()
    try:                                   # cached once, so reruns stay instant
        ss.preview = ds.sample(50).head(50)
    except Exception:
        ss.preview = None


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>🧪 AutoPrep Studio</h1>
      <p>Drop in a messy dataset — it is <b>cleaned, checked for leakage, preprocessed, and tested with a whole
      lineup of ML models automatically</b>. Your only job: download the result.</p>
      <span class="pill">🧹 cleaning</span><span class="pill teal">🛡️ leakage scan</span>
      <span class="pill pink">⚗️ preprocessing search</span><span class="pill amber">🤖 supervised + unsupervised models</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 1 · Input (runs once per new file — never on every rerun)
# ---------------------------------------------------------------------------
up_col, demo_col = st.columns([3, 1])
with up_col:
    uploaded = st.file_uploader(
        "Upload your dataset (CSV, TSV, Excel, JSON, NDJSON, Parquet)",
        type=["csv", "tsv", "txt", "dat", "xlsx", "xls", "json", "ndjson", "jsonl", "parquet"],
    )
with demo_col:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🎯 Try the messy demo dataset"):
        try:
            set_dataset(load_from_path(DEMO_PATH, original_filename="messy_churn_dataset.csv"), ("demo",))
        except Exception as e:
            ss.load_error = f"Could not load the demo file: {e}"

if uploaded is not None:
    data = uploaded.getvalue()
    upload_key = ("upload", uploaded.name, hashlib.md5(data).hexdigest())
    if upload_key != ss.last_upload_key:         # only when the file actually changed
        ss.last_upload_key = upload_key
        try:
            safe = re.sub(r"[^\w.\-]", "_", os.path.basename(uploaded.name))
            path = os.path.join(tempfile.mkdtemp(prefix="autoprep_upload_"), safe)
            with open(path, "wb") as fh:
                fh.write(data)
            set_dataset(load_from_path(path, original_filename=uploaded.name), upload_key)
        except Exception as e:
            ss.load_error = f"Failed to load file: {e}"
            ss.load_trace = traceback.format_exc()

with st.expander("✍️ …or type a small table by hand"):
    if ss.manual_df is None:
        ss.manual_df = pd.DataFrame({"column_1": [None, None, None], "column_2": [None, None, None]})
    edited = _stretch(st.data_editor, ss.manual_df, num_rows="dynamic", key="manual_editor")
    ss.manual_df = edited
    if st.button("Use this table"):
        tbl = ss.manual_df.dropna(how="all")
        if tbl.empty:
            ss.load_error = "The table is empty — add some data first."
        else:
            set_dataset(load_from_manual_table(tbl), ("manual", hashlib.md5(tbl.to_csv().encode()).hexdigest()))

if ss.load_error:
    st.error(ss.load_error)
    if ss.get("load_trace"):
        with st.expander("Technical details"):
            st.code(ss.load_trace)

ds = ss.ds
if ds is not None:
    if getattr(ds, "warning", None):
        st.warning(ds.warning)
    if ss.preview is not None:
        with st.expander(f"👀 Raw data as uploaded — {ds.source_name} (first rows)"):
            show_df(ss.preview, height=300)

# ---------------------------------------------------------------------------
# Sidebar: OPTIONAL settings (applied together, so nothing re-runs while you fiddle)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Optional settings")
    st.caption("You never have to touch these — everything is automatic. "
               "Change them only if you disagree with a choice, then press **Apply**.")
    columns = []
    if ds is not None:
        try:
            columns = list(ds.columns)
        except Exception:
            columns = []
    with st.form("settings_form"):
        target_choice = st.selectbox("Target column (what to predict)",
                                     ["Auto-detect", "None — unsupervised analysis"] + columns)
        task_choice = st.selectbox("Task type", ["Auto", "Classification", "Regression", "Unsupervised"])
        speed_choice = st.select_slider("Model search depth", options=["fast", "balanced", "thorough"],
                                        value="balanced")
        keep_extra = st.checkbox("Keep ID / contact / raw-date columns in the export", value=False)
        with st.expander("Advanced"):
            max_missing = st.slider("Drop features with more than … % missing", 30, 95, 70)
            risk_thr = st.slider("Auto-exclude leakage risk ≥", 0.30, 0.95, 0.60, 0.05)
        applied = st.form_submit_button("Apply & re-run")
    if applied:
        tgt = AUTO if target_choice == "Auto-detect" else (NONE if target_choice.startswith("None") else target_choice)
        ss.settings = Settings(target=tgt, task=task_choice.lower(), speed=speed_choice,
                               keep_extra_columns=keep_extra, max_missing_frac=max_missing / 100.0,
                               auto_exclude_risk=float(risk_thr))
    st.divider()
    st.markdown("**Models in the lineup**")
    st.caption("Classification: " + ", ".join(available_model_names("classification", ss.settings.speed)))
    st.caption("Regression: " + ", ".join(available_model_names("regression", ss.settings.speed)))
    st.caption("No target → KMeans, MiniBatch KMeans, Gaussian Mixture, BIRCH, DBSCAN, Agglomerative, "
               "Isolation Forest, LOF, One-Class SVM, PCA")

# ---------------------------------------------------------------------------
# 2 · Automatic pipeline
# ---------------------------------------------------------------------------
if ds is None:
    st.info("👆 Upload a file (or click the demo button) and everything will run automatically.")
    st.stop()

run_key = (ss.data_key, ss.settings.key())
if ss.run_key != run_key:
    bar = st.progress(0.0, text="Starting…")

    def on_progress(frac: float, message: str):
        bar.progress(float(min(max(frac, 0.0), 1.0)), text=message)

    try:
        ss.result = run_autopilot(ds, ss.settings, progress=on_progress)
        ss.error = None
    except Exception as e:
        ss.result = None
        ss.error = (str(e), traceback.format_exc())
    ss.run_key = run_key
    bar.empty()

if ss.error:
    st.error(f"The pipeline could not finish: {ss.error[0]}")
    with st.expander("Technical details"):
        st.code(ss.error[1])
    st.stop()

R: AutoResult = ss.result
if R is None:
    st.stop()
X = R.export

# ---------------------------------------------------------------------------
# 3 · Results
# ---------------------------------------------------------------------------
n_cells_fixed = int(R.cleaning.actions["cells"].sum()) if len(R.cleaning.actions) else 0

st.markdown('<div class="big-download">', unsafe_allow_html=True)
d1, d2 = st.columns([3, 1.3])
with d1:
    st.markdown("### ✅ Done — your cleaned dataset is ready")
    st.markdown(f"**{X.rows_out:,} rows × {len(X.cols_out)} columns** · {n_cells_fixed:,} cells fixed · "
                f"{len(R.excluded)} column(s) set aside · preprocessing: *{R.prep.describe()}*")
with d2:
    st.download_button("⬇️ Download cleaned CSV", data=file_bytes(X.cleaned_path), file_name="cleaned_dataset.csv",
                       mime="text/csv", key="dl_main")
st.markdown("</div>", unsafe_allow_html=True)

for w in R.warnings:
    st.warning(w)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Rows", f"{R.n_rows_total:,} → {X.rows_out:,}", help="Duplicates and rows with a missing target are removed.")
k2.metric("Columns", f"{R.n_cols_raw} → {len(X.cols_out)}")
k3.metric("Cells fixed", f"{n_cells_fixed:,}")
k4.metric("Task", R.task.capitalize())
if R.models is not None and R.models.best_name:
    k5.metric(f"Best model ({R.models.metric_name})", f"{R.models.best_score:.3f}", R.models.best_name, delta_color="off")
elif R.unsup is not None and R.unsup.best_algorithm:
    k5.metric("Best clustering", R.unsup.best_algorithm, f"k = {R.unsup.best_k}", delta_color="off")
else:
    k5.metric("Models", "—")

is_sup = R.task != "unsupervised"
tab_names = ["📋 Overview", "🧹 Cleaning", "📊 Data quality", "🛡️ Leakage & columns", "⚗️ Preprocessing",
             "🤖 Models" if is_sup else "🧩 Unsupervised", "📦 Final dataset"]
tabs = st.tabs(tab_names)

# ----------------------------------------------------------------- overview
with tabs[0]:
    st.subheader("What happened, step by step")
    stages = [
        f"**Loaded** {R.n_rows_total:,} rows × {R.n_cols_raw} columns"
        + (f" (rules learned on a {R.sample_rows:,}-row sample, applied to all rows)" if R.used_sampling else ""),
        (f"**Target:** `{R.target_clean}` — {R.target_reason}  →  **{R.task}**" if R.target_clean
         else f"**No target column** — {R.target_reason}"),
        f"**Cleaned** {n_cells_fixed:,} cells across {len(R.cleaning.column_table)} columns "
        f"({X.duplicates_removed + X.key_duplicates_removed:,} duplicate rows removed)",
        f"**Leakage scan:** {len(R.leakage)} column(s) flagged, "
        f"{sum(1 for f in R.leakage if f.risk_score >= R.settings.auto_exclude_risk)} auto-excluded",
        f"**Features used:** {len(R.numeric_cols)} numeric + {len(R.categorical_cols)} categorical",
        f"**Preprocessing chosen:** {R.prep.describe()}",
    ]
    if R.models is not None and R.models.best_name:
        stages.append(f"**Best model:** {R.models.best_name} — {R.models.metric_name} {R.models.best_score:.4f} "
                      f"({R.models.cv_folds}-fold CV, {len(R.models.table)} models compared)")
    if R.unsup is not None and R.unsup.best_algorithm:
        stages.append(f"**Unsupervised:** best clustering = {R.unsup.best_algorithm} with k = {R.unsup.best_k}; "
                      f"{int(R.unsup.anomalies.iloc[-1]['flagged rows'])} consensus anomalies")
    stages.append(f"**Exported** {X.rows_out:,} rows in {R.timings.get('total', 0):.1f}s total")
    for s in stages:
        st.markdown(f'<div class="stage"><span class="dot"></span><span>{s}</span></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        show_fig(ch.fig_type_donut({c.name: c.kind for c in R.clean_profile.columns.values()}))
    with c2:
        if R.class_counts:
            show_fig(ch.fig_class_balance(R.class_counts))
        else:
            show_fig(ch.fig_timings(R.timings))

# ----------------------------------------------------------------- cleaning
with tabs[1]:
    st.subheader("Data cleaning & normalisation (runs before profiling)")
    st.caption("Every rule is learned from your data — nothing is hard-coded to a dataset.")
    show_fig(ch.fig_cleaning_actions(R.cleaning.actions))
    st.markdown("**Column by column**")
    show_df(R.cleaning.column_table, height=420)
    if len(R.cleaning.merges):
        st.markdown("**Category spellings merged** (variant → canonical label)")
        cols_with = sorted(R.cleaning.merges["column"].unique())
        pick = st.selectbox("Column", ["all"] + cols_with, key="merge_col")
        m = R.cleaning.merges if pick == "all" else R.cleaning.merges[R.cleaning.merges["column"] == pick]
        show_df(m, height=320)
    if len(R.cleaning.dropped):
        st.markdown("**Columns dropped by the cleaner**")
        show_df(R.cleaning.dropped)

# ----------------------------------------------------------------- data quality
with tabs[2]:
    st.subheader("Data quality report")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Duplicate rows (raw)", f"{R.dup_raw:,}", help="Counted ignoring the row-counter column")
    q2.metric("Missing cells (raw, blanks only)", f"{R.raw_profile.total_missing_cells:,}")
    q3.metric("Missing cells (real)", f"{R.clean_profile.total_missing_cells:,}",
              help="After turning NA / ? / - / null / unparseable text into real missing values")
    q4.metric("Missing after imputation", f"{R.final_profile.total_missing_cells:,}")
    show_fig(ch.fig_missing_compare(R.missing_compare))
    summary = R.clean_profile.to_summary_frame()
    st.markdown("**Cleaned column summary**")
    show_df(summary, height=380, column_config={
        "missing %": st.column_config.ProgressColumn("missing %", min_value=0, max_value=100, format="%.1f")})
    v1, v2 = st.columns(2)
    with v1:
        show_fig(ch.fig_missing_bar(summary))
    with v2:
        show_fig(ch.fig_skew(summary))
    v3, v4 = st.columns(2)
    with v3:
        show_fig(ch.fig_outliers(summary))
    with v4:
        show_fig(ch.fig_corr(R.correlation))
    st.caption("Outliers use the 1.5×IQR rule; skewness is the Fisher–Pearson coefficient. "
               + ("Statistics come from a representative sample." if R.used_sampling else ""))

# ----------------------------------------------------------------- leakage
with tabs[3]:
    st.subheader("Leakage & low-value column detection (automatic)")
    if not is_sup:
        st.info("Leakage detection needs a target column. Nothing to check in unsupervised mode.")
    elif not R.leakage:
        st.success("No suspicious columns detected.")
    else:
        fdf = flags_to_frame(R.leakage)
        st.warning(f"{len(fdf)} column(s) flagged; those with risk ≥ {R.settings.auto_exclude_risk:.2f} "
                   "were excluded automatically (change the threshold in the sidebar).")
        show_fig(ch.fig_leakage(fdf, R.settings.auto_exclude_risk))
        show_df(fdf)
    st.markdown("**Every column that was NOT used as a feature, and why**")
    if len(R.excluded):
        show_df(R.excluded)
    else:
        st.write("None — every column was used.")
    pills(R.numeric_cols[:30], "teal")
    pills(R.categorical_cols[:30], "pink")
    st.caption("🟢 numeric features · 🟣 categorical features")

# ----------------------------------------------------------------- preprocessing
with tabs[4]:
    st.subheader("Preprocessing ablation study (automatic)")
    if R.ablation is None:
        st.info("The ablation study needs a target column, so it is skipped in unsupervised mode "
                "(median imputation, scaling and one-hot encoding are used for the clustering).")
    else:
        from core.transform import results_to_frame
        rdf = results_to_frame(R.ablation)
        st.caption(f"Each candidate was scored with a {R.ablation.probe_model} probe using {R.ablation.cv_folds}-fold "
                   f"cross-validation on {R.ablation.n_rows:,} rows. A change is adopted only if it beats the baseline "
                   "by more than the noise floor.")
        show_fig(ch.fig_ablation(rdf, R.ablation.metric_name))
        show_df(rdf)
        st.success(f"**Selected:** {R.ablation.best.name} — {R.ablation.best.cfg.describe()}")
        st.write(R.ablation.explanation)

# ----------------------------------------------------------------- models / unsupervised
with tabs[5]:
    if is_sup:
        M = R.models
        st.subheader("Model comparison (automatic)")
        if M is None or M.best_name is None:
            st.warning("No model could be evaluated. " + " ".join(M.notes if M else []))
        else:
            st.caption(f"{len(M.table)} models · {M.cv_folds}-fold cross-validation on {M.n_rows:,} rows · "
                       f"preprocessing: {R.prep.describe()}")
            for n in M.notes:
                st.info(n)
            show_fig(ch.fig_leaderboard(M.table, M.metric_name))
            tbl = M.table.copy()
            if "error" in tbl.columns and not tbl["error"].astype(str).str.len().gt(0).any():
                tbl = tbl.drop(columns=["error"])
            tbl = tbl.round(4)
            show_df(tbl, column_config={M.metric_name: progress_col(M.metric_name, tbl[M.metric_name].dropna())})
            g1, g2 = st.columns(2)
            with g1:
                show_fig(ch.fig_time_vs_score(M.table, M.metric_name))
            with g2:
                if M.importance is not None:
                    show_fig(ch.fig_importance(M.importance))
            h1, h2 = st.columns(2)
            with h1:
                if M.confusion is not None:
                    show_fig(ch.fig_confusion(M.confusion, M.classes))
                elif M.pred_vs_actual is not None:
                    show_fig(ch.fig_pred_vs_actual(M.pred_vs_actual))
            with h2:
                if M.roc is not None:
                    show_fig(ch.fig_roc(M.roc["fpr"], M.roc["tpr"], M.roc["auc"]))
    else:
        U = R.unsup
        st.subheader("Unsupervised analysis (no target column)")
        if U is None or U.algorithms.empty:
            st.warning("Not enough usable data for unsupervised analysis.")
        else:
            for n in U.notes:
                st.info(n)
            u1, u2 = st.columns(2)
            with u1:
                show_fig(ch.fig_kscores(U.k_scores))
            with u2:
                show_fig(ch.fig_pca_variance(U.pca_variance))
            st.markdown(f"**Clustering algorithms** — best: **{U.best_algorithm}** (k = {U.best_k})")
            show_df(U.algorithms.round(3))
            show_fig(ch.fig_pca_scatter(U.projection))
            u3, u4 = st.columns(2)
            with u3:
                show_fig(ch.fig_cluster_heatmap(U.cluster_profile))
            with u4:
                show_fig(ch.fig_cluster_sizes(U.cluster_sizes))
            st.markdown("**Anomaly detection**")
            u5, u6 = st.columns(2)
            with u5:
                show_df(U.anomalies)
            with u6:
                show_fig(ch.fig_anomaly_hist(U.projection))

# ----------------------------------------------------------------- final dataset
with tabs[6]:
    st.subheader("Final dataset & reproducible pipeline")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.download_button("⬇️ cleaned_dataset.csv", data=file_bytes(X.cleaned_path), file_name="cleaned_dataset.csv",
                           mime="text/csv", key="dl_clean")
        st.caption("Tidy, imputed, human-readable.")
    with f2:
        if X.ml_path and os.path.exists(X.ml_path):
            st.download_button("⬇️ ml_ready_dataset.csv", data=file_bytes(X.ml_path), file_name="ml_ready_dataset.csv",
                               mime="text/csv", key="dl_ml")
            st.caption("Fully numeric: one-hot / frequency encoded and scaled.")
    with f3:
        if X.report_path and os.path.exists(X.report_path):
            st.download_button("⬇️ cleaning_report.md", data=file_bytes(X.report_path), file_name="cleaning_report.md",
                               mime="text/markdown", key="dl_report")
            st.caption("Everything that was changed and why.")
    g1, g2 = st.columns(2)
    with g1:
        if X.script_path and os.path.exists(X.script_path):
            st.download_button("⬇️ preprocessing_pipeline.py", data=file_bytes(X.script_path),
                               file_name="preprocessing_pipeline.py", mime="text/x-python", key="dl_script")
    with g2:
        if X.bundle_path and os.path.exists(X.bundle_path):
            st.download_button("⬇️ autoprep_pipeline.joblib", data=file_bytes(X.bundle_path),
                               file_name="autoprep_pipeline.joblib", mime="application/octet-stream", key="dl_bundle")
    st.markdown("**Preview of the cleaned dataset** (first 300 rows)")
    show_df(R.preview_clean, height=360)
    st.markdown("**Row accounting (full dataset)**")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Rows in", f"{X.rows_in:,}")
    a2.metric("Duplicates removed", f"{X.duplicates_removed + X.key_duplicates_removed:,}")
    a3.metric("Missing target removed", f"{X.missing_target_removed:,}")
    a4.metric("Rows out", f"{X.rows_out:,}")
    st.markdown("**Summary of decisions**")
    st.markdown(
        f"- **Preprocessing:** {R.prep.describe()}\n"
        f"- **Excluded columns:** {', '.join('`%s`' % c for c in R.excluded['column']) if len(R.excluded) else 'none'}\n"
        + (f"- **Best model:** {R.models.best_name} ({R.models.metric_name} {R.models.best_score:.4f})\n"
           if R.models is not None and R.models.best_name else "")
    )
    if R.pipeline_code:
        with st.expander("Reproducible preprocessing code"):
            st.code(R.pipeline_code, language="python")
