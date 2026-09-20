"""
charts.py
---------
Colourful, dark-theme-friendly Plotly figures. Every function takes plain DataFrames /
arrays and returns a plotly Figure (no Streamlit code in here).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .theme import DIV, GRID, MUTED, PALETTE, RISK, SEQ, TEXT


def style(fig: go.Figure, title: Optional[str] = None, height: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Inter, Segoe UI, Arial, sans-serif"), colorway=PALETTE,
        margin=dict(l=10, r=10, t=56 if title else 20, b=10), height=height, showlegend=legend,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=16)))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def _empty(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=14, color=MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, height=220, legend=False)


# ------------------------------------------------------------------ data quality
def fig_missing_compare(df: pd.DataFrame, top: int = 22) -> go.Figure:
    d = df.copy()
    d = d[(d["before cleaning %"] > 0) | (d["after cleaning %"] > 0)]
    if d.empty:
        return _empty("No missing values detected")
    d = d.sort_values("before cleaning %", ascending=False).head(top).iloc[::-1]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Before cleaning (only blanks counted)", y=d["column"], x=d["before cleaning %"],
                         orientation="h", marker_color=PALETTE[2]))
    fig.add_trace(go.Bar(name="After cleaning (hidden NA / ? / - found)", y=d["column"], x=d["after cleaning %"],
                         orientation="h", marker_color=PALETTE[1]))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="missing %")
    return style(fig, "Missing values — before vs after cleaning", height=max(320, 26 * len(d) + 120))


def fig_type_donut(kinds: Dict[str, str]) -> go.Figure:
    if not kinds:
        return _empty("No columns")
    counts = pd.Series(kinds).value_counts().rename_axis("type").reset_index(name="columns")
    fig = px.pie(counts, names="type", values="columns", hole=0.62, color_discrete_sequence=PALETTE)
    fig.update_traces(textinfo="label+value", marker=dict(line=dict(color="#0B1020", width=2)))
    return style(fig, "Detected column types", height=340)


def fig_cleaning_actions(actions: pd.DataFrame) -> go.Figure:
    if actions is None or actions.empty:
        return _empty("Nothing needed fixing")
    d = actions.sort_values("cells")
    fig = px.bar(d, x="cells", y="action", orientation="h", color="cells", color_continuous_scale=SEQ, text="cells")
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_coloraxes(showscale=False)
    fig.update_yaxes(title_text="")
    return style(fig, "What the cleaner fixed (cells)", height=max(300, 42 * len(d) + 100), legend=False)


def fig_missing_bar(summary: pd.DataFrame) -> go.Figure:
    d = summary[summary["missing %"] > 0].sort_values("missing %", ascending=False)
    if d.empty:
        return _empty("No missing values after cleaning")
    fig = px.bar(d, x="column", y="missing %", color="missing %", color_continuous_scale=RISK, range_color=[0, 100])
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(tickangle=-40, title_text="")
    return style(fig, "Missing % by column (cleaned data)", legend=False)


def fig_skew(summary: pd.DataFrame) -> go.Figure:
    d = summary.dropna(subset=["skewness"])
    if d.empty:
        return _empty("No numeric columns to show skewness for")
    d = d.reindex(d["skewness"].abs().sort_values(ascending=False).index)
    fig = px.bar(d, x="column", y="skewness", color="skewness", color_continuous_scale=DIV, color_continuous_midpoint=0)
    fig.update_coloraxes(showscale=False)
    for y in (1, -1):
        fig.add_hline(y=y, line_dash="dash", line_color=MUTED)
    fig.update_xaxes(tickangle=-40, title_text="")
    return style(fig, "Skewness (dashed = ±1, log-transform candidates)", legend=False)


def fig_outliers(summary: pd.DataFrame) -> go.Figure:
    d = summary.dropna(subset=["outlier %"]).sort_values("outlier %", ascending=False)
    if d.empty:
        return _empty("No numeric columns to show outliers for")
    fig = px.bar(d, x="column", y="outlier %", color="outlier %", color_continuous_scale=SEQ)
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(tickangle=-40, title_text="")
    return style(fig, "IQR outliers by column (%)", legend=False)


def fig_corr(corr: pd.DataFrame) -> go.Figure:
    if corr is None or corr.empty:
        return _empty("Not enough numeric columns for a correlation matrix")
    z = corr.values
    fig = go.Figure(go.Heatmap(z=z, x=list(corr.columns), y=list(corr.index), colorscale=DIV, zmin=-1, zmax=1,
                               text=np.round(z, 2), texttemplate="%{text}", xgap=2, ygap=2,
                               colorbar=dict(thickness=12)))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-40)
    return style(fig, "Correlation matrix (cleaned numeric columns)", height=max(380, 26 * len(corr) + 160), legend=False)


def fig_hist(df: pd.DataFrame, col: str) -> go.Figure:
    fig = px.histogram(df, x=col, nbins=40, marginal="box", color_discrete_sequence=[PALETTE[0]])
    fig.update_traces(marker_line_width=0)
    return style(fig, f"Distribution of {col}", legend=False)


def fig_class_balance(counts: Dict[str, int]) -> go.Figure:
    if not counts:
        return _empty("No classes")
    d = pd.DataFrame({"class": list(counts.keys()), "rows": list(counts.values())})
    fig = px.pie(d, names="class", values="rows", hole=0.6, color_discrete_sequence=PALETTE)
    fig.update_traces(textinfo="label+percent", marker=dict(line=dict(color="#0B1020", width=2)))
    return style(fig, "Target class balance", height=340)


# ------------------------------------------------------------------ leakage / ablation
def fig_leakage(flags_df: pd.DataFrame, threshold: float) -> go.Figure:
    if flags_df is None or flags_df.empty:
        return _empty("No suspicious columns")
    d = flags_df.sort_values("risk score")
    fig = px.bar(d, x="risk score", y="column", orientation="h", color="risk score", color_continuous_scale=RISK,
                 range_color=[0, 1], text="risk score")
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_coloraxes(showscale=False)
    fig.add_vline(x=threshold, line_dash="dash", line_color=TEXT, annotation_text="auto-exclude", annotation_position="top")
    fig.update_xaxes(range=[0, 1.1])
    fig.update_yaxes(title_text="")
    return style(fig, "Leakage / low-value risk score", height=max(260, 44 * len(d) + 100), legend=False)


def fig_ablation(results: pd.DataFrame, metric: str) -> go.Figure:
    d = results.dropna(subset=[metric]).copy()
    if d.empty:
        return _empty("No candidate could be scored")
    colors = []
    for i, r in d.reset_index(drop=True).iterrows():
        if r.get("selected") == "✅":
            colors.append("#2EC4B6")
        elif str(r["Candidate"]).startswith("Baseline"):
            colors.append("#4DA3FF")
        elif "reference only" in str(r.get("note", "")):
            colors.append("#5A6390")
        else:
            colors.append(PALETTE[i % len(PALETTE)])
    fig = go.Figure(go.Bar(x=d["Candidate"], y=d[metric], marker_color=colors, text=d[metric].round(4),
                           textposition="outside", cliponaxis=False))
    lo = float(d[metric].min())
    hi = float(d[metric].max())
    pad = max((hi - lo) * 0.6, 0.02)
    fig.update_yaxes(range=[max(0, lo - pad), min(1.0, hi + pad) if hi <= 1 else hi + pad], title_text=metric)
    fig.update_xaxes(tickangle=-20, title_text="")
    return style(fig, f"{metric} by preprocessing candidate (green = selected, blue = baseline)", legend=False)


# ------------------------------------------------------------------ supervised models
def fig_leaderboard(table: pd.DataFrame, metric: str) -> go.Figure:
    d = table.dropna(subset=[metric]).copy()
    if d.empty:
        return _empty("No model could be evaluated")
    d = d.sort_values(metric, ascending=True)
    d["label"] = d[metric].round(4)
    fig = px.bar(d, x=metric, y="Model", color="Family", orientation="h", error_x="± std",
                 color_discrete_sequence=PALETTE, text="label")
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(title_text="")
    lo, hi = float(d[metric].min()), float(d[metric].max())
    fig.update_xaxes(range=[min(0, lo) if lo < 0 else max(0, lo - (hi - lo) * 1.2 - 0.05), hi + (hi - lo) * 0.25 + 0.03])
    return style(fig, f"Model leaderboard — {metric} (cross-validated, ± std)", height=max(340, 40 * len(d) + 120))


def fig_time_vs_score(table: pd.DataFrame, metric: str) -> go.Figure:
    d = table.dropna(subset=[metric, "seconds"]).copy()
    if d.empty:
        return _empty("No timing data")
    d["seconds"] = d["seconds"].clip(lower=0.05)
    fig = px.scatter(d, x="seconds", y=metric, color="Family", hover_name="Model", text="Model",
                     color_discrete_sequence=PALETTE, log_x=True)
    fig.update_traces(marker=dict(size=15, line=dict(width=1, color="#0B1020")), textposition="top center")
    fig.update_xaxes(title_text="training time per CV run (s, log scale)")
    return style(fig, "Accuracy vs speed trade-off", height=380)


def fig_confusion(cm: np.ndarray, labels: Sequence[str]) -> go.Figure:
    cm = np.asarray(cm)
    pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    text = [[f"{cm[i, j]}<br>{pct[i, j]:.0%}" for j in range(cm.shape[1])] for i in range(cm.shape[0])]
    lab = [str(x) for x in labels] if labels else [str(i) for i in range(cm.shape[0])]
    fig = go.Figure(go.Heatmap(z=pct, x=lab, y=lab, colorscale=SEQ, zmin=0, zmax=1, text=text, texttemplate="%{text}",
                               xgap=3, ygap=3, showscale=False))
    fig.update_yaxes(autorange="reversed", title_text="actual")
    fig.update_xaxes(title_text="predicted")
    return style(fig, "Confusion matrix (cross-validated; % of each actual class)", height=380, legend=False)


def fig_roc(fpr: np.ndarray, tpr: np.ndarray, auc_value: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {auc_value:.3f})", fill="tozeroy",
                             line=dict(color=PALETTE[1], width=3), fillcolor="rgba(0,212,180,0.18)"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance", line=dict(color=MUTED, dash="dash")))
    fig.update_xaxes(title_text="false positive rate")
    fig.update_yaxes(title_text="true positive rate")
    return style(fig, "ROC curve of the best model", height=380)


def fig_importance(imp: pd.DataFrame) -> go.Figure:
    if imp is None or imp.empty:
        return _empty("Feature importance unavailable")
    d = imp.sort_values("importance", ascending=True)
    fig = px.bar(d, x="importance", y="feature", orientation="h", color="importance", color_continuous_scale=SEQ,
                 error_x="std")
    fig.update_coloraxes(showscale=False)
    fig.update_yaxes(title_text="")
    fig.update_xaxes(title_text="permutation importance (score drop when shuffled)")
    return style(fig, "What drives the predictions", height=max(320, 30 * len(d) + 120), legend=False)


def fig_pred_vs_actual(df: pd.DataFrame) -> go.Figure:
    d = df.sample(min(len(df), 3000), random_state=1)
    fig = px.scatter(d, x="actual", y="predicted", opacity=0.55, color_discrete_sequence=[PALETTE[0]])
    lo, hi = float(min(d["actual"].min(), d["predicted"].min())), float(max(d["actual"].max(), d["predicted"].max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect", line=dict(color=PALETTE[1], dash="dash")))
    return style(fig, "Predicted vs actual (cross-validated)", height=380, legend=False)


# ------------------------------------------------------------------ unsupervised
def fig_kscores(k_df: pd.DataFrame) -> go.Figure:
    if k_df is None or k_df.empty:
        return _empty("No k-scores")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=k_df["k"], y=k_df["silhouette"], name="silhouette (higher = better)", mode="lines+markers",
                             line=dict(color=PALETTE[1], width=3), marker=dict(size=9)), secondary_y=False)
    fig.add_trace(go.Scatter(x=k_df["k"], y=k_df["inertia"], name="inertia (elbow)", mode="lines+markers",
                             line=dict(color=PALETTE[2], width=3, dash="dot"), marker=dict(size=9)), secondary_y=True)
    fig.update_xaxes(title_text="number of clusters k", dtick=1)
    fig.update_yaxes(title_text="silhouette", secondary_y=False)
    fig.update_yaxes(title_text="inertia", secondary_y=True, showgrid=False)
    return style(fig, "Choosing the number of clusters", height=360)


def fig_pca_scatter(proj: pd.DataFrame) -> go.Figure:
    d = proj.sample(min(len(proj), 4000), random_state=1) if len(proj) > 4000 else proj
    fig = px.scatter(d, x="PC1", y="PC2", color="cluster", symbol="anomaly",
                     symbol_map={"normal": "circle", "anomaly": "x"}, opacity=0.75,
                     color_discrete_sequence=PALETTE, hover_data=["anomaly score"])
    fig.update_traces(marker=dict(size=8))
    return style(fig, "Clusters in PCA space (× = consensus anomalies)", height=460)


def fig_pca_variance(var: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=var["component"], y=var["explained"], name="explained", marker_color=PALETTE[0]))
    fig.add_trace(go.Scatter(x=var["component"], y=var["cumulative"], name="cumulative", mode="lines+markers",
                             line=dict(color=PALETTE[3], width=3)))
    fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
    return style(fig, "PCA — variance explained", height=340)


def fig_cluster_heatmap(profile: pd.DataFrame) -> go.Figure:
    if profile is None or profile.empty:
        return _empty("No numeric columns to profile")
    z = profile.values
    fig = go.Figure(go.Heatmap(z=z, x=list(profile.columns), y=list(profile.index), colorscale=DIV, zmid=0,
                               text=np.round(z, 2), texttemplate="%{text}", xgap=2, ygap=2, colorbar=dict(thickness=12)))
    fig.update_xaxes(tickangle=-40)
    fig.update_yaxes(autorange="reversed")
    return style(fig, "Cluster profiles (mean z-score per feature)", height=max(300, 60 * len(profile) + 180), legend=False)


def fig_cluster_sizes(sizes: pd.DataFrame) -> go.Figure:
    fig = px.pie(sizes, names="cluster", values="rows", hole=0.6, color_discrete_sequence=PALETTE)
    fig.update_traces(textinfo="label+percent", marker=dict(line=dict(color="#0B1020", width=2)))
    return style(fig, "Cluster sizes", height=340)


def fig_anomaly_hist(proj: pd.DataFrame) -> go.Figure:
    fig = px.histogram(proj, x="anomaly score", color="anomaly", nbins=50, barmode="overlay", opacity=0.75,
                       color_discrete_map={"normal": PALETTE[1], "anomaly": PALETTE[2]})
    return style(fig, "Anomaly scores (Isolation Forest)", height=340)


def fig_timings(timings: Dict[str, float]) -> go.Figure:
    d = pd.DataFrame([{"stage": k, "seconds": v} for k, v in timings.items() if k != "total" and v > 0])
    if d.empty:
        return _empty("No timings")
    d = d.sort_values("seconds")
    d["label"] = d["seconds"].round(1)
    fig = px.bar(d, x="seconds", y="stage", orientation="h", color="stage",
                 color_discrete_sequence=PALETTE, text="label")
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(title_text="")
    return style(fig, "Where the time went", height=300, legend=False)
