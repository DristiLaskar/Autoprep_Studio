# 🧪 AutoPrep Studio — automatic data cleaning, leakage detection, preprocessing search & model comparison

Upload a messy file. AutoPrep Studio **cleans it, scans for leakage, picks the best preprocessing, compares a
whole lineup of ML models and exports the cleaned dataset — all by itself.** The only thing left for you to do is
click **Download cleaned CSV**.

No LLM / API calls anywhere: every decision is measured with pandas / scikit-learn on your own machine.

## Quick start

```bash
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then drop a file on the page (or press **Try the messy demo dataset**). That's it.

## What the pipeline does (automatically, in this order)

| # | Step | Module | What happens |
|---|------|--------|--------------|
| 1 | Load | `core/data_io.py` | CSV/TSV/text, Excel, JSON/NDJSON, Parquet. Delimited text is read as *text* so nothing is mis-typed; big files are streamed with Polars. |
| 2 | Target detection | `core/autopilot.py` | Picks the target from column names / position (`churn`, `target`, `label`, last column …). No obvious target → unsupervised mode. Override in the sidebar. |
| 3 | **Cleaning & normalisation** | `core/cleaning.py` | Runs **before** profiling — see below. |
| 4 | Profiling | `core/profiling.py` | Missing %, duplicates, outliers, skewness, before-vs-after cleaning. |
| 5 | Leakage scan | `core/leakage.py` | ID-like columns, near-copies of the target, and columns that (nearly) determine it. Risky columns are excluded automatically. |
| 6 | Preprocessing ablation | `core/transform.py` | Tries missing-value strategy, outlier clipping, log-transform, encoding; keeps only what *measurably* helps (3-fold CV, noise floor). |
| 7 | Models | `core/models.py` | Compares many models with cross-validation (below). |
| 8 | Export | `core/apply_final.py` | Streams the **complete** file through the fitted pipeline in chunks → `cleaned_dataset.csv`, `ml_ready_dataset.csv`, reusable pipeline + report. |

### The cleaning step (`core/cleaning.py`)
Rules are *learned from your data* (fit on a sample, applied to every row):

- tidy `snake_case` column names; hidden missing values (`NA`, `N/A`, `?`, `-`, `null`, `missing` …) → real NaN
- numbers stored as text: currency (`$1,234.50`, `USD 23`, `18.52$`, `(45)`), thousands/decimal formats (US & European), `%`, number words (`forty`), `8/10` scores
- units: `512 MB`, `0.8 TB`, `35 yrs` … converted to **one** unit per column (the unit in the column name wins, e.g. `data_usage_gb`)
- impossible values (negative ages, `999` sentinels, ×100 typos, ages > 120, out-of-scale ratings) → missing — cut only where there is a *clear gap* between plausible and absurd values, so genuine heavy tails survive
- Yes / Y / true / 1 / `TRUE` … → 1 and No / N / false / 0 … → 0 (target synonyms like `Churned` / `Retained` too)
- mixed date formats (`24.07.21`, `02-21-2022`, `28/10/2021`, `Sep 03, 2021`, ISO …): day/month order is learned from unambiguous values; adds year / month / weekday / epoch-day features
- category spellings: case, spacing, accents (`São Paulo`/`Sao Paulo`), typos (`Londn`), abbreviations (`Std` → `Standard`, `Bsc` → `Basic`)
- junk columns (index artifacts like `Unnamed: 0`, empty, constant), IDs, emails and phones are recognised and never used as features
- duplicate rows (also across chunks) and rows with a missing target are removed

Everything that was changed is listed in the **Cleaning** tab (and in `cleaning_report.md`).

## Models

**Supervised** (a target exists — classification or regression is detected automatically)

| Classification | Regression |
|---|---|
| Logistic Regression, Random Forest, Extra Trees, Gradient Boosting, HistGradient Boosting, Decision Tree, k-NN, Naive Bayes, SVM, AdaBoost, Neural Network (MLP), *XGBoost / LightGBM if installed* | Linear, Ridge, Lasso, ElasticNet, Random Forest, Extra Trees, Gradient Boosting, HistGradient Boosting, Decision Tree, k-NN, SVR, AdaBoost, Neural Network (MLP), *XGBoost / LightGBM if installed* |

All are cross-validated on identical folds through the same preprocessing (scaling only where the model needs it).
Outputs: leaderboard, accuracy-vs-speed chart, confusion matrix + ROC curve (or predicted-vs-actual), permutation feature importance.

**Unsupervised** (no target): KMeans with automatic *k*, MiniBatch KMeans, Gaussian Mixture, BIRCH, DBSCAN, Agglomerative (Ward) scored by silhouette / Davies-Bouldin / Calinski-Harabasz; anomaly detection with Isolation Forest, Local Outlier Factor and One-Class SVM (+ consensus); PCA variance and a 2-D cluster map.

Use the sidebar's **Model search depth** (fast / balanced / thorough) to trade time for coverage.

## Outputs (Final dataset tab)

- `cleaned_dataset.csv` — tidy, imputed, human-readable (the main deliverable)
- `ml_ready_dataset.csv` — fully numeric (encoded + scaled)
- `preprocessing_pipeline.py` + `autoprep_pipeline.joblib` — re-apply the *same* pipeline to new data / train the winning model (run from this project folder, or set `AUTOPREP_HOME`)
- `cleaning_report.md` — what changed and why

## Optional settings (sidebar → *Apply & re-run*)
Target column, task type, model search depth, whether to keep ID/contact/raw-date columns, missing-value cut-off, leakage auto-exclude threshold.

## Project layout
```
app.py                    Streamlit UI (fully automatic, colourful)
core/
  data_io.py              loading, sampling, chunk streaming (Polars)
  cleaning.py             cleaning & normalisation (fit on sample, apply to any chunk)
  profiling.py            data-quality profile
  leakage.py              leakage / low-value detection
  transform.py            preprocessors + ablation study
  models.py               supervised model zoo + unsupervised analysis
  autopilot.py            orchestration (the automatic pipeline)
  apply_final.py          full-dataset export, pipeline code, report
  charts.py / theme.py    Plotly figures, palette, CSS
sample_data/              demo files (incl. messy_churn_dataset.csv + its answer key)
tests/                    pytest suite (pandas/scikit-learn only)
```

## Tests
```bash
pip install pytest
pytest -q tests
```

## Notes & limitations
- Leakage detection is a heuristic aid. Columns that nearly determine the target are excluded automatically — check the
  **Leakage & columns** tab, and raise the threshold in the sidebar if a genuine strong predictor was dropped.
- Cleaning rules are learned from a sample of up to 100,000 rows; imputation values come from the same sample.
- Category fuzzy-merging never joins two large categories — only rare variants into much more frequent ones. Genuinely different
  aliases (e.g. *Bombay* vs *Mumbai*) can't be known and stay separate.
- Date/time-series forecasting and free-text (NLP) models are out of scope; text columns are set aside.
- "Drop rows with missing values" is scored for reference but never auto-selected, because it evaluates on a different set of rows.

## Why the old export step showed "Run the ablation study first"
The old `app.py` re-loaded the uploaded file — and reset every downstream result — on **every** Streamlit rerun, so clicking
*Apply to FULL dataset & export* wiped the ablation result before tab 5 rendered. The new app loads a file once per
upload and caches the pipeline result per (file, settings).
