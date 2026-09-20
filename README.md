<div align="center">

# AutoPrep Studio

**Upload a messy dataset. Get a clean one back, cleaned, leakage-checked, preprocessed and model-tested automatically.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-models-F7931E?logo=scikitlearn&logoColor=white)
![Polars](https://img.shields.io/badge/Polars-large%20files-CD792C)
![Plotly](https://img.shields.io/badge/Plotly-charts-3F4F75?logo=plotly&logoColor=white)

</div>

---

## Overview

AutoPrep Studio is a Streamlit app for data preparation. You upload a raw file and the whole pipeline runs by itself:
cleaning, profiling, leakage detection, preprocessing search, model comparison and export.
**The only manual step is clicking _Download cleaned CSV_.**

There are no LLM or API calls. Every decision is measured with pandas and scikit-learn on your own machine, and every change
is listed in a report so you can see what was done and why.

## Table of contents

- [Features](#features)
- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Using the app](#using-the-app)
- [The cleaning step in detail](#the-cleaning-step-in-detail)
- [Models](#models)
- [Outputs](#outputs)
- [Project structure](#project-structure)
- [Demo dataset](#demo-dataset)
- [Testing](#testing)
- [Author](#author)

## Features

| Feature | Description |
|---|---|
| **Smart cleaning** | Fixes currency and units, mixed date formats, category spellings, hidden missing values and impossible values before any analysis. |
| **Fully automatic** | Upload once and every stage runs. No per-stage buttons. |
| **Leakage detection** | Flags and excludes ID-like columns, near-copies of the target, and columns that almost determine it. |
| **Evidence-based preprocessing** | An ablation study keeps only the choices that measurably improve cross-validated scores. |
| **Many models** | Supervised (classification and regression) and unsupervised (clustering, anomaly detection, PCA). |
| **Interactive dashboards** | Plotly charts and a dark theme. |
| **Reproducible export** | Cleaned CSV, ML-ready CSV, reusable pipeline script and a cleaning report. |
| **Large file support** | Learns rules on a sample, then streams the full file in chunks with Polars. |

## How it works

```mermaid
flowchart LR
    A[Upload file] --> B[Load and sample]
    B --> C[Detect target]
    C --> D[Clean and normalise]
    D --> E[Profile data quality]
    E --> F[Leakage scan]
    F --> G[Preprocessing ablation]
    G --> H[Model comparison]
    H --> I[Export full dataset]
    I --> J[Download]
```

| # | Step | Module | What happens |
|---|------|--------|--------------|
| 1 | Load | `core/data_io.py` | CSV, TSV, Excel, JSON/NDJSON and Parquet. Delimited text is read as text so nothing is mistyped. |
| 2 | Target detection | `core/autopilot.py` | Picks the target from column names or position. If none is found, unsupervised mode runs. |
| 3 | Cleaning | `core/cleaning.py` | Rules are learned from the data, then applied to every row. |
| 4 | Profiling | `core/profiling.py` | Missing values, duplicates, outliers and skewness, before and after cleaning. |
| 5 | Leakage scan | `core/leakage.py` | Risky columns are excluded automatically. |
| 6 | Ablation study | `core/transform.py` | Tests missing-value handling, outlier clipping, log transform and encoding. |
| 7 | Models | `core/models.py` | Cross-validated comparison on identical folds. |
| 8 | Export | `core/apply_final.py` | Streams the complete dataset through the fitted pipeline. |

## Quick start

**Requirements:** Python 3.10 or newer.

```bash
git clone https://github.com/DristiLaskar/Autoprep_Studio.git
cd Autoprep_Studio

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

Optional: install `xgboost` and `lightgbm` and they are added to the model lineup automatically.

```bash
pip install xgboost lightgbm
```

## Using the app

1. Drag a file onto the uploader, or click **Try the messy demo dataset**.
2. Wait for the progress bar. Every stage runs automatically.
3. Click **Download cleaned CSV**.

Results are shown in tabs:

| Tab | Contents |
|---|---|
| Overview | Step-by-step summary of what happened |
| Cleaning | What was fixed, column by column, and every merged category spelling |
| Data quality | Missing values before and after cleaning, skewness, outliers, correlations |
| Leakage & columns | Flagged columns and every column that was not used as a feature, with the reason |
| Preprocessing | Ablation results and the selected configuration |
| Models / Unsupervised | Leaderboard, confusion matrix, ROC, feature importance, or clusters and anomalies |
| Final dataset | All downloads, a preview and row accounting |

**Optional settings** (sidebar, applied when you press *Apply & re-run*): target column, task type, model search depth
(`fast` / `balanced` / `thorough`), keeping ID/contact/raw-date columns, missing-value cut-off, and leakage exclusion threshold.

## The cleaning step in detail

Rules are learned from your data on a sample of up to 100,000 rows and applied to every row.

| Problem | What AutoPrep does |
|---|---|
| Messy column names | Converts to `snake_case` (`Tenure (months)` becomes `tenure_months`) |
| Hidden missing values | `NA`, `N/A`, `?`, `-`, `null`, `missing` and similar tokens become real missing values |
| Numbers stored as text | Strips currency (`$1,234.50`, `USD 23`, `18.52$`, `(45)`), handles US and European separators, `%`, number words (`forty`) and `8/10` scores |
| Mixed units | `512 MB`, `0.8 TB` and similar values are converted to one unit per column (the unit in the column name wins, e.g. `data_usage_gb`) |
| Impossible values | Negative ages, `999` sentinels, x100 typos and out-of-scale ratings become missing. Values are cut only where there is a clear gap, so genuine heavy tails survive. |
| Boolean variants | `Yes` / `Y` / `true` / `1` become 1 and `No` / `N` / `false` / `0` become 0. Target synonyms such as `Churned` / `Retained` are mapped too. |
| Mixed date formats | Parsed per format, with day/month order learned from unambiguous values. Adds year, month, weekday and epoch-day features. |
| Category spellings | Merges case, spacing, accent, typo and abbreviation variants (`Londn` becomes `London`, `Std` becomes `Standard`) |
| Junk columns | Drops index artifacts (`Unnamed: 0`), empty and constant columns. IDs, emails and phones are recognised and never used as features. |
| Duplicates | Removes duplicate rows, including across chunk boundaries, and rows with a missing target |

## Models

### Supervised

The task (classification or regression) is detected automatically.

| Classification | Regression |
|---|---|
| Logistic Regression | Linear Regression |
| Random Forest | Ridge |
| Extra Trees | Lasso |
| Gradient Boosting | ElasticNet |
| HistGradient Boosting | Random Forest |
| Decision Tree | Extra Trees |
| k-Nearest Neighbors | Gradient Boosting |
| Naive Bayes | HistGradient Boosting |
| SVM | Decision Tree |
| AdaBoost | k-Nearest Neighbors |
| Neural Network (MLP) | SVR |
| XGBoost / LightGBM *(optional)* | AdaBoost, Neural Network (MLP), XGBoost / LightGBM *(optional)* |

All models are cross-validated on identical folds through the same preprocessing. Scaling is applied only to models that need it.
Diagnostics for the best model include a leaderboard, an accuracy-versus-speed chart, a confusion matrix and ROC curve
(or predicted-versus-actual for regression), and permutation feature importance.

### Unsupervised

When no target exists:

- **Clustering:** KMeans (automatic *k*), MiniBatch KMeans, Gaussian Mixture, BIRCH, DBSCAN and Agglomerative (Ward), scored by
  silhouette, Davies-Bouldin and Calinski-Harabasz
- **Anomaly detection:** Isolation Forest, Local Outlier Factor and One-Class SVM, plus a consensus of the three
- **Dimensionality reduction:** PCA variance plot and a 2-D cluster map

## Outputs

| File | Description |
|---|---|
| `cleaned_dataset.csv` | Tidy, imputed, human-readable. **The main deliverable.** |
| `ml_ready_dataset.csv` | Fully numeric (encoded and scaled) |
| `preprocessing_pipeline.py` and `autoprep_pipeline.joblib` | Re-apply the same pipeline to new data, or train the winning model. Run from the project folder, or set `AUTOPREP_HOME`. |
| `cleaning_report.md` | Everything that was changed, and why |

## Project structure

```text
Autoprep_Studio/
├── app.py                 # Streamlit UI (fully automatic)
├── requirements.txt
├── .streamlit/
│   └── config.toml        # dark theme
├── core/
│   ├── data_io.py         # loading, sampling, chunk streaming (Polars)
│   ├── cleaning.py        # cleaning and normalisation
│   ├── profiling.py       # data-quality profile
│   ├── leakage.py         # leakage and low-value column detection
│   ├── transform.py       # preprocessors and ablation study
│   ├── models.py          # supervised model zoo and unsupervised analysis
│   ├── autopilot.py       # orchestration of the automatic pipeline
│   ├── apply_final.py     # full-dataset export, pipeline code, report
│   ├── charts.py          # Plotly figures
│   └── theme.py           # palette and CSS
├── sample_data/           # demo datasets
└── tests/                 # pytest suite
```

## Demo dataset

`sample_data/messy_churn_dataset.csv` is a synthetic churn dataset (2,126 rows, 24 columns) with deliberately injected problems:
hidden missing tokens, mixed date formats, currency and unit noise, category typos, impossible values, duplicate rows, a leakage
column (`churn_reason`) and junk columns. `messy_churn_dataset_ANSWER_KEY_clean.csv` holds the true clean values so the cleaner can be
checked against ground truth.

On this dataset the cleaner recovers 98-100% of recognised cells correctly, and `churn_reason` is excluded automatically as leakage.
Results are for this synthetic data only and will differ on real data.

## Testing

```bash
pip install pytest
pytest -q tests
```

The suite covers the cleaning rules, comparison against the answer key, the end-to-end pipeline, the unsupervised path and
chunked export.

## Author

Built by **Dristi Laskar**.
