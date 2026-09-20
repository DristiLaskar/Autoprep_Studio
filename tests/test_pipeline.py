"""End-to-end tests of the automatic pipeline on the bundled messy demo dataset."""
import os

import numpy as np
import pandas as pd
import pytest

import core.apply_final as af
from core.autopilot import NONE, Settings, run_autopilot
from helpers import PandasDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESSY = os.path.join(ROOT, "sample_data", "messy_churn_dataset.csv")
TRUTH = os.path.join(ROOT, "sample_data", "messy_churn_dataset_ANSWER_KEY_clean.csv")


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    ds = PandasDataset.from_csv(MESSY, "messy_churn_dataset.csv")
    return run_autopilot(ds, Settings(speed="fast"), out_dir=str(tmp_path_factory.mktemp("out")))


def test_target_task_and_leakage(result):
    assert result.target_clean == "churn" and result.task == "classification"
    assert "churn_reason" not in result.numeric_cols + result.categorical_cols
    assert any(f.column == "churn_reason" and f.risk_score >= 0.6 for f in result.leakage)


def test_export_is_clean(result):
    out = pd.read_csv(result.export.cleaned_path)
    assert len(out) == result.export.rows_out > 1900
    assert out.isna().sum().sum() == 0
    assert set(out["churn"].unique()) == {0, 1}
    assert all(c == c.lower() and " " not in c for c in out.columns)
    assert {"customer_id", "email", "phone", "churn_reason", "notes"}.isdisjoint(out.columns)
    ml = pd.read_csv(result.export.ml_path)
    assert ml.isna().sum().sum() == 0 and len(ml) == len(out)


def test_cleaning_matches_ground_truth(result):
    truth = pd.read_csv(TRUTH)
    raw = pd.read_csv(MESSY, dtype=str, keep_default_na=False).replace("", np.nan)
    clean = result.plan.cleaner.transform(raw).dropna(subset=["customer_id"]).drop_duplicates("customer_id")
    m = clean.merge(truth, on="customer_id", suffixes=("", "_t"))
    for col, tcol in [("monthly_charges", "monthly_charges_t"), ("total_charges", "total_charges_t"),
                      ("tenure_months", "tenure_months_t"), ("last_login_days_ago", "last_login_days_ago_t")]:
        ok = m[col].notna()
        assert ok.mean() > 0.9
        assert np.isclose(m.loc[ok, col].astype(float), m.loc[ok, tcol].astype(float), rtol=1e-3).mean() > 0.995
    ok = m["city"].notna()
    assert (m.loc[ok, "city"] == m.loc[ok, "city_t"]).mean() > 0.98
    assert (m["signup_date"].dropna() == m.loc[m["signup_date"].notna(), "signup_date_t"]).all()


def test_models_beat_chance(result):
    assert result.models.best_score > 0.6
    assert len(result.models.table) >= 5


def test_unsupervised_mode(tmp_path):
    ds = PandasDataset.from_csv(MESSY, "messy_churn_dataset.csv")
    r = run_autopilot(ds, Settings(target=NONE, speed="fast"), out_dir=str(tmp_path))
    assert r.task == "unsupervised" and r.unsup is not None and r.unsup.best_algorithm
    assert len(r.unsup.anomalies) >= 2 and pd.read_csv(r.export.cleaned_path).shape[0] > 1900


def test_chunked_export_dedupes_across_chunks(tmp_path, monkeypatch, result):
    monkeypatch.setattr(af, "CHUNK_ROWS", 300)
    ds = PandasDataset.from_csv(MESSY, "messy_churn_dataset.csv")
    r = run_autopilot(ds, Settings(speed="fast"), out_dir=str(tmp_path))
    assert r.export.rows_out == result.export.rows_out
