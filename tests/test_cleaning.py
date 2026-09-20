"""Unit tests for the cleaning step (pandas only - no Streamlit / Polars needed)."""
import numpy as np
import pandas as pd

from core.cleaning import DataCleaner, parse_dates, parse_numeric_strings, words_to_number, as_text


def _col(values):
    return as_text(pd.Series(values, dtype=object))


def test_currency_thousands_and_negatives():
    p = parse_numeric_strings(_col(["$1,234.50", "USD 23", "18.52$", "(45)", "1.234,56", "abc"]))
    assert p.value.iloc[0] == 1234.50
    assert p.value.iloc[1] == 23
    assert p.value.iloc[2] == 18.52
    assert p.value.iloc[3] == -45
    assert abs(p.value.iloc[4] - 1234.56) < 1e-9          # European format
    assert np.isnan(p.value.iloc[5])


def test_number_words():
    assert words_to_number("forty") == 40
    assert words_to_number("twenty five") == 25
    assert np.isnan(words_to_number("banana"))


def test_mixed_date_formats_learn_day_month_order():
    s = _col(["24.07.21", "28/11/2023", "2019-08-19", "Sep 03, 2021", "05/01/2024", "13/02/2024", "2019-13-45"])
    rules = {"/|4": "%d/%m/%Y", ".|2": "%d.%m.%y"}
    out = parse_dates(s, rules)
    assert str(out.iloc[0].date()) == "2021-07-24"
    assert str(out.iloc[1].date()) == "2023-11-28"
    assert str(out.iloc[2].date()) == "2019-08-19"
    assert str(out.iloc[3].date()) == "2021-09-03"
    assert pd.isna(out.iloc[6])                                # impossible date -> missing


def test_units_are_converted_to_the_unit_in_the_column_name():
    df = pd.DataFrame({"data_usage_gb": (["10 GB", "20 GB", "30 GB", "2048 MB", "0.5 TB", "40", "15", "25"] * 6)})
    clean = DataCleaner().fit(df).transform(df)
    assert clean["data_usage_gb"].iloc[3] == 2.0               # 2048 MB
    assert clean["data_usage_gb"].iloc[4] == 512.0             # 0.5 TB
    assert clean["data_usage_gb"].iloc[5] == 40.0              # unit-less value stays


def test_category_variants_typos_and_abbreviations_merge():
    vals = (["Premium"] * 60 + ["premium"] * 10 + ["PREMIUM"] * 5 + ["Prem"] * 4 +
            ["Standard"] * 60 + ["Std"] * 5 + ["standard "] * 5 + ["Basic"] * 60 + ["Bsc"] * 4)
    df = pd.DataFrame({"plan_type": vals})
    clean = DataCleaner().fit(df).transform(df)
    assert set(clean["plan_type"].unique()) == {"Premium", "Standard", "Basic"}


def test_binary_target_synonyms_become_0_1():
    vals = ["Yes"] * 30 + ["yes"] * 5 + ["Y"] * 5 + ["1"] * 5 + ["Churned"] * 10 + \
           ["No"] * 60 + ["N"] * 5 + ["0"] * 5 + ["Retained"] * 10
    df = pd.DataFrame({"Churn": vals, "x": range(len(vals))})
    c = DataCleaner(target="Churn").fit(df)
    y = c.transform(df)["churn"]
    assert set(y.unique()) == {0.0, 1.0}
    assert y.iloc[:55].eq(1.0).all() and y.iloc[55:].eq(0.0).all()


def test_impossible_values_become_missing_but_skewed_tails_survive():
    rng = np.random.RandomState(0)
    age = list(rng.randint(18, 80, 300)) + [-5, 999, 150]
    income = list(rng.lognormal(10, 1.0, 300))
    df = pd.DataFrame({"age": age, "income": income + [np.nan] * 3})
    clean = DataCleaner().fit(df).transform(df)
    assert clean["age"].iloc[-3:].isna().all()
    assert clean["income"].notna().sum() == 300               # legit heavy tail untouched


def test_junk_columns_dropped_and_ids_flagged():
    n = 60
    df = pd.DataFrame({"Unnamed: 0": range(n), "customer_id": [f"C{i:04d}" for i in range(n)],
                       "empty": [None] * n, "const": ["x"] * n, "v": np.arange(n) % 7})
    c = DataCleaner().fit(df)
    out = c.transform(df)
    assert "unnamed_0" not in out.columns and "empty" not in out.columns and "const" not in out.columns
    assert "customer_id" in c.roles()["identifier"]
