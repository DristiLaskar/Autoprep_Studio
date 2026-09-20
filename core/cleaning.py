"""
cleaning.py
-----------
Explainable data cleaning & normalization that runs BEFORE profiling.

The cleaner is *fit* on a (sample of the) raw data to learn a per-column
plan, then *applied* to any number of chunks, so a 300k-row file gets exactly
the same treatment as the sample used for experimentation.

What it does (every action is counted and reported):

  * column names            -> tidy snake_case ("Tenure (months)" -> tenure_months)
  * hidden missing values   -> "NA", "N/A", "?", "-", "null", "missing" ... -> real NaN
  * numbers stored as text  -> strips currency ($, USD, EUR ...), thousands
                               separators, %, number words ("forty"), "8/10"
  * units                   -> "512 MB", "0.8 TB", "35 yrs" ... converted to ONE unit
                               per column (the unit in the column name wins)
  * impossible values       -> negative ages, 999 sentinels, x100 typos -> NaN
  * booleans                -> Yes / Y / true / 1 / "TRUE" -> 1, No / n / false / 0 -> 0
  * mixed date formats      -> parsed per separator style, day/month order learned
                               from the data, plus year / month / weekday features
  * category spellings      -> case, spacing, accents, typos, abbreviations merged
  * junk columns            -> index artifacts, empty, constant columns dropped
  * emails / phones         -> validated (invalid -> NaN), flagged as non-features
  * identifiers             -> flagged so they never become model features

Only pandas/numpy are used here, so it is fast and easy to test.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NULL_TOKENS = frozenset({
    "", "na", "n/a", "n.a.", "n.a", "nan", "null", "none", "nil", "?", "??", "-", "--", "---",
    "missing", "#n/a", "#na", "undefined", "not available", "not applicable", "blank",
    "empty", "<na>", "nat", "(blank)", "n\\a",
})
TRUE_TOKENS = frozenset({"yes", "y", "true", "t", "1", "on"})
FALSE_TOKENS = frozenset({"no", "n", "false", "f", "0", "off"})

_MINUS_TABLE = {ord("\u2212"): "-", ord("\u2013"): "-", ord("\u2014"): "-"}
_CUR_SYMBOL_RE = r"[$€£¥₹₩₽₺₫฿₪₦¢]"
_CUR_CODES = r"(?:usd|eur|gbp|inr|jpy|cny|rmb|cad|aud|chf|nzd|sgd|hkd|aed|sar|brl|mxn|zar|krw|rub|rs)"

# unit -> factor to the family's smallest reference unit
UNIT_FAMILIES: Dict[str, Dict[str, float]] = {
    "data": {"b": 1, "byte": 1, "bytes": 1, "kb": 1024, "kib": 1024, "mb": 1024 ** 2, "mib": 1024 ** 2,
             "gb": 1024 ** 3, "gib": 1024 ** 3, "tb": 1024 ** 4, "tib": 1024 ** 4, "pb": 1024 ** 5},
    "weight": {"mg": 0.001, "g": 1, "gram": 1, "grams": 1, "kg": 1000, "kgs": 1000, "lb": 453.592,
               "lbs": 453.592, "oz": 28.3495},
    "length": {"mm": 0.001, "cm": 0.01, "m": 1, "km": 1000, "in": 0.0254, "inch": 0.0254,
               "inches": 0.0254, "ft": 0.3048, "feet": 0.3048, "yd": 0.9144, "mi": 1609.344,
               "mile": 1609.344, "miles": 1609.344},
    "time": {"ms": 0.001, "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1, "min": 60,
             "mins": 60, "minute": 60, "minutes": 60, "h": 3600, "hr": 3600, "hrs": 3600,
             "hour": 3600, "hours": 3600, "d": 86400, "day": 86400, "days": 86400,
             "wk": 604800, "wks": 604800, "week": 604800, "weeks": 604800,
             "mo": 2629800, "mos": 2629800, "month": 2629800, "months": 2629800,
             "y": 31557600, "yr": 31557600, "yrs": 31557600, "year": 31557600, "years": 31557600},
    "percent": {"%": 1, "pct": 1, "percent": 1},
}
UNIT_TO_FAMILY = {u: fam for fam, d in UNIT_FAMILIES.items() for u in d}

NONNEG_HINT = re.compile(
    r"(^|_)(age|tenure|charge|charges|price|cost|amount|count|calls|usage|days|weeks|months|years|"
    r"hours|minutes|qty|quantity|duration|size|weight|height|distance|salary|income|fee|fees|"
    r"revenue|sales|score|rating|visits|orders|logins?|ago)($|_)", re.I)
SCORE_HINT = re.compile(r"(score|rating|grade|satisfaction)", re.I)
PCT_HINT = re.compile(r"(percent|percentage|pct)", re.I)
AGE_HINT = re.compile(r"(^|_)age($|_)", re.I)
ID_NAME_RE = re.compile(r"(^|_)(id|uuid|guid|key|ref|reference|number|no|num|code)($|_)", re.I)
INDEX_NAME_RE = re.compile(r"^(unnamed(_\d+)?|index|level_0|idx|row|row_id|rowid|row_number|rownum)$", re.I)
PHONE_NAME_RE = re.compile(r"(phone|mobile|tel|fax|whatsapp|contact_no)", re.I)
DATE_NAME_RE = re.compile(r"(date|time|timestamp|dob|born|created|updated|_at$|_on$)", re.I)

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def snake_case(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s.strip()).strip("_").lower()
    return s or "column"


def unique_names(names: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 1
            out.append(n)
    return out


def as_text(s: pd.Series) -> pd.Series:
    """Object Series of stripped strings; null-like tokens become NaN."""
    s = s.astype(object)
    m = s.notna()
    out = pd.Series(np.nan, index=s.index, dtype=object)
    if m.any():
        out.loc[m] = s.loc[m].astype(str)
    m = out.notna()
    if m.any():
        out.loc[m] = out.loc[m].str.strip().str.replace(r"\s+", " ", regex=True)
    low = out.str.lower()
    return out.mask(low.isin(NULL_TOKENS))


def _bool(s: pd.Series) -> pd.Series:
    """Boolean mask from a str-accessor result that may contain NaN/NA."""
    return s.fillna(False).astype(bool)


def words_to_number(text: str) -> float:
    toks = [t for t in re.split(r"[\s\-,]+", text.lower().strip()) if t and t != "and"]
    if not toks:
        return float("nan")
    total, current = 0, 0
    for t in toks:
        if t in _NUM_WORDS:
            current += _NUM_WORDS[t]
        elif t == "hundred":
            current = (current or 1) * 100
        elif t == "thousand":
            total += (current or 1) * 1000
            current = 0
        else:
            return float("nan")
    return float(total + current)


def _norm_key(v: str) -> str:
    v = unicodedata.normalize("NFKD", v)
    v = "".join(ch for ch in v if not unicodedata.combining(ch))
    v = v.casefold()
    v = re.sub(r"[\s_\-/.]+", " ", v).strip()
    v = re.sub(r"[^\w ]", "", v)
    return v or v


def _is_subsequence(short: str, long: str) -> bool:
    it = iter(long)
    return all(ch in it for ch in short)


# ---------------------------------------------------------------------------
# Numeric text parsing
# ---------------------------------------------------------------------------
@dataclass
class ParsedNumbers:
    value: pd.Series          # float, NaN where not parsed
    unit: pd.Series           # object, "" where none
    had_currency: pd.Series   # bool
    nonnull: pd.Series        # bool: input was not null


def _normalize_number_strings(num: pd.Series) -> pd.Series:
    n = num.fillna("").str.replace("'", "", regex=False)
    has_c = n.str.contains(",", regex=False)
    has_d = n.str.contains(".", regex=False)
    out = n.copy()
    both = has_c & has_d
    comma_last = both & (n.str.rfind(",") > n.str.rfind("."))
    dot_last = both & ~comma_last
    out = out.where(~comma_last, n.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    out = out.where(~dot_last, n.str.replace(",", "", regex=False))
    only_c = has_c & ~has_d
    thousands = only_c & n.str.match(r"^\d{1,3}(,\d{3})+$")
    dec_c = only_c & ~thousands & n.str.match(r"^\d+,\d{1,2}$")
    other_c = only_c & ~thousands & ~dec_c
    out = out.where(~thousands, n.str.replace(",", "", regex=False))
    out = out.where(~dec_c, n.str.replace(",", ".", regex=False))
    out = out.where(~other_c, n.str.replace(",", "", regex=False))
    only_d = has_d & ~has_c
    multi_dot = only_d & n.str.match(r"^\d{1,3}(\.\d{3}){2,}$")
    out = out.where(~multi_dot, n.str.replace(".", "", regex=False))
    return out


def parse_numeric_strings(t: pd.Series) -> ParsedNumbers:
    """Vectorised parse of strings like '$1,234.50', '12.5 GB', '(45)', '18.52$', 'USD 23'."""
    nonnull = t.notna()
    w = t.where(nonnull, "").astype(object).str.translate(_MINUS_TABLE)
    neg_paren = w.str.match(r"^\(.*\)$")
    w = w.str.replace(r"^\((.*)\)$", r"\1", regex=True)
    had_sym = w.str.contains(_CUR_SYMBOL_RE, regex=True)
    w = w.str.replace(_CUR_SYMBOL_RE, "", regex=True)
    code_pat = rf"(?i)^(?:{_CUR_CODES})\.?\s*|\s*\b(?:{_CUR_CODES})\.?$"
    had_code = w.str.contains(rf"(?i)^(?:{_CUR_CODES})\b|\b(?:{_CUR_CODES})$", regex=True)
    w = w.str.replace(code_pat, "", regex=True)

    ex = w.str.extract(
        r"^\s*(?P<sign>[+-]?)\s*(?P<num>\d[\d.,']*\d|\d|\.\d+)\s*"
        r"(?P<unit>[A-Za-z%°µμ][A-Za-z%°µμ/²³. ]*)?\s*$"
    )
    ok = ex["num"].notna()
    norm = _normalize_number_strings(ex["num"])
    val = pd.to_numeric(norm.mask(norm == ""), errors="coerce")
    sign = np.where(ex["sign"].fillna("") == "-", -1.0, 1.0)
    val = val * sign
    val = val.where(~neg_paren, -val.abs())
    val = val.where(ok)
    # scientific notation / inf fallbacks
    left = nonnull & ~ok
    if left.any():
        alt = pd.to_numeric(w[left], errors="coerce")
        alt = alt.where(np.isfinite(alt))
        val = val.copy()
        val.loc[left] = alt
    unit = ex["unit"].fillna("").str.strip().str.lower().str.rstrip(".").str.strip()
    return ParsedNumbers(val.astype(float), unit.astype(object), (had_sym | had_code) & nonnull, nonnull)


_RATIO_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)$")


@dataclass
class NumericParse:
    values: pd.Series
    unit: pd.Series
    n_currency: int
    n_words: int
    n_ratio: int
    ok_rate: float
    plain_share: float


def try_numeric(t: pd.Series) -> Optional[NumericParse]:
    """Try to interpret a text column as numeric. Returns None if it is not numeric."""
    nn = t.notna()
    n = int(nn.sum())
    if n == 0:
        return None
    p = parse_numeric_strings(t)
    parsed = p.value.notna()
    plain = parsed & (p.unit == "")
    plain_share = float(plain.sum()) / n
    unit_known = p.unit.map(lambda u: u in UNIT_TO_FAMILY).astype(bool)
    unit_ok = (p.unit == "") | unit_known | (plain_share >= 0.6)
    good = parsed & unit_ok
    values = p.value.where(good)

    left = nn & ~good
    n_words = n_ratio = 0
    if left.any():
        vals = t[left].unique()
        wmap = {v: words_to_number(v) for v in vals}
        wv = t[left].map(wmap).astype(float)
        got = wv.notna()
        if got.any():
            values.loc[wv.index[got]] = wv[got]
            n_words = int(got.sum())
            left = left & ~values.notna()
        if left.any():
            m = t[left].str.extract(_RATIO_RE)
            if m.iloc[:, 0].notna().any():
                dens = m.iloc[:, 1].dropna().astype(float)
                if dens.nunique() == 1 and plain_share >= 0.5:
                    numer = m.iloc[:, 0].astype(float)
                    hit = numer.notna()
                    values.loc[numer.index[hit]] = numer[hit]
                    n_ratio = int(hit.sum())
    ok_rate = float(values.notna().sum()) / n
    # leading-zero codes (zip codes, product codes) are not numbers
    lead0 = t[nn].str.match(r"^0\d+$").mean() if n else 0.0
    if lead0 > 0.3:
        return None
    return NumericParse(values.astype(float), p.unit, int((p.had_currency & good).sum()),
                        n_words, n_ratio, ok_rate, plain_share)


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------
_DATE_NUM_RE = r"^(\d{1,4})([/\-.])(\d{1,2})\2(\d{1,4})$"


def _learn_date_rules(t: pd.Series) -> Dict[str, str]:
    """Learn day/month order per (separator, year-length) from unambiguous values."""
    ex = t.dropna().str.extract(_DATE_NUM_RE).dropna()
    rules: Dict[str, str] = {}
    if ex.empty:
        return rules
    ex.columns = ["a", "sep", "b", "c"]
    ex = ex[ex["a"].str.len() <= 2]
    if ex.empty:
        return rules
    a = ex["a"].astype(int)
    b = ex["b"].astype(int)
    ex = ex.assign(ylen=ex["c"].str.len(), day_first=(a > 12), month_first=(b > 12))
    global_day = int((a > 12).sum())
    global_month = int((b > 12).sum())
    for (sep, ylen), g in ex.groupby(["sep", "ylen"]):
        d, m = int(g["day_first"].sum()), int(g["month_first"].sum())
        if d == m == 0:
            d, m = global_day, global_month
        day_first = d >= m
        y = "%Y" if ylen == 4 else "%y"
        rules[f"{sep}|{ylen}"] = f"%d{sep}%m{sep}{y}" if day_first else f"%m{sep}%d{sep}{y}"
    return rules


def parse_dates(t: pd.Series, rules: Dict[str, str]) -> pd.Series:
    """Parse mixed-format date strings; unparseable values become NaT."""
    out = pd.Series(pd.NaT, index=t.index, dtype="datetime64[ns]")
    nn = t.notna()
    if not nn.any():
        return out
    ex = t.str.extract(_DATE_NUM_RE)
    is_num = ex[0].notna()
    parsed_any = pd.Series(False, index=t.index)

    def _assign(mask: pd.Series, fmt: Optional[str], **kw):
        nonlocal out
        if not mask.any():
            return
        try:
            if fmt:
                res = pd.to_datetime(t[mask], format=fmt, errors="coerce")
            else:
                res = pd.to_datetime(t[mask], errors="coerce", **kw)
        except Exception:
            return
        res = pd.Series(res, index=t.index[mask]).astype("datetime64[ns]")
        out.loc[mask] = res

    # year-first numeric (2021-05-03, 2021/05/03)
    for sep in ("-", "/", "."):
        m = is_num & (ex[0].str.len() == 4) & (ex[1] == sep)
        _assign(m, f"%Y{sep}%m{sep}%d")
        parsed_any |= m
    # day/month ambiguous numeric
    m_rest = is_num & (ex[0].str.len() <= 2)
    if m_rest.any():
        sub = ex.loc[m_rest, [1, 3]].copy()
        sub["_y"] = sub[3].str.len()
        combos = sub[[1, "_y"]].drop_duplicates()
        for _, row in combos.iterrows():
            sep, ylen = row[1], int(row["_y"])
            if ylen not in (2, 4):
                continue
            y = "%Y" if ylen == 4 else "%y"
            fmt = rules.get(f"{sep}|{ylen}", f"%d{sep}%m{sep}{y}")
            m = m_rest & (ex[1] == sep) & (ex[3].str.len() == ylen)
            _assign(m, fmt)
            parsed_any |= m
    # ISO timestamps
    m_iso = nn & ~parsed_any & _bool(t.str.match(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"))
    if m_iso.any():
        try:
            res = pd.to_datetime(t[m_iso], errors="coerce", utc=True, format="ISO8601")
            res = pd.Series(res, index=t.index[m_iso]).dt.tz_localize(None).astype("datetime64[ns]")
            out.loc[m_iso] = res
        except Exception:
            pass
        parsed_any |= m_iso
    # named months: "Jul 19, 2019", "19 July 2019", "19-Jul-2019"
    m_named = nn & ~parsed_any & _bool(t.str.contains(r"[A-Za-z]{3,}", regex=True)) & _bool(t.str.contains(r"\d"))
    if m_named.any():
        _assign(m_named, None, format="mixed")
    # sanity: plausible years only
    bad = out.notna() & ((out.dt.year < 1900) | (out.dt.year > 2100))
    out = out.mask(bad)
    return out


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------
@dataclass
class ColumnPlan:
    original: str
    name: str
    kind: str = "categorical"    # numeric|boolean|datetime|categorical|email|phone|empty|constant|index
    role: str = "feature"        # feature|identifier|contact|text|datetime_raw|index|dropped
    drop: bool = False
    drop_reason: str = ""
    was_numeric_dtype: bool = False
    # numeric
    unit_base: Optional[str] = None
    unit_factors: Dict[str, float] = field(default_factory=dict)
    unit_family: Optional[str] = None
    nonneg: bool = False
    lower: Optional[float] = None
    upper: Optional[float] = None
    is_integer: bool = False
    # categorical
    cat_map: Dict[str, str] = field(default_factory=dict)     # normalized key -> canonical label
    cat_merges: List[dict] = field(default_factory=list)
    id_case: Optional[str] = None                             # upper|lower|None for identifiers
    # datetime
    date_rules: Dict[str, str] = field(default_factory=dict)
    # boolean-like target labels {key: 0/1}
    label_map: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


@dataclass
class CleaningReport:
    column_table: pd.DataFrame
    actions: pd.DataFrame
    merges: pd.DataFrame
    dropped: pd.DataFrame
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# The cleaner
# ---------------------------------------------------------------------------
class DataCleaner:
    """Fit a per-column cleaning plan on a raw sample, apply it to any chunk."""

    def __init__(self, target: Optional[str] = None, max_categories_fuzzy: int = 300):
        self.target_raw = target
        self.plans: Dict[str, ColumnPlan] = {}          # keyed by original column name
        self.order: List[str] = []
        self.max_categories_fuzzy = max_categories_fuzzy
        self.stats_: Counter = Counter()
        self.stats_by_col_: Dict[str, Counter] = {}
        self.warnings: List[str] = []
        self.fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> "DataCleaner":
        self.order = list(df.columns)
        clean_names = unique_names([snake_case(c) for c in self.order])
        for orig, cname in zip(self.order, clean_names):
            plan = self._fit_column(orig, cname, df[orig], is_target=(orig == self.target_raw))
            self.plans[orig] = plan
        self.fitted = True
        return self

    @property
    def target_name(self) -> Optional[str]:
        if self.target_raw is None or self.target_raw not in self.plans:
            return None
        return self.plans[self.target_raw].name

    def _fit_column(self, orig: str, cname: str, raw: pd.Series, is_target: bool) -> ColumnPlan:
        plan = ColumnPlan(original=orig, name=cname)
        n_total = len(raw)
        if cname != orig:
            plan.notes.append("column name standardized")

        # ---- already-typed columns -------------------------------------
        if pd.api.types.is_bool_dtype(raw):
            plan.kind = "boolean"
            plan.notes.append("boolean -> 0/1")
            return self._finish_fit(plan, is_target, n_total, raw.astype(float))
        if pd.api.types.is_datetime64_any_dtype(raw):
            plan.kind, plan.role = "datetime", "datetime_raw"
            plan.notes.append("datetime column -> ISO date + year/month/weekday features")
            return plan
        if pd.api.types.is_numeric_dtype(raw):
            plan.kind = "numeric"
            plan.was_numeric_dtype = True
            vals = pd.to_numeric(raw, errors="coerce").astype(float)
            vals = vals.where(np.isfinite(vals))
            if vals.notna().sum() == 0:
                plan.kind, plan.drop, plan.drop_reason, plan.role = "empty", True, "column is completely empty", "dropped"
                return plan
            if INDEX_NAME_RE.match(cname) and vals.nunique() >= 0.95 * vals.notna().sum():
                plan.kind, plan.role, plan.drop = "index", "index", True
                plan.drop_reason = "index artifact (row counter exported with the file)"
                return plan
            self._learn_numeric_bounds(plan, vals)
            return self._finish_fit(plan, is_target, n_total, vals)

        # ---- text-like columns -----------------------------------------
        t = as_text(raw)
        nn = t.notna()
        n = int(nn.sum())
        n_null_tokens = int(raw.notna().sum()) - n
        if n == 0:
            plan.kind, plan.drop, plan.drop_reason, plan.role = "empty", True, "column is completely empty", "dropped"
            return plan

        if INDEX_NAME_RE.match(cname):
            probe = try_numeric(t)
            if probe is not None and probe.values.nunique() >= 0.95 * n:
                plan.kind, plan.role, plan.drop = "index", "index", True
                plan.drop_reason = "index artifact (row counter exported with the file)"
                return plan

        if n_null_tokens > 0:
            plan.notes.append("hidden missing values (NA / ? / - / null ...) -> NaN")

        # email / phone
        share_at = t[nn].str.contains("@", regex=False).mean()
        if share_at >= 0.5:
            plan.kind, plan.role = "email", "contact"
            plan.notes.append("emails lower-cased; invalid ones -> missing")
            return plan
        if PHONE_NAME_RE.search(cname):
            plan.kind, plan.role = "phone", "contact"
            plan.notes.append("phone numbers normalized to digits; invalid ones -> missing")
            return plan

        # boolean tokens
        low = t[nn].str.lower()
        in_bool = low.isin(TRUE_TOKENS | FALSE_TOKENS)
        if in_bool.mean() >= 0.95 and low[in_bool].str.contains(r"[a-z]", regex=True).any() \
                and low[in_bool].nunique() >= 2 and not is_target:
            plan.kind = "boolean"
            plan.notes.append("Yes/No/True/False/Y/N/1/0 variants -> 1/0")
            return self._finish_fit(plan, is_target, n_total, None)

        # numeric
        num = try_numeric(t)
        if num is not None and num.ok_rate >= 0.85:
            plan.kind = "numeric"
            self._learn_units(plan, num, cname)
            vals = self._apply_units(num.values, num.unit, plan)
            self._learn_numeric_bounds(plan, vals)
            if num.n_currency:
                plan.notes.append("currency symbols/codes stripped")
            if plan.unit_factors and any(abs(f - 1) > 1e-12 for f in plan.unit_factors.values()):
                plan.notes.append(f"units converted to {plan.unit_base}")
            elif num.unit.astype(bool).any():
                plan.notes.append("unit suffixes stripped")
            if num.n_words:
                plan.notes.append("number words (e.g. 'forty') -> digits")
            if num.n_ratio:
                plan.notes.append("'x/10' style scores -> numerator")
            if (~num.values.notna() & nn).any():
                plan.notes.append("unparseable text -> missing")
            return self._finish_fit(plan, is_target, n_total, vals)

        # dates
        if not is_target:
            rules = _learn_date_rules(t)
            parsed = parse_dates(t, rules)
            rate = float(parsed.notna().sum()) / n
            has_sep = t[nn].str.contains(r"[/\-.:]|[A-Za-z]{3,}", regex=True).mean()
            thresh = 0.6 if DATE_NAME_RE.search(cname) else 0.75
            if rate >= thresh and has_sep >= 0.9:
                plan.kind, plan.role = "datetime", "datetime_raw"
                plan.date_rules = rules
                shapes = t[nn].str.replace(r"\d", "9", regex=True).str.replace(r"[A-Za-z]+", "A", regex=True).nunique()
                plan.notes.append(f"{shapes} date format(s) parsed -> ISO date + year/month/weekday features")
                return plan

        # categorical / identifier / text
        return self._fit_categorical(plan, t, cname, is_target, n_total)

    # -------------------------------------------------------------- numeric
    def _learn_units(self, plan: ColumnPlan, num: NumericParse, cname: str) -> None:
        units = num.unit[num.values.notna() & (num.unit != "")]
        if units.empty:
            return
        counts = Counter(units)
        fam_counts: Counter = Counter()
        for u, c in counts.items():
            fam = UNIT_TO_FAMILY.get(u)
            if fam:
                fam_counts[fam] += c
        if not fam_counts:
            return
        fam = fam_counts.most_common(1)[0][0]
        if fam_counts[fam] < 0.8 * sum(fam_counts.values()):
            return  # mixed families -> just strip units
        plan.unit_family = fam
        table = UNIT_FAMILIES[fam]
        name_units = [tok for tok in cname.split("_") if tok in table]
        in_col = {u: c for u, c in counts.items() if u in table}
        base = name_units[-1] if name_units else max(in_col, key=in_col.get)
        plan.unit_base = base
        plan.unit_factors = {u: table[u] / table[base] for u in in_col}
        if fam == "percent":
            plan.unit_factors = {u: 1.0 for u in in_col}

    def _apply_units(self, values: pd.Series, unit: pd.Series, plan: ColumnPlan) -> pd.Series:
        if not plan.unit_factors:
            return values
        f = unit.map(plan.unit_factors).astype(float).fillna(1.0)
        return values * f

    def _learn_numeric_bounds(self, plan: ColumnPlan, vals: pd.Series) -> None:
        v = vals.dropna()
        n = len(v)
        cname = plan.name
        plan.nonneg = bool(NONNEG_HINT.search(cname))
        if n and np.all(np.isclose(v.values, np.round(v.values))):
            plan.is_integer = True
        if AGE_HINT.search(cname):
            plan.upper = 120.0
        if PCT_HINT.search(cname):
            plan.lower, plan.upper = 0.0, 100.0
        if SCORE_HINT.search(cname) and n >= 20:
            p95 = float(v[v >= 0].quantile(0.95)) if (v >= 0).any() else 0.0
            for scale in (1, 5, 10, 100):
                if p95 <= scale:
                    plan.upper = float(scale)
                    plan.nonneg = True
                    break
        if n < 30:
            self._note_validity(plan)
            return
        # 0 used as a placeholder in a column whose real values are all far from 0 (e.g. adult ages)
        if plan.nonneg or AGE_HINT.search(cname):
            pos = v[v > 0]
            if len(pos) >= 30 and float(pos.quantile(0.01)) >= 10 and float((v == 0).mean()) < 0.02 and (v == 0).any():
                plan.lower = 0.5
        # implausible extremes: only cut where there is a clear gap between legit and absurd values
        rest = v
        if plan.nonneg:
            rest = rest[rest >= 0]
        if plan.upper is not None:
            rest = rest[rest <= plan.upper]
        if len(rest) < 30:
            self._note_validity(plan)
            return
        up = self._gap_bound(rest)
        if up is not None:
            plan.upper = up if plan.upper is None else min(plan.upper, up)
        lo = self._gap_bound(-rest)
        if lo is not None and not plan.nonneg:
            plan.lower = -lo if plan.lower is None else max(plan.lower, -lo)
        self._note_validity(plan)

    @staticmethod
    def _note_validity(plan: ColumnPlan) -> None:
        rules = []
        if plan.nonneg:
            rules.append("negative")
        if plan.lower is not None:
            rules.append(f"< {plan.lower:g}")
        if plan.upper is not None:
            rules.append(f"> {plan.upper:g}")
        if rules:
            plan.notes.append("impossible/extreme values (" + ", ".join(rules) + ") -> missing")

    @staticmethod
    def _gap_bound(x: pd.Series) -> Optional[float]:
        q1, q3 = float(x.quantile(0.25)), float(x.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            return None
        p95 = float(x.quantile(0.95))
        cand = max(q3 + 8 * iqr, 6 * max(p95, 0.0))
        above = x[x > cand]
        below = x[x <= cand]
        if above.empty or below.empty or len(above) / len(x) > 0.05:
            return None
        if below.max() > 0 and above.min() > 2.5 * below.max():
            return float(cand)
        return None

    def _apply_bounds(self, v: pd.Series, plan: ColumnPlan) -> Tuple[pd.Series, int]:
        invalid = pd.Series(False, index=v.index)
        if plan.nonneg:
            invalid |= v < 0
        if plan.lower is not None:
            invalid |= v < plan.lower
        if plan.upper is not None:
            invalid |= v > plan.upper
        return v.mask(invalid), int(invalid.sum())

    # ---------------------------------------------------------- categorical
    def _fit_categorical(self, plan: ColumnPlan, t: pd.Series, cname: str, is_target: bool,
                         n_total: int) -> ColumnPlan:
        nn = t.notna()
        n = int(nn.sum())
        counts = t[nn].value_counts()
        n_unique = len(counts)
        ratio = n_unique / max(n, 1)
        mean_len = float(t[nn].str.len().mean())
        plan.kind = "categorical"

        if not is_target:
            if ID_NAME_RE.search(cname) and ratio >= 0.5:
                plan.role = "identifier"
            elif ratio >= 0.9 and n_unique >= 20:
                plan.role = "identifier"
            elif mean_len >= 30 and ratio > 0.5:
                plan.role = "text"
            elif n_unique > 100 and ratio > 0.5:
                plan.role = "text"
            if plan.role in ("identifier", "text"):
                notes = "identifier" if plan.role == "identifier" else "free text"
                plan.notes.append(f"treated as {notes} (not used as a model feature)")
                if plan.role == "identifier":
                    letters = t[nn].str.replace(r"[^A-Za-z]", "", regex=True)
                    letters = letters[letters != ""]
                    if len(letters):
                        if float((letters == letters.str.upper()).mean()) >= 0.9:
                            plan.id_case = "upper"
                        elif float((letters == letters.str.lower()).mean()) >= 0.9:
                            plan.id_case = "lower"
                return plan

        if n_unique <= 1 and not is_target:
            plan.kind, plan.role, plan.drop, plan.drop_reason = "constant", "dropped", True, "constant column (no information)"
            return plan

        cat_map, merges = self._build_category_map(counts, allow_fuzzy=(n_unique <= self.max_categories_fuzzy))
        plan.cat_map = cat_map
        plan.cat_merges = merges
        if merges:
            plan.notes.append(f"{sum(m['rows'] for m in merges)} cells with variant spellings merged "
                              f"into {len({m['canonical'] for m in merges})} canonical label(s)")
        if is_target:
            self._maybe_binary_labels(plan, counts)
        return plan

    def _build_category_map(self, counts: pd.Series, allow_fuzzy: bool) -> Tuple[Dict[str, str], List[dict]]:
        # group raw variants by normalized key
        groups: Dict[str, Counter] = {}
        for raw, c in counts.items():
            groups.setdefault(_norm_key(raw), Counter())[raw] += int(c)

        def looks_title(s: str) -> bool:
            return s[:1].isupper() and s[1:] == s[1:].lower() or s.istitle()

        canon: Dict[str, str] = {}
        gcount: Dict[str, int] = {}
        for key, variants in groups.items():
            best = sorted(variants.items(), key=lambda kv: (-kv[1], not looks_title(kv[0]), kv[0]))[0][0]
            canon[key] = best
            gcount[key] = sum(variants.values())
        key_target: Dict[str, str] = {k: k for k in groups}

        if allow_fuzzy and len(groups) >= 2:
            digit_heavy = np.mean([bool(re.search(r"\d", k)) for k in groups]) > 0.3
            keys_sorted = sorted(groups, key=lambda k: -gcount[k])
            majors = [k for k in keys_sorted if gcount[k] >= 5]
            if not digit_heavy:
                for kb in keys_sorted:
                    cb = gcount[kb]
                    cands = [ka for ka in majors if ka != kb and gcount[ka] >= 3 * cb]
                    if not cands or not kb:
                        continue
                    target = self._match_variant(kb, cands, gcount)
                    if target is not None:
                        key_target[kb] = target
        # resolve chains
        def root(k):
            seen = set()
            while key_target[k] != k and k not in seen:
                seen.add(k)
                k = key_target[k]
            return k

        cat_map: Dict[str, str] = {}
        merges: List[dict] = []
        for key, variants in groups.items():
            r = root(key)
            cat_map[key] = canon[r]
            for raw, c in variants.items():
                if raw != canon[r]:
                    merges.append({"variant": raw, "canonical": canon[r], "rows": int(c)})
        merges.sort(key=lambda m: -m["rows"])
        return cat_map, merges

    @staticmethod
    def _match_variant(kb: str, cands: List[str], gcount: Dict[str, int]) -> Optional[str]:
        # 1) abbreviation as prefix ("prem" -> "premium", "m" -> "male")
        pref = [ka for ka in cands if ka.startswith(kb) and len(kb) < len(ka)]
        if len(pref) == 1:
            return pref[0]
        if len(pref) > 1:
            return None
        # 2) abbreviation as subsequence ("std" -> "standard", "bsc" -> "basic")
        if len(kb) >= 3:
            sub = [ka for ka in cands if len(kb) < len(ka) and ka[0] == kb[0] and _is_subsequence(kb, ka)]
            if len(sub) == 1:
                return sub[0]
            if len(sub) > 1:
                return None
        # 3) typos via string similarity
        if len(kb) >= 4:
            best, best_r = None, 0.0
            for ka in cands:
                if re.sub(r"\d+", "#", ka) == re.sub(r"\d+", "#", kb) and re.search(r"\d", kb):
                    continue
                sm = difflib.SequenceMatcher(None, kb, ka)
                if sm.real_quick_ratio() < 0.8 or sm.quick_ratio() < 0.8:
                    continue
                r = sm.ratio()
                if r > best_r or (abs(r - best_r) < 1e-9 and best is not None and gcount[ka] > gcount[best]):
                    best, best_r = ka, r
            if best is not None and best_r >= 0.82 and (best[0] == kb[0] or best_r >= 0.9):
                return best
        return None

    def _maybe_binary_labels(self, plan: ColumnPlan, counts: pd.Series) -> None:
        """Map synonym labels (Yes/Y/1/TRUE/Churned vs No/N/0/FALSE/Retained) onto 0/1."""
        # collapse counts by canonical label
        by_canon: Counter = Counter()
        for raw, c in counts.items():
            by_canon[plan.cat_map.get(_norm_key(raw), raw)] += int(c)
        total = sum(by_canon.values())
        if len(by_canon) < 2:
            return
        pos, neg, left = [], [], []
        for lab in by_canon:
            k = _norm_key(lab)
            if k in TRUE_TOKENS:
                pos.append(lab)
            elif k in FALSE_TOKENS:
                neg.append(lab)
            else:
                left.append(lab)
        if not pos and not neg:
            return
        stem = re.sub(r"[^a-z]", "", plan.name.lower())
        stem = stem[:5] if len(stem) > 5 else stem
        if left:
            # words that contain the column-name stem ("churned" for "churn") are the positive class
            hits = [l for l in left if stem and stem in _norm_key(l).replace(" ", "")]
            rest = [l for l in left if l not in hits]
            if hits and len(hits) == len(left):
                pos += hits
            elif hits and rest:
                pos += hits
                neg += rest
            elif not hits and pos and not neg and len(left) == 1:
                neg += left
            elif not hits and neg and not pos and len(left) == 1:
                pos += left
            else:
                return
        if not pos or not neg:
            return
        covered = sum(by_canon[l] for l in pos + neg)
        if covered < 0.97 * total:
            return
        plan.label_map = {_norm_key(l): 1.0 for l in pos}
        plan.label_map.update({_norm_key(l): 0.0 for l in neg})
        plan.kind = "boolean"
        plan.notes.append(f"binary target: {sorted(set(pos))} -> 1, {sorted(set(neg))} -> 0")

    # -------------------------------------------------------------- finish
    def _finish_fit(self, plan: ColumnPlan, is_target: bool, n_total: int,
                    vals: Optional[pd.Series]) -> ColumnPlan:
        if vals is not None and vals.notna().any():
            v = vals.dropna()
            if plan.kind == "numeric" and not is_target:
                if ID_NAME_RE.search(plan.name) and v.nunique() / max(len(v), 1) >= 0.9:
                    plan.role = "identifier"
                    plan.notes.append("identifier (not used as a model feature)")
                elif v.nunique() <= 1:
                    plan.kind, plan.role, plan.drop, plan.drop_reason = "constant", "dropped", True, "constant column (no information)"
        return plan

    # ------------------------------------------------------------ transform
    def transform(self, df: pd.DataFrame, collect_stats: bool = False) -> pd.DataFrame:
        """Apply the fitted plan to a raw chunk. Dropped columns are removed."""
        if not self.fitted:
            raise RuntimeError("DataCleaner must be fit before transform().")
        pieces: Dict[str, pd.Series] = {}
        for orig in self.order:
            plan = self.plans[orig]
            if plan.drop or orig not in df.columns:
                continue
            cs: Counter = Counter()
            cols = self._transform_column(plan, df[orig], cs)
            for k, s in cols.items():
                pieces[k] = s
            if collect_stats:
                self.stats_by_col_.setdefault(plan.name, Counter()).update(cs)
                self.stats_.update(cs)
        out = pd.DataFrame(pieces, index=df.index)
        return out

    def _transform_column(self, plan: ColumnPlan, raw: pd.Series, cs: Counter) -> Dict[str, pd.Series]:
        name = plan.name
        if plan.kind == "datetime":
            t = as_text(raw) if not pd.api.types.is_datetime64_any_dtype(raw) else None
            if t is None:
                dt = pd.to_datetime(raw, errors="coerce")
                if getattr(dt.dt, "tz", None) is not None:
                    dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
                n_raw = int(raw.notna().sum())
            else:
                dt = parse_dates(t, plan.date_rules)
                n_raw = int(t.notna().sum())
            cs["Dates parsed to ISO format"] += int(dt.notna().sum())
            cs["Invalid dates -> missing"] += max(n_raw - int(dt.notna().sum()), 0)
            iso = dt.dt.strftime("%Y-%m-%d").astype(object).where(dt.notna(), np.nan)
            out = {name: iso,
                   f"{name}_year": dt.dt.year.astype(float),
                   f"{name}_month": dt.dt.month.astype(float),
                   f"{name}_dayofweek": dt.dt.dayofweek.astype(float),
                   f"{name}_epoch_days": ((dt - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400.0)}
            return out

        if plan.was_numeric_dtype:
            v = pd.to_numeric(raw, errors="coerce").astype(float)
            v = v.where(np.isfinite(v))
            v, bad = self._apply_bounds(v, plan)
            cs["Impossible / extreme values -> missing"] += bad
            return {name: v}

        if pd.api.types.is_bool_dtype(raw):
            return {name: raw.astype(float)}

        t = as_text(raw)
        n_null_tokens = int(raw.notna().sum()) - int(t.notna().sum())
        cs["Hidden missing tokens (NA, ?, -, null...) -> NaN"] += max(n_null_tokens, 0)

        if plan.kind == "numeric":
            num = try_numeric(t)
            if num is None:
                return {name: pd.Series(np.nan, index=raw.index)}
            v = self._apply_units(num.values, num.unit, plan)
            cs["Currency symbols stripped"] += num.n_currency
            if plan.unit_factors:
                conv = num.unit.map(lambda u: abs(plan.unit_factors.get(u, 1.0) - 1.0) > 1e-12)
                cs["Unit conversions"] += int((conv & num.values.notna()).sum())
            cs["Number words / ratios parsed"] += num.n_words + num.n_ratio
            unparsed = int((t.notna() & v.isna()).sum())
            cs["Unparseable text -> missing"] += unparsed
            v, bad = self._apply_bounds(v, plan)
            cs["Impossible / extreme values -> missing"] += bad
            return {plan.name: v}

        if plan.kind == "boolean":
            low = t.str.lower()
            if plan.label_map:
                uniq = {u: _norm_key(plan.cat_map.get(_norm_key(u), u)) for u in t.dropna().unique()}
                keys = t.map(uniq)
                v = keys.map(plan.label_map)
                fallback = t.str.lower().map(lambda x: 1.0 if x in TRUE_TOKENS else (0.0 if x in FALSE_TOKENS else np.nan))
                v = pd.to_numeric(v, errors="coerce").astype(float)
                v = v.where(v.notna(), pd.to_numeric(fallback, errors="coerce").astype(float))
            else:
                m = pd.Series(np.nan, index=raw.index, dtype=float)
                m[low.isin(TRUE_TOKENS)] = 1.0
                m[low.isin(FALSE_TOKENS)] = 0.0
                v = m
            changed = int((t.notna() & t.str.lower().isin({"yes", "y", "true", "t", "on", "no", "n", "false", "f", "off"})).sum())
            cs["Boolean variants -> 1/0"] += changed
            return {name: v}

        if plan.kind == "email":
            e = t.str.lower()
            valid = e.map(lambda x: bool(_EMAIL_RE.match(x)) if isinstance(x, str) else False)
            cs["Invalid emails -> missing"] += int((e.notna() & ~valid).sum())
            return {name: e.where(valid)}

        if plan.kind == "phone":
            digits = t.str.replace(r"[^\d+]", "", regex=True)
            nd = digits.str.replace("+", "", regex=False).str.len()
            same = _bool(digits.str.replace("+", "", regex=False).str.match(r"^(\d)\1+$"))
            valid = _bool(nd.between(10, 15)) & ~same
            cs["Invalid phone numbers -> missing"] += int((t.notna() & ~valid).sum())
            return {name: digits.where(valid)}

        # categorical / identifier / text
        if plan.role == "text":
            return {name: t}
        if plan.role == "identifier":
            if plan.id_case == "upper":
                t2 = t.str.upper()
            elif plan.id_case == "lower":
                t2 = t.str.lower()
            else:
                t2 = t
            cs["Identifier casing normalized"] += int((t2.notna() & (t2 != t)).sum())
            return {name: t2}
        uniq = t.dropna().unique()
        mapping = {}
        for u in uniq:
            mapping[u] = plan.cat_map.get(_norm_key(u), u)
        mapped = t.map(mapping)
        changed = int((t.notna() & (mapped != t)).sum())
        cs["Category spellings merged"] += changed
        return {name: mapped}

    # --------------------------------------------------------------- report
    def roles(self) -> Dict[str, List[str]]:
        """Cleaned column names grouped by how they may be used."""
        out = {"numeric": [], "categorical": [], "identifier": [], "contact": [], "text": [],
               "datetime_raw": [], "target": []}
        for orig in self.order:
            p = self.plans[orig]
            if p.drop:
                continue
            if p.kind == "datetime":
                out["datetime_raw"].append(p.name)
                out["numeric"] += [f"{p.name}_year", f"{p.name}_month", f"{p.name}_dayofweek", f"{p.name}_epoch_days"]
                continue
            if p.role == "identifier":
                out["identifier"].append(p.name)
            elif p.role == "contact":
                out["contact"].append(p.name)
            elif p.role == "text":
                out["text"].append(p.name)
            elif p.kind in ("numeric", "boolean"):
                out["numeric"].append(p.name)
            else:
                out["categorical"].append(p.name)
        return out

    def kinds(self) -> Dict[str, str]:
        """Cleaned column -> final type label for display."""
        k = {}
        for orig in self.order:
            p = self.plans[orig]
            if p.drop:
                continue
            k[p.name] = p.role if p.role in ("identifier", "contact", "text") else p.kind
        return k

    def report(self) -> CleaningReport:
        rows = []
        dropped = []
        for orig in self.order:
            p = self.plans[orig]
            if p.drop:
                dropped.append({"column": orig, "reason": p.drop_reason})
                rows.append({"original name": orig, "clean name": "—", "detected type": p.kind,
                             "role": "dropped", "what was done": p.drop_reason})
                continue
            label = p.role if p.role in ("identifier", "contact", "text") else p.kind
            rows.append({"original name": orig, "clean name": p.name, "detected type": label,
                         "role": p.role, "what was done": "; ".join(p.notes) if p.notes else "no changes needed"})
        col_table = pd.DataFrame(rows)
        actions = pd.DataFrame(
            [{"action": k, "cells": int(v)} for k, v in self.stats_.most_common() if v > 0]
        )
        merges = []
        for orig in self.order:
            p = self.plans[orig]
            for m in p.cat_merges:
                merges.append({"column": p.name, **m})
        merges_df = pd.DataFrame(merges)
        return CleaningReport(col_table, actions, merges_df, pd.DataFrame(dropped), list(self.warnings))
