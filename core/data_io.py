"""
data_io.py
----------
Loads datasets from files (CSV/TSV/other delimited text, Excel, JSON, Parquet)
or from manual UI entry, and normalizes them into a common internal
representation: a polars LazyFrame plus metadata (row count estimate, source
path, whether the source supports true lazy/streaming reads).

Design notes
------------
- CSV/TSV/Parquet/NDJSON are read with polars `scan_*` functions, which are
  lazy: no data is pulled into memory until `.collect()` (optionally in
  streaming mode) or `.fetch(n)` is called. This is what lets us "support"
  very large files without loading them fully.
- Plain JSON (a single JSON array/object, not NDJSON) and Excel do not have
  a true streaming reader available in this environment, so we read them
  eagerly but guard with a file-size warning so the user knows why.
- Manual entry comes from a pandas DataFrame (Streamlit's data_editor
  returns pandas) and is converted to a polars DataFrame, then wrapped as
  a LazyFrame so it flows through the same downstream code paths.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Iterator, List, Optional

import pandas as pd
import polars as pl

# Above this size, plain (non-lazy-capable) formats trigger a warning to the user.
EAGER_FORMAT_WARN_BYTES = 200 * 1024 * 1024  # 200 MB

# Above this row count, profiling/leakage/ablation steps switch to sampling.
LARGE_ROW_THRESHOLD = 200_000


@dataclass
class LoadedDataset:
    lazy_frame: pl.LazyFrame
    source_name: str
    source_format: str
    is_lazy_native: bool  # True if the reader is truly streaming/lazy end-to-end
    approx_size_bytes: Optional[int] = None
    warning: Optional[str] = None
    source_path: Optional[str] = None  # on-disk path, set for lazy-native formats
                                        # (csv/tsv/delimited/parquet/ndjson) so DuckDB
                                        # can query the file directly, out-of-core

    # ---- helpers used by the automatic pipeline (core/autopilot.py) ----------
    @property
    def columns(self) -> List[str]:
        return list(self.lazy_frame.collect_schema().keys())

    def sample(self, n: int) -> pd.DataFrame:
        """Up to n rows as a pandas DataFrame (the whole file if it is smaller)."""
        total = self.row_count_estimate()
        return sample_to_pandas(self.lazy_frame, n, total_rows=total)

    def iter_chunks(self, chunk_rows: int = 100_000) -> Iterator[pd.DataFrame]:
        """Yield the COMPLETE dataset as pandas chunks (bounded memory for lazy sources)."""
        if not self.is_lazy_native:
            full = self.lazy_frame.collect()
            for offset in range(0, full.height, chunk_rows):
                yield full.slice(offset, chunk_rows).to_pandas()
            return
        total = self.row_count_estimate()
        for offset in range(0, total, chunk_rows):
            chunk_lf = self.lazy_frame.slice(offset, chunk_rows)
            try:
                chunk = chunk_lf.collect(engine="streaming")
            except Exception:
                chunk = chunk_lf.collect()
            if chunk.height == 0:
                break
            yield chunk.to_pandas()

    def row_count_estimate(self) -> int:
        """Cheap-ish row count. Uses streaming count for lazy-native sources,
        falls back to a full collect for eager sources (already in memory)."""
        try:
            if self.is_lazy_native:
                return int(
                    self.lazy_frame.select(pl.len()).collect(engine="streaming")[0, 0]
                )
            return int(self.lazy_frame.select(pl.len()).collect()[0, 0])
        except Exception:
            # Fallback: non-streaming collect count
            return int(self.lazy_frame.select(pl.len()).collect()[0, 0])


def _detect_format(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    if ext in ("csv",):
        return "csv"
    if ext in ("tsv",):
        return "tsv"
    if ext in ("txt", "dat"):
        return "delimited"
    if ext in ("xlsx", "xls"):
        return "excel"
    if ext in ("json",):
        return "json"
    if ext in ("ndjson", "jsonl"):
        return "ndjson"
    if ext in ("parquet", "pq"):
        return "parquet"
    return "unknown"


def sniff_delimiter(sample_bytes: bytes) -> str:
    """Very small heuristic delimiter sniffer for .txt/.dat files."""
    text = sample_bytes[:5000].decode("utf-8", errors="ignore")
    first_line = text.splitlines()[0] if text.splitlines() else ""
    counts = {d: first_line.count(d) for d in [",", "\t", ";", "|"]}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def load_from_path(path: str, original_filename: Optional[str] = None) -> LoadedDataset:
    """Load a dataset file from disk into a LoadedDataset, choosing the most
    memory-efficient reader available for the detected format."""
    name = original_filename or os.path.basename(path)
    fmt = _detect_format(name)
    size = os.path.getsize(path) if os.path.exists(path) else None
    warning = None

    if fmt == "csv":
        lf = pl.scan_csv(path, infer_schema_length=0)
        return LoadedDataset(lf, name, fmt, True, size, source_path=path)

    if fmt == "tsv":
        lf = pl.scan_csv(path, separator="\t", infer_schema_length=0)
        return LoadedDataset(lf, name, fmt, True, size, source_path=path)

    if fmt == "delimited":
        with open(path, "rb") as f:
            sample = f.read(5000)
        delim = sniff_delimiter(sample)
        lf = pl.scan_csv(path, separator=delim, infer_schema_length=0)
        return LoadedDataset(lf, name, fmt, True, size, source_path=path)

    if fmt == "parquet":
        lf = pl.scan_parquet(path)
        return LoadedDataset(lf, name, fmt, True, size, source_path=path)

    if fmt == "ndjson":
        lf = pl.scan_ndjson(path)
        return LoadedDataset(lf, name, fmt, True, size, source_path=path)

    if fmt == "json":
        if size and size > EAGER_FORMAT_WARN_BYTES:
            warning = (
                f"Plain JSON files are read fully into memory (no streaming reader "
                f"available for JSON arrays). This file is {size / 1e6:.1f} MB — "
                f"consider converting to NDJSON or Parquet for very large data."
            )
        df = pl.read_json(path)
        return LoadedDataset(df.lazy(), name, fmt, False, size, warning)

    if fmt == "excel":
        if size and size > EAGER_FORMAT_WARN_BYTES:
            warning = (
                f"Excel files are read fully into memory (no streaming reader for "
                f".xlsx/.xls). This file is {size / 1e6:.1f} MB — consider exporting "
                f"to CSV or Parquet for very large data."
            )
        pdf = pd.read_excel(path, dtype=str)   # text in, the cleaner types it
        df = pl.from_pandas(pdf)
        return LoadedDataset(df.lazy(), name, fmt, False, size, warning)

    # Unknown extension: try sniffing as delimited text, else raise.
    try:
        with open(path, "rb") as f:
            sample = f.read(5000)
        delim = sniff_delimiter(sample)
        lf = pl.scan_csv(path, separator=delim, infer_schema_length=0)
        return LoadedDataset(lf, name, "delimited (guessed)", True, size,
                              "Unrecognized extension — parsed as delimited text.",
                              source_path=path)
    except Exception as e:
        raise ValueError(f"Could not parse file '{name}': {e}")


def load_from_manual_table(pdf: pd.DataFrame, name: str = "manual_entry") -> LoadedDataset:
    """Convert a manually-entered pandas DataFrame (from st.data_editor) into
    a LoadedDataset. Empty rows/columns are dropped."""
    pdf = pdf.dropna(how="all").dropna(axis=1, how="all")
    pdf = pdf.astype("string")            # user-typed cells are text; the cleaner types them
    df = pl.from_pandas(pdf)
    return LoadedDataset(df.lazy(), name, "manual", False, None)


def sample_to_pandas(lf: pl.LazyFrame, n: int, total_rows: Optional[int] = None,
                      seed: int = 42) -> pd.DataFrame:
    """Return a pandas DataFrame sample of at most n rows, WITHOUT ever
    materializing the full dataset in memory when it's large.

    Strategy: add a row index, derive a deterministic pseudo-random keep/drop
    decision per row via a multiplicative hash, and filter — all of which the
    polars streaming engine can execute chunk-by-chunk without holding the
    whole table in RAM. Only the sampled rows are ever collected fully.
    """
    if total_rows is not None and total_rows <= n:
        return lf.collect(engine="streaming").to_pandas()

    if total_rows is None:
        # Unknown size (e.g. manual entry / already-eager source) — safe to collect.
        collected = lf.collect()
        if len(collected) <= n:
            return collected.to_pandas()
        return collected.sample(n=n, seed=seed).to_pandas()

    frac = min(1.0, (n / total_rows) * 1.15)  # slight overshoot, trimmed below
    hashed = (
        lf.with_row_index("__row_idx")
        .with_columns(
            (((pl.col("__row_idx") * 2654435761 + seed) % 1_000_000) / 1_000_000)
            .alias("__rand")
        )
        .filter(pl.col("__rand") < frac)
        .drop(["__row_idx", "__rand"])
    )
    try:
        sampled = hashed.collect(engine="streaming").to_pandas()
    except Exception:
        sampled = hashed.collect().to_pandas()
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed)
    return sampled.reset_index(drop=True)
