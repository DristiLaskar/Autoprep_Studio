"""Small pandas-backed stand-in for core.data_io.LoadedDataset (same interface, no polars needed)."""
import numpy as np
import pandas as pd


class PandasDataset:
    def __init__(self, df: pd.DataFrame, name: str = "test.csv"):
        self.df = df
        self.source_name = name
        self.source_format = "csv"
        self.is_lazy_native = True
        self.approx_size_bytes = None
        self.warning = None

    @classmethod
    def from_csv(cls, path: str, name: str = None):
        df = pd.read_csv(path, dtype=str, keep_default_na=False).replace("", np.nan)
        return cls(df, name or path.split("/")[-1])

    @property
    def columns(self):
        return list(self.df.columns)

    def row_count_estimate(self) -> int:
        return len(self.df)

    def sample(self, n: int) -> pd.DataFrame:
        return self.df if len(self.df) <= n else self.df.sample(n, random_state=42)

    def iter_chunks(self, chunk_rows: int = 100_000):
        for i in range(0, len(self.df), chunk_rows):
            yield self.df.iloc[i:i + chunk_rows]
