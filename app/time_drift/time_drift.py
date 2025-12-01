import pandas as pd
import numpy as np
from typing import Literal, Dict
from drift_metrics import psi

Freq = Literal["D", "W", "M"]

def compute_time_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    datetime_col: str,
    freq: Freq = "M",
) -> pd.DataFrame:
    """Return PSI over time for each feature."""
    # concat for resampling convenience
    base = baseline_df.copy()
    curr = current_df.copy()

    base["__dataset"] = "baseline"
    curr["__dataset"] = "current"

    df = pd.concat([base, curr], ignore_index=True)
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    numeric_cols = [
        c for c in df.columns
        if c not in [datetime_col, "__dataset"] and pd.api.types.is_numeric_dtype(df[c])
    ]

    # define baseline period as earliest freq bucket
    df["__period"] = df[datetime_col].dt.to_period(freq).dt.to_timestamp()

    baseline_period = df[df["__dataset"] == "baseline"]["__period"].min()
    base_ref = df[(df["__dataset"] == "baseline") & (df["__period"] == baseline_period)]

    rows = []
    for period, group in df[df["__dataset"] == "current"].groupby("__period"):
        for col in numeric_cols:
            psi_val = psi(base_ref[col], group[col])
            rows.append(
                {
                    "period": period,
                    "feature": col,
                    "psi": psi_val,
                }
            )

    return pd.DataFrame(rows).sort_values(["feature", "period"])
