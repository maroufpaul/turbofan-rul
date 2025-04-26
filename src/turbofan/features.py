"""
Light-weight feature factory for C-MAPSS FD001.


comments try to explain the "why" as well as the "how".

Functions
---------
add_mrl_feature(df, lam, k)           # Weibull mean-residual-life per row
add_rolling_features(df, windows)     # rolling mean/std/min/max sensors
add_delta_features(df)                # first-order difference of sensors
build_feature_matrix(df, windows, ...)# high-level convenience wrapper
"""

from __future__ import annotations
from typing import List, Sequence

import numpy as np
import pandas as pd


# helper: which columns are sensors?
RAW_SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]  # 21 of them



# 1. Weibull Mean-Residual-Life as a feature
def add_mrl_feature(df: pd.DataFrame, lam: float, k: float) -> pd.Series:
    """
    Appends a column 'mrl_weibull' to *df* (does *not* modify in-place;
    always returns a copy, because side-effects bite).

    Parameters
    ----------
    lam, k : float
        Weibull parameters from reliability.fit_weibull().
    """
    from turbofan.reliability import weibull_mrl  # local import avoids cycle

    out = df.copy()
    out["mrl_weibull"] = weibull_mrl(out["cycle"].values, lam, k)
    return out


# ------------------------------------------------------------------
# 2. Rolling window statistics (mean / std / min / max)
# ------------------------------------------------------------------
def add_rolling_features(
    df: pd.DataFrame,
    windows: Sequence[int] = (5, 30),
    sensor_cols: Sequence[str] = RAW_SENSOR_COLS,
) -> pd.DataFrame:
    """
    Group-by 'unit', then compute windowed stats; prefix encodes window length.

    Result columns example:
        mean_5_sensor_1 , std_30_sensor_17 , min_5_sensor_3 , ...
    """
    out = df.copy()
    g = out.groupby("unit")

    for w in windows:
        rolled = (
            g[sensor_cols]
            .rolling(window=w, min_periods=1)
            .agg(["mean", "std", "min", "max"])
            .reset_index(level=0, drop=True)
        )
        # flatten MultiIndex ("sensor_1", "mean") -> "mean_5_sensor_1"
        rolled.columns = [f"{stat}_{w}_{col}" for col, stat in rolled.columns]
        out = pd.concat([out, rolled], axis=1)

    return out


# ------------------------------------------------------------------
# 3. First-order deltas (current - previous value)
# ------------------------------------------------------------------
def add_delta_features(
    df: pd.DataFrame, sensor_cols: Sequence[str] = RAW_SENSOR_COLS
) -> pd.DataFrame:
    """
    Δx_t = x_t − x_{t-1}; helps the model see short-term trends.
    """
    out = df.copy()
    deltas = (
        out.groupby("unit")[sensor_cols]
        .diff()  # NaN for the very first cycle of each engine
        .add_prefix("d1_")
    )
    return pd.concat([out, deltas], axis=1)


# ------------------------------------------------------------------
# 4. High-level convenience: build full matrix
# ------------------------------------------------------------------
def build_feature_matrix(
    df: pd.DataFrame,
    lam: float,
    k: float,
    windows: Sequence[int] = (5, 30),
) -> pd.DataFrame:
    """
    Pipe the three steps in order: MRL -> rolling -> deltas.

    Returns dataframe ready for ML (features + target 'rul').
    """
    x = add_mrl_feature(df, lam, k)
    x = add_rolling_features(x, windows=windows)
    x = add_delta_features(x)

    # Drop any rows with NaNs introduced by diff (only first cycle of units)
    x = x.dropna().reset_index(drop=True)
    return x
