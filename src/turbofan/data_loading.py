"""
Data ingestion & basic label utilities for NASA C-MAPSS datasets.
"""

from pathlib import Path
import pandas as pd

#parents[0] = …/src/turbofan
#parents[1] = …/src
#parents[2] = …/turbofan-rul ← our project root
ROOT = Path(__file__).resolve().parents[2]



 
# column names per the competition description
COL_NAMES = (
    ["unit", "cycle", "setting_1", "setting_2", "setting_3"] 
    + [f"sensor_{i}" for i in range(1,22)]
    
)



def read_cmaps(file_path: Path | str) -> pd.DataFrame: # can be str or Path
    # build an absolute Path object first
    path = Path(file_path) 
    if not path.is_absolute():
        path = ROOT / path           # resolve relative to project root

    
    df = pd.read_csv(
        path,                        # << changed
        sep=r"\s+",
        header=None,
        names=COL_NAMES,
        engine="c",
        dtype="float32",
    )
    df["unit"] = df["unit"].astype("int16")
    df["cycle"] = df["cycle"].astype("int32")
    return df


def add_rul_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add Remaining-Useful-Life column (training set only)."""
    max_cycle = df.groupby("unit")["cycle"].transform("max") # based on unique vaules in unit column group the data, then select the cycle column
    df = df.copy()
    df["rul"] = max_cycle - df["cycle"]
    return df
