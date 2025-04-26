"""
Train an XGBoost regressor for FD001 and save it to models/xgb_fd001.json.
Run from the project root:

    python -m turbofan.train_xgb
"""

from pathlib import Path
import numpy as np, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from turbofan.data_loading import read_cmaps, add_rul_labels
from turbofan.reliability  import fit_weibull, lifetimes_from_training
from turbofan.features     import build_feature_matrix

# -------- paths -----------------------------------------------------------
ROOT     = Path(__file__).resolve().parents[2]          # D:/turbofan-rul
RAW_DIR  = ROOT / "data/raw/FD001"
MODEL_FP = ROOT / "models/xgb_fd001.json"
MODEL_FP.parent.mkdir(exist_ok=True)

# -------- load + label training data -------------------------------------
train_raw = read_cmaps(RAW_DIR / "train_FD001.txt")
train     = add_rul_labels(train_raw)                   # <-- TRUE labels

# -------- Weibull params for MRL feature ---------------------------------
lam, k = fit_weibull(lifetimes_from_training(train))

# -------- build feature matrix -------------------------------------------
df = build_feature_matrix(train, lam, k, windows=(5, 30))
X  = df.drop(columns=["rul", "unit", "cycle"]).values
y  = df["rul"].values

# split by engine id (avoid leakage)
units = df["unit"].unique()
tr_u, val_u = train_test_split(units, test_size=0.2, random_state=42)
mask_tr = df["unit"].isin(tr_u)
mask_va = df["unit"].isin(val_u)

# -------- XGBoost config (fixed 800 trees) -------------------------------
xgb = XGBRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="mae",
    n_jobs=-1,
    random_state=42,
)

xgb.fit(
    X[mask_tr], y[mask_tr],
    eval_set=[(X[mask_va], y[mask_va])],
    verbose=False
)

best_mae = min(xgb.evals_result()['validation_0']['mae'])
print(f"Validation MAE: {best_mae:.2f} cycles")

# -------- save model ------------------------------------------------------
xgb.save_model(MODEL_FP)
print("✓ model saved to", MODEL_FP.relative_to(ROOT))
