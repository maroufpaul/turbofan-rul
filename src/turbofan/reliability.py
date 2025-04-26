"""
Classical reliability helpers: fit a 2-param Weibull to C-MAPSS engine
lifetimes and compute survivor, hazard, and mean-residual-life curves.

Everything is implemented with *very* explicit intermediate variables so
math could be followed line by line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from lifelines import WeibullFitter  # MLE under the hood
from scipy.special import gamma, gammaincc


# 1.  Extract failure/lifetime time for each engine and return that as a numpy array
def lifetimes_from_training(df: pd.DataFrame) -> np.ndarray:
    """
    Parameters
    ----------
    df : training DataFrame that *already* has columns
         'unit' and 'cycle' (RUL column ignored here).

    Returns
    -------
    np.ndarray, shape (N_engines,)
        Total cycles each engine survived before failure.
    """
    # for each engine take the maximum cycle number (= lifetime)
    life = df.groupby("unit")["cycle"].max().values
    return life.astype(np.float64)  # lifelines prefers float


# 2.  Fit Weibull(λ, κ) via maximum-likelihood (take failure times as complete data and return λ, κ)
def fit_weibull(lifetimes: np.ndarray) -> Tuple[float, float]:
    """
    Returns
    -------
    λ (scale), κ (shape)
    """
    wf = WeibullFitter()
    # event_observed = 1  → all lifetimes are complete, no censoring
    wf.fit(lifetimes, event_observed=np.ones_like(lifetimes))
    lam = wf.lambda_ # scale parameter (λ)
    kappa = wf.rho_ # shape parameter (κ)
    return float(lam), float(kappa)



# 3.  Closed-form survivor, hazard, mean residual life
# --- survivor, hazard, MRL corrected to use  (t / λ)^κ  -----------------
def weibull_survivor(t: np.ndarray | float, lam: float, k: float):
    t = np.asarray(t, dtype=np.float64)
    return np.exp(- (t / lam) ** k)


def weibull_hazard(t: np.ndarray | float, lam: float, k: float):
    t = np.asarray(t, dtype=np.float64)
    return (k / lam) * (t / lam) ** (k - 1)


def weibull_mrl(t: np.ndarray | float, lam: float, k: float):
    """
    Mean Residual Life for S(t) = exp(-(t/lam)^k).

    L(t) = lam * Gamma(1+1/k) * gammaincc(1+1/k, (t/lam)^k) * exp((t/lam)^k) - t
    """
    t = np.asarray(t, dtype=np.float64)
    a = 1.0 + 1.0 / k                  # 1 + 1/k
    x = (t / lam) ** k
    # upper incomplete gamma piece
    integral = (
        lam
        * gamma(a)
        * gammaincc(a, x)              # Γ(a,x) / Γ(a)
        * np.exp(x)                    # divide by S(t) = exp(-x)
    )
    return integral - t
