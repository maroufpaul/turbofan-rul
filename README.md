# Turbofan RUL Prediction — FD001 Case Study

> *A hybrid reliability + machine‑learning*

---

## 1  Project snapshot

| item | value |
|------|-------|
| Dataset | NASA C‑MAPSS **FD001** (100 train + 100 test engines) |
| Goal | Predict Remaining Useful Life (RUL) per engine |
| Best model | Random‑Forest (400× trees, min leaf 2) |
| Test results | **MAE 19.3 cycles  ·  RMSE 26.8  ·  PHM08 79 (sum)** |

---

## 2  Repo layout

```
turbofan-rul/
├─ data/            # raw NASA txt files
│  └─ raw/FD001/
├─ notebooks/       # Jupyter exploration & reports
│  └─ 01_eda_fd001.ipynb
├─ src/turbofan/    # feature & model code (importable)
│  ├─ data_loading.py
│  ├─ reliability.py
│  ├─ features.py
│  ├─ train_xgb.py
│  └─ …
├─ models/          # saved joblib / json models
│  ├─ rf_fd001.joblib
│  └─ xgb_fd001.json
├─ reports/
│  ├─ rf_fd001_predictions.csv
│  └─ figures/      # screenshots go here
└─ README.md        # this file
```

---

## 3  How to reproduce

```bash
# 0. clone & create venv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1.  Random‑Forest baseline
python -m turbofan.train_rf         # ~45 s

# 2.  Evaluate on test set & write CSV
jupyter nbconvert --execute notebooks/02_evaluate_rf.ipynb

# 3.  Optional: XGBoost experiment
python -m turbofan.train_xgb        # ~3 min
```

---

## 4  Workflow in pictures

> Add each screenshot to `reports/figures/` and commit.<br>
> Use <kbd>Win</kbd>+<kbd>Shift</kbd>+<kbd>S</kbd> (Windows Snip) or
> <kbd>⇧⌘4</kbd> (macOS) to grab the plot area.

| step | what to capture | embed code |
|------|-----------------|------------|
| **A** | Lifetime histogram | `![life_dist](reports/figures/lifetime_hist.png)` |
| **B** | Sensor drift plot (3 engines) | `![sensor_traces](reports/figures/sensor_traces.png)` |
| **C** | Weibull vs KM survivor | `![survivor](reports/figures/survivor_fit.png)` |
| **D** | MRL curve | `![mrl](reports/figures/mrl_curve.png)` |
| **E** | Feature‑importance RF | `![rf_feat_imp](reports/figures/rf_feat_imp.png)` |
| **F** | Test‑set error histogram | `![err_hist](reports/figures/error_hist.png)` |

*(copy‑paste the Markdown snippet after saving each PNG)*

---

## 5  Key equations

> See `src/turbofan/reliability.py` for full derivations.

* **Weibull survivor**  \(S(t)=e^{-(t/\lambda)^{\kappa}}\)
* **Mean residual life**  \(L(t)=\frac{1}{S(t)}\int_t^\infty S(u)\,du\;−\;t\)
* **PHM08 penalty**  \(\text{penalty}(d)=\begin{cases}e^{-d/13}-1,&d<0\\ e^{d/10}-1,&d\ge0\end{cases}\)

---

## 6  Results summary

| model | val MAE | test MAE | PHM08 (sum) |
|-------|--------:|---------:|-------------:|
| Weibull MRL baseline | 45.2 | 48.8 | 150.3 |
| **Random‑Forest** | **21.5** | **19.3** | **79.0** |
| XGBoost (quick) | 23.0 | 20.6 | 76.2 |

**Take‑away:** tree‑based ML with rolling‑sensor features halves error over a pure reliability model while staying fully interpretable.

---

## 7  Presenting live

1. Open the repo in **VS Code → Markdown preview** (`Ctrl+Shift+V`).
2. Scroll through each section while talking; click image thumbnails to zoom.
3. Run `python -m turbofan.train_rf` live (takes < 1 min) to show reproducibility.

---

## 8  References

* Saxena et al., “Damage Propagation Modeling for Aircraft Engine Run‑to‑Failure Simulation”, *PHM08.*
* XGBoost 3.0.0 documentation, callbacks & training API.

---

> © 2025 — Reliability Engineering final project, Marouf Paul & team.


