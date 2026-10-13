# EXP-11 — XGBoost donor return-propensity model (UCI RFMT)

Generated 2026-10-13 12:35 UTC by `experiments/exp11_xgboost_donor_propensity.py`
(xgboost 3.4.1, scikit-learn 1.9.1, Python 3.13.16).
Runtime 76 s.

## Data

* **Source:** UCI Blood Transfusion Service Center (Yeh, Yang & Ting, 2008). 748 donors from the
  Hsin-Chu City (Taiwan) centre; label = donated in March 2007. Positive rate 0.238 (178 / 748).
* **Retrieved from:** the UCI host is blocked from this build environment, so the file was taken from
  PyCaret's public dataset mirror and **accepted only after matching** 748 rows, 178 positives and the
  content checksum `0c5bfc6d287ced28…` (see `src/data/uci_loader.py`).
* **Features:** Recency (months since last donation), Frequency (donations), Time (months since first donation).
  Monetary is dropped because it equals 250 × Frequency in every row. Engineered: F/T, R/T,
  (T − R)/max(F − 1, 1).
* **Why not the Kaggle availability data:** in `reports/tables/availability_model_metrics.csv` every model
  scored held-out ROC-AUC 0.47–0.51, i.e. chance. That label is not predictable from its features.

## Protocol

5 × 5 repeated stratified cross-validation (25 outer folds). `xgb_tuned` and
`xgb_mono_platt` run a 30-draw randomised search (inner 3-fold, log loss) and — for the latter —
Platt calibration on inner 5-fold out-of-fold margins, **inside each outer training fold**. Values are
mean ± std over outer folds. Brier skill = 1 − Brier / Brier(prior) on the same fold.

## Results

| Variant | ROC-AUC | PR-AUC | Brier | Brier skill | Log loss | Fit s |
|---|---|---|---|---|---|---|
| `prior` | 0.500 ± 0.000 | 0.238 ± 0.003 | 0.1813 ± 0.0013 | 0.000 ± 0.000 | 0.549 ± 0.003 | 0.00 |
| `bernoulli_rate` | 0.717 ± 0.030 | 0.393 ± 0.034 | 0.1716 ± 0.0035 | 0.054 ± 0.019 | 0.521 ± 0.008 | 0.01 |
| `logistic_rfmt` | 0.753 ± 0.037 | 0.528 ± 0.081 | 0.1539 ± 0.0099 | 0.151 ± 0.053 | 0.477 ± 0.026 | 0.01 |
| `xgb_repo_default` | 0.680 ± 0.043 | 0.429 ± 0.052 | 0.2072 ± 0.0167 | -0.143 ± 0.094 | 0.652 ± 0.051 | 0.09 |
| `xgb_tuned` | 0.753 ± 0.039 | 0.509 ± 0.066 | 0.1530 ± 0.0106 | 0.157 ± 0.057 | 0.475 ± 0.029 | 4.80 |
| `xgb_mono_platt` | 0.749 ± 0.036 | 0.500 ± 0.057 | 0.1535 ± 0.0096 | 0.154 ± 0.051 | 0.476 ± 0.026 | 5.04 |

### Deployed candidate vs references (Nadeau–Bengio corrected resampled t-test)

Negative differences are better for Brier and log loss; positive is better for ROC-AUC.

| Comparison | Metric | Mean diff | t | p |
|---|---|---|---|---|
| xgb_mono_platt − logistic_rfmt | roc_auc | -0.0036 | -0.29 | 0.774 |
| xgb_mono_platt − logistic_rfmt | brier | -0.0004 | -0.15 | 0.878 |
| xgb_mono_platt − logistic_rfmt | log_loss | -0.0008 | -0.09 | 0.930 |
| xgb_mono_platt − xgb_repo_default | roc_auc | +0.0695 | 3.40 | 0.002 |
| xgb_mono_platt − xgb_repo_default | brier | -0.0537 | -6.89 | 0.000 |
| xgb_mono_platt − xgb_repo_default | log_loss | -0.1756 | -7.86 | 0.000 |
| xgb_mono_platt − xgb_tuned | roc_auc | -0.0037 | -0.42 | 0.677 |
| xgb_mono_platt − xgb_tuned | brier | +0.0005 | 0.23 | 0.820 |
| xgb_mono_platt − xgb_tuned | log_loss | +0.0010 | 0.16 | 0.877 |

## Deployed model

`models/donor_propensity_xgb.json` — `xgb_mono_platt` refit on all 748 rows.

* Hyperparameters: `{"subsample": 0.6, "reg_lambda": 0.1, "n_estimators": 50, "min_child_weight": 10, "max_depth": 4, "learning_rate": 0.05, "gamma": 0.5, "colsample_bytree": 1.0}`
* Monotone constraints: `{"recency_months": -1, "frequency": 1, "tenure_months": 0, "donation_rate": 1, "recency_ratio": -1, "mean_gap_months": 0}`
* Platt calibration: a = 1.0273, b = 0.0116
* 50 trees. The pure-Python evaluator reproduces the XGBoost booster to
  max |Δp| = 9.71e-08 on all training rows.

Example predictions:

| Recency (months) | Frequency | Time (months) | P(donates next month) |
|---|---|---|---|
| 2 | 1 | 2 | 0.250 |
| 2 | 6 | 24 | 0.457 |
| 4 | 12 | 48 | 0.531 |
| 14 | 3 | 36 | 0.067 |
| 24 | 1 | 24 | 0.055 |
| 35 | 10 | 98 | 0.099 |

Figures: `visuals/xgboost_propensity/roc_curves.png`, `calibration.png`, `feature_gain.png`, `response_curves.png`.

## Limitations

* 748 donors from one Taiwanese centre, observed for a single target month in 2007. Transfer to other
  populations, countries or years is unvalidated.
* The label is voluntary return donation, **not** a response to an emergency invitation. The UDI engine
  keeps its synthetic propensity; this probability is reported alongside rankings only.
* 215 of the 748 rows exactly duplicate an earlier row (distinct donors with identical histories);
  standard stratified CV is used because those are separate individuals, not repeated records.
* Donors without any prior donation are outside the training population and receive no prediction.
