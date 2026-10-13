"""
experiments/exp11_xgboost_donor_propensity.py
EXP-11: XGBoost donor return-propensity model on real UCI RFMT data.

Why this experiment exists
--------------------------
The Kaggle "availability" models (exp2) score ROC-AUC ≈ 0.47–0.51 on held-out
data — no better than chance — so that label carries no learnable signal from
its features. This experiment trains on the real UCI Blood Transfusion Service
Center data (748 donors, Hsin-Chu City, Taiwan; label = donated in March 2007)
and compares architecturally distinct predictors under one protocol:

  prior              constant base rate (reference for Brier skill)
  bernoulli_rate     one-feature logistic model on donations-per-month (F/T)
  logistic_rfmt      logistic regression on log-transformed engineered RFMT
  xgb_repo_default   XGBoost with configs/ml_models.yaml defaults on raw R/F/T
  xgb_tuned          XGBoost, engineered features, nested tuning (log loss)
  xgb_mono_platt     xgb_tuned + monotone constraints + Platt calibration

Protocol: 5 × 5 repeated stratified CV (25 outer folds). Every tuned or
calibrated variant does its tuning/calibration inside the outer training fold
(nested), so no test row influences a model scored on it. Differences against
the deployed candidate use the Nadeau–Bengio corrected resampled t-test,
because CV folds overlap and a plain paired t-test is anti-conservative.

Outputs
-------
  reports/xgboost_propensity/fold_metrics.csv
  reports/xgboost_propensity/summary.csv
  reports/xgboost_propensity/paired_tests.csv
  reports/xgboost_propensity/experiment_report.md
  visuals/xgboost_propensity/*.png
  models/donor_propensity_xgb.json        (portable deployed artefact)

Usage:
    python experiments/exp11_xgboost_donor_propensity.py [--repeats 5] [--n-iter 30] [--jobs 4]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import sklearn
import xgboost
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from src.data.uci_loader import EXPECTED_CONTENT_SHA256, content_sha256, download_uci_dataset, load_uci_dataframe
from src.models.donor_propensity import DonorPropensityModel, FEATURE_SETS
from src.models.donor_propensity_training import XGBPropensityEstimator, build_features

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
FINAL_VARIANT = "xgb_mono_platt"
REFERENCE_VARIANTS = ("logistic_rfmt", "xgb_repo_default", "xgb_tuned")

# Reference palette (dataviz skill, light mode): categorical slots in fixed order.
COLORS = {"prior": "#8a8984", "bernoulli_rate": "#6250d6", "logistic_rfmt": "#eb6834",
          "xgb_repo_default": "#e87ba4", "xgb_tuned": "#1baf7a", "xgb_mono_platt": "#2a78d6"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


# =================================================================
# Variants
# =================================================================

def _rate(X):
    X = np.asarray(X, dtype=float)
    return (X[:, 1] / np.maximum(X[:, 2], 1.0)).reshape(-1, 1)


def _log_engineered(X):
    return np.log1p(build_features(X, "engineered"))


def build_variants(n_iter: int, seed: int) -> dict:
    return {
        "prior": DummyClassifier(strategy="prior"),
        "bernoulli_rate": make_pipeline(FunctionTransformer(_rate), LogisticRegression(max_iter=1000)),
        "logistic_rfmt": make_pipeline(FunctionTransformer(_log_engineered), StandardScaler(),
                                       LogisticRegression(C=1.0, max_iter=1000)),
        # configs/ml_models.yaml xgboost.default (early stopping omitted: no validation split here)
        "xgb_repo_default": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8,
                                          colsample_bytree=0.8, min_child_weight=1, gamma=0.0,
                                          scale_pos_weight=3.0, eval_metric="logloss", tree_method="hist",
                                          random_state=seed, n_jobs=1),
        "xgb_tuned": XGBPropensityEstimator(feature_set="engineered", monotone=False, tune=True,
                                            n_iter=n_iter, calibrate=False, random_state=seed),
        "xgb_mono_platt": XGBPropensityEstimator(feature_set="engineered", monotone=True, tune=True,
                                                 n_iter=n_iter, calibrate=True, random_state=seed),
    }


def _fit_fold(name, estimator, X, y, train, test, fold_id):
    start = time.perf_counter()
    model = clone(estimator).fit(X[train], y[train])
    fit_seconds = time.perf_counter() - start
    p = np.clip(model.predict_proba(X[test])[:, 1], 1e-6, 1 - 1e-6)
    return {
        "variant": name, "fold": fold_id, "n_test": len(test),
        "roc_auc": roc_auc_score(y[test], p), "pr_auc": average_precision_score(y[test], p),
        "brier": brier_score_loss(y[test], p), "log_loss": log_loss(y[test], p),
        "fit_seconds": fit_seconds,
    }, (name, fold_id, test, p)


# =================================================================
# Statistics
# =================================================================

def corrected_resampled_ttest(diffs: np.ndarray, n_train: int, n_test: int) -> tuple[float, float]:
    """Nadeau & Bengio (2003) variance correction for repeated k-fold CV differences."""
    k = len(diffs)
    var = np.var(diffs, ddof=1)
    if var == 0:
        return float("nan"), float("nan")
    t = diffs.mean() / np.sqrt((1.0 / k + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), df=k - 1))


# =================================================================
# Figures
# =================================================================

def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def make_figures(oof: dict, y: np.ndarray, final_model: DonorPropensityModel, booster_gain: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import roc_curve

    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE})

    # 1. ROC — pooled out-of-fold predictions from the first CV repeat
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, p in oof.items():
        if name == "prior":
            continue
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=COLORS[name], linewidth=2, label=f"{name}  (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], color=COLORS["prior"], linewidth=1.2, linestyle="--", label="chance / prior (AUC 0.500)")
    _style(ax, "ROC — out-of-fold predictions (repeat 1 of CV)", "False positive rate", "True positive rate")
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)
    fig.tight_layout(); fig.savefig(out / "roc_curves.png", dpi=150); plt.close(fig)

    # 2. Reliability diagram
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], color=COLORS["prior"], linewidth=1.2, linestyle="--", label="perfect calibration")
    for name in ("logistic_rfmt", "xgb_repo_default", "xgb_tuned", "xgb_mono_platt"):
        frac, mean_pred = calibration_curve(y, oof[name], n_bins=8, strategy="quantile")
        ax.plot(mean_pred, frac, color=COLORS[name], linewidth=2, marker="o", markersize=5, label=name)
    _style(ax, "Calibration — out-of-fold, 8 quantile bins", "Mean predicted probability", "Observed donation rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    fig.tight_layout(); fig.savefig(out / "calibration.png", dpi=150); plt.close(fig)

    # 3. Feature importance (gain) of the deployed booster
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(booster_gain)
    values = [booster_gain[n] for n in names]
    order = np.argsort(values)
    ax.barh([names[i] for i in order], [values[i] for i in order], color=COLORS["xgb_mono_platt"], height=0.55)
    for i, idx in enumerate(order):
        ax.text(values[idx], i, f"  {values[idx]:.2f}", va="center", color=INK_2, fontsize=9)
    _style(ax, "Deployed model — mean split gain per feature", "Mean gain", "")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(out / "feature_gain.png", dpi=150); plt.close(fig)

    # 4. How the deployed model responds to recency, for several donation counts
    fig, ax = plt.subplots(figsize=(7, 5))
    recency = np.arange(0, 37)
    for colour, freq in zip(("#2a78d6", "#eb6834", "#1baf7a", "#6250d6"), (1, 3, 6, 12)):
        tenure = 24.0
        probs = [final_model.predict_proba(r, freq, max(tenure, r)) for r in recency]
        ax.plot(recency, probs, color=colour, linewidth=2, label=f"{freq} donation{'s' if freq > 1 else ''}, 24-month tenure")
    ax.axhline(final_model.base_rate, color=COLORS["prior"], linewidth=1.2, linestyle="--", label=f"base rate {final_model.base_rate:.3f}")
    _style(ax, "Deployed model — predicted return probability vs recency", "Months since last donation", "P(donates next month)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK)
    fig.tight_layout(); fig.savefig(out / "response_curves.png", dpi=150); plt.close(fig)


# =================================================================
# Main
# =================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--data-dir", default="data/raw/uci")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=30, help="Randomised-search draws per tuning run")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", default="models/donor_propensity_xgb.json")
    args = parser.parse_args()

    started = time.perf_counter()
    path = download_uci_dataset(str(ROOT / args.data_dir))
    df = load_uci_dataframe(path, verify=True)
    # Monetary is exactly 250 × Frequency in every row, so it is dropped as redundant.
    X = df[["recency", "frequency", "time"]].to_numpy(dtype=float)
    y = df["donated"].to_numpy(dtype=int)
    print(f"Loaded {len(y)} donors, positive rate {y.mean():.4f}, content sha256 {content_sha256(df)[:16]}…")

    variants = build_variants(args.n_iter, args.seed)
    cv = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=args.seed)
    splits = list(cv.split(X, y))
    tasks = [(name, est, train, test, i) for i, (train, test) in enumerate(splits) for name, est in variants.items()]
    print(f"Running {len(tasks)} fits ({len(variants)} variants × {len(splits)} outer folds, nested tuning inside)…")
    results = Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(_fit_fold)(name, est, X, y, train, test, i) for name, est, train, test, i in tasks)

    folds = pd.DataFrame([r[0] for r in results])
    oof = {name: np.zeros(len(y)) for name in variants}
    for name, fold_id, test, p in (r[1] for r in results):
        if fold_id < args.folds:  # first repeat covers every row exactly once
            oof[name][test] = p

    prior_brier = folds[folds.variant == "prior"].set_index("fold")["brier"]
    folds["brier_skill"] = 1 - folds["brier"] / folds["fold"].map(prior_brier)

    metrics = ["roc_auc", "pr_auc", "brier", "brier_skill", "log_loss", "fit_seconds"]
    summary = folds.groupby("variant")[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.loc[list(variants)].reset_index()

    n_test = len(y) // args.folds
    n_train = len(y) - n_test
    tests = []
    final_scores = folds[folds.variant == FINAL_VARIANT].set_index("fold")
    for other in REFERENCE_VARIANTS:
        other_scores = folds[folds.variant == other].set_index("fold")
        for metric in ("roc_auc", "brier", "log_loss"):
            diffs = (final_scores[metric] - other_scores[metric]).to_numpy()
            t, p = corrected_resampled_ttest(diffs, n_train, n_test)
            tests.append({"comparison": f"{FINAL_VARIANT} − {other}", "metric": metric,
                          "mean_diff": diffs.mean(), "std_diff": diffs.std(ddof=1), "t_corrected": t, "p_value": p})
    tests = pd.DataFrame(tests)

    # ---- Final model on all data -------------------------------------------------
    final_est = clone(variants[FINAL_VARIANT]).set_params(n_jobs=args.jobs).fit(X, y)
    final_row = summary[summary.variant == FINAL_VARIANT].iloc[0]
    metadata = {
        "model_version": f"uci-rfmt-xgb-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": "donated (1 = donated blood in March 2007)",
        "prediction_meaning": "Probability that a donor with >= 1 previous donation makes a voluntary "
                              "donation in the next month, given recency/frequency/tenure in months.",
        "not_for": "Emergency invitation-response probability; clinical eligibility; individual-level decisions "
                   "without human review.",
        "training_data": {
            "name": "UCI Blood Transfusion Service Center",
            "citation": "Yeh, I-Cheng, Yang, King-Jang, and Ting, Tao-Ming, 'Knowledge discovery on RFM model "
                        "using Bernoulli sequence', Expert Systems with Applications, 2008, "
                        "doi:10.1016/j.eswa.2008.07.018",
            "uci_page": "https://archive.ics.uci.edu/dataset/176/blood+transfusion+service+center",
            "population": "748 donors randomly selected from the donor database of the Blood Transfusion "
                          "Service Center, Hsin-Chu City, Taiwan",
            "n_rows": int(len(y)), "positive_rate": float(y.mean()),
            "content_sha256": EXPECTED_CONTENT_SHA256,
        },
        "training_ranges": {"recency_months": [float(X[:, 0].min()), float(X[:, 0].max())],
                            "frequency": [float(X[:, 1].min()), float(X[:, 1].max())],
                            "tenure_months": [float(X[:, 2].min()), float(X[:, 2].max())]},
        "nested_cv_estimate": {
            "protocol": f"{args.repeats}x{args.folds} repeated stratified CV, tuning + calibration nested",
            **{f"{m}_{s}": float(final_row[f"{m}_{s}"]) for m in ("roc_auc", "pr_auc", "brier", "log_loss") for s in ("mean", "std")},
        },
        "software": {"xgboost": xgboost.__version__, "scikit-learn": sklearn.__version__,
                     "python": platform.python_version()},
    }
    artefact = final_est.export_portable(X, metadata)
    model_path = ROOT / args.model_out
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(artefact, indent=1), encoding="utf-8")
    deployed = DonorPropensityModel.load(model_path)
    portable_p = np.array(deployed.predict_many(X))
    max_prob_diff = float(np.max(np.abs(portable_p - final_est.predict_proba(X)[:, 1])))
    assert max_prob_diff < 1e-5, max_prob_diff

    names = list(FEATURE_SETS[final_est.feature_set])
    gain = final_est.model_.get_booster().get_score(importance_type="gain")
    booster_gain = {names[int(k.lstrip("f"))] if k.startswith("f") and k[1:].isdigit() else k: v for k, v in gain.items()}
    for n in names:
        booster_gain.setdefault(n, 0.0)

    # ---- Write outputs -------------------------------------------------------------
    rep = ROOT / "reports/xgboost_propensity"
    rep.mkdir(parents=True, exist_ok=True)
    folds.to_csv(rep / "fold_metrics.csv", index=False)
    summary.to_csv(rep / "summary.csv", index=False)
    tests.to_csv(rep / "paired_tests.csv", index=False)
    make_figures(oof, y, deployed, booster_gain, ROOT / "visuals/xgboost_propensity")

    def fmt(row, m, d=3):
        return f"{row[f'{m}_mean']:.{d}f} ± {row[f'{m}_std']:.{d}f}"

    table = "\n".join(
        f"| `{r['variant']}` | {fmt(r, 'roc_auc')} | {fmt(r, 'pr_auc')} | {fmt(r, 'brier', 4)} | "
        f"{fmt(r, 'brier_skill')} | {fmt(r, 'log_loss')} | {r['fit_seconds_mean']:.2f} |"
        for _, r in summary.iterrows())
    test_table = "\n".join(
        f"| {r.comparison} | {r.metric} | {r.mean_diff:+.4f} | {r.t_corrected:.2f} | {r.p_value:.3f} |"
        for r in tests.itertuples())
    examples = [(2, 1, 2), (2, 6, 24), (4, 12, 48), (14, 3, 36), (24, 1, 24), (35, 10, 98)]
    example_rows = "\n".join(f"| {r} | {f} | {t} | {deployed.predict_proba(r, f, t):.3f} |" for r, f, t in examples)

    report = f"""# EXP-11 — XGBoost donor return-propensity model (UCI RFMT)

Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `experiments/exp11_xgboost_donor_propensity.py`
(xgboost {xgboost.__version__}, scikit-learn {sklearn.__version__}, Python {platform.python_version()}).
Runtime {time.perf_counter() - started:.0f} s.

## Data

* **Source:** UCI Blood Transfusion Service Center (Yeh, Yang & Ting, 2008). 748 donors from the
  Hsin-Chu City (Taiwan) centre; label = donated in March 2007. Positive rate {y.mean():.3f} (178 / 748).
* **Retrieved from:** the UCI host is blocked from this build environment, so the file was taken from
  PyCaret's public dataset mirror and **accepted only after matching** 748 rows, 178 positives and the
  content checksum `{EXPECTED_CONTENT_SHA256[:16]}…` (see `src/data/uci_loader.py`).
* **Features:** Recency (months since last donation), Frequency (donations), Time (months since first donation).
  Monetary is dropped because it equals 250 × Frequency in every row. Engineered: F/T, R/T,
  (T − R)/max(F − 1, 1).
* **Why not the Kaggle availability data:** in `reports/tables/availability_model_metrics.csv` every model
  scored held-out ROC-AUC 0.47–0.51, i.e. chance. That label is not predictable from its features.

## Protocol

{args.repeats} × {args.folds} repeated stratified cross-validation ({len(splits)} outer folds). `xgb_tuned` and
`xgb_mono_platt` run a {args.n_iter}-draw randomised search (inner 3-fold, log loss) and — for the latter —
Platt calibration on inner 5-fold out-of-fold margins, **inside each outer training fold**. Values are
mean ± std over outer folds. Brier skill = 1 − Brier / Brier(prior) on the same fold.

## Results

| Variant | ROC-AUC | PR-AUC | Brier | Brier skill | Log loss | Fit s |
|---|---|---|---|---|---|---|
{table}

### Deployed candidate vs references (Nadeau–Bengio corrected resampled t-test)

Negative differences are better for Brier and log loss; positive is better for ROC-AUC.

| Comparison | Metric | Mean diff | t | p |
|---|---|---|---|---|
{test_table}

## Deployed model

`{args.model_out}` — `{FINAL_VARIANT}` refit on all 748 rows.

* Hyperparameters: `{json.dumps(final_est.best_params_)}`
* Monotone constraints: `{json.dumps(artefact['monotone_constraints'])}`
* Platt calibration: a = {final_est.calib_a_:.4f}, b = {final_est.calib_b_:.4f}
* {len(artefact['trees'])} trees. The pure-Python evaluator reproduces the XGBoost booster to
  max |Δp| = {max_prob_diff:.2e} on all training rows.

Example predictions:

| Recency (months) | Frequency | Time (months) | P(donates next month) |
|---|---|---|---|
{example_rows}

Figures: `visuals/xgboost_propensity/roc_curves.png`, `calibration.png`, `feature_gain.png`, `response_curves.png`.

## Limitations

* 748 donors from one Taiwanese centre, observed for a single target month in 2007. Transfer to other
  populations, countries or years is unvalidated.
* The label is voluntary return donation, **not** a response to an emergency invitation. The UDI engine
  keeps its synthetic propensity; this probability is reported alongside rankings only.
* {int(df.duplicated().sum())} of the 748 rows exactly duplicate an earlier row (distinct donors with identical histories);
  standard stratified CV is used because those are separate individuals, not repeated records.
* Donors without any prior donation are outside the training population and receive no prediction.
"""
    (rep / "experiment_report.md").write_text(report, encoding="utf-8")
    print(summary.round(4).to_string(index=False))
    print(tests.round(4).to_string(index=False))
    print(f"Saved model to {model_path} (max |Δp| portable vs booster = {max_prob_diff:.2e})")


if __name__ == "__main__":
    main()
