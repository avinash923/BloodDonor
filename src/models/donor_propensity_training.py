"""
src/models/donor_propensity_training.py
Training-side helpers for the XGBoost donor return-propensity model.

``XGBPropensityEstimator`` is a scikit-learn compatible estimator that takes
the raw UCI Recency / Frequency / Time matrix and, inside ``fit``:

  1. builds the requested feature set (raw RFT or RFT + engineered ratios),
  2. optionally tunes XGBoost hyperparameters with an inner stratified CV
     scored by log loss (a proper scoring rule, so probabilities stay honest),
  3. optionally fits Platt scaling on out-of-fold margins, and
  4. refits the booster on all rows it was given.

Because all tuning and calibration happen inside ``fit``, wrapping the
estimator in an outer cross-validation loop gives a nested, leakage-free
estimate of generalisation performance.

``export_portable`` writes the artefact read by src/models/donor_propensity.py.

DATA STREAM: UCI_PILOT
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.models.donor_propensity import FEATURE_SETS, PortableTreeEnsemble, engineer_row

# Domain priors: lapsing longer lowers, donating more often raises return probability.
MONOTONE_DIRECTIONS: dict[str, int] = {
    "recency_months": -1,
    "frequency": 1,
    "tenure_months": 0,
    "donation_rate": 1,
    "recency_ratio": -1,
    "mean_gap_months": 0,
}

SEARCH_SPACE: dict[str, list] = {
    "max_depth": [1, 2, 3, 4],
    "learning_rate": [0.01, 0.02, 0.05, 0.1, 0.2],
    "n_estimators": [50, 100, 200, 400, 800],
    "min_child_weight": [1, 3, 5, 10, 20],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.67, 1.0],
    "reg_lambda": [0.1, 1.0, 5.0, 20.0],
    "gamma": [0.0, 0.5, 1.0],
}


def build_features(X_rft: np.ndarray, feature_set: str) -> np.ndarray:
    """Vectorised version of ``engineer_row`` for an (n, 3) R/F/T matrix."""
    X_rft = np.asarray(X_rft, dtype=np.float64)
    return np.array([engineer_row(r, f, t, feature_set) for r, f, t in X_rft], dtype=np.float64)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class XGBPropensityEstimator(BaseEstimator, ClassifierMixin):
    """Tuned, optionally monotone and Platt-calibrated XGBoost on RFMT features."""

    def __init__(
        self,
        feature_set: str = "engineered",
        monotone: bool = True,
        tune: bool = True,
        n_iter: int = 30,
        calibrate: bool = True,
        params: Optional[dict] = None,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.feature_set = feature_set
        self.monotone = monotone
        self.tune = tune
        self.n_iter = n_iter
        self.calibrate = calibrate
        self.params = params
        self.random_state = random_state
        self.n_jobs = n_jobs

    # ------------------------------------------------------------
    def _base_params(self) -> dict:
        base = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                    random_state=self.random_state, n_jobs=self.n_jobs)
        if self.monotone:
            names = FEATURE_SETS[self.feature_set]
            base["monotone_constraints"] = "(" + ",".join(str(MONOTONE_DIRECTIONS[n]) for n in names) + ")"
        return base

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBPropensityEstimator":
        y = np.asarray(y).astype(int)
        F = build_features(X, self.feature_set)
        self.classes_ = np.array([0, 1])
        base = self._base_params()

        if self.tune:
            search = RandomizedSearchCV(
                XGBClassifier(**base), SEARCH_SPACE, n_iter=self.n_iter, scoring="neg_log_loss",
                cv=StratifiedKFold(3, shuffle=True, random_state=self.random_state),
                random_state=self.random_state, n_jobs=1, refit=False,
            )
            search.fit(F, y)
            self.best_params_ = dict(search.best_params_)
            self.inner_cv_log_loss_ = float(-search.best_score_)
        else:
            self.best_params_ = dict(self.params or {})

        params = {**base, **self.best_params_}
        self.calib_a_, self.calib_b_ = 1.0, 0.0
        if self.calibrate:
            oof = np.zeros(len(y))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=self.random_state).split(F, y):
                oof[te] = XGBClassifier(**params).fit(F[tr], y[tr]).predict(F[te], output_margin=True)
            platt = LogisticRegression(C=1e6, max_iter=1000).fit(oof.reshape(-1, 1), y)
            self.calib_a_, self.calib_b_ = float(platt.coef_[0, 0]), float(platt.intercept_[0])

        self.model_ = XGBClassifier(**params).fit(F, y)
        self.positive_rate_ = float(y.mean())
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        margin = self.model_.predict(build_features(X, self.feature_set), output_margin=True)
        return self.calib_a_ * margin + self.calib_b_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # ------------------------------------------------------------
    def export_portable(self, X_reference: np.ndarray, metadata: dict) -> dict:
        """Export trees + calibration as JSON and prove the export reproduces the booster.

        The booster's base margin is recovered empirically (booster margin minus
        the summed leaves) instead of parsing version-specific config strings.
        """
        booster = self.model_.get_booster()
        names = list(FEATURE_SETS[self.feature_set])
        booster.feature_names = None  # dump splits as f0, f1, ... for a stable format
        trees = [json.loads(t) for t in booster.get_dump(dump_format="json")]
        F = build_features(X_reference, self.feature_set)
        booster_margin = self.model_.predict(F, output_margin=True).astype(np.float64)
        leaves_only = PortableTreeEnsemble(trees, names, 0.0)
        offsets = booster_margin - np.array([leaves_only.margin(row) for row in F])
        base_margin = float(np.median(offsets))
        portable = PortableTreeEnsemble(trees, names, base_margin)
        max_abs_margin_diff = float(np.max(np.abs(np.array([portable.margin(r) for r in F]) - booster_margin)))
        if max_abs_margin_diff > 1e-4:
            raise RuntimeError(f"Portable export diverges from booster (max |Δmargin| = {max_abs_margin_diff})")
        return {
            "format": "portable-xgboost-trees/v1",
            "feature_set": self.feature_set,
            "feature_names": names,
            "monotone_constraints": {n: MONOTONE_DIRECTIONS[n] for n in names} if self.monotone else None,
            "hyperparameters": self.best_params_,
            "base_margin": base_margin,
            "calibration": {"method": "platt_on_out_of_fold_margins" if self.calibrate else "none",
                            "a": self.calib_a_, "b": self.calib_b_},
            "export_check": {"rows_compared": int(len(F)), "max_abs_margin_diff": max_abs_margin_diff},
            "trees": trees,
            **metadata,
        }
