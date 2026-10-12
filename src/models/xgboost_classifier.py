"""
src/models/xgboost_classifier.py
Gradient Boosted Decision Trees via XGBoost.
DATA STREAM: UCI_PILOT (Phase 2 ML benchmark)
"""

from __future__ import annotations

from typing import Optional
import numpy as np

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError(
        "XGBoost is not installed. Run: pip install xgboost>=1.7.0"
    ) from e

from src.models.base_model import BaseClassifier


class XGBoostClassifier(BaseClassifier):
    """
    Gradient-boosted decision trees using XGBoost.

    scale_pos_weight compensates for the UCI dataset's class imbalance
    (~24% positive rate → recommended ratio ≈ (1-0.24)/0.24 ≈ 3.2).

    Early stopping is supported when eval_set is provided to fit().

    Args:
        n_estimators:       Maximum boosting rounds (default 300).
        learning_rate:      Step shrinkage (default 0.05).
        max_depth:          Max tree depth (default 6).
        subsample:          Row sampling ratio per tree (default 0.8).
        colsample_bytree:   Column sampling ratio per tree (default 0.8).
        min_child_weight:   Minimum sum of instance weight in child (default 1).
        gamma:              Minimum loss reduction for split (default 0.0).
        scale_pos_weight:   Balancing ratio for positive class (default 3.0).
        early_stopping_rounds: Stop if no improvement after N rounds.
        eval_metric:        Metric for early stopping (default 'logloss').
        random_state:       Random seed.
        n_jobs:             Parallel threads (-1 = all).
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        gamma: float = 0.0,
        scale_pos_weight: float = 3.0,
        early_stopping_rounds: int = 30,
        eval_metric: str = "logloss",
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self._early_stopping_rounds = early_stopping_rounds
        self._params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            scale_pos_weight=scale_pos_weight,
            eval_metric=eval_metric,
            use_label_encoder=False,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self._model = XGBClassifier(**self._params)

    @property
    def model_name(self) -> str:
        return "XGBoost"

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "XGBoostClassifier":
        """
        Fit XGBoost. If X_val and y_val provided, enables early stopping.

        Args:
            X:     Training feature matrix.
            y:     Training target vector.
            X_val: Optional validation features for early stopping.
            y_val: Optional validation targets for early stopping.
        """
        fit_kwargs: dict = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["early_stopping_rounds"] = self._early_stopping_rounds
            fit_kwargs["verbose"] = False

        self._model.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def feature_importances(self) -> Optional[np.ndarray]:
        return self._model.feature_importances_

    def get_params(self) -> dict:
        return dict(self._params)
