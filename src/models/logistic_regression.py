"""
src/models/logistic_regression.py
Regularized Logistic Regression classifier wrapping scikit-learn.
DATA STREAM: UCI_PILOT (Phase 2 ML benchmark)
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from src.models.base_model import BaseClassifier


class LogisticRegressionClassifier(BaseClassifier):
    """
    L2-regularized Logistic Regression baseline with optional Platt scaling
    calibration. Serves as the linear boundary reference model.

    Args:
        C:              Regularization inverse strength (default 1.0).
        penalty:        Regularization type: 'l1' or 'l2' (default 'l2').
        solver:         Optimization solver (default 'lbfgs').
        max_iter:       Maximum solver iterations (default 1000).
        class_weight:   Handle class imbalance; 'balanced' recommended
                        for the ~24% positive-rate UCI dataset.
        calibrate:      If True, wrap with CalibratedClassifierCV (Platt).
        calibration_cv: Number of inner CV folds for calibration (default 5).
        random_state:   Random seed.
    """

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        class_weight: Optional[str] = "balanced",
        calibrate: bool = True,
        calibration_cv: int = 5,
        random_state: int = 42,
    ) -> None:
        self._params = dict(
            C=C, penalty=penalty, solver=solver, max_iter=max_iter,
            class_weight=class_weight, random_state=random_state,
        )
        self._calibrate = calibrate
        self._calibration_cv = calibration_cv

        base = LogisticRegression(**self._params)

        if calibrate:
            self._model = CalibratedClassifierCV(
                estimator=base, method="sigmoid", cv=calibration_cv
            )
        else:
            self._model = base

    @property
    def model_name(self) -> str:
        return "Logistic Regression"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionClassifier":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def get_params(self) -> dict:
        return {**self._params, "calibrate": self._calibrate}
