"""
src/models/svm_classifier.py
Support Vector Machine classifier wrapping scikit-learn.
Probability calibration applied via Platt scaling (CalibratedClassifierCV).
DATA STREAM: UCI_PILOT (Phase 2 ML benchmark)
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from src.models.base_model import BaseClassifier


class SVMClassifier(BaseClassifier):
    """
    Kernel SVM with calibrated probability outputs.

    SVC does not natively output well-calibrated probabilities.
    We apply CalibratedClassifierCV with Platt scaling (sigmoid) to
    obtain reliable probability estimates for UDI ranking.

    Args:
        kernel:           Kernel type: 'rbf' (default) or 'linear'.
        C:                Regularization parameter (default 1.0).
        gamma:            Kernel coefficient: 'scale' or 'auto' or float.
        class_weight:     'balanced' recommended for UCI imbalance.
        calibration_cv:   Inner CV folds for Platt scaling.
        calibration_method: 'sigmoid' (Platt) or 'isotonic'.
        random_state:     Random seed.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str = "scale",
        class_weight: Optional[str] = "balanced",
        calibration_cv: int = 5,
        calibration_method: str = "sigmoid",
        random_state: int = 42,
    ) -> None:
        self._params = dict(
            kernel=kernel, C=C, gamma=gamma,
            class_weight=class_weight,
            probability=False,          # Platt scaling done externally
            random_state=random_state,
        )
        self._calibration_cv = calibration_cv
        self._calibration_method = calibration_method

        base_svc = SVC(**self._params)
        self._model = CalibratedClassifierCV(
            estimator=base_svc,
            method=calibration_method,
            cv=calibration_cv,
        )

    @property
    def model_name(self) -> str:
        return "SVM"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMClassifier":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def get_params(self) -> dict:
        return {
            **self._params,
            "calibration_cv": self._calibration_cv,
            "calibration_method": self._calibration_method,
        }
