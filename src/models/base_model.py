"""
src/models/base_model.py
Abstract base classifier interface.

Every ML classifier in this experiment system (LR, RF, SVM, XGBoost)
must inherit from BaseClassifier and implement its contract methods.
This guarantees the evaluator can treat all models uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseClassifier(ABC):
    """
    Abstract base class for all donor propensity classifiers.

    Subclasses must implement:
      - fit(X, y)
      - predict(X) -> np.ndarray of shape (n,) with integer class labels
      - predict_proba(X) -> np.ndarray of shape (n, 2) with class probabilities
      - model_name property -> str

    Optionally override:
      - get_params() -> dict  (for config / logging)
      - feature_importances() -> Optional[np.ndarray]
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier used in tables and logs."""
        ...

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseClassifier":
        """
        Fit the model on training data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Binary target vector of shape (n_samples,).

        Returns:
            Self (for method chaining).
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Integer array of shape (n_samples,) with values in {0, 1}.
        """
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict calibrated class probabilities.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Float array of shape (n_samples, 2) where column 1 gives
            Pr(donated=1 | features).
        """
        ...

    def get_params(self) -> dict:
        """Return model configuration parameters for logging."""
        return {}

    def feature_importances(self) -> Optional[np.ndarray]:
        """
        Return feature importance scores if available (e.g., RF, XGBoost).
        Returns None for models that do not expose feature importances (e.g., SVM).
        """
        return None

    def predict_proba_positive(self, X: np.ndarray) -> np.ndarray:
        """
        Convenience: return only the positive-class probability (column index 1).

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            1-D float array of shape (n_samples,) with Pr(donated=1).
        """
        return self.predict_proba(X)[:, 1]
