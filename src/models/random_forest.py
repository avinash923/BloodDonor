"""
src/models/random_forest.py
Random Forest ensemble classifier wrapping scikit-learn.
DATA STREAM: UCI_PILOT (Phase 2 ML benchmark)
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.base_model import BaseClassifier


class RandomForestClassifierWrapper(BaseClassifier):
    """
    Ensemble bagging classifier using randomly selected feature subsets.
    Exposes out-of-bag (OOB) validation score and feature importances.

    Args:
        n_estimators:    Number of decision trees (default 200).
        max_depth:       Maximum tree depth; None = fully grown (default).
        min_samples_split: Minimum samples to split an internal node.
        min_samples_leaf:  Minimum samples in each leaf node.
        max_features:    Features per split: 'sqrt' or 'log2'.
        class_weight:    'balanced' recommended for imbalanced UCI target.
        oob_score:       Compute out-of-bag accuracy estimate.
        n_jobs:          Parallel jobs (-1 = use all CPUs).
        random_state:    Random seed.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        class_weight: Optional[str] = "balanced",
        oob_score: bool = True,
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        self._params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self._model = RandomForestClassifier(**self._params)

    @property
    def model_name(self) -> str:
        return "Random Forest"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifierWrapper":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def feature_importances(self) -> Optional[np.ndarray]:
        return self._model.feature_importances_

    def oob_score(self) -> Optional[float]:
        """Return Out-of-Bag accuracy score if oob_score=True during fit."""
        return getattr(self._model, "oob_score_", None)

    def get_params(self) -> dict:
        return dict(self._params)
