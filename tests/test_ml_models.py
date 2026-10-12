"""
tests/test_ml_models.py
Unit tests for Phase 2: ML benchmark pipeline.

Verifies:
  1. UCI data loader integrity (schema, no missing values, M=250*F consistency).
  2. Stratified CV splits (class ratio preservation, no leakage).
  3. BaseClassifier interface contract for all four models.
  4. Evaluator output schema correctness.
  5. Determinism: identical results under fixed seed on repeated runs.
  6. Zero data leakage: scaler fitted only on training fold.

Run with: pytest tests/test_ml_models.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for import resolution
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.data.uci_loader import (
    load_uci_dataframe,
    extract_features_and_target,
    get_stratified_cv_splits,
    scale_fold,
    get_feature_names,
)
from src.models.base_model import BaseClassifier
from src.models.logistic_regression import LogisticRegressionClassifier
from src.models.random_forest import RandomForestClassifierWrapper
from src.models.svm_classifier import SVMClassifier
from src.models.xgboost_classifier import XGBoostClassifier
from src.models.evaluator import CrossValidatedEvaluator, ModelBenchmarkResult
from src.utils.seed import set_global_seed


# =================================================================
# Fixtures
# =================================================================

@pytest.fixture(scope="module")
def uci_dataframe(tmp_path_factory):
    """
    Load the UCI dataset from disk if available, else create a minimal
    synthetic stand-in with the correct schema for unit testing.
    This fixture does NOT download from the internet.
    """
    uci_path = Path("data/raw/uci/transfusion.data")
    if uci_path.exists():
        return load_uci_dataframe(uci_path)
    else:
        # Minimal synthetic stand-in: 100 rows, correct schema and M=250*F
        import pandas as pd
        rng = np.random.default_rng(42)
        n = 100
        frequency = rng.integers(1, 30, size=n)
        df = pd.DataFrame({
            "recency": rng.integers(0, 74, size=n),
            "frequency": frequency,
            "monetary": frequency * 250,
            "time": rng.integers(2, 98, size=n),
            "donated": rng.choice([0, 1], size=n, p=[0.76, 0.24]),
        })
        return df


@pytest.fixture(scope="module")
def Xy(uci_dataframe):
    """Return feature matrix X and target vector y."""
    return extract_features_and_target(uci_dataframe)


@pytest.fixture(scope="module")
def cv_splits(Xy):
    """Return stratified 5-fold splits."""
    X, y = Xy
    return get_stratified_cv_splits(X, y, n_splits=5, seed=42)


@pytest.fixture(params=[
    pytest.param("lr", id="LogisticRegression"),
    pytest.param("rf", id="RandomForest"),
    pytest.param("svm", id="SVM"),
    pytest.param("xgb", id="XGBoost"),
])
def classifier(request) -> BaseClassifier:
    """Parameterized fixture providing each of the four classifier types."""
    tag = request.param
    if tag == "lr":
        return LogisticRegressionClassifier(random_state=42, calibrate=False)
    elif tag == "rf":
        return RandomForestClassifierWrapper(n_estimators=10, random_state=42)
    elif tag == "svm":
        return SVMClassifier(random_state=42, calibration_cv=3)
    elif tag == "xgb":
        return XGBoostClassifier(n_estimators=10, random_state=42)
    raise ValueError(f"Unknown classifier tag: {tag}")


# =================================================================
# Test Group 1: Data Loader
# =================================================================

class TestUCIDataLoader:
    def test_column_names(self, uci_dataframe):
        assert list(uci_dataframe.columns) == [
            "recency", "frequency", "monetary", "time", "donated"
        ]

    def test_no_missing_values(self, uci_dataframe):
        assert uci_dataframe.isnull().sum().sum() == 0

    def test_monetary_equals_250_times_frequency(self, uci_dataframe):
        inconsistent = (uci_dataframe["monetary"] != uci_dataframe["frequency"] * 250).sum()
        assert inconsistent == 0, f"{inconsistent} rows violate M = 250*F"

    def test_target_is_binary(self, uci_dataframe):
        unique_values = set(uci_dataframe["donated"].unique())
        assert unique_values.issubset({0, 1}), f"Non-binary target values: {unique_values}"

    def test_minimum_instances(self, uci_dataframe):
        assert len(uci_dataframe) >= 50, "Dataset too small for meaningful CV"

    def test_feature_names(self):
        names = get_feature_names()
        assert names == ["recency", "frequency", "monetary", "time"]

    def test_feature_matrix_shape(self, Xy):
        X, y = Xy
        assert X.ndim == 2
        assert X.shape[1] == 4
        assert y.ndim == 1
        assert len(X) == len(y)


# =================================================================
# Test Group 2: Cross-Validation Splits
# =================================================================

class TestCVSplits:
    def test_correct_number_of_folds(self, cv_splits):
        assert len(cv_splits) == 5

    def test_indices_cover_all_samples(self, Xy, cv_splits):
        X, y = Xy
        n = len(y)
        all_test = set()
        for _, test_idx in cv_splits:
            all_test.update(test_idx.tolist())
        assert len(all_test) == n, "Test indices do not cover all samples exactly once"

    def test_no_train_test_overlap(self, cv_splits):
        for train_idx, test_idx in cv_splits:
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, "Data leakage: train and test indices overlap"

    def test_stratification_preserves_class_ratio(self, Xy, cv_splits):
        X, y = Xy
        global_rate = y.mean()
        for fold_idx, (_, test_idx) in enumerate(cv_splits):
            fold_rate = y[test_idx].mean()
            assert abs(fold_rate - global_rate) < 0.15, (
                f"Fold {fold_idx+1} positive rate {fold_rate:.3f} deviates more than "
                f"15pp from global rate {global_rate:.3f}"
            )

    def test_determinism_same_seed(self, Xy):
        X, y = Xy
        splits_a = get_stratified_cv_splits(X, y, n_splits=5, seed=42)
        splits_b = get_stratified_cv_splits(X, y, n_splits=5, seed=42)
        for (ta, va), (tb, vb) in zip(splits_a, splits_b):
            np.testing.assert_array_equal(ta, tb)
            np.testing.assert_array_equal(va, vb)

    def test_different_seeds_produce_different_splits(self, Xy):
        X, y = Xy
        splits_a = get_stratified_cv_splits(X, y, n_splits=5, seed=42)
        splits_b = get_stratified_cv_splits(X, y, n_splits=5, seed=99)
        # At least one fold should differ
        any_diff = any(
            not np.array_equal(va, vb)
            for (_, va), (_, vb) in zip(splits_a, splits_b)
        )
        assert any_diff, "Different seeds produced identical splits (unexpected)"


# =================================================================
# Test Group 3: Scaler — No Data Leakage
# =================================================================

class TestScalerLeakage:
    def test_scaler_fitted_on_train_only(self, Xy, cv_splits):
        X, y = Xy
        train_idx, test_idx = cv_splits[0]
        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]

        X_train_scaled, X_test_scaled, scaler = scale_fold(X_train_raw, X_test_raw)

        # Scaler mean should match training fold statistics
        np.testing.assert_allclose(
            scaler.mean_, X_train_raw.mean(axis=0), rtol=1e-5,
            err_msg="Scaler mean does not match training fold mean — possible leakage"
        )

    def test_scaled_train_is_zero_mean(self, Xy, cv_splits):
        X, y = Xy
        train_idx, test_idx = cv_splits[0]
        X_train_scaled, _, _ = scale_fold(X[train_idx], X[test_idx])
        np.testing.assert_allclose(
            X_train_scaled.mean(axis=0), np.zeros(4), atol=1e-9,
            err_msg="Scaled training data does not have zero mean"
        )


# =================================================================
# Test Group 4: Classifier Interface Contract
# =================================================================

class TestClassifierContract:
    def test_model_name_is_string(self, classifier):
        assert isinstance(classifier.model_name, str)
        assert len(classifier.model_name) > 0

    def test_fit_returns_self(self, Xy, classifier):
        X, y = Xy
        result = classifier.fit(X[:60], y[:60])
        assert result is classifier

    def test_predict_shape_and_dtype(self, Xy, classifier):
        X, y = Xy
        classifier.fit(X[:60], y[:60])
        preds = classifier.predict(X[60:])
        assert preds.ndim == 1
        assert preds.shape[0] == len(X[60:])
        assert set(preds.tolist()).issubset({0, 1}), "Predictions outside {0,1}"

    def test_predict_proba_shape(self, Xy, classifier):
        X, y = Xy
        classifier.fit(X[:60], y[:60])
        proba = classifier.predict_proba(X[60:])
        assert proba.ndim == 2
        assert proba.shape == (len(X[60:]), 2)

    def test_predict_proba_sums_to_one(self, Xy, classifier):
        X, y = Xy
        classifier.fit(X[:60], y[:60])
        proba = classifier.predict_proba(X[60:])
        np.testing.assert_allclose(
            proba.sum(axis=1), np.ones(len(X[60:])), atol=1e-6,
            err_msg="Probability rows do not sum to 1.0"
        )

    def test_predict_proba_in_unit_interval(self, Xy, classifier):
        X, y = Xy
        classifier.fit(X[:60], y[:60])
        proba = classifier.predict_proba(X[60:])
        assert proba.min() >= 0.0, "Negative probability detected"
        assert proba.max() <= 1.0, "Probability exceeds 1.0"

    def test_predict_proba_positive_is_1d(self, Xy, classifier):
        X, y = Xy
        classifier.fit(X[:60], y[:60])
        pos = classifier.predict_proba_positive(X[60:])
        assert pos.ndim == 1
        assert pos.shape[0] == len(X[60:])


# =================================================================
# Test Group 5: Evaluator
# =================================================================

class TestEvaluator:
    def test_evaluator_returns_correct_model_name(self, Xy, cv_splits):
        X, y = Xy
        model = LogisticRegressionClassifier(random_state=42, calibrate=False)
        evaluator = CrossValidatedEvaluator(cv_splits, X, y)
        result = evaluator.evaluate(model)
        assert result.model_name == "Logistic Regression"

    def test_evaluator_fold_count(self, Xy, cv_splits):
        X, y = Xy
        model = LogisticRegressionClassifier(random_state=42, calibrate=False)
        evaluator = CrossValidatedEvaluator(cv_splits, X, y)
        result = evaluator.evaluate(model)
        assert len(result.fold_results) == len(cv_splits)

    def test_aggregated_roc_auc_in_valid_range(self, Xy, cv_splits):
        X, y = Xy
        model = LogisticRegressionClassifier(random_state=42, calibrate=False)
        evaluator = CrossValidatedEvaluator(cv_splits, X, y)
        result = evaluator.evaluate(model)
        assert 0.0 <= result.mean_roc_auc <= 1.0

    def test_aggregated_brier_in_valid_range(self, Xy, cv_splits):
        X, y = Xy
        model = LogisticRegressionClassifier(random_state=42, calibrate=False)
        evaluator = CrossValidatedEvaluator(cv_splits, X, y)
        result = evaluator.evaluate(model)
        assert 0.0 <= result.mean_brier_score <= 1.0

    def test_pooled_predictions_length_matches_dataset(self, Xy, cv_splits):
        X, y = Xy
        model = LogisticRegressionClassifier(random_state=42, calibrate=False)
        evaluator = CrossValidatedEvaluator(cv_splits, X, y)
        result = evaluator.evaluate(model)
        assert len(result.all_y_true) == len(y)
        assert len(result.all_y_prob) == len(y)

    def test_summary_dict_has_all_keys(self, Xy, cv_splits):
        X, y = Xy
        model = LogisticRegressionClassifier(random_state=42, calibrate=False)
        evaluator = CrossValidatedEvaluator(cv_splits, X, y)
        result = evaluator.evaluate(model)
        d = result.summary_dict()
        expected_keys = {"Model", "ROC-AUC", "PR-AUC", "F1 (Binary)",
                         "F1 (Macro)", "Brier Score", "Accuracy", "Mean Fit Time (s)"}
        assert expected_keys.issubset(set(d.keys()))
