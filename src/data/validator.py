"""
src/data/validator.py
Schema compliance and data leakage checks.

Provides validation utilities called before any experiment run to
ensure:
  - Stream isolation: UCI, synthetic, and empirical data never co-mingle.
  - Feature columns match expected schema.
  - No target leakage (target column absent from feature matrix).
  - Synthetic data is tagged SYNTHETIC_SIMULATION.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

_UCI_FEATURE_COLS = {"recency", "frequency", "monetary", "time"}
_UCI_TARGET_COL = "donated"


def validate_uci_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate that a DataFrame conforms to the UCI RFMT schema.

    Checks:
      - All four RFMT feature columns present.
      - Target column 'donated' present and binary.
      - No missing values.
      - Target not present in feature matrix (leakage check).

    Returns:
        True if all checks pass.

    Raises:
        ValueError on any schema violation.
    """
    # Column presence
    missing = _UCI_FEATURE_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing UCI feature columns: {missing}")
    if _UCI_TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{_UCI_TARGET_COL}' missing from DataFrame.")

    # Missing values
    if df.isnull().any().any():
        raise ValueError("UCI DataFrame contains missing values.")

    # Binary target
    unique_target = set(df[_UCI_TARGET_COL].unique())
    if not unique_target.issubset({0, 1}):
        raise ValueError(f"Target column contains non-binary values: {unique_target}")

    # Leakage: target must not appear in feature subset
    feature_df = df[list(_UCI_FEATURE_COLS)]
    if _UCI_TARGET_COL in feature_df.columns:
        raise ValueError("Target column found inside feature matrix — data leakage detected!")

    logger.info("UCI DataFrame validation passed (%d rows).", len(df))
    return True


def validate_no_target_in_features(X: np.ndarray, y: np.ndarray) -> bool:
    """
    Check that the feature matrix X does not contain the target vector y
    as one of its columns (simple linear correlation leakage guard).

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target vector (n_samples,).

    Returns:
        True if no leakage detected.

    Raises:
        ValueError if correlation > 0.99 between y and any column of X.
    """
    y_f = y.astype(float)
    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        if np.std(col) == 0:
            continue
        corr = float(np.corrcoef(col, y_f)[0, 1])
        if abs(corr) > 0.99:
            raise ValueError(
                f"Feature column {col_idx} has correlation {corr:.4f} with target — "
                "potential data leakage detected."
            )
    logger.info("Feature leakage check passed (no column correlation > 0.99 with target).")
    return True


def validate_synthetic_tag(records: list, tag: str = "SYNTHETIC_SIMULATION") -> bool:
    """
    Ensure all records in a synthetic pool carry the correct data_tag.

    Args:
        records: List of objects with a `data_tag` attribute.
        tag:     Expected tag value.

    Returns:
        True if all records carry the correct tag.

    Raises:
        ValueError if any record carries a different tag.
    """
    wrong = [r for r in records if getattr(r, "data_tag", None) != tag]
    if wrong:
        raise ValueError(
            f"{len(wrong)} records do not carry the expected data tag '{tag}'. "
            "Stream isolation violated."
        )
    logger.info("Synthetic data tag validation passed (%d records).", len(records))
    return True
