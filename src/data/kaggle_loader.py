"""
src/data/kaggle_loader.py
Kaggle Blood Donor Dataset loader and preprocessor.

DATA STREAM TAG: KAGGLE_PILOT (Availability classification)
This module handles the Kaggle availability dataset.
Target variable 'availability' does NOT represent emergency response probability.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.utils.logger import get_logger

logger = get_logger(__name__)

_EXPECTED_COLUMNS = [
    "city", "blood_group", "months_since_first_donation",
    "number_of_donation", "pints_donated", "availability"
]
_TARGET_COL = "availability"

def load_kaggle_dataframe(filepath: str | Path) -> pd.DataFrame:
    filepath = Path(filepath)
    logger.info("Loading Kaggle dataset from %s ...", filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    df = pd.read_csv(filepath, header=0)

    actual_cols = set(df.columns)
    expected_cols = set(_EXPECTED_COLUMNS)
    
    if not actual_cols.issubset(expected_cols) or not expected_cols.issubset(actual_cols):
        forbidden = actual_cols - expected_cols
        raise ValueError(
            f"Dataset contains unexpected or missing columns. Expected: {expected_cols}. Found extra/missing. "
            f"Forbidden columns (like PII) might be present: {forbidden}"
        )

    # Missing value check
    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    # Target conversion
    df[_TARGET_COL] = df[_TARGET_COL].map({"Yes": 1, "No": 0})
    
    if df[_TARGET_COL].isnull().any():
        raise ValueError("Target column contains unrecognized values (expected 'Yes' or 'No').")

    df[_TARGET_COL] = df[_TARGET_COL].astype(np.int32)
    return df

def extract_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    X = df.drop(columns=[_TARGET_COL])
    y = df[_TARGET_COL].to_numpy(dtype=np.int32)
    return X, y

def get_preprocessor():
    numeric_features = ['months_since_first_donation', 'number_of_donation', 'pints_donated']
    categorical_features = ['city', 'blood_group']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    return preprocessor

def get_stratified_cv_splits(X: pd.DataFrame, y: np.ndarray, n_splits: int = 5, seed: int = 42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(X, y))

def prepare_kaggle_data(data_path: str = "data/raw/blood_donor_dataset_modeling.csv", n_splits: int = 5, seed: int = 42) -> dict:
    df = load_kaggle_dataframe(data_path)
    X, y = extract_features_and_target(df)
    splits = get_stratified_cv_splits(X, y, n_splits=n_splits, seed=seed)

    positive_rate = y.mean()
    class_weight_positive = (1.0 - positive_rate) / positive_rate if positive_rate > 0 else 1.0

    return {
        "X": X,
        "y": y,
        "splits": splits,
        "class_weight_positive": class_weight_positive,
        "positive_rate": positive_rate,
        "n_samples": len(y),
    }
