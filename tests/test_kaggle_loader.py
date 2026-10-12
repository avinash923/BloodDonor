"""
tests/test_kaggle_loader.py
Tests for the Kaggle dataset loader.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.kaggle_loader import load_kaggle_dataframe, prepare_kaggle_data

@pytest.fixture
def mock_valid_csv(tmp_path):
    df = pd.DataFrame({
        "city": ["Sydney", "Melbourne", "Sydney", "Melbourne"],
        "blood_group": ["A+", "O-", "A+", "O-"],
        "months_since_first_donation": [10, 20, 10, 20],
        "number_of_donation": [2, 5, 2, 5],
        "pints_donated": [50, 150, 50, 150],
        "availability": ["Yes", "No", "Yes", "No"]
    })
    filepath = tmp_path / "valid.csv"
    df.to_csv(filepath, index=False)
    return filepath

@pytest.fixture
def mock_invalid_cols_csv(tmp_path):
    df = pd.DataFrame({
        "city": ["Sydney"],
        "blood_group": ["A+"],
        "months_since_first_donation": [10],
        "number_of_donation": [2],
        "pints_donated": [50],
        "availability": ["Yes"],
        "name": ["John Doe"] # PII
    })
    filepath = tmp_path / "invalid_cols.csv"
    df.to_csv(filepath, index=False)
    return filepath

def test_load_valid(mock_valid_csv):
    df = load_kaggle_dataframe(mock_valid_csv)
    assert len(df) == 4
    assert set(df.columns) == {"city", "blood_group", "months_since_first_donation", "number_of_donation", "pints_donated", "availability"}
    assert df["availability"].tolist() == [1, 0, 1, 0]

def test_load_invalid_cols(mock_invalid_cols_csv):
    with pytest.raises(ValueError, match="Forbidden columns"):
        load_kaggle_dataframe(mock_invalid_cols_csv)

def test_missing_values(tmp_path):
    df = pd.DataFrame({
        "city": ["Sydney", None],
        "blood_group": ["A+", "O-"],
        "months_since_first_donation": [10, 20],
        "number_of_donation": [2, 5],
        "pints_donated": [50, 150],
        "availability": ["Yes", "No"]
    })
    filepath = tmp_path / "missing.csv"
    df.to_csv(filepath, index=False)
    
    with pytest.raises(ValueError, match="Dataset contains missing values"):
        load_kaggle_dataframe(filepath)

def test_invalid_target(tmp_path):
    df = pd.DataFrame({
        "city": ["Sydney"],
        "blood_group": ["A+"],
        "months_since_first_donation": [10],
        "number_of_donation": [2],
        "pints_donated": [50],
        "availability": ["Maybe"]
    })
    filepath = tmp_path / "invalid_target.csv"
    df.to_csv(filepath, index=False)
    
    with pytest.raises(ValueError, match="unrecognized values"):
        load_kaggle_dataframe(filepath)

def test_prepare_kaggle_data(mock_valid_csv):
    data = prepare_kaggle_data(data_path=str(mock_valid_csv), n_splits=2)
    assert "X" in data
    assert "y" in data
    assert len(data["y"]) == 4
    assert data["positive_rate"] == 0.5
