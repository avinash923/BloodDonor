"""
tests/test_phase2_kaggle_ml.py
Tests for Phase 2 Kaggle ML pipeline (availability classification).
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.data.kaggle_loader import load_kaggle_dataframe, get_preprocessor, extract_features_and_target

@pytest.fixture
def mock_dataset(tmp_path):
    df = pd.DataFrame({
        "city": ["Sydney", "Melbourne", "Sydney", "Brisbane", "Perth"],
        "blood_group": ["A+", "O-", "A+", "B+", "AB-"],
        "months_since_first_donation": [10, 20, 10, 15, 25],
        "number_of_donation": [2, 5, 2, 3, 6],
        "pints_donated": [50, 150, 50, 80, 180],
        "availability": ["Yes", "No", "Yes", "Yes", "No"]
    })
    filepath = tmp_path / "mock.csv"
    df.to_csv(filepath, index=False)
    return filepath

def test_kaggle_loader_target_and_features(mock_dataset):
    df = load_kaggle_dataframe(mock_dataset)
    X, y = extract_features_and_target(df)
    
    # Target validation
    assert y.tolist() == [1, 0, 1, 1, 0]
    
    # Forbidden-feature exclusion
    forbidden = {"name", "email", "password", "contact_number", "donor_id"}
    assert not any(f in X.columns for f in forbidden)

def test_preprocessing_pipeline_fit_transform(mock_dataset):
    df = load_kaggle_dataframe(mock_dataset)
    X, y = extract_features_and_target(df)
    
    preprocessor = get_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    
    assert X_transformed.shape[0] == 5
    assert X_transformed.shape[1] > X.shape[1]  # OHE expands columns

def test_pipeline_with_classifier(mock_dataset):
    df = load_kaggle_dataframe(mock_dataset)
    X, y = extract_features_and_target(df)
    
    preprocessor = get_preprocessor()
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    model.fit(X, y)
    probs = model.predict_proba(X)
    
    # Probability column ordering: [P(class=0), P(class=1)]
    assert probs.shape == (5, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert model.classes_.tolist() == [0, 1]

def test_unseen_categories(mock_dataset):
    df = load_kaggle_dataframe(mock_dataset)
    X, y = extract_features_and_target(df)
    
    preprocessor = get_preprocessor()
    preprocessor.fit(X)
    
    # Create new data with unseen city
    X_new = pd.DataFrame({
        "city": ["Hobart"],
        "blood_group": ["O-"],
        "months_since_first_donation": [12],
        "number_of_donation": [3],
        "pints_donated": [60]
    })
    
    # Should not raise error due to handle_unknown='ignore'
    X_new_transformed = preprocessor.transform(X_new)
    assert X_new_transformed is not None
