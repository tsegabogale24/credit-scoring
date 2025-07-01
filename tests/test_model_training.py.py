import pytest
import pandas as pd
import numpy as np
from train import preprocess_data

def test_preprocess_data_drops_customer_id():
    """Test that CustomerId column is dropped"""
    test_data = pd.DataFrame({
        'CustomerId': [1, 2, 3],
        'is_high_risk': [0, 1, 0],
        'feature1': [0.1, 0.2, 0.3]
    })
    
    X, _ = preprocess_data(test_data)
    assert 'CustomerId' not in X.columns

def test_preprocess_data_handles_missing_values():
    """Test that rows with missing values are dropped"""
    test_data = pd.DataFrame({
        'is_high_risk': [0, 1, np.nan],
        'feature1': [0.1, np.nan, 0.3],
        'feature2': [1.0, 2.0, 3.0]
    })
    
    X, y = preprocess_data(test_data)
    assert len(X) == 1  # Only one complete row
    assert len(y) == 1

def test_preprocess_data_returns_correct_target():
    """Test that target column is correctly separated"""
    test_data = pd.DataFrame({
        'is_high_risk': [0, 1, 0],
        'feature1': [0.1, 0.2, 0.3]
    })
    
    X, y = preprocess_data(test_data)
    assert 'is_high_risk' not in X.columns
    assert y.name == 'is_high_risk'
    assert list(y.values) == [0, 1, 0]