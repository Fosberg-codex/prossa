import pandas as pd
import numpy as np
import pytest
from prossa.utils import (
    check_missing_values,
    check_outliers,
    check_data_types,
    check_scaling_encoding,
    check_categorical_data,
    check_constant_columns,
    check_imputation
)

@pytest.fixture
def sample_dataset():
    return pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': ['x', 'y', 'z', 'x', 'y'],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5],
        'D': [10, 20, 30, 40, 50],
        'E': ['a', 'a', 'a', 'a', 'a']
    })

def test_check_missing_values(sample_dataset):
    recommendation = check_missing_values(sample_dataset)
    assert recommendation is None
    
    
def test_check_outliers(sample_dataset):
    recommendation = check_outliers(sample_dataset)
    assert recommendation is None

def test_check_data_types(sample_dataset):
    recommendation = check_data_types(sample_dataset)
    assert recommendation is None
    
def test_check_scaling_encoding(sample_dataset):
    recommendation = check_scaling_encoding(sample_dataset)
    assert recommendation is None

def test_check_categorical_data(sample_dataset):
    recommendation = check_categorical_data(sample_dataset)
    assert recommendation is None

def test_check_constant_columns(sample_dataset):
    recommendation = check_constant_columns(sample_dataset)
    assert len(recommendation) > 0

def test_check_imputation(sample_dataset):
    recommendation = check_imputation(sample_dataset)
    assert recommendation is None