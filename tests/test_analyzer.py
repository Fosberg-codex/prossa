import pandas as pd
import numpy as np
import pytest
from prossa.analyzer import DatasetAnalyzer, analyze_dataset

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
    analyzer = DatasetAnalyzer(sample_dataset)
    analyzer.check_missing_values()
    assert 'Missing Values' in analyzer.recommendations
    assert any('missing values' in rec.lower() for rec in analyzer.recommendations['Missing Values'])
    assert any("Column 'A' has 1 (20.00%) missing values" in rec for rec in analyzer.recommendations['Missing Values'])

def test_check_outliers(sample_dataset):
    analyzer = DatasetAnalyzer(sample_dataset)
    analyzer.check_outliers()
    assert 'Outliers' in analyzer.recommendations
    # No outliers in this sample dataset
    assert any("No significant outliers detected" in rec for rec in analyzer.recommendations['Outliers'])

def test_check_data_types(sample_dataset):
    analyzer = DatasetAnalyzer(sample_dataset)
    analyzer.check_data_types()
    assert 'Data Types' in analyzer.recommendations
    assert any("Column 'A' has data type: float64" in rec for rec in analyzer.recommendations['Data Types'])
    assert any("Column 'B' has data type: object" in rec for rec in analyzer.recommendations['Data Types'])

def test_check_scaling_encoding(sample_dataset):
    analyzer = DatasetAnalyzer(sample_dataset)
    analyzer.check_scaling_encoding()
    assert 'Scaling and Encoding' in analyzer.recommendations
    assert any("Consider scaling numeric features" in rec for rec in analyzer.recommendations['Scaling and Encoding'])
    assert any("Consider encoding categorical features" in rec for rec in analyzer.recommendations['Scaling and Encoding'])

def test_check_categorical_data(sample_dataset):
    analyzer = DatasetAnalyzer(sample_dataset)
    analyzer.check_categorical_data()
    assert 'Categorical Data' in analyzer.recommendations
    assert any("Column 'B' has 3 unique categories" in rec for rec in analyzer.recommendations['Categorical Data'])

def test_check_constant_columns(sample_dataset):
    analyzer = DatasetAnalyzer(sample_dataset)
    analyzer.check_constant_columns()
    recommendations = analyzer.recommendations['Constant Columns']
    assert "The following columns have constant values:" in recommendations
    assert "E" in recommendations

def test_check_imputation(sample_dataset):
    analyzer = DatasetAnalyzer(sample_dataset)
    analyzer.check_imputation()
    
    assert 'Imputation' in analyzer.recommendations, "Imputation recommendation not found"
    
    imputation_recommendations = analyzer.recommendations['Imputation']
    assert len(imputation_recommendations) > 0, "No imputation recommendations found"
    
    expected_phrase = "the data has missing imputations, consider the following imputation techniques:"
    assert any(expected_phrase in rec.lower() for rec in imputation_recommendations), \
        f"Expected phrase '{expected_phrase}' not found in recommendations"
    
    expected_techniques = [
        "simple imputation: mean, median, or mode",
        "advanced imputation: knn imputer, mice, or domain-specific methods",
        "for time series data, consider forward fill or backward fill",
        "create a binary column to indicate where values were imputed"
    ]
    
    for technique in expected_techniques:
        assert any(technique in rec.lower() for rec in imputation_recommendations), \
            f"Expected technique '{technique}' not found in recommendations"
            

def test_analyze(sample_dataset):
    analyzer = DatasetAnalyzer(sample_dataset)
    analyzer.analyze()
    assert all(key in analyzer.recommendations for key in [
        'Missing Values', 'Outliers', 'Data Types', 'Scaling and Encoding',
        'Categorical Data', 'Constant Columns', 'Imputation'
    ])

def test_print_recommendations(sample_dataset, capsys):
    analyzer = DatasetAnalyzer(sample_dataset)
    analyzer.analyze()
    analyzer.print_recommendations()
    captured = capsys.readouterr()
    assert "MISSING VALUES:" in captured.out
    assert "OUTLIERS:" in captured.out
    assert "DATA TYPES:" in captured.out
    assert "SCALING AND ENCODING:" in captured.out
    assert "CATEGORICAL DATA:" in captured.out
    assert "CONSTANT COLUMNS:" in captured.out
    assert "IMPUTATION:" in captured.out
