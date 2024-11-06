"""
Tests for the ProsaProcessor component.
"""

import pytest
import pandas as pd
import numpy as np
from prossa_agent.integrations.prossa.processor import ProsaProcessor
from prossa_agent.utils.errors import DatasetError

@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing"""
    return pd.DataFrame({
        'numeric': [1, 2, np.nan, 4, 5],
        'categorical': ['A', 'B', 'A', np.nan, 'B'],
        'integer': [1, 2, 3, 4, 5],
        'outlier': [1, 2, 100, 4, 5]
    })

@pytest.fixture
def processor():
    """Create ProsaProcessor instance"""
    return ProsaProcessor(use_gpu=False)

@pytest.mark.asyncio
async def test_analyze_dataset(processor, sample_dataset):
    """Test dataset analysis"""
    analysis = await processor.analyze_dataset(sample_dataset)
    
    assert 'missing_values' in analysis
    assert 'outliers' in analysis
    assert 'data_types' in analysis
    assert analysis['missing_values']['has_missing'] == True
    assert analysis['outliers']['has_outliers'] == True

@pytest.mark.asyncio
async def test_execute_preprocessing(processor, sample_dataset):
    """Test preprocessing execution"""
    analysis = await processor.analyze_dataset(sample_dataset)
    result = await processor.execute_preprocessing(sample_dataset, analysis)
    
    assert result.success == True
    assert isinstance(result.data, pd.DataFrame)
    assert result.data.isnull().sum().sum() == 0  # No missing values
    assert 'missing_values' in result.plan['applied_steps']

@pytest.mark.asyncio
async def test_invalid_dataset(processor):
    """Test handling of invalid datasets"""
    with pytest.raises(DatasetError):
        await processor.analyze_dataset(None)
    
    with pytest.raises(DatasetError):
        await processor.analyze_dataset(pd.DataFrame()) 