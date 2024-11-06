"""
Tests for the ValidationEngine component.
"""

import pytest
import pandas as pd
import numpy as np
from prossa_agent.core.execution.validator import ValidationEngine
from prossa_agent.utils.errors import ValidationError

@pytest.fixture
def validation_engine():
    """Create ValidationEngine instance"""
    return ValidationEngine(use_gpu=False)

@pytest.fixture
def original_df():
    """Create original dataset"""
    return pd.DataFrame({
        'numeric': [1, 2, np.nan, 4, 5],
        'categorical': ['A', 'B', 'A', np.nan, 'B']
    })

@pytest.fixture
def processed_df():
    """Create processed dataset"""
    return pd.DataFrame({
        'numeric': [1, 2, 3, 4, 5],
        'categorical': [0, 1, 0, 1, 1]
    })

@pytest.mark.asyncio
async def test_validate_preprocessing(validation_engine, original_df, processed_df):
    """Test preprocessing validation"""
    plan = {
        'applied_steps': ['missing_values', 'categorical_encoding']
    }
    
    result = await validation_engine.validate_preprocessing(
        original_df=original_df,
        processed_df=processed_df,
        preprocessing_plan=plan
    )
    
    assert result.passed == True
    assert result.confidence_score >= 0.7
    assert 'completeness' in result.data_quality
    assert 'missing_values' in result.preprocessing_quality

@pytest.mark.asyncio
async def test_validation_failure(validation_engine, original_df):
    """Test validation failure handling"""
    # Create bad processed data
    bad_df = pd.DataFrame({
        'numeric': [1, 2, np.nan, 4, 5],  # Still has missing values
        'categorical': ['A', 'B', 'A', np.nan, 'B']  # Not encoded
    })
    
    plan = {
        'applied_steps': ['missing_values', 'categorical_encoding']
    }
    
    with pytest.raises(ValidationError):
        await validation_engine.validate_preprocessing(
            original_df=original_df,
            processed_df=bad_df,
            preprocessing_plan=plan
        ) 