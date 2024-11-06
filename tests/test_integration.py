"""
Integration tests for the Prossa Agent framework.
"""

import pytest
import pandas as pd
import numpy as np
from prossa_agent.core.prossa_agent import ProsaAgent
from prossa_agent.utils.config import SystemConfig
from prossa_agent.utils.errors import DatasetError

@pytest.fixture
def complex_dataset():
    """Create a complex dataset for testing"""
    return pd.DataFrame({
        'numeric_clean': [1, 2, 3, 4, 5],
        'numeric_missing': [1, 2, np.nan, 4, 5],
        'numeric_outliers': [1, 2, 1000, 4, 5],
        'categorical_clean': ['A', 'B', 'A', 'B', 'A'],
        'categorical_missing': ['X', 'Y', np.nan, 'X', 'Y'],
        'mixed_types': [1, 'two', 3, 'four', 5]
    })

@pytest.fixture
def agent():
    """Create ProsaAgent instance"""
    config = SystemConfig(
        use_gpu=False,
        max_memory_items=100,
        min_confidence=0.7
    )
    return ProsaAgent(config)

@pytest.mark.asyncio
async def test_end_to_end_processing(agent, complex_dataset):
    """Test complete processing pipeline"""
    result = await agent.process_dataset(
        dataset=complex_dataset,
        task_description="Prepare this dataset for machine learning"
    )
    
    assert result.success == True
    assert isinstance(result.data, pd.DataFrame)
    assert result.confidence >= 0.7
    
    processed_df = result.data
    
    # Verify preprocessing results
    assert processed_df.isnull().sum().sum() == 0  # No missing values
    assert all(dtype in ['int64', 'float64'] for dtype in processed_df.dtypes)  # All numeric
    assert len(processed_df) == len(complex_dataset)  # Row count preserved

@pytest.mark.asyncio
async def test_invalid_input_handling(agent):
    """Test handling of invalid inputs"""
    with pytest.raises(DatasetError):
        await agent.process_dataset(None)
    
    with pytest.raises(DatasetError):
        await agent.process_dataset(pd.DataFrame())

@pytest.mark.asyncio
async def test_memory_integration(agent, complex_dataset):
    """Test memory system integration"""
    # Process dataset twice
    result1 = await agent.process_dataset(complex_dataset)
    result2 = await agent.process_dataset(complex_dataset)
    
    # Check if memory is working
    history = await agent.get_processing_history()
    assert len(history) >= 2
    
    # Verify memory contents
    assert any(entry['metadata'].get('dataset_shape') == complex_dataset.shape 
              for entry in history) 