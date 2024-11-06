"""
Core integration between Prossa library and the agent framework.
"""

from typing import Dict, Any, Optional, Union, List
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from dataclasses import dataclass
from prossa import (
    check_missing_values,
    check_outliers,
    check_data_types,
    check_scaling_encoding,
    check_categorical_data,
    check_constant_columns,
    check_imputation
)
from ...core.preprocessing.handlers import PreprocessingHandlers
from ...llm.selector import ModelSelector
from ...memory.buffer import MemoryBuffer
from ...utils.errors import (
    PreprocessingError, 
    DatasetError,
    handle_preprocessing_error
)

@dataclass
class ProcessingResult:
    """Result of preprocessing operations"""
    success: bool
    data: Union[pd.DataFrame, str]
    analysis: Dict[str, Any]
    plan: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

class ProsaProcessor:
    """
    Core processor integrating Prossa library capabilities with the agent framework.
    """
    
    def __init__(self, use_gpu: bool = True):
        self._setup_logging()
        self.handlers = PreprocessingHandlers()
        self.model_selector = ModelSelector()
        self.memory = MemoryBuffer(use_gpu=use_gpu)

    def _setup_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def _validate_dataset(self, df: pd.DataFrame) -> None:
        """Validate input dataset"""
        if not isinstance(df, pd.DataFrame):
            raise DatasetError("Input must be a pandas DataFrame")
        if df.empty:
            raise DatasetError("DataFrame cannot be empty")
        if df.columns.empty:
            raise DatasetError("DataFrame must have at least one column")

    async def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Complete dataset analysis using Prossa library"""
        try:
            self._validate_dataset(df)
            
            analysis = {
                'missing_values': check_missing_values(df),
                'outliers': check_outliers(df),
                'data_types': check_data_types(df),
                'scaling_needs': check_scaling_encoding(df),
                'categorical_data': check_categorical_data(df),
                'constant_columns': check_constant_columns(df),
                'imputation_needs': check_imputation(df)
            }

            # Store analysis in memory for context
            await self.memory.add_interaction(
                query="dataset_analysis",
                response=str(analysis),
                metadata={
                    'timestamp': datetime.now(),
                    'dataset_shape': df.shape,
                    'columns': list(df.columns)
                }
            )

            return analysis

        except Exception as e:
            error_context = {'dataset_shape': df.shape if isinstance(df, pd.DataFrame) else None}
            raise handle_preprocessing_error(e, error_context)

    async def execute_preprocessing(self, 
                                 df: pd.DataFrame, 
                                 analysis: Dict[str, Any]) -> ProcessingResult:
        """Execute preprocessing based on analysis"""
        try:
            self._validate_dataset(df)
            processed_df = df.copy()
            
            # Get preprocessing plan from LLM
            model = self.model_selector.select_model(
                task_complexity=0.8,
                required_confidence=0.9
            )
            
            plan = await self._generate_preprocessing_plan(model, analysis)
            
            # Track applied transformations
            applied_steps = []
            
            # Execute preprocessing steps with validation
            if analysis['missing_values']['has_missing']:
                processed_df = self.handlers._handle_missing_values(processed_df)
                applied_steps.append('missing_values')
                
            if analysis['outliers']['has_outliers']:
                processed_df = self.handlers._handle_outliers(processed_df)
                applied_steps.append('outliers')
                
            if analysis['scaling_needs']['needs_scaling']:
                processed_df = self.handlers._apply_scaling(processed_df)
                applied_steps.append('scaling')
                
            if analysis['categorical_data']['has_categorical']:
                processed_df = self.handlers._encode_categorical(processed_df)
                applied_steps.append('categorical_encoding')
                
            return ProcessingResult(
                success=True,
                data=processed_df,
                analysis=analysis,
                plan={
                    **plan,
                    'applied_steps': applied_steps,
                    'execution_timestamp': datetime.now()
                }
            )
            
        except Exception as e:
            error_context = {
                'dataset_shape': df.shape if isinstance(df, pd.DataFrame) else None,
                'analysis': analysis
            }
            error = handle_preprocessing_error(e, error_context)
            
            return ProcessingResult(
                success=False,
                data=str(error),
                analysis=analysis,
                error={
                    'message': str(error),
                    'details': error.details,
                    'timestamp': datetime.now()
                }
            )

    async def _generate_preprocessing_plan(self, 
                                        model: Any, 
                                        analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate preprocessing plan using LLM"""
        prompt = f"""Based on the following dataset analysis, generate a detailed preprocessing plan:
        {analysis}
        
        Include specific steps for:
        1. Missing value handling
        2. Outlier treatment
        3. Feature scaling
        4. Encoding categorical variables
        5. Data type conversions
        
        Format the response as a structured plan with clear steps."""
        
        response = await model.generate(prompt)
        return {
            'steps': response.content,
            'confidence': response.confidence,
            'model_used': response.metadata.get('model')
        }