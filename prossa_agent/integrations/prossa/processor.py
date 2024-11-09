"""
Core integration between Prossa library and the agent framework.
"""

from typing import Dict, Any, Optional, Union
import pandas as pd
import logging
from datetime import datetime
from dataclasses import dataclass
from src.prossa.analyzer import (
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
from ...utils.errors import DatasetError, PreprocessingError

@dataclass
class ProcessingResult:
    """Result of preprocessing operations"""
    success: bool
    data: Union[pd.DataFrame, str]
    analysis: Dict[str, Any]
    plan: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

class ProsaProcessor:
    """Core processor integrating Prossa library capabilities"""
    
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
            
            # Perform analysis and ensure we get results
            missing_analysis = check_missing_values(df)
            outliers_analysis = check_outliers(df)
            dtypes_analysis = check_data_types(df)
            scaling_analysis = check_scaling_encoding(df)
            categorical_analysis = check_categorical_data(df)
            constant_analysis = check_constant_columns(df)
            imputation_analysis = check_imputation(df)
            
            analysis = {
                'missing_values': {
                    'has_missing': missing_analysis is not None,
                    'details': missing_analysis
                },
                'outliers': {
                    'has_outliers': outliers_analysis is not None,
                    'details': outliers_analysis
                },
                'data_types': dtypes_analysis,
                'scaling_needs': {
                    'needs_scaling': scaling_analysis is not None,
                    'details': scaling_analysis
                },
                'categorical_data': {
                    'has_categorical': categorical_analysis is not None,
                    'details': categorical_analysis
                },
                'constant_columns': constant_analysis,
                'imputation_needs': {
                    'needs_imputation': imputation_analysis is not None,
                    'details': imputation_analysis
                }
            }

            # Store analysis in memory
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
            raise PreprocessingError(f"Analysis failed: {str(e)}", error_context)

    async def execute_preprocessing(self, 
                                 df: pd.DataFrame, 
                                 analysis: Dict[str, Any]) -> ProcessingResult:
        """Execute preprocessing based on analysis"""
        try:
            processed_df = df.copy()
            
            # Get preprocessing plan from LLM
            model = self.model_selector.select_model(
                task_complexity=0.8,
                required_confidence=0.9
            )
            
            plan = await self._generate_preprocessing_plan(model, analysis)
            
            # Execute preprocessing steps based on analysis
            if analysis['missing_values']['has_missing']:
                processed_df = self.handlers._handle_missing_values(processed_df)
                
            if analysis['outliers']['has_outliers']:
                processed_df = self.handlers._handle_outliers(processed_df)
                
            if analysis['scaling_needs']['needs_scaling']:
                processed_df = self.handlers._apply_scaling(processed_df)
                
            if analysis['categorical_data']['has_categorical']:
                processed_df = self.handlers._encode_categorical(processed_df)
                
            return ProcessingResult(
                success=True,
                data=processed_df,
                analysis=analysis,
                plan=plan
            )
            
        except Exception as e:
            error_context = {
                'dataset_shape': df.shape if isinstance(df, pd.DataFrame) else None,
                'analysis': analysis
            }
            error = PreprocessingError(f"Preprocessing execution failed: {str(e)}", error_context)
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