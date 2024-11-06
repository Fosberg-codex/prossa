"""
RAG-focused validation engine for the Prossa Agent framework.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from ...utils.errors import ValidationError, handle_preprocessing_error

@dataclass
class ValidationResult:
    """Comprehensive validation results"""
    confidence_score: float
    data_quality: Dict[str, float]
    preprocessing_quality: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: datetime
    
    @property
    def passed(self) -> bool:
        """Check if validation passed minimum thresholds"""
        return (
            self.confidence_score >= 0.7 and
            all(score >= 0.5 for score in self.data_quality.values()) and
            all(score >= 0.5 for score in self.preprocessing_quality.values())
        )

class ValidationEngine:
    """Validates preprocessing results and data quality"""
    
    def __init__(self, use_gpu: bool = True):
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self._setup_logging()
        self._initialize_components()
        
    def _setup_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def _initialize_components(self):
        try:
            # Initialize semantic similarity model for text comparison
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            self.logger.info("Validation engine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize validation engine: {str(e)}")
            raise

    async def validate_preprocessing(self, 
                                  original_df: pd.DataFrame,
                                  processed_df: pd.DataFrame,
                                  preprocessing_plan: Dict[str, Any]) -> ValidationResult:
        """Comprehensive validation of preprocessing results"""
        try:
            # Validate data quality
            data_quality = self._validate_data_quality(original_df, processed_df)
            
            # Validate preprocessing steps
            preprocessing_quality = self._validate_preprocessing_steps(
                processed_df, 
                preprocessing_plan
            )
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(
                data_quality,
                preprocessing_quality
            )
            
            result = ValidationResult(
                confidence_score=confidence,
                data_quality=data_quality,
                preprocessing_quality=preprocessing_quality,
                metadata={
                    "original_shape": original_df.shape,
                    "processed_shape": processed_df.shape,
                    "plan_steps": preprocessing_plan.get('applied_steps', [])
                },
                timestamp=datetime.now()
            )
            
            if not result.passed:
                raise ValidationError(
                    "Validation failed to meet minimum thresholds",
                    {
                        'confidence': confidence,
                        'data_quality': data_quality,
                        'preprocessing_quality': preprocessing_quality
                    }
                )
            
            return result
            
        except Exception as e:
            error_context = {
                'original_shape': original_df.shape,
                'processed_shape': processed_df.shape
            }
            if isinstance(e, ValidationError):
                raise
            raise handle_preprocessing_error(e, error_context)

    def _validate_data_quality(self, 
                             original_df: pd.DataFrame, 
                             processed_df: pd.DataFrame) -> Dict[str, float]:
        """Validate data quality metrics"""
        try:
            return {
                'completeness': self._calculate_completeness(processed_df),
                'consistency': self._calculate_consistency(original_df, processed_df),
                'validity': self._validate_data_types(processed_df),
                'integrity': self._check_data_integrity(processed_df)
            }
        except Exception as e:
            self.logger.error(f"Data quality validation error: {str(e)}")
            raise

    def _validate_preprocessing_steps(self,
                                   df: pd.DataFrame,
                                   plan: Dict[str, Any]) -> Dict[str, float]:
        """Validate preprocessing step execution"""
        try:
            results = {}
            applied_steps = plan.get('applied_steps', [])
            
            if 'missing_values' in applied_steps:
                results['missing_values'] = 1.0 if df.isnull().sum().sum() == 0 else 0.0
                
            if 'scaling' in applied_steps:
                results['scaling'] = self._validate_scaling(df)
                
            if 'categorical_encoding' in applied_steps:
                results['encoding'] = self._validate_encoding(df)
                
            if 'outliers' in applied_steps:
                results['outliers'] = self._validate_outliers(df)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Preprocessing validation error: {str(e)}")
            raise

    def _calculate_completeness(self, df: pd.DataFrame) -> float:
        """Calculate data completeness score"""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        return 1.0 - (missing_cells / total_cells)

    def _calculate_consistency(self, 
                             original_df: pd.DataFrame, 
                             processed_df: pd.DataFrame) -> float:
        """Calculate data consistency score"""
        if len(processed_df) != len(original_df):
            return 0.0
            
        consistent_cols = sum(
            1 for col in processed_df.columns
            if col in original_df.columns
        )
        return consistent_cols / len(original_df.columns)

    def _validate_data_types(self, df: pd.DataFrame) -> float:
        """Validate data types"""
        valid_types = 0
        for dtype in df.dtypes:
            if dtype in ['int64', 'float64', 'object', 'bool', 'datetime64[ns]']:
                valid_types += 1
        return valid_types / len(df.dtypes)

    def _check_data_integrity(self, df: pd.DataFrame) -> float:
        """Check data integrity"""
        try:
            integrity_checks = [
                df.index.is_unique,
                not df.empty,
                all(col.strip() == col for col in df.columns),  # Clean column names
                all(df[col].notna().any() for col in df.columns)  # No empty columns
            ]
            return sum(1 for check in integrity_checks if check) / len(integrity_checks)
        except Exception:
            return 0.0

    def _validate_scaling(self, df: pd.DataFrame) -> float:
        """Validate numerical scaling"""
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_cols) == 0:
            return 1.0
            
        scaled_cols = 0
        for col in numerical_cols:
            stats = df[col].describe()
            if -3 <= stats['mean'] <= 3 and 0 <= stats['std'] <= 2:
                scaled_cols += 1
                
        return scaled_cols / len(numerical_cols)

    def _validate_encoding(self, df: pd.DataFrame) -> float:
        """Validate categorical encoding"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) == 0:
            return 1.0
            
        return 0.0 if len(categorical_cols) > 0 else 1.0

    def _validate_outliers(self, df: pd.DataFrame) -> float:
        """Validate outlier removal"""
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_cols) == 0:
            return 1.0
            
        outlier_free_cols = 0
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[col][(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            if len(outliers) / len(df) < 0.01:  # Less than 1% outliers
                outlier_free_cols += 1
                
        return outlier_free_cols / len(numerical_cols)

    def _calculate_confidence(self,
                            data_quality: Dict[str, float],
                            preprocessing_quality: Dict[str, float]) -> float:
        """Calculate overall confidence score"""
        quality_score = sum(data_quality.values()) / len(data_quality)
        preprocessing_score = sum(preprocessing_quality.values()) / len(preprocessing_quality)
        
        # Weight quality higher than preprocessing
        return 0.7 * quality_score + 0.3 * preprocessing_score