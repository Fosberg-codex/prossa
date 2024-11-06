"""
Core preprocessing handlers for the Prossa Agent framework.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Any
import logging
from ...utils.errors import PreprocessingError

class PreprocessingHandlers:
    """Handles various preprocessing operations"""
    
    def __init__(self):
        self._setup_logging()
        self.encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, StandardScaler] = {}

    def _setup_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on data types"""
        try:
            df_copy = df.copy()
            for column in df_copy.columns:
                if df_copy[column].dtype in ['int64', 'float64']:
                    # Numerical - use median
                    df_copy[column].fillna(df_copy[column].median(), inplace=True)
                else:
                    # Categorical - use mode
                    df_copy[column].fillna(df_copy[column].mode()[0], inplace=True)
            return df_copy
        except Exception as e:
            self.logger.error(f"Missing value handling failed: {str(e)}")
            raise PreprocessingError("Missing value handling failed", {"error": str(e)})

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method"""
        try:
            df_copy = df.copy()
            for column in df_copy.select_dtypes(include=['int64', 'float64']).columns:
                Q1 = df_copy[column].quantile(0.25)
                Q3 = df_copy[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df_copy[column] = df_copy[column].clip(lower=lower_bound, upper=upper_bound)
            return df_copy
        except Exception as e:
            self.logger.error(f"Outlier handling failed: {str(e)}")
            raise PreprocessingError("Outlier handling failed", {"error": str(e)})

    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply standard scaling to numerical columns"""
        try:
            df_copy = df.copy()
            numerical_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns
            
            if len(numerical_cols) > 0:
                for col in numerical_cols:
                    scaler = StandardScaler()
                    df_copy[col] = scaler.fit_transform(df_copy[[col]])
                    self.scalers[col] = scaler
                    
            return df_copy
        except Exception as e:
            self.logger.error(f"Scaling failed: {str(e)}")
            raise PreprocessingError("Scaling failed", {"error": str(e)})

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        try:
            df_copy = df.copy()
            categorical_cols = df_copy.select_dtypes(include=['object']).columns
            
            for column in categorical_cols:
                encoder = LabelEncoder()
                df_copy[column] = encoder.fit_transform(df_copy[column].astype(str))
                self.encoders[column] = encoder
                
            return df_copy
        except Exception as e:
            self.logger.error(f"Categorical encoding failed: {str(e)}")
            raise PreprocessingError("Categorical encoding failed", {"error": str(e)})

    def get_feature_names(self) -> Dict[str, Any]:
        """Get encoded feature names and scaling info"""
        return {
            'encoders': {col: list(enc.classes_) for col, enc in self.encoders.items()},
            'scaled_features': list(self.scalers.keys())
        }