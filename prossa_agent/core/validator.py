from typing import Dict, List, Optional, Union, Any
from enum import Enum
import numpy as np
from dataclasses import dataclass

class ValidationType(Enum):
    ANALYSIS = "analysis"
    FEATURE_ENGINEERING = "feature_engineering"
    OUTLIER_DETECTION = "outlier_detection"
    MISSING_VALUES = "missing_values"
    SCALING = "scaling"
    ENCODING = "encoding"

@dataclass
class ValidationThresholds:
    """Defines confidence thresholds for different validation types"""
    analysis: float = 0.85
    feature_engineering: float = 0.90
    outlier_detection: float = 0.95
    missing_values: float = 0.90
    scaling: float = 0.85
    encoding: float = 0.85

class ConfidenceValidator:
    def __init__(self):
        """Initialize the validator with default thresholds"""
        self.thresholds = ValidationThresholds()
        self.validation_history: List[Dict[str, Any]] = []
    
    def validate_recommendation(self,
                              recommendation: Dict[str, Any],
                              validation_type: ValidationType,
                              dataset_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Validate a single recommendation against defined criteria
        Returns validation result with confidence score and status
        """
        confidence_score = self._calculate_confidence(
            recommendation,
            validation_type,
            dataset_metadata
        )
        
        threshold = getattr(self.thresholds, validation_type.value)
        
        validation_result = {
            "recommendation_id": recommendation.get("id"),
            "validation_type": validation_type.value,
            "confidence_score": confidence_score,
            "threshold": threshold,
            "passed": confidence_score >= threshold,
            "metadata": {
                "model": recommendation.get("model"),
                "dataset_type": dataset_metadata.get("type") if dataset_metadata else None,
                "timestamp": recommendation.get("timestamp")
            }
        }
        
        self.validation_history.append(validation_result)
        return validation_result
    
    def _calculate_confidence(self,
                            recommendation: Dict[str, Any],
                            validation_type: ValidationType,
                            dataset_metadata: Optional[Dict]) -> float:
        """
        Calculate confidence score based on multiple factors:
        1. Model reliability score
        2. Task-specific validation
        3. Historical performance
        4. Dataset compatibility
        """
        scores = []
        
        # Model reliability score
        model_score = self._get_model_reliability_score(
            recommendation.get("model", "")
        )
        scores.append(model_score)
        
        # Task-specific validation
        task_score = self._validate_task_specific(
            recommendation,
            validation_type
        )
        scores.append(task_score)
        
        # Historical performance
        if dataset_metadata:
            history_score = self._check_historical_performance(
                validation_type,
                dataset_metadata.get("type")
            )
            scores.append(history_score)
        
        # Dataset compatibility
        if dataset_metadata:
            compatibility_score = self._check_dataset_compatibility(
                recommendation,
                dataset_metadata
            )
            scores.append(compatibility_score)
        
        # Weighted average of all scores
        weights = [0.4, 0.3, 0.15, 0.15][:len(scores)]
        return float(np.average(scores, weights=weights))
    
    def _get_model_reliability_score(self, model_name: str) -> float:
        """Calculate reliability score based on model type"""
        reliability_scores = {
            "claude-3-sonnet": 0.95,
            "claude-3-haiku": 0.90,
            "gpt-4": 0.95,
            "gemini-1.5-pro": 0.85,
            "gemini-1.5-flash": 0.80
        }
        return reliability_scores.get(model_name, 0.70)
    
    def _validate_task_specific(self,
                              recommendation: Dict[str, Any],
                              validation_type: ValidationType) -> float:
        """Validate recommendation based on task-specific criteria"""
        validation_funcs = {
            ValidationType.ANALYSIS: self._validate_analysis,
            ValidationType.FEATURE_ENGINEERING: self._validate_feature_engineering,
            ValidationType.OUTLIER_DETECTION: self._validate_outlier_detection,
            ValidationType.MISSING_VALUES: self._validate_missing_values,
            ValidationType.SCALING: self._validate_scaling,
            ValidationType.ENCODING: self._validate_encoding
        }
        
        validator = validation_funcs.get(validation_type, lambda x: 0.0)
        return validator(recommendation)
    
    def _check_historical_performance(self,
                                    validation_type: ValidationType,
                                    dataset_type: Optional[str]) -> float:
        """Check historical validation performance for similar cases"""
        if not dataset_type:
            return 0.85
            
        relevant_history = [
            h for h in self.validation_history
            if h["validation_type"] == validation_type.value
            and h["metadata"]["dataset_type"] == dataset_type
        ]
        
        if not relevant_history:
            return 0.85
            
        return np.mean([h["confidence_score"] for h in relevant_history])
    
    def _check_dataset_compatibility(self,
                                   recommendation: Dict[str, Any],
                                   dataset_metadata: Dict[str, Any]) -> float:
        """Verify compatibility between recommendation and dataset"""
        # Implementation specific to dataset type and recommendation
        # Default high compatibility score for now
        return 0.90
    
    # Task-specific validation methods
    def _validate_analysis(self, recommendation: Dict[str, Any]) -> float:
        """Validate analysis recommendations"""
        required_keys = ["statistical_summary", "data_quality", "recommendations"]
        return self._check_required_keys(recommendation, required_keys)
    
    def _validate_feature_engineering(self, recommendation: Dict[str, Any]) -> float:
        """Validate feature engineering recommendations"""
        required_keys = ["features", "transformations", "impact"]
        return self._check_required_keys(recommendation, required_keys)
    
    def _validate_outlier_detection(self, recommendation: Dict[str, Any]) -> float:
        """Validate outlier detection recommendations"""
        required_keys = ["method", "threshold", "identified_outliers"]
        return self._check_required_keys(recommendation, required_keys)
    
    def _validate_missing_values(self, recommendation: Dict[str, Any]) -> float:
        """Validate missing value handling recommendations"""
        required_keys = ["strategy", "affected_columns", "justification"]
        return self._check_required_keys(recommendation, required_keys)
    
    def _validate_scaling(self, recommendation: Dict[str, Any]) -> float:
        """Validate scaling recommendations"""
        required_keys = ["method", "features", "parameters"]
        return self._check_required_keys(recommendation, required_keys)
    
    def _validate_encoding(self, recommendation: Dict[str, Any]) -> float:
        """Validate encoding recommendations"""
        required_keys = ["method", "categorical_columns", "encoding_map"]
        return self._check_required_keys(recommendation, required_keys)
    
    def _check_required_keys(self,
                           recommendation: Dict[str, Any],
                           required_keys: List[str]) -> float:
        """Helper method to check for required keys in recommendations"""
        content = recommendation.get("content", {})
        if not isinstance(content, dict):
            return 0.0
        
        present_keys = sum(1 for key in required_keys if key in content)
        return present_keys / len(required_keys) 