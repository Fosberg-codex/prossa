"""
Simplify ModelSelector to focus on RAG requirements:
"""

from typing import Dict, List, Optional, Type
from .base import BaseLLM
from dataclasses import dataclass

@dataclass
class ModelFeatures:
    """Features and capabilities of an LLM model"""
    name: str
    context_length: int
    supports_gpu: bool
    cost_per_1k_tokens: float
    average_latency: float  # in seconds
    reliability: float  # 0-1 score

class ModelSelector:
    """Selects the most appropriate LLM model for a given task"""
    
    def __init__(self):
        self.models: Dict[str, Type[BaseLLM]] = {}
        self.features: Dict[str, ModelFeatures] = {}

    def register_model(self, 
                      name: str, 
                      model_class: Type[BaseLLM], 
                      features: ModelFeatures):
        """Register a model with its features"""
        self.models[name] = model_class
        self.features[name] = features

    def select_model(self, 
                    task_complexity: float,
                    required_confidence: float,
                    max_latency: Optional[float] = None,
                    max_cost: Optional[float] = None) -> Type[BaseLLM]:
        """
        Select the most appropriate model based on task requirements
        """
        candidates = []
        
        for name, features in self.features.items():
            # Check hard constraints
            if max_latency and features.average_latency > max_latency:
                continue
            if max_cost and features.cost_per_1k_tokens > max_cost:
                continue
                
            # Score the model
            score = self._calculate_model_score(
                features, 
                task_complexity,
                required_confidence
            )
            
            candidates.append((score, name))

        if not candidates:
            raise ValueError("No suitable model found for the given requirements")

        # Select the highest scoring model
        best_model_name = max(candidates, key=lambda x: x[0])[1]
        return self.models[best_model_name]

    def _calculate_model_score(self, 
                             features: ModelFeatures,
                             task_complexity: float,
                             required_confidence: float) -> float:
        """Calculate a score for model suitability"""
        # Base score is the model's reliability
        score = features.reliability
        
        # Adjust based on task complexity
        if task_complexity > 0.7:
            score *= (features.context_length / 8192)  # Normalize by typical context length
        
        # Adjust based on required confidence
        if required_confidence > 0.8:
            score *= features.reliability
            
        return score 