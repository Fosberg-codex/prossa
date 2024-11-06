"""
Intelligent model selection based on task requirements.
"""

from typing import Dict, Type, Optional, Any
from dataclasses import dataclass
import logging
from .base import BaseLLM
from .openai.gpt_o1_preview import GPTPreview
from .anthropic.claude_sonnet import ClaudeSonnet
from .gemini.gemini_pro import GeminiPro
from ..utils.errors import ModelError

@dataclass
class ModelFeatures:
    """Features for model selection"""
    name: str
    max_context: int
    average_latency: float
    cost_per_1k_tokens: float
    capabilities: Dict[str, float]  # Capability scores (0-1)

class ModelSelector:
    """Selects the most appropriate LLM model for a given task"""
    
    def __init__(self):
        self._setup_logging()
        self.models: Dict[str, Type[BaseLLM]] = {}
        self.features: Dict[str, ModelFeatures] = {}
        self._initialize_models()

    def _setup_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def _initialize_models(self):
        """Initialize available models"""
        # Register OpenAI models
        self.register_model(
            "gpt-4-preview",
            GPTPreview,
            ModelFeatures(
                name="gpt-4-preview",
                max_context=128000,
                average_latency=2.0,
                cost_per_1k_tokens=0.01,
                capabilities={
                    "analysis": 0.95,
                    "planning": 0.95,
                    "reasoning": 0.95,
                    "accuracy": 0.95
                }
            )
        )
        
        # Register Anthropic models
        self.register_model(
            "claude-3-sonnet",
            ClaudeSonnet,
            ModelFeatures(
                name="claude-3-sonnet",
                max_context=200000,
                average_latency=1.5,
                cost_per_1k_tokens=0.008,
                capabilities={
                    "analysis": 0.92,
                    "planning": 0.90,
                    "reasoning": 0.92,
                    "accuracy": 0.92
                }
            )
        )
        
        # Register Gemini models
        self.register_model(
            "gemini-pro",
            GeminiPro,
            ModelFeatures(
                name="gemini-pro",
                max_context=30720,
                average_latency=1.0,
                cost_per_1k_tokens=0.005,
                capabilities={
                    "analysis": 0.88,
                    "planning": 0.85,
                    "reasoning": 0.88,
                    "accuracy": 0.88
                }
            )
        )

    def register_model(self, 
                      name: str, 
                      model_class: Optional[Type[BaseLLM]] = None,
                      features: Optional[ModelFeatures] = None):
        """Register a model with its features"""
        if model_class:
            self.models[name] = model_class
        if features:
            self.features[name] = features

    def select_model(self, 
                    task_complexity: float,
                    required_confidence: float,
                    max_latency: Optional[float] = None,
                    max_cost: Optional[float] = None) -> BaseLLM:
        """Select and instantiate the most appropriate model"""
        try:
            # Get best model name
            model_name = self._select_best_model(
                task_complexity,
                required_confidence,
                max_latency,
                max_cost
            )
            
            # Instantiate model
            model_class = self.models.get(model_name)
            if not model_class:
                raise ModelError(f"Model {model_name} not found")
                
            return model_class()
            
        except Exception as e:
            raise ModelError(f"Model selection failed: {str(e)}")

    def _select_best_model(self,
                          task_complexity: float,
                          required_confidence: float,
                          max_latency: Optional[float] = None,
                          max_cost: Optional[float] = None) -> str:
        """Select the best model name based on requirements"""
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
            raise ModelError("No suitable model found for the given requirements")

        # Select the highest scoring model
        return max(candidates, key=lambda x: x[0])[1]

    def _calculate_model_score(self,
                             features: ModelFeatures,
                             task_complexity: float,
                             required_confidence: float) -> float:
        """Calculate score for model selection"""
        # Base score from capabilities
        capability_score = (
            features.capabilities['analysis'] * 0.3 +
            features.capabilities['planning'] * 0.3 +
            features.capabilities['reasoning'] * 0.2 +
            features.capabilities['accuracy'] * 0.2
        )
        
        # Adjust for task complexity
        complexity_match = 1.0 - abs(task_complexity - capability_score)
        
        # Adjust for required confidence
        confidence_match = 1.0 if capability_score >= required_confidence else 0.5
        
        return (capability_score * 0.4 + 
                complexity_match * 0.3 + 
                confidence_match * 0.3)