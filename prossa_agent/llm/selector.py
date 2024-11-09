"""
Intelligent model selection based on task requirements.
"""

from typing import Dict, Type, Optional, Any
from dataclasses import dataclass
import logging
import os
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
    capabilities: Dict[str, float]

class ModelSelector:
    """Selects the most appropriate LLM model based on available API keys"""
    
    def __init__(self):
        self._setup_logging()
        self.models: Dict[str, Type[BaseLLM]] = {}
        self.features: Dict[str, ModelFeatures] = {}
        self._initialize_available_models()

    def _setup_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def _initialize_available_models(self):
        """Initialize only models with available API keys"""
        # Check Gemini
        if os.getenv("GOOGLE_GEMINI_API_KEY"):
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
            
        # Check OpenAI
        if os.getenv("OPENAI_API_KEY"):
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
            
        # Check Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
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

        if not self.models:
            raise ModelError("No LLM API keys found in environment")

    def register_model(self, 
                      name: str, 
                      model_class: Type[BaseLLM],
                      features: ModelFeatures):
        """Register an available model"""
        self.models[name] = model_class
        self.features[name] = features
        self.logger.info(f"Registered model: {name}")

    def select_model(self, 
                    task_complexity: float,
                    required_confidence: float,
                    max_latency: Optional[float] = None,
                    max_cost: Optional[float] = None) -> BaseLLM:
        """Select best available model based on requirements"""
        try:
            # Get best model name from available models
            model_name = self._select_best_model(
                task_complexity,
                required_confidence,
                max_latency,
                max_cost
            )
            
            # Instantiate selected model
            model_class = self.models.get(model_name)
            if not model_class:
                raise ModelError(f"Selected model {model_name} not found")
                
            return model_class()
            
        except Exception as e:
            raise ModelError(f"Model selection failed: {str(e)}")
    def _select_best_model(self,
                          task_complexity: float,
                          required_confidence: float,
                          max_latency: Optional[float] = None,
                          max_cost: Optional[float] = None) -> str:
        """Select best model from available ones"""
        candidates = []
        
        for name, features in self.features.items():
            # Apply constraints
            if max_latency and features.average_latency > max_latency:
                continue
            if max_cost and features.cost_per_1k_tokens > max_cost:
                continue
                
            score = self._calculate_model_score(
                features, 
                task_complexity,
                required_confidence
            )
            candidates.append((score, name))

        if not candidates:
            # If no model meets constraints, return any available model
            return next(iter(self.models.keys()))

        return max(candidates, key=lambda x: x[0])[1]

    def _calculate_model_score(self,
                             features: ModelFeatures,
                             task_complexity: float,
                             required_confidence: float) -> float:
        """Calculate model score based on requirements"""
        capability_score = sum(features.capabilities.values()) / len(features.capabilities)
        complexity_match = 1.0 - abs(task_complexity - capability_score)
        confidence_match = 1.0 if capability_score >= required_confidence else 0.5
        
        return (capability_score * 0.4 + 
                complexity_match * 0.3 + 
                confidence_match * 0.3)

