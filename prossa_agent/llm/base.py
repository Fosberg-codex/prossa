"""
Base LLM interface for the Prossa Agent framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
import logging
from dataclasses import dataclass
from .prompts.system_prompts import get_system_prompt

@dataclass
class LLMResponse:
    """Standardized response format for LLM outputs"""
    content: str
    confidence: float
    metadata: Dict[str, Any]
    timestamp: datetime

class BaseLLM(ABC):
    """Abstract base class for LLM implementations"""
    
    def __init__(self, api_key: Optional[str] = None, system_prompt: Optional[str] = None):
        self.api_key = api_key
        # Use provided system prompt or fall back to default IQ-PACE prompt
        self.system_prompt = system_prompt if system_prompt is not None else get_system_prompt()
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for the LLM"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='agent.log'
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def generate_content(self, prompt: str) -> LLMResponse:
        """Generate content from the LLM"""
        pass

    @abstractmethod
    def validate_response(self, response: Any) -> bool:
        """Validate the raw response from the LLM"""
        pass

    def update_system_prompt(self, new_prompt: str):
        """Update the system prompt"""
        self.logger.info("Updating system prompt")
        self.system_prompt = new_prompt

    def _format_prompt(self, user_prompt: str) -> str:
        """Format the complete prompt with system and user content"""
        if self.system_prompt:
            return f"{self.system_prompt}\n\n{user_prompt}"
        return user_prompt

    def _create_response(self, 
                        content: str, 
                        confidence: float = 1.0, 
                        metadata: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """Create a standardized LLMResponse object"""
        return LLMResponse(
            content=content,
            confidence=confidence,
            metadata=metadata or {},
            timestamp=datetime.now()
        ) 