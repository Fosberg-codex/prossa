"""
Anthropic Claude 3.5 Haiku implementation for the Prossa Agent framework.
"""

import os
from typing import Any, Optional, Dict
from anthropic import Anthropic
from ..base import BaseLLM, LLMResponse
from ..prompts.system_prompts import get_system_prompt

class ClaudeHaiku(BaseLLM):
    """Claude 3.5 Haiku implementation"""
    
    MODEL_NAME = "claude-3-haiku-20240307"
    
    def __init__(self, api_key: Optional[str] = None, system_prompt: Optional[str] = None):
        super().__init__(api_key, system_prompt or get_system_prompt())
        self._initialize_model()

    def _initialize_model(self):
        try:
            api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key not found in constructor or environment")
            
            self.client = Anthropic(api_key=api_key)
            self.logger.info(f"Initialized {self.MODEL_NAME} successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Anthropic client: {str(e)}")
            raise

    def generate_content(self, prompt: str) -> LLMResponse:
        try:
            formatted_prompt = self._format_prompt(prompt)
            
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.messages.create(
                model=self.MODEL_NAME,
                messages=messages,
                max_tokens=1024  # Smaller context for Haiku
            )
            
            if not self.validate_response(response):
                raise ValueError("Invalid response from Claude Haiku")
            
            content = response.content[0].text
            metadata = self._extract_metadata(response, formatted_prompt)
            
            return self._create_response(
                content=content,
                confidence=self._calculate_confidence(response),
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Content generation error: {str(e)}")
            raise

    def validate_response(self, response: Any) -> bool:
        try:
            return (
                hasattr(response, 'content') and 
                len(response.content) > 0 and
                hasattr(response.content[0], 'text') and
                len(response.content[0].text.strip()) > 0
            )
        except Exception as e:
            self.logger.error(f"Response validation error: {str(e)}")
            return False

    def _calculate_confidence(self, response: Any) -> float:
        try:
            if hasattr(response, 'stop_reason'):
                if response.stop_reason == 'end_turn':
                    return 0.85  # Slightly lower than Sonnet
                elif response.stop_reason == 'max_tokens':
                    return 0.65
            return 0.75
            
        except Exception as e:
            self.logger.warning(f"Confidence calculation error: {str(e)}")
            return 0.4

    def _extract_metadata(self, response: Any, prompt: str) -> Dict[str, Any]:
        return {
            "model": self.MODEL_NAME,
            "input_tokens": getattr(response, 'usage', {}).get('input_tokens', 0),
            "output_tokens": getattr(response, 'usage', {}).get('output_tokens', 0),
            "stop_reason": getattr(response, 'stop_reason', None),
            "system": getattr(response, 'system', None),
        } 