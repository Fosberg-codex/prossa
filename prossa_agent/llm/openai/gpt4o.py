"""
OpenAI GPT-4 implementation for the Prossa Agent framework.
"""

import os
from typing import Any, Optional, Dict
import openai
from ..base import BaseLLM, LLMResponse
from ..prompts.system_prompts import get_system_prompt

class GPT4O(BaseLLM):
    """GPT-4 implementation"""
    
    MODEL_NAME = "gpt-4"
    
    def __init__(self, api_key: Optional[str] = None, system_prompt: Optional[str] = None):
        super().__init__(api_key, system_prompt or get_system_prompt())
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the OpenAI client with API key configuration"""
        try:
            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in constructor or environment")
            
            self.client = openai.OpenAI(api_key=api_key)
            self.logger.info(f"Initialized {self.MODEL_NAME} successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise

    def generate_content(self, prompt: str) -> LLMResponse:
        """
        Generate content using GPT-4.
        
        Args:
            prompt: The input prompt for generation
            
        Returns:
            LLMResponse containing the generated content and metadata
        """
        try:
            formatted_prompt = self._format_prompt(prompt)
            
            messages = [
                {"role": "system", "content": self.system_prompt} if self.system_prompt else None,
                {"role": "user", "content": prompt}
            ]
            messages = [m for m in messages if m is not None]
            
            response = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=messages,
                temperature=0.7
            )
            
            if not self.validate_response(response):
                raise ValueError("Invalid response from OpenAI")
            
            content = response.choices[0].message.content
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
        """Validate the OpenAI response"""
        try:
            return (
                hasattr(response, 'choices') and 
                len(response.choices) > 0 and
                hasattr(response.choices[0], 'message') and
                hasattr(response.choices[0].message, 'content') and
                len(response.choices[0].message.content.strip()) > 0
            )
        except Exception as e:
            self.logger.error(f"Response validation error: {str(e)}")
            return False

    def _calculate_confidence(self, response: Any) -> float:
        """Calculate confidence score for the response"""
        try:
            # Use finish_reason and other metadata to estimate confidence
            if response.choices[0].finish_reason == 'stop':
                return 0.9
            elif response.choices[0].finish_reason == 'length':
                return 0.7
            return 0.8
            
        except Exception as e:
            self.logger.warning(f"Confidence calculation error: {str(e)}")
            return 0.5

    def _extract_metadata(self, response: Any, prompt: str) -> Dict[str, Any]:
        """Extract metadata from the response"""
        return {
            "model": self.MODEL_NAME,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "finish_reason": response.choices[0].finish_reason,
        } 