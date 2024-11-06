"""
Gemini Pro implementation for the Prossa Agent framework.
"""

import os
from typing import Any, Optional, Dict
import google.generativeai as genai
from ..base import BaseLLM, LLMResponse
from ..prompts.system_prompts import get_system_prompt

class GeminiPro(BaseLLM):
    """Gemini Pro 1.5 implementation"""
    
    MODEL_NAME = "gemini-1.5-pro-latest"
    
    def __init__(self, api_key: Optional[str] = None, system_prompt: Optional[str] = None):
        super().__init__(api_key, system_prompt or get_system_prompt())
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the Gemini model with API key configuration"""
        try:
            api_key = self.api_key or os.getenv("GOOGLE_GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Gemini API key not found in constructor or environment")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.MODEL_NAME)
            self.logger.info(f"Initialized {self.MODEL_NAME} successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini model: {str(e)}")
            raise

    def generate_content(self, prompt: str) -> LLMResponse:
        """
        Generate content using Gemini Pro.
        
        Args:
            prompt: The input prompt for generation
            
        Returns:
            LLMResponse containing the generated content and metadata
            
        Raises:
            ValueError: If response validation fails
            Exception: For any other generation errors
        """
        try:
            # Gemini uses generation_config for system prompts
            generation_config = {
                "temperature": 0.7,
                "top_p": 1.0,
                "top_k": 40,
            }

            # Create safety settings
            safety_settings = {
                "harassment": "block_none",
                "hate_speech": "block_none",
                "sexually_explicit": "block_none",
                "dangerous_content": "block_none",
            }

            # Initialize chat
            chat = self.model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [self.system_prompt]
                    },
                    {
                        "role": "model",
                        "parts": ["Understood. I will act according to the IQ-PACE framework."]
                    }
                ]
            )

            # Generate response
            response = chat.send_message(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            if not self.validate_response(response):
                raise ValueError("Invalid response from Gemini")

            metadata = self._extract_metadata(response, prompt)

            return self._create_response(
                content=response.text,
                confidence=self._calculate_confidence(response),
                metadata=metadata
            )

        except Exception as e:
            self.logger.error(f"Content generation error: {str(e)}")
            raise

    def validate_response(self, response: Any) -> bool:
        """
        Validate the Gemini response.
        
        Args:
            response: Raw response from Gemini
            
        Returns:
            bool: True if response is valid, False otherwise
        """
        try:
            return (
                hasattr(response, 'text') and 
                isinstance(response.text, str) and 
                len(response.text.strip()) > 0
            )
        except Exception as e:
            self.logger.error(f"Response validation error: {str(e)}")
            return False

    def _calculate_confidence(self, response: Any) -> float:
        """
        Calculate confidence score for the response.
        
        Args:
            response: Raw response from Gemini
            
        Returns:
            float: Confidence score between 0 and 1
        """
        try:
            # Basic confidence calculation
            # Could be enhanced based on response properties
            if hasattr(response, 'candidates') and response.candidates:
                return min(1.0, response.candidates[0].score)
            return 0.8  # Default confidence for valid responses
            
        except Exception as e:
            self.logger.warning(f"Confidence calculation error: {str(e)}")
            return 0.5  # Default fallback confidence

    def _extract_metadata(self, response: Any, prompt: str) -> Dict[str, Any]:
        """
        Extract metadata from the response.
        
        Args:
            response: Raw response from Gemini
            prompt: Original formatted prompt
            
        Returns:
            Dict containing response metadata
        """
        return {
            "model": self.MODEL_NAME,
            "prompt_tokens": len(prompt.split()),
            "response_tokens": len(response.text.split()),
            "timestamp_utc": response.candidates[0].finish_time if hasattr(response, 'candidates') else None,
            "safety_ratings": getattr(response, 'safety_ratings', None),
        } 