"""
Gemini gemini-1.5-flash-8b-001 implementation for the Prossa Agent framework.
"""

import os
from typing import Any, Optional, Dict
import google.generativeai as genai
from ..base import BaseLLM, LLMResponse
from ..prompts.system_prompts import get_system_prompt

class GeminiBase(BaseLLM):
    """gemini-1.5-flash-8b-001 implementation"""
    
    MODEL_NAME = "gemini-1.5-flash-8b-001"
    
    def __init__(self, api_key: Optional[str] = None, system_prompt: Optional[str] = None):
        super().__init__(api_key, system_prompt or get_system_prompt())
        self._initialize_model()

    def _initialize_model(self):
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
        try:
            # Gemini uses generation_config for system prompts
            generation_config = {
                "temperature": 0.7,
                "top_p": 1.0,
                "top_k": 40,
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
                generation_config=generation_config
            )
            
            if not self.validate_response(response):
                raise ValueError("Invalid response from Gemini Base")
            
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
        try:
            if hasattr(response, 'candidates') and response.candidates:
                return min(0.8, response.candidates[0].score)  # Lower max confidence for base model
            return 0.6  # Lower default confidence for base model
            
        except Exception as e:
            self.logger.warning(f"Confidence calculation error: {str(e)}")
            return 0.4

    def _extract_metadata(self, response: Any, prompt: str) -> Dict[str, Any]:
        return {
            "model": self.MODEL_NAME,
            "prompt_tokens": len(prompt.split()),
            "response_tokens": len(response.text.split()),
            "timestamp_utc": response.candidates[0].finish_time if hasattr(response, 'candidates') else None,
            "safety_ratings": getattr(response, 'safety_ratings', None),
        } 