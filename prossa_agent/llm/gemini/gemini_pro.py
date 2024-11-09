"""
Google Gemini Pro implementation for the Prossa Agent framework.
"""

import os
from typing import Any, Optional, Dict, List
import google.generativeai as genai
from datetime import datetime
from ..base import BaseLLM, LLMResponse
from ..prompts.system_prompts import get_system_prompt

class GeminiPro(BaseLLM):
    """Gemini Pro implementation"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    async def generate_content(self, 
                             prompt: str,
                             **kwargs) -> str:
        """Generate content using Gemini Pro"""
        response = await self.model.generate_content_async(prompt)
        return response.text

    async def validate_response(self, 
                              response: str,
                              context: Optional[Dict[str, Any]] = None) -> bool:
        """Validate the generated response"""
        # Basic validation - ensure response is not empty
        if not response or not response.strip():
            return False
            
        # Check if response seems coherent
        min_length = 10  # Minimum reasonable response length
        max_length = 10000  # Maximum reasonable response length
        response_length = len(response.split())
        
        return min_length <= response_length <= max_length

    async def generate(self, 
                      query: str, 
                      context: Optional[List[str]] = None,
                      max_tokens: int = 1000) -> LLMResponse:
        """Generate response using Gemini Pro"""
        try:
            # Prepare prompt
            system_prompt = get_system_prompt()
            user_prompt = self._prepare_prompt(query, context)
            
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Generate content
            content = await self.generate_content(
                full_prompt,
                max_tokens=max_tokens
            )
            
            # Validate response
            is_valid = await self.validate_response(content)
            if not is_valid:
                raise ValueError("Generated response failed validation")

            return LLMResponse(
                content=content,
                confidence=0.90,  # Gemini Pro baseline confidence
                metadata={
                    "model": "gemini-pro",
                    "prompt_tokens": len(full_prompt.split())
                },
                timestamp=datetime.now()
            )

        except Exception as e:
            self.logger.error(f"Gemini Pro generation error: {str(e)}")
            raise

    def _prepare_prompt(self, query: str, context: Optional[List[str]] = None) -> str:
        """Prepare prompt with context"""
        if not context:
            return query
            
        context_str = "\n".join(context)
        return f"""Context:
{context_str}

Query: {query}

Analyze the context and provide a detailed response focusing on data preprocessing requirements."""