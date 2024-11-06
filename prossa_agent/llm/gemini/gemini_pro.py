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
            
            response = await self.model.generate_content_async(
                full_prompt,
                generation_config={
                    'max_output_tokens': max_tokens,
                    'temperature': 0.7
                }
            )

            return LLMResponse(
                content=response.text,
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