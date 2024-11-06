"""
OpenAI GPT-o1-preview implementation for the Prossa Agent framework.
"""

import os
from typing import Any, Optional, Dict, List
import openai
from ..base import BaseLLM, LLMResponse
from ..prompts.system_prompts import get_system_prompt
from datetime import datetime

class GPTPreview(BaseLLM):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        openai.api_key = self.api_key

    async def generate(self, 
                      query: str, 
                      context: Optional[List[str]] = None,
                      max_tokens: int = 1000) -> LLMResponse:
        """Generate response using GPT-4-preview"""
        try:
            messages = [
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": self._prepare_prompt(query, context)}
            ]

            response = await openai.ChatCompletion.acreate(
                model="gpt-4-preview",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                confidence=0.95,  # GPT-4 typically has high confidence
                metadata={
                    "model": "gpt-4-preview",
                    "tokens_used": response.usage.total_tokens
                },
                timestamp=datetime.now()
            )

        except Exception as e:
            self.logger.error(f"GPT-4-preview generation error: {str(e)}")
            raise

    def _prepare_prompt(self, query: str, context: Optional[List[str]] = None) -> str:
        """Prepare prompt with context if available"""
        if not context:
            return query
            
        context_str = "\n".join(context)
        return f"Context:\n{context_str}\n\nQuery: {query}"