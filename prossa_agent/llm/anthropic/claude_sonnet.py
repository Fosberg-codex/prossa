"""
Anthropic Claude 3.5 Sonnet implementation for the Prossa Agent framework.
"""

import os
from typing import Any, Optional, Dict, List
from anthropic import Anthropic
from datetime import datetime
from ..base import BaseLLM, LLMResponse
from ..prompts.system_prompts import get_system_prompt

class ClaudeSonnet(BaseLLM):
    """Claude 3.5 Sonnet implementation"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        self.client = Anthropic(api_key=self.api_key)

    async def generate(self, 
                      query: str, 
                      context: Optional[List[str]] = None,
                      max_tokens: int = 1000) -> LLMResponse:
        """Generate response using Claude Sonnet"""
        try:
            # Prepare messages
            system_prompt = get_system_prompt()
            user_prompt = self._prepare_prompt(query, context)
            
            response = await self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=max_tokens,
                temperature=0.7,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )

            return LLMResponse(
                content=response.content[0].text,
                confidence=0.92,  # Claude Sonnet baseline confidence
                metadata={
                    "model": "claude-3-sonnet",
                    "tokens_used": response.usage.output_tokens
                },
                timestamp=datetime.now()
            )

        except Exception as e:
            self.logger.error(f"Claude Sonnet generation error: {str(e)}")
            raise

    def _prepare_prompt(self, query: str, context: Optional[List[str]] = None) -> str:
        """Prepare prompt with context"""
        if not context:
            return query
            
        context_str = "\n".join(context)
        return f"""Context:
{context_str}

Query: {query}

Please analyze the context and provide a detailed response focusing on data preprocessing requirements."""