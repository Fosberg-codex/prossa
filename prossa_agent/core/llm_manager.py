from typing import Dict, Optional, List, Union, Any
import os
from enum import Enum
from dotenv import load_dotenv
import anthropic
import openai
import google.generativeai as genai
import yaml

load_dotenv()

class ModelType(Enum):
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    FEATURE_ENGINEERING = "feature_engineering"
    INITIAL_SCREENING = "initial_screening"

class LLMManager:
    def __init__(self):
        """Initialize LLM clients and load configurations"""
        self._init_clients()
        self._load_config()
        self.active_models = self._verify_available_models()
    
    def _init_clients(self):
        """Initialize API clients for different LLM providers"""
        # Anthropic initialization
        self.anthropic_client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        
        # OpenAI initialization
        self.openai_client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Gemini initialization (simplified)
        genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))
    
    def _load_config(self):
        """Load model configurations from YAML"""
        config_path = os.path.join(
            os.path.dirname(__file__), 
            "../config/llm_config.yaml"
        )
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def _verify_available_models(self) -> Dict[str, bool]:
        """Verify which models are available for use"""
        available_models = {}
        
        # Check Anthropic models
        try:
            self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            available_models["claude-3-haiku"] = True
        except:
            available_models["claude-3-haiku"] = False
            
        # Similar checks for other models...
        return available_models
    
    def select_model(self, 
                    task_type: ModelType, 
                    dataset_complexity: float) -> str:
        """Select the most appropriate model for a given task"""
        model_config = self.config["model_assignments"][task_type.value]
        
        # Select primary model based on complexity and availability
        for model in model_config["models"]:
            if (dataset_complexity >= model["min_complexity"] and 
                dataset_complexity <= model["max_complexity"] and
                self.active_models.get(model["name"], False)):
                return model["name"]
        
        # Return fallback model if primary not available
        return model_config["fallback"]
    
    async def process_task(self,
                          task_type: ModelType,
                          prompt: str,
                          dataset_complexity: float = 0.5,
                          **kwargs) -> Dict[str, Any]:
        """Process a task with the selected model"""
        model_name = self.select_model(task_type, dataset_complexity)
        
        if "claude" in model_name:
            return await self._process_anthropic(model_name, prompt, **kwargs)
        elif "gpt" in model_name:
            return await self._process_openai(model_name, prompt, **kwargs)
        else:
            return await self._process_gemini(model_name, prompt, **kwargs)
    
    async def _process_anthropic(self,
                               model: str,
                               prompt: str,
                               **kwargs) -> Dict[str, Any]:
        """Process task with Anthropic's Claude"""
        response = await self.anthropic_client.messages.create(
            model=model,
            max_tokens=kwargs.get("max_tokens", 1000),
            messages=[{"role": "user", "content": prompt}]
        )
        return {
            "content": response.content[0].text,
            "model": model,
            "usage": response.usage
        }
    
    async def _process_openai(self,
                             model: str,
                             prompt: str,
                             **kwargs) -> Dict[str, Any]:
        """Process task with OpenAI models"""
        response = await self.openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", 1000)
        )
        return {
            "content": response.choices[0].message.content,
            "model": model,
            "usage": response.usage
        }
    
    async def _process_gemini(self,
                             model: str,
                             prompt: str,
                             **kwargs) -> Dict[str, Any]:
        """Process task with Google's Gemini"""
        model = genai.GenerativeModel(model)
        response = model.generate_content(prompt)
        return {
            "content": response.text,
            "model": model,
            "usage": None  # Gemini doesn't provide usage stats
        } 