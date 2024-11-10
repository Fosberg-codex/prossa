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
    OUTLIER_DETECTION = "outlier_detection"
    MISSING_VALUES = "missing_values"
    SCALING = "scaling"
    ENCODING = "encoding"

class LLMManager:
    def __init__(self):
        """Initialize LLM clients and load configurations"""
        self._init_clients()
        self._load_config()
        self.active_models = self._verify_available_models()
        self.fallback_chain = self._setup_fallback_chain()
    
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
    
    def _setup_fallback_chain(self) -> Dict[str, List[str]]:
        """Setup fallback chain for each model"""
        return {
            "claude-3-sonnet": ["gpt-4", "o1-preview", "gemini-1.5-pro", "claude-3-haiku"],
            "claude-3-haiku": ["claude-3-sonnet", "o1-preview", "gemini-1.5-pro", "gpt-4"],
            "gpt-4": ["o1-preview", "claude-3-sonnet", "gemini-1.5-pro", "claude-3-haiku"],
            "o1-preview": ["gpt-4", "claude-3-haiku", "gemini-1.5-pro", "gemini-1.5-flash"],
            "gemini-1.5-pro": ["o1-preview", "gpt-4", "claude-3-haiku", "gemini-1.5-flash"],
            "gemini-1.5-flash": ["gemini-1.5-pro", "o1-preview", "claude-3-haiku", "gpt-4"]
        }
    
    async def process_task(self,
                          task_type: ModelType,
                          prompt: str,
                          dataset_complexity: float = 0.5,
                          **kwargs) -> Dict[str, Any]:
        """Process a task with the selected model, with fallback handling"""
        tried_models = set()
        errors = {}
        
        # Get initial model
        current_model = self.select_model(task_type, dataset_complexity)
        
        while current_model not in tried_models:
            tried_models.add(current_model)
            
            try:
                if "claude" in current_model:
                    return await self._process_anthropic(current_model, prompt, **kwargs)
                elif "gpt" in current_model:
                    return await self._process_openai(current_model, prompt, **kwargs)
                else:
                    return await self._process_gemini(current_model, prompt, **kwargs)
                    
            except Exception as e:
                errors[current_model] = str(e)
                # Get next model from fallback chain
                current_model = self._get_next_available_model(
                    current_model,
                    tried_models
                )
        
        # If we've exhausted all options, raise comprehensive error
        raise RuntimeError(
            f"All available models failed. Errors: {errors}"
        )
    
    def _get_next_available_model(self,
                                current_model: str,
                                tried_models: set) -> Optional[str]:
        """Get next available model from fallback chain"""
        fallback_options = self.fallback_chain.get(current_model, [])
        
        for model in fallback_options:
            if (model not in tried_models and 
                self.active_models.get(model, False)):
                return model
                
        # If no fallbacks left, try any available model not yet tried
        all_models = set(self.active_models.keys())
        available_models = all_models - tried_models
        
        return next(iter(available_models)) if available_models else None
    
    def _verify_available_models(self) -> Dict[str, bool]:
        """Verify which models are available for use"""
        available_models = {}
        
        # Test Anthropic models
        try:
            self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            available_models["claude-3-haiku"] = True
            available_models["claude-3-sonnet"] = True
        except:
            available_models["claude-3-haiku"] = False
            available_models["claude-3-sonnet"] = False
        
        # Test OpenAI models
        try:
            # Test GPT-4
            self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10
            )
            available_models["gpt-4"] = True
            
            # Test o1-preview
            self.openai_client.chat.completions.create(
                model="o1-preview",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10
            )
            available_models["o1-preview"] = True
        except:
            available_models["gpt-4"] = False
            available_models["o1-preview"] = False
        
        # Test Gemini models
        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            model.generate_content("test")
            available_models["gemini-1.5-pro"] = True
            available_models["gemini-1.5-flash"] = True
        except:
            available_models["gemini-1.5-pro"] = False
            available_models["gemini-1.5-flash"] = False
        
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
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(prompt)
        
        # Structure the response properly
        return {
            "content": {
                "text": response.text,
                "statistical_summary": self._extract_section(response.text, "statistical_summary"),
                "data_quality": self._extract_section(response.text, "data_quality"),
                "recommendations": self._extract_section(response.text, "recommendations"),
                "features": self._extract_section(response.text, "features"),
                "transformations": self._extract_section(response.text, "transformations"),
                "impact": self._extract_section(response.text, "impact"),
                "method": self._extract_section(response.text, "method"),
                "threshold": self._extract_section(response.text, "threshold"),
                "identified_outliers": self._extract_section(response.text, "identified_outliers"),
                "strategy": self._extract_section(response.text, "strategy"),
                "affected_columns": self._extract_section(response.text, "affected_columns"),
                "justification": self._extract_section(response.text, "justification"),
                "parameters": self._extract_section(response.text, "parameters"),
                "categorical_columns": self._extract_section(response.text, "categorical_columns"),
                "encoding_map": self._extract_section(response.text, "encoding_map")
            },
            "model": model,
            "usage": None
        }
    
    def _extract_section(self, text: str, section: str) -> str:
        """Extract specific sections from the response text"""
        try:
            # Simple extraction - can be made more sophisticated
            if section.lower() in text.lower():
                return text
            return ""
        except:
            return ""