"""
Configuration management for the Prossa Agent.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
from .logging import LogConfig

@dataclass
class APIConfig:
    """API configuration settings"""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

@dataclass
class SystemConfig:
    """System configuration settings"""
    use_gpu: bool = True
    max_memory_items: int = 1000
    embedding_model: str = 'all-MiniLM-L6-v2'
    min_confidence: float = 0.7
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    def __post_init__(self):
        """Initialize logging configuration"""
        self.log_config = LogConfig(
            log_level=self.log_level,
            log_file=self.log_file
        )

def load_config() -> tuple[APIConfig, SystemConfig]:
    """Load configuration from environment"""
    load_dotenv()
    
    api_config = APIConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        gemini_api_key=os.getenv("GOOGLE_GEMINI_API_KEY")
    )
    
    system_config = SystemConfig(
        use_gpu=os.getenv("USE_GPU", "true").lower() == "true",
        max_memory_items=int(os.getenv("MAX_MEMORY_ITEMS", "1000")),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.7")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE")
    )
    
    return api_config, system_config 