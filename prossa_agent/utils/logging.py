"""
Centralized logging configuration for the Prossa Agent framework.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

class LogConfig:
    """Logging configuration for the Prossa Agent"""
    
    def __init__(self, 
                 log_level: str = "INFO",
                 log_file: Optional[str] = None,
                 log_format: Optional[str] = None):
        self.log_level = getattr(logging, log_level.upper())
        self.log_file = log_file or f"prossa_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.log_format = log_format or '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        self._configure_logging()

    def _configure_logging(self):
        """Configure logging with both file and console handlers"""
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create root logger
        logger = logging.getLogger("prossa_agent")
        logger.setLevel(self.log_level)
        
        # Remove existing handlers
        logger.handlers = []
        
        # File handler
        file_handler = logging.FileHandler(log_dir / self.log_file)
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(logging.Formatter(self.log_format))
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(logging.Formatter(self.log_format))
        logger.addHandler(console_handler)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the proper configuration"""
    return logging.getLogger(f"prossa_agent.{name}") 