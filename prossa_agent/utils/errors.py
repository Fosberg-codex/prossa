"""
Custom exceptions and error handling for the Prossa Agent framework.
"""

from typing import Optional, Dict, Any

class ProssaAgentError(Exception):
    """Base exception class for Prossa Agent"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class DatasetError(ProssaAgentError):
    """Raised when there are issues with the input dataset"""
    pass

class PreprocessingError(ProssaAgentError):
    """Raised when preprocessing operations fail"""
    pass

class ModelError(ProssaAgentError):
    """Raised when LLM operations fail"""
    pass

class ValidationError(ProssaAgentError):
    """Raised when validation checks fail"""
    pass

class MemoryError(ProssaAgentError):
    """Raised when memory operations fail"""
    pass

def handle_preprocessing_error(error: Exception, context: Dict[str, Any]) -> PreprocessingError:
    """Convert preprocessing exceptions to PreprocessingError"""
    if isinstance(error, PreprocessingError):
        return error
        
    return PreprocessingError(
        message=f"Preprocessing failed: {str(error)}",
        details={
            'original_error': str(error),
            'error_type': error.__class__.__name__,
            'context': context
        }
    ) 