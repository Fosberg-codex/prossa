"""
Main agent class implementing the RAG workflow for dataset processing.
"""

from typing import Optional, Dict, Any, Union, List
import logging
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from .execution.actions import ActionManager, ActionResult
from .execution.validator import ValidationEngine, ValidationResult
from ..memory.buffer import MemoryBuffer
from ..llm.selector import ModelSelector
from ..utils.config import SystemConfig
from ..utils.errors import (
    ProssaAgentError,
    DatasetError,
    ValidationError,
    handle_preprocessing_error
)

@dataclass
class AgentResult:
    """Comprehensive result from the agent's processing"""
    success: bool
    data: Union[pd.DataFrame, str]
    confidence: float
    preprocessing_steps: Dict[str, Any]
    validation: Optional[ValidationResult] = None
    error: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = datetime.now()

class ProsaAgent:
    """
    Main agent class implementing RAG workflow:
    Dataset -> Analyze -> Plan -> Process -> Validate -> Return
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self._setup_logging()
        self._initialize_components()
        
    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def _initialize_components(self):
        """Initialize core components"""
        try:
            # Initialize components
            self.action_manager = ActionManager()
            self.validator = ValidationEngine(use_gpu=self.config.use_gpu)
            self.memory = MemoryBuffer(
                max_items=self.config.max_memory_items,
                use_gpu=self.config.use_gpu
            )
            
            self.logger.info("ProsaAgent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ProsaAgent: {str(e)}")
            raise ProssaAgentError("Agent initialization failed", {"error": str(e)})

    async def process_dataset(self, 
                            dataset: pd.DataFrame,
                            task_description: Optional[str] = None) -> AgentResult:
        """
        Process a dataset using RAG workflow
        
        Args:
            dataset: Input DataFrame
            task_description: Optional description of processing requirements
            
        Returns:
            AgentResult containing processed dataset and comprehensive metadata
        """
        try:
            self.logger.info(f"Processing dataset of shape {dataset.shape}")
            
            # 1. Get relevant context from memory
            context = await self.memory.get_relevant_context(
                query=task_description or "Process dataset with standard preprocessing"
            )
            
            # 2. Process dataset through action manager
            action_result = await self.action_manager.process_dataset(
                df=dataset.copy(),
                context=context
            )
            
            if not action_result.success:
                return AgentResult(
                    success=False,
                    data=action_result.data,
                    confidence=0.0,
                    preprocessing_steps={},
                    error={
                        'source': action_result.source,
                        'message': str(action_result.data),
                        'metadata': action_result.metadata
                    }
                )
            
            # 3. Validate results
            try:
                validation = await self.validator.validate_preprocessing(
                    original_df=dataset,
                    processed_df=action_result.data,
                    preprocessing_plan=action_result.metadata.get('plan', {})
                )
            except ValidationError as ve:
                return AgentResult(
                    success=False,
                    data=str(ve),
                    confidence=0.0,
                    preprocessing_steps=action_result.metadata.get('plan', {}),
                    error={
                        'source': 'validation',
                        'message': str(ve),
                        'details': ve.details
                    }
                )
            
            # 4. Store successful processing
            if validation.passed:
                await self.memory.add_interaction(
                    query=task_description or "dataset_processing",
                    response=str(action_result.metadata),
                    metadata={
                        'dataset_shape': dataset.shape,
                        'confidence': validation.confidence_score,
                        'timestamp': datetime.now()
                    }
                )
            
            return AgentResult(
                success=True,
                data=action_result.data,
                confidence=validation.confidence_score,
                preprocessing_steps=action_result.metadata.get('plan', {}),
                validation=validation,
                metadata={
                    'original_shape': dataset.shape,
                    'processed_shape': action_result.data.shape,
                    'context_used': context,
                    'task_description': task_description
                }
            )
            
        except Exception as e:
            error = handle_preprocessing_error(e, {'dataset_shape': dataset.shape})
            self.logger.error(f"Dataset processing error: {str(error)}")
            
            return AgentResult(
                success=False,
                data=str(error),
                confidence=0.0,
                preprocessing_steps={},
                error={
                    'message': str(error),
                    'details': error.details,
                    'timestamp': datetime.now()
                }
            )

    async def get_processing_history(self) -> List[Dict[str, Any]]:
        """Get history of processing operations"""
        try:
            return await self.memory.get_all_interactions()
        except Exception as e:
            self.logger.error(f"Failed to retrieve processing history: {str(e)}")
            return []