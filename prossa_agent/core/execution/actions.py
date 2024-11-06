"""
RAG-focused action system for the Prossa Agent framework.
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import pandas as pd
from ...integrations.prossa.processor import ProsaProcessor, ProcessingResult
from ...memory.buffer import MemoryBuffer
from ...llm.selector import ModelSelector
from ...utils.errors import (
    ProssaAgentError,
    handle_preprocessing_error,
    ValidationError
)

@dataclass
class ActionResult:
    """Result of an action execution"""
    success: bool
    data: Union[pd.DataFrame, str]
    confidence: float
    source: str
    timestamp: datetime
    metadata: Dict[str, Any]

class ActionManager:
    """Manages data processing and LLM interactions"""
    
    def __init__(self):
        self._setup_logging()
        self._initialize_components()
        
    def _setup_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def _initialize_components(self):
        try:
            self.processor = ProsaProcessor()
            self.model_selector = ModelSelector()
            self.memory = MemoryBuffer()
            self.logger.info("Action components initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize action components: {str(e)}")
            raise ProssaAgentError("Component initialization failed", {"error": str(e)})

    async def process_dataset(self, df: pd.DataFrame, context: List[str]) -> ActionResult:
        """Main processing pipeline"""
        try:
            # 1. Analyze dataset using ProsaProcessor
            self.logger.info(f"Starting dataset analysis for shape {df.shape}")
            analysis = await self.processor.analyze_dataset(df)
            
            # 2. Execute preprocessing with integrated plan generation
            self.logger.info("Executing preprocessing pipeline")
            processing_result: ProcessingResult = await self.processor.execute_preprocessing(
                df=df,
                analysis=analysis
            )
            
            if not processing_result.success:
                self.logger.error(f"Processing failed: {processing_result.error}")
                return ActionResult(
                    success=False,
                    data=processing_result.error['message'],
                    confidence=0.0,
                    source="processor_error",
                    timestamp=datetime.now(),
                    metadata={
                        'error': processing_result.error,
                        'analysis': processing_result.analysis
                    }
                )
            
            # 3. Validate results
            self.logger.info("Validating processing results")
            try:
                validation = await self._validate_results(
                    processed_df=processing_result.data,
                    original_df=df
                )
            except ValidationError as ve:
                self.logger.error(f"Validation failed: {str(ve)}")
                return ActionResult(
                    success=False,
                    data=str(ve),
                    confidence=0.0,
                    source="validation_error",
                    timestamp=datetime.now(),
                    metadata={
                        'error': {'message': str(ve), 'details': ve.details},
                        'processing_result': processing_result.plan
                    }
                )
            
            # 4. Store successful result in memory
            await self.memory.add_interaction(
                query="dataset_processing",
                response=str(processing_result.plan),
                metadata={
                    'dataset_shape': df.shape,
                    'validation': validation,
                    'timestamp': datetime.now()
                }
            )
            
            return ActionResult(
                success=True,
                data=processing_result.data,
                confidence=validation['confidence'],
                source="prossa_processor",
                timestamp=datetime.now(),
                metadata={
                    'analysis': analysis,
                    'plan': processing_result.plan,
                    'validation': validation,
                    'context_used': context
                }
            )
            
        except Exception as e:
            error = handle_preprocessing_error(e, {'dataset_shape': df.shape})
            self.logger.error(f"Processing pipeline error: {str(error)}")
            return ActionResult(
                success=False,
                data=str(error),
                confidence=0.0,
                source="pipeline_error",
                timestamp=datetime.now(),
                metadata={"error": error.details}
            )

    async def _validate_results(self, 
                              processed_df: pd.DataFrame, 
                              original_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate preprocessing results"""
        try:
            validation = {
                'row_count_match': len(processed_df) == len(original_df),
                'no_new_nulls': processed_df.isnull().sum().sum() <= original_df.isnull().sum().sum(),
                'data_types_valid': all(dt in ['int64', 'float64', 'object'] 
                                     for dt in processed_df.dtypes),
                'value_ranges_valid': self._check_value_ranges(processed_df)
            }
            
            # Calculate overall confidence
            confidence = sum(1 for v in validation.values() if v) / len(validation)
            
            if confidence < 0.5:
                raise ValidationError(
                    "Validation confidence too low",
                    {
                        'confidence': confidence,
                        'checks': validation
                    }
                )
            
            return {
                'checks': validation,
                'confidence': confidence,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            self.logger.error(f"Validation error: {str(e)}")
            raise ValidationError("Validation failed", {'error': str(e)})

    def _check_value_ranges(self, df: pd.DataFrame) -> bool:
        """Check if numerical values are within reasonable ranges"""
        try:
            for col in df.select_dtypes(include=['int64', 'float64']).columns:
                if df[col].min() < -1e9 or df[col].max() > 1e9:
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Value range check error: {str(e)}")
            return False