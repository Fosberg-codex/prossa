"""
Main RAG-focused agent class for the Prossa framework.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from dataclasses import dataclass

from ..llm.base import BaseLLM
from ..llm.selector import ModelSelector
from ..memory.buffer import MemoryBuffer
from ..integrations.prosa.processor import ProsaProcessor
from .execution.validator import ValidationEngine
from .execution.actions import ActionManager, ActionResult

@dataclass
class AgentConfig:
    """Configuration settings for the Prosa Agent"""
    max_memory_items: int = 1000
    enable_gpu: bool = True
    min_confidence: float = 0.7
    log_level: str = "INFO"

class ProsaAgent:
    """
    Main agent class implementing RAG workflow:
    Query -> Process -> Retrieve Context -> Generate -> Validate -> Return
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self._setup_logging()
        self._initialize_components()
        
    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='agent.log'
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def _initialize_components(self):
        """Initialize core components"""
        try:
            # Initialize model selector
            self.model_selector = ModelSelector()
            
            # Initialize memory system
            self.memory = MemoryBuffer(
                max_items=self.config.max_memory_items,
                use_gpu=self.config.enable_gpu
            )
            
            # Initialize processor
            self.processor = ProsaProcessor(use_gpu=self.config.enable_gpu)
            
            # Initialize action manager
            self.action_manager = ActionManager()
            
            # Initialize validator
            self.validator = ValidationEngine(use_gpu=self.config.enable_gpu)
            
            self.logger.info("Core components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {str(e)}")
            raise

    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the RAG workflow.
        
        Args:
            query: The user's query
            
        Returns:
            Dict containing the response and metadata
        """
        try:
            self.logger.info(f"Processing query: {query}")
            
            # 1. Preprocess query
            processed = await self.action_manager.preprocess_text(query)
            if not processed.success:
                raise ValueError(f"Preprocessing failed: {processed.data}")
            
            # 2. Retrieve relevant context
            context = await self.action_manager.retrieve_context(processed.data)
            if not context.success:
                raise ValueError(f"Context retrieval failed: {context.data}")
            
            # 3. Generate response
            response = await self.action_manager.generate_response(
                query=processed.data,
                context=context.data
            )
            if not response.success:
                raise ValueError(f"Response generation failed: {response.data}")
            
            # 4. Validate response
            validation = await self.validator.validate(
                response=response.data,
                context=context.data,
                query=query
            )
            
            # 5. Store in memory
            await self.memory.add_interaction(
                query=query,
                response=response.data,
                metadata={
                    "confidence": validation.confidence_score,
                    "context_relevance": validation.context_relevance,
                    "response_consistency": validation.response_consistency,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            return {
                "content": response.data,
                "confidence": validation.confidence_score,
                "metadata": {
                    "context_relevance": validation.context_relevance,
                    "response_consistency": validation.response_consistency,
                    "context_count": len(context.data),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return {
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }