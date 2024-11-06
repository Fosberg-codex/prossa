"""
RAG-focused action system for the Prossa Agent framework.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import logging
from ...integrations.prosa.processor import ProsaProcessor
from ...memory.buffer import MemoryBuffer
from ...llm.selector import ModelSelector

@dataclass
class ActionResult:
    """Result of an action execution"""
    success: bool
    data: Any
    confidence: float
    source: str
    timestamp: datetime
    metadata: Dict[str, Any]

class ActionManager:
    """
    Manages RAG-specific actions:
    - Text preprocessing
    - Context retrieval
    - Response generation
    """
    
    def __init__(self):
        self._setup_logging()
        self._initialize_components()
        
    def _setup_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def _initialize_components(self):
        """Initialize required components"""
        try:
            self.processor = ProsaProcessor()
            self.memory = MemoryBuffer()
            self.model_selector = ModelSelector()
            self.logger.info("Action components initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {str(e)}")
            raise

    async def preprocess_text(self, text: str) -> ActionResult:
        """Preprocess text using Prosa"""
        try:
            processed = await self.processor.process_text(text)
            
            return ActionResult(
                success=True,
                data=processed.content,
                confidence=1.0,  # Preprocessing is deterministic
                source="preprocessor",
                timestamp=datetime.now(),
                metadata={
                    "embeddings_shape": processed.embeddings.shape if processed.embeddings is not None else None,
                    "original_length": len(text),
                    "processed_length": len(processed.content)
                }
            )
        except Exception as e:
            self.logger.error(f"Text preprocessing error: {str(e)}")
            return ActionResult(
                success=False,
                data=str(e),
                confidence=0.0,
                source="preprocessor",
                timestamp=datetime.now(),
                metadata={"error": str(e)}
            )

    async def retrieve_context(self, query: str) -> ActionResult:
        """Retrieve relevant context"""
        try:
            results = await self.memory.search_similar(
                query=query,
                limit=5,
                threshold=0.7
            )
            
            if not results:
                return ActionResult(
                    success=True,
                    data=[],
                    confidence=0.0,
                    source="context_retrieval",
                    timestamp=datetime.now(),
                    metadata={"message": "No relevant context found"}
                )
            
            return ActionResult(
                success=True,
                data=[r["content"] for r in results],
                confidence=sum(r["similarity"] for r in results) / len(results),
                source="context_retrieval",
                timestamp=datetime.now(),
                metadata={
                    "result_count": len(results),
                    "similarities": [r["similarity"] for r in results]
                }
            )
            
        except Exception as e:
            self.logger.error(f"Context retrieval error: {str(e)}")
            return ActionResult(
                success=False,
                data=str(e),
                confidence=0.0,
                source="context_retrieval",
                timestamp=datetime.now(),
                metadata={"error": str(e)}
            )

    async def generate_response(self, 
                              query: str,
                              context: List[str]) -> ActionResult:
        """Generate response using LLM"""
        try:
            # Select appropriate model
            model_class = self.model_selector.select_model(
                task_complexity=0.7,  # Default for RAG
                required_confidence=0.8
            )
            
            # Initialize model
            model = model_class()
            
            # Format prompt with context
            context_str = "\n".join(f"Context {i+1}: {ctx}" for i, ctx in enumerate(context))
            prompt = f"""
            Based on the following context, answer the query.
            
            {context_str}
            
            Query: {query}
            """
            
            # Generate response
            response = model.generate_content(prompt)
            
            return ActionResult(
                success=True,
                data=response.content,
                confidence=response.confidence,
                source="llm",
                timestamp=datetime.now(),
                metadata={
                    "model": model.MODEL_NAME,
                    "context_length": len(context),
                    **response.metadata
                }
            )
            
        except Exception as e:
            self.logger.error(f"Response generation error: {str(e)}")
            return ActionResult(
                success=False,
                data=str(e),
                confidence=0.0,
                source="llm",
                timestamp=datetime.now(),
                metadata={"error": str(e)}
            )