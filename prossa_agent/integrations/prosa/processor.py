"""
Prosa integration for data preprocessing and RAG capabilities.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
import torch
from sentence_transformers import SentenceTransformer
import prossa  # The actual data preprocessing library

@dataclass
class ProcessedData:
    """Processed data structure"""
    content: Any
    metadata: Dict[str, Any]
    embeddings: Optional[torch.Tensor] = None
    timestamp: datetime = datetime.now()

class ProsaProcessor:
    """
    Handles data preprocessing using prossa library and RAG capabilities.
    """
    
    def __init__(self, 
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 use_gpu: bool = True):
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self._setup_logging()
        self._initialize_components(embedding_model)

    def _setup_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def _initialize_components(self, embedding_model: str):
        try:
            # Initialize embedding model
            self.encoder = SentenceTransformer(embedding_model, device=self.device)
            
            # Initialize prossa preprocessor
            self.preprocessor = prossa.Preprocessor()
            
            self.logger.info(f"Initialized Prosa processor with {embedding_model} on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Prosa processor: {str(e)}")
            raise

    async def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> ProcessedData:
        """Process text using prossa library"""
        try:
            # Use prossa's preprocessing capabilities
            processed_text = self.preprocessor.process_text(
                text,
                remove_stopwords=True,
                remove_punctuation=True,
                lowercase=True
            )
            
            # Generate embeddings
            embeddings = await self._generate_embeddings(processed_text)
            
            return ProcessedData(
                content=processed_text,
                metadata=metadata or {},
                embeddings=embeddings,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Text processing error: {str(e)}")
            raise

    async def _generate_embeddings(self, text: str) -> Optional[torch.Tensor]:
        """Generate embeddings for processed text"""
        try:
            with torch.no_grad():
                return self.encoder.encode(
                    text,
                    convert_to_tensor=True,
                    show_progress_bar=False
                ).to(self.device)
        except Exception as e:
            self.logger.error(f"Embedding generation error: {str(e)}")
            return None