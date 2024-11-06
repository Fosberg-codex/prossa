"""
RAG-focused validation engine for the Prossa Agent framework.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

@dataclass
class ValidationResult:
    """Result of validation process"""
    confidence_score: float
    context_relevance: float
    response_consistency: float
    metadata: Dict[str, Any]
    timestamp: datetime

class ValidationEngine:
    """
    Handles validation focused on RAG operations:
    - Context relevance checking
    - Response-context consistency
    - Basic quality checks
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
            self.encoder = SentenceTransformer(embedding_model, device=self.device)
            self.logger.info(f"Initialized validation engine with {embedding_model} on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to initialize validation components: {str(e)}")
            raise

    async def validate(self,
                      response: str,
                      context: List[str],
                      query: str) -> ValidationResult:
        """
        Validate response against context and query.
        
        Args:
            response: Generated response
            context: Retrieved context
            query: Original query
            
        Returns:
            ValidationResult containing validation metrics
        """
        try:
            # Generate embeddings
            response_embedding = self._generate_embedding(response)
            context_embeddings = self._generate_embeddings(context)
            query_embedding = self._generate_embedding(query)
            
            # Calculate metrics
            context_relevance = self._calculate_context_relevance(
                query_embedding, context_embeddings
            )
            
            response_consistency = self._calculate_response_consistency(
                response_embedding, context_embeddings
            )
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(
                context_relevance,
                response_consistency
            )
            
            return ValidationResult(
                confidence_score=confidence,
                context_relevance=context_relevance,
                response_consistency=response_consistency,
                metadata=self._create_metadata(len(context)),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            raise

    def _generate_embedding(self, text: str) -> torch.Tensor:
        """Generate embedding for single text"""
        with torch.no_grad():
            return self.encoder.encode(
                text,
                convert_to_tensor=True,
                show_progress_bar=False
            ).to(self.device)

    def _generate_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Generate embeddings for multiple texts"""
        with torch.no_grad():
            return self.encoder.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False
            ).to(self.device)

    def _calculate_context_relevance(self,
                                   query_embedding: torch.Tensor,
                                   context_embeddings: torch.Tensor) -> float:
        """Calculate relevance of context to query"""
        similarities = F.cosine_similarity(
            query_embedding.unsqueeze(0),
            context_embeddings
        )
        return float(torch.mean(similarities).cpu())

    def _calculate_response_consistency(self,
                                     response_embedding: torch.Tensor,
                                     context_embeddings: torch.Tensor) -> float:
        """Calculate consistency of response with context"""
        similarities = F.cosine_similarity(
            response_embedding.unsqueeze(0),
            context_embeddings
        )
        return float(torch.mean(similarities).cpu())

    def _calculate_confidence(self,
                            context_relevance: float,
                            response_consistency: float) -> float:
        """Calculate overall confidence score"""
        return (context_relevance + response_consistency) / 2.0

    def _create_metadata(self, context_count: int) -> Dict[str, Any]:
        """Create metadata for validation result"""
        return {
            "context_count": context_count,
            "validation_device": self.device,
            "timestamp": datetime.now().isoformat()
        }