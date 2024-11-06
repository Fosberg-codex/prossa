"""
Memory buffer focused on RAG operations for dataset processing.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import torch
from datetime import datetime
from sentence_transformers import SentenceTransformer
import logging
from dataclasses import dataclass

@dataclass
class MemoryEntry:
    """Single memory entry with embeddings"""
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    timestamp: datetime

class MemoryBuffer:
    """Memory buffer for storing and retrieving processing context"""
    
    def __init__(self, 
                 max_items: int = 1000,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 use_gpu: bool = True):
        self.max_items = max_items
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self._setup_logging()
        self._initialize_components(embedding_model)
        
    def _setup_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def _initialize_components(self, embedding_model: str):
        try:
            self.encoder = SentenceTransformer(embedding_model, device=self.device)
            self.memories: List[MemoryEntry] = []
            self.embeddings: List[np.ndarray] = []
            
            self.logger.info(f"Initialized memory buffer with {embedding_model} on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory components: {str(e)}")
            raise

    async def add_interaction(self, 
                            query: str,
                            response: str,
                            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a query-response interaction to memory"""
        try:
            content = f"Q: {query}\nA: {response}"
            embedding = await self._generate_embedding(content)
            
            entry = MemoryEntry(
                content=content,
                embedding=embedding,
                metadata=metadata or {},
                timestamp=datetime.now()
            )
            
            self._add_entry(entry)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add interaction: {str(e)}")
            return False

    async def get_relevant_context(self, 
                                 query: str, 
                                 k: int = 3) -> List[str]:
        """Retrieve relevant context for a query"""
        try:
            if not self.memories:
                return []
                
            query_embedding = await self._generate_embedding(query)
            
            # Calculate similarities
            similarities = [
                self._cosine_similarity(query_embedding, mem.embedding)
                for mem in self.memories
            ]
            
            # Get top-k relevant memories
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            
            return [self.memories[i].content for i in top_k_indices]
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve context: {str(e)}")
            return []

    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        try:
            with torch.no_grad():
                embedding = self.encoder.encode(text)
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {str(e)}")
            raise

    def _add_entry(self, entry: MemoryEntry):
        """Add entry to memory buffer"""
        if len(self.memories) >= self.max_items:
            self.memories.pop(0)
            self.embeddings.pop(0)
            
        self.memories.append(entry)
        self.embeddings.append(entry.embedding)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    async def get_all_interactions(self) -> List[Dict[str, Any]]:
        """Get all stored interactions"""
        return [
            {
                'content': mem.content,
                'metadata': mem.metadata,
                'timestamp': mem.timestamp
            }
            for mem in self.memories
        ]