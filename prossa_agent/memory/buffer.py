"""
Memory buffer implementation focused on RAG vector storage and retrieval.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from dataclasses import dataclass

@dataclass
class MemoryEntry:
    """A single memory entry"""
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    timestamp: datetime

class MemoryBuffer:
    """
    Memory buffer focused on RAG operations:
    - Vector storage for context
    - Efficient retrieval
    - Basic cleanup
    """
    
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

    async def search_similar(self, 
                           query: str,
                           limit: int = 5,
                           threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar memories"""
        try:
            if not self.memories:
                return []
            
            query_embedding = await self._generate_embedding(query)
            
            similarities = []
            for idx, memory in enumerate(self.memories):
                similarity = self._calculate_similarity(query_embedding, memory.embedding)
                if similarity >= threshold:
                    similarities.append((similarity, idx))
            
            similarities.sort(reverse=True)
            results = []
            
            for similarity, idx in similarities[:limit]:
                memory = self.memories[idx]
                results.append({
                    "content": memory.content,
                    "similarity": float(similarity),
                    "timestamp": memory.timestamp.isoformat(),
                    "metadata": memory.metadata
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search error: {str(e)}")
            return []

    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        with torch.no_grad():
            return self.encoder.encode(text, convert_to_numpy=True)

    def _calculate_similarity(self, 
                            embedding1: np.ndarray,
                            embedding2: np.ndarray) -> float:
        """Calculate cosine similarity"""
        return float(np.dot(embedding1, embedding2) / 
                    (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))

    def _add_entry(self, entry: MemoryEntry):
        """Add entry to memory, maintaining size limit"""
        self.memories.append(entry)
        self.embeddings.append(entry.embedding)
        
        if len(self.memories) > self.max_items:
            self.memories.pop(0)
            self.embeddings.pop(0)

    def clear(self):
        """Clear all memories"""
        self.memories.clear()
        self.embeddings.clear()