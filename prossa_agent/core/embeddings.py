from typing import Dict, List, Optional, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

class EmbeddingManager:
    def __init__(self, persist_directory: Optional[str] = "./chroma_db"):
        """Initialize the EmbeddingManager with ChromaDB and SentenceTransformer"""
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create collections for different types of embeddings
        self.recommendations = self.client.get_or_create_collection(
            name="preprocessing_recommendations",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.dataset_patterns = self.client.get_or_create_collection(
            name="dataset_patterns",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize embedding model from environment or default
        model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.embedding_model = SentenceTransformer(model_name)
    
    def store_recommendation(self, 
                           content: str, 
                           metadata: Dict[str, Any],
                           id: str) -> None:
        """Store a preprocessing recommendation with metadata"""
        embeddings = self.embedding_model.encode([content])
        
        self.recommendations.add(
            embeddings=embeddings.tolist(),
            documents=[content],
            metadatas=[metadata],
            ids=[id]
        )
    
    def store_dataset_pattern(self,
                            pattern: str,
                            metadata: Dict[str, Any],
                            id: str) -> None:
        """Store dataset patterns for future reference"""
        embeddings = self.embedding_model.encode([pattern])
        
        self.dataset_patterns.add(
            embeddings=embeddings.tolist(),
            documents=[pattern],
            metadatas=[metadata],
            ids=[id]
        )
    
    def find_similar_recommendations(self,
                                   query: str,
                                   n_results: int = 5,
                                   filters: Optional[Dict] = None) -> Dict[str, List]:
        """Find similar preprocessing recommendations"""
        query_embedding = self.embedding_model.encode(query)
        
        return self.recommendations.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            where=filters
        )
    
    def find_similar_patterns(self,
                            query: str,
                            n_results: int = 5,
                            filters: Optional[Dict] = None) -> Dict[str, List]:
        """Find similar dataset patterns"""
        query_embedding = self.embedding_model.encode(query)
        
        return self.dataset_patterns.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            where=filters
        )
    
    def clear_collections(self):
        """Clear all collections"""
        self.client.delete_collection("preprocessing_recommendations")
        self.client.delete_collection("dataset_patterns")
        
        # Recreate collections
        self.recommendations = self.client.create_collection(
            name="preprocessing_recommendations",
            metadata={"hnsw:space": "cosine"}
        )
        self.dataset_patterns = self.client.create_collection(
            name="dataset_patterns",
            metadata={"hnsw:space": "cosine"}
        ) 