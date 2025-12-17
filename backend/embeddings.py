"""Embedding generation using sentence-transformers."""

from sentence_transformers import SentenceTransformer
from typing import List, Optional
import numpy as np


class EmbeddingModel:
    """Wrapper for sentence-transformers embedding model."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        """Initialize embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self._model: Optional[SentenceTransformer] = None
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            print(f"Embedding model loaded. Dimension: {self.dimension}")
        return self._model
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text.
        
        Args:
            text: Text to encode
            
        Returns:
            Numpy array of embedding
        """
        return self.model.encode([text], convert_to_numpy=True)[0]


# Global embedding model instance
_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu") -> EmbeddingModel:
    """Get or create the global embedding model.
    
    Args:
        model_name: Name of the sentence-transformers model
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        EmbeddingModel instance
    """
    global _embedding_model
    if _embedding_model is None or _embedding_model.model_name != model_name:
        _embedding_model = EmbeddingModel(model_name, device)
    return _embedding_model
