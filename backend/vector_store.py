import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from backend.document_processor import DocumentChunk


class FAISSVectorStore:
    def __init__(self, dimension: int, index_path: str = "./data/faiss_index", metadata_path: str = "./data/metadata.pkl"):

        self.dimension = dimension
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks: List[DocumentChunk] = []
        self.chunk_texts: List[str] = []
        
        if self.index_path.exists():
            self.load()
    
    def add_documents(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")
        
        self.index.add(embeddings.astype('float32'))
        
        self.chunks.extend(chunks)
        self.chunk_texts.extend([chunk.text for chunk in chunks])
        
        print(f"Added {len(chunks)} chunks to vector store. Total: {len(self.chunks)}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:

        if len(self.chunks) == 0:
            return []
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        distances, indices = self.index.search(query_embedding.astype('float32'), min(top_k, len(self.chunks)))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks): 
                similarity = 1.0 / (1.0 + dist)
                results.append((self.chunks[idx], similarity))
        
        return results
    
    def clear(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
        self.chunk_texts = []
    
    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(self.index_path))
        
        with open(self.metadata_path, 'wb') as f:
            pickle.dump({
                'chunks': [chunk.to_dict() for chunk in self.chunks],
                'chunk_texts': self.chunk_texts
            }, f)
        
        print(f"Vector store saved to {self.index_path}")
    
    def load(self):
        if not self.index_path.exists() or not self.metadata_path.exists():
            print("No existing vector store found")
            return
        
        self.index = faiss.read_index(str(self.index_path))
        
        with open(self.metadata_path, 'rb') as f:
            data = pickle.load(f)
            
            self.chunks = [
                DocumentChunk(
                    text=chunk_dict['text'],
                    chunk_id=chunk_dict['chunk_id'],
                    page_number=chunk_dict['page_number'],
                    line_number=chunk_dict['line_number'],
                    section=chunk_dict.get('section', ''),
                    file_name=chunk_dict.get('file_name', '')
                )
                for chunk_dict in data['chunks']
            ]
            self.chunk_texts = data['chunk_texts']
        
        print(f"Loaded vector store with {len(self.chunks)} chunks")
    
    def get_all_texts(self) -> List[str]:
        return self.chunk_texts
    
    def __len__(self) -> int:
        return len(self.chunks)
