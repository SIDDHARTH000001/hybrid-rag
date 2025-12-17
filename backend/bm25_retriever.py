from rank_bm25 import BM25Okapi
from typing import List, Tuple
import re
from backend.document_processor import DocumentChunk


class BM25Retriever:
    def __init__(self):
        self.bm25: BM25Okapi = None
        self.chunks: List[DocumentChunk] = []
        self.tokenized_corpus: List[List[str]] = []
    
    def add_documents(self, chunks: List[DocumentChunk]):

        self.chunks.extend(chunks)
        
        new_tokenized = [self._tokenize(chunk.text) for chunk in chunks]
        self.tokenized_corpus.extend(new_tokenized)
        
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        print(f"Added {len(chunks)} chunks to BM25 index. Total: {len(self.chunks)}")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:

        if not self.bm25 or len(self.chunks) == 0:
            return []
        
        tokenized_query = self._tokenize(query)
        
        scores = self.bm25.get_scores(tokenized_query)
        
        top_k = min(top_k, len(scores))
        top_indices = scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  
                results.append((self.chunks[idx], float(scores[idx])))
        
        return results
    
    def clear(self):
        self.bm25 = None
        self.chunks = []
        self.tokenized_corpus = []
        print("BM25 index cleared")
    
    def _tokenize(self, text: str) -> List[str]:

        text = text.lower()
        
        tokens = re.findall(r'\b\w+\b', text)
        
        return tokens
    
    def __len__(self) -> int:
        """Get number of chunks in index."""
        return len(self.chunks)
