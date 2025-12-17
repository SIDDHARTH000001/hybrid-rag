from typing import List, Tuple, Dict
import numpy as np
from backend.document_processor import DocumentChunk
from backend.vector_store import FAISSVectorStore
from backend.bm25_retriever import BM25Retriever
from backend.embeddings import EmbeddingModel


class HybridSearcher:

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        bm25_retriever: BM25Retriever,
        embedding_model: EmbeddingModel,
        bm25_weight: float = 0.4,
        semantic_weight: float = 0.6
    ):

        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.embedding_model = embedding_model
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight

        total = bm25_weight + semantic_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float, Dict[str, float]]]:

        bm25_results = self.bm25_retriever.search(query, top_k=top_k * 2)
        
        query_embedding = self.embedding_model.encode_single(query)
        semantic_results = self.vector_store.search(query_embedding, top_k=top_k * 2)
        
        bm25_scores = self._normalize_scores(bm25_results)
        semantic_scores = self._normalize_scores(semantic_results)
        
        combined_scores: Dict[int, Tuple[DocumentChunk, float, float, float]] = {}
        
        for chunk, norm_score in bm25_scores:
            chunk_id = chunk.chunk_id
            combined_scores[chunk_id] = (chunk, norm_score * self.bm25_weight, 0.0, 0.0)
        
        for chunk, norm_score in semantic_scores:
            chunk_id = chunk.chunk_id
            if chunk_id in combined_scores:
                chunk_obj, bm25_contrib, _, _ = combined_scores[chunk_id]
                semantic_contrib = norm_score * self.semantic_weight
                combined_scores[chunk_id] = (
                    chunk_obj,
                    bm25_contrib,
                    semantic_contrib,
                    bm25_contrib + semantic_contrib
                )
            else:
                semantic_contrib = norm_score * self.semantic_weight
                combined_scores[chunk_id] = (chunk, 0.0, semantic_contrib, semantic_contrib)
        
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x[3], 
            reverse=True
        )
        
        final_results = []
        for chunk, bm25_score, semantic_score, combined_score in sorted_results[:top_k]:
            score_details = {
                'bm25_score': bm25_score,
                'semantic_score': semantic_score,
                'combined_score': combined_score
            }
            final_results.append((chunk, combined_score, score_details))
        
        return final_results
    
    def _normalize_scores(self, results: List[Tuple[DocumentChunk, float]]) -> List[Tuple[DocumentChunk, float]]:

        if not results:
            return []
        
        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score - min_score < 1e-6:
            return [(chunk, 1.0) for chunk, _ in results]
        
        normalized = []
        for chunk, score in results:
            norm_score = (score - min_score) / (max_score - min_score)
            normalized.append((chunk, norm_score))
        
        return normalized
    
    def add_documents(self, chunks: List[DocumentChunk]):

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress=True)
        
        self.vector_store.add_documents(chunks, embeddings)
        self.bm25_retriever.add_documents(chunks)
    
    def clear(self):
        self.vector_store.clear()
        self.bm25_retriever.clear()
        print("Hybrid searcher cleared")
