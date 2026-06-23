"""
MediBot Retrieval Service — Hybrid search + cross-encoder reranking.

Orchestrates the embedding adapter and Qdrant adapter to perform:
  1. Dense + sparse embedding of the query
  2. Hybrid search with RBAC filtering
  3. Cross-encoder reranking of candidates
"""
from typing import List, Dict, Any

from adapters.embedding_adapter import EmbeddingAdapter
from adapters.qdrant_adapter import QdrantAdapter
from config.settings import HYBRID_SEARCH_LIMIT, RERANKER_TOP_K
from utils.logger import get_logger
from utils.timing import PipelineTimer

logger = get_logger("services.retrieval")


class RetrievalService:
    """Handles the full retrieval pipeline: embed → search → rerank."""

    def __init__(self, embedding_adapter: EmbeddingAdapter, qdrant_adapter: QdrantAdapter):
        self.embedding = embedding_adapter
        self.qdrant = qdrant_adapter

    def retrieve_hybrid(self, query: str, role: str, limit: int = HYBRID_SEARCH_LIMIT) -> List[Dict[str, Any]]:
        """
        Retrieves candidate document chunks using Qdrant's Reciprocal Rank Fusion (RRF)
        over dense and sparse vectors, applying strict role-based access filtering.
        """
        with PipelineTimer("hybrid_retrieval"):
            # 1. Encode query (dense + sparse)
            query_dense = self.embedding.encode_dense(query)
            query_sparse = self.embedding.encode_sparse(query)

            # 2. Hybrid search with RBAC filter
            chunks = self.qdrant.hybrid_search(query_dense, query_sparse, role, limit)

        logger.info(f"Retrieved {len(chunks)} chunks for role '{role}'")
        return chunks

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = RERANKER_TOP_K) -> List[Dict[str, Any]]:
        """
        Re-evaluates and scores the relevance of candidate chunks using a Cross-Encoder.
        Cross-Encoder joint scoring runs joint attention over query and document tokens.
        """
        if not chunks:
            return []

        with PipelineTimer("cross_encoder_reranking"):
            # Prepare input pairs for cross encoder
            pairs = [[query, chunk["chunk_text"]] for chunk in chunks]
            scores = self.embedding.rerank(pairs)

            # Attach scores to chunks
            for i, score in enumerate(scores):
                chunks[i]["rerank_score"] = score

            # Sort by score descending
            sorted_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

        # Log reranker scores for auditing
        for idx, c in enumerate(sorted_chunks[:top_k]):
            logger.info(
                f"Rank {idx+1}: {c['source_document']} | "
                f"Section: {c['section_title']} | Score: {c['rerank_score']:.4f}"
            )

        return sorted_chunks[:top_k]
