"""
MediBot Embedding Adapter — Dense and Sparse embedding model wrappers.

Encapsulates model loading so that services never need to know
which specific model library or checkpoint is in use.
"""
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding
from qdrant_client import models

from config.settings import DENSE_EMBEDDING_MODEL, SPARSE_EMBEDDING_MODEL, RERANKER_MODEL
from utils.logger import get_logger
from utils.timing import PipelineTimer

logger = get_logger("adapters.embedding")


class EmbeddingAdapter:
    """Wraps dense, sparse, and cross-encoder models behind a clean interface."""

    def __init__(self):
        logger.info(f"Loading dense embedding model: {DENSE_EMBEDDING_MODEL}")
        self.dense_model = SentenceTransformer(DENSE_EMBEDDING_MODEL)

        logger.info(f"Loading sparse embedding model: {SPARSE_EMBEDDING_MODEL}")
        self.sparse_model = SparseTextEmbedding(model_name=SPARSE_EMBEDDING_MODEL)

        logger.info(f"Loading cross-encoder reranker: {RERANKER_MODEL}")
        self.reranker_model = CrossEncoder(RERANKER_MODEL)
        
        logger.info("All embedding models loaded successfully")

    def encode_dense(self, text: str) -> List[float]:
        """Returns a dense vector for a single query string."""
        return self.dense_model.encode(text).tolist()

    def encode_dense_batch(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Returns dense vectors for a batch of texts."""
        return self.dense_model.encode(texts, show_progress_bar=show_progress)

    def encode_sparse(self, text: str) -> models.SparseVector:
        """Returns a Qdrant-compatible sparse vector for a single query string."""
        emb = list(self.sparse_model.query_embed(text))[0]
        return models.SparseVector(
            indices=emb.indices.tolist(),
            values=emb.values.tolist()
        )

    def encode_sparse_batch(self, texts: List[str]):
        """Returns sparse embeddings for a batch of texts."""
        return list(self.sparse_model.embed(texts))

    def rerank(self, pairs: List[List[str]]) -> List[float]:
        """Scores query-document pairs using the cross-encoder."""
        return [float(s) for s in self.reranker_model.predict(pairs)]
