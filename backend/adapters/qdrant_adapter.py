"""
MediBot Qdrant Adapter — Vector database client wrapper.

Handles connection management and provides typed methods for
hybrid search, collection management, and point upload.
"""
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Prefetch, Filter, FieldCondition, MatchValue,
    Distance, VectorParams, SparseVectorParams, SparseIndexParams, PointStruct
)

from config.settings import QDRANT_PATH, COLLECTION_NAME, DENSE_EMBEDDING_DIM
from utils.logger import get_logger

logger = get_logger("adapters.qdrant")


class QdrantAdapter:
    """Wraps all Qdrant operations behind a clean interface."""

    def __init__(self, path: str = QDRANT_PATH):
        logger.info(f"Connecting to Qdrant at: {path}")
        self.client = QdrantClient(path=path)
        self.collection_name = COLLECTION_NAME

    def hybrid_search(
        self,
        query_dense: List[float],
        query_sparse: models.SparseVector,
        role: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Performs hybrid (dense + sparse) search with RBAC metadata filtering
        using Qdrant's native Reciprocal Rank Fusion (RRF).
        """
        # Build RBAC metadata filter — admin gets no filter
        rbac_filter = None
        if role != "admin":
            rbac_filter = Filter(
                must=[
                    FieldCondition(
                        key="access_roles",
                        match=MatchValue(value=role)
                    )
                ]
            )

        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    Prefetch(
                        query=query_dense,
                        using="text-dense",
                        filter=rbac_filter,
                        limit=limit * 2
                    ),
                    Prefetch(
                        query=query_sparse,
                        using="text-sparse",
                        filter=rbac_filter,
                        limit=limit * 2
                    )
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit
            )

            chunks = []
            for hit in results.points:
                chunks.append({
                    "id": hit.id,
                    "score": hit.score,
                    "content": hit.payload["content"],
                    "chunk_text": hit.payload["chunk_text"],
                    "source_document": hit.payload["source_document"],
                    "collection": hit.payload["collection"],
                    "section_title": hit.payload["section_title"],
                    "chunk_type": hit.payload["chunk_type"]
                })
            return chunks
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}", exc_info=True)
            return []

    def create_collection(self):
        """Creates the medibot collection with dense + sparse vector config."""
        if self.client.collection_exists(self.collection_name):
            logger.info(f"Deleting existing collection '{self.collection_name}'")
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "text-dense": VectorParams(
                    size=DENSE_EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            },
            sparse_vectors_config={
                "text-sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=True)
                )
            }
        )

        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="access_roles",
            field_schema="keyword"
        )
        logger.info(f"Created collection '{self.collection_name}' with hybrid vector config + payload index")

    def upload_points(self, points: List[PointStruct], batch_size: int = 100):
        """Uploads points to Qdrant in batches."""
        total_batches = (len(points) - 1) // batch_size + 1
        for offset in range(0, len(points), batch_size):
            batch = points[offset:offset + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=batch
            )
            batch_num = offset // batch_size + 1
            logger.info(f"Uploaded batch {batch_num}/{total_batches}")
