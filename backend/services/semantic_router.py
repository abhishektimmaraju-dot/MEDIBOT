"""
MediBot Semantic Router — Embedding-based query classification.

Replaces the LLM-based classify_query() with a true semantic router that
pre-computes embedding centroids for each route category and classifies
incoming queries by cosine similarity — no LLM call needed.

How It Works:
  1. At initialization, we define "route utterances" — example queries
     that represent each route category (analytical vs. document).
  2. Each utterance is embedded using the same dense model used for RAG.
  3. The mean embedding for each route becomes the "route centroid."
  4. At query time, we embed the user's query and compute cosine similarity
     against each route centroid.
  5. The route with the highest similarity wins, provided it exceeds the
     confidence threshold. Below the threshold, we fall back to keyword matching.

Advantages over LLM Classification:
  - Latency: ~5ms (local embedding) vs. ~200-500ms (Groq API call)
  - Cost: Zero (runs locally) vs. 1 API call per message
  - Determinism: Fully deterministic (same input = same output)
  - Offline: Works without internet (no API dependency)
"""
import numpy as np
from typing import List, Tuple

from config.settings import SEMANTIC_ROUTER_THRESHOLD
from utils.logger import get_logger
from utils.timing import PipelineTimer

logger = get_logger("services.semantic_router")

# ── Route Definitions ────────────────────────────────────────────────────────
# Each route has a name and a list of representative utterances.
# These utterances are embedded and averaged to form a "route centroid."

ANALYTICAL_UTTERANCES = [
    "How many claims were approved last week?",
    "Give me the total count of radiology maintenance tickets.",
    "List the claims with status escalated.",
    "What is the average claimed amount in cardiology?",
    "Show me the number of pending cashless claims.",
    "How many maintenance tickets are open?",
    "Total rejected claims by insurer.",
    "Count of escalated claims in neurology department.",
    "What is the sum of approved amounts for Star Health?",
    "How many tickets have fault code sensor failure?",
    "Show claims statistics by department.",
    "Give me the breakdown of claims by status.",
    "What is the total claimed amount across all departments?",
    "List all maintenance tickets for infusion equipment.",
    "How many resolved tickets are there in the Pune campus?",
]

DOCUMENT_UTTERANCES = [
    "What is the room eligibility for clinical staff?",
    "How to handle an emergency ICU admission?",
    "What are the steps for battery replacement on an infusion pump?",
    "What is the NSTEMI treatment protocol?",
    "Tell me about infection control procedures.",
    "What drugs are listed in the formulary for cardiac arrest?",
    "What is the leave policy for nurses?",
    "How do I calibrate a defibrillator?",
    "Explain the cashless pre-authorization process.",
    "What are the billing code guidelines for reimbursement?",
    "What is the staff code of conduct?",
    "Describe the ICU nursing shift handoff procedure.",
    "What diagnostic reference tests are needed for nephrology?",
    "What is the equipment maintenance SOP for ventilators?",
    "How to file a claim submission for cashless patients?",
]


class SemanticRouter:
    """
    Embedding-based semantic router that classifies queries into
    'analytical' or 'document' categories using cosine similarity
    against pre-computed route centroids.
    """

    def __init__(self, embedding_adapter):
        """
        Initializes the router by computing route centroids from example utterances.
        
        Args:
            embedding_adapter: An EmbeddingAdapter instance (provides encode_dense_batch).
        """
        logger.info("Initializing Semantic Router — computing route centroids...")

        with PipelineTimer("semantic_router_init"):
            # Encode all utterances in a single batch for efficiency
            all_utterances = ANALYTICAL_UTTERANCES + DOCUMENT_UTTERANCES
            all_embeddings = embedding_adapter.encode_dense_batch(all_utterances, show_progress=False)

            # Split embeddings back into route groups
            n_analytical = len(ANALYTICAL_UTTERANCES)
            analytical_embeddings = all_embeddings[:n_analytical]
            document_embeddings = all_embeddings[n_analytical:]

            # Compute centroids (mean of each route's embeddings)
            self.analytical_centroid = np.mean(analytical_embeddings, axis=0)
            self.document_centroid = np.mean(document_embeddings, axis=0)

            # Normalize centroids for fast cosine similarity via dot product
            self.analytical_centroid = self.analytical_centroid / np.linalg.norm(self.analytical_centroid)
            self.document_centroid = self.document_centroid / np.linalg.norm(self.document_centroid)

        self.embedding_adapter = embedding_adapter
        self.threshold = SEMANTIC_ROUTER_THRESHOLD

        logger.info(f"Semantic Router ready — threshold: {self.threshold}")

    def classify(self, query: str) -> Tuple[str, float]:
        """
        Classifies a query as 'analytical' or 'document' using cosine similarity.
        
        Returns:
            Tuple of (route_name, confidence_score)
            - route_name: 'analytical' or 'document'
            - confidence_score: cosine similarity to the winning centroid
        """
        with PipelineTimer("semantic_routing"):
            # Encode the query
            query_vec = np.array(self.embedding_adapter.encode_dense(query))
            query_vec = query_vec / np.linalg.norm(query_vec)

            # Compute cosine similarity to each centroid
            analytical_score = float(np.dot(query_vec, self.analytical_centroid))
            document_score = float(np.dot(query_vec, self.document_centroid))

        logger.info(
            f"Semantic Router scores — analytical: {analytical_score:.4f}, "
            f"document: {document_score:.4f}"
        )

        # Return the route with the highest similarity
        if analytical_score > document_score:
            return "analytical", analytical_score
        else:
            return "document", document_score

    def classify_with_fallback(self, query: str) -> str:
        """
        Classifies a query with keyword fallback if the semantic similarity
        scores are too close together (below the margin threshold).
        
        This is the primary entry point for the chat router.
        """
        route, confidence = self.classify(query)

        # Compute margin between top and second route
        query_vec = np.array(self.embedding_adapter.encode_dense(query))
        query_vec = query_vec / np.linalg.norm(query_vec)
        analytical_score = float(np.dot(query_vec, self.analytical_centroid))
        document_score = float(np.dot(query_vec, self.document_centroid))
        margin = abs(analytical_score - document_score)

        if margin < 0.05:
            # Scores too close — use keyword fallback for safety
            logger.info(f"Semantic margin too narrow ({margin:.4f}), falling back to keywords")
            return self._keyword_fallback(query)

        logger.info(f"Semantic Router decision: {route.upper()} (confidence: {confidence:.4f}, margin: {margin:.4f})")
        return route

    @staticmethod
    def _keyword_fallback(query: str) -> str:
        """
        Keyword-based fallback classification when semantic similarity is inconclusive.
        """
        analytical_keywords = [
            "how many", "count", "average", "total", "sum", "stats",
            "tickets", "claims", "resolved", "rejected", "escalated",
            "number of", "breakdown", "statistics"
        ]
        query_lower = query.lower()
        if any(kw in query_lower for kw in analytical_keywords):
            logger.info("Keyword fallback classified as: ANALYTICAL")
            return "analytical"
        logger.info("Keyword fallback classified as: DOCUMENT")
        return "document"
