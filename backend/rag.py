import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient, models
from qdrant_client.models import Prefetch, Filter, FieldCondition, MatchValue
from groq import Groq

# Import BM25Vectorizer from ingest to keep vectorization consistent
from ingest import BM25Vectorizer, QDRANT_PATH, COLLECTION_NAME, EMBEDDING_MODEL, BM25_MODEL_PATH

# Load environment variables
load_dotenv()

class RAGPipeline:
    def __init__(self):
        print("Initializing models and loading database client...")
        # Local Qdrant client
        self.qdrant_client = QdrantClient(path=QDRANT_PATH)
        
        # Load SentenceTransformer for dense embeddings
        self.dense_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Load CrossEncoder for reranking
        self.reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # Load BM25 vectorizer from saved state
        if os.path.exists(BM25_MODEL_PATH):
            self.bm25 = BM25Vectorizer.load(BM25_MODEL_PATH)
            print("Loaded BM25 Vectorizer model successfully.")
        else:
            self.bm25 = None
            print("WARNING: BM25 Vectorizer model file not found. Sparse search will not function.")
            
        # Initialize Groq client
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if self.groq_api_key:
            self.llm_client = Groq(api_key=self.groq_api_key)
            print("Groq Client initialized successfully.")
        else:
            self.llm_client = None
            print("WARNING: GROQ_API_KEY not found in environment. LLM generation will be unavailable.")

    def retrieve_hybrid(self, query: str, role: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves candidate document chunks using Qdrant's Reciprocal Rank Fusion (RRF) 
        over dense and sparse vectors, applying strict role-based access filtering.
        
        Enforces user role restrictions at the database level so that unauthorized chunks 
        are completely hidden from retrieval.
        """
        if not self.bm25:
            print("ERROR: BM25 model not loaded. Cannot perform hybrid search.")
            return []

        # 1. Encode query (dense + sparse)
        query_dense = self.dense_model.encode(query).tolist()
        query_sparse = self.bm25.transform(query)
        
        # 2. Build RBAC metadata filter
        # Restricted chunks must carry access roles. If user's role is in access_roles, allow retrieval.
        # Admin has access to all, others are filtered.
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

        # 3. Native Qdrant Fusion query using RRF
        try:
            results = self.qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
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

            # Map search results into a clean dictionary list
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
            print(f"Error during Qdrant search: {e}")
            return []

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Re-evaluates and scores the relevance of candidate chunks using a Cross-Encoder.
        
        Cross-Encoder joint scoring runs joint attention over query and document tokens, 
        improving accuracy by promoting the most contextually relevant chunks to the top.
        """
        if not chunks:
            return []
            
        # Prepare inputs for cross encoder: pairs of (query, chunk_text)
        pairs = [[query, chunk["chunk_text"]] for chunk in chunks]
        scores = self.reranker_model.predict(pairs)
        
        # Update chunks with scores
        for i, score in enumerate(scores):
            chunks[i]["rerank_score"] = float(score)
            
        # Sort chunks by rerank score descending
        sorted_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        
        # Log reranker scores for visibility/auditing
        print(f"\nReranking results for query: '{query}'")
        for idx, c in enumerate(sorted_chunks[:top_k]):
            print(f"Rank {idx+1}: Document: {c['source_document']} | Section: {c['section_title']} | Score: {c['rerank_score']:.4f}")
            
        return sorted_chunks[:top_k]

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Takes contextual chunks and query, prompts Groq LLM, and formats response.
        """
        # Format sources (deduplicated)
        sources = []
        seen_sources = set()
        for c in context_chunks:
            source_key = (c["source_document"], c["section_title"], c["collection"])
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append({
                    "source_document": c["source_document"],
                    "section_title": c["section_title"],
                    "collection": c["collection"]
                })

        if not self.llm_client:
            return {
                "answer": f"Offline Mode (Missing API Key): Retrieved matching context chunks:\n\n" + "\n\n".join([c["content"] for c in context_chunks]),
                "sources": sources
            }

        if not context_chunks:
            return {
                "answer": "I do not have access to any documents that could answer your question based on your role, or no matching documents were found.",
                "sources": []
            }

        # Build context string
        context_str = ""
        for i, c in enumerate(context_chunks):
            context_str += f"--- Document {i+1} ---\n"
            context_str += f"Source: {c['source_document']}\n"
            context_str += f"Section: {c['section_title']}\n"
            context_str += f"Content:\n{c['content']}\n\n"

        system_prompt = (
            "You are MediBot, an internal intelligent healthcare assistant for MediAssist Health Network.\n"
            "Your task is to answer the user's question accurately using ONLY the provided document contexts.\n\n"
            "Guidelines:\n"
            "1. Rely ONLY on the clear facts mentioned in the context. If the context does not contain the answer, say "
            "\"I cannot find the answer in the provided documents.\"\n"
            "2. Provide exact references when possible. Do not extrapolate, assume, or hallucinate.\n"
            "3. Be professional, concise, and direct in your answer."
        )

        user_content = f"CONTEXT DOCUMENTS:\n{context_str}\n\nQUESTION: {query}"

        try:
            completion = self.llm_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0
            )
            answer = completion.choices[0].message.content
            
            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            return {
                "answer": f"Error generating LLM response: {e}",
                "sources": []
            }
