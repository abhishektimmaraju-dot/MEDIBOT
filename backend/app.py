"""
MediBot Application — FastAPI entry point.

Initializes all adapters and services at startup, then mounts the API router.

Usage:
    uvicorn app:app --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from adapters.embedding_adapter import EmbeddingAdapter
from adapters.qdrant_adapter import QdrantAdapter
from adapters.llm_adapter import LLMAdapter
from services.retrieval_service import RetrievalService
from services.answer_service import AnswerService
from services.sql_rag_service import SQLRAGService
from services.semantic_router import SemanticRouter
from api.chat_router import router, init_services
from utils.logger import get_logger

logger = get_logger("app")

# ── Create FastAPI App ───────────────────────────────────────────────────────
app = FastAPI(title="MediBot Backend API", version="2.0.0")

# Enable CORS for Next.js frontend calls
# NOTE: allow_origins=["*"] is configured for demo/local development purposes only.
# For production environments, this must be restricted to the exact authorized domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialize Adapters (infrastructure layer) ──────────────────────────────
logger.info("Initializing adapters...")
embedding_adapter = EmbeddingAdapter()
qdrant_adapter = QdrantAdapter()
llm_adapter = LLMAdapter()

# ── Initialize Services (business logic layer) ──────────────────────────────
logger.info("Initializing services...")
retrieval_service = RetrievalService(embedding_adapter, qdrant_adapter)
answer_service = AnswerService(llm_adapter)
sql_rag_service = SQLRAGService(llm_adapter)
semantic_router = SemanticRouter(embedding_adapter)

# ── Inject Services into API Router ─────────────────────────────────────────
init_services(retrieval_service, answer_service, sql_rag_service, semantic_router)

# ── Mount Router ─────────────────────────────────────────────────────────────
app.include_router(router)

logger.info("MediBot API v2.0.0 ready — all systems initialized")
