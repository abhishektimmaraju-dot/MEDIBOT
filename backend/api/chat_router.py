"""
MediBot API Router — All FastAPI route handlers.

Routes:
  POST /login          — Authenticate and receive JWT token
  POST /chat           — Main chat endpoint (routes to Semantic Router → SQL RAG or Hybrid RAG)
  GET  /collections/{role} — Returns accessible collections for a role
  GET  /health         — API healthcheck
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Header, Depends, status

from models.schemas import LoginRequest, ChatRequest
from security.auth import authenticate_user, create_access_token, decode_access_token
from config.settings import ROLE_COLLECTIONS, COLLECTION_KEYWORDS, RERANKER_SCORE_THRESHOLD
from utils.logger import get_logger
from utils.timing import PipelineTimer

logger = get_logger("api.chat_router")

router = APIRouter()

# ── These will be injected by app.py at startup ──────────────────────────────
_retrieval_service = None
_answer_service = None
_sql_rag_service = None
_semantic_router = None


def init_services(retrieval_service, answer_service, sql_rag_service, semantic_router):
    """Called once by app.py to inject service instances into the router."""
    global _retrieval_service, _answer_service, _sql_rag_service, _semantic_router
    _retrieval_service = retrieval_service
    _answer_service = answer_service
    _sql_rag_service = sql_rag_service
    _semantic_router = semantic_router
    logger.info("API router services injected")


# ── Auth Dependency ──────────────────────────────────────────────────────────

def get_role_from_token(authorization: Optional[str] = Header(None)) -> str:
    """Extracts role from Bearer token. Raises HTTPException if invalid."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization Header"
        )
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token scheme"
            )
        payload = decode_access_token(token)
        if not payload or "role" not in payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired session token"
            )
        return payload["role"]
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed authorization header"
        )


# ── RBAC Helpers ─────────────────────────────────────────────────────────────

def is_query_restricted_for_role(role: str, query: str) -> bool:
    """Checks if the query targets collections that are restricted for the user's role."""
    if role == "admin":
        return False

    query_lower = query.lower()
    allowed = ROLE_COLLECTIONS.get(role, [])

    for coll, keywords in COLLECTION_KEYWORDS.items():
        if coll not in allowed:
            if any(kw in query_lower for kw in keywords):
                return True

    return False


def get_rbac_rejection_message(role: str, query: str) -> str:
    """Returns a tailored rejection message listing what the role can and cannot access."""
    allowed = ROLE_COLLECTIONS.get(role, [])
    allowed_str = ", ".join(allowed)

    query_lower = query.lower()
    denied_topic = "billing and finance"
    if "calibrat" in query_lower or "equipment" in query_lower or "maintenance" in query_lower or "pump" in query_lower:
        denied_topic = "medical equipment manuals"
    elif "drug" in query_lower or "dosage" in query_lower or "clinical" in query_lower or "treatment" in query_lower:
        denied_topic = "clinical treatment protocols"
    elif "infect" in query_lower or "icu" in query_lower or "nurs" in query_lower:
        denied_topic = "nursing procedures"

    return (
        f"Access Denied: As a {role.replace('_', ' ')}, you don't have access to {denied_topic} documents. "
        f"I can only retrieve information from your permitted collections: [{allowed_str}]."
    )


# ── Route Handlers ───────────────────────────────────────────────────────────

@router.post("/login")
def login(req: LoginRequest):
    """Authenticates username & password and returns a token."""
    user = authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid username or password")
    token = create_access_token(user)
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user["role"],
        "name": user["name"]
    }


@router.post("/chat")
def chat(req: ChatRequest, active_role: str = Depends(get_role_from_token)):
    """
    Main chat router. Uses Semantic Router to classify query, then forwards
    to SQL RAG or Document Hybrid RAG with strict role-based access filtering.
    """
    with PipelineTimer("total_chat_request"):
        raw_question = req.question

        # Step 0: Contextualize follow-up questions
        question = _answer_service.contextualize_question(raw_question, req.history)
        logger.info(f"Question: '{raw_question}' (Standalone: '{question}') | Role: '{active_role}'")

        # Step 1: Classify query via Semantic Router (embedding-based)
        q_type = _semantic_router.classify_with_fallback(question)
        logger.info(f"Route decision: {q_type.upper()}")

        # Step 2: Route request
        if q_type == "off_topic":
            logger.info("Routing to Off-Topic Rejection")
            return {
                "answer": "I can only assist with queries related to MediAssist operations, billing, claims, equipment maintenance, and clinical procedures. Your question appears to be outside this scope.",
                "sources": [],
                "retrieval_type": "off_topic",
                "role": active_role
            }
        elif q_type == "analytical":
            # Check permissions: only billing_executive and admin can access SQL RAG
            if active_role in ["billing_executive", "admin"]:
                logger.info("Routing to SQL RAG Chain")
                answer = _sql_rag_service.run(question)
                return {
                    "answer": answer,
                    "sources": [{"source_document": "mediassist.db", "section_title": "SQL Database Tables", "collection": "relational_db"}],
                    "retrieval_type": "sql_rag",
                    "role": active_role
                }
            else:
                logger.warning(f"SQL RAG access denied for role: {active_role}")
                return {
                    "answer": f"As a {active_role.replace('_', ' ')}, you do not have permission to run analytical queries over the billing or ticket databases. This feature is restricted to billing executives and administrators.",
                    "sources": [],
                    "retrieval_type": "sql_rag",
                    "role": active_role
                }
        else:
            # Document search: Hybrid RAG + Rerank
            logger.info("Routing to Hybrid + Reranker RAG pipeline")

            # Retrieve candidates (applies RBAC filter at vector DB layer)
            retrieved_chunks = _retrieval_service.retrieve_hybrid(question, active_role, limit=10)

            # Primary Security Guardrail: zero chunks = RBAC block or no relevant content
            if not retrieved_chunks:
                logger.info(f"Zero chunks returned for role '{active_role}'")
                if is_query_restricted_for_role(active_role, question):
                    answer_msg = get_rbac_rejection_message(active_role, question)
                else:
                    answer_msg = "I could not find relevant information in the provided documents to answer your question."
                return {
                    "answer": answer_msg,
                    "sources": [],
                    "retrieval_type": "hybrid_rag",
                    "role": active_role
                }

            # Rerank candidates (narrow top-10 to top-3)
            reranked_chunks = _retrieval_service.rerank(question, retrieved_chunks, top_k=3)

            # Secondary Confidence Guardrail: cross-encoder score threshold
            if not reranked_chunks or reranked_chunks[0]["rerank_score"] < RERANKER_SCORE_THRESHOLD:
                score = reranked_chunks[0]['rerank_score'] if reranked_chunks else None
                logger.info(f"Low confidence block: rank-1 score {score} is below {RERANKER_SCORE_THRESHOLD}")
                if is_query_restricted_for_role(active_role, question):
                    answer_msg = get_rbac_rejection_message(active_role, question)
                else:
                    answer_msg = "I could not find relevant information in the provided documents to answer your question."
                return {
                    "answer": answer_msg,
                    "sources": [],
                    "retrieval_type": "hybrid_rag",
                    "role": active_role
                }

            # Generate LLM answer using context
            logger.info("Invoking LLM for response generation")
            rag_res = _answer_service.generate(question, reranked_chunks)

            return {
                "answer": rag_res["answer"],
                "sources": rag_res["sources"],
                "retrieval_type": "hybrid_rag",
                "role": active_role
            }


@router.get("/collections/{role}")
def get_collections(role: str):
    """Returns accessible collections for a specific role."""
    if role not in ROLE_COLLECTIONS:
        raise HTTPException(status_code=400, detail="Invalid role specified")
    return {
        "role": role,
        "collections": ROLE_COLLECTIONS[role]
    }


@router.get("/health")
def health():
    """Simple API healthcheck."""
    return {"status": "healthy"}
