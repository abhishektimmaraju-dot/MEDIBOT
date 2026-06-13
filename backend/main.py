import os
from fastapi import FastAPI, HTTPException, Header, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

from auth import authenticate_user, create_access_token, decode_access_token, ROLE_COLLECTIONS
from rag import RAGPipeline
from sql_rag import sql_rag_chain

# Load environment variables
load_dotenv()

app = FastAPI(title="MediBot Backend API", version="1.0.0")

# Enable CORS for Next.js frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate RAG pipeline
# We initialize it globally so it loads once at startup
rag = RAGPipeline()

# API schemas
class LoginRequest(BaseModel):
    username: str
    password: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    role: Optional[str] = None  # Can be passed in body or extracted from token
    history: Optional[List[ChatMessage]] = None

def contextualize_question(question: str, history: Optional[List[ChatMessage]]) -> str:
    """
    Rewrites the user's latest question into a standalone question
    using the conversation history.
    """
    if not history or not rag.llm_client:
        return question

    system_prompt = (
        "You are an expert conversational assistant for a healthcare organization's MediBot.\n"
        "Given the conversation history and a follow-up question, rephrase the follow-up question "
        "into a standalone, self-contained question that can be understood on its own without the history.\n"
        "Do NOT answer the question. Just return the rephrased question.\n\n"
        "Example 1:\n"
        "History:\n"
        "User: Which equipment category has the most open maintenance tickets?\n"
        "Assistant: The equipment category with the most open maintenance tickets is Radiology.\n"
        "Follow-up: how many\n"
        "Standalone: How many open maintenance tickets are there in the Radiology category?\n\n"
        "Example 2:\n"
        "History:\n"
        "User: How many open maintenance tickets are there in the Radiology category?\n"
        "Assistant: There are 85 open maintenance tickets in the Radiology category.\n"
        "Follow-up: can you list them\n"
        "Standalone: Can you list all open maintenance tickets in the Radiology category?\n\n"
        "Example 3:\n"
        "History:\n"
        "User: What is the NSTEMI protocol?\n"
        "Assistant: [Detailed protocol steps]\n"
        "Follow-up: show me claims of cardiology\n"
        "Standalone: Show me claims of cardiology\n"
    )

    # Format history for LLM
    formatted_messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-5:]: # limit to last 5 messages for speed
        formatted_messages.append({"role": msg.role, "content": msg.content})
    formatted_messages.append({"role": "user", "content": f"Follow-up: {question}"})

    try:
        completion = rag.llm_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=formatted_messages,
            temperature=0.0
        )
        rewritten = completion.choices[0].message.content.strip()
        print(f"[Contextualization] Rewrote '{question}' to standalone: '{rewritten}'")
        return rewritten
    except Exception as e:
        print(f"Error in contextualizing question: {e}")
        return question

def get_role_from_token(authorization: Optional[str] = Header(None)) -> str:
    """Extracts role from Bearer token. Falls back or raises HTTPException if invalid."""
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

def classify_query(query: str) -> str:
    """
    Classifies if a question is analytical (database-bound) or document-bound.
    Uses Groq LLM logic.
    """
    if not rag.llm_client:
        # Fallback to keyword matching if LLM is unavailable
        analytical_keywords = ["how many", "count", "average", "total", "sum", "stats", "tickets", "claims", "resolved", "rejected", "escalated"]
        query_lower = query.lower()
        if any(kw in query_lower for kw in analytical_keywords):
            return "analytical"
        return "document"

    system_prompt = (
        "You are an expert query classifier for a hospital RAG system.\n"
        "Classify if the user's question is:\n"
        "1. \"analytical\" (relates to numbers, metrics, statistics, counts, statuses, or records stored in SQLite database tables like claims and maintenance tickets)\n"
        "2. \"document\" (relates to standard operating procedures, clinical guidelines, drug details, leave policy, or general FAQs in text documents)\n\n"
        "Examples of analytical:\n"
        "- \"How many claims were approved last week?\"\n"
        "- \"Give me the total count of radiology maintenance tickets.\"\n"
        "- \"List the claims with status escalated.\"\n\n"
        "Examples of document:\n"
        "- \"What is the room eligibility for clinical staff?\"\n"
        "- \"How to handle an emergency ICU admission?\"\n"
        "- \"What are the steps for battery replacement on an infusion pump?\"\n\n"
        "Output ONLY the word \"analytical\" or \"document\". Do not include any other text."
    )

    try:
        completion = rag.llm_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.0
        )
        classification = completion.choices[0].message.content.strip().lower()
        if "analytical" in classification:
            return "analytical"
        return "document"
    except Exception as e:
        print(f"Error in LLM classification: {e}. Falling back to keywords.")
        # Fallback keyword logic
        analytical_keywords = ["how many", "count", "average", "total", "sum", "stats", "tickets", "claims", "resolved", "rejected", "escalated"]
        query_lower = query.lower()
        if any(kw in query_lower for kw in analytical_keywords):
            return "analytical"
        return "document"

def get_rbac_rejection_message(role: str, query: str) -> str:
    """Returns a tailored rejection message listing what the role can and cannot access."""
    allowed = ROLE_COLLECTIONS.get(role, [])
    allowed_str = ", ".join(allowed)
    
    # Identify what they were likely asking about
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

@app.post("/login")
def login(req: LoginRequest):
    """Authenticates username & password and returns a token."""
    user = authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(
            status_code=400,
            detail="Invalid username or password"
        )
    # Create JWT token
    token = create_access_token(user)
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user["role"],
        "name": user["name"]
    }

@app.post("/chat")
def chat(req: ChatRequest, active_role: str = Depends(get_role_from_token)):
    """
    Main chat router. Classifies query and forwards to SQL RAG or Document Hybrid RAG
    with strict role-based access filtering.
    """
    # Overwrite token role if role is explicitly passed in request body (for testing convenience)
    if req.role:
        active_role = req.role

    raw_question = req.question
    question = contextualize_question(raw_question, req.history)

    print(f"\n[Chat Route] Question: '{raw_question}' (Standalone: '{question}') | Active Role: '{active_role}'")

    # Step 1: Classify question type
    q_type = classify_query(question)
    print(f"[Chat Route] Classified as: {q_type.upper()}")

    # Step 2: Route request
    if q_type == "analytical":
        # Check permissions: only billing_executive and admin can access SQL RAG
        if active_role in ["billing_executive", "admin"]:
            print("[Chat Route] Routing to SQL RAG Chain...")
            answer = sql_rag_chain(question)
            return {
                "answer": answer,
                "sources": [{"source_document": "mediassist.db", "section_title": "SQL Database Tables", "collection": "relational_db"}],
                "retrieval_type": "sql_rag",
                "role": active_role
            }
        else:
            print("[Chat Route] SQL RAG Access Denied for role:", active_role)
            return {
                "answer": f"As a {active_role.replace('_', ' ')}, you do not have permission to run analytical queries over the billing or ticket databases. This feature is restricted to billing executives and administrators.",
                "sources": [],
                "retrieval_type": "sql_rag",
                "role": active_role
            }
    else:
        # Document search: Hybrid RAG + Rerank
        print("[Chat Route] Routing to Hybrid + Reranker RAG pipeline...")
        # Retrieve candidates (applies RBAC filter at vector DB layer)
        retrieved_chunks = rag.retrieve_hybrid(question, active_role, limit=10)
        
        # Rerank candidates (narrow top-10 to top-3)
        reranked_chunks = rag.rerank(question, retrieved_chunks, top_k=3)
        
        # Check if chunks are empty or if the top relevance score is extremely low (meaning no relevant allowed documents exist)
        if not reranked_chunks or reranked_chunks[0]["rerank_score"] < -6.0:
            # Check if user asked about restricted topics
            answer_msg = get_rbac_rejection_message(active_role, question)
            return {
                "answer": answer_msg,
                "sources": [],
                "retrieval_type": "hybrid_rag",
                "role": active_role
            }

            
        # Generate LLM answer using context
        print("[Chat Route] Invoking LLM for response generation...")
        rag_res = rag.generate_answer(question, reranked_chunks)
        
        return {
            "answer": rag_res["answer"],
            "sources": rag_res["sources"],
            "retrieval_type": "hybrid_rag",
            "role": active_role
        }

@app.get("/collections/{role}")
def get_collections(role: str):
    """Returns accessible collections for a specific role."""
    if role not in ROLE_COLLECTIONS:
        raise HTTPException(status_code=400, detail="Invalid role specified")
    return {
        "role": role,
        "collections": ROLE_COLLECTIONS[role]
    }

@app.get("/health")
def health():
    """Simple API healthcheck."""
    return {"status": "healthy"}
