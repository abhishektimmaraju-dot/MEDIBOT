"""
MediBot Pydantic Schemas — Request and Response Models

All API-facing data structures are defined here for validation,
serialization, and documentation.
"""
from pydantic import BaseModel
from typing import Optional, List


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = None


class SourceInfo(BaseModel):
    source_document: str
    section_title: str
    collection: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    retrieval_type: str
    role: str
