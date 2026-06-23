"""
MediBot Answer Generation Service — LLM-powered response formatting.

Takes retrieved context chunks and the user's query, then prompts the LLM
to generate a grounded, professional answer.
"""
from typing import List, Dict, Any

from adapters.llm_adapter import LLMAdapter
from utils.logger import get_logger
from utils.timing import PipelineTimer

logger = get_logger("services.answer")


class AnswerService:
    """Generates LLM answers from retrieved context chunks."""

    def __init__(self, llm_adapter: LLMAdapter):
        self.llm = llm_adapter

    def generate(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
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

        if not self.llm.is_available:
            return {
                "answer": "Offline Mode (Missing API Key): Retrieved matching context chunks:\n\n" +
                          "\n\n".join([c["content"] for c in context_chunks]),
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

        with PipelineTimer("answer_generation"):
            response = self.llm.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ])

        if response:
            logger.info("LLM answer generated successfully")
            return {"answer": response, "sources": sources}
        else:
            logger.error("LLM response was empty or failed")
            return {
                "answer": "Error generating LLM response. Please try again.",
                "sources": []
            }

    def contextualize_question(self, question: str, history) -> str:
        """
        Rewrites the user's latest question into a standalone question
        using the conversation history.
        """
        if not history or not self.llm.is_available:
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
        for msg in history[-5:]:
            formatted_messages.append({"role": msg.role, "content": msg.content})
        formatted_messages.append({"role": "user", "content": f"Follow-up: {question}"})

        with PipelineTimer("question_contextualization"):
            rewritten = self.llm.chat(formatted_messages)

        if rewritten:
            logger.info(f"Rewrote '{question}' → '{rewritten}'")
            return rewritten

        return question
