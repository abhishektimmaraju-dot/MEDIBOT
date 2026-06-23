"""
MediBot LLM Adapter — Groq client wrapper.

Provides a unified interface for all LLM calls (query classification,
conversation contextualization, answer generation, SQL translation).
"""
import os
from typing import List, Dict, Optional
from groq import Groq

from config.settings import GROQ_API_KEY, LLM_MODEL
from utils.logger import get_logger
from utils.timing import PipelineTimer

logger = get_logger("adapters.llm")


class LLMAdapter:
    """Wraps the Groq LLM client behind a clean interface."""

    def __init__(self):
        if GROQ_API_KEY:
            self.client = Groq(api_key=GROQ_API_KEY)
            logger.info("Groq LLM client initialized successfully")
        else:
            self.client = None
            logger.warning("GROQ_API_KEY not found — LLM generation will be unavailable")

    @property
    def is_available(self) -> bool:
        return self.client is not None

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        model: Optional[str] = None
    ) -> Optional[str]:
        """
        Sends a chat completion request and returns the response text.
        Returns None if the client is unavailable or the call fails.
        """
        if not self.client:
            logger.warning("LLM call skipped — no API key")
            return None

        use_model = model or LLM_MODEL
        try:
            with PipelineTimer("llm_call"):
                completion = self.client.chat.completions.create(
                    model=use_model,
                    messages=messages,
                    temperature=temperature
                )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            return None
