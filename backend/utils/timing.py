"""
MediBot Performance Timing — Tracks latency per pipeline stage.

Provides a context manager and decorator for measuring execution time
of any pipeline stage (routing, retrieval, reranking, LLM generation, SQL execution).
"""
import time
import functools
from typing import Optional
from utils.logger import get_logger

logger = get_logger("timing")


class PipelineTimer:
    """
    Context manager that logs execution time for a named pipeline stage.
    
    Usage:
        with PipelineTimer("hybrid_retrieval"):
            results = rag.retrieve_hybrid(query, role)
        
        # Output: PERF | hybrid_retrieval completed in 0.234s
    """
    def __init__(self, stage_name: str):
        self.stage_name = stage_name
        self.start_time: Optional[float] = None
        self.elapsed: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start_time
        if exc_type:
            logger.error(f"PERF | {self.stage_name} FAILED after {self.elapsed:.3f}s")
        else:
            logger.info(f"PERF | {self.stage_name} completed in {self.elapsed:.3f}s")
        return False  # Do not suppress exceptions


def timed(stage_name: str):
    """
    Decorator that logs execution time for any function.
    
    Usage:
        @timed("sql_execution")
        def execute_sql(query: str) -> dict:
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info(f"PERF | {stage_name} completed in {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error(f"PERF | {stage_name} FAILED after {elapsed:.3f}s: {e}")
                raise
        return wrapper
    return decorator
