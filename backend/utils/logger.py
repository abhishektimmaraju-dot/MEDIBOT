"""
MediBot Structured Logging — Replaces all print() calls with leveled logging.

Configures a shared logger for the entire application with:
  - Console output (StreamHandler) with colored formatting
  - Structured format: timestamp | level | module | message
"""
import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger configured with console output and structured formatting.
    
    Usage:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Pipeline initialized")
        logger.warning("API key not found")
        logger.error("Query failed", exc_info=True)
    """
    logger = logging.getLogger(name)
    
    # Avoid adding duplicate handlers if get_logger is called multiple times
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Prevent log propagation to root logger (avoids duplicate output)
        logger.propagate = False
    
    return logger
