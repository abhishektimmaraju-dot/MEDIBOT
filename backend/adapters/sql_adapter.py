"""
MediBot SQL Adapter — SQLite database operations.

Handles read-only database connections and query execution
with safety validation to prevent SQL injection or mutation.
"""
import os
import re
import sqlite3
from typing import Dict, Any

from config.settings import DB_PATH
from utils.logger import get_logger
from utils.timing import timed

logger = get_logger("adapters.sql")


def is_safe_select(query: str) -> bool:
    """
    Checks that the query is a single read-only SELECT statement.
    Rejects any statement that begins with other commands or contains data-modifying keywords.
    """
    q = query.strip()
    if not q:
        return False

    # Block multiple statements separated by semicolon
    parts = q.split(";")
    non_empty_parts = [p.strip() for p in parts if p.strip()]
    if len(non_empty_parts) > 1:
        return False

    q_upper = q.upper()

    # Must start with SELECT or WITH
    if not (q_upper.startswith("SELECT") or q_upper.startswith("WITH")):
        return False

    # Block forbidden data-modifying keywords using word boundaries
    forbidden = [
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
        "REPLACE", "TRUNCATE", "RENAME", "GRANT", "REVOKE", "ATTACH", "DETACH"
    ]

    for kw in forbidden:
        pattern = rf"\b{kw}\b"
        if re.search(pattern, q_upper):
            return False

    return True


def clean_sql(raw_sql: str) -> str:
    """
    Extracts the raw SQL statement from the LLM's output.
    Removes any surrounding markdown fence blocks (e.g., ```sql ... ```) or whitespace.
    """
    sql = raw_sql.strip()
    if sql.startswith("```"):
        newline_idx = sql.find("\n")
        if newline_idx != -1:
            sql = sql[newline_idx + 1:]
        if sql.endswith("```"):
            sql = sql[:-3]

    sql = re.sub(r"^sql\s+", "", sql.strip(), flags=re.IGNORECASE)
    sql = sql.strip().rstrip(";")
    return sql


@timed("sql_execution")
def execute_sql(query: str) -> Dict[str, Any]:
    """
    Executes the SQLite query and returns results.
    Opens the SQLite connection in read-only mode.
    """
    if not is_safe_select(query):
        logger.warning(f"Blocked unsafe SQL query: {query[:100]}")
        return {
            "success": False,
            "error": "Query validation failed. Only read-only SELECT statements are allowed."
        }

    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        conn.close()
        logger.info(f"SQL executed successfully — {len(results)} rows returned")
        return {"success": True, "columns": col_names, "data": results}
    except Exception as e:
        if 'conn' in locals():
            conn.close()
        logger.error(f"SQL execution failed: {e}")
        return {"success": False, "error": str(e)}
