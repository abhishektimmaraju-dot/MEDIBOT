"""
MediBot SQL RAG Service — Natural language to SQL translation and execution.

Pipeline:
  1. Translate NL question to SQL using LLM
  2. Clean and validate the SQL
  3. Execute against SQLite (read-only)
  4. Format results conversationally via LLM
"""
from adapters.llm_adapter import LLMAdapter
from adapters.sql_adapter import clean_sql, execute_sql, is_safe_select
from config.settings import SQL_SCHEMA_PROMPT
from utils.logger import get_logger
from utils.timing import PipelineTimer

logger = get_logger("services.sql_rag")


class SQLRAGService:
    """Translates natural language questions into SQL, executes them, and formats results."""

    def __init__(self, llm_adapter: LLMAdapter):
        self.llm = llm_adapter

    def run(self, question: str) -> str:
        """
        Full SQL RAG pipeline: NL → SQL → Execute → Format response.
        Falls back to rule-based SQL when no API key is available.
        """
        if not self.llm.is_available:
            return self._offline_fallback(question)

        # Step 1: Translate NL to SQL
        with PipelineTimer("nl_to_sql_translation"):
            raw_sql = self.llm.chat([
                {"role": "system", "content": SQL_SCHEMA_PROMPT},
                {"role": "user", "content": f"Translate this question into SQLite: {question}"}
            ])

        if not raw_sql:
            return "Error: Could not translate question to SQL."

        # Step 2: Clean the raw SQL output
        sql_query = clean_sql(raw_sql)
        logger.info(f"Generated SQL: {sql_query}")

        # Step 3: Execute the SQL query
        execution_res = execute_sql(sql_query)

        if not execution_res["success"]:
            return f"Database Execution Error: {execution_res['error']}\nSQL Query tried: {sql_query}"

        # Step 4: Format results conversationally
        with PipelineTimer("sql_response_formatting"):
            user_content = (
                f"Question: {question}\n\n"
                f"Executed SQL Query: {sql_query}\n\n"
                f"Query Result Columns: {execution_res['columns']}\n"
                f"Query Result Rows: {execution_res['data']}\n\n"
                f"Please formulate a natural language answer to the question based on these query results. "
                f"Keep it professional, structured, and concise (using markdown tables where appropriate)."
            )

            s_prompt = (
                "You are a professional assistant reporting data analytics to healthcare executives. "
                "Use the SQL results provided to answer the user's question directly. "
                "If the query result is a single cell or a single number (such as a COUNT, SUM, or AVG), do NOT use a markdown table. Just state the number/result directly in a clear, conversational sentence.\n"
                "If the user's question explicitly asks to list, show, or detail multiple individual records, and the query results contain multiple rows: "
                "1. Present the records in a clean, aligned markdown table. "
                "2. If there are more than 15 rows in the query result, print only the first 10 rows in the markdown table "
                "to keep the response readable, and add a brief summary paragraph below it summarizing the remaining records and stating the total count."
            )

            formatted_answer = self.llm.chat([
                {"role": "system", "content": s_prompt},
                {"role": "user", "content": user_content}
            ])

        return formatted_answer or f"SQL Query Result: {execution_res['data']}"

    @staticmethod
    def _offline_fallback(question: str) -> str:
        """Rule-based fallback for offline testing without a Groq API key."""
        q_lower = question.lower()

        if "cashless" in q_lower and "pending" in q_lower and "cardiology" in q_lower:
            sql_query = "SELECT COUNT(*) FROM claims WHERE claim_type='cashless' AND status='pending' AND department='cardiology'"
            execution_res = execute_sql(sql_query)
            if not execution_res["success"]:
                return f"Database Execution Error: {execution_res['error']}\nSQL Query tried: {sql_query}"
            return f"Offline SQL RAG: The count of cashless claims that are pending in the cardiology department is {execution_res['data'][0][0]}."

        elif "total claimed amount" in q_lower:
            sql_query = "SELECT SUM(claimed_amount) FROM claims"
            execution_res = execute_sql(sql_query)
            if not execution_res["success"]:
                return f"Database Execution Error: {execution_res['error']}\nSQL Query tried: {sql_query}"
            val = execution_res['data'][0][0] or 0.0
            return f"Offline SQL RAG: The total claimed amount across all claims is ${val:,.2f}."

        elif "escalated claims" in q_lower or "escalated claim" in q_lower:
            sql_query = "SELECT COUNT(*) FROM claims WHERE status='escalated'"
            execution_res = execute_sql(sql_query)
            if not execution_res["success"]:
                return f"Database Execution Error: {execution_res['error']}\nSQL Query tried: {sql_query}"
            return f"Offline SQL RAG: The number of escalated claims is {execution_res['data'][0][0]}."

        elif "equipment category with most open tickets" in q_lower or ("category" in q_lower and "open" in q_lower and "tickets" in q_lower):
            sql_query = "SELECT category, COUNT(*) as cnt FROM maintenance_tickets WHERE status='open' GROUP BY category ORDER BY cnt DESC LIMIT 1"
            execution_res = execute_sql(sql_query)
            if not execution_res["success"]:
                return f"Database Execution Error: {execution_res['error']}\nSQL Query tried: {sql_query}"
            if execution_res['data']:
                cat, cnt = execution_res['data'][0]
                return f"Offline SQL RAG: The equipment category with the most open tickets is {cat} (Count: {cnt})."
            return "Offline SQL RAG: No open maintenance tickets found."

        elif "claims by insurer" in q_lower or ("claims" in q_lower and "insurer" in q_lower):
            sql_query = "SELECT insurer, COUNT(*) FROM claims GROUP BY insurer"
            execution_res = execute_sql(sql_query)
            if not execution_res["success"]:
                return f"Database Execution Error: {execution_res['error']}\nSQL Query tried: {sql_query}"
            rows = [f"- {insurer}: {count} claims" for insurer, count in execution_res['data']]
            return "Offline SQL RAG: Claims by insurer:\n" + "\n".join(rows)

        else:
            sql_query = "SELECT COUNT(*) FROM claims"
            execution_res = execute_sql(sql_query)
            if not execution_res["success"]:
                return f"Database Execution Error: {execution_res['error']}\nSQL Query tried: {sql_query}"
            return f"Offline SQL RAG: The total number of claims is {execution_res['data'][0][0]}."
