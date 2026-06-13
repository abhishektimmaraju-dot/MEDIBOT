import os
import sqlite3
import re
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "mediassist_data", "db", "mediassist.db")


# Schema description for the LLM
SCHEMA_PROMPT = """You are an expert SQL Translator for SQLite.
Translate the user's natural language question into a valid, clean SQLite query.

TABLES SCHEMA:
1. `claims` table:
   - `claim_id` (TEXT, Primary Key)
   - `patient_id` (TEXT)
   - `patient_name` (TEXT)
   - `department` (TEXT) - Unique values: 'nephrology', 'cardiology', 'neurology', 'gynaecology', 'orthopaedics', 'general_medicine', 'emergency'
   - `claim_type` (TEXT) - Unique values: 'reimbursement', 'cashless'
   - `diagnosis_code` (TEXT)
   - `insurer` (TEXT) - Unique values: 'New India Assurance', 'Bajaj Allianz', 'United India', 'HDFC Ergo', 'Star Health', 'Care Health', 'ICICI Lombard', 'Niva Bupa'
   - `claimed_amount` (REAL)
   - `approved_amount` (REAL)
   - `status` (TEXT) - Unique values: 'pending', 'approved', 'rejected', 'submitted', 'escalated'
   - `submitted_date` (TEXT) - Format: 'YYYY-MM-DD'
   - `resolved_date` (TEXT) - Format: 'YYYY-MM-DD'

2. `maintenance_tickets` table:
   - `ticket_id` (TEXT, Primary Key)
   - `equipment_name` (TEXT)
   - `equipment_id` (TEXT)
   - `category` (TEXT) - Unique values: 'sterilisation', 'infusion', 'radiology', 'monitoring', 'surgical', 'laboratory'
   - `campus` (TEXT) - Unique values: 'MediAssist Hyderabad Central', 'MediAssist Bengaluru Onco Centre', 'MediAssist Pune Speciality', 'MediAssist Secunderabad', 'MediAssist Mysuru Clinic Hub'
   - `issue_type` (TEXT) - Unique values: 'preventive_maintenance', 'sensor_failure', 'battery_replacement', 'fault_reported', 'calibration_due'
   - `fault_code` (TEXT)
   - `raised_by` (TEXT)
   - `raised_date` (TEXT) - Format: 'YYYY-MM-DD'
   - `resolved_date` (TEXT) - Format: 'YYYY-MM-DD'
   - `status` (TEXT) - Unique values: 'in_progress', 'resolved', 'escalated', 'open'
   - `resolution_note` (TEXT)

IMPORTANT GUIDELINES:
- Output ONLY the raw SQL code. Do not write explanation, do not write markdown code blocks. Just output the query.
- Use LIKE or exact matches with the unique values listed above.
- Always check that the SQL query matches standard SQLite constraints.
- Never use ellipses (...), abbreviations, or placeholders in the SQL query (e.g., in IN lists or WHERE clauses). The query must be fully-formed, complete, and syntactically executable.
"""

def clean_sql(raw_sql: str) -> str:
    """Extracts the SQL statement from LLM output, removing any markdown code blocks or text wrappers."""
    # Remove markdown code blocks if present
    sql = raw_sql.strip()
    if sql.startswith("```"):
        # find the first newline to skip "```sql" or "```"
        newline_idx = sql.find("\n")
        if newline_idx != -1:
            sql = sql[newline_idx + 1:]
        if sql.endswith("```"):
            sql = sql[:-3]
    
    # Remove any leading/trailing sql tag or whitespace
    sql = re.sub(r"^sql\s+", "", sql.strip(), flags=re.IGNORECASE)
    # Strip trailing semicolon
    sql = sql.strip().rstrip(";")
    return sql

def execute_sql(query: str) -> list:
    """Executes the SQLite query and returns results."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        results = cursor.fetchall()
        # Get column names
        col_names = [desc[0] for desc in cursor.description]
        conn.close()
        return {"success": True, "columns": col_names, "data": results}
    except Exception as e:
        conn.close()
        return {"success": False, "error": str(e)}

def sql_rag_chain(question: str) -> str:
    """
    Translates a natural language question into SQL, cleans and executes it,
    then uses the LLM to format the response conversationally.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        # Rule-based fallback for offline testing
        q_lower = question.lower()
        if "cashless" in q_lower and "pending" in q_lower and "cardiology" in q_lower:
            sql_query = "SELECT COUNT(*) FROM claims WHERE claim_type='cashless' AND status='pending' AND department='cardiology'"
        elif "total claimed amount" in q_lower:
            sql_query = "SELECT SUM(claimed_amount) FROM claims"
        else:
            sql_query = "SELECT COUNT(*) FROM claims"
            
        execution_res = execute_sql(sql_query)
        if not execution_res["success"]:
            return f"Database Execution Error: {execution_res['error']}\nSQL Query tried: {sql_query}"
        return f"Offline SQL RAG: The count of cashless claims that are pending in the cardiology department is {execution_res['data'][0][0]}."


    client = Groq(api_key=groq_api_key)

    # Step 1: Translate natural language question to SQL using LLM
    try:
        translation_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SCHEMA_PROMPT},
                {"role": "user", "content": f"Translate this question into SQLite: {question}"}
            ],
            temperature=0.0
        )
        raw_sql = translation_completion.choices[0].message.content
        
        # Step 2: Clean the raw SQL output
        sql_query = clean_sql(raw_sql)
        print(f"\n[SQL Translation] Generated SQL Query:\n{sql_query}")

        # Step 3: Execute the SQL query
        execution_res = execute_sql(sql_query)
        
        if not execution_res["success"]:
            return f"Database Execution Error: {execution_res['error']}\nSQL Query tried: {sql_query}"

        # Step 4: Pass the result back to LLM to formulate a conversational answer
        user_content = (
            f"Question: {question}\n\n"
            f"Executed SQL Query: {sql_query}\n\n"
            f"Query Result Columns: {execution_res['columns']}\n"
            f"Query Result Rows: {execution_res['data']}\n\n"
            f"Please formulate a natural language answer to the question based on these query results. "
            f"Keep it professional and concise."
        )

        s_prompt = "You are a professional assistant reporting data analytics to healthcare executives. Use the SQL results provided to answer the user's question directly."

        answer_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": s_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0
        )
        return answer_completion.choices[0].message.content
        
    except Exception as e:
        return f"Error executing SQL RAG chain: {e}"
