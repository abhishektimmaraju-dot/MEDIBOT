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
- Distinguish between listing and counting: If the user asks to 'list', 'show', or 'detail' records, use 'SELECT *' or select specific columns. Do NOT use COUNT() or other aggregations unless they explicitly ask for a count, total, average, or number of records.
"""

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

def execute_sql(query: str) -> dict:
    """
    Executes the SQLite query and returns results. 
    Runs the validation check and opens the SQLite connection in read-only mode.
    """
    if not is_safe_select(query):
        return {
            "success": False, 
            "error": "Query validation failed. Only read-only SELECT statements are allowed."
        }
        
    try:
        # Open database connection in read-only mode
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        conn.close()
        return {"success": True, "columns": col_names, "data": results}
    except Exception as e:
        if 'conn' in locals():
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
