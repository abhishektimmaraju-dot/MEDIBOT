"""
MediBot Configuration — Central Settings Module

All application constants, file paths, model identifiers, and threshold values
are defined here so every other module draws from a single source of truth.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
DATA_DIR = os.path.join(SCRIPT_DIR, "mediassist_data")
QDRANT_PATH = os.path.join(SCRIPT_DIR, "mediassist_data", "qdrant_db")
DB_PATH = os.path.join(SCRIPT_DIR, "mediassist_data", "db", "mediassist.db")

# ── Vector DB ────────────────────────────────────────────────────────────────
COLLECTION_NAME = "medibot"

# ── Embedding Models ─────────────────────────────────────────────────────────
DENSE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DENSE_EMBEDDING_DIM = 384
SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── JWT / Auth ───────────────────────────────────────────────────────────────
JWT_SECRET = os.getenv("JWT_SECRET", "mediassist-super-secret-key-12345!")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 12

# ── Retrieval Thresholds ─────────────────────────────────────────────────────
# Cross-encoder scores from ms-marco-MiniLM-L-6-v2:
#   Valid in-scope queries:    rank-1 scores between -1.0 and 4.0
#   Out-of-scope/irrelevant:   rank-1 scores below -10.0
#   Restricted (RBAC-blocked): rank-1 scores below -8.0
# The threshold of -6.0 sits safely between valid content and noise.
RERANKER_SCORE_THRESHOLD = -6.0
HYBRID_SEARCH_LIMIT = 10
RERANKER_TOP_K = 3

# ── Semantic Router ──────────────────────────────────────────────────────────
# Cosine similarity threshold for semantic route classification.
# Above this value, the router is confident enough to classify without LLM.
SEMANTIC_ROUTER_THRESHOLD = 0.45

# ── RBAC Access Matrix ───────────────────────────────────────────────────────
# Shared between ingest.py and runtime — single source of truth.
ACCESS_MATRIX = {
    "general": {
        "access_roles": ["doctor", "nurse", "billing_executive", "technician", "admin"],
    },
    "clinical": {
        "access_roles": ["doctor", "admin"],
    },
    "nursing": {
        "access_roles": ["nurse", "doctor", "admin"],
    },
    "billing": {
        "access_roles": ["billing_executive", "admin"],
    },
    "equipment": {
        "access_roles": ["technician", "admin"],
    }
}

ROLE_COLLECTIONS = {
    "doctor": ["clinical", "nursing", "general"],
    "nurse": ["nursing", "general"],
    "billing_executive": ["billing", "general"],
    "technician": ["equipment", "general"],
    "admin": ["general", "clinical", "nursing", "billing", "equipment"]
}

# ── Demo Users ───────────────────────────────────────────────────────────────
# NOTE: In production, passwords must be securely hashed (e.g. bcrypt).
# Plaintext is only for demo convenience.
DEMO_USERS = {
    "dr.mehta": {"password": "password", "role": "doctor", "name": "Dr. Mehta"},
    "nurse.priya": {"password": "password", "role": "nurse", "name": "Nurse Priya"},
    "billing.ravi": {"password": "password", "role": "billing_executive", "name": "Billing Exec Ravi"},
    "tech.anand": {"password": "password", "role": "technician", "name": "Technician Anand"},
    "admin.sys": {"password": "password", "role": "admin", "name": "Admin Sys"}
}

# ── SQL RAG Schema Prompt ────────────────────────────────────────────────────
SQL_SCHEMA_PROMPT = """You are an expert SQL Translator for SQLite.
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

# ── Collection Keywords (for RBAC query restriction heuristic) ───────────────
COLLECTION_KEYWORDS = {
    "clinical": ["clinical", "treatment", "protocol", "drug", "dosage", "medical", "guideline", "patient care", "nstemi", "cardiac", "heart", "physician", "doctor", "diagnosis", "therapy", "cardiologist", "medication", "prescrib", "remedy"],
    "nursing": ["nurse", "nursing", "icu", "ward", "infection control", "hygiene", "shift", "patient hygiene", "handwash", "dressings", "wound", "sterile"],
    "billing": ["billing", "claim", "reimbursement", "cashless", "insurance", "insurer", "invoice", "finance", "sla", "copay", "co-pay", "policy limit", "pre-auth", "pre-authorisation", "pre-authorization", "star health", "hdfc", "bajaj", "new india", "settlement", "turnaround", "tpa", "pricing", "deductible", "tariff", "rates", "fee", "charge"],
    "equipment": ["equipment", "manual", "calibration", "infusion pump", "defibrillator", "radiology", "ventilator", "maintenance", "sensor", "battery", "pump", "fault code", "calibration steps", "device", "hardware", "malfunction", "fault"]
}
