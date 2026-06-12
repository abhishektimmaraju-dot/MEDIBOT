import jwt
import datetime
from typing import Dict, Any, Optional

# Secret key to sign JWT tokens (for demo purposes, keeping it static)
JWT_SECRET = "mediassist-super-secret-key-12345!"
JWT_ALGORITHM = "HS256"

# Demo user accounts and their roles
DEMO_USERS = {
    "dr.mehta": {"password": "password", "role": "doctor", "name": "Dr. Mehta"},
    "nurse.priya": {"password": "password", "role": "nurse", "name": "Nurse Priya"},
    "billing.ravi": {"password": "password", "role": "billing_executive", "name": "Billing Exec Ravi"},
    "tech.anand": {"password": "password", "role": "technician", "name": "Technician Anand"},
    "admin.sys": {"password": "password", "role": "admin", "name": "Admin Sys"}
}

ROLE_COLLECTIONS = {
    "doctor": ["clinical", "nursing", "general"],
    "nurse": ["nursing", "general"],
    "billing_executive": ["billing", "general"],
    "technician": ["equipment", "general"],
    "admin": ["general", "clinical", "nursing", "billing", "equipment"]
}

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticates username and password against demo list."""
    user = DEMO_USERS.get(username.lower())
    if user and user["password"] == password:
        return {
            "username": username,
            "role": user["role"],
            "name": user["name"]
        }
    return None

def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None) -> str:
    """Generates a signed JWT token containing user metadata."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(hours=12)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decodes and validates a signed JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None
