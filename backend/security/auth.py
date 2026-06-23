"""
MediBot Authentication & Authorization — JWT token management and user verification.

Handles:
  - Demo user authentication (plaintext for demo purposes)
  - JWT token creation and validation
  - Role-based collection mappings
"""
import datetime
import jwt
from typing import Dict, Any, Optional

from config.settings import (
    JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRY_HOURS,
    DEMO_USERS, ROLE_COLLECTIONS
)
from utils.logger import get_logger

logger = get_logger("security.auth")


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Authenticates username and password against demo list.
    NOTE: In production, passwords must be securely hashed and verified (e.g. using bcrypt).
    Plaintext verification is only for demo convenience.
    """
    user = DEMO_USERS.get(username.lower())
    if user and user["password"] == password:
        logger.info(f"User '{username}' authenticated successfully as '{user['role']}'")
        return {
            "username": username,
            "role": user["role"],
            "name": user["name"]
        }
    logger.warning(f"Failed authentication attempt for username '{username}'")
    return None


def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None) -> str:
    """Generates a signed JWT token containing user metadata."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(hours=JWT_EXPIRY_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decodes and validates a signed JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.PyJWTError:
        logger.warning("Failed to decode JWT token — expired or malformed")
        return None
