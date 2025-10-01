"""Authentication and rate limiting for the web API."""

from __future__ import annotations

import os
from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

# API key authentication
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def get_api_key() -> Optional[str]:
    """Get API key from environment.

    Returns
    -------
    Optional[str]
        API key if set, None otherwise
    """
    return os.getenv("RAG_API_KEY")


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Verify API key from request header.

    Parameters
    ----------
    api_key : str
        API key from request header

    Returns
    -------
    str
        Validated API key

    Raises
    ------
    HTTPException
        If API key is missing or invalid
    """
    expected_key = get_api_key()

    # If no API key is configured, authentication is disabled
    if not expected_key:
        return "auth_disabled"

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is required. Set X-API-Key header.",
        )

    if api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return api_key


def is_auth_enabled() -> bool:
    """Check if authentication is enabled.

    Returns
    -------
    bool
        True if RAG_API_KEY environment variable is set
    """
    return get_api_key() is not None


__all__ = ["verify_api_key", "is_auth_enabled", "API_KEY_NAME"]
