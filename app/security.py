from hmac import compare_digest

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from app.config import settings


api_key_header = APIKeyHeader(
    name="X-API-Key",
    scheme_name="APIKey",
    description="API key required to access protected endpoints",
    auto_error=False
)


def verify_api_key(
    api_key: str | None = Depends(api_key_header)
):
    if not api_key or not compare_digest(api_key, settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "APIKey"}
        )

    return api_key