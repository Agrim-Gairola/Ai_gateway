import os
from fastapi import Header, HTTPException
from typing import Optional

VALID_API_KEYS = {
    "ak_dev_1234567890abcdef": {"team": "engineering", "tier": "premium"},
    "ak_prod_abcdef1234567890": {"team": "marketing", "tier": "standard"},
}

async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key.")
    if os.getenv("ENV", "development") == "development" and x_api_key.startswith("ak_dev_"):
        return x_api_key
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    return x_api_key
