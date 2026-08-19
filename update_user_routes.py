code = '''import os
import uuid
import base64
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from observability.database import get_db
from auth_models import UserAPIKey, UserBudget
from auth_utils import require_user, get_current_user
from auth_models import User

router = APIRouter(prefix="/user", tags=["user"])


def encrypt_key(key: str) -> str:
    return base64.b64encode(key.encode()).decode()


def decrypt_key(encrypted: str) -> str:
    return base64.b64decode(encrypted.encode()).decode()


def get_user_keys(user_id: str, db: Session) -> dict:
    """Get decrypted API keys for a user. Falls back to .env keys."""
    keys = db.query(UserAPIKey).filter(UserAPIKey.user_id == user_id).all()
    result = {
        "groq": os.getenv("GROQ_API_KEY", ""),
        "gemini": os.getenv("GEMINI_API_KEY", ""),
    }
    for k in keys:
        result[k.provider] = decrypt_key(k.encrypted_key)
    return result


class AddKeyRequest(BaseModel):
    provider: str
    api_key: str


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not user:
        return RedirectResponse("/login")
    with open("gateway/settings.html", encoding="utf-8") as f:
        return f.read()


@router.get("/keys")
async def get_keys(user: User = Depends(require_user), db: Session = Depends(get_db)):
    keys = db.query(UserAPIKey).filter(UserAPIKey.user_id == user.id).all()
    return [{"id": k.id, "provider": k.provider, "key_preview": "••••••••" + decrypt_key(k.encrypted_key)[-4:]} for k in keys]


@router.post("/keys")
async def add_key(data: AddKeyRequest, user: User = Depends(require_user), db: Session = Depends(get_db)):
    existing = db.query(UserAPIKey).filter(
        UserAPIKey.user_id == user.id,
        UserAPIKey.provider == data.provider
    ).first()
    if existing:
        existing.encrypted_key = encrypt_key(data.api_key)
        db.commit()
        return {"message": f"{data.provider} key updated"}
    key = UserAPIKey(
        id=str(uuid.uuid4()),
        user_id=user.id,
        provider=data.provider,
        encrypted_key=encrypt_key(data.api_key)
    )
    db.add(key)
    db.commit()
    return {"message": f"{data.provider} key added"}


@router.delete("/keys/{key_id}")
async def delete_key(key_id: str, user: User = Depends(require_user), db: Session = Depends(get_db)):
    key = db.query(UserAPIKey).filter(UserAPIKey.id == key_id, UserAPIKey.user_id == user.id).first()
    if not key:
        raise HTTPException(status_code=404, detail="Key not found")
    db.delete(key)
    db.commit()
    return {"message": "Key deleted"}


@router.get("/budget")
async def get_budget(user: User = Depends(require_user), db: Session = Depends(get_db)):
    budget = db.query(UserBudget).filter(UserBudget.user_id == user.id).first()
    if not budget:
        return {"daily_limit_usd": 10.0, "monthly_limit_usd": 100.0, "daily_spent_usd": 0.0, "monthly_spent_usd": 0.0}
    return {"daily_limit_usd": budget.daily_limit_usd, "monthly_limit_usd": budget.monthly_limit_usd, "daily_spent_usd": budget.daily_spent_usd, "monthly_spent_usd": budget.monthly_spent_usd}


@router.post("/budget")
async def update_budget(data: dict, user: User = Depends(require_user), db: Session = Depends(get_db)):
    budget = db.query(UserBudget).filter(UserBudget.user_id == user.id).first()
    if not budget:
        budget = UserBudget(id=str(uuid.uuid4()), user_id=user.id)
        db.add(budget)
    if "daily_limit_usd" in data:
        budget.daily_limit_usd = data["daily_limit_usd"]
    if "monthly_limit_usd" in data:
        budget.monthly_limit_usd = data["monthly_limit_usd"]
    db.commit()
    return {"message": "Budget updated"}
'''

with open('user_routes.py', 'w', encoding='utf-8') as f:
    f.write(code)
print('Done')