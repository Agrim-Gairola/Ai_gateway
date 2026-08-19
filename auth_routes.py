import os
import uuid
import httpx
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Response, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from observability.database import get_db
from auth_models import User, UserBudget
from auth_utils import hash_password, verify_password, create_access_token, get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:8000")


class SignupRequest(BaseModel):
    email: str
    password: str
    full_name: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


def create_user_with_budget(db: Session, user: User):
    db.add(user)
    db.flush()
    budget = UserBudget(user_id=user.id)
    db.add(budget)
    db.commit()
    db.refresh(user)
    return user


@router.post("/signup")
async def signup(data: SignupRequest, request: Request, response: Response, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    from email_service import generate_token, send_verification_email
    verification_token = generate_token()
    user = User(
        id=str(uuid.uuid4()),
        email=data.email,
        hashed_password=hash_password(data.password),
        full_name=data.full_name,
        is_verified=False,
        verification_token=verification_token,
    )
    user = create_user_with_budget(db, user)
    base_url = str(request.base_url).rstrip("/")
    await send_verification_email(data.email, verification_token, base_url)
    token = create_access_token({"sub": user.id, "email": user.email})
    response.set_cookie("access_token", token, httponly=True, max_age=604800, samesite="lax")
    return {"token": token, "user": {"id": user.id, "email": user.email, "name": user.full_name}, "message": "Account created. Please check your email to verify your account."}


@router.post("/login")
async def login(data: LoginRequest, response: Response, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user or not user.hashed_password:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token({"sub": user.id, "email": user.email})
    response.set_cookie("access_token", token, httponly=True, max_age=604800, samesite="lax")
    return {"token": token, "user": {"id": user.id, "email": user.email, "name": user.full_name, "verified": user.is_verified}}


@router.get("/verify-email")
async def verify_email(token: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.verification_token == token).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired verification link")
    user.is_verified = True
    user.verification_token = None
    db.commit()
    return RedirectResponse("/app?verified=1")


@router.post("/resend-verification")
async def resend_verification(request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if user.is_verified:
        return {"message": "Email already verified"}
    from email_service import generate_token, send_verification_email
    token = generate_token()
    user.verification_token = token
    db.commit()
    base_url = str(request.base_url).rstrip("/")
    await send_verification_email(user.email, token, base_url)
    return {"message": "Verification email sent"}


@router.post("/forgot-password")
async def forgot_password(data: ForgotPasswordRequest, request: Request, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user:
        return {"message": "If that email exists, a reset link has been sent"}
    from email_service import generate_token, send_password_reset_email
    token = generate_token()
    user.reset_token = token
    user.reset_token_expires = datetime.utcnow() + timedelta(hours=1)
    db.commit()
    base_url = str(request.base_url).rstrip("/")
    await send_password_reset_email(user.email, token, base_url)
    return {"message": "If that email exists, a reset link has been sent"}


@router.post("/reset-password")
async def reset_password(data: ResetPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.reset_token == data.token).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired reset link")
    if user.reset_token_expires and datetime.utcnow() > user.reset_token_expires:
        raise HTTPException(status_code=400, detail="Reset link has expired")
    if len(data.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    user.hashed_password = hash_password(data.new_password)
    user.reset_token = None
    user.reset_token_expires = None
    db.commit()
    return {"message": "Password reset successfully"}


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie("access_token")
    return {"message": "Logged out"}


@router.get("/me")
async def me(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"id": user.id, "email": user.email, "name": user.full_name, "avatar": user.avatar_url, "verified": user.is_verified}


@router.get("/google")
async def google_login():
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Google OAuth not configured")
    from urllib.parse import urlencode
    params = {"client_id": GOOGLE_CLIENT_ID, "redirect_uri": f"{FRONTEND_URL}/auth/google/callback", "response_type": "code", "scope": "openid email profile", "access_type": "offline"}
    return RedirectResponse("https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params))


@router.get("/google/callback")
async def google_callback(code: str, response: Response, db: Session = Depends(get_db)):
    async with httpx.AsyncClient(timeout=30.0) as client:
        token_res = await client.post("https://oauth2.googleapis.com/token", data={"code": code, "client_id": GOOGLE_CLIENT_ID, "client_secret": GOOGLE_CLIENT_SECRET, "redirect_uri": f"{FRONTEND_URL}/auth/google/callback", "grant_type": "authorization_code"})
        tokens = token_res.json()
        userinfo_res = await client.get("https://www.googleapis.com/oauth2/v3/userinfo", headers={"Authorization": f"Bearer {tokens['access_token']}"})
        userinfo = userinfo_res.json()
    email = userinfo.get("email")
    google_id = userinfo.get("sub")
    name = userinfo.get("name", "")
    avatar = userinfo.get("picture", "")
    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(id=str(uuid.uuid4()), email=email, google_id=google_id, full_name=name, avatar_url=avatar, is_verified=True)
        create_user_with_budget(db, user)
    else:
        user.google_id = google_id
        user.avatar_url = avatar
        db.commit()
    token = create_access_token({"sub": user.id, "email": user.email})
    resp = RedirectResponse(url="/app")
    resp.set_cookie("access_token", token, httponly=True, max_age=604800, samesite="lax")
    return resp
