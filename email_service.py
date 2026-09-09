import os
import secrets
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the repository root, wherever the repo happens to live.
load_dotenv(Path(__file__).resolve().parent / ".env")


def generate_token() -> str:
    return secrets.token_urlsafe(32)


async def send_verification_email(email: str, token: str, base_url: str = "http://localhost:8000"):
    try:
        from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
        conf = ConnectionConfig(
            MAIL_USERNAME=os.getenv("MAIL_USERNAME", ""),
            MAIL_PASSWORD=os.getenv("MAIL_PASSWORD", ""),
            MAIL_FROM=os.getenv("MAIL_FROM", ""),
            MAIL_PORT=int(os.getenv("MAIL_PORT", 587)),
            MAIL_SERVER=os.getenv("MAIL_SERVER", "smtp.gmail.com"),
            MAIL_STARTTLS=True,
            MAIL_SSL_TLS=False,
            USE_CREDENTIALS=True,
        )
        verify_url = f"{base_url}/auth/verify-email?token={token}"
        html = f"""
        <div style="font-family:Inter,sans-serif;max-width:480px;margin:0 auto;padding:40px 20px;background:#0a0a0f;color:#e2e2f0">
            <div style="text-align:center;margin-bottom:32px">
                <div style="width:48px;height:48px;background:#7c6af7;border-radius:12px;display:inline-flex;align-items:center;justify-content:center;font-weight:700;font-size:16px;margin-bottom:16px">AI</div>
                <h1 style="font-size:24px;font-weight:700;margin:0">Verify your email</h1>
            </div>
            <p style="color:#6b6b8a;margin-bottom:32px;line-height:1.6">Click the button below to verify your email address and activate your Enterprise AI Gateway account.</p>
            <a href="{verify_url}" style="display:block;text-align:center;padding:14px 32px;background:#7c6af7;color:white;text-decoration:none;border-radius:8px;font-weight:600;font-size:16px">Verify Email</a>
            <p style="color:#6b6b8a;font-size:12px;margin-top:32px;text-align:center">This link expires in 24 hours. If you did not create an account, ignore this email.</p>
        </div>
        """
        message = MessageSchema(subject="Verify your email — AI Gateway", recipients=[email], body=html, subtype=MessageType.html)
        fm = FastMail(conf)
        await fm.send_message(message)
        return True
    except Exception as e:
        print(f"Email send failed: {e}")
        return False


async def send_password_reset_email(email: str, token: str, base_url: str = "http://localhost:8000"):
    try:
        from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
        conf = ConnectionConfig(
            MAIL_USERNAME=os.getenv("MAIL_USERNAME", ""),
            MAIL_PASSWORD=os.getenv("MAIL_PASSWORD", ""),
            MAIL_FROM=os.getenv("MAIL_FROM", ""),
            MAIL_PORT=int(os.getenv("MAIL_PORT", 587)),
            MAIL_SERVER=os.getenv("MAIL_SERVER", "smtp.gmail.com"),
            MAIL_STARTTLS=True,
            MAIL_SSL_TLS=False,
            USE_CREDENTIALS=True,
        )
        reset_url = f"{base_url}/reset-password?token={token}"
        html = f"""
        <div style="font-family:Inter,sans-serif;max-width:480px;margin:0 auto;padding:40px 20px;background:#0a0a0f;color:#e2e2f0">
            <div style="text-align:center;margin-bottom:32px">
                <div style="width:48px;height:48px;background:#7c6af7;border-radius:12px;display:inline-flex;align-items:center;justify-content:center;font-weight:700;font-size:16px;margin-bottom:16px">AI</div>
                <h1 style="font-size:24px;font-weight:700;margin:0">Reset your password</h1>
            </div>
            <p style="color:#6b6b8a;margin-bottom:32px;line-height:1.6">Click the button below to reset your password. This link expires in 1 hour.</p>
            <a href="{reset_url}" style="display:block;text-align:center;padding:14px 32px;background:#7c6af7;color:white;text-decoration:none;border-radius:8px;font-weight:600;font-size:16px">Reset Password</a>
            <p style="color:#6b6b8a;font-size:12px;margin-top:32px;text-align:center">If you did not request a password reset, ignore this email.</p>
        </div>
        """
        message = MessageSchema(subject="Reset your password — AI Gateway", recipients=[email], body=html, subtype=MessageType.html)
        fm = FastMail(conf)
        await fm.send_message(message)
        return True
    except Exception as e:
        print(f"Email send failed: {e}")
        return False