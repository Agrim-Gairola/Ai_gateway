from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def landing():
    with open("gateway/landing.html", encoding="utf-8") as f:
        return f.read()


@router.get("/login", response_class=HTMLResponse)
async def login_page():
    with open("gateway/login.html", encoding="utf-8") as f:
        return f.read()


@router.get("/signup", response_class=HTMLResponse)
async def signup_page():
    with open("gateway/login.html", encoding="utf-8") as f:
        return f.read()


@router.get("/reset-password", response_class=HTMLResponse)
async def reset_password_page():
    with open("gateway/login.html", encoding="utf-8") as f:
        return f.read()