from fastapi import APIRouter
from fastapi.responses import HTMLResponse
import os

router = APIRouter()

HTML_PATH = os.path.join(os.path.dirname(__file__), 'dashboard.html')

@router.get("/", response_class=HTMLResponse)
async def dashboard():
    with open(HTML_PATH, encoding='utf-8') as f:
        return f.read()
