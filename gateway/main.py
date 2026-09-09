import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
# Load .env from the repository root, wherever the repo happens to live.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from gateway.auth import verify_api_key
from gateway.rate_limiter import RateLimiter
from gateway.middleware import LoggingMiddleware, MetricsMiddleware
from gateway.router import GatewayRouter
from gateway.dashboard_route import router as dashboard_router
from gateway.landing_route import router as landing_router
from gateway.schemas import GenerateRequest, GenerateResponse, HealthResponse
from observability.database import init_db, get_db
from observability.metrics import metrics_registry
from sqlalchemy.orm import Session
import auth_routes
import user_routes
from auth_utils import get_current_user

rate_limiter = RateLimiter()
gateway_router = GatewayRouter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    # Create auth tables
    from observability.database import engine
    from auth_models import User, UserAPIKey, UserBudget
    from sqlalchemy import inspect
    inspector = inspect(engine)
    from observability.database import Base
    Base.metadata.create_all(bind=engine)
    print("AI Gateway started — database initialized")
    yield
    print("AI Gateway shutting down")


app = FastAPI(
    title="Enterprise AI Gateway",
    description="LLM Orchestrator + Cost Governance Layer",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(LoggingMiddleware)
app.add_middleware(MetricsMiddleware)

# Auth routes
app.include_router(auth_routes.router)
app.include_router(user_routes.router)

# Landing page (replaces dashboard as root)
app.include_router(landing_router)

# Dashboard at /app
@app.get("/app", response_class=HTMLResponse)
async def app_dashboard(request: Request):
    with open("gateway/dashboard.html", encoding="utf-8") as f:
        return f.read()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", version="2.0.0")


@app.get("/v1/cache/stats")
async def cache_stats():
    return gateway_router.prompt_dna.get_stats()


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    http_request: Request,
    api_key: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    from auth_utils import get_current_user
    from user_routes import get_user_keys
    team_id = http_request.headers.get("X-Team-ID", "default")
    allowed, reason = await rate_limiter.check(team_id)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {reason}")
    request_id = str(uuid.uuid4())
    start_time = time.time()
    try:
        user = get_current_user(http_request, db=db)
        if user:
            user_keys = get_user_keys(user.id, db)
            os.environ["GROQ_API_KEY"] = user_keys.get("groq", os.getenv("GROQ_API_KEY", ""))
            os.environ["GEMINI_API_KEY"] = user_keys.get("gemini", os.getenv("GEMINI_API_KEY", ""))
        result = await gateway_router.route(request=request, team_id=team_id, request_id=request_id, db=db)
        result.latency_ms = round((time.time() - start_time) * 1000, 2)
        result.request_id = request_id
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/stream")
async def stream(
    request: GenerateRequest,
    http_request: Request,
    api_key: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    from fastapi.responses import StreamingResponse
    import json
    import asyncio
    team_id = http_request.headers.get("X-Team-ID", "default")
    request_id = str(uuid.uuid4())

    async def event_stream():
        try:
            compressed_prompt, compression_stats = gateway_router.compressor.compress(request.prompt)
            request.prompt = compressed_prompt

            cache_hit, cached_entry = gateway_router.prompt_dna.lookup(request.prompt)
            if cache_hit:
                meta = json.dumps({"type":"meta","model":cached_entry["model"]+" (cached)","provider":"prompt_dna_cache","cost":0.0,"tokens":0,"routing_reason":f"DNA cache hit — similarity {cached_entry['similarity_score']}","compression_applied":compression_stats["compressed"],"original_tokens":compression_stats["original_tokens"],"compressed_tokens":compression_stats["compressed_tokens"],"spd_similarity":None})
                yield f"data: {meta}\n\n"
                words = cached_entry["response"].split()
                for i, word in enumerate(words):
                    chunk = json.dumps({"type":"token","text":word+(" " if i<len(words)-1 else "")})
                    yield f"data: {chunk}\n\n"
                yield f'data: {json.dumps({"type":"done"})}\n\n'
                return

            spd_similarity = None
            selected_model = "llama-3.1-8b-instant"
            provider = "groq"
            routing_reason = "Direct routing"
            full_content = ""
            tokens_used = 0
            actual_cost = 0.0

            if gateway_router.arbiter._available and request.strategy != "performance":
                cheap_models = [("llama-3.1-8b-instant","groq"),("gemini-2.5-flash","gemini")]
                results = await asyncio.gather(*[gateway_router._execute_model(m,p,request) for m,p in cheap_models], return_exceptions=True)
                responses = []
                for i, result in enumerate(results):
                    if not isinstance(result, Exception):
                        c,t,cost = result
                        responses.append({"model":cheap_models[i][0],"provider":cheap_models[i][1],"content":c,"tokens":t,"cost":cost})
                if len(responses) >= 2:
                    agreed, best, similarity = gateway_router.arbiter.check_agreement(responses)
                    spd_similarity = similarity
                    if agreed:
                        selected_model,provider,full_content,tokens_used,actual_cost = best["model"],best["provider"],best["content"],best["tokens"],best["cost"]
                        routing_reason = f"SPD consensus (similarity={similarity}) — {best['model']} selected"
                    else:
                        full_content,tokens_used,actual_cost = await gateway_router._execute_model("llama-3.3-70b-versatile","groq",request)
                        selected_model,provider = "llama-3.3-70b-versatile","groq"
                        routing_reason = f"SPD no consensus (similarity={similarity}) — escalated to Llama 70B"
                elif len(responses) == 1:
                    selected_model,provider,full_content,tokens_used,actual_cost = responses[0]["model"],responses[0]["provider"],responses[0]["content"],responses[0]["tokens"],responses[0]["cost"]
                else:
                    full_content,tokens_used,actual_cost = await gateway_router._execute_model(selected_model,provider,request)
            else:
                full_content,tokens_used,actual_cost = await gateway_router._execute_model(selected_model,provider,request)

            compression_cost_saved = round((compression_stats["original_tokens"]-compression_stats["compressed_tokens"])*0.0000025,8)
            gateway_router.prompt_dna.store(prompt=request.prompt,response=full_content,model=selected_model,query_type="unknown",cost=actual_cost,tokens=tokens_used)

            meta = json.dumps({"type":"meta","model":selected_model,"provider":provider,"cost":round(actual_cost,6),"tokens":tokens_used,"routing_reason":routing_reason,"compression_applied":compression_stats["compressed"],"original_tokens":compression_stats["original_tokens"],"compressed_tokens":compression_stats["compressed_tokens"],"compression_savings":compression_cost_saved,"spd_similarity":spd_similarity})
            yield f"data: {meta}\n\n"

            words = full_content.split(" ")
            for i, word in enumerate(words):
                chunk = json.dumps({"type":"token","text":word+(" " if i<len(words)-1 else "")})
                yield f"data: {chunk}\n\n"
                await asyncio.sleep(0.015)

            yield f'data: {json.dumps({"type":"done"})}\n\n'

        except Exception as e:
            yield f'data: {json.dumps({"type":"error","message":str(e)})}\n\n'

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


@app.get("/v1/metrics")
async def get_metrics(db: Session = Depends(get_db)):
    from observability.analytics import get_summary_metrics
    return await get_summary_metrics(db)


@app.get("/v1/models")
async def list_models():
    return {"models": [
        {"id":"llama-3.1-8b-instant","provider":"groq","tier":"fast","cost_per_1k_tokens":0.00005},
        {"id":"llama-3.3-70b-versatile","provider":"groq","tier":"quality","cost_per_1k_tokens":0.00059},
        {"id":"gemini-2.5-flash","provider":"gemini","tier":"balanced","cost_per_1k_tokens":0.0002},
    ]}
