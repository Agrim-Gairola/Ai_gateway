f = open('gateway/main.py', encoding='utf-8')
content = f.read()
f.close()

old = '''@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    http_request: Request,
    api_key: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    team_id = http_request.headers.get("X-Team-ID", "default")
    allowed, reason = await rate_limiter.check(team_id)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {reason}")
    request_id = str(uuid.uuid4())
    start_time = time.time()
    try:
        result = await gateway_router.route(request=request, team_id=team_id, request_id=request_id, db=db)
        result.latency_ms = round((time.time() - start_time) * 1000, 2)
        result.request_id = request_id
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))'''

new = '''@app.post("/v1/generate", response_model=GenerateResponse)
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
        raise HTTPException(status_code=500, detail=str(e))'''

if old in content:
    f = open('gateway/main.py', 'w', encoding='utf-8')
    f.write(content.replace(old, new, 1))
    f.close()
    print('Done')
else:
    print('Not found')
    i = content.find('/v1/generate')
    print(repr(content[i:i+100]))