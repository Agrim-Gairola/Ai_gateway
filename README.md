# ⚡ Enterprise AI Gateway

> **LLM Orchestrator + Cost Governance Layer**
> An AWS API Gateway — but for AI models. Routes requests intelligently, enforces budgets, tracks every token.

---

## 🏗 Architecture

```
Client Request
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│                    GATEWAY LAYER                        │
│  FastAPI  ·  Auth (API Keys)  ·  Rate Limiter           │
│  Logging Middleware  ·  Metrics Middleware               │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                 INTELLIGENCE LAYER                      │
│  Query Classifier → Complexity Scorer → Cost Estimator  │
│                  → Routing Engine                       │
└─────────────────────┬───────────────────────────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
      OpenAI      Anthropic     Mock
    (GPT-4o)    (Claude)      (Dev)
          │           │           │
          └───────────┼───────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│               OBSERVABILITY LAYER                       │
│  PostgreSQL (logs)  ·  Prometheus (metrics)             │
│  Policy Engine (budgets)  ·  Analytics                  │
└─────────────────────────────────────────────────────────┘
                      │
                      ▼
              Streamlit Dashboard
```

---

## 🚀 Quick Start (5 minutes)

### 1. Clone & configure
```bash
git clone https://github.com/YOUR_USERNAME/ai-gateway.git
cd ai-gateway
cp .env.example .env
# Edit .env — set OPENAI_API_KEY or leave as "mock" for dev mode
```

### 2. Run with Docker Compose (recommended)
```bash
docker compose up --build
```

Services started:
| Service | URL |
|---|---|
| Gateway API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

### 3. Run locally (dev)
```bash
pip install -r requirements.txt
uvicorn gateway.main:app --reload
streamlit run dashboard/app.py  # separate terminal
```

---

## 📡 API Usage

### Send a request
```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ak_dev_1234567890abcdef" \
  -H "X-Team-ID: engineering" \
  -d '{
    "prompt": "Write a Python function to parse nested JSON",
    "max_tokens": 500,
    "strategy": "balanced"
  }'
```

### Response
```json
{
  "request_id": "a1b2c3d4-...",
  "content": "Here is a clean implementation...",
  "model_used": "gpt-4o",
  "provider": "openai",
  "query_type": "coding",
  "complexity_score": 0.72,
  "tokens_used": 387,
  "estimated_cost_usd": 0.00193,
  "latency_ms": 842.3,
  "routing_reason": "Complex coding task → GPT-4o",
  "fallback_used": false
}
```

### Routing strategies
| Strategy | Behavior |
|---|---|
| `balanced` | Smart routing by query complexity (default) |
| `cost_optimized` | Always use cheapest model |
| `performance` | Always use best model |

### Other endpoints
```bash
GET  /health          # Health check
GET  /v1/models       # List available models
GET  /v1/metrics      # Aggregated analytics
GET  /v1/usage/{team} # Per-team usage history
```

---

## 🧠 How Routing Works

```
prompt → Classifier → query_type (coding/summarization/reasoning/creative/simple)
                 ↓
       Complexity Scorer → 0.0 ... 1.0
                 ↓
       Routing Engine (strategy × complexity × type)
                 ↓
       Policy Engine (budget check)
                 ↓
       Provider Adapter → API call
                 ↓
       DB log + response
```

**Example routing decisions (balanced strategy):**
```
coding + complexity 0.85  →  gpt-4o          ($0.005/1k)
coding + complexity 0.50  →  claude-sonnet   ($0.003/1k)
coding + complexity 0.20  →  gpt-4o-mini     ($0.00015/1k)
simple + any complexity   →  gpt-4o-mini     ($0.00015/1k)
```

**Cost savings:** A naive "always GPT-4o" approach costs ~33x more than smart routing.

---

## ⚙️ Policy Engine

Configure per-team budgets in `observability/policy_engine.py`:

```python
DEFAULT_BUDGETS = {
    "engineering": {"daily": 50.0,  "monthly": 1000.0},
    "marketing":   {"daily": 10.0,  "monthly": 200.0},
    "qa":          {"daily": 5.0,   "monthly": 100.0},
}
```

Enforcement happens automatically before every request:
- Daily budget exceeded → 403 Forbidden
- Monthly budget exceeded → 403 Forbidden
- Team blocked → 403 Forbidden
- Premium model not allowed → routes to allowed tier

---

## 🧪 Running Tests

```bash
pytest tests/ -v

# Expected output:
# PASSED test_coding_detection
# PASSED test_complex_coding_high_score
# PASSED test_gpt4o_more_expensive_than_mini
# PASSED test_cost_optimized_always_cheap
# PASSED test_balanced_complex_coding_gets_strong_model
# ... 15 tests total
```

---

## 🏗 Adding a New Provider

```python
# models/provider_registry.py

class MistralProvider(BaseProvider):
    async def generate(self, model, prompt, max_tokens, temperature):
        # Your implementation
        return content, tokens_used, cost_usd

# Register it:
registry.register("mistral", MistralProvider())
```

Then add routing rules in `intelligence/routing_engine.py`.

---

## ☁️ Deploy to AWS

```bash
cd infra/
terraform init
terraform plan -var="key_name=your-key" \
               -var="openai_api_key=$OPENAI_API_KEY"
terraform apply
# → gateway_url = http://1.2.3.4:8000
# → dashboard_url = http://1.2.3.4:8501
```

---

## 📁 Project Structure

```
ai-gateway/
├── gateway/
│   ├── main.py            # FastAPI app + endpoints
│   ├── auth.py            # API key authentication
│   ├── rate_limiter.py    # Token bucket rate limiting
│   ├── middleware.py      # Logging + metrics middleware
│   ├── router.py          # Core orchestration pipeline
│   └── schemas.py         # Pydantic request/response models
├── intelligence/
│   ├── classifier.py      # Query type classification
│   ├── complexity.py      # Complexity scoring (0.0–1.0)
│   ├── cost_estimator.py  # Pre-flight cost prediction
│   └── routing_engine.py  # Model selection logic
├── models/
│   └── provider_registry.py  # OpenAI + Anthropic + Mock adapters
├── observability/
│   ├── database.py        # SQLAlchemy models + session
│   ├── policy_engine.py   # Budget enforcement
│   ├── analytics.py       # Dashboard data queries
│   └── metrics.py         # Prometheus metrics
├── dashboard/
│   └── app.py             # Streamlit dashboard
├── tests/
│   └── test_core.py       # Unit tests
├── infra/
│   ├── main.tf            # Terraform AWS config
│   └── prometheus.yml     # Prometheus scrape config
├── .github/workflows/
│   └── ci.yml             # GitHub Actions CI/CD
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| API Gateway | FastAPI + Uvicorn |
| Database | PostgreSQL (prod) / SQLite (dev) |
| ORM | SQLAlchemy |
| Cache | Redis |
| AI Providers | OpenAI API, Anthropic API |
| Dashboard | Streamlit + Plotly |
| Monitoring | Prometheus + Grafana |
| Container | Docker + Docker Compose |
| Cloud | AWS EC2 + Terraform |
| CI/CD | GitHub Actions |

---

## 💡 What Makes This "Enterprise"

1. **Multi-provider routing** — not locked to one model
2. **Cost governance** — budget caps enforced per team, per day/month
3. **Audit trail** — every request logged with tokens, cost, latency, model
4. **Policy engine** — block teams, restrict premium models, enforce limits
5. **Fallback logic** — automatic failover when primary model errors
6. **Observable** — Prometheus metrics, Grafana dashboards, structured logs
7. **Deployable** — Dockerized, Terraform-ready, CI/CD pipeline included
