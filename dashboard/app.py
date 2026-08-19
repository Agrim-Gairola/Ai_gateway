"""
Enterprise AI Gateway — Analytics Dashboard
Streamlit app showing real-time usage, costs, latency, and forecasts.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
import os
from datetime import datetime

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")
API_KEY = os.getenv("DASHBOARD_API_KEY", "ak_dev_1234567890abcdef")

st.set_page_config(
    page_title="AI Gateway Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stMetric { background: #0f1117; border: 1px solid #1e2130; border-radius: 12px; padding: 1rem; }
.stMetric label { color: #6b7280 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.1em; }
.stMetric [data-testid="metric-container"] > div:nth-child(2) { font-size: 2rem !important; font-weight: 700; color: #f9fafb !important; }
.block-container { padding: 1.5rem 2rem; }
h1, h2, h3 { font-weight: 700; }
.stSelectbox label, .stSlider label { color: #9ca3af; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ─── Data fetching ───────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def fetch_metrics():
    try:
        r = httpx.get(
            f"{GATEWAY_URL}/v1/metrics",
            headers={"X-API-Key": API_KEY},
            timeout=5,
        )
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        return None


@st.cache_data(ttl=30)
def fetch_models():
    try:
        r = httpx.get(f"{GATEWAY_URL}/v1/models", timeout=5)
        return r.json().get("models", []) if r.status_code == 200 else []
    except:
        return []


def send_test_request(prompt, strategy, max_tokens):
    try:
        r = httpx.post(
            f"{GATEWAY_URL}/v1/generate",
            headers={"X-API-Key": API_KEY, "X-Team-ID": "engineering"},
            json={"prompt": prompt, "strategy": strategy, "max_tokens": max_tokens},
            timeout=30,
        )
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ AI Gateway")
    st.markdown("---")
    st.markdown("**Gateway URL**")
    st.code(GATEWAY_URL, language=None)

    st.markdown("**Status**")
    try:
        health = httpx.get(f"{GATEWAY_URL}/health", timeout=3)
        if health.status_code == 200:
            st.success("🟢 Online")
        else:
            st.error("🔴 Unhealthy")
    except:
        st.warning("🟡 Unreachable (using mock data)")

    st.markdown("---")
    st.markdown("**Navigation**")
    page = st.radio("", ["📊 Overview", "🔀 Routing Playground", "📋 Model Catalog", "⚙️ Policy Config"], label_visibility="collapsed")
    st.markdown("---")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# ─── Mock data for demo mode ──────────────────────────────────────────────────
MOCK_METRICS = {
    "summary": {
        "total_requests": 1842,
        "total_cost_usd": 12.4871,
        "total_tokens": 2847392,
        "avg_latency_ms": 643.2,
        "success_rate": 98.7,
    },
    "daily_usage": [
        {"date": "2026-03-15", "cost": 1.23, "requests": 187},
        {"date": "2026-03-16", "cost": 1.87, "requests": 253},
        {"date": "2026-03-17", "cost": 2.41, "requests": 318},
        {"date": "2026-03-18", "cost": 1.92, "requests": 274},
        {"date": "2026-03-19", "cost": 2.18, "requests": 301},
        {"date": "2026-03-20", "cost": 1.74, "requests": 245},
        {"date": "2026-03-21", "cost": 1.11, "requests": 264},
    ],
    "model_breakdown": [
        {"model": "gpt-4o-mini",               "requests": 891,  "total_cost": 3.21,  "avg_latency_ms": 412},
        {"model": "claude-sonnet-4-20250514",  "requests": 523,  "total_cost": 5.87,  "avg_latency_ms": 721},
        {"model": "gpt-4o",                    "requests": 287,  "total_cost": 2.14,  "avg_latency_ms": 893},
        {"model": "claude-3-haiku-20240307",   "requests": 141,  "total_cost": 1.28,  "avg_latency_ms": 318},
    ],
    "query_types": [
        {"type": "coding",         "count": 612},
        {"type": "reasoning",      "count": 487},
        {"type": "summarization",  "count": 341},
        {"type": "creative",       "count": 228},
        {"type": "simple",         "count": 174},
    ],
    "top_teams": [
        {"team": "engineering", "cost": 8.21,  "requests": 1124},
        {"team": "marketing",   "cost": 2.87,  "requests": 487},
        {"team": "qa",          "cost": 1.39,  "requests": 231},
    ],
    "forecast": {
        "current_month_cost": 12.49,
        "forecasted_monthly_cost": 19.87,
        "days_elapsed": 21,
    },
}

data = fetch_metrics() or MOCK_METRICS
summary = data["summary"]

# ─── OVERVIEW PAGE ────────────────────────────────────────────────────────────
if "Overview" in page:
    st.title("⚡ Enterprise AI Gateway")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Requests", f"{summary['total_requests']:,}")
    c2.metric("Total Cost", f"${summary['total_cost_usd']:.4f}")
    c3.metric("Total Tokens", f"{summary['total_tokens']:,}")
    c4.metric("Avg Latency", f"{summary['avg_latency_ms']:.0f}ms")
    c5.metric("Success Rate", f"{summary['success_rate']}%")

    st.markdown("---")

    # Row 2: Daily usage + Model pie
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📈 Daily Usage")
        daily_df = pd.DataFrame(data["daily_usage"])
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=daily_df["date"], y=daily_df["requests"],
            name="Requests", marker_color="#3b82f6", opacity=0.7
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=daily_df["date"], y=daily_df["cost"],
            name="Cost ($)", line=dict(color="#f59e0b", width=2),
            mode="lines+markers"
        ), secondary_y=True)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#9ca3af", height=300, margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(orientation="h", y=1.1),
        )
        fig.update_yaxes(title_text="Requests", secondary_y=False, gridcolor="#1e2130")
        fig.update_yaxes(title_text="Cost USD", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("🔵 Model Usage")
        model_df = pd.DataFrame(data["model_breakdown"])
        model_df["model_short"] = model_df["model"].apply(lambda x: x.split("-")[0] + "..." + x[-8:] if len(x) > 15 else x)
        fig2 = px.pie(
            model_df, names="model_short", values="requests",
            color_discrete_sequence=["#3b82f6", "#8b5cf6", "#f59e0b", "#10b981"],
            hole=0.5,
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#9ca3af",
            height=300, margin=dict(l=0, r=0, t=20, b=0),
            showlegend=True, legend=dict(font_size=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Row 3: Latency bar + Query types + Forecast
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("⚡ Latency by Model")
        model_df = pd.DataFrame(data["model_breakdown"]).sort_values("avg_latency_ms")
        model_df["label"] = model_df["model"].apply(lambda x: x[:20])
        fig3 = px.bar(
            model_df, x="avg_latency_ms", y="label", orientation="h",
            color="avg_latency_ms",
            color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#9ca3af", height=250, margin=dict(l=0, r=0, t=20, b=0),
            coloraxis_showscale=False, yaxis_title="", xaxis_title="Avg ms",
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.subheader("🏷️ Query Types")
        qt_df = pd.DataFrame(data["query_types"])
        fig4 = px.bar(
            qt_df, x="count", y="type", orientation="h",
            color_discrete_sequence=["#8b5cf6"],
        )
        fig4.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#9ca3af", height=250, margin=dict(l=0, r=0, t=20, b=0),
            yaxis_title="", xaxis_title="Requests",
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col3:
        st.subheader("💰 Monthly Forecast")
        fc = data["forecast"]
        days_left = 30 - fc["days_elapsed"]
        remaining = fc["forecasted_monthly_cost"] - fc["current_month_cost"]

        fig5 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=fc["current_month_cost"],
            delta={"reference": fc["forecasted_monthly_cost"], "valueformat": ".2f", "suffix": " forecast"},
            number={"prefix": "$", "valueformat": ".2f"},
            gauge={
                "axis": {"range": [0, fc["forecasted_monthly_cost"] * 1.3]},
                "bar": {"color": "#3b82f6"},
                "steps": [
                    {"range": [0, fc["forecasted_monthly_cost"] * 0.6], "color": "#1e2130"},
                    {"range": [fc["forecasted_monthly_cost"] * 0.6, fc["forecasted_monthly_cost"]], "color": "#292d3e"},
                ],
                "threshold": {"line": {"color": "#f59e0b", "width": 2}, "thickness": 0.75, "value": fc["forecasted_monthly_cost"]},
            },
        ))
        fig5.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#9ca3af",
            height=250, margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig5, use_container_width=True)
        st.caption(f"📅 Day {fc['days_elapsed']}/30 · {days_left} days left · ${remaining:.2f} projected remaining")

    # Team spend table
    st.subheader("👥 Team Spend")
    team_df = pd.DataFrame(data["top_teams"])
    team_df["cost_bar"] = team_df["cost"] / team_df["cost"].max()
    st.dataframe(
        team_df[["team", "requests", "cost"]].rename(columns={"team": "Team", "requests": "Requests", "cost": "Cost ($)"}),
        use_container_width=True, hide_index=True,
    )


# ─── ROUTING PLAYGROUND ──────────────────────────────────────────────────────
elif "Playground" in page:
    st.title("🔀 Routing Playground")
    st.caption("Send real requests through the gateway and see how routing decisions are made.")

    col1, col2 = st.columns([2, 1])
    with col1:
        prompt = st.text_area("Prompt", height=150, placeholder="Write a Python function to parse JSON from a REST API response...", value="Write a Python function that parses a nested JSON response from a REST API and extracts all email addresses recursively.")
    with col2:
        strategy = st.selectbox("Routing Strategy", ["balanced", "cost_optimized", "performance"])
        max_tokens = st.slider("Max Tokens", 100, 2000, 500, 100)
        team = st.selectbox("Team", ["engineering", "marketing", "qa"])

    if st.button("🚀 Send Request", type="primary"):
        with st.spinner("Routing request through gateway..."):
            result = send_test_request(prompt, strategy, max_tokens)

        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.success("✅ Request completed")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Model Used", result.get("model_used", "—").split("-")[0])
            c2.metric("Tokens", f"{result.get('tokens_used', 0):,}")
            c3.metric("Cost", f"${result.get('estimated_cost_usd', 0):.6f}")
            c4.metric("Latency", f"{result.get('latency_ms', 0):.0f}ms")

            col_a, col_b = st.columns(2)
            with col_a:
                st.info(f"**Query Type:** {result.get('query_type', '—')}\n\n**Complexity:** {result.get('complexity_score', 0):.3f}\n\n**Routing Reason:** {result.get('routing_reason', '—')}")
            with col_b:
                st.info(f"**Provider:** {result.get('provider', '—')}\n\n**Fallback Used:** {result.get('fallback_used', False)}\n\n**Request ID:** `{result.get('request_id', '—')[:16]}...`")

            st.subheader("Response")
            st.markdown(result.get("content", ""))


# ─── MODEL CATALOG ───────────────────────────────────────────────────────────
elif "Catalog" in page:
    st.title("📋 Model Catalog")
    models = fetch_models() or [
        {"id": "gpt-4o", "provider": "openai", "tier": "premium", "cost_per_1k_tokens": 0.005},
        {"id": "gpt-4o-mini", "provider": "openai", "tier": "standard", "cost_per_1k_tokens": 0.00015},
        {"id": "claude-3-haiku-20240307", "provider": "anthropic", "tier": "standard", "cost_per_1k_tokens": 0.00025},
        {"id": "claude-sonnet-4-20250514", "provider": "anthropic", "tier": "premium", "cost_per_1k_tokens": 0.003},
    ]
    df = pd.DataFrame(models)
    df["cost_per_1k_tokens"] = df["cost_per_1k_tokens"].apply(lambda x: f"${x:.5f}")
    df.columns = [c.replace("_", " ").title() for c in df.columns]
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("💸 Cost Comparison (per 1K tokens)")
    models_raw = [
        {"model": "gpt-4o", "cost": 0.005},
        {"model": "claude-sonnet-4-20250514", "cost": 0.003},
        {"model": "claude-3-haiku-20240307", "cost": 0.00025},
        {"model": "gpt-4o-mini", "cost": 0.00015},
    ]
    cdf = pd.DataFrame(models_raw).sort_values("cost", ascending=True)
    fig = px.bar(cdf, x="model", y="cost", color="cost", color_continuous_scale=["#10b981", "#ef4444"], text_auto=".5f")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#9ca3af", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


# ─── POLICY CONFIG ───────────────────────────────────────────────────────────
elif "Policy" in page:
    st.title("⚙️ Policy Configuration")
    st.caption("Configure budget caps and access policies per team.")

    teams = ["engineering", "marketing", "qa"]
    selected_team = st.selectbox("Select Team", teams)

    defaults = {
        "engineering": {"daily": 50.0, "monthly": 1000.0},
        "marketing":   {"daily": 10.0, "monthly": 200.0},
        "qa":          {"daily": 5.0,  "monthly": 100.0},
    }
    d = defaults[selected_team]

    col1, col2 = st.columns(2)
    with col1:
        daily = st.number_input("Daily Budget (USD)", value=d["daily"], min_value=0.0, step=1.0)
    with col2:
        monthly = st.number_input("Monthly Budget (USD)", value=d["monthly"], min_value=0.0, step=10.0)

    st.subheader("Model Access")
    allow_premium = st.toggle("Allow Premium Models (GPT-4o, Claude Opus)", value=selected_team == "engineering")
    block_team = st.toggle("Block All Requests", value=False)

    if st.button("💾 Save Policy", type="primary"):
        st.success(f"✅ Policy saved for team '{selected_team}'")
        st.info(f"Daily: ${daily} | Monthly: ${monthly} | Premium: {allow_premium} | Blocked: {block_team}")
