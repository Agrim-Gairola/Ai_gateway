"""
Analytics Engine — Powers the dashboard with real DB queries.
"""
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from observability.database import RequestLog


async def get_summary_metrics(db: Session) -> dict:
    total = db.query(func.count(RequestLog.id)).scalar() or 0
    total_cost = db.query(func.sum(RequestLog.cost_usd)).scalar() or 0.0
    total_tokens = db.query(func.sum(RequestLog.tokens_used)).scalar() or 0
    avg_latency = db.query(func.avg(RequestLog.latency_ms)).scalar() or 0.0
    success_count = db.query(func.count(RequestLog.id)).filter(RequestLog.success == True).scalar() or 0

    # Last 7 days cost per day
    daily = []
    for i in range(6, -1, -1):
        day = (datetime.utcnow() - timedelta(days=i)).date()
        day_cost = db.query(func.sum(RequestLog.cost_usd)).filter(
            func.date(RequestLog.created_at) == day
        ).scalar() or 0.0
        day_requests = db.query(func.count(RequestLog.id)).filter(
            func.date(RequestLog.created_at) == day
        ).scalar() or 0
        daily.append({"date": str(day), "cost": round(day_cost, 4), "requests": day_requests})

    # Model usage breakdown
    model_usage = db.query(
        RequestLog.model,
        func.count(RequestLog.id).label("count"),
        func.sum(RequestLog.cost_usd).label("total_cost"),
        func.avg(RequestLog.latency_ms).label("avg_latency"),
    ).group_by(RequestLog.model).all()

    # Query type distribution
    type_dist = db.query(
        RequestLog.query_type,
        func.count(RequestLog.id).label("count"),
    ).group_by(RequestLog.query_type).all()

    # Top teams by spend
    team_spend = db.query(
        RequestLog.team_id,
        func.sum(RequestLog.cost_usd).label("total_cost"),
        func.count(RequestLog.id).label("requests"),
    ).group_by(RequestLog.team_id).order_by(desc("total_cost")).limit(10).all()

    # Monthly forecast (linear extrapolation)
    days_elapsed = datetime.utcnow().day
    month_cost = db.query(func.sum(RequestLog.cost_usd)).filter(
        RequestLog.created_at >= datetime.utcnow().replace(day=1, hour=0, minute=0)
    ).scalar() or 0.0
    forecasted_monthly = (month_cost / max(days_elapsed, 1)) * 30

    return {
        "summary": {
            "total_requests": total,
            "total_cost_usd": round(total_cost, 4),
            "total_tokens": total_tokens,
            "avg_latency_ms": round(avg_latency, 2),
            "success_rate": round((success_count / max(total, 1)) * 100, 1),
        },
        "daily_usage": daily,
        "model_breakdown": [
            {
                "model": r.model,
                "requests": r.count,
                "total_cost": round(r.total_cost or 0, 4),
                "avg_latency_ms": round(r.avg_latency or 0, 2),
            }
            for r in model_usage
        ],
        "query_types": [{"type": r.query_type, "count": r.count} for r in type_dist],
        "top_teams": [
            {"team": r.team_id, "cost": round(r.total_cost, 4), "requests": r.requests}
            for r in team_spend
        ],
        "forecast": {
            "current_month_cost": round(month_cost, 4),
            "forecasted_monthly_cost": round(forecasted_monthly, 4),
            "days_elapsed": days_elapsed,
        },
    }


async def get_team_usage(team_id: str, db: Session) -> dict:
    records = db.query(RequestLog).filter(
        RequestLog.team_id == team_id
    ).order_by(desc(RequestLog.created_at)).limit(50).all()

    return {
        "team_id": team_id,
        "recent_requests": [
            {
                "id": r.id[:8],
                "model": r.model,
                "query_type": r.query_type,
                "tokens": r.tokens_used,
                "cost": round(r.cost_usd, 6),
                "latency_ms": r.latency_ms,
                "success": r.success,
                "timestamp": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ],
    }
