"""
Policy Engine — Budget caps, model blocking, team restrictions.
This is what makes this an "enterprise" gateway.
"""
import logging
from datetime import datetime, timedelta
from typing import Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func

from observability.database import RequestLog, TeamBudget

logger = logging.getLogger("ai_gateway.policy")

# Default budgets per team (USD)
DEFAULT_BUDGETS = {
    "engineering": {"daily": 50.0, "monthly": 1000.0},
    "marketing":   {"daily": 10.0, "monthly": 200.0},
    "qa":          {"daily": 5.0,  "monthly": 100.0},
    "default":     {"daily": 2.0,  "monthly": 50.0},
}

# Premium models that require special approval
PREMIUM_MODELS = {"gpt-4o", "claude-opus-4-20250514", "gpt-4-turbo"}


class PolicyEngine:
    """
    Checks before every request:
    1. Is the team's daily budget exceeded?
    2. Is the team's monthly budget exceeded?
    3. Is the team blocked from using premium models?
    4. Is the estimated cost above single-request limit?
    """

    MAX_SINGLE_REQUEST_COST = 0.50  # $0.50 per request max

    async def check(
        self,
        team_id: str,
        model: str,
        estimated_cost: float,
        db: Session,
    ) -> Tuple[bool, str]:
        """Returns (allowed, reason)."""

        # 1. Single request cost check
        if estimated_cost > self.MAX_SINGLE_REQUEST_COST:
            return False, f"Request cost ${estimated_cost:.4f} exceeds limit ${self.MAX_SINGLE_REQUEST_COST}"

        # 2. Load/create team budget config
        budget = db.query(TeamBudget).filter(TeamBudget.team_id == team_id).first()
        if not budget:
            defaults = DEFAULT_BUDGETS.get(team_id, DEFAULT_BUDGETS["default"])
            budget = TeamBudget(
                team_id=team_id,
                daily_budget_usd=defaults["daily"],
                monthly_budget_usd=defaults["monthly"],
            )
            db.add(budget)
            db.commit()

        # 3. Check if team is blocked
        if budget.blocked:
            return False, f"Team '{team_id}' is blocked: {budget.block_reason}"

        # 4. Daily spend check
        today = datetime.utcnow().date()
        daily_spend = db.query(func.sum(RequestLog.cost_usd)).filter(
            RequestLog.team_id == team_id,
            func.date(RequestLog.created_at) == today,
            RequestLog.success == True,
        ).scalar() or 0.0

        if daily_spend + estimated_cost > budget.daily_budget_usd:
            return False, f"Daily budget exhausted for '{team_id}': ${daily_spend:.4f} / ${budget.daily_budget_usd:.2f}"

        # 5. Monthly spend check
        month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0)
        monthly_spend = db.query(func.sum(RequestLog.cost_usd)).filter(
            RequestLog.team_id == team_id,
            RequestLog.created_at >= month_start,
            RequestLog.success == True,
        ).scalar() or 0.0

        if monthly_spend + estimated_cost > budget.monthly_budget_usd:
            return False, f"Monthly budget exhausted for '{team_id}': ${monthly_spend:.4f} / ${budget.monthly_budget_usd:.2f}"

        # 6. Premium model access check
        if model in PREMIUM_MODELS and team_id in {"qa", "default"}:
            return False, f"Team '{team_id}' does not have access to premium model '{model}'"

        logger.info(f"Policy OK for {team_id}: daily=${daily_spend:.4f}, monthly=${monthly_spend:.4f}")
        return True, "OK"
