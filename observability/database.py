"""
Database - SQLAlchemy models + session factory
PostgreSQL in production, SQLite for dev.
"""
import os
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the repository root, wherever the repo happens to live.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

Base = declarative_base()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ai_gateway.db")


class RequestLog(Base):
    __tablename__ = "request_logs"
    id               = Column(String, primary_key=True, index=True)
    team_id          = Column(String, index=True, nullable=False)
    model            = Column(String, nullable=False)
    provider         = Column(String, nullable=False)
    query_type       = Column(String, nullable=False)
    complexity_score = Column(Float, nullable=False)
    tokens_used      = Column(Integer, nullable=False)
    cost_usd         = Column(Float, nullable=False)
    latency_ms       = Column(Float, default=0.0)
    success          = Column(Boolean, default=True)
    error_msg        = Column(Text, nullable=True)
    created_at       = Column(DateTime, default=datetime.utcnow)


class TeamBudget(Base):
    __tablename__ = "team_budgets"
    team_id            = Column(String, primary_key=True)
    daily_budget_usd   = Column(Float, default=10.0)
    monthly_budget_usd = Column(Float, default=200.0)
    blocked            = Column(Boolean, default=False)
    block_reason       = Column(Text, nullable=True)
    updated_at         = Column(DateTime, default=datetime.utcnow)


engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


async def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
