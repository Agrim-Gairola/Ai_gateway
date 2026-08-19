"""
Test Suite — Gateway core components
Run: pytest tests/ -v
"""
import pytest
import asyncio
from intelligence.classifier import QueryClassifier
from intelligence.complexity import ComplexityScorer
from intelligence.cost_estimator import CostEstimator
from intelligence.routing_engine import RoutingEngine
from gateway.schemas import RoutingStrategy


# ── Classifier tests ──────────────────────────────────────────────────────────
class TestQueryClassifier:
    clf = QueryClassifier()

    def test_coding_detection(self):
        result = self.clf.classify("Write a Python function to parse JSON from a REST API")
        assert result == "coding"

    def test_summarization_detection(self):
        result = self.clf.classify("Summarize this article for me in 3 bullet points")
        assert result == "summarization"

    def test_reasoning_detection(self):
        result = self.clf.classify("Why is Rust faster than Python? Compare their trade-offs")
        assert result == "reasoning"

    def test_simple_detection(self):
        result = self.clf.classify("What is the capital of France?")
        assert result == "simple"

    def test_creative_detection(self):
        result = self.clf.classify("Write a blog post about the future of AI")
        assert result == "creative"


# ── Complexity scorer tests ───────────────────────────────────────────────────
class TestComplexityScorer:
    scorer = ComplexityScorer()

    def test_simple_prompt_low_score(self):
        score = self.scorer.score("What is Python?", "simple")
        assert score < 0.4

    def test_complex_coding_high_score(self):
        prompt = """
        Implement a distributed rate limiter using Redis with token bucket algorithm.
        It should support concurrent access, handle network partitions, and guarantee
        at-most-once delivery semantics. Include unit tests and benchmark results.
        What are the trade-offs vs sliding window approach?
        """
        score = self.scorer.score(prompt, "coding")
        assert score > 0.5

    def test_score_range(self):
        for prompt in ["hi", "explain quantum computing in detail", "write code"]:
            score = self.scorer.score(prompt, "unknown")
            assert 0.0 <= score <= 1.0


# ── Cost estimator tests ──────────────────────────────────────────────────────
class TestCostEstimator:
    est = CostEstimator()

    def test_gpt4o_more_expensive_than_mini(self):
        cost_4o   = self.est.estimate("gpt-4o",      "hello world", 100)
        cost_mini = self.est.estimate("gpt-4o-mini", "hello world", 100)
        assert cost_4o > cost_mini

    def test_cost_positive(self):
        cost = self.est.estimate("gpt-4o-mini", "test prompt", 500)
        assert cost > 0

    def test_longer_prompt_costs_more(self):
        short = self.est.estimate("gpt-4o", "hi", 100)
        long  = self.est.estimate("gpt-4o", "hi " * 500, 100)
        assert long > short


# ── Routing engine tests ──────────────────────────────────────────────────────
class TestRoutingEngine:
    engine = RoutingEngine()

    def test_cost_optimized_always_cheap(self):
        model, provider, reason = self.engine.select(
            "coding", 0.9, RoutingStrategy.COST_OPTIMIZED
        )
        assert "mini" in model or "haiku" in model

    def test_performance_always_premium(self):
        model, provider, reason = self.engine.select(
            "simple", 0.1, RoutingStrategy.PERFORMANCE
        )
        assert model == "gpt-4o"

    def test_preferred_model_honored(self):
        model, provider, reason = self.engine.select(
            "coding", 0.5, RoutingStrategy.BALANCED,
            preferred_model="claude-3-haiku-20240307"
        )
        assert model == "claude-3-haiku-20240307"

    def test_balanced_complex_coding_gets_strong_model(self):
        model, provider, reason = self.engine.select(
            "coding", 0.85, RoutingStrategy.BALANCED
        )
        assert model in ("gpt-4o", "claude-sonnet-4-20250514")

    def test_balanced_simple_gets_cheap_model(self):
        model, provider, reason = self.engine.select(
            "simple", 0.05, RoutingStrategy.BALANCED
        )
        assert "mini" in model or "haiku" in model
