from typing import Tuple, Optional
from gateway.schemas import QueryType, RoutingStrategy


class RoutingEngine:
    BALANCED_ROUTES = {
        QueryType.CODING: [
            (0.3, "llama-3.3-70b-versatile", "groq",   "Complex coding → Llama 70B on Groq"),
            (0.0, "llama-3.1-8b-instant",    "groq",   "Simple coding → Llama 8B on Groq"),
        ],
        QueryType.REASONING: [
            (0.2, "gemini-2.5-flash",        "gemini", "Deep reasoning → Gemini 2.5 Flash"),
            (0.0, "llama-3.1-8b-instant",    "groq",   "Light reasoning → Llama 8B on Groq"),
        ],
        QueryType.SUMMARIZATION: [
            (0.0, "gemini-2.5-flash",        "gemini", "Summarization → Gemini 2.5 Flash"),
        ],
        QueryType.CREATIVE: [
            (0.3, "gemini-2.5-flash",        "gemini", "Creative → Gemini 2.5 Flash"),
            (0.0, "llama-3.1-8b-instant",    "groq",   "Simple creative → Llama 8B"),
        ],
        QueryType.SIMPLE: [
            (0.0, "llama-3.1-8b-instant",    "groq",   "Simple query → Llama 8B (fastest)"),
        ],
        QueryType.UNKNOWN: [
            (0.3, "gemini-2.5-flash",        "gemini", "Unknown complex → Gemini 2.5 Flash"),
            (0.0, "llama-3.1-8b-instant",    "groq",   "Unknown simple → Llama 8B"),
        ],
    }

    PREMIUM_MODEL = ("llama-3.3-70b-versatile", "groq")
    ECONOMY_MODEL = ("llama-3.1-8b-instant", "groq")
    BASELINE_MODEL_COST_PER_1K = 0.0025

    def select(self, query_type, complexity, strategy, preferred_model=None) -> Tuple[str, str, str]:
        if preferred_model:
            provider = self._provider_for_model(preferred_model)
            return preferred_model, provider, f"User-specified: {preferred_model}"

        if strategy == RoutingStrategy.PERFORMANCE:
            m, p = self.PREMIUM_MODEL
            return m, p, "Performance strategy → Llama 70B on Groq"

        if strategy == RoutingStrategy.COST_OPTIMIZED:
            m, p = self.ECONOMY_MODEL
            return m, p, "Cost-optimized → Llama 8B on Groq (free)"

        qt = QueryType(query_type) if query_type in [e.value for e in QueryType] else QueryType.UNKNOWN
        routes = self.BALANCED_ROUTES.get(qt, self.BALANCED_ROUTES[QueryType.UNKNOWN])

        for threshold, model, provider, reason in routes:
            if complexity >= threshold:
                return model, provider, reason

        m, p = self.ECONOMY_MODEL
        return m, p, "Default → Llama 8B on Groq"

    def _provider_for_model(self, model: str) -> str:
        if "llama" in model or "mixtral" in model: return "groq"
        if "gemini" in model: return "gemini"
        if "gpt" in model: return "openai"
        if "claude" in model: return "anthropic"
        return "groq"