"""
Cost Estimator
Predicts cost BEFORE calling the model, using token heuristics.
Uses actual 2025 pricing per model.
"""


class CostEstimator:
    """
    Pricing per 1M tokens (input/output), USD.
    Source: OpenAI + Anthropic pricing pages (update as needed).
    """

    # model → (input_cost_per_1k, output_cost_per_1k)
    PRICING = {
        "gpt-4o":                       (0.0025,  0.010),
        "gpt-4o-mini":                  (0.00015, 0.0006),
        "gpt-4-turbo":                  (0.010,   0.030),
        "gpt-3.5-turbo":                (0.0005,  0.0015),
        "claude-sonnet-4-20250514":     (0.003,   0.015),
        "claude-3-haiku-20240307":      (0.00025, 0.00125),
        "claude-opus-4-20250514":       (0.015,   0.075),
    }

    DEFAULT_PRICING = (0.001, 0.002)

    def estimate(self, model: str, prompt: str, max_tokens: int) -> float:
        """
        Estimates cost using:
        - Input tokens ≈ words * 1.3
        - Output tokens = max_tokens (worst case)
        """
        input_tokens = len(prompt.split()) * 1.3
        output_tokens = max_tokens

        input_cost_per_1k, output_cost_per_1k = self.PRICING.get(
            model, self.DEFAULT_PRICING
        )

        cost = (
            (input_tokens / 1000) * input_cost_per_1k
            + (output_tokens / 1000) * output_cost_per_1k
        )
        return round(cost, 8)

    def actual_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculates actual cost after API call with real token counts."""
        input_cost_per_1k, output_cost_per_1k = self.PRICING.get(
            model, self.DEFAULT_PRICING
        )
        cost = (
            (input_tokens / 1000) * input_cost_per_1k
            + (output_tokens / 1000) * output_cost_per_1k
        )
        return round(cost, 8)

    @staticmethod
    def get_pricing_table() -> dict:
        return CostEstimator.PRICING
