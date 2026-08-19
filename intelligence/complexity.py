"""
Complexity Scorer
Returns a 0.0 → 1.0 score based on multiple heuristics.
High score = needs a powerful (expensive) model.
"""
import math
import re


class ComplexityScorer:
    """
    Scores based on:
    - Token count (length proxy)
    - Sentence count and structure
    - Question depth (multi-part questions)
    - Technical term density
    - Presence of code blocks
    """

    TECHNICAL_TERMS = {
        "algorithm", "architecture", "concurrent", "distributed", "microservices",
        "transformer", "embedding", "gradient", "optimization", "inference",
        "kubernetes", "orchestration", "latency", "throughput", "consensus",
        "cryptography", "polynomial", "derivative", "eigenvector", "bayesian",
    }

    def score(self, prompt: str, query_type: str) -> float:
        scores = []

        # 1. Length score (log scale, capped at 1.0)
        word_count = len(prompt.split())
        length_score = min(1.0, math.log(max(word_count, 1) + 1) / math.log(500))
        scores.append(("length", length_score, 0.25))

        # 2. Technical density
        words = set(prompt.lower().split())
        tech_overlap = len(words & self.TECHNICAL_TERMS)
        tech_score = min(1.0, tech_overlap / 3)
        scores.append(("technical", tech_score, 0.30))

        # 3. Multi-part question detection
        question_marks = prompt.count("?")
        numbered_items = len(re.findall(r"\d+\.", prompt))
        multi_part_score = min(1.0, (question_marks + numbered_items) / 5)
        scores.append(("multi_part", multi_part_score, 0.20))

        # 4. Code complexity
        has_code = "```" in prompt or re.search(r"\bdef |class |import |SELECT |FROM ", prompt)
        code_score = 0.8 if has_code else 0.0
        scores.append(("code", code_score, 0.15))

        # 5. Query type baseline
        type_baselines = {
            "coding": 0.7,
            "reasoning": 0.6,
            "summarization": 0.4,
            "creative": 0.5,
            "simple": 0.1,
            "unknown": 0.4,
        }
        baseline = type_baselines.get(query_type, 0.4)
        scores.append(("type_baseline", baseline, 0.10))

        # Weighted sum
        total = sum(s * w for _, s, w in scores)
        return round(min(1.0, max(0.0, total)), 3)
