"""
pcr_config.py
=============
Single source of truth for every PCR pair: which models play which role,
which provider each model lives on, and how many questions to run per
benchmark. Add a new pair by adding one entry to PAIRS -- nothing else in
the pipeline needs to change.

Historical pairs (1 and 2) are included with is_historical=True purely so
the comparison-table generator can pull their already-exported summaries
in alongside newly-run pairs. They are NOT re-run by pcr_runner.py.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ModelSpec:
    provider: str            # "groq" | "mistral" | "gemini"
    model: str               # provider-specific model id
    cost_tier: str = "free"  # "free" | "paid" -- pcr_runner.py refuses to run any
                              # pair containing a "paid" model unless --allow-paid
                              # is passed explicitly. Verified as of Aug 2026:
                              #   groq: entire self-serve catalog is $0 until you
                              #     hit a model's daily token cap, then it blocks
                              #     the request rather than billing you -- as long
                              #     as no card/auto-recharge is on the account.
                              #   mistral: "Experiment" tier on La Plateforme is
                              #     $0 for every model (incl. mistral-large) up to
                              #     a generous monthly token cap, rate-limited but
                              #     no card required.
                              #   gemini: Pro-tier models (gemini-3.1-pro etc.)
                              #     went paid-only in April 2026. Only Flash /
                              #     Flash-Lite tiers remain free.
                              # Re-verify against each provider's live pricing
                              # page before a real run -- these move often.


@dataclass(frozen=True)
class PairConfig:
    pair_id: str                    # used as checkpoint/export file prefix, e.g. "pair3"
    label: str                      # human-readable, goes in report headers
    tests: str                      # one-line note on what this pair isolates
    cheap_a: ModelSpec
    cheap_b: ModelSpec
    frontier: ModelSpec
    benchmarks: dict                # {"mmlu": 250, "gsm8k": 150, "arc": 100}
    is_historical: bool = False     # True => don't run, just reference in tables
    notes: str = ""


PAIRS: dict[str, PairConfig] = {

    "pair1": PairConfig(
        pair_id="pair1",
        label="Pair 1 (original paper, historical)",
        tests="Original baseline",
        cheap_a=ModelSpec("groq", "llama-3.1-8b-instant"),
        cheap_b=ModelSpec("gemini", "gemini-2.5-flash-lite"),
        frontier=ModelSpec("groq", "llama-3.3-70b-versatile"),
        benchmarks={"mmlu": 1000, "gsm8k": 500, "arc": 300},
        is_historical=True,
        notes="Frontier model (llama-3.3-70b-versatile) deprecated by Groq "
              "2026-06-17, shut down 2026-08-16 -- cannot be re-run, kept "
              "for table reference only.",
    ),

    "pair2": PairConfig(
        pair_id="pair2",
        label="Pair 2 (cross-family, cross-provider)",
        tests="Cross-family, cross-provider replication (already run)",
        # NOTE: cheap_a here reflects the FINAL model used (allam-2-7b, used
        # for GSM8K + ARC). MMLU q1-800 actually used openai/gpt-oss-20b --
        # see the mixed-pair disclosure note in pcr_config.PAIR2_MMLU_CHEAP_A
        # below. This is intentional and must be footnoted in the paper.
        cheap_a=ModelSpec("groq", "allam-2-7b"),
        cheap_b=ModelSpec("mistral", "mistral-small-latest"),
        frontier=ModelSpec("groq", "openai/gpt-oss-120b"),
        benchmarks={"mmlu": 1000, "gsm8k": 500, "arc": 300},
        is_historical=True,
        notes="MIXED PAIR: MMLU (q1-800) used openai/gpt-oss-20b as cheap-A; "
              "GSM8K and ARC used allam-2-7b as cheap-A, after two forced "
              "swaps (gpt-oss-20b hit a 200k TPD quota wall; qwen/qwen3-32b "
              "was tried and 404'd, deprecated before use). Disclose this "
              "explicitly in methods -- do not present as one consistent "
              "cheap-A model across benchmarks.",
    ),

    "pair3": PairConfig(
        pair_id="pair3",
        label="Pair 3 (same-family control)",
        tests="Isolates family-relatedness -- everything same lab/lineage, "
              "unlike Pair 2",
        cheap_a=ModelSpec("groq", "openai/gpt-oss-20b"),
        cheap_b=ModelSpec("groq", "openai/gpt-oss-safeguard-20b"),
        frontier=ModelSpec("groq", "openai/gpt-oss-120b"),
        benchmarks={"mmlu": 250, "gsm8k": 150, "arc": 100, "truthfulqa": 150,
                    "mmlu_pro": 150, "openbookqa": 200, "commonsenseqa": 200},
        notes="Pilot N (not full-scale) -- report as pilot replication.",
    ),

    "pair4": PairConfig(
        pair_id="pair4",
        label="Pair 4 (cross-family, capability-matched)",
        tests="Closest replication of Pair 1's design (small Groq model + "
            "Gemini-Flash-Lite) with current live models",
        cheap_a=ModelSpec("groq", "allam-2-7b"),
        cheap_b=ModelSpec("gemini", "gemini-3.5-flash-lite"),   # was gemini-2.5-flash-lite (404, sunset for new users)
        frontier=ModelSpec("groq", "openai/gpt-oss-120b"),
        benchmarks={"mmlu": 250, "gsm8k": 800, "arc": 100},   # gsm8k bumped from pilot N=150
        notes="Pilot N (not full-scale) -- report as pilot replication. "
            "GSM8K expanded to N=800 on [today's date] -- see run log.",
    ),

    "pair5": PairConfig(
        pair_id="pair5",
        label="Pair 5 (frontier-swap, subsample)",
        tests="Tests whether escalation accuracy changes with a different "
              "Gemini-family frontier vs. an open 120B model",
        # Reuses Pair 4's cheap layer so the ONLY variable that changes vs.
        # Pair 4 is the frontier model. If you'd rather reuse Pair 2's cheap
        # layer instead, swap cheap_a/cheap_b below and update `tests`.
        cheap_a=ModelSpec("groq", "allam-2-7b"),
        cheap_b=ModelSpec("gemini", "gemini-2.5-flash-lite"),
        # NOTE: originally specced as gemini-3.1-pro, but Pro-tier Gemini
        # models are paid-only as of April 2026 (confirmed live Aug 2026) --
        # switched to gemini-3-flash, Google's current recommended free-tier
        # model, to keep this pair at $0. This is a WEAKER test than the
        # original intent (Flash, not Pro, so less of a true "genuinely
        # stronger/pricier frontier" comparison) -- document that limitation
        # if you report this pair. If you later want the original paid
        # design, set frontier=ModelSpec("gemini", "gemini-3.1-pro",
        # cost_tier="paid") and pass --allow-paid to pcr_runner.py.
        frontier=ModelSpec("gemini", "gemini-3-flash"),
        benchmarks={"mmlu": 100},   # MMLU only first; expand later if interesting
        notes="Subsample, MMLU only. Frontier swapped from gemini-3.1-pro "
              "(paid-only) to gemini-3-flash (free tier) to keep this pair "
              "at $0 -- see cost_tier note on the frontier ModelSpec above.",
    ),

    # -----------------------------------------------------------------------
    # Focused-wave pairs (added 2026-09-03) -- a matched within-family /
    # cross-family contrast run on the two new benchmarks (TruthfulQA,
    # MMLU-Pro) plus MMLU/ARC anchors. Frontier is held constant
    # (openai/gpt-oss-120b) across BOTH so the only thing that varies between
    # pair6 and pair7 is whether the two cheap models share a family.
    # -----------------------------------------------------------------------
    "pair6": PairConfig(
        pair_id="pair6",
        label="Pair 6 (cross-family: gpt-oss-20B + Mistral-Small)",
        tests="Cross-family / cross-provider arm. Two capability-comparable "
              "mid-size instruction models from different labs (OpenAI-oss "
              "via Groq vs Mistral). Tests whether C-Hall on adversarial "
              "(TruthfulQA) and hard (MMLU-Pro) knowledge is lower when the "
              "cheap pair does NOT share pre-training lineage.",
        cheap_a=ModelSpec("groq", "openai/gpt-oss-20b"),
        cheap_b=ModelSpec("mistral", "mistral-small-latest"),
        frontier=ModelSpec("groq", "openai/gpt-oss-120b"),
        benchmarks={"truthfulqa": 150, "mmlu_pro": 150, "mmlu": 250, "arc": 200,
                    "openbookqa": 200, "commonsenseqa": 200, "gsm8k": 150},
        notes="Focused-wave pilot N. Cross-family arm; matched to pair7 on "
              "frontier and benchmark set.",
    ),

    "pair7": PairConfig(
        pair_id="pair7",
        label="Pair 7 (within-family: Qwen3 27B pair)",
        tests="Within-family control arm. Both cheap models are Qwen3 27B "
              "variants (same lab / same lineage, minor version apart), so "
              "capability is held ~constant while family relatedness is "
              "maximised. Direct within-family counterpart to pair6.",
        cheap_a=ModelSpec("groq", "qwen/qwen3.8-27b"),
        cheap_b=ModelSpec("groq", "qwen/qwen3.6-27b"),
        frontier=ModelSpec("groq", "openai/gpt-oss-120b"),
        benchmarks={"truthfulqa": 150, "mmlu_pro": 150, "mmlu": 250, "arc": 200,
                    "openbookqa": 200, "commonsenseqa": 200, "gsm8k": 150},
        notes="Focused-wave pilot N. Within-family arm (Qwen); third "
              "within-family data point alongside pair3 (gpt-oss) and the "
              "legacy Mistral pair3. Matched to pair6 on frontier + benchmarks.",
    ),

    # -----------------------------------------------------------------------
    # Wave-3 pairs (added 2026-09-03), NON-GROQ: a second matched
    # within-/cross-family contrast on an entirely different stack
    # (Mistral + Gemini, mistral-medium frontier). This decouples the
    # within-family condition from "hosted on Groq", which was a stated
    # confound in the wave-1/2 family analysis. Frontier held constant
    # (mistral-medium-latest) across pair8 and pair9, same design as
    # pair6/pair7. mistral-large-latest is NOT free-tier any more (403
    # tier_not_allowed as of 2026-09-03), so mistral-medium is the frontier.
    # -----------------------------------------------------------------------
    "pair8": PairConfig(
        pair_id="pair8",
        label="Pair 8 (within-family, Mistral / non-Groq)",
        tests="Within-family arm on a non-Groq stack: both cheap models are "
              "Ministral series (same lab/lineage). Independent replicate of "
              "the pair7 within-family result without the Groq-hosting "
              "confound. NB size gap (3B vs 8B) is wider than the report's "
              "2-3pp capability-match ideal -- check Always-Cheap-A vs -B.",
        cheap_a=ModelSpec("mistral", "ministral-3b-latest"),
        cheap_b=ModelSpec("mistral", "ministral-8b-latest"),
        frontier=ModelSpec("mistral", "mistral-medium-latest"),
        benchmarks={"arc": 150, "truthfulqa": 150, "mmlu_pro": 150, "mmlu": 250,
                    "openbookqa": 200, "commonsenseqa": 200, "gsm8k": 150},
        notes="Wave-3 pilot N. Within-family (Mistral), non-Groq. Matched to "
              "pair9 on frontier + benchmark set.",
    ),

    "pair9": PairConfig(
        pair_id="pair9",
        label="Pair 9 (cross-family, Mistral + Gemini / non-Groq)",
        tests="Cross-family / cross-provider arm on a non-Groq stack "
              "(Mistral cheap-A + Gemini cheap-B). Direct counterpart to "
              "pair8; closest available echo of the original paper's "
              "small-model + Gemini-Flash-Lite design.",
        cheap_a=ModelSpec("mistral", "ministral-8b-latest"),
        cheap_b=ModelSpec("gemini", "gemini-flash-lite-latest"),
        frontier=ModelSpec("mistral", "mistral-medium-latest"),
        benchmarks={"arc": 150, "truthfulqa": 150, "mmlu_pro": 150, "mmlu": 250,
                    "openbookqa": 200, "commonsenseqa": 200, "gsm8k": 150},
        notes="Wave-3 pilot N. Cross-family, non-Groq. Gemini free tier is "
              "~500 requests/day -- 3 benchmarks x ~150 is close to that "
              "ceiling; resume next day if it 429s on the last run.",
    ),

    # -----------------------------------------------------------------------
    # Wave-4 pairs (added 2026-09-04): two more all-Groq cross-family pairs
    # the solo sweep flagged as capability-matched to ~1pp. Together with
    # pair7 (qwen within-family) and pair3 (gpt-oss within-family) they give a
    # 4-pair, all-Groq, all-matched field to pick the practitioner default
    # from -- head to head on the same benchmarks, same frontier.
    # -----------------------------------------------------------------------
    "pair10": PairConfig(
        pair_id="pair10",
        label="Pair 10 (cross-family, all-Groq: gpt-oss-20B + Qwen3.6-27B)",
        tests="Best-matched cross-family pair from the solo sweep (0.6pp mean "
              "solo gap). Different labs (OpenAI-oss vs Alibaba), both on Groq "
              "LPU. Candidate for the recommended practitioner default.",
        cheap_a=ModelSpec("groq", "openai/gpt-oss-20b"),
        cheap_b=ModelSpec("groq", "qwen/qwen3.6-27b"),
        frontier=ModelSpec("groq", "openai/gpt-oss-120b"),
        benchmarks={"arc": 150, "mmlu": 250, "mmlu_pro": 150, "truthfulqa": 150,
                    "openbookqa": 200, "commonsenseqa": 200, "gsm8k": 150},
        notes="Wave-4 pilot N. All-Groq cross-family. Matched to pair7/pair11 "
              "on frontier + benchmark set.",
    ),

    "pair11": PairConfig(
        pair_id="pair11",
        label="Pair 11 (cross-family, all-Groq: gpt-oss-20B + Qwen3.8-27B)",
        tests="Second-best-matched cross-family pair (1.3pp mean solo gap). "
              "Same idea as pair10 but the newer Qwen minor version.",
        cheap_a=ModelSpec("groq", "openai/gpt-oss-20b"),
        cheap_b=ModelSpec("groq", "qwen/qwen3.8-27b"),
        frontier=ModelSpec("groq", "openai/gpt-oss-120b"),
        benchmarks={"arc": 150, "mmlu": 250, "mmlu_pro": 150, "truthfulqa": 150,
                    "openbookqa": 200, "commonsenseqa": 200, "gsm8k": 150},
        notes="Wave-4 pilot N. All-Groq cross-family. Matched to pair7/pair10 "
              "on frontier + benchmark set.",
    ),
}


def get_runnable_pairs() -> list[str]:
    """Pair ids that pcr_runner.py should actually execute."""
    return [pid for pid, cfg in PAIRS.items() if not cfg.is_historical]