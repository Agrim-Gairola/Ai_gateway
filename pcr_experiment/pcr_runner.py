"""
pcr_runner.py
=============
Unified PCR (Parallel Cheap-model Routing) runner. Replaces
run_pcr_pair2.py -- instead of one script per pair, every pair is defined
once in pcr_config.py and this script executes any of them by id.

Checkpoint files are named "{pair_id}_{benchmark}.json" (e.g.
"pair3_mmlu.json"), a drop-in generalization of the old "pair2_mmlu.json"
naming -- so summarize_checkpoint.py works unmodified for pair2 and just
needs --pair pair3 / --pair pair4 / --pair pair5 for the new ones.

Requires (only install what you need for the pairs you're running):
    pip install groq mistralai google-generativeai datasets sentence-transformers python-dotenv

Env vars (set only what the pairs you're running actually need):
    GROQ_API_KEY, MISTRAL_API_KEY, GEMINI_API_KEY

Usage:
    python pcr_runner.py --pair pair3 --benchmark mmlu
    python pcr_runner.py --pair pair3 --benchmark mmlu --n 50   # override config N (e.g. smoke test)
    python pcr_runner.py --pair pair4 --benchmark gsm8k
    python pcr_runner.py --pair pair5 --benchmark mmlu

Checkpoints every CHECKPOINT_EVERY questions; rerun the same command to
resume after a rate-limit failure or interruption.

IMPORTANT -- things you should verify before a real run, since I can't hit
live APIs from here to confirm current behavior:
  - Gemini parameter names (thinking_budget / generation_config keys) below
    are my best understanding as of my knowledge cutoff, not a live check.
    Your account has already found docs/blogs stale on model availability
    once (qwen3-32b) -- do the same live-check for Gemini 3.1 Pro's exact
    SDK method/params before a full run, and adjust call_gemini() if needed.
  - openai/gpt-oss-safeguard-20b (Pair 3 cheap_b) -- confirm it's live on
    your Groq account (`GET /v1/models`) the same way you did for the
    cheap_a swaps in Pair 2; it's a safety-tuned variant and may have
    different refusal behavior on some MMLU questions worth a note in the
    paper if it triggers.
"""

import os
import re
import json
import time
import asyncio
import argparse
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[warn] python-dotenv not installed; relying on real env vars only. "
          "Run: pip install python-dotenv")

from .pcr_config import PAIRS, ModelSpec, get_runnable_pairs

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHECKPOINT_EVERY = 50
# Anchor to this file's own directory, not the caller's cwd. Running as
# `python -m pcr_experiment.pcr_runner` from D:\files would otherwise create
# D:\files\checkpoints instead of pcr_experiment\checkpoints, silently
# diverging from where pooled_analysis.py / summarize_checkpoint.py look.
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

INTER_QUERY_SLEEP_SECONDS = 4.5   # matches original paper's throughput (~13 rpm);
                                   # raise this per-pair if a provider's free tier
                                   # is tighter than Groq/Mistral's
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2.0

# Cheap models need only a short answer budget for MMLU/ARC; GSM8K needs a
# generous budget because allam-2-7b and Mistral-Small write full
# step-by-step solutions regardless of "answer with just the final number"
# (this bit us on Pair 2 -- a short cap truncates mid-calculation and the
# extraction regex grabs an intermediate number instead of the real answer).
# New MC benchmarks (truthfulqa, mmlu_pro, openbookqa, commonsenseqa) get a
# 256-token cheap budget. Measured on 2026-09-03: gpt-oss-20b, even with
# reasoning_effort="low", spends ~200 hidden-reasoning tokens before emitting
# a letter -- at 64 or 160 tokens it truncates on most questions and burns the
# expensive 3x-retry path; at 256 it answers first try. Non-reasoning cheap
# models (mistral-small) and reasoning_effort="none" ones (qwen3) emit ~2
# tokens regardless, so the wider budget costs them nothing.
CHEAP_MAX_TOKENS = {
    # mmlu bumped 20 -> 256 for the pair6/pair7 runs: pair6's cheap_a is
    # gpt-oss-20b, which truncates at 20 tokens and burns the 3x-retry path on
    # nearly every question (this is the same effect that hit pair3/mmlu).
    # qwen (pair7) emits ~2 tokens regardless, so the wider budget is free for it.
    # Historical pair2/pair3 mmlu cells were collected at 20/40 and are not re-run.
    "mmlu": 256, "gsm8k": 300, "arc": 150,
    "truthfulqa": 256, "mmlu_pro": 256, "openbookqa": 256, "commonsenseqa": 256,
}
FRONTIER_MAX_TOKENS = {
    "mmlu": 400, "gsm8k": 600, "arc": 500,
    "truthfulqa": 400, "mmlu_pro": 500, "openbookqa": 400, "commonsenseqa": 400,
}

# ---------------------------------------------------------------------------
# Free-tier budget tracking (informational, not enforced beyond the paid-tier
# guard below). Numbers are approximate daily caps as of Aug 2026 -- verify
# against each provider's live console before trusting them for a real run;
# these change without notice and vary by account.
# ---------------------------------------------------------------------------
FREE_TIER_DAILY_CAPS = {
    # (provider, model) -> (requests_per_day, tokens_per_day) ; None = unknown/not the binding limit
    ("groq", "allam-2-7b"): (None, None),                 # not in the commonly-published table; watch console
    ("groq", "openai/gpt-oss-20b"): (1000, 200_000),
    ("groq", "openai/gpt-oss-safeguard-20b"): (1000, 200_000),
    ("groq", "openai/gpt-oss-120b"): (1000, 200_000),
    ("gemini", "gemini-2.5-flash-lite"): (1000, None),    # RPD-bound, not TPD-bound
    ("gemini", "gemini-3-flash"): (1500, None),
    # mistral: Experiment tier caps at ~1B tokens/MONTH, not day -- not
    # tracked here since a single pilot run won't come close to that; the
    # binding constraint in practice is its low RPM, handled by
    # INTER_QUERY_SLEEP_SECONDS below, not a daily counter.
}

_usage_tracker = {}  # (provider, model) -> {"requests": int, "tokens": int}


def _track_usage(provider: str, model: str, tokens: int):
    key = (provider, model)
    stats = _usage_tracker.setdefault(key, {"requests": 0, "tokens": 0})
    stats["requests"] += 1
    stats["tokens"] += tokens

    cap = FREE_TIER_DAILY_CAPS.get(key)
    if cap:
        rpd_cap, tpd_cap = cap
        if rpd_cap and stats["requests"] in (int(rpd_cap * 0.8), rpd_cap):
            pct = stats["requests"] / rpd_cap
            print(f"  [budget] {provider}:{model} at {stats['requests']}/{rpd_cap} "
                  f"requests today ({pct:.0%} of known free-tier daily cap)")
        if tpd_cap and stats["tokens"] >= int(tpd_cap * 0.8) and stats["tokens"] - tokens < int(tpd_cap * 0.8):
            print(f"  [budget] {provider}:{model} at {stats['tokens']}/{tpd_cap} "
                  f"tokens today (80%+ of known free-tier daily cap -- expect "
                  f"429s soon, this just blocks, it will not bill you)")


def print_usage_summary():
    if not _usage_tracker:
        return
    print("\n" + "-" * 60)
    print("SESSION USAGE (this process only, not a full-day total)")
    print("-" * 60)
    for (provider, model), stats in _usage_tracker.items():
        print(f"  {provider}:{model:<40} {stats['requests']:>5} req  {stats['tokens']:>8} tok (est.)")
    print("-" * 60)

# ---------------------------------------------------------------------------
# Lazy client init (so this file can be imported/tested without keys present)
# ---------------------------------------------------------------------------
_clients = {}


def get_groq_client():
    if "groq" not in _clients:
        from groq import AsyncGroq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Set GROQ_API_KEY environment variable.")
        _clients["groq"] = AsyncGroq(api_key=api_key)
    return _clients["groq"]


def get_mistral_client():
    if "mistral" not in _clients:
        try:
            from mistralai.client import Mistral
        except ImportError:
            from mistralai import Mistral
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("Set MISTRAL_API_KEY environment variable.")
        _clients["mistral"] = Mistral(api_key=api_key)
    return _clients["mistral"]


def get_gemini_client():
    if "gemini" not in _clients:
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY environment variable.")
        genai.configure(api_key=api_key)
        _clients["gemini"] = genai
    return _clients["gemini"]


# ---------------------------------------------------------------------------
# Provider call wrappers with retry/backoff -- each returns raw text
# ---------------------------------------------------------------------------
class CallResult:
    """Return type for every provider call -- carries the answer text plus
    the instrumentation (real token usage where the API reports it, wall-clock
    latency always) that Track A's cost/latency analysis needs."""
    __slots__ = ("text", "tokens_in", "tokens_out", "latency_seconds", "tokens_are_estimated")

    def __init__(self, text, tokens_in, tokens_out, latency_seconds, tokens_are_estimated=False):
        self.text = text
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.latency_seconds = latency_seconds
        self.tokens_are_estimated = tokens_are_estimated

    def to_dict(self):
        return {
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "latency_seconds": round(self.latency_seconds, 3),
            "tokens_estimated": self.tokens_are_estimated,
        }


def _estimate_tokens(text: str) -> int:
    """Fallback when a provider doesn't report usage: ~4 chars/token, the
    same heuristic used for the retroactive estimate on old checkpoints."""
    return max(1, len(text) // 4)


async def call_groq(model: str, prompt: str, max_tokens: int) -> CallResult:
    client = get_groq_client()
    extra_kwargs = {}
    if "gpt-oss" in model:
        # Reasoning model: force minimal reasoning effort so short token
        # budgets aren't consumed entirely by hidden reasoning before any
        # answer text appears (this is what caused the empty-content bug on
        # Pair 2's MMLU run).
        extra_kwargs["reasoning_effort"] = "low"
        extra_kwargs["reasoning_format"] = "parsed"
    elif "qwen3" in model:
        extra_kwargs["reasoning_effort"] = "none"

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
                **extra_kwargs,
            )
            content = resp.choices[0].message.content
            finish_reason = resp.choices[0].finish_reason
            if not content and finish_reason == "length":
                print(f"  [groq:{model}] empty content (finish_reason=length); "
                      f"retrying with 3x max_tokens")
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens * 3,
                    temperature=0.0,
                    **extra_kwargs,
                )
                content = resp.choices[0].message.content
            latency = time.perf_counter() - t0
            text = (content or "").strip()

            usage = getattr(resp, "usage", None)
            if usage is not None:
                tin = getattr(usage, "prompt_tokens", None)
                tout = getattr(usage, "completion_tokens", None)
                estimated = tin is None or tout is None
                tin = tin if tin is not None else _estimate_tokens(prompt)
                tout = tout if tout is not None else _estimate_tokens(text)
            else:
                tin, tout, estimated = _estimate_tokens(prompt), _estimate_tokens(text), True

            _track_usage("groq", model, tin + tout)
            return CallResult(text, tin, tout, latency, estimated)
        except Exception as e:
            wait = RETRY_BACKOFF_BASE ** attempt
            print(f"  [groq:{model}] error ({e}); retrying in {wait:.1f}s")
            await asyncio.sleep(wait)
    raise RuntimeError(f"Groq call failed after {MAX_RETRIES} retries: {model}")


async def call_mistral(model: str, prompt: str, max_tokens: int) -> CallResult:
    client = get_mistral_client()
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            resp = await client.chat.complete_async(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            latency = time.perf_counter() - t0
            text = resp.choices[0].message.content.strip()

            usage = getattr(resp, "usage", None)
            if usage is not None:
                tin = getattr(usage, "prompt_tokens", None)
                tout = getattr(usage, "completion_tokens", None)
                estimated = tin is None or tout is None
                tin = tin if tin is not None else _estimate_tokens(prompt)
                tout = tout if tout is not None else _estimate_tokens(text)
            else:
                tin, tout, estimated = _estimate_tokens(prompt), _estimate_tokens(text), True

            _track_usage("mistral", model, tin + tout)
            return CallResult(text, tin, tout, latency, estimated)
        except Exception as e:
            wait = RETRY_BACKOFF_BASE ** attempt
            print(f"  [mistral:{model}] error ({e}); retrying in {wait:.1f}s")
            await asyncio.sleep(wait)
    raise RuntimeError(f"Mistral call failed after {MAX_RETRIES} retries: {model}")


async def call_gemini(model: str, prompt: str, max_tokens: int) -> CallResult:
    """
    NOTE: if this model is a "thinking" variant, it can burn its token
    budget on hidden thinking the way gpt-oss did on Groq -- watch for
    empty/truncated output, and set a thinking_budget in generation_config
    if the live SDK supports it for this model (verify the param name
    against current docs; don't trust this comment's naming blindly).
    """
    genai = get_gemini_client()
    for attempt in range(MAX_RETRIES):
        try:
            gen_model = genai.GenerativeModel(model)
            t0 = time.perf_counter()
            resp = await gen_model.generate_content_async(
                prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": 0.0,
                },
            )
            text = (resp.text or "").strip() if hasattr(resp, "text") else ""
            if not text:
                print(f"  [gemini:{model}] empty response; retrying with 3x max_tokens")
                gen_model = genai.GenerativeModel(model)
                resp = await gen_model.generate_content_async(
                    prompt,
                    generation_config={
                        "max_output_tokens": max_tokens * 3,
                        "temperature": 0.0,
                    },
                )
                text = (resp.text or "").strip() if hasattr(resp, "text") else ""
            latency = time.perf_counter() - t0

            usage = getattr(resp, "usage_metadata", None)
            if usage is not None:
                tin = getattr(usage, "prompt_token_count", None)
                tout = getattr(usage, "candidates_token_count", None)
                estimated = tin is None or tout is None
                tin = tin if tin is not None else _estimate_tokens(prompt)
                tout = tout if tout is not None else _estimate_tokens(text)
            else:
                tin, tout, estimated = _estimate_tokens(prompt), _estimate_tokens(text), True

            _track_usage("gemini", model, tin + tout)
            return CallResult(text, tin, tout, latency, estimated)
        except Exception as e:
            wait = RETRY_BACKOFF_BASE ** attempt
            print(f"  [gemini:{model}] error ({e}); retrying in {wait:.1f}s")
            await asyncio.sleep(wait)
    raise RuntimeError(f"Gemini call failed after {MAX_RETRIES} retries: {model}")


PROVIDER_CALLERS = {
    "groq": call_groq,
    "mistral": call_mistral,
    "gemini": call_gemini,
}


async def call_model(spec: ModelSpec, prompt: str, max_tokens: int) -> CallResult:
    caller = PROVIDER_CALLERS[spec.provider]
    return await caller(spec.model, prompt, max_tokens)


# ---------------------------------------------------------------------------
# Benchmark loaders (unchanged from run_pcr_pair2.py)
# ---------------------------------------------------------------------------
def load_mmlu(n=1000, seed=42):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    items = []
    for row in ds:
        choices = row["choices"]
        letters = ["A", "B", "C", "D"]
        q_text = row["question"] + "\n" + "\n".join(
            f"{l}. {c}" for l, c in zip(letters, choices)
        )
        items.append({
            "prompt": f"Answer with just the letter (A, B, C, or D):\n{q_text}",
            "gold": letters[row["answer"]],
            "domain": row["subject"],
        })
    return items


def load_gsm8k(n=500, seed=42):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    items = []
    for row in ds:
        gold_numeric = row["answer"].split("####")[-1].strip().replace(",", "")
        items.append({
            "prompt": f"Solve. Answer with just the final number:\n{row['question']}",
            "gold": gold_numeric,
            "domain": "gsm8k",
        })
    return items


def load_arc_challenge(n=300, seed=42):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    items = []
    for row in ds:
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        q_text = row["question"] + "\n" + "\n".join(
            f"{l}. {t}" for l, t in zip(labels, texts)
        )
        items.append({
            "prompt": f"Answer with just the letter:\n{q_text}",
            "gold": row["answerKey"],
            "domain": "arc_challenge",
        })
    return items


def load_truthfulqa(n=150, seed=42):
    """TruthfulQA MC1 -- adversarial questions written to elicit common human
    misconceptions; exactly one option is true (label == 1). We keep the true
    option plus up to 5 sampled distractors (<=6 total) and shuffle with a
    per-question seed so the answer isn't fixed at 'A'. This is the sharpest
    probe for coordinated hallucination: when two independent cheap models
    *agree* on a wrong option here, they are agreeing on a shared false belief
    absorbed from overlapping pre-training text -- exactly PCR's C-Hall thesis.
    """
    import random
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    letters = ["A", "B", "C", "D", "E", "F"]
    items = []
    for qi, row in enumerate(ds):
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        correct = next(c for c, l in zip(choices, labels) if l == 1)
        wrong = [c for c, l in zip(choices, labels) if l != 1]
        rng = random.Random(seed + qi)
        rng.shuffle(wrong)
        opts = wrong[:5] + [correct]
        rng.shuffle(opts)
        gold = letters[opts.index(correct)]
        q_text = row["question"] + "\n" + "\n".join(
            f"{l}. {c}" for l, c in zip(letters, opts)
        )
        items.append({
            "prompt": f"Answer with just the letter:\n{q_text}",
            "gold": gold,
            "domain": "truthfulqa",
        })
    return items


def load_mmlu_pro(n=150, seed=42):
    """MMLU-Pro -- harder MMLU successor: up to 10 options (A-J), heavier
    reasoning, fewer trivially-retrievable answers. Keeps the dataset's own
    `category` as the domain so the per-domain C-Hall breakdown still works."""
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    items = []
    for row in ds:
        opts = row["options"]
        q_text = row["question"] + "\n" + "\n".join(
            f"{l}. {c}" for l, c in zip(letters, opts)
        )
        items.append({
            "prompt": f"Answer with just the letter:\n{q_text}",
            "gold": row["answer"],
            "domain": row["category"],
        })
    return items


def load_openbookqa(n=200, seed=42):
    """OpenBookQA -- elementary-science MC (A-D) needing a fact plus a bit of
    reasoning. Broadens the 'knowledge/science' task family beyond MMLU/ARC."""
    from datasets import load_dataset
    ds = load_dataset("allenai/openbookqa", "main", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    items = []
    for row in ds:
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        q_text = row["question_stem"] + "\n" + "\n".join(
            f"{l}. {t}" for l, t in zip(labels, texts)
        )
        items.append({
            "prompt": f"Answer with just the letter:\n{q_text}",
            "gold": row["answerKey"],
            "domain": "openbookqa",
        })
    return items


def load_commonsenseqa(n=200, seed=42):
    """CommonsenseQA -- 5-option (A-E) commonsense reasoning; validation split
    (test has no public labels)."""
    from datasets import load_dataset
    ds = load_dataset("tau/commonsense_qa", split="validation")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    items = []
    for row in ds:
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        q_text = row["question"] + "\n" + "\n".join(
            f"{l}. {t}" for l, t in zip(labels, texts)
        )
        items.append({
            "prompt": f"Answer with just the letter:\n{q_text}",
            "gold": row["answerKey"],
            "domain": "commonsenseqa",
        })
    return items


LOADERS = {
    "mmlu": load_mmlu, "gsm8k": load_gsm8k, "arc": load_arc_challenge,
    "truthfulqa": load_truthfulqa, "mmlu_pro": load_mmlu_pro,
    "openbookqa": load_openbookqa, "commonsenseqa": load_commonsenseqa,
}


# ---------------------------------------------------------------------------
# Answer extraction (unchanged)
# ---------------------------------------------------------------------------
def extract_letter(text: str):
    """Pull a single choice letter (A-J, to cover MMLU-Pro's 10 options) from a
    model response. Ordered from most to least reliable so a verbose answer
    still resolves: (1) the whole response is just a letter, (2) it starts with
    'B)' / 'B.' / 'B:', (3) '... answer is B', (4) last standalone A-J token."""
    if not text:
        return None
    t = text.strip().upper()
    m = re.match(r"^\(?\*?\s*([A-J])\s*[\).:,\-\]]*\s*$", t)
    if m:
        return m.group(1)
    m = re.match(r"^\(?\*?\s*([A-J])\s*[\).:,\-\]]", t)
    if m:
        return m.group(1)
    m = re.search(r"ANSWER\s*(?:IS)?\s*:?\s*\(?([A-J])\b", t)
    if m:
        return m.group(1)
    hits = re.findall(r"\b([A-J])\b", t)
    return hits[-1] if hits else None


def extract_number(text: str):
    nums = re.findall(r"-?\d+\.\d+|-?\d+", text.replace(",", ""))
    if not nums:
        return None
    return nums[-1].rstrip(".")


EXTRACTORS = {
    "mmlu": extract_letter, "gsm8k": extract_number, "arc": extract_letter,
    "truthfulqa": extract_letter, "mmlu_pro": extract_letter,
    "openbookqa": extract_letter, "commonsenseqa": extract_letter,
}


# ---------------------------------------------------------------------------
# Core PCR loop -- now parametrized by PairConfig instead of hardcoded models
# ---------------------------------------------------------------------------
def _check_free_tier(cfg):
    """Refuse to run any pair containing a paid-tier model unless the caller
    explicitly opts in. This is a config-level check (cost_tier on each
    ModelSpec), not a live billing check -- keep pcr_config.py's cost_tier
    annotations current when providers change their free-tier policies."""
    paid_models = [
        (role, spec) for role, spec in
        (("cheap_a", cfg.cheap_a), ("cheap_b", cfg.cheap_b), ("frontier", cfg.frontier))
        if spec.cost_tier == "paid"
    ]
    return paid_models


async def run_benchmark(pair_id: str, benchmark: str, n: int = None, allow_paid: bool = False):
    if pair_id not in PAIRS:
        raise ValueError(f"Unknown pair_id '{pair_id}'. Known: {list(PAIRS)}")
    cfg = PAIRS[pair_id]
    if cfg.is_historical:
        raise ValueError(
            f"'{pair_id}' is marked historical in pcr_config.py and won't be "
            f"re-run (models may be deprecated). Use build_comparison_table.py "
            f"to pull its already-exported summary instead."
        )
    if benchmark not in cfg.benchmarks:
        raise ValueError(
            f"'{pair_id}' has no configured N for benchmark '{benchmark}' "
            f"(configured: {list(cfg.benchmarks)}). Check pcr_config.py -- "
            f"e.g. Pair 5 is MMLU-only by design."
        )

    paid_models = _check_free_tier(cfg)
    if paid_models and not allow_paid:
        details = ", ".join(f"{role}={spec.provider}:{spec.model}" for role, spec in paid_models)
        raise RuntimeError(
            f"'{pair_id}' includes a paid-tier model ({details}) and this run "
            f"was not explicitly opted into paid usage. Re-run with --allow-paid "
            f"if you actually want to spend money, or swap that model for a "
            f"free-tier one in pcr_config.py."
        )
    if paid_models and allow_paid:
        details = ", ".join(f"{role}={spec.provider}:{spec.model}" for role, spec in paid_models)
        print(f"[WARNING] --allow-paid set: this run WILL incur real cost via {details}")

    n = n or cfg.benchmarks[benchmark]
    extract = EXTRACTORS[benchmark]
    cheap_max_tokens = CHEAP_MAX_TOKENS[benchmark]
    frontier_max_tokens = FRONTIER_MAX_TOKENS[benchmark]

    items = LOADERS[benchmark](n=n)

    ckpt_path = CHECKPOINT_DIR / f"{pair_id}_{benchmark}.json"
    results = []
    start_idx = 0
    if ckpt_path.exists():
        with open(ckpt_path, encoding="utf-8") as f:
            saved = json.load(f)
            results = saved["results"]
            start_idx = len(results)
            print(f"Resuming {pair_id}/{benchmark} from question {start_idx}/{len(items)}")

    def _write_checkpoint():
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({
                "pair_id": pair_id,
                "benchmark": benchmark,
                "n_total": len(items),
                "cheap_a": f"{cfg.cheap_a.provider}:{cfg.cheap_a.model}",
                "cheap_b": f"{cfg.cheap_b.provider}:{cfg.cheap_b.model}",
                "frontier": f"{cfg.frontier.provider}:{cfg.frontier.model}",
                "results": results,
            }, f, indent=2)

    embedder = None
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"[warn] could not load sentence-transformers ({e}); similarity will be null")

    print(f"\n=== {cfg.label} -- {benchmark.upper()} (n={len(items)}) ===")
    print(f"cheap_a={cfg.cheap_a.provider}:{cfg.cheap_a.model}  "
          f"cheap_b={cfg.cheap_b.provider}:{cfg.cheap_b.model}  "
          f"frontier={cfg.frontier.provider}:{cfg.frontier.model}")
    if cfg.notes:
        print(f"[note] {cfg.notes}")

    try:
      for i in range(start_idx, len(items)):
        item = items[i]
        prompt = item["prompt"]

        call_a, call_b = await asyncio.gather(
            call_model(cfg.cheap_a, prompt, cheap_max_tokens),
            call_model(cfg.cheap_b, prompt, cheap_max_tokens),
        )
        resp_a, resp_b = call_a.text, call_b.text
        ans_a, ans_b = extract(resp_a), extract(resp_b)
        agree = (ans_a is not None) and (ans_a == ans_b)

        sim = None
        if embedder is not None:
            try:
                emb = embedder.encode([resp_a, resp_b])
                import numpy as np
                sim = float(
                    np.dot(emb[0], emb[1])
                    / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]) + 1e-8)
                )
            except Exception:
                pass

        frontier_ans = None
        frontier_resp = None
        call_frontier = None
        if not agree:
            call_frontier = await call_model(cfg.frontier, prompt, frontier_max_tokens)
            frontier_resp = call_frontier.text
            frontier_ans = extract(frontier_resp)
            final_ans = frontier_ans
        else:
            final_ans = ans_a

        correct = (final_ans is not None) and (
            str(final_ans).strip().upper() == str(item["gold"]).strip().upper()
        )

        # Wall-clock latency for this question: A and B ran in parallel, so
        # it's max(A, B), not the sum; frontier (if called) runs serially
        # after, so it's additive. This is what Track A's cost/latency
        # analysis needs -- see the report's flagged gap.
        parallel_latency = max(call_a.latency_seconds, call_b.latency_seconds)
        total_latency = parallel_latency + (call_frontier.latency_seconds if call_frontier else 0.0)

        results.append({
            "idx": i,
            "domain": item["domain"],
            "gold": item["gold"],
            "resp_a": resp_a, "resp_b": resp_b,
            "ans_a": ans_a, "ans_b": ans_b,
            "agree": agree,
            "similarity": sim,
            "escalated": not agree,
            "frontier_ans": frontier_ans,
            "final_ans": final_ans,
            "correct": correct,
            "usage": {
                "cheap_a": call_a.to_dict(),
                "cheap_b": call_b.to_dict(),
                "frontier": call_frontier.to_dict() if call_frontier else None,
                "parallel_latency_seconds": round(parallel_latency, 3),
                "total_latency_seconds": round(total_latency, 3),
            },
        })

        if (i + 1) % CHECKPOINT_EVERY == 0 or i == len(items) - 1:
            _write_checkpoint()
            print(f"[{pair_id}/{benchmark}] checkpointed {i+1}/{len(items)}")

        await asyncio.sleep(INTER_QUERY_SLEEP_SECONDS)
    except BaseException:
        # Whatever's in `results` was already paid for in real API calls --
        # flush it before the exception propagates, so a mid-run rate-limit
        # crash (or Ctrl-C) never discards questions between checkpoints.
        if len(results) > start_idx:
            _write_checkpoint()
            print(f"[{pair_id}/{benchmark}] emergency-checkpointed "
                  f"{len(results)}/{len(items)} before exit")
        raise

    print_quick_summary(pair_id, benchmark, results)
    print_usage_summary()
    return results


def print_quick_summary(pair_id, benchmark, results):
    n = len(results)
    if n == 0:
        return
    agreed = [r for r in results if r["agree"]]
    disagreed = [r for r in results if not r["agree"]]
    n_agree = len(agreed)
    n_correct_agreed = sum(r["correct"] for r in agreed)
    n_chall = n_agree - n_correct_agreed
    overall_acc = sum(r["correct"] for r in results) / n

    print("\n" + "=" * 60)
    print(f"QUICK SUMMARY -- {pair_id}/{benchmark}")
    print("=" * 60)
    print(f"N questions:               {n}")
    print(f"Agreement rate:            {n_agree}/{n} = {n_agree/n:.1%}")
    print(f"Escalation rate:           {len(disagreed)}/{n} = {len(disagreed)/n:.1%}")
    if n_agree:
        print(f"Accuracy when agreed:      {n_correct_agreed}/{n_agree} = {n_correct_agreed/n_agree:.1%}")
        print(f"Coordinated hallucination: {n_chall}/{n_agree} = {n_chall/n_agree:.1%}")
    print(f"Overall PCR accuracy:      {overall_acc:.1%}")
    print("=" * 60)
    print(f"For full stats + CIs + domain breakdown, run:")
    print(f"  python summarize_checkpoint.py --pair {pair_id} --benchmark {benchmark} --export\n")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=list(PAIRS.keys()), required=True,
                         help=f"Runnable pairs: {get_runnable_pairs()} "
                              f"(pair1/pair2 are historical, not re-runnable)")
    parser.add_argument("--benchmark", choices=list(LOADERS.keys()), required=True)
    parser.add_argument("--n", type=int, default=None,
                         help="Override the N configured in pcr_config.py (e.g. for a smoke test)")
    parser.add_argument("--allow-paid", action="store_true",
                         help="Explicitly opt into running a pair that includes a paid-tier "
                              "model (per cost_tier in pcr_config.py). Without this flag, "
                              "such pairs are refused before any API calls are made.")
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.pair, args.benchmark, args.n, allow_paid=args.allow_paid))