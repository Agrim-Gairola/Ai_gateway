"""
Parallel Consensus Routing -- Second Model Pair Replication
=============================================================
Pair A (held constant from original paper): Llama-3.1-8B-Instant  (via Groq)
Pair B (NEW, replaces Gemini-2.5-Flash-Lite): Mistral-Small-Latest (via Mistral La Plateforme)
Frontier / escalation model: Llama-3.3-70B-Versatile (via Groq)

Benchmarks: MMLU (1000 q), GSM8K (500 q), ARC-Challenge (300 q)

Run this on a machine with network access to api.groq.com and api.mistral.ai.
Requires env vars: GROQ_API_KEY, MISTRAL_API_KEY

    pip install groq mistralai datasets sentence-transformers

    export GROQ_API_KEY=...
    export MISTRAL_API_KEY=...
    python run_pcr_pair2.py --benchmark mmlu --n 1000
    python run_pcr_pair2.py --benchmark gsm8k --n 500
    python run_pcr_pair2.py --benchmark arc --n 300

Checkpoints to JSON every CHECKPOINT_EVERY questions so a rate-limit failure
never loses progress -- rerun the same command and it resumes automatically.
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
    load_dotenv()  # reads .env in the current working directory
except ImportError:
    print("[warn] python-dotenv not installed; relying on real env vars only. "
          "Run: pip install python-dotenv")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHECKPOINT_EVERY = 50
CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Free-tier rate limits differ by provider -- adjust these if you hit 429s.
# Mistral free tier: conservatively ~1 req/sec. Groq free tier: ~30 RPM combined
# across two models called per question (cheap A + occasional frontier), so we
# throttle to the slower of the two providers to keep both models in sync.
INTER_QUERY_SLEEP_SECONDS = 4.5   # matches original paper's throughput (~13 rpm)
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2.0

MODEL_CHEAP_A = "allam-2-7b"                  # Groq -- plain (non-reasoning) small instruct model
MODEL_CHEAP_B = "mistral-small-latest"        # Mistral  <-- the new addition
MODEL_FRONTIER = "openai/gpt-oss-120b"        # Groq -- reasoning model, used only on escalation
# NOTE (2026-08-19): third model swap for cheap-A. Timeline: llama-3.1-8b-instant
# (deprecated) -> openai/gpt-oss-20b (hit 200k TPD quota due to hidden reasoning
# tokens even at reasoning_effort="low") -> qwen/qwen3-32b (also deprecated by
# the time we tried it, confirmed via live `GET /v1/models` against this
# account) -> allam-2-7b. Verified against the account's actual live model
# list (not docs/blog posts, which were inconsistent) before landing here.
# allam-2-7b is a plain instruct model with no reasoning-token overhead, so
# it should behave like the original llama-3.1-8b-instant in terms of terse
# output and token/quota usage.

# ---------------------------------------------------------------------------
# Lazy client init (so this file can be imported/tested without keys present)
# ---------------------------------------------------------------------------
_groq_client = None
_mistral_client = None


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import AsyncGroq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Set GROQ_API_KEY environment variable.")
        _groq_client = AsyncGroq(api_key=api_key)
    return _groq_client


def get_mistral_client():
    global _mistral_client
    if _mistral_client is None:
        # mistralai>=2.x restructured: Mistral is no longer exported at the
        # top-level `mistralai` package (which is now an empty namespace
        # package) -- it lives in `mistralai.client`. Fall back to the older
        # top-level import in case an older SDK version is installed.
        try:
            from mistralai.client import Mistral
        except ImportError:
            from mistralai import Mistral
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("Set MISTRAL_API_KEY environment variable.")
        _mistral_client = Mistral(api_key=api_key)
    return _mistral_client


# ---------------------------------------------------------------------------
# Model call wrappers with retry/backoff
# ---------------------------------------------------------------------------
async def call_groq(model: str, prompt: str, max_tokens: int = 12) -> str:
    client = get_groq_client()
    # gpt-oss models on Groq are reasoning models: they spend tokens on hidden
    # reasoning before the final answer. Force minimal reasoning effort so the
    # short max_tokens budgets (matching the original paper's prompt style)
    # don't get consumed before any answer text is produced.
    extra_kwargs = {}
    if "gpt-oss" in model:
        extra_kwargs["reasoning_effort"] = "low"
        # "parsed" keeps hidden reasoning tokens out of message.content, so
        # content contains only the final answer text. ("hidden" has a known
        # Groq bug where reasoning occasionally leaks into content anyway.)
        extra_kwargs["reasoning_format"] = "parsed"
    elif "qwen3" in model:
        # qwen3 models (unlike gpt-oss) can fully disable reasoning, giving
        # short, direct answers with no hidden-token overhead -- this is
        # what keeps token/day usage low enough for the free tier.
        extra_kwargs["reasoning_effort"] = "none"
    for attempt in range(MAX_RETRIES):
        try:
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
                # Reasoning consumed the entire token budget before any
                # answer was written. Retry once with a much larger budget
                # rather than silently recording an empty/None answer.
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
            return (content or "").strip()
        except Exception as e:
            wait = RETRY_BACKOFF_BASE ** attempt
            print(f"  [groq:{model}] error ({e}); retrying in {wait:.1f}s")
            await asyncio.sleep(wait)
    raise RuntimeError(f"Groq call failed after {MAX_RETRIES} retries: {model}")


async def call_mistral(model: str, prompt: str, max_tokens: int = 12) -> str:
    client = get_mistral_client()
    for attempt in range(MAX_RETRIES):
        try:
            # mistralai's async client
            resp = await client.chat.complete_async(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            wait = RETRY_BACKOFF_BASE ** attempt
            print(f"  [mistral:{model}] error ({e}); retrying in {wait:.1f}s")
            await asyncio.sleep(wait)
    raise RuntimeError(f"Mistral call failed after {MAX_RETRIES} retries: {model}")


# ---------------------------------------------------------------------------
# Benchmark loaders (mirrors the original paper's sampling: 1000 MMLU / 500
# GSM8K / 300 ARC-Challenge). Uses HuggingFace `datasets`.
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
    # HF migrated the bare "gsm8k" repo to the namespaced "openai/gsm8k" --
    # same config ("main") and fields, just a different repo id.
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
    # Same HF namespace migration as gsm8k: bare "ai2_arc" -> "allenai/ai2_arc".
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


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
def extract_letter(text: str):
    m = re.search(r"\b([A-D])\b", text.upper())
    return m.group(1) if m else None


def extract_number(text: str):
    # Match decimals before bare integers so "42." doesn't get parsed with a
    # dangling period; strip any trailing punctuation just in case.
    nums = re.findall(r"-?\d+\.\d+|-?\d+", text.replace(",", ""))
    if not nums:
        return None
    return nums[-1].rstrip(".")


# ---------------------------------------------------------------------------
# Core PCR loop
# ---------------------------------------------------------------------------
async def run_benchmark(benchmark: str, n: int):
    loaders = {"mmlu": load_mmlu, "gsm8k": load_gsm8k, "arc": load_arc_challenge}
    extractors = {"mmlu": extract_letter, "gsm8k": extract_number, "arc": extract_letter}
    # Cheap models (qwen3 w/ reasoning_effort="none", Mistral) need only a
    # short budget, matching the original paper's terse single-letter style.
    # Frontier (gpt-oss-120b) still reasons internally and needs much more
    # headroom, per Groq's own docs showing 200+ output tokens even on
    # simple questions at low reasoning effort.
    # Neither allam-2-7b nor Mistral-Small reliably follow "answer with just
    # the final number" on GSM8K -- they write full step-by-step solutions
    # regardless of the instruction. A short cap truncates mid-calculation
    # and the extraction regex then grabs an intermediate number instead of
    # the real final answer, corrupting both agreement and correctness.
    # Budget generously so the full worked solution can complete.
    cheap_max_tokens_map = {"mmlu": 20, "gsm8k": 300, "arc": 150}
    frontier_max_tokens_map = {"mmlu": 400, "gsm8k": 600, "arc": 500}

    assert benchmark in loaders, f"unknown benchmark {benchmark}"
    items = loaders[benchmark](n=n)
    extract = extractors[benchmark]
    max_tokens = cheap_max_tokens_map[benchmark]  # used for cheap A + B calls

    ckpt_path = CHECKPOINT_DIR / f"pair2_{benchmark}.json"
    results = []
    start_idx = 0
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            saved = json.load(f)
            results = saved["results"]
            start_idx = len(results)
            print(f"Resuming {benchmark} from question {start_idx}/{len(items)}")

    # Optional: semantic similarity, same embedding model as original paper
    embedder = None
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"[warn] could not load sentence-transformers ({e}); similarity will be null")

    for i in range(start_idx, len(items)):
        item = items[i]
        prompt = item["prompt"]

        # 1. Query both cheap models in parallel
        resp_a, resp_b = await asyncio.gather(
            call_groq(MODEL_CHEAP_A, prompt, max_tokens=max_tokens),
            call_mistral(MODEL_CHEAP_B, prompt, max_tokens=max_tokens),
        )
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
        if not agree:
            frontier_resp = await call_groq(
                MODEL_FRONTIER, prompt, max_tokens=frontier_max_tokens_map[benchmark]
            )
            frontier_ans = extract(frontier_resp)
            final_ans = frontier_ans
        else:
            final_ans = ans_a

        correct = (final_ans is not None) and (
            str(final_ans).strip().upper() == str(item["gold"]).strip().upper()
        )

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
        })

        if (i + 1) % CHECKPOINT_EVERY == 0 or i == len(items) - 1:
            with open(ckpt_path, "w") as f:
                json.dump({"benchmark": benchmark, "n_total": len(items), "results": results}, f, indent=2)
            print(f"[{benchmark}] checkpointed {i+1}/{len(items)}")

        await asyncio.sleep(INTER_QUERY_SLEEP_SECONDS)

    summarize(benchmark, results)
    return results


def summarize(benchmark, results):
    n = len(results)
    agreed = [r for r in results if r["agree"]]
    disagreed = [r for r in results if not r["agree"]]
    n_agree = len(agreed)
    n_correct_agreed = sum(r["correct"] for r in agreed)
    n_chall = n_agree - n_correct_agreed
    overall_acc = sum(r["correct"] for r in results) / n if n else 0

    print("\n" + "=" * 60)
    print(f"SUMMARY -- {benchmark.upper()}  (Llama-3.1-8B + Mistral-Small pair)")
    print("=" * 60)
    print(f"N questions:              {n}")
    print(f"Agreement rate:           {n_agree}/{n} = {n_agree/n:.1%}" if n else "n/a")
    print(f"Escalation rate:          {len(disagreed)}/{n} = {len(disagreed)/n:.1%}" if n else "n/a")
    if n_agree:
        print(f"Accuracy when agreed:     {n_correct_agreed}/{n_agree} = {n_correct_agreed/n_agree:.1%}")
        print(f"Coordinated hallucination:{n_chall}/{n_agree} = {n_chall/n_agree:.1%}")
    print(f"Overall PCR accuracy:     {overall_acc:.1%}")
    print("=" * 60 + "\n")
    print("Compare these numbers directly against your original")
    print("(Llama-3.1-8B, Gemini-2.5-Flash-Lite) pair results to test")
    print("whether the knowledge-vs-math C-Hall asymmetry replicates")
    print("across a different model pair.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["mmlu", "gsm8k", "arc"], required=True)
    parser.add_argument("--n", type=int, default=None,
                         help="Number of questions (default: 1000/500/300 matching original paper)")
    args = parser.parse_args()

    defaults = {"mmlu": 1000, "gsm8k": 500, "arc": 300}
    n = args.n or defaults[args.benchmark]

    asyncio.run(run_benchmark(args.benchmark, n))