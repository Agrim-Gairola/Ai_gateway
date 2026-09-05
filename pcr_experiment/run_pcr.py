"""Unified PCR runner with robust checkpointing and provider-aware rate limiting.

Examples:
  python -m pcr_experiment.run_pcr --pair pair3 --benchmark mmlu --n 250
  python -m pcr_experiment.run_pcr --pair pair3 --benchmark gsm8k --n 150

Key properties:
- Saves after EVERY completed question.
- Writes checkpoints atomically, so a crash cannot leave half-written JSON.
- Resumes by question index and de-duplicates old checkpoints.
- Treats HTTP 429/rate-limit errors separately from ordinary failures.
- Honors Retry-After when the provider exposes it.
- Uses per-provider/model pacing to reduce repeated 429s.
- Never records a rate-limited frontier call as an experimental answer.
- Writes a .events.jsonl operational log for failures/retries.
"""

import os
import re
import json
import time
import asyncio
import argparse
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
CFG = ROOT / "configs" / "pairs.json"
CK = ROOT / "checkpoints"
CK.mkdir(exist_ok=True)

MAX_RETRIES = 8
BACKOFF_BASE = 5.0
BACKOFF_MAX = 180.0
QUESTION_SLEEP = 2.0

# Conservative pacing. The Mistral frontier is intentionally slower because
# the previous Pair-3 run hit HTTP 429 on mistral-large-latest.
MODEL_MIN_INTERVAL = {
    "groq": 1.0,
    "mistral": 3.0,
    "gemini": 1.5,
}

# Extra conservative pacing for known frontier models. This can be changed
# without changing the experiment logic.
MODEL_OVERRIDES = {
    ("mistral", "mistral-large-latest"): 10.0,
}

_clients = {}
_rate_locks = {}
_next_allowed = {}


class RateLimitError(RuntimeError):
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class ModelCallError(RuntimeError):
    pass


def event_log_path(pair: str, benchmark: str) -> Path:
    return CK / f"{pair}_{benchmark}.events.jsonl"


def log_event(pair: str, benchmark: str, event: dict):
    event = {"timestamp": time.time(), **event}
    with open(event_log_path(pair, benchmark), "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def client(provider):
    if provider in _clients:
        return _clients[provider]

    if provider == "groq":
        from groq import AsyncGroq
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY missing")
        _clients[provider] = AsyncGroq(api_key=key)

    elif provider == "mistral":
        try:
            from mistralai.client import Mistral
        except ImportError:
            from mistralai import Mistral
        key = os.getenv("MISTRAL_API_KEY")
        if not key:
            raise RuntimeError("MISTRAL_API_KEY missing")
        _clients[provider] = Mistral(api_key=key)

    elif provider == "gemini":
        from google import genai
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY missing")
        _clients[provider] = genai.Client(api_key=key)

    else:
        raise ValueError(provider)

    return _clients[provider]


def interval_for(provider: str, model: str) -> float:
    return MODEL_OVERRIDES.get(
        (provider, model),
        MODEL_MIN_INTERVAL.get(provider, 2.0),
    )


async def pace(provider: str, model: str):
    key = (provider, model)
    lock = _rate_locks.setdefault(key, asyncio.Lock())

    async with lock:
        now = time.monotonic()
        allowed = _next_allowed.get(key, now)
        if allowed > now:
            await asyncio.sleep(allowed - now)
        _next_allowed[key] = time.monotonic() + interval_for(provider, model)


def retry_after_from_exception(exc) -> Optional[float]:
    # SDK exceptions differ between versions. Inspect common locations and
    # fall back to parsing the string representation.
    candidates = []
    for obj in (exc, getattr(exc, "response", None), getattr(exc, "body", None)):
        if obj is None:
            continue
        for attr in ("retry_after", "retryAfter"):
            value = getattr(obj, attr, None)
            if value is not None:
                candidates.append(value)
        headers = getattr(obj, "headers", None)
        if headers:
            for key in ("retry-after", "Retry-After"):
                if key in headers:
                    candidates.append(headers[key])

    for value in candidates:
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            pass

    text = str(exc)
    m = re.search(r"retry[- _]?after[^0-9]*([0-9]+(?:\.[0-9]+)?)", text, re.I)
    if m:
        return float(m.group(1))

    return None


def is_rate_limit_exception(exc) -> bool:
    text = str(exc).lower()
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 429:
        return True
    return (
        "429" in text
        or "rate limit" in text
        or "rate_limit" in text
        or "too many requests" in text
        or "rate_limited" in text
    )


def groq_kwargs(model: str) -> dict:
    if "gpt-oss" in model:
        return {
            "reasoning_effort": "low",
            "reasoning_format": "parsed",
        }
    if "qwen3" in model:
        return {"reasoning_effort": "none"}
    return {}


async def call(provider, model, prompt, max_tokens, pair=None, benchmark=None, role=None):
    """Call one model with robust 429 handling.

    Returns (text, latency). Raises ModelCallError only after all retries are
    exhausted. A rate limit is never converted into an empty answer.
    """
    last_exc = None

    for attempt in range(MAX_RETRIES):
        await pace(provider, model)
        started = time.perf_counter()

        try:
            if provider == "groq":
                kw = groq_kwargs(model)
                r = await client(provider).chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0,
                    **kw,
                )
                text = r.choices[0].message.content or ""

                # gpt-oss can spend the initial budget on hidden reasoning.
                if not text and r.choices[0].finish_reason == "length":
                    await pace(provider, model)
                    r = await client(provider).chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens * 3,
                        temperature=0,
                        **kw,
                    )
                    text = r.choices[0].message.content or ""

            elif provider == "mistral":
                r = await client(provider).chat.complete_async(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0,
                )
                text = r.choices[0].message.content or ""

            elif provider == "gemini":
                c = client(provider)

                def invoke():
                    return c.models.generate_content(
                        model=model,
                        contents=prompt,
                        config={
                            "temperature": 0,
                            "max_output_tokens": max_tokens,
                        },
                    )

                r = await asyncio.to_thread(invoke)
                text = getattr(r, "text", "") or ""

            else:
                raise ValueError(provider)

            return text.strip(), time.perf_counter() - started

        except Exception as exc:
            last_exc = exc
            rate_limited = is_rate_limit_exception(exc)
            retry_after = retry_after_from_exception(exc)

            if rate_limited:
                # Honor server-provided Retry-After, otherwise use exponential
                # backoff with a generous floor for provider quota recovery.
                wait = retry_after if retry_after is not None else BACKOFF_BASE * (2 ** attempt)
                wait = min(max(wait, 5.0), BACKOFF_MAX)
                kind = "429/rate-limit"
            else:
                wait = min(BACKOFF_BASE * (2 ** attempt), BACKOFF_MAX)
                kind = "error"

            if pair and benchmark:
                log_event(pair, benchmark, {
                    "event": "retry",
                    "provider": provider,
                    "model": model,
                    "role": role,
                    "attempt": attempt + 1,
                    "kind": kind,
                    "error": str(exc),
                    "wait_seconds": wait,
                })

            print(
                f"[{provider}:{model}] {kind}; "
                f"attempt {attempt + 1}/{MAX_RETRIES}; "
                f"waiting {wait:.1f}s"
            )
            await asyncio.sleep(wait)

    raise ModelCallError(
        f"{provider}:{model} failed after {MAX_RETRIES} retries; "
        f"last error: {last_exc}"
    )


def load_items(benchmark, n, seed):
    from datasets import load_dataset

    if benchmark == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=seed)
        letters = "ABCD"
        out = []
        for x in ds.select(range(min(n, len(ds)))):
            out.append({
                "prompt": (
                    "Answer with just the letter (A, B, C, or D):\n"
                    + x["question"]
                    + "\n"
                    + "\n".join(
                        f"{l}. {c}" for l, c in zip(letters, x["choices"])
                    )
                ),
                "gold": letters[x["answer"]],
                "domain": x["subject"],
            })
        return out

    if benchmark == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=seed)
        out = []
        for x in ds.select(range(min(n, len(ds)))):
            out.append({
                "prompt": (
                    "Solve the problem. Give the final numerical answer clearly:\n"
                    + x["question"]
                ),
                "gold": x["answer"].split("####")[-1].strip().replace(",", ""),
                "domain": "gsm8k",
            })
        return out

    if benchmark == "arc":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test").shuffle(seed=seed)
        out = []
        for x in ds.select(range(min(n, len(ds)))):
            out.append({
                "prompt": (
                    "Answer with just the letter:\n"
                    + x["question"]
                    + "\n"
                    + "\n".join(
                        f"{l}. {t}"
                        for l, t in zip(x["choices"]["label"], x["choices"]["text"])
                    )
                ),
                "gold": x["answerKey"],
                "domain": "arc_challenge",
            })
        return out

    raise ValueError(benchmark)


def extract_letter(s):
    s = s or ""
    m = (
        re.search(r"(?:FINAL ANSWER|ANSWER)\s*[:\-]?\s*\**([A-D])\**", s.upper())
        or re.search(r"\b([A-D])\b", s.upper())
    )
    return m.group(1) if m else None


def extract_number(s):
    s = (s or "").replace(",", "")
    m = re.findall(
        r"(?:FINAL ANSWER|ANSWER)\s*[:\-]?\s*\$?\s*(-?\d+(?:\.\d+)?)",
        s,
        re.I,
    )
    if m:
        return m[-1].rstrip(".")
    m = re.findall(r"-?\d+\.\d+|-?\d+", s)
    return m[-1].rstrip(".") if m else None


def config(pair):
    with open(CFG, encoding="utf8") as f:
        d = json.load(f)["pairs"]
    if pair not in d:
        raise ValueError(f"Unknown pair {pair}; available: {list(d)}")
    return d[pair]


def atomic_json_write(path: Path, payload: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def checkpoint_payload(pair, benchmark, seed, n_total, results, status, started_at, error=None):
    return {
        "schema_version": 2,
        "pair": pair,
        "benchmark": benchmark,
        "seed": seed,
        "n_total": n_total,
        "completed": len(results),
        "status": status,
        "started_at": started_at,
        "updated_at": time.time(),
        "error": error,
        "results": results,
    }


def load_checkpoint(path, pair, benchmark, seed, n_total):
    if not path.exists():
        return [], None

    with open(path, encoding="utf8") as f:
        d = json.load(f)

    compatible = (
        d.get("pair") == pair
        and d.get("benchmark") == benchmark
        and d.get("seed", seed) == seed
    )

    if not compatible:
        backup = path.with_suffix(path.suffix + f".incompatible-{int(time.time())}")
        os.replace(path, backup)
        print(f"[checkpoint] incompatible checkpoint moved to {backup.name}")
        return [], None

    # Repair old checkpoints that may contain duplicate indices after a crash.
    by_idx = {}
    for row in d.get("results", []):
        if isinstance(row, dict) and "idx" in row:
            by_idx[int(row["idx"])] = row

    results = [by_idx[i] for i in sorted(by_idx) if i < n_total]
    started_at = d.get("started_at")

    print(
        f"[checkpoint] loaded {len(results)}/{n_total} completed "
        f"(status={d.get('status', 'legacy')})"
    )
    return results, started_at


async def run(pair, benchmark, n, seed):
    p = config(pair)
    defaults = {"mmlu": 1000, "gsm8k": 500, "arc": 300}
    n = n or defaults[benchmark]
    items = load_items(benchmark, n, seed)
    extract = extract_number if benchmark == "gsm8k" else extract_letter

    path = CK / f"{pair}_{benchmark}.json"
    results, started_at = load_checkpoint(path, pair, benchmark, seed, len(items))
    started_at = started_at or time.time()

    # Results are indexed by dataset position. This makes resumption robust.
    done = {int(r["idx"]): r for r in results}

    tokens = {"mmlu": 30, "gsm8k": 350, "arc": 150}
    ftokens = {"mmlu": 500, "gsm8k": 700, "arc": 500}

    print("\n" + "=" * 72)
    print(f"PCR {pair.upper()} / {benchmark.upper()}")
    print(f"A: {p['cheap_a']['provider']} / {p['cheap_a']['model']}")
    print(f"B: {p['cheap_b']['provider']} / {p['cheap_b']['model']}")
    print(f"F: {p['frontier']['provider']} / {p['frontier']['model']}")
    print(f"Progress: {len(done)}/{len(items)}")
    print("=" * 72)

    # Mark the checkpoint as running before any new calls.
    current = [done[i] for i in sorted(done)]
    atomic_json_write(
        path,
        checkpoint_payload(
            pair, benchmark, seed, len(items), current, "running", started_at
        ),
    )

    try:
        for i, x in enumerate(items):
            if i in done:
                continue

            # Cheap calls remain parallel, preserving PCR's intended mechanism.
            a, b = await asyncio.gather(
                call(
                    p["cheap_a"]["provider"],
                    p["cheap_a"]["model"],
                    x["prompt"],
                    tokens[benchmark],
                    pair, benchmark, "cheap_a",
                ),
                call(
                    p["cheap_b"]["provider"],
                    p["cheap_b"]["model"],
                    x["prompt"],
                    tokens[benchmark],
                    pair, benchmark, "cheap_b",
                ),
            )

            ra, la = a
            rb, lb = b
            aa, ab = extract(ra), extract(rb)
            agree = aa is not None and aa == ab

            fr = fa = fl = None
            if not agree:
                fr, fl = await call(
                    p["frontier"]["provider"],
                    p["frontier"]["model"],
                    x["prompt"],
                    ftokens[benchmark],
                    pair, benchmark, "frontier",
                )
                fa = extract(fr)
                final = fa
            else:
                final = aa

            gold = str(x["gold"]).strip().upper()
            correct = final is not None and str(final).strip().upper() == gold
            ca = aa is not None and str(aa).strip().upper() == gold
            cb = ab is not None and str(ab).strip().upper() == gold
            fc = None if fa is None else str(fa).strip().upper() == gold

            row = {
                "idx": i,
                "pair": pair,
                "benchmark": benchmark,
                "seed": seed,
                "domain": x["domain"],
                "gold": x["gold"],
                "cheap_a_model": p["cheap_a"]["model"],
                "cheap_a_provider": p["cheap_a"]["provider"],
                "cheap_b_model": p["cheap_b"]["model"],
                "cheap_b_provider": p["cheap_b"]["provider"],
                "frontier_model": p["frontier"]["model"],
                "frontier_provider": p["frontier"]["provider"],
                "resp_a": ra,
                "resp_b": rb,
                "ans_a": aa,
                "ans_b": ab,
                "correct_a": ca,
                "correct_b": cb,
                "agree": agree,
                "escalated": not agree,
                "frontier_resp": fr,
                "frontier_ans": fa,
                "frontier_correct": fc,
                "final_ans": final,
                "correct": correct,
                "latency_a": la,
                "latency_b": lb,
                "frontier_latency": fl,
            }

            done[i] = row
            results = [done[j] for j in sorted(done)]

            # CRITICAL: save after every completed question.
            atomic_json_write(
                path,
                checkpoint_payload(
                    pair,
                    benchmark,
                    seed,
                    len(items),
                    results,
                    "running",
                    started_at,
                ),
            )

            print(
                f"[{i + 1}/{len(items)}] A={aa} B={ab} "
                + ("AGREE" if agree else "ESCALATE")
                + f" final={final} "
                + ("OK" if correct else "WRONG")
            )

            await asyncio.sleep(QUESTION_SLEEP)

        # Final atomic checkpoint.
        results = [done[j] for j in sorted(done)]
        atomic_json_write(
            path,
            checkpoint_payload(
                pair,
                benchmark,
                seed,
                len(items),
                results,
                "complete",
                started_at,
            ),
        )
        print(f"[complete] {len(results)}/{len(items)}")

    except KeyboardInterrupt:
        results = [done[j] for j in sorted(done)]
        atomic_json_write(
            path,
            checkpoint_payload(
                pair, benchmark, seed, len(items), results, "paused", started_at
            ),
        )
        print(f"\n[paused] checkpoint saved at {len(results)}/{len(items)}")
        raise

    except Exception as exc:
        results = [done[j] for j in sorted(done)]
        atomic_json_write(
            path,
            checkpoint_payload(
                pair,
                benchmark,
                seed,
                len(items),
                results,
                "error",
                started_at,
                error=str(exc),
            ),
        )
        log_event(pair, benchmark, {
            "event": "run_error",
            "error": str(exc),
            "completed": len(results),
        })
        print(f"[error] checkpoint preserved at {len(results)}/{len(items)}")
        raise


def main():
    q = argparse.ArgumentParser()
    q.add_argument("--pair", required=True)
    q.add_argument("--benchmark", required=True, choices=["mmlu", "gsm8k", "arc"])
    q.add_argument("--n", type=int)
    q.add_argument("--seed", type=int, default=42)
    a = q.parse_args()
    asyncio.run(run(a.pair, a.benchmark, a.n, a.seed))


if __name__ == "__main__":
    main()
