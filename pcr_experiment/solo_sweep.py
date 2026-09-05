"""
solo_sweep.py
=============
Measure every candidate cheap model's *solo* accuracy + latency on a fixed
question set, so any pair's capability gap can be read off a table instead of
running full PCR for every combination. This is the groundwork for choosing:

  (a) capability-matched pairs for a clean family-effect experiment, and
  (b) a recommended "just works" pair for practitioners (accurate, fast, cheap).

One row per (model, benchmark): solo accuracy, mean/p90 latency, mean output
tokens. Cached to exports/solo_sweep.json; re-run only fills missing cells.

Usage:
    python -m pcr_experiment.solo_sweep
    python -m pcr_experiment.solo_sweep --n 80 --benchmarks arc mmlu_pro truthfulqa
"""
import json
import time
import asyncio
import argparse
import statistics as st
from pathlib import Path

from . import pcr_runner as R
from .pcr_config import ModelSpec
from .pooled_analysis import EXPORT_DIR

# provider, model id, short label, family tag (for later pair grouping)
CANDIDATES = [
    # frontier-tier first -- the Always-Frontier baseline is the priority cell
    ("groq",    "openai/gpt-oss-120b",      "gpt-oss-120b",    "gpt-oss-FRONTIER"),
    ("mistral", "mistral-medium-latest",    "mistral-medium",  "mistral-FRONTIER"),
    # the strong cheap trio (already swept)
    ("groq",   "openai/gpt-oss-20b",        "gpt-oss-20b",     "gpt-oss"),
    ("groq",   "qwen/qwen3.6-27b",          "qwen3.6-27b",     "qwen"),
    ("groq",   "qwen/qwen3.8-27b",          "qwen3.8-27b",     "qwen"),
    # expanded free-tier candidates (2026-09-04)
    ("groq",   "openai/gpt-oss-safeguard-20b", "gpt-oss-safeguard-20b", "gpt-oss"),
    ("mistral", "magistral-small-latest",   "magistral-small", "mistral"),
    ("mistral", "ministral-3b-latest",      "ministral-3b",    "mistral"),
    ("gemini",  "gemini-2.5-flash",         "gemini-2.5-flash", "gemini"),
    # weaker / slower, kept for completeness
    ("groq",   "allam-2-7b",                "allam-2-7b",      "allam"),
    ("gemini",  "gemini-flash-lite-latest", "gemini-flash-lite", "gemini"),
    ("gemini",  "gemini-flash-latest",      "gemini-flash",    "gemini"),
    ("mistral", "ministral-8b-latest",      "ministral-8b",    "mistral"),
    ("mistral", "ministral-14b-latest",     "ministral-14b",   "mistral"),
    ("mistral", "mistral-small-latest",     "mistral-small",   "mistral"),
]

SWEEP_PATH = EXPORT_DIR / "solo_sweep.json"


async def run_model_on(provider, model, benchmark, n, seed=42):
    items = R.LOADERS[benchmark](n=n, seed=seed)
    extract = R.EXTRACTORS[benchmark]
    budget = R.CHEAP_MAX_TOKENS[benchmark]
    correct = 0
    lats, touts, unparsed = [], [], 0
    for it in items:
        call = await R.call_model(ModelSpec(provider, model), it["prompt"], budget)
        ans = extract(call.text)
        if ans is None:
            unparsed += 1
        elif str(ans).strip().upper() == str(it["gold"]).strip().upper():
            correct += 1
        lats.append(call.latency_seconds)
        touts.append(call.tokens_out)
        await asyncio.sleep(1.5)  # gentle (Mistral RPM-safe); solo runs, no parallel pressure
    lats.sort()
    return {
        "n": len(items),
        "accuracy": correct / len(items),
        "unparsed": unparsed,
        "latency_mean": round(st.mean(lats), 3),
        "latency_p90": round(lats[min(len(lats) - 1, int(0.9 * len(lats)))], 3),
        "tokens_out_mean": round(st.mean(touts), 1),
    }


def load_cache():
    if SWEEP_PATH.exists():
        with open(SWEEP_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(d):
    EXPORT_DIR.mkdir(exist_ok=True)
    with open(SWEEP_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


async def main(n, benchmarks, retry_errors=False):
    cache = load_cache()
    for provider, model, label, family in CANDIDATES:
        cache.setdefault(label, {"provider": provider, "model": model, "family": family, "cells": {}})
        for bm in benchmarks:
            key = f"{bm}_n{n}"
            cached = cache[label]["cells"].get(key)
            is_error = isinstance(cached, dict) and "error" in cached
            if cached is not None and not (retry_errors and is_error):
                print(f"[skip] {label} / {bm} ({'error, --retry-errors to redo' if is_error else 'cached'})")
                continue
            print(f"[run ] {label} / {bm} (n={n}) ...", flush=True)
            try:
                res = await run_model_on(provider, model, bm, n)
                cache[label]["cells"][key] = res
                print(f"       acc={res['accuracy']:.1%}  lat={res['latency_mean']}s "
                      f"p90={res['latency_p90']}s  unparsed={res['unparsed']}")
            except Exception as e:
                print(f"       FAILED: {type(e).__name__}: {str(e)[:160]}")
                cache[label]["cells"][key] = {"error": str(e)[:200]}
            save_cache(cache)

    # ---- report ----
    print("\n" + "=" * 96)
    print(f"SOLO SWEEP  (n={n} per cell)")
    print("=" * 96)
    print(f"{'model':<20}{'family':<10}", end="")
    for bm in benchmarks:
        print(f"{bm[:11]:>13}", end="")
    print(f"{'lat(mean)':>11}{'lat(p90)':>10}")
    print("-" * 96)
    rows = []
    for label, d in cache.items():
        cells = d["cells"]
        accs = {bm: cells.get(f"{bm}_n{n}", {}).get("accuracy") for bm in benchmarks}
        lat = [cells.get(f"{bm}_n{n}", {}).get("latency_mean") for bm in benchmarks]
        latp90 = [cells.get(f"{bm}_n{n}", {}).get("latency_p90") for bm in benchmarks]
        lat = [x for x in lat if x is not None]
        latp90 = [x for x in latp90 if x is not None]
        print(f"{label:<20}{d['family']:<10}", end="")
        for bm in benchmarks:
            a = accs[bm]
            print(f"{(f'{a:.1%}' if a is not None else '--'):>13}", end="")
        print(f"{(f'{st.mean(lat):.2f}s' if lat else '--'):>11}"
              f"{(f'{max(latp90):.2f}s' if latp90 else '--'):>10}")
        mean_acc = st.mean([v for v in accs.values() if v is not None]) if any(accs.values()) else None
        rows.append((label, d["family"], accs, mean_acc, st.mean(lat) if lat else None))

    # ---- best-matched pairs ----
    print("\n" + "=" * 96)
    print("BEST CAPABILITY-MATCHED PAIRS  (mean |accuracy gap| across benchmarks, ascending)")
    print("=" * 96)
    pairs = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            la, fa, aa, ma, lta = rows[i]
            lb, fb, ab, mb, ltb = rows[j]
            gaps = [abs(aa[bm] - ab[bm]) for bm in benchmarks
                    if aa[bm] is not None and ab[bm] is not None]
            if not gaps:
                continue
            mean_gap = st.mean(gaps)
            cond = "within" if fa == fb else "cross"
            worst_solo = min(ma, mb) if (ma is not None and mb is not None) else None
            pairs.append((mean_gap, cond, la, lb, gaps, worst_solo,
                          (lta + ltb) / 2 if (lta and ltb) else None))
    pairs.sort()
    print(f"{'gap':>6}  {'cond':<7}{'pair':<38}{'per-bm gaps':<26}{'min mean acc':>13}{'~lat':>8}")
    print("-" * 96)
    for mean_gap, cond, la, lb, gaps, worst_solo, lat in pairs[:18]:
        gs = " ".join(f"{g*100:.0f}" for g in gaps)
        print(f"{mean_gap*100:5.1f}p  {cond:<7}{(la+' + '+lb):<38}{gs:<26}"
              f"{(f'{worst_solo:.1%}' if worst_solo else '--'):>13}"
              f"{(f'{lat:.2f}s' if lat else '--'):>8}")

    print(f"\nCache -> {SWEEP_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--benchmarks", nargs="+", default=["arc", "mmlu_pro", "truthfulqa"])
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-attempt cells previously cached as an error (e.g. after a quota reset)")
    a = ap.parse_args()
    asyncio.run(main(a.n, a.benchmarks, retry_errors=a.retry_errors))
