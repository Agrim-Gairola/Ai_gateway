"""
cost_latency_analysis.py
========================
Track A (report's Medium-priority item): turn the per-call instrumentation
pcr_runner.py now records (`usage` block on every result row) into measured --
not estimated -- cost and latency numbers for PCR vs the two reference
policies:

  * always-frontier : every query goes straight to the frontier model
  * PCR             : two cheap calls in parallel; frontier only on disagree

Only checkpoints that carry the `usage` block are included (runs made after
the Track A instrumentation landed -- pair4 onward). Older pair2/pair3
checkpoints are skipped with a note.

Cost model
----------
Per-query token cost = sum of (tokens_in + tokens_out) actually billed on
each call made for that query. PCR pays cheap_a + cheap_b on every query and
frontier only on escalations; always-frontier pays one frontier call per
query (its input tokens are the same prompt; its output tokens we approximate
with the frontier output tokens actually observed on the escalated subset,
which is the best available estimate for the non-escalated queries too).

Dollar figures use per-provider $/1M-token rates in RATES below (public
list prices; PCR's own runs were on free tiers). Override with --rate.

Latency
-------
Wall-clock per query:
  * PCR         : max(cheap_a, cheap_b) + (frontier if escalated)   [parallel]
  * cascade ref : cheap_a + cheap_b + (frontier if escalated)       [serial]
  * always-front: frontier latency only
Reported as mean and p50/p90 over all queries in the pool.

Usage:
    python -m pcr_experiment.cost_latency_analysis
    python -m pcr_experiment.cost_latency_analysis --export
"""
import json
import argparse
import statistics as st
from pathlib import Path

from .pooled_analysis import CHECKPOINT_DIR, EXPORT_DIR, BENCHMARKS, discover_available

# Public list prices, USD per 1M tokens (blended in/out for a rough figure).
# These are only used to translate token counts into dollars; adjust freely.
RATES = {
    "groq:openai/gpt-oss-120b": 0.59,
    "groq:openai/gpt-oss-20b": 0.10,
    "groq:qwen/qwen3.8-27b": 0.20,
    "groq:qwen/qwen3.6-27b": 0.20,
    "groq:allam-2-7b": 0.05,
    "mistral:mistral-small-latest": 0.20,
    "gemini:gemini-3.5-flash-lite": 0.10,
    "gemini:gemini-flash-lite-latest": 0.10,
    "_default_cheap": 0.15,
    "_default_frontier": 0.60,
}


def rate_for(model_str, is_frontier):
    if model_str in RATES:
        return RATES[model_str]
    return RATES["_default_frontier" if is_frontier else "_default_cheap"]


def _toks(call):
    if not call:
        return 0
    return (call.get("tokens_in") or 0) + (call.get("tokens_out") or 0)


def analyse_cell(data):
    rows = [r for r in data["results"] if r.get("usage")]
    if not rows:
        return None
    n = len(rows)
    cheap_a_model = data.get("cheap_a", "")
    cheap_b_model = data.get("cheap_b", "")
    frontier_model = data.get("frontier", "")
    ra = rate_for(cheap_a_model, False)
    rb = rate_for(cheap_b_model, False)
    rf = rate_for(frontier_model, True)

    esc = sum(1 for r in rows if r["usage"].get("frontier"))
    esc_rate = esc / n

    # token totals
    tok_a = sum(_toks(r["usage"]["cheap_a"]) for r in rows)
    tok_b = sum(_toks(r["usage"]["cheap_b"]) for r in rows)
    tok_f = sum(_toks(r["usage"]["frontier"]) for r in rows if r["usage"].get("frontier"))
    # typical frontier tokens per call, to price always-frontier on ALL queries
    front_calls = [_toks(r["usage"]["frontier"]) for r in rows if r["usage"].get("frontier")]
    mean_front_tok = st.mean(front_calls) if front_calls else 0

    pcr_cost_per_q = (tok_a * ra + tok_b * rb + tok_f * rf) / 1e6 / n
    always_front_cost_per_q = (mean_front_tok * rf) / 1e6
    cheap_solo_cost_per_q = (tok_a * ra + tok_b * rb) / 1e6 / n  # consensus, no esc

    # latency
    def lat_pcr(r):
        u = r["usage"]
        base = u.get("parallel_latency_seconds")
        if base is None:
            base = max(u["cheap_a"]["latency_seconds"], u["cheap_b"]["latency_seconds"])
        f = u["frontier"]["latency_seconds"] if u.get("frontier") else 0.0
        return base + f

    def lat_cascade(r):
        u = r["usage"]
        f = u["frontier"]["latency_seconds"] if u.get("frontier") else 0.0
        return u["cheap_a"]["latency_seconds"] + u["cheap_b"]["latency_seconds"] + f

    def lat_front(r):
        u = r["usage"]
        if u.get("frontier"):
            return u["frontier"]["latency_seconds"]
        return mean_front_lat  # estimate for non-escalated queries

    front_lats = [r["usage"]["frontier"]["latency_seconds"] for r in rows if r["usage"].get("frontier")]
    mean_front_lat = st.mean(front_lats) if front_lats else 0.0

    pcr_l = [lat_pcr(r) for r in rows]
    cas_l = [lat_cascade(r) for r in rows]
    fro_l = [lat_front(r) for r in rows]

    def p(vs, q):
        vs = sorted(vs)
        return vs[min(len(vs) - 1, int(q * len(vs)))]

    return {
        "n": n,
        "escalation_rate": esc_rate,
        "cost_per_1k_queries_usd": {
            "pcr": pcr_cost_per_q * 1000,
            "always_frontier": always_front_cost_per_q * 1000,
            "consensus_no_escalation": cheap_solo_cost_per_q * 1000,
        },
        "cost_reduction_vs_always_frontier": (
            1 - pcr_cost_per_q / always_front_cost_per_q) if always_front_cost_per_q else None,
        "tokens_per_query": {
            "cheap_a": tok_a / n, "cheap_b": tok_b / n,
            "frontier_amortised": tok_f / n,
            "total_pcr": (tok_a + tok_b + tok_f) / n,
        },
        "latency_seconds": {
            "pcr_mean": st.mean(pcr_l), "pcr_p50": p(pcr_l, 0.5), "pcr_p90": p(pcr_l, 0.9),
            "cascade_mean": st.mean(cas_l), "cascade_p50": p(cas_l, 0.5), "cascade_p90": p(cas_l, 0.9),
            "always_frontier_mean": st.mean(fro_l),
            "pcr_vs_cascade_speedup": st.mean(cas_l) / st.mean(pcr_l) if st.mean(pcr_l) else None,
        },
        "models": {"cheap_a": cheap_a_model, "cheap_b": cheap_b_model, "frontier": frontier_model},
    }


def run(export=False):
    available = discover_available()
    results = {}
    skipped = []
    for pair_id, bm in available:
        path = CHECKPOINT_DIR / f"{pair_id}_{bm}.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cell = analyse_cell(data)
        if cell is None:
            skipped.append(f"{pair_id}/{bm}")
            continue
        results[f"{pair_id}_{bm}"] = cell

    print("=" * 92)
    print("COST & LATENCY  --  measured from per-call instrumentation (usage block)")
    print("=" * 92)
    if skipped:
        print(f"[skipped -- no instrumentation] {', '.join(sorted(skipped))}\n")
    hdr = (f"{'cell':<20}{'N':>5}{'esc%':>7}{'$/1k PCR':>10}{'$/1k AF':>9}"
           f"{'cost cut':>9}{'lat PCR':>9}{'lat AF':>8}{'vs casc':>9}")
    print(hdr)
    print("-" * 92)
    for k, c in results.items():
        cr = c["cost_reduction_vs_always_frontier"]
        L = c["latency_seconds"]
        print(f"{k:<20}{c['n']:>5}{c['escalation_rate']*100:>6.1f}%"
              f"{c['cost_per_1k_queries_usd']['pcr']:>10.3f}"
              f"{c['cost_per_1k_queries_usd']['always_frontier']:>9.3f}"
              f"{(cr*100 if cr is not None else 0):>8.1f}%"
              f"{L['pcr_mean']:>8.2f}s{L['always_frontier_mean']:>7.2f}s"
              f"{(L['pcr_vs_cascade_speedup'] or 0):>8.2f}x")
    print("-" * 92)
    print("$/1k AF = always-frontier.  cost cut = 1 - PCR/AF.  vs casc = serial-cascade")
    print("latency / PCR latency (parallel speedup on the cheap stage).")
    print("=" * 92)

    if export:
        EXPORT_DIR.mkdir(exist_ok=True)
        p = EXPORT_DIR / "cost_latency_analysis.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"cells": results, "skipped": sorted(skipped), "rates_usd_per_1m": RATES},
                      f, indent=2)
        print(f"\nExported -> {p}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()
    run(export=a.export)
