"""
estimate_historical_usage.py
=============================
Retroactively estimates token usage for checkpoints written BEFORE the
usage-instrumentation update to pcr_runner.py (i.e. anything without a
"usage" field per result -- your existing pair2/pair3 runs).

Makes ZERO API calls and costs $0. It works by re-loading the exact same
benchmark dataset with the exact same seed (seed=42, same as pcr_runner.py's
loaders) so it can reconstruct the real prompt text for each question by
index, then estimates:
    input tokens  ~= len(prompt) / 4        (reconstructed exactly, not guessed)
    output tokens ~= len(response text) / 4  (from resp_a/resp_b/frontier_resp
                                               already stored in the checkpoint)

This is an approximation (OpenAI/Groq's own tokenizer isn't exactly 4
chars/token), but it's the same heuristic used as a fallback anywhere the
live APIs don't report usage, so historical and future numbers stay
comparable. Two things can't be reconstructed retroactively from old
checkpoints, since neither run_pcr_pair2.py nor early runs stored them:
  - Latency: no timestamps were ever recorded. Only available for runs
    made after the pcr_runner.py instrumentation update.
  - Frontier output tokens (exact): raw frontier response text was never
    saved, only the extracted answer letter/number. Falls back to an
    upper-bound estimate using the benchmark's max_tokens budget. This is
    also moot going forward -- new checkpoints capture real/estimated
    frontier tokens live in their "usage" field, so this script skips them.
Both gaps are flagged explicitly in this script's output.

Usage:
    python estimate_historical_usage.py                     # all checkpoints found
    python estimate_historical_usage.py --pair pair2         # just one pair
    python estimate_historical_usage.py --pair pair2 --benchmark mmlu
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

from pcr_runner import LOADERS, _estimate_tokens
from pcr_config import PAIRS

CHECKPOINT_DIR = Path("./checkpoints")
EXPORT_DIR = Path("./exports")


def estimate_checkpoint(ckpt_path: Path):
    with open(ckpt_path) as f:
        data = json.load(f)

    benchmark = data["benchmark"]
    n_total = data["n_total"]
    results = data["results"]

    if not results:
        return None

    # Skip files that already have real usage instrumentation -- no need to
    # estimate what we already measured.
    if "usage" in results[0]:
        return {"skipped": True, "reason": "already has real usage data"}

    # Reconstruct the exact prompts this run used (deterministic loader).
    items = LOADERS[benchmark](n=n_total)

    totals = {
        "cheap_a": {"tokens_in": 0, "tokens_out": 0, "calls": 0},
        "cheap_b": {"tokens_in": 0, "tokens_out": 0, "calls": 0},
        "frontier": {"tokens_in": 0, "tokens_out": 0, "calls": 0},
    }

    for r in results:
        idx = r["idx"]
        if idx >= len(items):
            continue  # shouldn't happen, but don't crash a whole report over one row
        prompt = items[idx]["prompt"]
        tin = _estimate_tokens(prompt)

        totals["cheap_a"]["tokens_in"] += tin
        totals["cheap_a"]["tokens_out"] += _estimate_tokens(r.get("resp_a", "") or "")
        totals["cheap_a"]["calls"] += 1

        totals["cheap_b"]["tokens_in"] += tin
        totals["cheap_b"]["tokens_out"] += _estimate_tokens(r.get("resp_b", "") or "")
        totals["cheap_b"]["calls"] += 1

        if r.get("escalated"):
            frontier_resp = r.get("frontier_resp")
            # NOTE: neither run_pcr_pair2.py nor pcr_runner.py store the raw
            # frontier response text in the checkpoint (only frontier_ans,
            # the extracted letter/number) -- this script only exists for
            # checkpoints from BEFORE the usage-instrumentation update, and
            # those never had frontier_resp text either way. Without it we
            # can't estimate frontier output tokens from real text; count
            # the call and use max_tokens as an upper-bound placeholder
            # instead of silently underestimating. This whole fallback is
            # moot for anything run after the instrumentation update, since
            # those checkpoints carry a "usage" field with real/estimated
            # tokens captured live and this script skips them entirely.
            totals["frontier"]["tokens_in"] += tin
            if frontier_resp:
                totals["frontier"]["tokens_out"] += _estimate_tokens(frontier_resp)
            # else: frontier_resp text wasn't stored historically -- leave
            # tokens_out as-is for now, backfilled with an upper-bound
            # estimate after the loop (see frontier_out_is_upper_bound below)
            totals["frontier"]["calls"] += 1

    frontier_out_is_upper_bound = totals["frontier"]["calls"] > 0 and any(
        r.get("escalated") and not r.get("frontier_resp") for r in results
    )
    if frontier_out_is_upper_bound:
        # frontier_resp wasn't stored historically -- fall back to the known
        # max_tokens budget as a (loose) upper bound per call instead of None
        from pcr_runner import FRONTIER_MAX_TOKENS
        upper_bound_per_call = FRONTIER_MAX_TOKENS.get(benchmark, 400)
        totals["frontier"]["tokens_out"] = totals["frontier"]["calls"] * upper_bound_per_call

    return {
        "skipped": False,
        "benchmark": benchmark,
        "n_total": n_total,
        "n_completed": len(results),
        "totals": totals,
        "frontier_output_is_upper_bound_estimate": frontier_out_is_upper_bound,
        "latency": None,  # never available retroactively -- see docstring
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default=None, help="Only process this pair id (default: all found)")
    parser.add_argument("--benchmark", default=None, choices=["mmlu", "gsm8k", "arc"])
    args = parser.parse_args()

    if not CHECKPOINT_DIR.exists():
        print(f"No checkpoint directory found at {CHECKPOINT_DIR}")
        return

    all_ckpts = sorted(CHECKPOINT_DIR.glob("*.json"))
    if args.pair:
        all_ckpts = [p for p in all_ckpts if p.stem.startswith(f"{args.pair}_")]
    if args.benchmark:
        all_ckpts = [p for p in all_ckpts if p.stem.endswith(f"_{args.benchmark}")]

    if not all_ckpts:
        print("No matching checkpoints found.")
        return

    report = {}
    print("\n" + "=" * 78)
    print("RETROACTIVE TOKEN ESTIMATE (no API calls made, $0 cost)")
    print("=" * 78)

    for ckpt_path in all_ckpts:
        pair_id = ckpt_path.stem.rsplit("_", 1)[0]
        result = estimate_checkpoint(ckpt_path)
        if result is None:
            continue
        if result.get("skipped"):
            print(f"\n{ckpt_path.name}: skipped ({result['reason']})")
            continue

        report[ckpt_path.stem] = result
        t = result["totals"]
        print(f"\n{pair_id} / {result['benchmark'].upper()} "
              f"({result['n_completed']}/{result['n_total']} questions)")
        print(f"  cheap_a:  {t['cheap_a']['calls']:>4} calls  "
              f"~{t['cheap_a']['tokens_in']:>7,} in  ~{t['cheap_a']['tokens_out']:>6,} out")
        print(f"  cheap_b:  {t['cheap_b']['calls']:>4} calls  "
              f"~{t['cheap_b']['tokens_in']:>7,} in  ~{t['cheap_b']['tokens_out']:>6,} out")
        frontier_note = " (output is an UPPER-BOUND estimate -- see note)" if result["frontier_output_is_upper_bound_estimate"] else ""
        print(f"  frontier: {t['frontier']['calls']:>4} calls  "
              f"~{t['frontier']['tokens_in']:>7,} in  ~{t['frontier']['tokens_out']:>6,} out{frontier_note}")
        print(f"  latency:  not available (checkpoint predates instrumentation)")

    if not report:
        print("\nNothing to report -- either no checkpoints matched, or all had real usage data already.")
        return

    EXPORT_DIR.mkdir(exist_ok=True)
    out_path = EXPORT_DIR / "historical_usage_estimate.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n\nWritten to {out_path}")
    print("\nReminder: these are $0 estimates from response text length, not billed usage --")
    print("your account's dashboard (Groq console / La Plateforme / AI Studio) is the source")
    print("of truth if you need exact numbers.")


if __name__ == "__main__":
    main()