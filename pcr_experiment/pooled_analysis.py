"""
pooled_analysis.py
===================
Track A, Step 3 (report's own priority #1): a benchmark-balanced pooled
analysis across every completed pair/benchmark checkpoint you have.

"Benchmark-balanced" matters because your benchmarks have very different N
(MMLU pilots at 250, GSM8K at 150-800, ARC at 100-300) -- naively pooling
raw per-question counts across benchmarks would let whichever benchmark has
the most questions dominate the pooled accuracy/C-Hall numbers. Instead,
this computes each metric per (pair, benchmark) cell first, then averages
those cell-level rates *unweighted* across benchmarks to get a per-pair
pooled score, and again unweighted across pairs for a grand pooled score.
This is standard practice for combining heterogeneous-N benchmarks and
matches how the original paper's Table-style summaries treat benchmarks as
equally important regardless of question count.

WHAT THIS DOES NOT DO (yet, on purpose):
- It does NOT group pairs into "within-family" vs "cross-family" conditions
  for you. The report's Pair 3 (described as "a within-family Mistral
  configuration") does not match this codebase's current pair3 (gpt-oss on
  Groq, no Mistral involved) -- see the FAMILY_CONDITIONS dict below, which
  is deliberately left unfilled until that discrepancy is resolved. Filling
  it in incorrectly would silently produce a wrong Track B conclusion,
  which is worse than not having one yet.
- It does NOT compute always-frontier or random-50/50 baselines. Those need
  the escalated subset's frontier answers (already recorded) plus backfilled
  frontier answers for the *non-escalated* subset (not recorded -- requires
  new API calls). That's Track A Step 2's remaining piece, not this script.

Usage:
    python -m pcr_experiment.pooled_analysis
    python -m pcr_experiment.pooled_analysis --export
    python -m pcr_experiment.pooled_analysis --pairs pair2 pair3 pair4
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
EXPORT_DIR = Path(__file__).resolve().parent / "exports"

# Longer suffixes first so discover_available()'s endswith() check matches
# "pair6_mmlu_pro" as mmlu_pro, not as mmlu.
BENCHMARKS = ["mmlu_pro", "mmlu", "gsm8k", "arc", "truthfulqa", "openbookqa", "commonsenseqa"]

# Knowledge/retrieval-style benchmarks vs procedural/other -- used only for
# the optional task-family grouping in the printout, not for pooling math.
KNOWLEDGE_BENCHMARKS = {"mmlu", "mmlu_pro", "arc", "truthfulqa", "openbookqa", "commonsenseqa"}

# ---------------------------------------------------------------------------
# Family-condition tagging for Track B pooling -- INTENTIONALLY LEFT UNSET.
#
# The Detailed Report (20 Aug 2026) describes its two validated pairs as:
#   "Pair 2 is cross-family; Pair 3 is a within-family Mistral configuration"
# But this codebase's pcr_config.py currently defines pair3 as:
#   cheap_a=groq:openai/gpt-oss-20b, cheap_b=groq:openai/gpt-oss-safeguard-20b
#   -- no Mistral model anywhere in it.
#
# Either the pair numbering was reshuffled after the report was written, or
# "Pair 3" refers to two different experiments in the report vs. the current
# config. Confirm which before filling this in -- an incorrect mapping here
# would silently mislabel a cross-family pair as within-family (or vice
# versa) in every Track B number this script produces downstream.
#
# CONFIRMED 24-Aug-2026 by inspecting checkpoints/pair3_mmlu.json's first
# result row directly: the legacy (Mistral-era) pair3 used
# cheap_a=ministral-3b-latest, cheap_b=ministral-8b-latest,
# frontier=mistral-large-latest -- all one provider/family. The current
# gpt-oss-era pair3 (all Groq) is the same experimental role, just
# reimplemented after Mistral's rate limits forced a swap. Both eras are
# within_family; only the specific models changed, not the design intent.
FAMILY_CONDITIONS = {
    "pair2": "cross_family",
    "pair3": "within_family",
    "pair4": "cross_family",
    "pair5": "cross_family",    # reuses pair4's cheap layer
    # Focused-wave matched pair (added 2026-09-03), frontier held constant:
    "pair6": "cross_family",    # groq:gpt-oss-20b  +  mistral:mistral-small-latest
    "pair7": "within_family",   # groq:qwen3.8-27b  +  groq:qwen3.6-27b (same lineage)
    # Wave-3 non-Groq matched pair, frontier held constant (mistral-medium):
    "pair8": "within_family",   # mistral:ministral-3b  +  mistral:ministral-8b (same lineage)
    "pair9": "cross_family",    # mistral:ministral-8b  +  gemini:gemini-flash-lite-latest
    # Wave-4 all-Groq matched cross-family pairs:
    "pair10": "cross_family",   # groq:gpt-oss-20b  +  groq:qwen3.6-27b
    "pair11": "cross_family",   # groq:gpt-oss-20b  +  groq:qwen3.8-27b
}
# NOTE: pair3_gptoss_safeguard_mmlu.json was renamed to the canonical
# pair3_mmlu.json on 24-Aug-2026 (after archiving the legacy Mistral data
# as pair3_mistral_mmlu_legacy.json) so pcr_runner.py resumes it correctly
# going forward -- no separate tag needed for it anymore.

# ---------------------------------------------------------------------------
# Cells excluded from the family-condition grand pooling, because they fail
# the report's own matching criterion (pairs must be capability-matched
# within ~2-3pp before a disagreement/C-Hall comparison can be attributed to
# family relatedness rather than a skill gap).
#
# pair2/gsm8k: diagnosed 24-Aug-2026 via diagnose_pair2_gsm8k.py. 437/438
# disagreements are genuinely different extracted numbers (not an
# extraction bug -- only 1 was a same-value formatting artifact), and the
# disagreement rate is flat across the run (88.6/86.7/87.5% by third), so
# it's not an artifact of the documented mid-run model swap either. Reading
# the raw responses shows why: cheap_a (allam-2-7b) reliably shows full
# step-by-step work and gets many questions right; cheap_b
# (mistral-small-latest) frequently outputs a bare, unexplained number and
# is wrong more often. This is a real capability mismatch between the two
# cheap models on GSM8K specifically, not evidence about family
# relatedness -- including it in the cross_family pooled average would
# mix a skill-gap effect into what's supposed to be an isolated family
# comparison. Kept visible in the per-cell table below, just excluded from
# the grand family-condition means.
EXCLUDE_FROM_FAMILY_POOLING = {
    ("pair2", "gsm8k"): "capability mismatch (allam-2-7b vs mistral-small-latest on "
                         "GSM8K), not a family effect -- see diagnose_pair2_gsm8k.py output",
    # Wave-3 pairs turned out capability-mismatched on the hard benchmarks: the
    # two cheap models' solo accuracies differ by far more than the report's
    # +/-2-3pp matching criterion, so their (very high) C-Hall reflects a skill
    # gap, not family relatedness. Measured solo gaps (cheap_a - cheap_b):
    #   pair8 (ministral-3b vs ministral-8b): -16.7pp truthfulqa, -7.3pp mmlu_pro
    #   pair9 (ministral-8b vs gemini-flash): -24.0pp truthfulqa, -18.0pp mmlu_pro
    # The ARC cells (gaps -7.3 / -4.7pp) are closer and kept, with a caveat.
    ("pair8", "truthfulqa"): "capability mismatch (ministral-3b vs ministral-8b, -16.7pp solo)",
    ("pair8", "mmlu_pro"): "capability mismatch (ministral-3b vs ministral-8b, -7.3pp solo)",
    ("pair9", "truthfulqa"): "capability mismatch (ministral-8b vs gemini-flash-lite, -24.0pp solo)",
    ("pair9", "mmlu_pro"): "capability mismatch (ministral-8b vs gemini-flash-lite, -18.0pp solo)",
}


def load_checkpoint(pair_id: str, benchmark: str):
    path = CHECKPOINT_DIR / f"{pair_id}_{benchmark}.json"
    if not path.exists():
        return None
    # encoding="utf-8" is required, not optional: on Windows, open() defaults
    # to the OS locale (cp1252), which can't decode characters some model
    # responses contain (smart quotes, em-dashes, math symbols). Without
    # this, reading a perfectly valid UTF-8 checkpoint crashes here.
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def discover_available(pairs_filter=None):
    """Find every (pair, benchmark) checkpoint that actually exists on disk."""
    found = []
    if not CHECKPOINT_DIR.exists():
        return found
    for path in sorted(CHECKPOINT_DIR.glob("*.json")):
        stem = path.stem  # e.g. "pair3_gsm8k"
        for bm in BENCHMARKS:
            suffix = f"_{bm}"
            if stem.endswith(suffix):
                pair_id = stem[: -len(suffix)]
                if pairs_filter and pair_id not in pairs_filter:
                    continue
                found.append((pair_id, bm))
                break
    return found


def cell_stats(data: dict) -> dict:
    """Compute the core rates for one (pair, benchmark) checkpoint -- same
    definitions as summarize_checkpoint.py, kept consistent on purpose so
    numbers match between the two tools."""
    results = data["results"]
    n = len(results)
    n_planned = data.get("n_total", n)
    if n == 0:
        return {
            "n": 0, "n_planned": n_planned, "complete": False,
            "agreement_rate": None, "escalation_rate": None,
            "accuracy_when_agreed": None, "chall_rate": None,
            "overall_accuracy": None,
        }

    agreed = [r for r in results if r["agree"]]
    n_agree = len(agreed)
    n_correct_agreed = sum(r["correct"] for r in agreed)
    n_chall = n_agree - n_correct_agreed
    overall_correct = sum(r["correct"] for r in results)

    return {
        "n": n,
        "n_planned": n_planned,
        "complete": n >= n_planned,
        "agreement_rate": n_agree / n,
        "escalation_rate": (n - n_agree) / n,
        "accuracy_when_agreed": (n_correct_agreed / n_agree) if n_agree else None,
        "chall_rate": (n_chall / n_agree) if n_agree else None,
        "overall_accuracy": overall_correct / n,
    }


def _mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def pooled_analysis(pairs_filter=None, export=False):
    available = discover_available(pairs_filter)
    if not available:
        print(f"No checkpoints found under {CHECKPOINT_DIR}. "
              f"Run pcr_runner.py for at least one pair/benchmark first.")
        return

    pairs_seen = sorted({p for p, _ in available})
    per_pair_benchmark = {}  # (pair, benchmark) -> cell_stats dict
    per_pair_meta = {}       # pair -> {"cheap_a":..., "cheap_b":..., "frontier":...}

    for pair_id, benchmark in available:
        data = load_checkpoint(pair_id, benchmark)
        stats = cell_stats(data)
        per_pair_benchmark[(pair_id, benchmark)] = stats
        if pair_id not in per_pair_meta:
            cheap_a, cheap_b, frontier = data.get("cheap_a"), data.get("cheap_b"), data.get("frontier")
            # Legacy checkpoints (schema_version-based, e.g. the original
            # Mistral pair3 files) don't have these at the top level -- fall
            # back to the first result row, which does carry per-model info
            # under different key names (cheap_a_provider/cheap_a_model etc).
            if cheap_a is None and data.get("results"):
                r0 = data["results"][0]
                if "cheap_a_provider" in r0:
                    cheap_a = f"{r0.get('cheap_a_provider')}:{r0.get('cheap_a_model')}"
                    cheap_b = f"{r0.get('cheap_b_provider')}:{r0.get('cheap_b_model')}"
                    frontier = f"{r0.get('frontier_provider')}:{r0.get('frontier_model')}"
            per_pair_meta[pair_id] = {
                "cheap_a": cheap_a,
                "cheap_b": cheap_b,
                "frontier": frontier,
            }

    print("\n" + "=" * 78)
    print("POOLED ANALYSIS -- per (pair, benchmark) cell")
    print("=" * 78)
    print(f"{'Pair':<8}{'Benchmark':<10}{'N':>6}{'Complete':>10}"
          f"{'Agree%':>9}{'Acc(agr)':>10}{'C-Hall%':>10}{'Overall':>9}  {'Flag'}")
    print("-" * 78)
    for pair_id in pairs_seen:
        for bm in BENCHMARKS:
            s = per_pair_benchmark.get((pair_id, bm))
            if s is None:
                continue
            complete_str = "yes" if s["complete"] else f"partial({s['n']}/{s['n_planned']})"
            agree_str = f"{s['agreement_rate']:.1%}" if s["agreement_rate"] is not None else "n/a"
            acc_str = f"{s['accuracy_when_agreed']:.1%}" if s["accuracy_when_agreed"] is not None else "n/a"
            chall_str = f"{s['chall_rate']:.1%}" if s["chall_rate"] is not None else "n/a"
            overall_str = f"{s['overall_accuracy']:.1%}" if s["overall_accuracy"] is not None else "n/a"
            flag = "EXCLUDED FROM FAMILY POOLING" if (pair_id, bm) in EXCLUDE_FROM_FAMILY_POOLING else ""
            print(f"{pair_id:<8}{bm:<10}{s['n']:>6}{complete_str:>10}"
                  f"{agree_str:>9}{acc_str:>10}{chall_str:>10}{overall_str:>9}  {flag}")
    print("-" * 78)
    for (p, bm), reason in EXCLUDE_FROM_FAMILY_POOLING.items():
        if (p, bm) in per_pair_benchmark:
            print(f"[excluded] {p}/{bm}: {reason}")

    # ---- Benchmark-balanced pooling per pair: average the per-benchmark
    # rates unweighted, so MMLU's larger N doesn't dominate GSM8K/ARC. ----
    print("\n" + "=" * 78)
    print("BENCHMARK-BALANCED POOLED SCORE -- per pair "
          "(unweighted mean across benchmarks, NOT weighted by N)")
    print("=" * 78)
    print(f"{'Pair':<8}{'#Benchmarks':>12}{'Agree%':>9}{'Acc(agr)':>10}"
          f"{'C-Hall%':>10}{'Overall':>9}{'Family':>14}")
    print("-" * 78)

    pair_pooled = {}
    pair_pooled_for_family = {}  # excludes EXCLUDE_FROM_FAMILY_POOLING cells
    for pair_id in pairs_seen:
        cells = [per_pair_benchmark[(pair_id, bm)] for bm in BENCHMARKS
                 if (pair_id, bm) in per_pair_benchmark]
        n_bm = len(cells)
        pooled = {
            "n_benchmarks": n_bm,
            "agreement_rate": _mean([c["agreement_rate"] for c in cells]),
            "accuracy_when_agreed": _mean([c["accuracy_when_agreed"] for c in cells]),
            "chall_rate": _mean([c["chall_rate"] for c in cells]),
            "overall_accuracy": _mean([c["overall_accuracy"] for c in cells]),
        }
        pair_pooled[pair_id] = pooled

        family_cells = [per_pair_benchmark[(pair_id, bm)] for bm in BENCHMARKS
                         if (pair_id, bm) in per_pair_benchmark
                         and (pair_id, bm) not in EXCLUDE_FROM_FAMILY_POOLING]
        if family_cells:
            pair_pooled_for_family[pair_id] = {
                "n_benchmarks": len(family_cells),
                "agreement_rate": _mean([c["agreement_rate"] for c in family_cells]),
                "accuracy_when_agreed": _mean([c["accuracy_when_agreed"] for c in family_cells]),
                "chall_rate": _mean([c["chall_rate"] for c in family_cells]),
                "overall_accuracy": _mean([c["overall_accuracy"] for c in family_cells]),
            }
        # else: every benchmark for this pair got excluded -- it contributes
        # nothing to the family grand mean, which is correct, not a bug.

        family = FAMILY_CONDITIONS.get(pair_id, "UNSET")

        def f(v, pct=True):
            return f"{v:.1%}" if (pct and v is not None) else (f"{v}" if v is not None else "n/a")

        print(f"{pair_id:<8}{n_bm:>12}{f(pooled['agreement_rate']):>9}"
              f"{f(pooled['accuracy_when_agreed']):>10}{f(pooled['chall_rate']):>10}"
              f"{f(pooled['overall_accuracy']):>9}{family:>14}")
    print("-" * 78)
    if any(FAMILY_CONDITIONS.get(p, "UNSET") == "UNSET" for p in pairs_seen):
        print("[note] Family column shows UNSET for pairs not yet tagged in "
              "FAMILY_CONDITIONS -- see the comment block at the top of this "
              "file. Grand pooled-by-family numbers below are skipped until "
              "every completed pair has a confirmed tag.")

    # ---- Grand pooled-by-family-condition (only if every pair is tagged) ----
    if pairs_seen and all(FAMILY_CONDITIONS.get(p) in ("within_family", "cross_family") for p in pairs_seen):
        print("\n" + "=" * 78)
        print("POOLED BY FAMILY CONDITION (unweighted mean across pairs in each group; "
              "excludes cells flagged above)")
        print("=" * 78)
        by_condition = defaultdict(list)
        skipped_pairs = []
        for pair_id in pairs_seen:
            if pair_id in pair_pooled_for_family:
                by_condition[FAMILY_CONDITIONS[pair_id]].append(pair_pooled_for_family[pair_id])
            else:
                skipped_pairs.append(pair_id)
        for condition, pooled_list in by_condition.items():
            grand = {
                "agreement_rate": _mean([p["agreement_rate"] for p in pooled_list]),
                "accuracy_when_agreed": _mean([p["accuracy_when_agreed"] for p in pooled_list]),
                "chall_rate": _mean([p["chall_rate"] for p in pooled_list]),
                "overall_accuracy": _mean([p["overall_accuracy"] for p in pooled_list]),
            }
            print(f"{condition:<16} n_pairs={len(pooled_list)}  "
                  f"agree={grand['agreement_rate']:.1%}  "
                  f"acc_agreed={grand['accuracy_when_agreed']:.1%}  "
                  f"chall={grand['chall_rate']:.1%}  "
                  f"overall={grand['overall_accuracy']:.1%}")
        if skipped_pairs:
            print(f"[note] {', '.join(skipped_pairs)} contributed nothing here -- "
                  f"every one of its benchmarks is in EXCLUDE_FROM_FAMILY_POOLING.")
        print("-" * 78)
        print("[note] This is a descriptive pooled comparison, not a "
              "significance test -- see the power-analysis discussion "
              "(N required per condition) before treating any gap here as "
              "more than suggestive.")

    print("=" * 78 + "\n")

    if export:
        EXPORT_DIR.mkdir(exist_ok=True)
        out = {
            "cells": {
                f"{p}_{bm}": per_pair_benchmark[(p, bm)]
                for (p, bm) in per_pair_benchmark
            },
            "pooled_by_pair": pair_pooled,
            "pooled_by_pair_for_family": pair_pooled_for_family,
            "family_conditions": FAMILY_CONDITIONS,
            "excluded_from_family_pooling": {
                f"{p}_{bm}": reason for (p, bm), reason in EXCLUDE_FROM_FAMILY_POOLING.items()
            },
            "meta": per_pair_meta,
        }
        out_path = EXPORT_DIR / "pooled_analysis.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Exported -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="*", default=None,
                         help="Restrict to specific pair ids (default: all pairs with checkpoints present)")
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    pooled_analysis(pairs_filter=args.pairs, export=args.export)