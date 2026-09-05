"""
family_significance.py
======================
Turns the descriptive within-family vs cross-family pooled comparison
(pooled_analysis.py) into an inferential one: bootstrap confidence intervals
on the between-condition gap for the two quantities the family-relatedness
hypothesis is actually about --

  1. agreement rate            (do same-family cheap models agree more often?)
  2. coordinated hallucination (when they agree, are they wrong together
                                more often -- i.e. is the extra agreement
                                "bad" agreement?)

Method
------
Every completed (pair, benchmark) checkpoint is tagged within_family /
cross_family via pcr_experiment.pooled_analysis.FAMILY_CONDITIONS. Cells in
EXCLUDE_FROM_FAMILY_POOLING (capability-mismatched pairs) are dropped.

Two pooling schemes, reported side by side because they answer slightly
different questions:

  * question-level pool: concatenate the raw per-question rows of every cell
    in a condition and resample rows with replacement. Tight CIs, but treats
    one big pair as more informative than one small pair.
  * pair-level (benchmark-balanced) pool: compute each pair's benchmark-mean
    rate first (matching pooled_analysis.py), then resample *pairs* with
    replacement. Wider CIs, but each pair counts once -- the honest unit of
    replication for a "family" claim. With only 2-3 pairs per side this is
    deliberately conservative.

A gap whose 95% CI excludes 0 is evidence of a real between-condition
difference at this (still pilot) scale; one that spans 0 is not.

Usage:
    python -m pcr_experiment.family_significance
    python -m pcr_experiment.family_significance --iterations 20000 --export
"""
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

from .pooled_analysis import (
    CHECKPOINT_DIR, EXPORT_DIR, BENCHMARKS,
    FAMILY_CONDITIONS, EXCLUDE_FROM_FAMILY_POOLING, discover_available,
)


def load_rows(pair_id, benchmark):
    path = CHECKPOINT_DIR / f"{pair_id}_{benchmark}.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for r in data["results"]:
        # normalise across modern + legacy schemas: both carry agree/correct
        rows.append({"agree": bool(r["agree"]), "correct": bool(r["correct"])})
    return rows


def agree_rate(rows):
    return sum(r["agree"] for r in rows) / len(rows) if rows else 0.0


def chall_rate(rows):
    agreed = [r for r in rows if r["agree"]]
    if not agreed:
        return None
    return sum(not r["correct"] for r in agreed) / len(agreed)


def _pct(x):
    return "n/a" if x is None else f"{x*100:5.1f}%"


def _ci_str(lo, hi):
    excl = (lo > 0) or (hi < 0)
    return f"[{lo*100:+5.1f}, {hi*100:+5.1f}] pp" + ("   <-- excludes 0" if excl else "")


def bootstrap_question_level(rows_a, rows_b, metric, iters, seed):
    rng = random.Random(seed)
    diffs = []
    na, nb = len(rows_a), len(rows_b)
    for _ in range(iters):
        ra = [rows_a[rng.randrange(na)] for _ in range(na)]
        rb = [rows_b[rng.randrange(nb)] for _ in range(nb)]
        va, vb = metric(ra), metric(rb)
        if va is None or vb is None:
            continue
        diffs.append(va - vb)
    diffs.sort()
    return diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]


def bootstrap_pair_level(pair_rates_a, pair_rates_b, iters, seed):
    """Resample whole pairs. pair_rates_* is a list of per-pair scalar rates."""
    rng = random.Random(seed)
    a = [v for v in pair_rates_a if v is not None]
    b = [v for v in pair_rates_b if v is not None]
    if not a or not b:
        return None, None
    diffs = []
    for _ in range(iters):
        sa = sum(a[rng.randrange(len(a))] for _ in range(len(a))) / len(a)
        sb = sum(b[rng.randrange(len(b))] for _ in range(len(b))) / len(b)
        diffs.append(sa - sb)
    diffs.sort()
    return diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]


def run(iterations=10000, seed=42, export=False):
    available = discover_available()
    # condition -> benchmark -> list of (pair_id, rows)
    cells = defaultdict(lambda: defaultdict(list))
    used, skipped = [], []
    for pair_id, bm in available:
        cond = FAMILY_CONDITIONS.get(pair_id)
        if cond not in ("within_family", "cross_family"):
            skipped.append(f"{pair_id}/{bm} (untagged)")
            continue
        if (pair_id, bm) in EXCLUDE_FROM_FAMILY_POOLING:
            skipped.append(f"{pair_id}/{bm} (capability-mismatch exclusion)")
            continue
        rows = load_rows(pair_id, bm)
        if len(rows) < 50:                       # skip partial / in-progress cells
            skipped.append(f"{pair_id}/{bm} (n={len(rows)}, partial)")
            continue
        cells[cond][bm].append((pair_id, rows))
        used.append(f"{pair_id}/{bm} [{cond}] n={len(rows)}")

    print("=" * 78)
    print("WITHIN-FAMILY vs CROSS-FAMILY  --  bootstrap significance of the gap")
    print("(gap sign convention: within_family minus cross_family)")
    print("=" * 78)
    print("cells used:")
    for u in sorted(used):
        print("  +", u)
    for s in sorted(skipped):
        print("  -", s)

    out = {"cells_used": sorted(used), "cells_skipped": sorted(skipped),
           "iterations": iterations, "per_benchmark": {}, "benchmark_balanced": {}}

    # ---- per-benchmark, question-level ----
    print("\n" + "-" * 78)
    print(f"{'benchmark':<14}{'metric':<14}{'within':>9}{'cross':>9}{'gap':>9}   95% bootstrap CI (question-level)")
    print("-" * 78)
    common_bms = [b for b in BENCHMARKS
                  if cells['within_family'].get(b) and cells['cross_family'].get(b)]
    for bm in common_bms:
        win_rows = [r for _, rr in cells['within_family'][bm] for r in rr]
        cro_rows = [r for _, rr in cells['cross_family'][bm] for r in rr]
        for mname, metric in (("agreement", agree_rate), ("c_hall", chall_rate)):
            wv, cv = metric(win_rows), metric(cro_rows)
            gap = None if (wv is None or cv is None) else wv - cv
            lo, hi = bootstrap_question_level(win_rows, cro_rows, metric, iterations, seed)
            print(f"{bm:<14}{mname:<14}{_pct(wv):>9}{_pct(cv):>9}"
                  f"{('n/a' if gap is None else f'{gap*100:+.1f}pp'):>9}   {_ci_str(lo, hi)}")
            out["per_benchmark"].setdefault(bm, {})[mname] = {
                "within": wv, "cross": cv, "gap": gap, "ci95": [lo, hi],
            }

    # ---- benchmark-balanced, pair-level ----
    print("\n" + "-" * 78)
    print("BENCHMARK-BALANCED, PAIR-LEVEL  (each pair = one draw; conservative)")
    print("-" * 78)

    def pair_balanced_rates(cond, metric):
        # per pair: mean of its per-benchmark rates
        by_pair = defaultdict(list)
        for bm, lst in cells[cond].items():
            for pair_id, rows in lst:
                v = metric(rows)
                if v is not None:
                    by_pair[pair_id].append(v)
        return {p: sum(vs) / len(vs) for p, vs in by_pair.items() if vs}

    for mname, metric in (("agreement", agree_rate), ("c_hall", chall_rate)):
        wr = pair_balanced_rates("within_family", metric)
        cr = pair_balanced_rates("cross_family", metric)
        if min(len(wr), len(cr)) < 2:
            print(f"[warn] {mname}: one condition has <2 pairs "
                  f"(within={len(wr)}, cross={len(cr)}) -- the pair-level CI "
                  f"below is under-dispersed on that side; treat as indicative only.")
        wv = sum(wr.values()) / len(wr) if wr else None
        cv = sum(cr.values()) / len(cr) if cr else None
        gap = None if (wv is None or cv is None) else wv - cv
        lo, hi = bootstrap_pair_level(list(wr.values()), list(cr.values()), iterations, seed)
        print(f"{mname:<14}"
              f"within={_pct(wv)} (pairs={sorted(wr)})")
        print(f"{'':<14}"
              f" cross={_pct(cv)} (pairs={sorted(cr)})")
        if gap is not None and lo is not None:
            print(f"{'':<14} gap={gap*100:+.1f}pp   95% CI {_ci_str(lo, hi)}")
        out["benchmark_balanced"][mname] = {
            "within": wv, "cross": cv, "gap": gap,
            "ci95": [lo, hi], "within_pairs": wr, "cross_pairs": cr,
        }
        print()

    print("=" * 78)
    print("Read: a CI that excludes 0 = the between-condition gap is unlikely to be")
    print("noise at this sample size. A CI spanning 0 = current pilot data cannot")
    print("distinguish the conditions on that metric.")
    print("=" * 78)

    if export:
        EXPORT_DIR.mkdir(exist_ok=True)
        p = EXPORT_DIR / "family_significance.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nExported -> {p}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()
    run(iterations=a.iterations, seed=a.seed, export=a.export)
