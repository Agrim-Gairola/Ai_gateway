"""
pcr3_runner.py
==============
PCR-3 ablation: three cheap models instead of two. Decision rule:

  - if >= 2 of the 3 cheap answers agree  -> take that majority answer
  - if all 3 differ                        -> one frontier escalation

vs PCR-2 this trades a third cheap call (cheap, parallel) for a lower
escalation rate and a majority-vote safety margin. This script only exists to
produce the PCR-3 rows in the ablation study; PCR-2 stays in pcr_runner.py.

Checkpoints: pcr3_{tag}_{benchmark}.json  in the same checkpoints/ dir.

Usage:
    python -m pcr_experiment.pcr3_runner --benchmark arc --n 120
    (models + tag are set in TRIPLE below)
"""
import json
import asyncio
import argparse
import time
from collections import Counter
from pathlib import Path

from . import pcr_runner as R
from .pcr_config import ModelSpec

# The one triple we run: three different labs, mistral-medium as frontier.
TRIPLE = {
    # All-Groq capability-matched trio: the solo sweep put gpt-oss-20b,
    # qwen3.6-27b and qwen3.8-27b within ~1pp of each other on every benchmark,
    # and all three are on the fast free path. This is also the practitioner-
    # relevant PCR-3 config. (A cross-provider trio was the original plan but
    # Gemini's 500/day and Mistral's RPM limit are exhausted -- see run_phase3b.)
    "tag": "allgroq_gptoss20_qwen36_qwen38",
    "cheap": [
        ModelSpec("groq", "openai/gpt-oss-20b"),
        ModelSpec("groq", "qwen/qwen3.6-27b"),
        ModelSpec("groq", "qwen/qwen3.8-27b"),
    ],
    "frontier": ModelSpec("groq", "openai/gpt-oss-120b"),
}

CKPT = Path(__file__).resolve().parent / "checkpoints"
SLEEP = 4.5


async def run(benchmark, n):
    loader = R.LOADERS[benchmark]
    extract = R.EXTRACTORS[benchmark]
    cbud = R.CHEAP_MAX_TOKENS[benchmark]
    fbud = R.FRONTIER_MAX_TOKENS[benchmark]
    items = loader(n=n)
    path = CKPT / f"pcr3_{TRIPLE['tag']}_{benchmark}.json"
    results = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            results = json.load(f)["results"]
    start = len(results)
    print(f"=== PCR-3 [{TRIPLE['tag']}] -- {benchmark} n={len(items)} (from {start}) ===")
    print("cheap:", [f"{m.provider}:{m.model}" for m in TRIPLE["cheap"]],
          "frontier:", f"{TRIPLE['frontier'].provider}:{TRIPLE['frontier'].model}")

    def _save():
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"tag": TRIPLE["tag"], "benchmark": benchmark,
                       "n_total": len(items),
                       "cheap": [f"{m.provider}:{m.model}" for m in TRIPLE["cheap"]],
                       "frontier": f"{TRIPLE['frontier'].provider}:{TRIPLE['frontier'].model}",
                       "results": results}, f, indent=2)

    try:
        for i in range(start, len(items)):
            it = items[i]
            calls = await asyncio.gather(*[
                R.call_model(m, it["prompt"], cbud) for m in TRIPLE["cheap"]
            ])
            ans = [extract(c.text) for c in calls]
            counts = Counter(a for a in ans if a is not None)
            majority, maj_n = (counts.most_common(1)[0] if counts else (None, 0))
            has_majority = maj_n >= 2

            fcall = None
            if has_majority:
                final = majority
            else:
                fcall = await R.call_model(TRIPLE["frontier"], it["prompt"], fbud)
                final = extract(fcall.text)

            gold = str(it["gold"]).strip().upper()
            correct = final is not None and str(final).strip().upper() == gold
            par_lat = max(c.latency_seconds for c in calls)
            tot_lat = par_lat + (fcall.latency_seconds if fcall else 0.0)

            results.append({
                "idx": i, "domain": it["domain"], "gold": it["gold"],
                "ans": ans, "majority": majority, "maj_n": maj_n,
                "agree": has_majority,              # 'agree' == a 2/3 majority formed
                "escalated": not has_majority,
                "frontier_ans": (extract(fcall.text) if fcall else None),
                "final_ans": final, "correct": correct,
                "usage": {
                    "cheap": [c.to_dict() for c in calls],
                    "frontier": fcall.to_dict() if fcall else None,
                    "parallel_latency_seconds": round(par_lat, 3),
                    "total_latency_seconds": round(tot_lat, 3),
                },
            })
            if (i + 1) % 50 == 0 or i == len(items) - 1:
                _save()
                print(f"  checkpointed {i + 1}/{len(items)}")
            await asyncio.sleep(SLEEP)
    except BaseException:
        if len(results) > start:
            _save()
            print(f"  emergency-checkpointed {len(results)}/{len(items)}")
        raise

    n_tot = len(results)
    maj = [r for r in results if r["agree"]]
    ok_maj = sum(r["correct"] for r in maj)
    esc = [r for r in results if not r["agree"]]
    print("\n" + "=" * 56)
    print(f"PCR-3 SUMMARY -- {TRIPLE['tag']}/{benchmark}")
    print("=" * 56)
    print(f"N:                          {n_tot}")
    print(f"Majority formed (2/3+):     {len(maj)}/{n_tot} = {len(maj)/n_tot:.1%}")
    print(f"Escalation (all 3 differ):  {len(esc)}/{n_tot} = {len(esc)/n_tot:.1%}")
    if maj:
        print(f"Accuracy | majority:        {ok_maj}/{len(maj)} = {ok_maj/len(maj):.1%}")
        print(f"C-Hall | majority:          {len(maj)-ok_maj}/{len(maj)} = {(len(maj)-ok_maj)/len(maj):.1%}")
    print(f"Overall PCR-3 accuracy:     {sum(r['correct'] for r in results)/n_tot:.1%}")
    print("=" * 56)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=list(R.LOADERS.keys()))
    ap.add_argument("--n", type=int, default=120)
    a = ap.parse_args()
    asyncio.run(run(a.benchmark, a.n))
