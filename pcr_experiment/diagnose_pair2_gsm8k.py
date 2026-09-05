"""
diagnose_pair2_gsm8k.py
========================
One-off diagnostic for the pair2/gsm8k anomaly flagged during pooled
analysis: 12.4% agreement and 29.0% C-Hall, wildly out of line with every
other GSM8K result (2.6-4.3% C-Hall elsewhere). Two live hypotheses:

  1. REAL SIGNAL -- pair2's documented mid-run cheap_a swap (gpt-oss-20b ->
     allam-2-7b, per the MIXED PAIR note in pcr_config.py) genuinely changed
     answer-formatting behavior partway through, causing real disagreement.
  2. EXTRACTION ARTIFACT -- extract_number()'s "grab the last number in the
     text" regex is misfiring on one model's output style, manufacturing
     disagreement that isn't a real behavioral difference.

This script doesn't decide between them -- it prints enough raw evidence
(the actual resp_a/resp_b text and what got extracted from each) for a
human to eyeball and judge. Read the printed samples before touching
pcr_config.py or the pooled numbers based on this pair.

Usage:
    python -m pcr_experiment.diagnose_pair2_gsm8k
    python -m pcr_experiment.diagnose_pair2_gsm8k --n 15
"""
import json
import argparse
from pathlib import Path

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def load(pair_id: str, benchmark: str):
    path = CHECKPOINT_DIR / f"{pair_id}_{benchmark}.json"
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def diagnose(pair_id="pair2", benchmark="gsm8k", n_samples=10):
    data = load(pair_id, benchmark)
    results = data["results"]
    total = len(results)
    disagreed = [r for r in results if not r["agree"]]

    print("\n" + "=" * 78)
    print(f"DIAGNOSTIC -- {pair_id}/{benchmark}  ({total} total, "
          f"{len(disagreed)} disagreed = {len(disagreed)/total:.1%})")
    print("=" * 78)

    # Split disagreements into two buckets to help distinguish the two
    # hypotheses at a glance:
    #  - "same number, different formatting" (e.g. "42" vs "42.0" vs "$42")
    #    would point at an extraction-regex artifact, not a real split.
    #  - "genuinely different numbers" points at a real behavioral / accuracy
    #    difference between the two models.
    same_numeric_value = []
    genuinely_different = []
    unparseable = []

    for r in disagreed:
        a, b = r.get("ans_a"), r.get("ans_b")
        if a is None or b is None:
            unparseable.append(r)
            continue
        try:
            if float(a) == float(b):
                same_numeric_value.append(r)
            else:
                genuinely_different.append(r)
        except (TypeError, ValueError):
            unparseable.append(r)

    print(f"\nOf {len(disagreed)} disagreements:")
    print(f"  Same numeric value, flagged as disagreeing anyway (likely a "
          f"string-comparison/formatting bug): {len(same_numeric_value)}")
    print(f"  Genuinely different extracted numbers (real model split, or "
          f"one/both extractions wrong): {len(genuinely_different)}")
    print(f"  One or both sides failed to extract any number at all "
          f"(ans_a/ans_b is None): {len(unparseable)}")

    if same_numeric_value:
        print("\n" + "-" * 78)
        print(f"SAME-VALUE DISAGREEMENTS (up to {n_samples}) -- if these "
              f"exist, it's a formatting/comparison bug, not real disagreement")
        print("-" * 78)
        for r in same_numeric_value[:n_samples]:
            print(f"\n[idx {r['idx']}] gold={r['gold']!r}")
            print(f"  ans_a={r['ans_a']!r}   ans_b={r['ans_b']!r}")
            print(f"  resp_a (last 150 chars): ...{r['resp_a'][-150:]!r}")
            print(f"  resp_b (last 150 chars): ...{r['resp_b'][-150:]!r}")

    if genuinely_different:
        print("\n" + "-" * 78)
        print(f"GENUINELY-DIFFERENT DISAGREEMENTS (up to {n_samples}) -- "
              f"eyeball whether these look like real reasoning differences "
              f"or truncated/garbled extraction")
        print("-" * 78)
        for r in genuinely_different[:n_samples]:
            print(f"\n[idx {r['idx']}] gold={r['gold']!r}  correct={r['correct']}")
            print(f"  ans_a={r['ans_a']!r}   ans_b={r['ans_b']!r}")
            print(f"  resp_a (last 150 chars): ...{r['resp_a'][-150:]!r}")
            print(f"  resp_b (last 150 chars): ...{r['resp_b'][-150:]!r}")

    if unparseable:
        print("\n" + "-" * 78)
        print(f"UNPARSEABLE (up to {n_samples}) -- extraction returned None "
              f"on one or both sides; these count as 'disagree' by definition "
              f"(None != any number) even though it's not a real model split")
        print("-" * 78)
        for r in unparseable[:n_samples]:
            print(f"\n[idx {r['idx']}] gold={r['gold']!r}")
            print(f"  ans_a={r['ans_a']!r}   ans_b={r['ans_b']!r}")
            print(f"  resp_a (last 150 chars): ...{r['resp_a'][-150:]!r}")
            print(f"  resp_b (last 150 chars): ...{r['resp_b'][-150:]!r}")

    # If the checkpoint has per-question usage/model info (modern schema),
    # check whether disagreements cluster around the documented cheap_a
    # swap point -- pcr_config.py's MIXED PAIR note says GSM8K used
    # allam-2-7b throughout (unlike MMLU's split), so this should be a
    # no-op check, but worth confirming rather than assuming.
    print("\n" + "-" * 78)
    print("DISAGREEMENT RATE BY POSITION (first third / middle third / last "
          "third of the run) -- a swap-related bug would concentrate in one "
          "segment; a genuine steady-state behavior split would look even")
    print("-" * 78)
    third = total // 3 or 1
    segments = [
        ("first third", results[:third]),
        ("middle third", results[third:2 * third]),
        ("last third", results[2 * third:]),
    ]
    for label, seg in segments:
        if not seg:
            continue
        seg_disagree = sum(1 for r in seg if not r["agree"])
        print(f"  {label:<14} n={len(seg):<5} disagree={seg_disagree:<5} "
              f"({seg_disagree/len(seg):.1%})")

    print("\n" + "=" * 78)
    print("VERDICT GUIDE (read the samples above, don't just trust this):")
    print(f"  - If SAME-VALUE count is large -> fix extract_number() or the "
          f"agree comparison, then re-derive C-Hall from corrected data.")
    print(f"  - If UNPARSEABLE count is large -> check whether one model's "
          f"GSM8K responses are getting truncated (CHEAP_MAX_TOKENS budget) "
          f"before a number appears at all.")
    print(f"  - If disagreement rate is uneven across thirds -> look at what "
          f"changed at that point in the run (model swap, quota-driven "
          f"retry, etc.) before trusting the pooled C-Hall number as-is.")
    print(f"  - If none of the above and disagreements look like clean, "
          f"complete, differently-reasoned answers -> this may be a real "
          f"signal, not an artifact. Worth a footnote either way in the "
          f"eventual write-up.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="pair2")
    parser.add_argument("--benchmark", default="gsm8k")
    parser.add_argument("--n", type=int, default=10,
                         help="Max samples to print per bucket")
    args = parser.parse_args()
    diagnose(args.pair, args.benchmark, args.n)