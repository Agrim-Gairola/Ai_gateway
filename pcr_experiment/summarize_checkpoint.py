"""
Summarize a PCR checkpoint file (partial or complete) without needing to
finish the full benchmark run. Also prints a per-domain breakdown for MMLU,
matching the style of the original paper's Table VI.

Works across all pairs now (pcr_runner.py writes checkpoints as
"{pair}_{benchmark}.json"). --pair defaults to "pair2" so your existing
Pair 2 commands keep working unchanged.

Usage:
    python summarize_checkpoint.py --benchmark mmlu                       # pair2 (default)
    python summarize_checkpoint.py --pair pair3 --benchmark mmlu --export
    python summarize_checkpoint.py --pair pair4 --benchmark gsm8k --export
    python summarize_checkpoint.py --pair pair5 --benchmark mmlu --export
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict

CHECKPOINT_DIR = Path("./checkpoints")


def wilson_ci(successes, n, z=1.96):
    """95% Wilson score confidence interval -- more honest than a raw
    percentage for small-n domains (see the fact-check discussion on your
    original paper's virology/econometrics rows)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    adj = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)
    lo = (centre - adj) / denom
    hi = (centre + adj) / denom
    return (max(0, lo), min(1, hi))


def compute_ablation(results):
    """Replicates the original paper's Table II ablation rows from raw
    per-question results: Always Cheap A, Always Cheap B, Consensus/No
    Escalation (falls back to A's answer on disagreement, matching the
    original paper's finding that this equals Always-Cheap-A), and full PCR."""
    n = len(results)
    if n == 0:
        return {}

    def acc_against(key):
        correct = sum(
            1 for r in results
            if r.get(key) is not None
            and str(r[key]).strip().upper() == str(r["gold"]).strip().upper()
        )
        return correct / n

    always_a = acc_against("ans_a")
    always_b = acc_against("ans_b")

    # Consensus, No Escalation: use ans_a when agreed, fall back to ans_a
    # when disagreed too (no frontier available) -- matches original paper's
    # observation that this equals Always-Cheap-A exactly.
    consensus_no_esc = always_a

    pcr_correct = sum(1 for r in results if r["correct"])
    pcr_full = pcr_correct / n

    return {
        "always_cheap_a": always_a,
        "always_cheap_b": always_b,
        "consensus_no_escalation": consensus_no_esc,
        "pcr_full": pcr_full,
    }


def summarize(benchmark: str, pair: str = "pair2", export: bool = False):
    ckpt_path = CHECKPOINT_DIR / f"{pair}_{benchmark}.json"
    if not ckpt_path.exists():
        print(f"No checkpoint found at {ckpt_path}")
        return

    with open(ckpt_path) as f:
        data = json.load(f)
    results = data["results"]
    n = len(results)
    n_total_planned = data.get("n_total", n)

    agreed = [r for r in results if r["agree"]]
    disagreed = [r for r in results if not r["agree"]]
    n_agree = len(agreed)
    n_correct_agreed = sum(r["correct"] for r in agreed)
    n_chall = n_agree - n_correct_agreed
    overall_correct = sum(r["correct"] for r in results)
    overall_acc = overall_correct / n if n else 0

    agree_lo, agree_hi = wilson_ci(n_agree, n)
    acc_agreed_lo, acc_agreed_hi = wilson_ci(n_correct_agreed, n_agree) if n_agree else (0, 0)
    chall_lo, chall_hi = wilson_ci(n_chall, n_agree) if n_agree else (0, 0)
    overall_lo, overall_hi = wilson_ci(overall_correct, n)

    print("\n" + "=" * 70)
    print(f"SUMMARY -- {pair}/{benchmark.upper()}  "
          f"({n}/{n_total_planned} questions completed)")
    print("=" * 70)
    print(f"Agreement rate:            {n_agree}/{n} = {n_agree/n:.1%}  "
          f"(95% CI: {agree_lo:.1%}-{agree_hi:.1%})")
    print(f"Escalation rate:           {len(disagreed)}/{n} = {len(disagreed)/n:.1%}")
    if n_agree:
        print(f"Accuracy when agreed:      {n_correct_agreed}/{n_agree} = "
              f"{n_correct_agreed/n_agree:.1%}  (95% CI: {acc_agreed_lo:.1%}-{acc_agreed_hi:.1%})")
        print(f"Coordinated hallucination: {n_chall}/{n_agree} = "
              f"{n_chall/n_agree:.1%}  (95% CI: {chall_lo:.1%}-{chall_hi:.1%})")
    print(f"Overall PCR accuracy:      {overall_correct}/{n} = {overall_acc:.1%}  "
          f"(95% CI: {overall_lo:.1%}-{overall_hi:.1%})")

    ablation = compute_ablation(results)
    print("\n" + "-" * 70)
    print("ABLATION (matches original paper's Table II style)")
    print("-" * 70)
    print(f"{'Configuration':<30}{'Accuracy':>12}")
    print(f"{'Always Cheap A':<30}{ablation['always_cheap_a']:>12.1%}")
    print(f"{'Always Cheap B':<30}{ablation['always_cheap_b']:>12.1%}")
    print(f"{'Consensus, No Esc.':<30}{ablation['consensus_no_escalation']:>12.1%}")
    print(f"{'PCR, Full (this pair)':<30}{ablation['pcr_full']:>12.1%}")
    print("-" * 70)

    # Similarity stats (correct vs C-Hall), matching original paper's Table III
    sims_correct = [r["similarity"] for r in agreed if r["correct"] and r["similarity"] is not None]
    sims_chall = [r["similarity"] for r in agreed if not r["correct"] and r["similarity"] is not None]
    if sims_correct and sims_chall:
        mean_correct = sum(sims_correct) / len(sims_correct)
        mean_chall = sum(sims_chall) / len(sims_chall)
        print(f"\nMean cosine sim | correct agreement:  {mean_correct:.3f}  (n={len(sims_correct)})")
        print(f"Mean cosine sim | C-Hall agreement:    {mean_chall:.3f}  (n={len(sims_chall)})")
        print(f"Similarity delta (correct - C-Hall):   {mean_correct - mean_chall:+.3f}")
    elif any(r["similarity"] is None for r in agreed):
        print("\n[note] similarity is null for some/all rows -- sentence-transformers "
              "likely failed to load during this run; C-Hall/accuracy numbers above are "
              "still valid since they only depend on letter agreement, not similarity.")

    # Domain-level breakdown (MMLU only)
    if benchmark == "mmlu":
        by_domain = defaultdict(list)
        for r in results:
            by_domain[r["domain"]].append(r)

        print("\n" + "-" * 70)
        print(f"{'Domain':<30}{'N':>4}{'Agree%':>9}{'Acc(agr)':>10}{'C-Hall%':>10}")
        print("-" * 70)
        for domain, rows in sorted(by_domain.items(), key=lambda x: -len(x[1])):
            d_agreed = [r for r in rows if r["agree"]]
            d_n = len(rows)
            d_agree_rate = len(d_agreed) / d_n if d_n else 0
            d_correct_agreed = sum(r["correct"] for r in d_agreed)
            d_acc_agreed = d_correct_agreed / len(d_agreed) if d_agreed else float("nan")
            d_chall = (len(d_agreed) - d_correct_agreed) / len(d_agreed) if d_agreed else float("nan")
            flag = " (n<5, unstable)" if d_n < 5 else (" (n<15)" if d_n < 15 else "")
            acc_str = f"{d_acc_agreed:.1%}" if d_agreed else "n/a"
            chall_str = f"{d_chall:.1%}" if d_agreed else "n/a"
            print(f"{domain:<30}{d_n:>4}{d_agree_rate:>9.1%}{acc_str:>10}{chall_str:>10}{flag}")
        print("-" * 70)
        print("Rows with n<5 have wide confidence intervals -- treat as suggestive, "
              "not conclusive (see Wilson CI note in your fact-check).")

    print("=" * 70 + "\n")

    if export:
        export_dir = Path("./exports")
        export_dir.mkdir(exist_ok=True)

        summary_json = {
            "pair": pair,
            "benchmark": benchmark,
            "n_completed": n,
            "n_planned": n_total_planned,
            "note": "PARTIAL RUN -- stopped early due to time constraint" if n < n_total_planned else "complete",
            "agreement_rate": n_agree / n,
            "agreement_rate_95ci": [agree_lo, agree_hi],
            "escalation_rate": len(disagreed) / n,
            "accuracy_when_agreed": (n_correct_agreed / n_agree) if n_agree else None,
            "accuracy_when_agreed_95ci": [acc_agreed_lo, acc_agreed_hi] if n_agree else None,
            "coordinated_hallucination_rate": (n_chall / n_agree) if n_agree else None,
            "coordinated_hallucination_rate_95ci": [chall_lo, chall_hi] if n_agree else None,
            "overall_pcr_accuracy": overall_acc,
            "overall_pcr_accuracy_95ci": [overall_lo, overall_hi],
            "ablation": ablation,
            "mean_similarity_correct": (sum(sims_correct) / len(sims_correct)) if sims_correct else None,
            "mean_similarity_chall": (sum(sims_chall) / len(sims_chall)) if sims_chall else None,
        }
        json_path = export_dir / f"{pair}_{benchmark}_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary_json, f, indent=2)
        print(f"Exported summary -> {json_path}")

        if benchmark == "mmlu":
            csv_path = export_dir / f"{pair}_{benchmark}_domain_breakdown.csv"
            with open(csv_path, "w") as f:
                f.write("domain,n,agree_rate,accuracy_when_agreed,chall_rate,flag\n")
                for domain, rows in sorted(by_domain.items(), key=lambda x: -len(x[1])):
                    d_agreed = [r for r in rows if r["agree"]]
                    d_n = len(rows)
                    d_agree_rate = len(d_agreed) / d_n if d_n else 0
                    d_correct_agreed = sum(r["correct"] for r in d_agreed)
                    d_acc_agreed = (d_correct_agreed / len(d_agreed)) if d_agreed else ""
                    d_chall = ((len(d_agreed) - d_correct_agreed) / len(d_agreed)) if d_agreed else ""
                    flag = "small_n" if d_n < 15 else ""
                    f.write(f"{domain},{d_n},{d_agree_rate:.4f},{d_acc_agreed},{d_chall},{flag}\n")
            print(f"Exported domain breakdown -> {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["mmlu", "gsm8k", "arc"], required=True)
    parser.add_argument("--pair", default="pair2",
                         help="Pair id (default: pair2, for backward compatibility). "
                              "Use pair3/pair4/pair5 for the new pairs.")
    parser.add_argument("--export", action="store_true",
                         help="Write a clean JSON summary + CSV domain breakdown to ./exports/")
    args = parser.parse_args()
    summarize(args.benchmark, pair=args.pair, export=args.export)