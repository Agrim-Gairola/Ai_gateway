"""
build_comparison_table.py
==========================
Reads every exported summary JSON (from summarize_checkpoint.py --export)
across all pairs and benchmarks and builds one markdown comparison table,
ready to paste into the paper. Also flags any pair/benchmark combo defined
in pcr_config.py that's missing an export yet, so you know what's left.

Run this AFTER exporting each pair/benchmark you care about:
    python summarize_checkpoint.py --pair pair3 --benchmark mmlu --export
    python summarize_checkpoint.py --pair pair3 --benchmark gsm8k --export
    ... etc for every pair/benchmark you've run ...
    python build_comparison_table.py
"""

import json
from pathlib import Path

from pcr_config import PAIRS

EXPORT_DIR = Path("./exports")
OUT_PATH = Path("./exports/comparison_table.md")


def load_summary(pair_id: str, benchmark: str):
    path = EXPORT_DIR / f"{pair_id}_{benchmark}_summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fmt_pct(x, ci=None):
    if x is None:
        return "—"
    base = f"{x:.1%}"
    if ci:
        base += f" ({ci[0]:.1%}–{ci[1]:.1%})"
    return base


def main():
    rows = []
    missing = []

    for pair_id, cfg in PAIRS.items():
        for benchmark in ("mmlu", "gsm8k", "arc"):
            if benchmark not in cfg.benchmarks:
                continue  # not applicable to this pair by design (e.g. Pair 5)
            summary = load_summary(pair_id, benchmark)
            if summary is None:
                missing.append((pair_id, cfg.label, benchmark))
                continue
            rows.append({
                "pair": cfg.label,
                "benchmark": benchmark.upper(),
                "n": f"{summary['n_completed']}/{summary['n_planned']}",
                "agreement": fmt_pct(summary["agreement_rate"], summary.get("agreement_rate_95ci")),
                "acc_agreed": fmt_pct(summary["accuracy_when_agreed"], summary.get("accuracy_when_agreed_95ci")),
                "chall": fmt_pct(summary["coordinated_hallucination_rate"], summary.get("coordinated_hallucination_rate_95ci")),
                "overall": fmt_pct(summary["overall_pcr_accuracy"], summary.get("overall_pcr_accuracy_95ci")),
                "note": summary.get("note", ""),
            })

    lines = []
    lines.append("# PCR Cross-Pair Comparison Table\n")
    lines.append("| Pair | Benchmark | N | Agreement | Acc. when agreed | C-Hall rate | Overall PCR acc. |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        flag = " *(partial)*" if "PARTIAL" in r["note"] else ""
        lines.append(
            f"| {r['pair']} | {r['benchmark']} | {r['n']}{flag} | "
            f"{r['agreement']} | {r['acc_agreed']} | {r['chall']} | {r['overall']} |"
        )

    lines.append("\n_95% Wilson score confidence intervals in parentheses._\n")

    if missing:
        lines.append("\n## Not yet exported (run + export these before finalizing)\n")
        for pair_id, label, benchmark in missing:
            lines.append(f"- {label} — {benchmark.upper()} "
                          f"(`python summarize_checkpoint.py --pair {pair_id} --benchmark {benchmark} --export`)")

    # Methodological caveats worth carrying straight into the paper
    lines.append("\n## Caveats to disclose alongside this table\n")
    lines.append("- **Pair 2 is a mixed pair**: MMLU (q1–800) used `openai/gpt-oss-20b` as "
                  "cheap-A; GSM8K/ARC used `allam-2-7b`, after two forced swaps due to Groq "
                  "deprecations. Footnote this explicitly — do not present Pair 2 as one "
                  "consistent cheap-A model across benchmarks.")
    lines.append("- **Pairs 3 and 4 are pilot-scale** (N=250/150/100), not full-scale like "
                  "Pairs 1–2 (N=1000/500/300). Report their CIs accordingly and label them "
                  "'pilot replication' in the text, not as equal-weight primary results.")
    lines.append("- **Pair 5 is MMLU-only, N=100 subsample**, reusing Pair 4's cheap layer "
                  "with Gemini 3.1 Pro as frontier — isolates the frontier-model effect only; "
                  "not a full replication in its own right.")
    lines.append("- **Pair 1's frontier model** (`llama-3.3-70b-versatile`) is deprecated/shut "
                  "down on Groq as of 2026-08-16 and cannot be re-run — its numbers are "
                  "historical and can't be refreshed or extended.")

    out_text = "\n".join(lines)
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(out_text)
    print(out_text)
    print(f"\n\nWritten to {OUT_PATH}")


if __name__ == "__main__":
    main()