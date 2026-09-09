# -*- coding: utf-8 -*-
"""Regenerate the two paper figures from ONLY the 4 complete pairs
(pair3, pair7, pair10, pair11) so figures and tables are consistent.

  fig_chall_ladder.png  -- Finding 1: C-Hall vs benchmark difficulty
  fig_acc_cost.png      -- Finding 3: PCR-2 accuracy vs cost per 1k queries

    python -m pcr_experiment.paper_figures
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(ROOT, "checkpoints")
FIGDIR = os.path.join(ROOT, "exports", "figures")

PAIRS = [("pair3", "P3 (within)"), ("pair7", "P7 (within)"),
         ("pair10", "P10 (cross)"), ("pair11", "P11 (cross)")]
BMS = [("openbookqa", "OpenBookQA"), ("arc", "ARC"), ("mmlu", "MMLU"),
       ("commonsenseqa", "CommonsenseQA"), ("truthfulqa", "TruthfulQA"),
       ("mmlu_pro", "MMLU-Pro")]
TIER = {"openbookqa": "easy", "arc": "easy", "commonsenseqa": "easy",
        "mmlu": "medium", "truthfulqa": "hard", "mmlu_pro": "hard"}
TIER_C = {"easy": "#4C9A5A", "medium": "#4C72B0", "hard": "#C44E52"}
AF_ACC = {"arc": 95.0, "mmlu": 82.5, "mmlu_pro": 71.2, "truthfulqa": 78.8}


def cell(pid, bm):
    r = json.load(open(os.path.join(CK, f"{pid}_{bm}.json"), encoding="utf-8"))["results"]
    n = len(r)
    ag = [x for x in r if x["agree"]]
    ok = sum(x["correct"] for x in ag)
    return {"chall": (len(ag) - ok) / len(ag) * 100 if ag else 0.0,
            "acc": sum(x["correct"] for x in r) / n * 100}


def costs():
    d = json.load(open(os.path.join(ROOT, "exports", "cost_latency_analysis.json"),
                       encoding="utf-8"))["cells"]
    return {k: (v["cost_per_1k_queries_usd"]["pcr"],
                v["cost_per_1k_queries_usd"]["always_frontier"]) for k, v in d.items()}


def fig_ladder():
    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    xs = list(range(len(BMS)))
    for pid, lab in PAIRS:
        ys = [cell(pid, k)["chall"] for k, _ in BMS]
        ax.plot(xs, ys, marker="o", ms=4, lw=1.1, alpha=0.5, label=lab)
    means = [sum(cell(p, k)["chall"] for p, _ in PAIRS) / len(PAIRS) for k, _ in BMS]
    ax.plot(xs, means, marker="s", ms=7, lw=2.6, color="black", label="mean (4 pairs)")
    for x, m in zip(xs, means):
        ax.annotate(f"{m:.0f}%", (x, m), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab in BMS], rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("C-Hall on agreed questions (%)", fontsize=9)
    ax.set_ylim(0, 30)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=7.5, ncol=2, loc="upper left", framealpha=0.9)
    ax.set_title("Coordinated hallucination climbs monotonically with benchmark difficulty",
                 fontsize=9.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        for d in (FIGDIR, ROOT):          # exports/figures/ AND next to the .tex
            fig.savefig(os.path.join(d, f"fig_chall_ladder.{ext}"),
                        dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_chall_ladder.{png,pdf} means", [round(m, 1) for m in means])


def fig_acc_cost():
    cm = costs()
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    seen = set()
    for pid, _ in PAIRS:
        for k, lab in BMS:
            cc = cm.get(f"{pid}_{k}")
            if not cc:
                continue
            pcr_c, _af_c = cc
            s = cell(pid, k)
            t = TIER[k]
            ax.scatter(pcr_c, s["acc"], s=48, color=TIER_C[t], edgecolor="white",
                       linewidth=0.6, zorder=3, label=t if t not in seen else None)
            seen.add(t)
            if k in ("mmlu_pro", "mmlu"):     # label only the high-cost points
                short = {"mmlu": "MMLU", "mmlu_pro": "MP"}[k]
                ax.annotate(pid.replace("pair", "P") + "/" + short,
                            (pcr_c, s["acc"]), fontsize=6.5, xytext=(4, 3),
                            textcoords="offset points", alpha=0.75)
    # always-frontier: x at its own cost, y at its solo accuracy
    afx, afy = [], []
    for k, acc in AF_ACC.items():
        afcs = [cm[f"{p}_{k}"][1] for p, _ in PAIRS if f"{p}_{k}" in cm]
        if afcs:
            afx.append(sum(afcs) / len(afcs)); afy.append(acc)
    ax.scatter(afx, afy, marker="x", s=70, color="#222", linewidth=2, zorder=4,
               label="always-frontier")
    ax.set_xlabel("PCR-2 cost, \\$ per 1,000 queries", fontsize=9)
    ax.set_ylabel("PCR-2 accuracy (%)", fontsize=9)
    ax.set_ylim(60, 100)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
    ax.set_title("Accuracy vs cost: 4 pairs $\\times$ 6 benchmarks", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        for d in (FIGDIR, ROOT):
            fig.savefig(os.path.join(d, f"fig_acc_cost.{ext}"),
                        dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_acc_cost.{png,pdf}")


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    fig_ladder()
    fig_acc_cost()
