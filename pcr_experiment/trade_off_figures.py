"""
trade_off_figures.py
====================
The accuracy vs cost and accuracy vs latency trade-off, per pair, per
difficulty tier -- the plots that answer "which pair beats the big model on
the trade-off, and where".

Reads exports/ablation_study.json (run ablation_study.py --export first).
Points that HAVE measured cost+latency (post-instrumentation checkpoints:
pair3_arc, pair4, pair6, pair7, pair8, pair9) are plotted; others are listed
as accuracy-only in the console.

Always-Frontier is drawn as a reference band per tier from the escalated-subset
frontier accuracy actually recorded (a lower bound -- the frontier's score on
the questions the cheap pair found hard) until the solo frontier sweep can run
after the Groq reset.

    python -m pcr_experiment.trade_off_figures
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .pooled_analysis import EXPORT_DIR, CHECKPOINT_DIR
from .ablation_study import TIER_OF, discover, _norm_rows

FIG = EXPORT_DIR / "figures"
TIER_COLOR = {"easy": "#4C9A5A", "medium": "#4C72B0", "hard": "#C44E52", "math": "#8C6BB1"}


def frontier_lb_by_tier():
    """Lower-bound Always-Frontier accuracy per tier: mean over cells of the
    frontier's accuracy on that cell's escalated (disagreement) subset."""
    acc = {}
    for pair, bm, path in discover():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rows = list(_norm_rows(data))
        esc = [r for r in rows if not r["agree"] and r["has_frontier"]]
        if len(esc) < 8:
            continue
        t = TIER_OF.get(bm)
        acc.setdefault(t, []).append(sum(r["cf"] for r in esc) / len(esc))
    return {t: sum(v) / len(v) for t, v in acc.items()}


def main():
    p = EXPORT_DIR / "ablation_study.json"
    with open(p, encoding="utf-8") as f:
        cells = json.load(f)
    fr_lb = frontier_lb_by_tier()

    # ---------- Fig A: accuracy vs cost ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    tiers = ["easy", "medium", "hard"]
    for ax, tier in zip(axes, tiers):
        af_vals = []
        for pair, bm, _ in discover():
            if TIER_OF.get(bm) != tier:
                continue
            s = cells.get(f"{pair}_{bm}")
            if not s:
                continue
            c2 = s["cost_per_1k"].get("pcr2")
            cc = s["cost_per_1k"].get("always_cheap_a")
            a2 = s["acc"]["pcr2"] * 100
            ac = s["acc"]["always_cheap_a"] * 100
            if c2 is not None:
                ax.scatter(c2, a2, s=55, color=TIER_COLOR[tier], zorder=3,
                           edgecolor="white", linewidth=.6)
                ax.annotate(pair.replace("pair", "P"), (c2, a2), fontsize=7,
                            xytext=(4, 3), textcoords="offset points")
            if cc is not None:
                ax.scatter(cc, ac, s=30, color=TIER_COLOR[tier], alpha=.35, marker="s", zorder=2)
            cf = s["cost_per_1k"].get("always_frontier")
            af = s["acc"].get("always_frontier")
            if cf is not None and af is not None:
                ax.scatter(cf, af * 100, s=55, color="#333", marker="x", zorder=4, linewidth=1.8)
                af_vals.append(af * 100)
        # solid line = measured Always-Frontier (gpt-oss-120b solo); shown only
        # where a real solo number exists for the tier.
        if af_vals:
            y = sum(af_vals) / len(af_vals)
            ax.axhline(y, color="#333", ls="--", lw=1, alpha=.8)
            ax.text(0.98, y + 0.8, "always-frontier (gpt-oss-120b, measured)",
                    transform=ax.get_yaxis_transform(), ha="right", fontsize=7, color="#333")
        ax.set_title(f"{tier} tier")
        ax.set_xlabel("$ per 1,000 queries")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("accuracy (%)")
    fig.suptitle("Accuracy vs cost  —  circle = PCR-2, faint square = single cheap model, × = always-frontier",
                 fontsize=11)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig7_tradeoff_cost.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG/'fig7_tradeoff_cost.png'}")

    # ---------- Fig B: accuracy vs latency ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, tier in zip(axes, tiers):
        for pair, bm, _ in discover():
            if TIER_OF.get(bm) != tier:
                continue
            s = cells.get(f"{pair}_{bm}")
            if not s:
                continue
            l2 = s["latency"].get("pcr2")
            a2 = s["acc"]["pcr2"] * 100
            if l2 is not None:
                ax.scatter(l2, a2, s=55, color=TIER_COLOR[tier], zorder=3,
                           edgecolor="white", linewidth=.6)
                ax.annotate(pair.replace("pair", "P"), (l2, a2), fontsize=7,
                            xytext=(4, 3), textcoords="offset points")
            lf = s["latency"].get("always_frontier")
            af = s["acc"].get("always_frontier")
            if lf is not None and af is not None:
                ax.scatter(lf, af * 100, s=55, color="#333", marker="x", zorder=4, linewidth=1.8)
        ax.set_title(f"{tier} tier")
        ax.set_xlabel("mean latency per query (s)")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("accuracy (%)")
    fig.suptitle("Accuracy vs latency  —  circle = PCR-2, × = always-frontier", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig8_tradeoff_latency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG/'fig8_tradeoff_latency.png'}")

    # ---------- Fig C: policy ladder for the recommended within-family pair (pair7) ----------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pols = [("always_cheap_a", "1 cheap model"), ("best_cheap", "best of 2 cheap"),
            ("consensus_no_escalation", "consensus, no esc."), ("pcr2", "PCR-2"),
            ("oracle_2cheap", "oracle (either cheap right)")]
    bms = ["arc", "mmlu", "truthfulqa", "mmlu_pro"]
    ncol = len(pols)
    for bmi, bm in enumerate(bms):
        s = cells.get(f"pair7_{bm}")
        if not s:
            continue
        for pi, (pol, _) in enumerate(pols):
            v = s["acc"].get(pol)
            if v is None:
                continue
            ax.bar(bmi * (ncol + 1.2) + pi, v * 100, 0.9,
                   color=plt.cm.viridis(pi / (ncol - 1)))
    ax.set_xticks([bmi * (ncol + 1.2) + (ncol - 1) / 2 for bmi in range(len(bms))])
    ax.set_xticklabels(bms)
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(50, 100)
    ax.set_title("Policy ladder for pair7 (qwen3.6 + qwen3.8, all-Groq) — PCR-2's lift is "
                 "biggest exactly where C-Hall is worst", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=plt.cm.viridis(pi / (ncol - 1)))
               for pi in range(ncol)]
    ax.legend(handles, [lbl for _, lbl in pols], fontsize=8, ncol=2, loc="lower left")
    ax.grid(alpha=.2, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "fig9_policy_ladder.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG/'fig9_policy_ladder.png'}")

    print("\nfrontier lower-bound accuracy by tier (from disagreement subset):",
          {k: round(v, 3) for k, v in fr_lb.items()})


if __name__ == "__main__":
    main()
