"""
make_figures.py
===============
Regenerates every analysis export, then renders publication figures (PNG, 200
dpi) into exports/figures/. Safe to re-run any time -- it only reads
checkpoints.

    python -m pcr_experiment.make_figures

Figures
  fig1_pcr_vs_bestcheap   PCR accuracy vs the stronger cheap model, every cell
  fig2_chall_by_benchmark grouped C-Hall bars, within vs cross family
  fig3_agreement_by_bench  grouped agreement bars, within vs cross family
  fig4_family_forest       bootstrap gap (within - cross) +/- 95% CI, per metric
  fig5_cost_per_1k         $/1k queries: PCR vs always-frontier, per instrumented cell
  fig6_newbench_chall      C-Hall on the new benchmarks vs the anchors (if present)
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .pooled_analysis import EXPORT_DIR, pooled_analysis, CHECKPOINT_DIR, BENCHMARKS
from .family_significance import run as family_run
from .cost_latency_analysis import run as cost_run

FIG_DIR = EXPORT_DIR / "figures"
C_WITHIN, C_CROSS = "#4C72B0", "#DD8452"
C_PCR, C_AF, C_BASE = "#4C72B0", "#C44E52", "#8C8C8C"


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG_DIR / (name + '.png')}")


def load(pair, bm):
    p = CHECKPOINT_DIR / f"{pair}_{bm}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def cell_rates(data):
    rows = data["results"]
    agreed = [r for r in rows if r["agree"]]
    n_ok_agr = sum(r["correct"] for r in agreed)
    return {
        "n": len(rows),
        "agree": len(agreed) / len(rows),
        "chall": (len(agreed) - n_ok_agr) / len(agreed) if agreed else None,
        "pcr_acc": sum(r["correct"] for r in rows) / len(rows),
        "best_cheap": max(
            sum(1 for r in rows if str(r.get("ans_a")).upper() == str(r["gold"]).upper()) / len(rows),
            sum(1 for r in rows if str(r.get("ans_b")).upper() == str(r["gold"]).upper()) / len(rows),
        ),
    }


FAMILY = {
    "pair2": "cross", "pair4": "cross", "pair6": "cross", "pair9": "cross",
    "pair10": "cross", "pair11": "cross",
    "pair3": "within", "pair7": "within", "pair8": "within",
}


def main():
    print("regenerating exports...")
    pooled_analysis(export=True)
    fam = family_run(iterations=10000, export=True)
    cost_run(export=True)
    print("\nrendering figures...")

    # gather per-cell rates
    cells = {}
    for pair in FAMILY:
        for bm in BENCHMARKS:
            d = load(pair, bm)
            if d and d["results"]:
                cells[(pair, bm)] = cell_rates(d)

    # ---- fig1: PCR vs best cheap ----
    keys = sorted(cells, key=lambda k: (k[1], k[0]))
    labels = [f"{p}/{b}" for p, b in keys]
    pcr = [cells[k]["pcr_acc"] * 100 for k in keys]
    base = [cells[k]["best_cheap"] * 100 for k in keys]
    x = range(len(keys))
    fig, ax = plt.subplots(figsize=(max(7, len(keys) * 0.7), 4.2))
    ax.bar([i - 0.2 for i in x], base, 0.4, label="best cheap model (solo)", color=C_BASE)
    ax.bar([i + 0.2 for i in x], pcr, 0.4, label="PCR (full)", color=C_PCR)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("PCR accuracy vs the stronger cheap model, per (pair, benchmark)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)
    _save(fig, "fig1_pcr_vs_bestcheap")

    # ---- fig2/3: chall & agreement grouped by benchmark, within vs cross ----
    for metric, fname, title, ylab in [
        ("chall", "fig2_chall_by_benchmark", "Coordinated hallucination by benchmark", "C-Hall rate (%)"),
        ("agree", "fig3_agreement_by_bench", "Agreement rate by benchmark", "agreement rate (%)"),
    ]:
        bms = [b for b in BENCHMARKS if any(k[1] == b for k in cells)]
        win_v, cro_v = [], []
        for b in bms:
            w = [cells[k][metric] for k in cells if k[1] == b and FAMILY[k[0]] == "within" and cells[k][metric] is not None]
            c = [cells[k][metric] for k in cells if k[1] == b and FAMILY[k[0]] == "cross" and cells[k][metric] is not None]
            win_v.append(100 * sum(w) / len(w) if w else 0)
            cro_v.append(100 * sum(c) / len(c) if c else 0)
        xi = range(len(bms))
        fig, ax = plt.subplots(figsize=(max(6, len(bms) * 1.1), 4))
        ax.bar([i - 0.2 for i in xi], win_v, 0.4, label="within-family", color=C_WITHIN)
        ax.bar([i + 0.2 for i in xi], cro_v, 0.4, label="cross-family", color=C_CROSS)
        ax.set_xticks(list(xi))
        ax.set_xticklabels(bms, rotation=20, ha="right")
        ax.set_ylabel(ylab)
        ax.set_title(title + "  (mean of pairs in each condition)")
        ax.legend(fontsize=8)
        _save(fig, fname)

    # ---- fig4: family forest plot (gap +/- CI) ----
    rows = []
    for bm, md in fam.get("per_benchmark", {}).items():
        for mname, v in md.items():
            if v["gap"] is None:
                continue
            rows.append((f"{bm} / {mname}", v["gap"] * 100, v["ci95"][0] * 100, v["ci95"][1] * 100))
    for mname, v in fam.get("benchmark_balanced", {}).items():
        if v["gap"] is not None and v["ci95"][0] is not None:
            rows.append((f"ALL (bal.) / {mname}", v["gap"] * 100, v["ci95"][0] * 100, v["ci95"][1] * 100))
    if rows:
        rows.reverse()
        fig, ax = plt.subplots(figsize=(7, max(3, len(rows) * 0.45)))
        for i, (lab, g, lo, hi) in enumerate(rows):
            ok = (lo > 0) or (hi < 0)
            ax.plot([lo, hi], [i, i], color="#333" if ok else "#999", lw=2)
            ax.plot(g, i, "o", color=C_PCR if ok else "#999", ms=6)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows], fontsize=8)
        ax.set_xlabel("within-family  minus  cross-family   (percentage points)")
        ax.set_title("Family-condition gap with 95% bootstrap CI\n(filled = CI excludes 0)")
        _save(fig, "fig4_family_forest")

    # ---- fig5: cost per 1k ----
    try:
        with open(EXPORT_DIR / "cost_latency_analysis.json", encoding="utf-8") as f:
            cl = json.load(f)["cells"]
    except FileNotFoundError:
        cl = {}
    if cl:
        ks = list(cl)
        pcrc = [cl[k]["cost_per_1k_queries_usd"]["pcr"] for k in ks]
        afc = [cl[k]["cost_per_1k_queries_usd"]["always_frontier"] for k in ks]
        xi = range(len(ks))
        fig, ax = plt.subplots(figsize=(max(6, len(ks) * 1.1), 4))
        ax.bar([i - 0.2 for i in xi], afc, 0.4, label="always-frontier", color=C_AF)
        ax.bar([i + 0.2 for i in xi], pcrc, 0.4, label="PCR", color=C_PCR)
        ax.set_xticks(list(xi))
        ax.set_xticklabels(ks, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("USD per 1,000 queries (list-price tokens)")
        ax.set_title("Measured cost: PCR vs always-frontier")
        ax.legend(fontsize=8)
        _save(fig, "fig5_cost_per_1k")

    # ---- fig6: C-Hall difficulty ladder across all knowledge benchmarks ----
    ladder = [b for b in ("openbookqa", "arc", "commonsenseqa", "mmlu", "truthfulqa", "mmlu_pro")
              if any(k[1] == b for k in cells)]
    if ladder:
        win_v, cro_v, all_v = [], [], []
        for b in ladder:
            w = [cells[k]["chall"] for k in cells if k[1] == b and FAMILY[k[0]] == "within" and cells[k]["chall"] is not None]
            c = [cells[k]["chall"] for k in cells if k[1] == b and FAMILY[k[0]] == "cross" and cells[k]["chall"] is not None]
            a = w + c
            win_v.append(100 * sum(w) / len(w) if w else None)
            cro_v.append(100 * sum(c) / len(c) if c else None)
            all_v.append(100 * sum(a) / len(a) if a else 0)
        xi = list(range(len(ladder)))
        fig, ax = plt.subplots(figsize=(8, 4.4))
        ax.bar([i - 0.19 for i in xi], [v or 0 for v in win_v], 0.38, label="within-family", color=C_WITHIN)
        ax.bar([i + 0.19 for i in xi], [v or 0 for v in cro_v], 0.38, label="cross-family", color=C_CROSS)
        ax.plot(xi, all_v, "-o", color="#333", lw=1.3, ms=4, label="mean (all pairs)")
        for i, v in enumerate(all_v):
            ax.annotate(f"{v:.0f}%", (i, v), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8, color="#333")
        ax.set_xticks(xi)
        ax.set_xticklabels([b.replace("_", "-") for b in ladder], rotation=15, ha="right")
        ax.set_ylabel("C-Hall rate on agreed questions (%)")
        ax.set_title("Coordinated hallucination scales with benchmark difficulty")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_ylim(0, max(all_v) * 1.25)
        _save(fig, "fig6_newbench_chall")

    print("\ndone.")


if __name__ == "__main__":
    main()
