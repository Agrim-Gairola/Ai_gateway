"""
ablation_study.py
=================
The full policy ablation the study is really about: for every (pair, benchmark)
checkpoint, score every routing policy on the SAME stored per-question data, so
they are directly comparable. No new API calls for anything except the
Always-Frontier baseline, which is read from exports/solo_sweep.json if present
(frontier models run solo the same way the cheap models are in solo_sweep.py).

Policies
--------
  always_cheap_a / always_cheap_b : one small model, every query
  best_cheap                      : the better of the two, every query
  consensus_no_escalation         : trust agreement, fall back to cheap-A on
                                    disagreement (== always_cheap_a, per paper)
  pcr2                            : trust agreement, ONE frontier call on
                                    disagreement                (the paper's PCR)
  oracle_2cheap                   : right whenever EITHER cheap model is right
                                    (upper bound for any 2-cheap method)
  random_cheap_frontier           : 50/50 cheap-A vs frontier (naive router);
                                    needs frontier solo accuracy
  always_frontier                 : the big model every query; needs solo sweep

Metrics per policy: accuracy, C-Hall (consensus policies), escalation rate,
$/1k queries, mean latency. Rolled up by difficulty tier.

Usage:
    python -m pcr_experiment.ablation_study
    python -m pcr_experiment.ablation_study --export
"""
import json
import argparse
import statistics as st
from pathlib import Path

from .pooled_analysis import CHECKPOINT_DIR, EXPORT_DIR, BENCHMARKS
from .cost_latency_analysis import RATES, rate_for, _toks

TIERS = {
    "easy": ["arc", "openbookqa", "commonsenseqa"],
    "medium": ["mmlu"],
    "hard": ["mmlu_pro", "truthfulqa"],
    "math": ["gsm8k"],
}
TIER_OF = {b: t for t, bs in TIERS.items() for b in bs}


def _norm_rows(data):
    """Yield a uniform per-row dict across the modern and legacy schemas."""
    for r in data["results"]:
        gold = str(r["gold"]).strip().upper()
        ans_a = r.get("ans_a")
        ans_b = r.get("ans_b")
        fa = r.get("frontier_ans")
        ca = ans_a is not None and str(ans_a).strip().upper() == gold
        cb = ans_b is not None and str(ans_b).strip().upper() == gold
        cf = fa is not None and str(fa).strip().upper() == gold
        yield {
            "agree": bool(r["agree"]),
            "ca": ca, "cb": cb, "cf": cf,
            "has_frontier": fa is not None,
            "pcr2_correct": bool(r["correct"]),
            "usage": r.get("usage"),
        }


def _models(data):
    return data.get("cheap_a"), data.get("cheap_b"), data.get("frontier")


def cost_latency_for_policy(rows, models, policy):
    """Approx $/1k and mean latency for a policy, using the usage block when
    present. Returns (usd_per_1k, latency_mean) or (None, None)."""
    ca_m, cb_m, fr_m = models
    ra = rate_for(ca_m or "", False)
    rb = rate_for(cb_m or "", False)
    rf = rate_for(fr_m or "", True)
    u = [r["usage"] for r in rows if r["usage"]]
    if not u:
        return None, None
    n = len(u)

    def toks(call):
        return _toks(call) if call else 0

    ta = [toks(x["cheap_a"]) for x in u]
    tb = [toks(x["cheap_b"]) for x in u]
    tf = [toks(x["frontier"]) for x in u if x.get("frontier")]
    mean_tf = st.mean(tf) if tf else 0
    la = [x["cheap_a"]["latency_seconds"] for x in u]
    lb = [x["cheap_b"]["latency_seconds"] for x in u]
    lf = [x["frontier"]["latency_seconds"] for x in u if x.get("frontier")]
    mean_lf = st.mean(lf) if lf else 0
    esc = [1 if x.get("frontier") else 0 for x in u]

    if policy == "always_cheap_a":
        cost = sum(ta) * ra / 1e6 / n
        lat = st.mean(la)
    elif policy == "always_cheap_b":
        cost = sum(tb) * rb / 1e6 / n
        lat = st.mean(lb)
    elif policy in ("consensus_no_escalation",):
        cost = (sum(ta) * ra + sum(tb) * rb) / 1e6 / n
        lat = st.mean([max(x, y) for x, y in zip(la, lb)])
    elif policy == "pcr2":
        cost = (sum(ta) * ra + sum(tb) * rb + sum(tf) * rf) / 1e6 / n
        lat = st.mean([max(x, y) + (z if e else 0)
                       for x, y, z, e in zip(la, lb, [mean_lf] * n, esc)])
    elif policy == "always_frontier":
        # price one frontier call per query at the observed mean frontier tokens
        cost = mean_tf * rf / 1e6
        lat = mean_lf
    else:
        return None, None
    return cost * 1000, lat


def score_cell(data, frontier_solo_acc=None):
    rows = list(_norm_rows(data))
    n = len(rows)
    agreed = [r for r in rows if r["agree"]]
    models = _models(data)

    acc = {
        "always_cheap_a": sum(r["ca"] for r in rows) / n,
        "always_cheap_b": sum(r["cb"] for r in rows) / n,
        "consensus_no_escalation": sum(r["ca"] for r in rows) / n,
        "pcr2": sum(r["pcr2_correct"] for r in rows) / n,
        "oracle_2cheap": sum(r["ca"] or r["cb"] for r in rows) / n,
    }
    acc["best_cheap"] = max(acc["always_cheap_a"], acc["always_cheap_b"])
    if frontier_solo_acc is not None:
        acc["always_frontier"] = frontier_solo_acc
        acc["random_cheap_frontier"] = 0.5 * acc["always_cheap_a"] + 0.5 * frontier_solo_acc

    chall = {
        "consensus_no_escalation": (len(agreed) - sum(r["ca"] for r in agreed)) / len(agreed) if agreed else None,
    }
    chall["pcr2"] = chall["consensus_no_escalation"]

    esc_rate = sum(1 for r in rows if not r["agree"]) / n

    out = {"n": n, "escalation_rate": esc_rate, "acc": acc, "chall": chall,
           "cost_per_1k": {}, "latency": {}}
    for pol in ("always_cheap_a", "always_cheap_b", "consensus_no_escalation",
                "pcr2", "always_frontier"):
        c, l = cost_latency_for_policy(rows, models, pol)
        if c is not None:
            out["cost_per_1k"][pol] = c
            out["latency"][pol] = l
    return out


def load_frontier_solo(n_hint=80):
    p = EXPORT_DIR / "solo_sweep.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        sweep = json.load(f)
    # label -> {benchmark -> accuracy}
    out = {}
    for label, d in sweep.items():
        accs = {}
        for k, v in d["cells"].items():
            if isinstance(v, dict) and "accuracy" in v:
                bm = k.rsplit("_n", 1)[0]
                accs[bm] = v["accuracy"]
        out[d["model"]] = accs
    return out


def discover():
    found = []
    for path in sorted(CHECKPOINT_DIR.glob("*.json")):
        stem = path.stem
        if any(x in stem for x in ("legacy", "provisional", "partial", "events", "pcr3")):
            continue
        for bm in sorted(BENCHMARKS, key=len, reverse=True):
            if stem.endswith("_" + bm):
                try:
                    import json as _j
                    if len(_j.load(open(path, encoding="utf-8")).get("results", [])) >= 50:
                        found.append((stem[: -len(bm) - 1], bm, path))
                except Exception:
                    pass
                break
    return found


def score_pcr3(data):
    """Score a PCR-3 checkpoint (3 cheap, majority vote, escalate if all differ)."""
    rows = data["results"]
    n = len(rows)
    maj = [r for r in rows if r["agree"]]
    n_ok_maj = sum(r["correct"] for r in maj)
    ra = [_toks(u) for r in rows for u in [r["usage"]["cheap"][0]]]
    # cost: 3 cheap calls always + frontier on no-majority; price all 3 cheap at
    # the mid rate, frontier at frontier rate.
    rc = RATES["_default_cheap"]
    rf = rate_for(data.get("frontier", ""), True)
    tc = sum(sum(_toks(c) for c in r["usage"]["cheap"]) for r in rows)
    tf = sum(_toks(r["usage"]["frontier"]) for r in rows if r["usage"].get("frontier"))
    lat = st.mean([r["usage"]["total_latency_seconds"] for r in rows])
    return {
        "n": n,
        "escalation_rate": sum(1 for r in rows if not r["agree"]) / n,
        "acc": {"pcr3": sum(r["correct"] for r in rows) / n},
        "chall": {"pcr3": (len(maj) - n_ok_maj) / len(maj) if maj else None},
        "cost_per_1k": {"pcr3": (tc * rc + tf * rf) / 1e6 / n * 1000},
        "latency": {"pcr3": lat},
    }


def main(export=False):
    frontier_solo = load_frontier_solo()
    cells = {}
    for pair, bm, path in discover():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        fr_model = (data.get("frontier") or "").split(":")[-1]
        fr_acc = None
        for k, accs in frontier_solo.items():
            if k.split("/")[-1] == fr_model or k == fr_model:
                fr_acc = accs.get(bm)
        cells[(pair, bm)] = score_cell(data, fr_acc)

    # PCR-3 checkpoints (different schema: pcr3_<tag>_<bm>.json)
    for path in sorted(CHECKPOINT_DIR.glob("pcr3_*.json")):
        stem = path.stem
        for bm in sorted(BENCHMARKS, key=len, reverse=True):
            if stem.endswith("_" + bm):
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                # skip partial PCR-3 cells (the hard-benchmark runs died on the
                # gpt-oss-20b daily cap at n=1/5 -- Phase 3b finishes them)
                if len(data.get("results", [])) >= 50:
                    cells[("pcr3", bm)] = score_pcr3(data)
                break

    POLS = ["always_cheap_a", "always_cheap_b", "best_cheap",
            "consensus_no_escalation", "pcr2", "oracle_2cheap",
            "random_cheap_frontier", "always_frontier"]

    print("=" * 118)
    print("POLICY ABLATION  --  accuracy (%) per (pair, benchmark), same stored data scored every way")
    print("=" * 118)
    print(f"{'pair/bench':<20}{'esc%':>6}", end="")
    for p in POLS:
        print(f"{p.replace('always_','A-').replace('_',' ')[:11]:>12}", end="")
    print()
    print("-" * 118)
    for (pair, bm), s in cells.items():
        print(f"{pair + '/' + bm:<20}{s['escalation_rate'] * 100:>5.0f}%", end="")
        for p in POLS:
            v = s["acc"].get(p)
            print(f"{(f'{v * 100:.1f}' if v is not None else '--'):>12}", end="")
        print()
    print("-" * 118)
    print("A-frontier / random need exports/solo_sweep.json with the frontier model run solo; '--' = not available yet")

    # tier roll-up: PCR-2 vs best-cheap vs always-frontier, + cost/latency
    print("\n" + "=" * 118)
    print("BY DIFFICULTY TIER  --  mean across cells (PCR-2 vs the baselines)")
    print("=" * 118)
    print(f"{'tier':<8}{'pair':<16}{'PCR2 acc':>9}{'bestcheap':>10}{'A-front':>9}"
          f"{'PCR2 Chall':>11}{'esc%':>6}{'$/1k PCR2':>10}{'$/1k front':>11}{'lat PCR2':>9}")
    print("-" * 118)
    by = {}
    for (pair, bm), s in cells.items():
        t = TIER_OF.get(bm, "?")
        by.setdefault((t, pair), []).append(s)
    for (t, pair), ss in sorted(by.items()):
        def m(f):
            xs = []
            for s in ss:
                try:
                    v = f(s)
                except (KeyError, TypeError):
                    v = None
                if v is not None:
                    xs.append(v)
            return st.mean(xs) if xs else None
        pcr2 = m(lambda s: s["acc"].get("pcr2") or s["acc"].get("pcr3"))
        bc = m(lambda s: s["acc"].get("best_cheap"))
        af = m(lambda s: s["acc"].get("always_frontier"))
        ch = m(lambda s: s["chall"].get("pcr2") or s["chall"].get("pcr3"))
        esc = m(lambda s: s["escalation_rate"])
        c2 = m(lambda s: s["cost_per_1k"].get("pcr2") or s["cost_per_1k"].get("pcr3"))
        cf = m(lambda s: s["cost_per_1k"].get("always_frontier"))
        l2 = m(lambda s: s["latency"].get("pcr2") or s["latency"].get("pcr3"))
        def f(x, suf=""):
            return f"{x * 100:.1f}{suf}" if x is not None else "--"
        print(f"{t:<8}{pair:<16}{f(pcr2):>9}{f(bc):>10}{f(af):>9}{f(ch):>11}"
              f"{f(esc,'%'):>6}{('$%.3f' % c2) if c2 else '--':>10}"
              f"{('$%.3f' % cf) if cf else '--':>11}{('%.2fs' % l2) if l2 else '--':>9}")

    if export:
        EXPORT_DIR.mkdir(exist_ok=True)
        out = {f"{p}_{b}": s for (p, b), s in cells.items()}
        with open(EXPORT_DIR / "ablation_study.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nExported -> {EXPORT_DIR / 'ablation_study.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()
    main(export=a.export)
