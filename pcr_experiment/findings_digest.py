"""
findings_digest.py -- one-screen dump of every number the write-up needs.
Run after make_figures.py (which regenerates the export JSONs).

    python -m pcr_experiment.findings_digest
"""
import json
from pathlib import Path
from .pooled_analysis import CHECKPOINT_DIR, EXPORT_DIR, BENCHMARKS

FAMILY = {"pair2": "cross", "pair4": "cross", "pair6": "cross", "pair9": "cross",
          "pair10": "cross", "pair11": "cross",
          "pair3": "within", "pair7": "within", "pair8": "within"}
MODELS = {
    "pair2": "allam-2-7b + mistral-small (mixed)", "pair3": "gpt-oss-20b + gpt-oss-safeguard-20b",
    "pair4": "allam-2-7b + gemini-flash-lite", "pair6": "gpt-oss-20b + mistral-small",
    "pair7": "qwen3.8-27b + qwen3.6-27b",
    "pair8": "ministral-3b + ministral-8b (Mistral)",
    "pair9": "ministral-8b + gemini-flash-lite",
    "pair10": "gpt-oss-20b + qwen3.6-27b (all-Groq)",
    "pair11": "gpt-oss-20b + qwen3.8-27b (all-Groq)",
}


def load(pair, bm):
    p = CHECKPOINT_DIR / f"{pair}_{bm}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def rates(d):
    r = d["results"]
    agr = [x for x in r if x["agree"]]
    ok = sum(x["correct"] for x in agr)
    def solo(key):
        return sum(1 for x in r if str(x.get(key)).upper() == str(x["gold"]).upper()) / len(r)
    return dict(
        n=len(r), nplan=d.get("n_total", len(r)),
        agree=len(agr) / len(r),
        chall=(len(agr) - ok) / len(agr) if agr else None,
        acc_agr=ok / len(agr) if agr else None,
        pcr=sum(x["correct"] for x in r) / len(r),
        cheap_a=solo("ans_a"), cheap_b=solo("ans_b"),
        best_cheap=max(solo("ans_a"), solo("ans_b")),
    )


def main():
    print("=" * 100)
    print("PER-CELL  (pair / benchmark)")
    print("=" * 100)
    print(f"{'cell':<18}{'fam':<7}{'N':>7}{'agree':>8}{'acc|agr':>9}{'C-Hall':>8}"
          f"{'PCR':>7}{'bestcheap':>10}{'PCR-gain':>9}   models")
    print("-" * 100)
    for pair in ["pair2", "pair3", "pair4", "pair6", "pair7"]:
        for bm in BENCHMARKS:
            d = load(pair, bm)
            if not d or not d["results"]:
                continue
            s = rates(d)
            comp = "" if s["n"] >= s["nplan"] else f"({s['n']}/{s['nplan']})"
            g = (s["pcr"] - s["best_cheap"]) * 100
            ch = "n/a" if s["chall"] is None else f"{s['chall']*100:.1f}%"
            aa = "n/a" if s["acc_agr"] is None else f"{s['acc_agr']*100:.1f}%"
            print(f"{pair+'/'+bm:<18}{FAMILY[pair]:<7}{str(s['n'])+comp:>7}"
                  f"{s['agree']*100:>7.1f}%{aa:>9}{ch:>8}{s['pcr']*100:>6.1f}%"
                  f"{s['best_cheap']*100:>9.1f}%{g:>+8.1f}p   {MODELS[pair]}")

    for name in ["pooled_analysis", "family_significance", "cost_latency_analysis"]:
        p = EXPORT_DIR / f"{name}.json"
        if not p.exists():
            continue
        print("\n" + "=" * 100)
        print(name.upper())
        print("=" * 100)
        print(p.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
