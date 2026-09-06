# -*- coding: utf-8 -*-
"""Emit LaTeX table bodies for the full pair x benchmark grid from checkpoints.
Prints two \\begin{tabular} bodies: C-Hall grid and agreement grid.
Cells with <50 questions -> '---' (pending)."""
import json, os, sys

CK = "pcr_experiment/checkpoints"
PAIRS = ["pair1","pair2","pair3","pair4","pair5","pair6","pair7","pair8","pair9","pair10","pair11"]
BM = [("arc","ARC"),("mmlu","MMLU"),("mmlu_pro","M-Pro"),("truthfulqa","TQA"),
      ("openbookqa","OBQA"),("commonsenseqa","CSQA")]
LABEL = {
 "pair1":"1$^{\\dagger}$","pair2":"2","pair3":"3\\,{\\footnotesize(w)}","pair4":"4",
 "pair5":"5$^{\\dagger}$","pair6":"6","pair7":"7\\,{\\footnotesize(w)}",
 "pair8":"8\\,{\\footnotesize(w)}","pair9":"9","pair10":"10","pair11":"11",
}

def stats(p, b):
    f = os.path.join(CK, f"{p}_{b}.json")
    if not os.path.exists(f):
        return None
    try:
        r = json.load(open(f, encoding="utf-8")).get("results", [])
    except Exception:
        return None
    if len(r) < 50:
        return None
    n = len(r)
    ag = [x for x in r if x["agree"]]
    okag = sum(x["correct"] for x in ag)
    acc = sum(x["correct"] for x in r) / n
    return dict(agree=len(ag)/n*100,
                chall=((len(ag)-okag)/len(ag)*100 if ag else 0),
                acc=acc*100)

def grid(metric):
    lines = []
    for p in PAIRS:
        cells = []
        for key, _ in BM:
            s = stats(p, key)
            if s is None:
                cells.append("---")
            elif metric == "chall":
                cells.append(f"{s['chall']:.1f}")
            elif metric == "agree":
                cells.append(f"{s['agree']:.0f}")
            elif metric == "acc":
                cells.append(f"{s['acc']:.1f}")
        # skip fully-empty rows for pair1/pair5 but keep a stub
        if all(c == "---" for c in cells):
            lines.append(f"{LABEL[p]} & " + " & ".join(cells) + r" \\")
        else:
            lines.append(f"{LABEL[p]} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)

hdr = "\\textbf{Pair} & " + " & ".join(f"\\textbf{{{lab}}}" for _, lab in BM) + r" \\"
print("% ===== C-HALL GRID (%) =====")
print(hdr); print(r"\midrule")
print(grid("chall"))
print()
print("% ===== AGREEMENT GRID (%) =====")
print(hdr); print(r"\midrule")
print(grid("agree"))
print()
print("% ===== PCR-2 ACCURACY GRID (%) =====")
print(hdr); print(r"\midrule")
print(grid("acc"))
