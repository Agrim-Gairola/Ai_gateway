import argparse, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
EXPORT_DIR = ROOT / "exports"
EXPORT_DIR.mkdir(exist_ok=True)

def load(pair, benchmark):
    path = CHECKPOINT_DIR / f"{pair}_{benchmark}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def wilson(x, n, z=1.96):
    if not n: return (0.0, 0.0)
    p=x/n; d=1+z*z/n; c=p+z*z/(2*n)
    m=z*((p*(1-p)/n)+(z*z/(4*n*n)))**0.5
    return ((c-m)/d, (c+m)/d)

def metrics(rows):
    n=len(rows)
    agree=[r for r in rows if r.get("agree") is True]
    disagree=[r for r in rows if r.get("agree") is False]
    def r(x,n): return x/n if n else 0
    ca=sum(bool(x.get("correct_a")) for x in rows)
    cb=sum(bool(x.get("correct_b")) for x in rows)
    cag=sum(bool(x.get("correct")) for x in agree)
    ct=sum(bool(x.get("correct")) for x in rows)
    fc=[x for x in disagree if x.get("frontier_correct") is not None]
    return {
        "agreement": (len(agree),n),
        "escalation": (len(disagree),n),
        "accuracy_when_agreed": (cag,len(agree)),
        "c_hall": (len(agree)-cag,len(agree)),
        "pcr_accuracy": (ct,n),
        "cheap_a": (ca,n),
        "cheap_b": (cb,n),
        "frontier_on_disagreement": (sum(bool(x.get("frontier_correct")) for x in fc),len(fc))
    }

def rate(t):
    return t[0]/t[1] if t[1] else 0

def bootstrap(rows_a, rows_b, fn, iterations=10000, seed=42):
    rng=random.Random(seed); diffs=[]
    for _ in range(iterations):
        a=[rows_a[rng.randrange(len(rows_a))] for _ in rows_a]
        b=[rows_b[rng.randrange(len(rows_b))] for _ in rows_b]
        diffs.append(fn(a)-fn(b))
    diffs.sort()
    return diffs[int(.025*iterations)], diffs[int(.975*iterations)-1]

def chall(rows):
    a=[r for r in rows if r.get("agree") is True]
    return sum(not bool(r.get("correct")) for r in a)/len(a) if a else 0

def acc(rows):
    return sum(bool(r.get("correct")) for r in rows)/len(rows) if rows else 0

def pct(x): return f"{x*100:.1f}%"
def pp(x): return f"{x*100:+.1f} pp"

def config(data):
    r=data["results"][0]
    return (f"A={r.get('cheap_a_provider')}/{r.get('cheap_a_model')} | "
            f"B={r.get('cheap_b_provider')}/{r.get('cheap_b_model')} | "
            f"F={r.get('frontier_provider')}/{r.get('frontier_model')}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--benchmark", required=True, choices=["mmlu","gsm8k","arc"])
    a=ap.parse_args()

    L=load(a.left,a.benchmark); R=load(a.right,a.benchmark)
    lr=L["results"]; rr=R["results"]
    lm=metrics(lr); rm=metrics(rr)

    print("\n"+"="*86)
    print(f"PCR COMPARISON — {a.left.upper()} vs {a.right.upper()} / {a.benchmark.upper()}")
    print("="*86)
    print("LEFT :",config(L))
    print("RIGHT:",config(R))
    print(f"N: {len(lr)} vs {len(rr)}\n")
    print(f"{'Metric':<32}{a.left:>16}{a.right:>16}{'Difference':>18}")
    print("-"*86)

    labels=[
        ("Agreement","agreement"),("Escalation","escalation"),
        ("Accuracy | Agreement","accuracy_when_agreed"),
        ("C-Hall | Agreement","c_hall"),("Overall PCR accuracy","pcr_accuracy"),
        ("Always Cheap A","cheap_a"),("Always Cheap B","cheap_b"),
        ("Frontier | Disagreement","frontier_on_disagreement")
    ]
    exported=[]
    for label,key in labels:
        x=rate(lm[key]); y=rate(rm[key])
        print(f"{label:<32}{pct(x):>16}{pct(y):>16}{pp(x-y):>18}")
        exported.append({"metric":key,"left":x,"right":y,"difference":x-y})

    print("\n"+"-"*86)
    print("95% WILSON CIs")
    print("-"*86)
    for label,key in labels[:5]:
        lx,ln=lm[key]; rx,rn=rm[key]
        l=wilson(lx,ln); r=wilson(rx,rn)
        print(f"{label:<32}{pct(l[0])}-{pct(l[1]):<13}{pct(r[0])}-{pct(r[1])}")

    ch=chall(lr)-chall(rr); ac=acc(lr)-acc(rr)
    chl,chh=bootstrap(lr,rr,chall)
    acl,ach=bootstrap(lr,rr,acc)

    print("\n"+"-"*86)
    print("BOOTSTRAP 95% CI FOR DIFFERENCES (LEFT - RIGHT)")
    print("-"*86)
    print(f"C-Hall difference:       {pp(ch)}  CI [{pp(chl)}, {pp(chh)}]")
    print(f"PCR accuracy difference: {pp(ac)}  CI [{pp(acl)}, {pp(ach)}]")

    print("\n"+"="*86)
    print("INTERPRETATION")
    print("="*86)
    if chl <= 0 <= chh:
        print("C-Hall difference: not clearly distinguishable from zero in this bootstrap.")
    else:
        print("C-Hall difference: bootstrap interval excludes zero in this sample.")
    print("This is a configuration-level comparison, not a causal test of model-family relatedness.")
    print("Differences in model size, capability, provider, frontier model, and model behavior are potential confounders.")

    out={
        "benchmark":a.benchmark,"left_pair":a.left,"right_pair":a.right,
        "left_n":len(lr),"right_n":len(rr),"metrics":exported,
        "bootstrap":{
            "c_hall_difference":ch,"c_hall_difference_95_ci":[chl,chh],
            "pcr_accuracy_difference":ac,"pcr_accuracy_difference_95_ci":[acl,ach]
        }
    }
    outpath=EXPORT_DIR/f"comparison_{a.left}_vs_{a.right}_{a.benchmark}.json"
    with open(outpath,"w",encoding="utf-8") as f: json.dump(out,f,indent=2)
    print(f"\nExported: {outpath}")

if __name__=="__main__": main()
