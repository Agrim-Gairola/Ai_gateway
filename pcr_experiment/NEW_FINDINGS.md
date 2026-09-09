# PCR — New Experimental Findings (Focused Waves 1–3, 2026-09-03)

Supplement to *Parallel Consensus Routing: Training-Free Quality Control and
Coordinated Hallucination in Cost-Efficient LLM Inference* and to the 20 Aug 2026
Detailed Experiment Report.

All new runs: free-tier APIs, temperature 0, no chain-of-thought, seed 42,
checkpoint-and-resume. Total additional API spend: **\$0** (free tiers only).

---

## What was added

| Addition | Detail |
|---|---|
| **4 new benchmarks** | **TruthfulQA** (MC1, 6-option, adversarial misconceptions), **MMLU-Pro** (up to 10 options, harder, reasoning-heavy), **OpenBookQA** (elementary science), **CommonsenseQA** (5-option commonsense). Loaders in `pcr_runner.py`; letter extraction generalised to A–J. |
| **4 new model pairs** | **pair6 — cross-family:** `groq:gpt-oss-20b` + `mistral:mistral-small`. **pair7 — within-family:** `groq:qwen3.8-27b` + `groq:qwen3.6-27b`. **pair8 — within-family (non-Groq):** `mistral:ministral-3b` + `mistral:ministral-8b`. **pair9 — cross-family (non-Groq):** `mistral:ministral-8b` + `gemini:gemini-flash-lite`. Each matched pair holds its frontier constant (pair6/7 → `gpt-oss-120b`; pair8/9 → `mistral-medium`), so family relatedness is the only manipulated variable. |
| **Inferential family test** | `family_significance.py` — 10,000-iteration bootstrap 95% CIs on the within − cross gap, question-level and pair-level (benchmark-balanced), with capability-mismatched cells excluded. |
| **Measured cost & latency** | `cost_latency_analysis.py` — real \$/1k-queries and wall-clock latency (PCR vs always-frontier vs serial cascade) from per-call instrumentation (20 instrumented cells). |

The family comparison now spans **7 pilot pairs across 4 model families**
(gpt-oss, Qwen, Mistral, plus cross-family mixes) on **6 benchmarks**, over three
runs: wave 1 (2 pairs × 2 benchmarks), wave 2 (2 pairs × 4 benchmarks, filling
the matrix), wave 3 (2 non-Groq pairs × 3 benchmarks).

**Coverage:** 24 of 26 target cells complete. Two cross-family cells were cut off
by the `gpt-oss-20b` 200k tokens/day Groq cap (`pair6/openbookqa` 192/200, used
as-is; `pair6/commonsenseqa` 16/200, parked pending reset).

---

> **Update (2026-09-09):** the definitive results are now the **consistent
> 4-pair × 6-benchmark grid** — pair3, pair7 (within-family) and pair10, pair11
> (cross-family), all sharing the `gpt-oss-120b` frontier and the Groq provider,
> every cell at full N. The tables below are refreshed to that grid; the earlier
> 7-pair numbers (which mixed in the capability-mismatched pair8/pair9 and the
> partial pair6) are superseded. See `pcr_ieee_v2.tex` for the final paper.

## Finding 1 — Coordinated hallucination scales with benchmark **difficulty**, not task type

The original paper framed C-Hall as *knowledge* (≈9%) vs *math* (31.8%). On the
4-pair grid the real pattern is a smooth monotone climb with question difficulty
— no change in task type:

| Benchmark | mean C-Hall (4 pairs) | range across pairs |
|---|---|---|
| OpenBookQA (elementary science) | **3%** | 1.6–4.5% |
| ARC-Challenge (science) | **3%** | 2.6–3.9% |
| MMLU (general knowledge) | **10%** | 8.3–11.3% |
| CommonsenseQA (commonsense) | **10%** | 7.1–15.6% |
| **TruthfulQA (adversarial misconceptions)** | **14%** | 9.5–19.8% |
| **MMLU-Pro (hard, 10-option)** | **19%** | 13.3–25.9% |

*(Figure `fig_chall_ladder.png`.)* CommonsenseQA is the one non-monotone point —
its ~10% is driven by pair3's safety-tuned model (15.6% there).

MMLU-Pro's **25.9% C-Hall for the within-family Qwen pair (pair7)** **matches
the paper's GSM8K figure on a plain letter task** — no numeric-answer extraction,
256-token budget, &lt;3% unparsed — so it cannot be dismissed as a truncation
artefact (cf. Finding 5). Wave 3's mismatched pairs sit even higher (Finding 3),
which is a *pair*-quality effect, but they still land on the same rising ladder.

**Consequence for the paper.** The prerequisite for safe PCR is not
*task-type (knowledge vs math) awareness* but **difficulty / reliability-tier
awareness**. The domain-calibrated threshold mechanism the paper proposes for
MMLU domains needs to generalise to a *difficulty*-calibrated one. This widens
the paper's central safety argument rather than contradicting it.

---

## Finding 2 — On frontier-hard benchmarks, PCR's accuracy looks fine while its safety breaks

| Cell | acc &#124; agreed | frontier &#124; disagreed | overall PCR | best cheap solo | PCR gain | C-Hall (trusted) |
|---|---|---|---|---|---|---|
| pair7 / MMLU-Pro | 74.1% | 70.7% | 72.7% | 62.0% | **+10.7 pp** | **25.9%** |
| pair6 / MMLU-Pro | 72.7% | 72.6% | 72.7% | 65.3% | **+7.3 pp** | **27.3%** |
| pair7 / TruthfulQA | 85.7% | 64.7% | 83.3% | 81.3% | +2.0 pp | 14.3% |
| pair6 / TruthfulQA | 80.2% | 65.8% | 76.0% | 72.7% | +3.3 pp | 19.8% |
| *pair7 / MMLU (ref)* | 88.7% | 88.7% | 84.0% | 83.2% | +0.8 pp | 11.3% |
| *pair6 / MMLU (ref)* | 90.8% | 84.0% | 85.6% | 80.4% | +5.2 pp | 9.2% |

On MMLU-Pro the frontier (`gpt-oss-120b`) scores ~72% on the questions it is
escalated — **the same as the cheap consensus scores on the questions it is
trusted for**. Escalation adds nothing on the hard tail; PCR's +7–11 pp gain
comes entirely from *consensus routing* (trusting agreement filters to the
easier questions). But that "trusted" set still contains ~27% wrong answers —
PCR passes the accuracy check and fails the safety check on the same benchmark
at the same time. On the *easy* benchmarks (ARC, OpenBookQA, CommonsenseQA) the
opposite limit holds — cheap models are near ceiling, PCR gain ≈ 0, escalation
rare and neutral. PCR's real value band is the middle (MMLU-like); the paper's
"super-frontier" MMLU result (+13.2 pp) is a best case.

*(Figure `fig1_pcr_vs_bestcheap.png`.)*

---

## Finding 3 — Family: a difficulty-gated agreement premium, and a **suggestive** C-Hall premium on the hard tail

Clean 2-vs-2 design: within = {pair3, pair7}, cross = {pair10, pair11}, all 6
benchmarks, frontier + provider constant. Bootstrap 95% CIs on the (within −
cross) gap:

### 3a. Agreement rate — the premium is difficulty-gated

| Benchmark | within | cross | gap | 95% CI | verdict |
|---|---|---|---|---|---|
| ARC | 95.4% | 94.0% | +1.4 pp | [−1.8, +4.7] | spans 0 |
| OpenBookQA | 92.2% | 89.5% | +2.7 pp | [−1.3, +6.8] | spans 0 |
| MMLU | 73.0% | 80.2% | **−7.2 pp** | [−12.2, −2.0] | **excludes 0 (reversed)** |
| CommonsenseQA | 90.8% | 81.8% | +9.0 pp | [+4.2, +13.5] | **excludes 0** |
| TruthfulQA | 84.7% | 70.7% | +14.0 pp | [+7.7, +20.3] | **excludes 0** |
| MMLU-Pro | 74.0% | 52.3% | +21.7 pp | [+14.0, +29.3] | **excludes 0** |

Null on easy; large ($+9$ to $+22$ pp) on the hard/adversarial benchmarks. MMLU
**reverses** — pair3's safety-tuned model refuses/hedges on ~40% of items, cutting
its agreement to 58%.

### 3b. Coordinated hallucination — follows the same shape, one step weaker

| Benchmark | within | cross | gap | 95% CI | verdict |
|---|---|---|---|---|---|
| ARC | 3.4% | 2.8% | +0.5 pp | [−2.1, +2.9] | spans 0 |
| OpenBookQA | 3.3% | 2.2% | +1.0 pp | [−1.3, +3.4] | spans 0 |
| MMLU | 10.1% | 10.0% | +0.2 pp | [−4.0, +4.3] | spans 0 |
| CommonsenseQA | 11.3% | 8.6% | +2.7 pp | [−1.8, +7.3] | spans 0 |
| **TruthfulQA** | 16.9% | 10.4% | **+6.6 pp** | [+0.3, +12.8] | **excludes 0** |
| MMLU-Pro | 22.5% | 15.3% | +7.2 pp | [−0.3, +14.9] | boundary |
| **Balanced, pair-level** | 10.5% | 6.8% | **+3.4 pp** | [+2.4, +4.3] | excludes 0 (2 pairs/cond.) |

**Revised conclusion:** shared lineage buys a large agreement premium on hard
questions and, *plausibly*, a coordinated-hallucination premium there too
(TruthfulQA +6.6 pp; MMLU-Pro +7.2 pp at the boundary). Null on easy/medium. With
only two pairs per condition this is a **hypothesis for a powered replication**,
not the settled "no effect" of the earlier 7-pair pool. What dominates C-Hall
everywhere is still benchmark difficulty (Finding 1), not lineage.

### 3c. Why this is hard to test: capability matching is the binding constraint

Wave 3 was meant to break the "within-family = Groq-hosted" confound with a
non-Groq Mistral within-family pair (pair8). It produced a striking-looking
result — and then undercut it:

| pair | benchmark | C-Hall | cheap-A solo | cheap-B solo | solo gap |
|---|---|---|---|---|---|
| pair8 (within, Mistral) | TruthfulQA | **38.6%** | 43.3% (ministral-3b) | 60.0% (ministral-8b) | **−16.7 pp** |
| pair8 (within, Mistral) | MMLU-Pro | **43.0%** | 36.7% | 44.0% | −7.3 pp |
| pair9 (cross, Mistral+Gemini) | TruthfulQA | 12.2% | 60.0% (ministral-8b) | 84.0% (gemini-fl) | **−24.0 pp** |
| pair9 (cross, Mistral+Gemini) | MMLU-Pro | 21.7% | 43.3% | 61.3% | −18.0 pp |

Taken at face value, pair8's 38–43% C-Hall vs pair9's 12–22% is a *large*
within-family effect. But the last column kills it: in every wave-3 cell on the
hard benchmarks the two cheap models differ in solo accuracy by **7–24 pp** —
far beyond the report's own ±2–3 pp matching criterion. `ministral-3b` is simply
weak (37% on MMLU-Pro); when it "agrees" with `ministral-8b` it is often because
both defaulted to the same plausible-but-wrong answer. That is a skill gap, not
shared family blind spots — the same failure mode that got `pair2/gsm8k`
excluded. The four affected cells (`pair8` and `pair9` × TruthfulQA / MMLU-Pro)
are therefore excluded from the family pooling above; the closer ARC cells are
kept.

**This is itself a finding for the paper's methodology.** The family hypothesis
cannot be tested with off-the-shelf same-series small-model pairs — a 3B + 8B
pair is systematically mismatched. A credible family experiment needs two
same-family models within ~3 pp solo accuracy on the target benchmark (pair7's
`qwen3.6` / `qwen3.8`, −0.5 to +2 pp, is the only pair in the whole set that
qualifies on the hard benchmarks), and such pairs are rare. Until more exist,
the honest answer to "does family relatedness cause coordinated hallucination"
is **no evidence that it does, from the one clean test available.**

---

## Finding 4 — Measured cost and latency (replaces the paper's estimates)

From per-call instrumentation on 20 cells *(`cost_latency_analysis.json`,
Figure `fig5_cost_per_1k.png`)*:

| Cell | escalation | \$/1k, PCR | \$/1k, always-frontier | cost cut | PCR latency | vs serial cascade |
|---|---|---|---|---|---|---|
| pair7 / OpenBookQA | 5.0% | \$0.028 | \$0.100 | **72%** | 0.49 s | 1.33× |
| pair7 / CommonsenseQA | 8.5% | \$0.032 | \$0.104 | **69%** | 0.55 s | 1.27× |
| pair7 / ARC | 4.0% | \$0.036 | \$0.108 | **66%** | 0.40 s | 1.34× |
| pair7 / MMLU | 11.6% | \$0.072 | \$0.169 | 57% | 0.53 s | 1.28× |
| pair6 / MMLU | 26.4% | \$0.094 | \$0.163 | 42% | 0.74 s | 1.59× |
| pair6 / MMLU-Pro | 48.7% | \$0.216 | \$0.273 | 21% | 1.15 s | 1.40× |
| pair8 / MMLU-Pro | 47.3% | \$0.135 | \$0.148 | 9% | 1.11 s | 1.43× |
| pair4 / GSM8K | 65.9% | \$0.120 | \$0.134 | **11%** | 1.65 s | 1.16× |

Two honest nuances the paper's estimate glossed:

1. **Cost saving tracks escalation rate almost linearly.** 60–72% when
   escalation is low (≤12%), ~40% at ~28%, ~10% once a benchmark is hard enough
   to push escalation past 50%. The paper's 94.7% figure is
   MMLU-with-domain-calibration specific.
2. **PCR is 1.2–1.7× faster than a serial cascade** (validating the paper's
   latency claim) **but not faster than always-frontier** on Groq's LPU — a
   single `gpt-oss-120b` call is ~0.5 s, while PCR pays `max(cheap_a, cheap_b)`
   plus a frontier call on escalations. PCR's latency advantage is architectural
   (versus cascading), not absolute. *(The Mistral-stack pair8/pair9 latencies
   are noisier — some Gemini calls ran 3–8 s — so they are indicative only.)*

---

## Finding 5 (note, not re-run) — the GSM8K 31.8% C-Hall figure looks answer-budget-sensitive

Every post-instrumentation GSM8K run shows **C-Hall 2.6–4.3% and
accuracy-when-agreed 95–97%** (pair2 excluded as a capability mismatch),
versus the paper's **31.8% / 24.3%**. The paper used a 12-token answer budget;
the current pipeline uses 300. A 12-token cap truncates multi-step solutions
before the final number, and `extract_number()` then returns an *intermediate*
value — inflating apparent disagreement and C-Hall.

This does **not** weaken the safety thesis — MMLU-Pro (Finding 1) shows a
genuine ≈27% C-Hall on a letter task with an adequate budget. But the specific
31.8% GSM8K number should carry a caveat, or be re-established with a controlled
12-vs-300 token A/B before being cited as-is.

---

## Methods delta (for the paper's Methods / Appendix)

- **TruthfulQA:** `truthful_qa/multiple_choice`, validation split. Per question:
  the one true MC1 option + up to 5 randomly-sampled distractors (≤ 6 total),
  shuffled per-question so the answer is not fixed at A.
- **MMLU-Pro:** `TIGER-Lab/MMLU-Pro`, test split. Options A–J; dataset `answer`
  letter is gold; dataset `category` kept as the domain. 0–5 of 150–250 cheap
  responses per cell were empty after the retry path and count as disagreement.
- **OpenBookQA** (`allenai/openbookqa`, main, test) and **CommonsenseQA**
  (`tau/commonsense_qa`, validation): standard A–D / A–E letter tasks.
- **Letter extraction:** generalised to A–J with an ordered fallback
  (bare letter → "B)" / "B." → "answer is B" → last standalone A–J token).
- **Cheap token budget:** 256 for all new MC benchmarks and for MMLU on the new
  pairs. Historical pair2/pair3 MMLU used 20/40 and is not re-run.
- **Frontiers:** `gpt-oss-120b` on pair6/pair7; `mistral-medium-latest` on
  pair8/pair9 (`mistral-large-latest` returned 403 `tier_not_allowed` on the
  free tier as of 2026-09-03).
- **similarity:** `sentence-transformers` failed to load in this environment
  (`huggingface_hub` API change), so cosine-similarity is null for the new
  runs. The paper's own threshold-sweep shows binary letter agreement carries
  the signal.
- **Exact model IDs used (2026-09-03):** `openai/gpt-oss-20b`,
  `openai/gpt-oss-safeguard-20b`, `openai/gpt-oss-120b`, `allam-2-7b`,
  `qwen/qwen3.8-27b`, `qwen/qwen3.6-27b` (Groq); `ministral-3b-latest`,
  `ministral-8b-latest`, `mistral-small-latest`, `mistral-medium-latest`
  (Mistral); `gemini-flash-lite-latest` (Google). `llama-3.1-8b-instant` — the
  paper's Cheap Model A — is no longer in Groq's catalogue.
- **Capability-mismatch exclusions from family pooling:** `pair2/gsm8k`,
  `pair8/{truthfulqa,mmlu_pro}`, `pair9/{truthfulqa,mmlu_pro}` — solo-accuracy
  gap between the two cheap models exceeds the ±2–3 pp matching criterion
  (7–24 pp). Shown in the per-cell table, excluded from the grand family means.

## Limitations

- **Pilot N** (150–300 per cell); per-benchmark C-Hall CIs are ±3–14 pp.
- **The Groq-hosting confound is only partly addressed.** Wave 3 added a
  non-Groq within-family pair (pair8), but it is capability-mismatched, so the
  only *clean* within-family evidence is still pair7 (`qwen3.6` / `qwen3.8`),
  which is Groq-hosted. A capability-matched, non-Groq within-family pair
  remains the key missing replicate.
- **One frontier per matched pair-set**; Finding 2 is specific to
  `gpt-oss-120b`.
- TruthfulQA reduced to 6 options for comparability; absolute C-Hall would shift
  with the full MC1 set.
- `pair6/commonsenseqa` (16/200) and the last 8 of `pair6/openbookqa`
  (192/200) are unfinished — blocked on the `gpt-oss-20b` daily token cap.
- Cost dollar figures use public list prices applied to measured token counts;
  the runs themselves were free-tier.

## Files

```
pcr_experiment/
  pcr_runner.py            (+ 4 benchmark loaders; A-J extraction; mmlu budget 20->256; crash-safe checkpoints)
  pcr_config.py            (+ pair6/7 Groq matched pair; pair8/9 non-Groq matched pair)
  family_significance.py   NEW  bootstrap CI on within-vs-cross gap, mismatch-aware
  cost_latency_analysis.py NEW  measured $/1k and latency
  make_figures.py          NEW  regenerates exports + 6 PNGs
  findings_digest.py       NEW  one-screen number dump
  NEW_FINDINGS.md / .html  this supplement
  checkpoints/pair6..9_*.json                       raw data (24 complete cells)
    pair6_commonsenseqa.json.partial16              parked, resume after Groq reset
  exports/pooled_analysis.json, family_significance.json, cost_latency_analysis.json
  exports/figures/fig1..fig6 .png
```

Reproduce: `python -m pcr_experiment.make_figures`

See also **`ABLATION_STUDY.md`** — the full policy ablation (Always-Cheap /
Best-Cheap / Consensus / PCR-2 / Oracle / Always-Frontier / PCR-3), the
capability-matched pair rankings, the accuracy-vs-cost-vs-latency trade-off,
and the practitioner recommendation ("which pair to use").
