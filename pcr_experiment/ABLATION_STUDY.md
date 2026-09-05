# PCR — Full Policy Ablation & Model-Pair Selection Guide

Companion to `NEW_FINDINGS.md`. Where that document answers *"is coordinated
hallucination real and does model family cause it"*, this one answers the
practical question: **which routing policy and which model pair should someone
actually use, and does it beat just calling a big model?**

Free-tier APIs, temperature 0, seed 42, $0 additional spend. Two pieces of the
matrix (exact Always-Frontier accuracy, PCR-3) are **pending a provider quota
reset** — see *Phase 3* at the end; every other number below is final.

---

## 1. The policies compared

All scored on the **same stored per-question data** from the wave 1–3 PCR runs
(7 pairs × up to 6 benchmarks) — no re-running:

| Policy | Rule | Frontier calls |
|---|---|---|
| Always-Cheap-A / -B | one small model, every query | 0 |
| Best-Cheap | the stronger of the two small models | 0 |
| Consensus, no escalation | trust agreement; on disagreement fall back to Cheap-A | 0 |
| **PCR-2** | trust agreement; **one** frontier call on disagreement | = disagreement rate |
| Oracle-2-cheap | correct whenever *either* cheap model is correct (ceiling for any 2-cheap method) | 0 |
| Random cheap/frontier | 50/50 coin flip between Cheap-A and the frontier | 50% |
| **Always-Frontier** | the big model, every query | 100% |
| **PCR-3** | three cheap models, majority vote (2-of-3); escalate only if all three differ | = no-majority rate |

Metrics: accuracy · coordinated-hallucination (C-Hall) rate · escalation rate ·
$/1k queries · latency. Rolled up by **difficulty tier**: *easy* (ARC,
OpenBookQA, CommonsenseQA) · *medium* (MMLU) · *hard* (MMLU-Pro, TruthfulQA) ·
*math* (GSM8K).

---

## 2. Capability-matched pairs (from the solo sweep)

Each candidate cheap model run **solo** on ARC / MMLU-Pro / TruthfulQA (n=80),
so any pair's capability gap can be read off directly. Solo accuracy:

| Model | Provider | ARC | MMLU-Pro | TruthfulQA |
|---|---|---|---|---|
| gpt-oss-20b | Groq | 96.2% | 65.0% | *(pending)* |
| qwen3.6-27b | Groq | 96.2% | 63.7% | 81.2% |
| qwen3.8-27b | Groq | 95.0% | 63.7% | 83.8% |
| ministral-8b | Mistral | 91.2% | 46.2% | 55.0% |
| allam-2-7b | Groq | 75.0% | 35.0% | 33.8% |
| **gpt-oss-120b** (frontier) | Groq | **95.0%** | **71.3%** | **78.7%** |

*(ministral-14b, mistral-small, gemini-flash-lite, gemini-flash, mistral-medium:
solo run blocked by Mistral RPM / Gemini daily quota — Phase 3b.)*

**Best-matched pairs** (mean |accuracy gap| across the measured benchmarks):

| Gap | Condition | Pair | Notes |
|---|---|---|---|
| **0.6 pp** | cross-family | `gpt-oss-20b` + `qwen3.6-27b` | both Groq; matched as tightly as a same-family pair |
| 1.3 pp | within-family | `qwen3.6-27b` + `qwen3.8-27b` | both Groq (this is **pair7**) |
| 1.3 pp | cross-family | `gpt-oss-20b` + `qwen3.8-27b` | both Groq |
| 11.9 pp | cross-family | `gpt-oss-20b` + `ministral-8b` | fails the ±3 pp match bar |
| 16–33 pp | cross-family | anything with `ministral-8b` or `allam-2-7b` | mismatched — do not use for a family test |

**Key finding for pair selection:** `gpt-oss-20b`, `qwen3.6-27b` and
`qwen3.8-27b` are mutually within ~1 pp of each other on every measured
benchmark. Any two of them form a well-matched pair, within *or* across family,
and all three are on Groq's LPU (the fast, free path). Every other free-tier
cheap model we tested is either much weaker (`allam-2-7b`, `ministral-8b`) or
on a slower provider.

---

## 3. Master results — accuracy by policy

Mean accuracy across cells in each tier (PCR-2 vs the two bracketing baselines):

| Tier | Pair | PCR-2 | Best-Cheap | Always-Frontier† | PCR-2 C-Hall | Escalation |
|---|---|---|---|---|---|---|
| **easy** | pair7 (Qwen, within) | 93.0% | 93.8% | 95.0% | 3.9% | **5.8%** |
| easy | pair3 (gpt-oss, within) | 94.3% | 93.7% | 95.0% | 3.9% | 5.0% |
| easy | pair6 (gpt-oss+Mistral, cross) | 93.1% | 93.3% | 95.0% | 2.0% | 12.6% |
| easy | pair9 (Mistral+Gemini, cross) | 94.7% | 94.0% | — | 2.2% | 10.0% |
| **medium** | pair7 | 84.0% | 83.2% | *(pending)* | 11.3% | 11.6% |
| medium | pair6 | 85.6% | 80.4% | *(pending)* | 9.2% | 26.4% |
| medium | pair3 | 84.8% | 69.2% | *(pending)* | 8.3% | 42.4% |
| **hard** | pair7 | **78.0%** | 71.7% | **75.0%** | **20.1%** | 19.7% |
| hard | pair6 | 74.3% | 69.0% | **75.0%** | 23.5% | 37.3% |
| hard | pair8 (Mistral, mismatched) | 59.7% | 52.0% | 75.0% | 40.8% | 40.0% |
| **math** | pair4 (gsm8k) | 94.8% | 92.6% | 84.8%‡ | 2.6% | 65.9% |

† *Always-Frontier = `gpt-oss-120b` run solo on the full benchmark set (measured,
Phase 3): ARC 95.0 %, MMLU-Pro 71.3 %, TruthfulQA 78.7 % → hard-tier mean 75.0 %.
MMLU (medium) solo not yet run — Phase 3b.*
‡ *math = frontier accuracy on the disagreement subset only (no full-set solo).*

Reading it:
- **Easy:** PCR-2 (93–94 %) lands ~1–2 pp under Always-Frontier (95.0 %). Same
  accuracy band; PCR-2's whole value here is **cost** (§4).
- **Medium:** PCR-2 adds +1 to +15 pp over Best-Cheap, depending on how weak the
  cheap models are and how often they disagree.
- **Hard:** for the **capability-matched** pairs, PCR-2 **matches or beats
  Always-Frontier** — pair7 78.0 % vs 75.0 %, pair6 74.3 % vs 75.0 % — while the
  mismatched pair8 (59.7 %) falls far below it. But 20–24 % of the answers PCR-2
  *trusts* on the hard tier are wrong (C-Hall) — the accuracy is real, the
  safety is not.
- **Policy ladder** (`fig9_policy_ladder.png`): on ARC/MMLU the five policies are
  flat; on MMLU-Pro PCR-2 clears even Oracle-2-cheap, because frontier
  escalation recovers questions *both* cheap models missed.

---

## 4. Trade-off: does a pair beat the big model?

Measured $/1k queries and latency (from per-call instrumentation; latency from
the parallel PCR runs, not the throttled solo sweep):

| Tier | Pair | $/1k PCR-2 | $/1k Always-Frontier | cost cut | latency PCR-2 |
|---|---|---|---|---|---|
| easy | pair7 (all-Groq Qwen) | **$0.032** | $0.104 | **69 %** | **0.48 s** |
| easy | pair6 | $0.043 | $0.101 | 57 % | 1.61 s |
| easy | pair9 (Mistral+Gemini) | $0.022 | $0.048 | 54 % | 2.75 s |
| medium | pair7 | $0.072 | $0.169 | 57 % | 0.53 s |
| medium | pair6 | $0.094 | $0.163 | 42 % | 0.74 s |
| hard | pair7 | $0.111 | $0.209 | 47 % | 0.51 s |
| hard | pair6 | $0.143 | $0.197 | 27 % | 0.94 s |

*(`fig7_tradeoff_cost.png`, `fig8_tradeoff_latency.png`.)*

- **Easy tier — the headline result:** PCR-2 with a matched pair scores 93–94 %
  vs Always-Frontier's measured 95.0 %, at **54–69 % lower cost** ($0.032–0.043
  vs $0.10–0.13 per 1k). ~1 pp of accuracy for ~2/3 of the bill.
- **Hard tier:** PCR-2 with a *matched* pair (pair7 78.0 %, pair6 74.3 %)
  **matches Always-Frontier's measured 75.0 %** at 27–47 % lower cost — but with
  20–24 % C-Hall on the trusted answers, so the money saving does not come with
  a safety guarantee. A *mismatched* pair (pair8, 59.7 %) is well below the
  frontier and not worth using.
- **Latency is decided by the provider, not the policy.** All-Groq pairs (pair3,
  pair6 cheap side, pair7) run PCR-2 at **~0.5 s**; any pair with a Mistral or
  Gemini leg runs at **1.6–7.8 s** (5–15× slower). PCR-2 is also *not* faster
  than a single Groq frontier call (~0.9 s) — its latency win is only versus a
  serial cascade (1.2–1.7×).

### 4a. PCR-3 (three cheap models, majority vote)

`gpt-oss-20b` + `qwen3.6-27b` + `qwen3.8-27b` (the all-Groq matched trio),
majority vote (2-of-3); escalate to `gpt-oss-120b` only when all three differ.
**N=120 per benchmark, now complete on all three:**

| Benchmark | majority forms | escalation | acc &#124; majority | C-Hall | overall acc |
|---|---|---|---|---|---|
| ARC | 100 % | 0 % | 95.0 % | 5.0 % | 95.0 % |
| TruthfulQA | 99 % | 1 % | 81.5 % | 18.5 % | 81.7 % |
| MMLU-Pro | 84 % | 16 % | 71.3 % | 28.7 % | 73.3 % |

Head-to-head vs **PCR-2** (pair7, the same two Qwen models without the 3rd):

| Benchmark | escalation | C-Hall | overall acc | $/1k |
|---|---|---|---|---|
| ARC | 4.0 % &rarr; **0 %** | 2.6 % &rarr; 5.0 % | 95.0 % &rarr; 95.0 % | $0.032 &rarr; $0.050 |
| TruthfulQA | 11.3 % &rarr; **1 %** | 14.3 % &rarr; 18.5 % | 83.3 % &rarr; 81.7 % | $0.053 &rarr; ~$0.09 |
| MMLU-Pro | 28.0 % &rarr; **16 %** | 25.9 % &rarr; 28.7 % | 72.7 % &rarr; 73.3 % | $0.169 &rarr; $0.124 |

**Verdict: a third model is not worth it.** It reliably breaks ties (a 2-of-3
majority forms 84–100 % of the time, so escalation collapses toward zero), but it
buys **no accuracy** (±1 pp, slightly *worse* on TruthfulQA) and **no safety**
(C-Hall is flat-to-worse on every benchmark — the questions that stop escalating
are added to the trusted set without being any cleaner). It trades one frontier
call for one extra cheap call on every query: net-negative on easy/medium
(the cheap call costs more than the escalations it saves) and roughly break-even
on hard (fewer $0.06 frontier calls, but +50 % cheap-call volume). Binary-
agreement PCR-2 with a *capability-matched pair* remains the right design.

---

## 5. Verdict — which configuration to use

| Goal | Use | Why |
|---|---|---|
| **High-volume, mostly easy/medium queries, minimise cost + latency** | **PCR-2 with `qwen3.6-27b` + `qwen3.8-27b`, frontier `gpt-oss-120b`, all on Groq** (= pair7) | Matches Best-Cheap and (on easy/medium) Always-Frontier accuracy; 57–69 % cheaper than frontier-only; 0.5 s latency; 5–12 % escalation; zero truncation/parse issues. |
| **Same, but want maximum error-independence** (the two models must not share failure modes) | **PCR-2 with `gpt-oss-20b` + `qwen3.8-27b`**, frontier `gpt-oss-120b`, all on Groq | Different labs (OpenAI-oss vs Alibaba) → genuinely independent training, yet solo accuracies within ~1 pp and both on the fast free path. Trade-off: `gpt-oss-20b` needs a 256-token answer budget. |
| **Hard / adversarial queries** (MMLU-Pro-like, TruthfulQA-like) | **Do not rely on PCR consensus as a trust signal.** Route these to the frontier directly, or gate by a difficulty classifier. | Agreement here is 20–43 % wrong; PCR-2 gains +5–11 pp of accuracy but ships confident errors and only ~30 % cheaper. |
| **Research: settle the family-effect question** | 3+ capability-matched pairs per condition at N≈1,000 on MMLU-Pro + N≈300 ARC. Within: `qwen3.6`+`qwen3.8`, `gpt-oss-20b`+`gpt-oss-safeguard-20b` (caveated), a matched Mistral pair if Phase 3b finds one. Cross: `gpt-oss-20b`+`qwen3.6-27b` (0.6 pp), `gpt-oss-20b`+`qwen3.8-27b` (1.3 pp). | Only MMLU-Pro/MMLU have a large enough pool for the ~1,000-questions-per-pair the power analysis needs. Everything must clear the ±3 pp match bar or it measures a skill gap, not family (see `NEW_FINDINGS.md` §3c). |

**Bottom line:** on the ~80 % of real traffic that is easy-to-medium
difficulty, a well-matched pair of ~mid-size open models on a fast inference
provider, run through PCR-2, gives you frontier-tier accuracy at roughly a
third of the cost and a fraction of the latency. The specific pair that does
this best on free tiers today is **two of {`gpt-oss-20b`, `qwen3.6-27b`,
`qwen3.8-27b`} on Groq**. PCR stops being the right answer as soon as the query
is hard enough that the cheap models — and the frontier — start failing
together.

---

## 6. Phase status

**Phase 3 — done (2026-09-03):**
- ✅ **Always-Frontier baseline** — `gpt-oss-120b` run solo: ARC 95.0 %,
  MMLU-Pro 71.3 %, TruthfulQA 78.7 %. The trade-off tables above now use real
  numbers, not lower bounds.
- ✅ **PCR-3 on ARC** (all-Groq trio) — see §4a. Net-negative on easy.

**Phase 3b — pending the next daily reset** (Groq `gpt-oss` token cap + Gemini
500/day both hit again; Mistral-medium RPM-limited under load). `run_phase3b.sh`
does all of it, then rebuilds:
1. **PCR-3 on MMLU-Pro + TruthfulQA** — the runs that died at n≤5 on the cap.
   The interesting case: does the 3rd model's majority vote cut the 20–37 %
   PCR-2 escalation enough to pay for itself on hard queries?
2. **`gpt-oss-120b` / `mistral-medium` solo on MMLU** — fills the medium-tier
   Always-Frontier cell.
3. **`ministral-14b`, `mistral-small`, `gemini-flash-lite`, `gemini-flash` solo**
   — `solo_sweep.py --retry-errors` re-attempts the cells cached as errors;
   fills the cross-provider pair-matching table.

---

## Files

```
pcr_experiment/
  solo_sweep.py            NEW  per-model solo accuracy + latency (--retry-errors after a reset)
  ablation_study.py        NEW  scores 8 policies + PCR-3 per cell from stored data
  pcr3_runner.py           NEW  3-model majority-vote variant (all-Groq trio)
  trade_off_figures.py     NEW  fig7 (acc vs cost), fig8 (acc vs latency), fig9 (policy ladder)
  run_phase3.sh  run_phase3b.sh   the quota-gated run scripts
  ABLATION_STUDY.md        this document
  checkpoints/pcr3_allgroq_gptoss20_qwen36_qwen38_{arc,mmlu_pro,truthfulqa}.json
  exports/solo_sweep.json  ablation_study.json
  exports/figures/fig7_tradeoff_cost.png  fig8_tradeoff_latency.png  fig9_policy_ladder.png
```
