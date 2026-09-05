#!/usr/bin/env bash
# Phase 3b -- run after the NEXT daily reset (Groq gpt-oss TPD + Gemini 500/day,
# ~00:00 America/Los_Angeles). Finishes the pieces blocked on 2026-09-03:
#   - PCR-3 (all-Groq trio) on mmlu_pro + truthfulqa   (died at n=1/5 on the cap)
#   - solo cells still missing: mistral-medium, gemini-flash-lite, gemini-flash,
#     ministral-14b, mistral-small   (Mistral RPM / Gemini quota)
set -u
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"; export PYTHONIOENCODING=utf-8
LOG="pcr_experiment/exports/phase3b.log"; : > "$LOG"
echo "=== phase 3b start $(date) ===" | tee -a "$LOG"

# 1. finish PCR-3 on the hard benchmarks (resumes from the emergency checkpoints)
for bm in mmlu_pro truthfulqa; do
  echo "--- PCR-3 / $bm ---" | tee -a "$LOG"
  $PY -u -m pcr_experiment.pcr3_runner --benchmark "$bm" --n 120 >> "$LOG" 2>&1
done

# 2. retry the solo cells that errored out (slow pacing already set to 1.5s;
#    Mistral still tight -- if mistral-* fails again, drop --n to 40)
$PY -u -m pcr_experiment.solo_sweep --n 80 --benchmarks arc mmlu_pro truthfulqa --retry-errors >> "$LOG" 2>&1

# 3. rebuild
$PY -m pcr_experiment.ablation_study --export >> "$LOG" 2>&1
$PY -m pcr_experiment.trade_off_figures >> "$LOG" 2>&1
echo "=== phase 3b done $(date) ===" | tee -a "$LOG"
