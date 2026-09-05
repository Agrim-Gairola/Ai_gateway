#!/usr/bin/env bash
# Phase 3 of the ablation study -- run after the Groq gpt-oss token cap and the
# Gemini 500/day quota reset (both ~00:00 America/Los_Angeles).
# Fills: exact Always-Frontier accuracy, PCR-3, and the missing solo cells.
set -u
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8
LOG="pcr_experiment/exports/phase3.log"; : > "$LOG"

echo "=== phase 3 start $(date) ===" | tee -a "$LOG"

# 1. frontier + remaining cheap solo cells (solo_sweep skips cached cells)
$PY -u -m pcr_experiment.solo_sweep --n 80 --benchmarks arc mmlu_pro truthfulqa >> "$LOG" 2>&1

# 2. PCR-3 on the difficulty ladder
for bm in arc mmlu_pro truthfulqa; do
  echo "--- PCR-3 / $bm ---" | tee -a "$LOG"
  $PY -u -m pcr_experiment.pcr3_runner --benchmark "$bm" --n 120 >> "$LOG" 2>&1
done

# 3. rebuild analysis + figures
$PY -m pcr_experiment.ablation_study --export >> "$LOG" 2>&1
$PY -m pcr_experiment.trade_off_figures >> "$LOG" 2>&1
$PY -m pcr_experiment.make_figures >> "$LOG" 2>&1
echo "=== phase 3 done $(date) ===" | tee -a "$LOG"
