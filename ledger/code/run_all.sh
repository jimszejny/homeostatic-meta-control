#!/usr/bin/env bash
# Reproduce every number of the REWARD module. Run from code/. Writes ../results and ../figures.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../results/logs ../figures
python3 step0_validate.py | tee ../results/logs/step0.log
grep -q "STEP0 PASS" ../results/logs/step0.log || { echo "Step 0 failed"; exit 1; }
python3 r1_shaping.py   | tee ../results/logs/r1.log
python3 r2_battery.py   > ../results/logs/r2.log
python3 r3_ledger.py    | tee ../results/logs/r3.log
python3 r4_death.py     | tee ../results/logs/r4.log
# read-only use of the earlier package (skipped if absent)
if [ -d /tmp/claude-0/-home-claude/0b0a5afe-fb2d-5015-8d90-aaae8c596e01/scratchpad/synth/v22/paper/code ]; then
  PYTHONDONTWRITEBYTECODE=1 python3 m2_alignment.py | tee ../results/logs/m2_alignment.log
fi
python3 posthoc.py     | tee ../results/logs/posthoc.log
python3 figures.py 2>&1 | grep -v timestamp
python3 merge_results.py
python3 verify_numbers.py
[ -d ../../paper ] && ./compile_check.sh || true
