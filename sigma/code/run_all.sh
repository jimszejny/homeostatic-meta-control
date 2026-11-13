#!/usr/bin/env bash
# Reproduce every number and figure of the sigma module. Run from sigma/code/.
# Note: predictions.md/.sha256 are NOT regenerated (written once before the first E1-E5 run).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../results/logs ../figures
for s in step0_validate e1_gate e2_mode e3_sweep e4_lockin e5_offswitch figures derived collect tables make_app verify_tex check_keys; do
  start=$(date +%s.%N)
  python3 $s.py > ../results/logs/$s.log 2>&1
  end=$(date +%s.%N)
  printf '%-16s %8.1f s\n' "$s" "$(echo "$end - $start" | bc)" | tee -a ../results/logs/wall_times.txt
done
sha256sum -c ../results/predictions.sha256 --status -- 2>/dev/null || (cd ../results && sha256sum -c predictions.sha256)
