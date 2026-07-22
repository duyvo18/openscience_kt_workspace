#!/usr/bin/env bash
# Run every full-model training + the ablation grid, sequentially, fast first.
# Skips a run if its test_metrics.json already exists. Safe to re-run / resume.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=./venv/bin/python

run() {  # run <dataset> <ablation>
  local ds="$1" ab="$2" tag
  tag=$([ "$ab" = full ] && echo "${ds}_full" || echo "${ds}_${ab}")
  if [ -f "runs-50-epochs/${tag}/test_metrics.json" ]; then
    echo "SKIP ${tag} (done)"; return; fi
  echo ">>> TRAIN ${tag}"
  $PY scripts/train.py --dataset "$ds" --ablation "$ab" 2>&1 | tee "runs-50-epochs/${tag}.log" | grep -E "epoch|TEST|early" || true
}

# 1) full model on every dataset, cheapest first
for ds in assist09 algebra05 bridge06 xes3g5m assist12 eedi junyi; do
  run "$ds" full
done

# 2) ablation grid on the two representative datasets
for ds in assist09 xes3g5m; do
  for ab in no_temporal no_samekc no_prereq no_neighbor no_mono no_gs no_distributional single_branch; do
    run "$ds" "$ab"
  done
done

echo "ALL DONE"
