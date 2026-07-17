#!/usr/bin/env bash
# Single controller for all pending training, with a HARD cap of 3 concurrent
# jobs + a RAM gate (only launch when >=12GB available). This prevents the
# memory exhaustion that OOM-killed eedi_full/junyi_full when two schedulers
# (run_all + a parallel sweep) competed for RAM. Skips any run whose
# test_metrics.json already exists, so it is safe to re-run / resume.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=./venv/bin/python
MAXJOBS=3
GATE_GB=12
log(){ echo "[$(date +%H:%M:%S)] $*"; }
avail(){ free -g | awk '/^Mem:/{print $7}'; }

# Priority-ordered task list. Format: "run_name|||<train.py args>"
TASKS=(
  # --- Phase A: assist09 improvement sweep (fast; answers the open question) ---
  "exp_smallreg|||--dataset assist09 --epochs 25 --overrides run_name=exp_smallreg batch_size=64 d_model=64 d_z=32 dropout=0.4 weight_decay=0.0003"
  "exp_wd|||--dataset assist09 --epochs 25 --overrides run_name=exp_wd batch_size=64 dropout=0.2 weight_decay=0.0005"
  "exp_reg2|||--dataset assist09 --epochs 25 --overrides run_name=exp_reg2 batch_size=64 dropout=0.5 weight_decay=0.0003"
  "exp_reg|||--dataset assist09 --epochs 25 --overrides run_name=exp_reg batch_size=64 dropout=0.4 weight_decay=0.0001"
  "exp_small|||--dataset assist09 --epochs 25 --overrides run_name=exp_small batch_size=64 d_model=64 d_z=32 dropout=0.3 weight_decay=0.0001"
  "exp_reg3|||--dataset assist09 --epochs 25 --overrides run_name=exp_reg3 batch_size=64 dropout=0.3 weight_decay=0.0001"
  "exp_nodist|||--dataset assist09 --epochs 25 --overrides run_name=exp_nodist batch_size=64 use_distributional=false dropout=0.3 weight_decay=0.0001"
  # --- Phase B: recover the OOM-killed full runs ---
  "eedi_full|||--dataset eedi --ablation full"
  "junyi_full|||--dataset junyi --ablation full"
  # --- Phase C: ablation grid (assist09 + xes3g5m) ---
  "assist09_no_temporal|||--dataset assist09 --ablation no_temporal"
  "assist09_no_samekc|||--dataset assist09 --ablation no_samekc"
  "assist09_no_prereq|||--dataset assist09 --ablation no_prereq"
  "assist09_no_neighbor|||--dataset assist09 --ablation no_neighbor"
  "assist09_no_mono|||--dataset assist09 --ablation no_mono"
  "assist09_no_gs|||--dataset assist09 --ablation no_gs"
  "assist09_no_distributional|||--dataset assist09 --ablation no_distributional"
  "assist09_single_branch|||--dataset assist09 --ablation single_branch"
  "xes3g5m_no_temporal|||--dataset xes3g5m --ablation no_temporal"
  "xes3g5m_no_samekc|||--dataset xes3g5m --ablation no_samekc"
  "xes3g5m_no_prereq|||--dataset xes3g5m --ablation no_prereq"
  "xes3g5m_no_neighbor|||--dataset xes3g5m --ablation no_neighbor"
  "xes3g5m_no_mono|||--dataset xes3g5m --ablation no_mono"
  "xes3g5m_no_gs|||--dataset xes3g5m --ablation no_gs"
  "xes3g5m_no_distributional|||--dataset xes3g5m --ablation no_distributional"
  "xes3g5m_single_branch|||--dataset xes3g5m --ablation single_branch"
)

log "=== queue start: ${#TASKS[@]} tasks, cap=$MAXJOBS, gate=${GATE_GB}GB ==="
for entry in "${TASKS[@]}"; do
  name="${entry%%|||*}"; cargs="${entry##*|||}"
  if [ -f "runs/$name/test_metrics.json" ]; then log "SKIP $name (done)"; continue; fi
  # wait for a free slot
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n; done
  # RAM gate: don't launch into a tight window
  while [ "$(avail)" -lt "$GATE_GB" ]; do log "gate wait (avail=$(avail)GB, running=$(jobs -rp|wc -l))"; sleep 30; done
  log "START $name  (avail=$(avail)GB, running=$(jobs -rp|wc -l))"
  ( $PY scripts/train.py $cargs > "runs/q_${name}.log" 2>&1
    a=$([ -f "runs/$name/test_metrics.json" ] && $PY -c "import json;m=json.load(open('runs/$name/test_metrics.json'));print(f\"auc={m['auc']:.4f} acc={m['acc']:.4f} val={m['best_val_auc']:.4f}\")" || echo FAILED)
    echo "[$(date +%H:%M:%S)] DONE $name :: $a" ) &
  sleep 5   # small stagger so simultaneous model-init spikes don't collide
done
wait
log "=== QUEUE COMPLETE ==="
