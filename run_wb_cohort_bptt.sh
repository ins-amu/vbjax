#!/bin/bash
# Card 4ec1331b: freeze cohort G/sigma, BPTT the shape controls (W_q, W_y,
# kv_scale, tau) on the FIT cohort split, then certify + low-rank structure on
# the HELD-OUT split with the certified 2400 s strict rule.
# Usage: sbatch run_wb_cohort_bptt.sh <tag> <loss-mode> <iters> <extra-args>
#SBATCH --partition=run
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --job-name=wbcb
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/marmaduke@lnn.timone.univ-mrs.fr/mhsa_e0/logs/wbcb.%j.out
#SBATCH --error=/home/marmaduke@lnn.timone.univ-mrs.fr/mhsa_e0/logs/wbcb.%j.err
set -e
TAG="${1:-bptt_current}"
MODE="${2:-current}"
ITERS="${3:-60}"
EXTRA="${4:-}"
SIF=/home/marmaduke@lnn.timone.univ-mrs.fr/mhsa_e0/vbjax-jax.sif
TREE="$HOME/mhsa_e0/work/tree-cobptt"
COH="$HOME/mhsa_e0/data/kb_cohort"
SHAPE="$HOME/mhsa_e0/runs_persist/wb_arm_ema_yout/params.pkl"
cd "$TREE"
export PYTHONUNBUFFERED=1 JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "=== COBPTT $TAG mode=$MODE iters=$ITERS start $(date) ==="
echo "=== BPTT shape controls with frozen G=4.0 sigma=0.005 (cohort-best)"
apptainer exec --nv --pwd "$TREE" "$SIF" python -m vbjax.app.wb_cohort_train \
  --cohort "$COH" --params "$SHAPE" \
  --out "runs/wb_cohort_bptt_${TAG}" \
  --g-init 4.0 --sigma-init 0.005 --loss-mode "$MODE" --iters "$ITERS" \
  --k-fit-train 6 --coupling ema $EXTRA
echo "=== certified re-score + low-rank structure on HELD-OUT split"
mkdir -p "runs/wb_cohort_bptt_${TAG}/holdout"
apptainer exec --nv --pwd "$TREE" "$SIF" python -m vbjax.app.wb_cohort_score \
  --cohort "$COH" \
  --params "runs/wb_cohort_bptt_${TAG}/params.pkl" \
  --out "runs/wb_cohort_bptt_${TAG}/holdout" \
  --g 4.0 --sigma 0.005 --k-hold 6 --confirm-s 2400
mkdir -p ~/mhsa_e0/runs_persist
rm -rf "$HOME/mhsa_e0/runs_persist/wb_cohort_bptt_${TAG}"
cp -r "runs/wb_cohort_bptt_${TAG}" ~/mhsa_e0/runs_persist/
echo "WB_COHORT_BPTT_DONE $TAG"
