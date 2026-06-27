#!/usr/bin/env bash
# Master queue runner: run the full internals>output pipeline for a LIST of
# models, one at a time (GPU holds one model at a time). One background process
# drives the whole queue; a failing model is logged and skipped, not fatal.
#
# Usage: bash scripts/run_queue.sh
#   (edit the QUEUE array below to change what runs)
#
# Logs: results/logs/<tag>.log per model, results/logs/queue.log master.
cd "$(dirname "$0")/.."

# force HF cache onto data mount (system disk is only 30G)
export HF_HOME="${HF_HOME:-/root/shared-nvme/hf_cache}"
export HF_HUB_DISABLE_XET=1
# proxy for downloads (credentials live in env / CLAUDE.md §七, never hardcoded)
export https_proxy="${https_proxy:-}"
[ -n "$https_proxy" ] && export http_proxy="$https_proxy"

mkdir -p results/logs

# (model_id, tag) pairs. Order = priority.
QUEUE=(
  "meta-llama/Llama-3.1-8B|llama31_8b"
  "meta-llama/Llama-3.2-3B-Instruct|llama32_3b_instruct"
  "mistralai/Mistral-7B-v0.3|mistral_7b"
  "meta-llama/Llama-3.1-8B-Instruct|llama31_8b_instruct"
  "mistralai/Mistral-7B-Instruct-v0.3|mistral_7b_instruct"
)

MASTER=results/logs/queue.log
echo "=== queue start $(date) ===" >> "$MASTER"
for item in "${QUEUE[@]}"; do
  MODEL="${item%%|*}"; TAG="${item##*|}"
  OUT="results/internals_vs_output_${TAG}.json"
  if [ -f "$OUT" ]; then
    echo "[$(date +%H:%M)] SKIP $TAG (already have $OUT)" >> "$MASTER"
    continue
  fi
  echo "[$(date +%H:%M)] START $TAG ($MODEL)" >> "$MASTER"
  if bash scripts/run_model.sh "$MODEL" "$TAG" > "results/logs/${TAG}.log" 2>&1; then
    echo "[$(date +%H:%M)] DONE  $TAG" >> "$MASTER"
    # aggregate after each success so partial progress is always visible
    python scripts/aggregate_models.py >> "$MASTER" 2>&1 || true
    # generate the layer-signal figure for this model
    python scripts/plot_layer_signal.py --cache "results/cache/${TAG}_instruct" \
      --judge-cache "results/cache/${TAG}_judge" --tag "$TAG" >> "$MASTER" 2>&1 || true
  else
    echo "[$(date +%H:%M)] FAIL  $TAG (see results/logs/${TAG}.log)" >> "$MASTER"
  fi
done
echo "=== queue end $(date) ===" >> "$MASTER"
