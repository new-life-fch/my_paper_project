#!/usr/bin/env bash
# Causal intervention on the head-probe direction, across models.
#
# Step 2 of route B: the head probe wins descriptively (8/8), and its gradient
# alignment came out mixed. This asks the causal question directly -- is the
# verdict something the probe's direction actually controls? -- using the
# E-11 corrected protocol (logit-shift calibration, label-grouped alignment gap,
# random-direction null, registered criterion).
#
# Serial by construction: GPU stages must not overlap on a 32 GB card.
#
# Usage:
#   bash scripts/run_causal_steer_heads.sh
#   MODELS=first2 bash scripts/run_causal_steer_heads.sh
#   WAIT_PID=<pid> bash scripts/run_causal_steer_heads.sh
set -uo pipefail
cd "$(dirname "$0")/.."

NQ="${NQ:-1500}"
NROWS="${NROWS:-0}"        # 0 = all test rows
NRAND="${NRAND:-12}"
SHIFTS="${SHIFTS:--2,2}"
BATCH="${BATCH:-16}"
LOG="${LOG:-results/logs/causal_steer_heads_queue_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p results/logs
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

if [ -n "${WAIT_PID:-}" ]; then
  log "waiting for PID $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  log "PID $WAIT_PID gone"
fi

ALL=(
  "llama32_3b|meta-llama/Llama-3.2-3B|float32"
  "llama31_8b|meta-llama/Llama-3.1-8B|float16"
  "llama32_3b_instruct|meta-llama/Llama-3.2-3B-Instruct|float32"
  "llama31_8b_instruct|meta-llama/Llama-3.1-8B-Instruct|float16"
  "mistral_7b|mistralai/Mistral-7B-v0.3|float16"
  "mistral_7b_instruct|mistralai/Mistral-7B-Instruct-v0.3|float16"
  "qwen25_7b|Qwen/Qwen2.5-7B|float16"
  "qwen25_7b_instruct|Qwen/Qwen2.5-7B-Instruct|float16"
)

MODE="${MODELS:-all8}"
LIST=()
case "$MODE" in
  all8)   LIST=("${ALL[@]}") ;;
  first2) LIST=("${ALL[@]:0:2}") ;;
  *)      for item in "${ALL[@]}"; do
            [ "${item%%|*}" = "$MODE" ] && LIST+=("$item")
          done ;;
esac
[ ${#LIST[@]} -eq 0 ] && { log "nothing to run for MODELS=$MODE"; exit 1; }

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for item in "${LIST[@]}"; do
  tag="${item%%|*}"; model="${item#*|}"; dtype="${model##*|}"; model="${model%%|*}"
  out="results/causal_steer_heads_${tag}_q${NQ}.json"
  if [ -f "$out" ] && [ "${FORCE:-0}" != "1" ]; then
    log "SKIP  $tag (artifact present)"
    continue
  fi
  log "START causal-heads $tag ($dtype)"
  # -u: without it python block-buffers stdout when redirected, and the log
  # shows nothing for the whole run (made a healthy job look hung).
  python -u scripts/causal_steer_heads.py --model "$model" \
    --cache "results/cache/${tag}_q${NQ}_instruct" --tag "$tag" \
    --n-rows "$NROWS" --n-random "$NRAND" --shift-targets="$SHIFTS" \
    --dtype "$dtype" --batch-size "$BATCH" --out "$out" >> "$LOG" 2>&1 \
    && log "DONE  $tag" || log "FAIL  $tag (see $LOG)"
done
log "causal-heads queue finished"
