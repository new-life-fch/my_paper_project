#!/usr/bin/env bash
# Gradient alignment over all 8 models (the sharpest misalignment control).
# Serial by construction: GPU stages must not overlap on a 32 GB card.
#
# Usage:
#   bash scripts/run_gradient_alignment.sh                       # 4 remaining models
#   MODELS=all8 bash scripts/run_gradient_alignment.sh           # force all 8
#   WAIT_PID=<pid> bash scripts/run_gradient_alignment.sh        # queue behind a job
set -uo pipefail
cd "$(dirname "$0")/.."

NQ="${NQ:-1500}"
LOG="${LOG:-results/logs/gradient_alignment_queue_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p results/logs
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

if [ -n "${WAIT_PID:-}" ]; then
  log "waiting for PID $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  log "PID $WAIT_PID gone"
fi

# tag|hf_id|dtype  (7B models need float16: fp32 weights alone are ~28 GB)
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

MODE="${MODELS:-remaining}"
LIST=()
if [ "$MODE" = "all8" ]; then
  LIST=("${ALL[@]}")
else
  for item in "${ALL[@]}"; do
    tag="${item%%|*}"
    [ -f "results/gradient_alignment_${tag}_q${NQ}.json" ] || LIST+=("$item")
  done
fi
[ ${#LIST[@]} -eq 0 ] && { log "nothing to do (all artifacts present)"; exit 0; }

for item in "${LIST[@]}"; do
  tag="${item%%|*}"
  rest="${item#*|}"; model="${rest%%|*}"; dtype="${rest##*|}"
  out="results/gradient_alignment_${tag}_q${NQ}.json"
  log "START gradient alignment $tag ($dtype)"
  python scripts/gradient_alignment.py --model "$model" \
    --cache "results/cache/${tag}_q${NQ}_instruct" --tag "$tag" \
    --n-rows 256 --dtype "$dtype" --out "$out" >> "$LOG" 2>&1 \
    && log "DONE  $tag" || log "FAIL  $tag (see $LOG)"
done
log "gradient alignment queue finished"
