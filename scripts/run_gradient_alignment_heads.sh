#!/usr/bin/env bash
# Head-probe gradient alignment across all 8 models.
#
# This is the experiment that closes Limitations (0) of the wide paper: every
# earlier mechanism measurement was taken on the residual stream, while the
# paper's headline effect is the ATTENTION-HEAD probe. Here the gradient is
# taken with respect to the selected heads' o_proj outputs, so the probe being
# tested is the same reader that carries the headline.
#
# Serial by construction: GPU stages must not overlap on a 32 GB card.
#
# Usage:
#   bash scripts/run_gradient_alignment_heads.sh                 # all 8
#   MODELS=first4 bash scripts/run_gradient_alignment_heads.sh   # LLaMA only
#   WAIT_PID=<pid> bash scripts/run_gradient_alignment_heads.sh  # queue behind a job
set -uo pipefail
cd "$(dirname "$0")/.."

NQ="${NQ:-1500}"
ROWS="${ROWS:-256}"
LOG="${LOG:-results/logs/gradient_alignment_heads_queue_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p results/logs
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

if [ -n "${WAIT_PID:-}" ]; then
  log "waiting for PID $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  log "PID $WAIT_PID gone"
fi

# tag|hf_id|batch|dtype
#   batch 4 keeps the graph small enough on 3B with fp32; the 7B/8B models are
#   run in fp16 (fp32 weights alone are ~28 GB, leaving no room for backward).
ALL=(
  "llama32_3b|meta-llama/Llama-3.2-3B|4|float32"
  "llama31_8b|meta-llama/Llama-3.1-8B|1|float16|gc"
  "llama32_3b_instruct|meta-llama/Llama-3.2-3B-Instruct|4|float32"
  "llama31_8b_instruct|meta-llama/Llama-3.1-8B-Instruct|1|float16|gc"
  "mistral_7b|mistralai/Mistral-7B-v0.3|1|float16|gc"
  "mistral_7b_instruct|mistralai/Mistral-7B-Instruct-v0.3|1|float16|gc"
  "qwen25_7b|Qwen/Qwen2.5-7B|1|float16|gc"
  "qwen25_7b_instruct|Qwen/Qwen2.5-7B-Instruct|1|float16|gc"
)

MODE="${MODELS:-all8}"
LIST=()
case "$MODE" in
  all8)   LIST=("${ALL[@]}") ;;
  first4) LIST=("${ALL[@]:0:4}") ;;
  *)      for item in "${ALL[@]}"; do
            [ "${item%%|*}" = "$MODE" ] && LIST+=("$item")
          done ;;
esac
[ ${#LIST[@]} -eq 0 ] && { log "nothing to run for MODELS=$MODE"; exit 1; }

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for item in "${LIST[@]}"; do
  # idempotent: an existing artifact is reused, so re-running the queue after a
  # failure only costs the models that actually failed.
  if [ -f "results/gradient_alignment_heads_${item%%|*}_q${NQ}.json" ] && [ "${FORCE:-0}" != "1" ]; then
    log "SKIP  ${item%%|*} (artifact present)"
    continue
  fi
  tag="${item%%|*}"; rest="${item#*|}"
  model="${rest%%|*}"; rest="${rest#*|}"
  batch="${rest%%|*}"; rest2="${rest#*|}"
  dtype="${rest2%%|*}"; flag="${rest2##*|}"
  extra=(); [ "$flag" = "gc" ] && extra=(--grad-checkpoint)
  out="results/gradient_alignment_heads_${tag}_q${NQ}.json"
  log "START head-gradient $tag ($dtype, batch=$batch)"
  python scripts/gradient_alignment_heads.py \
    --model "$model" --cache "results/cache/${tag}_q${NQ}_instruct" \
    --tag "$tag" --n-rows "$ROWS" --batch "$batch" --dtype "$dtype" \
    "${extra[@]}" \
    --out "$out" >> "$LOG" 2>&1 \
    && log "DONE  $tag" || log "FAIL  $tag (see $LOG)"
done
log "head-gradient queue finished"
