#!/usr/bin/env bash
# Run patch_internal_v2.py over a model list, optionally waiting for another
# job's PID to exit first. GPU stages MUST be serial: two 7B jobs each take
# ~15-16 GB and OOM'd this box twice (CLAUDE.md §七 trap 4).
#
# Usage:
#   bash scripts/run_patch_v2.sh                       # 4 default variants, now
#   WAIT_PID=1575 bash scripts/run_patch_v2.sh         # wait for that PID, then run
#   MODELS=all8 bash scripts/run_patch_v2.sh           # E-10: all 8 models
#   MODELS="llama32_3b qwen25_7b" bash scripts/run_patch_v2.sh
set -uo pipefail
cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true

NQ="${NQ:-1500}"
LOG="${LOG:-results/logs/patch_v2_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p results/logs
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

if [ -n "${WAIT_PID:-}" ]; then
  log "waiting for PID $WAIT_PID to exit ..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  log "PID $WAIT_PID gone; starting patch v2"
fi

# tag|hf_model_id -- first four are the same variants the v1 patch used, so the
# two can be compared directly; the last four complete E-10's eight models.
ALL_MODELS=(
  "llama32_3b|meta-llama/Llama-3.2-3B"
  "llama31_8b|meta-llama/Llama-3.1-8B"
  "qwen25_7b|Qwen/Qwen2.5-7B"
  "llama32_3b_instruct|meta-llama/Llama-3.2-3B-Instruct"
  "llama31_8b_instruct|meta-llama/Llama-3.1-8B-Instruct"
  "mistral_7b|mistralai/Mistral-7B-v0.3"
  "mistral_7b_instruct|mistralai/Mistral-7B-Instruct-v0.3"
  "qwen25_7b_instruct|Qwen/Qwen2.5-7B-Instruct"
)

MODELS="${MODELS:-first4}"
if [ "$MODELS" = "all8" ]; then
  LIST=("${ALL_MODELS[@]}")
elif [ "$MODELS" = "first4" ]; then
  LIST=("${ALL_MODELS[@]:0:4}")
else
  LIST=()
  for want in $MODELS; do
    hit=""
    for item in "${ALL_MODELS[@]}"; do
      [ "${item%%|*}" = "$want" ] && { LIST+=("$item"); hit=1; }
    done
    [ -z "$hit" ] && log "WARN unknown model tag '$want' (ignored)"
  done
fi
[ ${#LIST[@]} -eq 0 ] && { log "ERROR empty model list"; exit 1; }

for item in "${LIST[@]}"; do
  MID="${item%%|*}"; MODEL="${item##*|}"
  CACHE="results/cache/${MID}_q${NQ}_instruct"
  OUT="results/patch_v2_${MID}_q${NQ}.json"
  if [ ! -f "$CACHE/resid_test.pt" ]; then
    log "SKIP patch v2 $MID (no cache $CACHE)"; continue
  fi
  # only skip artifacts that came from THIS script's current format
  if [ -f "$OUT" ] && python -c "
import json,sys
d=json.load(open('$OUT'))
sys.exit(0 if {'sweep','controls_Lstar','row_alignment_rel_mae'} <= set(d) else 1)"; then
    log "SKIP patch v2 $MID (up-to-date artifact)"; continue
  fi
  log "START patch v2 $MID ($MODEL)"
  python scripts/patch_internal_v2.py --model "$MODEL" --cache "$CACHE" --tag "$MID" \
    --n-queries "$NQ" --out "$OUT" >> "$LOG" 2>&1 \
    && log "DONE  patch v2 $MID" || log "FAIL  patch v2 $MID (see $LOG)"
done

log "patch v2 queue finished"
