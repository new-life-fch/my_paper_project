#!/usr/bin/env bash
# =============================================================================
# 第二轮实验：主流水线跑完后的补充实验（都在 1500q 缓存上做）
# =============================================================================
# 每个 job 都幂等（产物在就跳过），可反复续跑。
#
#   1. capacity  : R-4 capacity-matched 对照（CPU；每模型 ~2-5 分钟）
#   2. causal    : 因果 steering，3B/8B/Qwen × base/instruct（GPU，串行）
#   3. prompt    : prompt 改写鲁棒性，3B/Qwen base（GPU，串行）
#
# 用法:
#   bash scripts/run_second_round.sh                 # 全部
#   bash scripts/run_second_round.sh capacity        # 只跑某一类
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

NQ="${NQ:-1500}"
KIND="${1:-all}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true

mkdir -p results/logs
LOG="results/logs/second_round_$(date +%Y%m%d_%H%M%S).log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# 因果与 prompt 实验的模型清单：tag|hf_id|n_random
# 12 个随机方向 × 9 个 alpha × 2250 条 test ≈ 每个模型 20-40 分钟，
# 8 个模型串行太久，故选三个家族各一个 base（+ 3B instruct 作对齐对照）。
CAUSAL=(
  "llama32_3b|meta-llama/Llama-3.2-3B|12"
  "llama31_8b|meta-llama/Llama-3.1-8B|12"
  "qwen25_7b|Qwen/Qwen2.5-7B|12"
  "llama32_3b_instruct|meta-llama/Llama-3.2-3B-Instruct|5"
)
PROMPT_MODELS=(
  "llama32_3b|meta-llama/Llama-3.2-3B"
  "qwen25_7b|Qwen/Qwen2.5-7B"
)

# ---------------------------------------------------------------- capacity ---
if [ "$KIND" = "all" ] || [ "$KIND" = "capacity" ]; then
  for item in "llama32_3b|meta-llama/Llama-3.2-3B" "llama31_8b|meta-llama/Llama-3.1-8B" \
              "mistral_7b|mistralai/Mistral-7B-v0.3" "qwen25_7b|Qwen/Qwen2.5-7B" \
              "llama32_3b_instruct|meta-llama/Llama-3.2-3B-Instruct" \
              "llama31_8b_instruct|meta-llama/Llama-3.1-8B-Instruct" \
              "mistral_7b_instruct|mistralai/Mistral-7B-Instruct-v0.3" \
              "qwen25_7b_instruct|Qwen/Qwen2.5-7B-Instruct"; do
    MID="${item%%|*}"
    IC="results/cache/${MID}_q${NQ}_instruct"
    JC="results/cache/${MID}_q${NQ}_judge"
    OUT="results/capacity_control_${MID}_q${NQ}.json"
    [ -f "$IC/resid_train.pt" ] || { log "SKIP capacity $MID (no cache)"; continue; }
    if [ -f "$OUT" ]; then log "SKIP capacity $MID (exists)"; continue; fi
    log "START capacity $MID"
    python scripts/capacity_control.py --cache "$IC" --judge-cache "$JC" \
      --tag "$MID" --out "$OUT" >> "$LOG" 2>&1 \
      && log "DONE  capacity $MID" || log "FAIL  capacity $MID"
  done
fi

# ------------------------------------------------------------------ causal ---
if [ "$KIND" = "all" ] || [ "$KIND" = "causal" ]; then
  for item in "${CAUSAL[@]}"; do
    MID="$(echo "$item" | cut -d'|' -f1)"
    MODEL="$(echo "$item" | cut -d'|' -f2)"
    NR="$(echo "$item" | cut -d'|' -f3)"
    IC="results/cache/${MID}_q${NQ}_instruct"
    OUT="results/causal_steer_${MID}_q${NQ}.json"
    [ -f "$IC/resid_test.pt" ] || { log "SKIP causal $MID (no cache)"; continue; }
    if [ -f "$OUT" ]; then log "SKIP causal $MID (exists)"; continue; fi
    log "START causal $MID (n_random=$NR)"
    python scripts/causal_steer.py --model "$MODEL" --cache "$IC" --tag "$MID" \
      --n-queries "$NQ" --n-random "$NR" --out "$OUT" >> "$LOG" 2>&1 \
      && log "DONE  causal $MID" || log "FAIL  causal $MID"
    python scripts/plot_causal.py --in "$OUT" >> "$LOG" 2>&1 || true
  done
fi

# ------------------------------------------------------------------ prompt ---
if [ "$KIND" = "all" ] || [ "$KIND" = "prompt" ]; then
  for item in "${PROMPT_MODELS[@]}"; do
    MID="${item%%|*}"; MODEL="${item##*|}"
    for TPL in terse verbose; do
      PIC="results/cache/${MID}_${TPL}_q${NQ}_instruct"
      PJC="results/cache/${MID}_${TPL}_q${NQ}_judge"
      [ -f "$PIC/resid_test.pt" ] && [ -f "$PJC/judge_test.pt" ] && { log "SKIP prompt $MID/$TPL (exists)"; continue; }
      log "START prompt $MID/$TPL"
      python scripts/extract_instruct.py --model "$MODEL" --n-queries "$NQ" \
        --max-passages 5 --seed 42 --prompt-template "$TPL" --out "$PIC" >> "$LOG" 2>&1 \
        && python scripts/llm_judge.py --model "$MODEL" --n-queries "$NQ" \
             --max-passages 5 --seed 42 --prompt-template "$TPL" --out "$PJC" >> "$LOG" 2>&1 \
        && python scripts/internals_vs_output.py --cache "$PIC" --judge-cache "$PJC" \
             --out "results/internals_vs_output_${MID}_${TPL}_q${NQ}.json" >> "$LOG" 2>&1 \
        && log "DONE  prompt $MID/$TPL" || log "FAIL  prompt $MID/$TPL"
    done
  done
fi

log "SECOND ROUND DONE (kind=$KIND) -- log: $LOG"
