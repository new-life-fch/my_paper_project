#!/usr/bin/env bash
# =============================================================================
# 小样本量受控复算：500q × 同一份新代码
# =============================================================================
# 为什么需要它（TODO.md「复核新发现 第 0 条」，结论级存疑项）：
#   论文 Limitations 写着"attention-head reader 在 Mistral 上弱"，依据是原主机的
#   **500q** artifact（base attn−final = +0.0021 p=0.474；instruct +0.0509 p=0.0526）。
#   但**原主机上 Mistral 根本没有 1500q 数据**（significance_q1500.json 只有 4 个 LLaMA，
#   significance_qwen_q1500.json 只有 2 个 Qwen），所以那个"弱"只建立在 test 仅 ~75 query
#   的一档上。新协议在 1500q 上测出 Mistral 两个模型的 attn−final 都是 +0.10~+0.12、p<2e-4。
#
#   于是必须问：差异来自**样本量**，还是来自**旧代码/旧口径**？
#   本脚本用同一份新代码在 500q 上重算，把两者并排比较：
#     - 500q 下也显著  -> 旧结论是旧代码/旧口径造成的假阴性，Limitations 必须删掉该条；
#     - 500q 下确实弱  -> 那是欠采样效应，同样不该写成架构性结论。
#   两种结局都不支持原文措辞，跑它只是为了让改法有据。
#
# ⚠️ 本机的 500q 缓存是**原主机遗留**的（mistral_7b_q500_instruct 等），
#    必须当成"不存在"重新提取 —— 用旧缓存跑新代码等于把两个变量（协议 + 缓存来源）
#    混在一起，正好毁掉这次受控比较。脚本用 meta.json 的日期判断，旧的直接重跑。
#
# 用法:
#   bash scripts/run_q500_control.sh                 # 默认 4 个 7B 模型
#   MODELS="mistral_7b llama31_8b" bash scripts/run_q500_control.sh
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

NQ=500
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true

mkdir -p results/logs
LOG="results/logs/q500_control_$(date +%Y%m%d_%H%M%S).log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

ALL=(
  "mistral_7b|mistralai/Mistral-7B-v0.3"
  "mistral_7b_instruct|mistralai/Mistral-7B-Instruct-v0.3"
  "qwen25_7b|Qwen/Qwen2.5-7B"
  "qwen25_7b_instruct|Qwen/Qwen2.5-7B-Instruct"
)
want=()
[ -n "${MODELS:-}" ] && read -r -a want <<< "$MODELS"

# 缓存是否由**当前代码**产出：看 meta.json 的 mtime 是否在 2026-10-13 之后
fresh() {
  local f="$1/meta.json"
  [ -f "$f" ] || return 1
  python - "$f" <<'PY'
import os, sys, time
# 今天之前生成的缓存一律不算数（本机今天才开始用当前代码跑）
p = sys.argv[1]
sys.exit(0 if os.path.getmtime(p) > time.mktime(time.strptime("2026-10-13", "%Y-%m-%d")) else 1)
PY
}

for item in "${ALL[@]}"; do
  MID="${item%%|*}"; MODEL="${item##*|}"
  if [ ${#want[@]} -gt 0 ]; then
    printf '%s\n' "${want[@]}" | grep -qx "$MID" || continue
  fi
  IC="results/cache/${MID}_q${NQ}_instruct"
  JC="results/cache/${MID}_q${NQ}_judge"
  IO="results/internals_vs_output_${MID}_q${NQ}.json"
  SG="results/significance_${MID}_q${NQ}.json"

  log "================ $MID ($MODEL) @ ${NQ}q ================"
  if fresh "$IC"; then log "SKIP extract (fresh cache)"; else
    [ -d "$IC" ] && log "STALE extract cache (原主机遗留) -> 重跑"
    log "START extract"
    python scripts/extract_instruct.py --model "$MODEL" --n-queries "$NQ" \
      --max-passages 5 --seed 42 --out "$IC" >> "$LOG" 2>&1 \
      && log "DONE extract" || { log "FAIL extract ($MID)"; continue; }
  fi
  if fresh "$JC"; then log "SKIP judge (fresh cache)"; else
    [ -d "$JC" ] && log "STALE judge cache -> 重跑"
    log "START judge"
    python scripts/llm_judge.py --model "$MODEL" --n-queries "$NQ" \
      --max-passages 5 --seed 42 --out "$JC" >> "$LOG" 2>&1 \
      && log "DONE judge" || { log "FAIL judge ($MID)"; continue; }
  fi
  log "START internals_vs_output"
  python scripts/internals_vs_output.py --cache "$IC" --judge-cache "$JC" \
    --out "$IO" >> "$LOG" 2>&1 && log "DONE internals_vs_output" || log "FAIL internals ($MID)"
  log "START significance"
  python scripts/significance.py --suffix "_q${NQ}" --only "$MID" \
    --out "$SG" >> "$LOG" 2>&1 && log "DONE significance" || log "FAIL significance ($MID)"
  log "================ $MID complete ================"
done

log "Q500 CONTROL DONE -- log: $LOG"
