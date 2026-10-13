#!/usr/bin/env bash
# =============================================================================
# 全量实验流水线（8 模型 × 1500 queries）—— 2026-10-13 新主机
# =============================================================================
# 顺序: extract_instruct -> llm_judge -> internals_vs_output
#       -> significance -> logit_lens -> plot_layer_signal
# GPU 步骤（extract/judge/logit_lens）串行，一台卡同时只跑一个模型。
# 每一步都有幂等跳过：产物在且是当前代码产出的就跳过，因此可以随时重跑续跑。
#
# 为什么必须重跑：原 GPU 主机已回收，results/cache/ 与全部 *_layer_signal.json、
# 4 个 internals_vs_output_*.json、2 个 causal_steer_8b*.json 都已永久丢失
# （见结果/REPORT.md 与 TODO.md P2）。新数字 = 复现版，允许第 3 位小数漂移。
#
# 用法:
#   bash scripts/run_experiments.sh                 # 全部 8 个模型
#   bash scripts/run_experiments.sh llama32_3b      # 只跑指定 tag
#   MODELS="llama32_3b qwen25_7b" bash scripts/run_experiments.sh
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

NQ="${NQ:-1500}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true

mkdir -p results/logs results/cache
MASTER="results/logs/experiments_$(date +%Y%m%d_%H%M%S).log"

# tag|hf_model_id —— 顺序 = 优先级（LLaMA 旗舰在前，论文 Table 3/Fig 1 依赖它们）
ALL=(
  "llama32_3b|meta-llama/Llama-3.2-3B"
  "llama32_3b_instruct|meta-llama/Llama-3.2-3B-Instruct"
  "llama31_8b|meta-llama/Llama-3.1-8B"
  "llama31_8b_instruct|meta-llama/Llama-3.1-8B-Instruct"
  "mistral_7b|mistralai/Mistral-7B-v0.3"
  "mistral_7b_instruct|mistralai/Mistral-7B-Instruct-v0.3"
  "qwen25_7b|Qwen/Qwen2.5-7B"
  "qwen25_7b_instruct|Qwen/Qwen2.5-7B-Instruct"
)

want=("$@")
if [ ${#want[@]} -eq 0 ] && [ -n "${MODELS:-}" ]; then
  # shellcheck disable=SC2206
  want=(${MODELS})
fi

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }

for item in "${ALL[@]}"; do
  MID="${item%%|*}"; MODEL="${item##*|}"
  if [ ${#want[@]} -gt 0 ]; then
    printf '%s\n' "${want[@]}" | grep -qx "$MID" || continue
  fi
  IC="results/cache/${MID}_q${NQ}_instruct"
  JC="results/cache/${MID}_q${NQ}_judge"
  IO="results/internals_vs_output_${MID}_q${NQ}.json"
  LL="results/logit_lens_${MID}_q${NQ}.json"

  log "================ $MID ($MODEL) ================"

  # --- 1. activations (GPU) ---
  if [ -f "$IC/meta.json" ] && [ -f "$IC/resid_test.pt" ]; then
    log "SKIP extract (cache exists: $IC)"
  else
    log "START extract -> $IC"
    python scripts/extract_instruct.py --model "$MODEL" --n-queries "$NQ" \
      --max-passages 5 --seed 42 --out "$IC" >> "$MASTER" 2>&1 \
      && log "DONE  extract" || { log "FAIL  extract ($MID)"; continue; }
  fi

  # --- 2. LLM-judge baseline (GPU) ---
  if [ -f "$JC/meta.json" ] && [ -f "$JC/judge_test.pt" ]; then
    log "SKIP judge (cache exists: $JC)"
  else
    log "START judge -> $JC"
    python scripts/llm_judge.py --model "$MODEL" --n-queries "$NQ" \
      --max-passages 5 --seed 42 --out "$JC" >> "$MASTER" 2>&1 \
      && log "DONE  judge" || { log "FAIL  judge ($MID)"; continue; }
  fi

  # --- 3. four-reader report (CPU) ---
  if [ -f "$IO" ] && python - "$IO" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if (d.get("resid_best_layer", {}).get("selection") == "val"
               and "resid_per_layer_val_auc" in d) else 1)
PY
  then
    log "SKIP internals_vs_output (current-code artifact exists)"
  else
    log "START internals_vs_output -> $IO"
    python scripts/internals_vs_output.py --cache "$IC" --judge-cache "$JC" \
      --out "$IO" >> "$MASTER" 2>&1 \
      && log "DONE  internals_vs_output" || log "FAIL  internals_vs_output ($MID)"
  fi

  # --- 4. permuted/paired bootstrap significance (CPU, per model) ---
  log "START significance ($MID)"
  python scripts/significance.py --suffix "_q${NQ}" --only "$MID" \
    --out "results/significance_${MID}_q${NQ}.json" >> "$MASTER" 2>&1 \
    && log "DONE  significance" || log "FAIL  significance ($MID)"

  # --- 5. logit-lens control (GPU; val-selected peak, test-reported) ---
  if [ -f "$LL" ]; then
    log "SKIP logit_lens (exists: $LL)"
  else
    log "START logit_lens -> $LL"
    python scripts/logit_lens.py --model "$MODEL" --cache "$IC" --judge-cache "$JC" \
      --tag "$MID" --out "$LL" >> "$MASTER" 2>&1 \
      && log "DONE  logit_lens" || log "FAIL  logit_lens ($MID)"
  fi

  # --- 6. depth figure + per-head heatmap (CPU) ---
  log "START plot_layer_signal ($MID)"
  python scripts/plot_layer_signal.py --cache "$IC" --judge-cache "$JC" \
    --tag "$MID" >> "$MASTER" 2>&1 \
    && log "DONE  plot_layer_signal" || log "FAIL  plot_layer_signal ($MID)"

  log "================ $MID complete ================"
done

log "ALL EXPERIMENTS DONE (master log: $MASTER)"
log "free space: $(df -h /root/autodl-tmp | tail -1 | awk '{print $4}')"
