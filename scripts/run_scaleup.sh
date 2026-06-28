#!/usr/bin/env bash
# Scale-up extraction to tighten per-model CIs (root cause: test had only ~75 queries).
# Runs extract->judge->internals for 1500-query caches on the flagship models.
set -u
export HF_HOME="${HF_HOME:-/root/shared-nvme/hf_cache}"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
cd /root/shared-nvme/paper_A_internals
mkdir -p results/logs

run() {  # model_id  tag
  local mid="$1" tag="$2"
  local ic="results/cache/${tag}_q1500_instruct" jc="results/cache/${tag}_q1500_judge"
  echo "[$(date +%H:%M:%S)] === $tag extract (1500q) ==="
  python scripts/extract_instruct.py --model "$mid" --n-queries 1500 --max-passages 5 --out "$ic" \
    && python scripts/llm_judge.py --model "$mid" --n-queries 1500 --max-passages 5 --seed 42 --out "$jc" \
    && python scripts/internals_vs_output.py --cache "$ic" --judge-cache "$jc" \
         --out "results/internals_vs_output_${tag}_q1500.json"
  echo "[$(date +%H:%M:%S)] === $tag done ==="
}

run meta-llama/Llama-3.2-3B                 llama32_3b
run meta-llama/Llama-3.2-3B-Instruct        llama32_3b_instruct
run meta-llama/Llama-3.1-8B                 llama31_8b
run meta-llama/Llama-3.1-8B-Instruct        llama31_8b_instruct
echo "[$(date +%H:%M:%S)] ALL SCALE-UP DONE"
