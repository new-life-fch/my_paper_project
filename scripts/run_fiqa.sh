#!/usr/bin/env bash
# Second IN-DOMAIN relevance dataset (BeIR/FiQA): train AND test the probe
# within FiQA (NO cross-domain transfer — that dead plan is workspace B).
# Goal: show the internals>output gap is not specific to MS MARCO.
# Usage: bash scripts/run_fiqa.sh <hf_model_id> <tag>
#   e.g. bash scripts/run_fiqa.sh meta-llama/Llama-3.2-3B llama32_3b_fiqa
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="$1"; TAG="$2"
DS="beir:fiqa"
NQ=1500
CACHE="results/cache/${TAG}_instruct"
JUDGE="results/cache/${TAG}_judge"
OUT="results/internals_vs_output_${TAG}.json"

export HF_HUB_DISABLE_XET=1
export HF_HOME="${HF_HOME:-/root/shared-nvme/hf_cache}"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "==== [$TAG] extract_instruct ($MODEL, $DS) ===="
python scripts/extract_instruct.py --model "$MODEL" --dataset "$DS" --n-queries "$NQ" --out "$CACHE"

echo "==== [$TAG] llm_judge ($DS) ===="
python scripts/llm_judge.py --model "$MODEL" --dataset "$DS" --n-queries "$NQ" --out "$JUDGE"

echo "==== [$TAG] internals_vs_output ===="
python scripts/internals_vs_output.py --cache "$CACHE" --judge-cache "$JUDGE" --out "$OUT"

echo "==== [$TAG] DONE -> $OUT ===="
