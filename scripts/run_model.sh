#!/usr/bin/env bash
# Drive the full internals>output pipeline for ONE model:
#   extract_instruct -> llm_judge -> internals_vs_output
# Usage: bash scripts/run_model.sh <hf_model_id> <tag>
#   e.g. bash scripts/run_model.sh meta-llama/Llama-3.1-8B llama31_8b
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="$1"; TAG="$2"
# Canonical cache naming (docs/CACHE_NAMING.md): always spell out the query count,
# so a 500q cache is never confusable with the `-Instruct` *model* of the same tag.
NQ=500
CACHE="results/cache/${TAG}_q${NQ}_instruct"
JUDGE="results/cache/${TAG}_q${NQ}_judge"
OUT="results/internals_vs_output_${TAG}.json"

# proxy only needed while downloading; harmless once cached
export HF_HUB_DISABLE_XET=1
# IMPORTANT: system disk is only 30G; force HF cache onto the data mount.
# (see CLAUDE.md §七 / memory disk-layout-server)
export HF_HOME="${HF_HOME:-/root/shared-nvme/hf_cache}"
# Proxy is needed while downloading; inherit from environment if set.
# NEVER hardcode credentials here — this repo is public.
#   export https_proxy="http://<user>:<pass>@<proxy-host>:<port>"
if [ -n "${https_proxy:-}" ]; then
    export http_proxy="$https_proxy"
fi

echo "==== [$TAG] extract_instruct ($MODEL, ${NQ}q) ===="
python scripts/extract_instruct.py --model "$MODEL" --n-queries "$NQ" --out "$CACHE"

echo "==== [$TAG] llm_judge (${NQ}q) ===="
python scripts/llm_judge.py --model "$MODEL" --n-queries "$NQ" --out "$JUDGE"

echo "==== [$TAG] internals_vs_output ===="
python scripts/internals_vs_output.py --cache "$CACHE" --judge-cache "$JUDGE" --out "$OUT"

echo "==== [$TAG] DONE -> $OUT ===="
