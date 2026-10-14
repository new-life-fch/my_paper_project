#!/usr/bin/env bash
# =============================================================================
# 第二个 in-domain 数据集（BeIR/FiQA）：在 FiQA 内部训练并测试探针
# （**不是跨域迁移** —— 那条死路属于工作区 B）。
# 目的：证明 "内部 > 输出" 这个 gap 不是 MS MARCO 特有的。
#
# 2026-10-14 重做（用户决策，见 TODO.md「已决定」）：
#   原实验用**随机负例**，结果 final 层探针到 0.976（近天花板）、internal_edge
#   只有 +0.0165 —— 一个太简单、输出表示已经到顶的数据集，对主论点是弱证据。
#   现改为 **BM25 难负例**（与 MS MARCO 自带的 hard negative 同构：都是检索到
#   但未被标注为相关的段落），使两个数据集的难度可比。
#
# Usage:
#   bash scripts/run_fiqa.sh <hf_model_id> <tag> [negatives] [n_queries]
#     bash scripts/run_fiqa.sh meta-llama/Llama-3.2-3B llama32_3b_fiqa bm25 1500
#   缓存命名（docs/CACHE_NAMING.md）: results/cache/<tag>_q<N>_{instruct,judge}
#   BM25 排名缓存在 results/cache/bm25/ 下，由 src/data.py 自动复用。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="$1"; TAG="$2"
NEG="${3:-bm25}"
NQ="${4:-1500}"
DS="beir:fiqa"

case "$NEG" in
  bm25)   IC="results/cache/${TAG}_bm25_q${NQ}_instruct"; JC="results/cache/${TAG}_bm25_q${NQ}_judge";;
  random) IC="results/cache/${TAG}_q${NQ}_instruct";      JC="results/cache/${TAG}_q${NQ}_judge";;
  *) echo "unknown negatives: $NEG (use bm25|random)" >&2; exit 2;;
esac
OUT="results/internals_vs_output_${TAG}_${NEG}_q${NQ}.json"

export HF_HUB_DISABLE_XET=1
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset http_proxy https_proxy 2>/dev/null || true

echo "==== [$TAG] FiQA negatives=$NEG nq=$NQ model=$MODEL ===="
[ -f "$IC/meta.json" ] && echo "  SKIP extract (exists)" || \
  python scripts/extract_instruct.py --model "$MODEL" --dataset "$DS" \
    --n-queries "$NQ" --max-passages 5 --seed 42 --out "$IC"
[ -f "$JC/meta.json" ] && echo "  SKIP judge (exists)" || \
  python scripts/llm_judge.py --model "$MODEL" --dataset "$DS" \
    --n-queries "$NQ" --max-passages 5 --seed 42 --out "$JC"

python scripts/internals_vs_output.py --cache "$IC" --judge-cache "$JC" --out "$OUT"
echo "==== [$TAG] DONE -> $OUT ===="
