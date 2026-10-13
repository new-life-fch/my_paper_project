#!/bin/bash
# =============================================================================
# 一键准备：环境变量 + 下载通道 + 全量模型下载（全部落在数据盘）
# =============================================================================
# 新 AutoDL 主机（2026-10-13）专用。系统盘只有 30G，所以：
#   - HF 缓存 -> /root/autodl-tmp/hf_cache（数据盘）
#   - torch/triton 编译缓存 -> /root/autodl-tmp/{torch,triton}_cache
#   - pip wheel 缓存 -> /root/autodl-tmp/pip_cache
#
# ── 下载通道怎么选（本机实测，2026-10-13）────────────────────────────────────
#   A) huggingface.co + AutoDL 代理（/etc/network_turbo）
#      → 小文件/API 很快（0.8s），但**大 shard 持续吞吐会停滞**：实测第 2 个模型
#        180MB 后再无字节增长（90s 净增 −24MB），进程 6 线程全部 poll 等 IO。
#        首个 3B 模型是靠"超时→断点续传"磨了 6.9 分钟才完成的。
#   B) hf-mirror.com + **不走代理**（本脚本默认）
#      → 实测稳定 4.25–5.06 MB/s（100MB 用 19.7s），100MB 分片无超时。
#      加代理反而卡死（200MB 120s 内零进展）→ 必须 unset 代理变量。
#   C) modelscope：AutoDL 的 no_proxy 白名单里有，但仓库 id/布局与 HF 不同，
#      需另写下载器，暂不用。
#   结论：走 B。若将来要回到 A，注释掉 HF_ENDPOINT 那几行并重新 source 代理。
#
# 用法：
#   bash scripts/setup_autodl.sh                  # 下载全部 8 个模型（已缓存的会跳过）
#   bash scripts/setup_autodl.sh --only qwen25_7b # 只下指定 tag（透传给 python）
# =============================================================================
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# --- 缓存全部指向数据盘 -------------------------------------------------------
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME=/root/autodl-tmp/torch_cache
export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache
export TRITON_CACHE_DIR=/root/autodl-tmp/triton_cache
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$PIP_CACHE_DIR" "$TRITON_CACHE_DIR"

# --- 下载通道 B：hf-mirror.com，直连（详情见文件头）--------------------------
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
# AutoDL 的 no_proxy 白名单仍留着，防止子进程自己捡到代理
export no_proxy="localhost,127.0.0.1,modelscope.com,aliyuncs.com,tencentyun.com,wisemodel.cn,hf-mirror.com"

# --- hf_xet 与"经典 HTTP 断点续传"的下载路径冲突，必须缺席 --------------------
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_TELEMETRY=1
# 镜像带宽会抖动，默认 10s 读超时对 4-5GB 的大 shard 太紧
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"

if [ -n "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN: 从环境读取（不落盘到仓库）"
fi

mkdir -p results/logs
LOG="results/logs/download_$(date +%Y%m%d_%H%M%S).log"

echo "REPO_DIR    = $REPO_DIR"
echo "HF_HOME     = $HF_HOME"
echo "HF_ENDPOINT = $HF_ENDPOINT  (直连，代理已 unset)"
echo "log         = $LOG"
echo

python scripts/download_all_models.py "$@" 2>&1 | tee "$LOG"
exit "${PIPESTATUS[0]}"
