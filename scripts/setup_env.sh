#!/bin/bash
# =============================================================================
# Attention-Probe-RAG: Environment Setup Script
# =============================================================================
# Usage: bash scripts/setup_env.sh
#
# 2026-10-13 更新（新 AutoDL 主机首次跑通）：
#   - 缓存全部指向数据盘（系统盘只有 30G，几个 7B 模型就能撑爆）
#   - 卸载 hf_xet（与代理不兼容会卡死，CLAUDE.md §七 坑 3）
#   - 装上 nnsight（激活提取必需，原来漏在 requirements.txt 之外）
#   - 修 torch 属性名 total_mem -> total_memory（TODO.md P1-14，已实测确认）
#   - torch 本身不装：本机自带 2.8.0+cu128 且 CUDA 可用
# =============================================================================

set -e

echo "=========================================="
echo " Attention-Probe-RAG Environment Setup"
echo "=========================================="

# --- Python venv (optional, uncomment if needed) ---
# python -m venv .venv
# source .venv/bin/activate

# --- 缓存全部指向数据盘 ------------------------------------------------------
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/pip_cache}"
export TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/torch_cache}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$TORCH_HOME"
echo "  HF_HOME = $HF_HOME"

# --- hf_xet 必须缺席（与代理不兼容，下载大文件卡死）--------------------------
pip uninstall -y hf_xet >/dev/null 2>&1 || true
export HF_HUB_DISABLE_XET=1

# --- Core dependencies（requirements.txt 已钉版本，含 nnsight）---------------
echo "[1/4] Installing core dependencies..."
pip install -r requirements.txt

# --- torch 由服务器提供 ------------------------------------------------------
echo "[2/4] torch 由服务器自带，不在此安装（requirements.txt 顶部有显式重装命令）"

# --- 验证 --------------------------------------------------------------------
echo "[3/4] Verifying installations..."
python -c "
import torch; print(f'  torch:        {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:          {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:         {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

import transformers; print(f'  transformers: {transformers.__version__}')
import nnsight; print(f'  nnsight:      {nnsight.__version__}')
import sklearn; print(f'  sklearn:      {sklearn.__version__}')
import datasets; print(f'  datasets:     {datasets.__version__}')
import sentence_transformers; print(f'  sentence-transformers: {sentence_transformers.__version__}')
"

# --- HuggingFace 认证（gated 模型，如 LLaMA 系列）----------------------------
echo "[4/4] HuggingFace authentication..."
echo "  gated 模型（4 个 meta-llama 仓库，gated=manual）需要 token，且账号须已同意 Meta 许可。"
echo "  用 Python API（本环境 huggingface-cli 不可靠）："
echo "    python -c \"from huggingface_hub import login; login()\""
echo "  或下载时注入环境变量：export HF_TOKEN=hf_xxx"
echo "  ⚠️ token 绝不写进仓库（见 CLAUDE.md §二 提交前必查密钥）。"

echo ""
echo "=========================================="
echo " Setup complete!"
echo "=========================================="
