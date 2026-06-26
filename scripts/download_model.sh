#!/bin/bash
# =============================================================================
# 手动下载 LLaMA-3.2-3B 权重（经典 HTTP，支持断点续传）
# =============================================================================
# 用途：绕过 xet 协议（与代理不兼容会卡住），强制经典 HTTP 下载。
#   - hf_xet 已卸载，huggingface_hub 会自动回退到经典 HTTP
#   - 断点续传：会接着已下载的 .incomplete 文件继续，不用从头来
# 运行：bash scripts/download_model.sh
# =============================================================================
set -e

# --- 代理（下载 HF 必需）---
export https_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
export http_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
export no_proxy="127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.paracloud.com,*.paratera.com,*.blsc.cn"

# --- 强制禁用 xet，走经典 HTTP ---
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_TELEMETRY=1
# 启用底层传输进度（经典 HTTP 下载器会显示每个文件的字节级进度条）
export HF_HUB_ENABLE_HF_TRANSFER=0

MODEL="${1:-meta-llama/Llama-3.2-3B}"

echo "=========================================="
echo " 下载模型: $MODEL"
echo " 协议: 经典 HTTP（hf_xet 已卸载）"
echo " 断点续传: 是"
echo "=========================================="

python - "$MODEL" <<'EOF'
import sys
from huggingface_hub import snapshot_download

model = sys.argv[1]
print(f"开始下载 {model} ...")
path = snapshot_download(
    repo_id=model,
    allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "*.model"],
    # resume_download 默认为 True（新版 hub），会接着 .incomplete 继续
    max_workers=4,
)
print(f"\n下载完成，模型位于:\n{path}")
EOF

echo ""
echo "=========================================="
echo " 完成。可运行实验:"
echo "   bash run_experiment.sh"
echo "=========================================="
