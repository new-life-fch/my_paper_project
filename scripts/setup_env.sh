#!/bin/bash
# =============================================================================
# Attention-Probe-RAG: Environment Setup Script
# =============================================================================
# Usage: bash scripts/setup_env.sh
# =============================================================================

set -e

echo "=========================================="
echo " Attention-Probe-RAG Environment Setup"
echo "=========================================="

# --- Python venv (optional, uncomment if needed) ---
# python -m venv .venv
# source .venv/bin/activate

# Core dependencies
echo "[1/4] Installing core dependencies..."
pip install -r requirements.txt

# nnsight (latest from pip)
echo "[2/4] Installing nnsight..."
pip install nnsight

# Verify installations
echo "[3/4] Verifying installations..."
python -c "
import torch; print(f'  torch:        {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')

import transformers; print(f'  transformers: {transformers.__version__}')
import nnsight; print(f'  nnsight:      {nnsight.__version__}')
import sklearn; print(f'  sklearn:      {sklearn.__version__}')
import datasets; print(f'  datasets:     {datasets.__version__}')
import sentence_transformers; print(f'  sentence-transformers: {sentence_transformers.__version__}')
"

# HuggingFace login (needed for gated models like LLaMA-3.2)
echo "[4/4] HuggingFace authentication..."
echo "If you haven't logged in yet, run: huggingface-cli login"
echo "(You need to accept the LLaMA-3.2 model license on HuggingFace first)"

echo ""
echo "=========================================="
echo " Setup complete!"
echo "=========================================="
