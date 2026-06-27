# Attention-Probe / Internals > Output

研究问题：**LLM 内部激活是否比模型自己的输出更"知道"一段检索文档与 query 是否相关？**

核心发现（LLaMA-3.2-3B, MS MARCO）：同一模型、同一 prompt、同一 answer-token——
- 让模型直接输出判断（LLM-judge）：相关性 AUC **0.535**（≈随机）
- 训练一个读**输出表示**（最终层 hidden）的探针：**0.593**
- 训练一个读**内部**（注意力头激活）的探针：**0.685**

→ 优势主要来自"读内部"而非"被训练"，且相关性信号沿深度中部（L12, 0.632）最强、向输出层衰减。
**Internals know more than the model can say.**

这是一篇可解释性 / 探针方向的工作，**不是** RAG reranker 优化（那条路已被证伪，详见 `CLAUDE.md` 第四节）。

## 快速开始

```bash
pip install -r requirements.txt   # 或 bash scripts/setup_env.sh
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
python scripts/internals_vs_output.py --cache results/cache/q500_instruct
```

完整上下文、方案史、脚本说明、进度与下一步见 **`CLAUDE.md`**。
本目录是 git worktree（分支 `paper-A`）；姊妹工作区 `paper-B` 做 cheap-adaptation 主线。
