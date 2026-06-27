# Cheap Adaptation — Probe vs Fine-tuned Reranker

研究问题：**面对新领域、只有少量目标域标注 + 有限算力时，"廉价适配一个冻结 LLM 上的线性探针"是否比"微调一个专用 reranker"更划算？**

口号：**Adapt a linear probe in minutes, not fine-tune a transformer for hours.**

已观测现象（适配增益大且两域一致）：

| 设置 | SciFact | FiQA |
|---|---|---|
| 探针 零样本 | 0.795 | 0.627 |
| reranker (bge-v2-m3) 零样本 | 0.921 | 0.789 |
| **探针 适配** | **0.964** | **0.860** |

⚠️ 当前对比**不公平**（探针看了目标域标注、reranker 没看）。本工作区的核心任务就是补上"LoRA 微调 reranker"作为公平对手，论证**性价比（精度/算力/标注代价）**优势——不是单纯精度。

注意：零样本下探针打不过 reranker（已证伪），本主线只在"给定目标域少量标注 + 公平算力预算"下立论。详见 `CLAUDE.md`。

## 快速开始

```bash
pip install -r requirements.txt   # 或 bash scripts/setup_env.sh
# 复现 FiQA 三方对比：
python scripts/zeroshot_transfer.py --src results/cache/q500_instruct --tgt results/cache/fiqa_ood --topk 20
python scripts/compare_ood.py --cache results/cache/fiqa_ood --topk 20
python scripts/strong_reranker.py --which cache --cache results/cache/fiqa_ood
```

完整上下文、方案史、脚本说明、进度与下一步见 **`CLAUDE.md`**。
本目录是 git worktree（分支 `paper-B`）；姊妹工作区 `paper-A` 做 internals>output 主线。
