# Attention-Probe / Internals > Output

研究问题：**LLM 内部激活是否比模型自己的输出更"知道"一段检索文档与 query 是否相关？**

核心发现（LLaMA-3.2-3B, MS MARCO）：同一模型、同一 prompt、同一 answer-token——
- 让模型直接输出判断（LLM-judge）：相关性 AUC **0.535**（≈随机）
- 训练一个读**输出表示**（最终层 hidden）的探针：**0.593**
- 训练一个读**内部**（注意力头激活）的探针：**0.685**

→ 优势主要来自"读内部"而非"被训练"，且相关性信号沿深度中部（L12, 0.632）最强、向输出层衰减。
**Internals know more than the model can say.**

这是一篇可解释性 / 探针方向的工作，**不是** RAG reranker 优化（那条路已被证伪，详见 `CLAUDE.md` 第四节）。

> ⚠️ 上面那几个数字（0.535 / 0.593 / 0.632）是 **P1 修复前**的口径（在 test 上选层）。
> 现行的 **val 选层**数字与 8 模型结果见 `CLAUDE.md` §3.2 的加注与 `TODO.md` P1-1。

## 先读什么（接手必看）

**→ `docs/READING_GUIDE.md`** —— 阅读顺序、每份 MD 的时效性与权威性、
项目进展记在哪、待办在哪、以及在云容器里从零开始怎么做。**先读它**，
否则会被互相打架的口径绕晕（这个项目有 6+ 份 MD，且不是同一时间写的）。

最短路径：**本文**（定位）→ **`CLAUDE.md`** §〇/§三（论点）+ §六（进展流水账）
→ **`TODO.md`**（待办与已知问题，含投稿前必须处理的 P0）。

## 快速开始

```bash
pip install -r requirements.txt   # 或 bash scripts/setup_env.sh
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
# 缓存目录命名规范见 docs/CACHE_NAMING.md
python scripts/internals_vs_output.py --cache results/cache/llama32_3b_q500_instruct
```

完整上下文、方案史、脚本说明、进度与下一步见 **`CLAUDE.md`**；
缓存命名规范与旧目录对照见 **`docs/CACHE_NAMING.md`**；
接手时的阅读顺序与各文档时效性见 **`docs/READING_GUIDE.md`**。

> 🖥️ **2026-10-13 起跑在 AutoDL 新实例上**（原 GPU 主机已回收）。当前**只有这一个 clone**：
> `/root/autodl-tmp/my_paper_project`，分支 `paper-A`。本文此前写的"本目录是 git worktree、
> 姊妹工作区 `paper-B` 做 cheap-adaptation 主线"是**旧主机上的情况**，新主机上不存在那个
> 主仓库，也没有工作区 B。
