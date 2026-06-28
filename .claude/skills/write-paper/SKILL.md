---
name: write-paper
description: Draft and compile the "Internals > Output" interpretability paper (LaTeX via tectonic). Use when writing, revising, or building the paper for workspace A (paper_A_internals). Encapsulates interpretability-paper structure, honesty rules, and this project's verified results.
---

# write-paper — 可解释性论文写作 skill（工作区 A）

## 何时用
撰写 / 修订 / 编译 "Internals know more than the model can say" 这篇可解释性论文时。

## 论文一句话
LLM 内部激活编码了 (query, passage) 相关性信号，但该信号在传向输出 logits 的过程中被衰减，导致模型"嘴上"表达不出来。

## 铁律（继承 CLAUDE.md 诚实优先原则）
1. **只写已验证的数字**。所有 AUC / p 值 / CI 必须来自 `results/significance.json`、`results/significance_q1500.json`、`results/causal_steer_*.json`、`results/internals_vs_output_*.json`、`results/REPORT.md`。不得编造或外推未跑的结果。
2. **不声称已证伪的论断**（见 CLAUDE.md 第四节）：不碰"探针打赢 reranker"、"跨域天生强"、"数据效率高"。本文是**可解释性发现**，非 RAG reranker 工程。
3. **效应强弱分级陈述**：attn−final(LLaMA 4/4 极显著 p<0.0001)是最强主张；best_resid−final 强形式仅在 **base 模型**扩样本后单模型显著(3B p=0.0008, 8B p=0.004)，instruct 上不显著且缩小（这是机制证据，不是失败）；跨模型符号检验 p=0.016 作辅证。
4. **因果证据以消融臂为主**：steering 负向注入 P(yes) 0.79→0.26 单调 + internal 唯一保持 AUC（vs final 摧毁 AUC、random 平）。增益臂受 judge 正例偏置饱和，需注明边界。
5. Mistral 的 attn-head 弱是**真实跨架构差异**，作 discussion，不藏。

## 标准结构（NeurIPS/ICLR/ACL 可解释性短文 8 页骨架）
1. **Abstract** — 发现 + 方法(线性探针 4 读取器 + steering) + 关键数字(attn−final, base internal_edge, 因果消融) + 一句机制(对齐改善读出但不闭合内部 gap)。
2. **Introduction** — 现象引入(模型嘴上≈随机 judge 0.51-0.62 但内部可解码)；贡献 bullet：①存在性 ②归因(读内部 vs 训练，train_edge 拆解) ③机制(深度衰减 + base/instruct 对照) ④因果(steering)。
3. **Related Work** — probing classifiers、logit lens / tuned lens、activation steering、RAG relevance、calibration / hallucination（模型知道但不说）。引用诚实，标注未确认项。
4. **Method** — ①4 读取器定义(judge / final-layer probe / best-resid probe / attn-head probe)，prompt 与 answer-token 约定(CLAUDE.md 3.3) ②两 gap 拆解 train_edge/internal_edge ③steering 干预协议 ④评测：query 70/15/15 防泄露、ROC-AUC、query 聚类 bootstrap 5000、val 选层。
5. **Experiments** —
   - 主表：6 模型 × 4 读取器(REPORT §2，标注点估计) + §9/§10 CI 表（判读以 CI 为准）。
   - 深度定位图：`results/figures/<tag>_layer_signal.png`（中部成峰、向输出衰减）。
   - 扩样本显著性：500q→1500q internal_edge p 值改善表（§10）。
   - base↔instruct 机制对照（§4 三配对：judge↑ / train_edge↓转负 / internal_edge）。
   - 因果 steering（§11 表 + 消融臂单调 + AUC 对照）。
6. **Discussion** — 机制解读(对齐改善输出读出、不闭合 gap)；跨架构差异(Mistral)；与校准/幻觉文献关系；局限(单数据集、judge 正例偏置、Mistral 未扩样本)。
7. **Conclusion** + **Limitations**（诚实列：相关性为主、因果限 3B、in-domain 单数据集）。

## 数据来源速查
| 要写的内容 | 取数文件 |
|---|---|
| 6 模型主表(点估计) | `results/multimodel_summary.json` / REPORT §2 |
| CI + p 值(500q) | `results/significance.json` / REPORT §9 |
| CI + p 值(1500q 加固) | `results/significance_q1500.json` / REPORT §10 |
| 因果 steering | `results/causal_steer_llama32_3b_q1500.json` / REPORT §11 |
| 深度/head 定位图 | `results/figures/*_layer_signal.{png,json}` |
| 实验设置/prompt | CLAUDE.md §3.3 |

## 编译
```bash
cd /root/shared-nvme/paper_A_internals/paper
tectonic main.tex          # 首次会下载宏包(需代理), 之后缓存
```
代理：`export https_proxy=$https_proxy`（见 CLAUDE.md §七）。产物 `main.pdf`。

## 产物位置
论文源码放 `paper/`（`main.tex`, `refs.bib`, `figures/` 软链到 `../results/figures/`）。`paper/` 不进 gitignore，可提交 .tex/.bib（PDF 与 figures 按需 -f）。
