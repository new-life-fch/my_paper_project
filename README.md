# Internals Know More Than the Model Can Say

> **一句话**：同一个模型、同一个 prompt、同一个 answer token 上，
> 「这段文档是否与 query 相关」这个信号，从**注意力头激活**里读出来，
> 明显比从**输出层表示**（最终层 hidden）里读出来更准——**跨 8 个模型无一例外**。
> 而模型自己"嘴上"（输出 logits）说得最差。

这是一篇**可解释性 / 探针（probing）**方向的论文。它**不是**「做一个更好的 RAG reranker」，
也**不主张**探针在重排精度上打赢 cross-encoder（那条路在本项目早期已被自己的实验证伪）。

> ⚠️ **你在读的是 `paper-wide` 分支（宽版论文）**。同一实验库支撑两版论文，
> 差别在**主张跨度**，不在数据。详见下方"两版论文"一节。

## 两版论文（同一个实验库，两种主张跨度）

| | **窄版** `paper-A` | **宽版** `paper-wide`（← 你在这里） |
|---|---|---|
| 主张 | 读注意力头 > 读输出层表示，8/8 模型，唯一过 BH 校正的对比 | 窄版全部 **+** 机制：为什么输出端读不出来 |
| 机制/因果 | **不主张**（正文明确写"本文不含机制断言"） | **主张 misalignment**：信息在、下游也确实用 $L^\ast$，但判决局部敏感的方向不是相关性方向 |
| 关键证据 | `attn−final` 8/8（q<0.001）；capacity 对照；训练贡献 0/8 | 窄版证据 + patch v2（8 模型）+ 梯度对齐（8 模型，分位 0.06–0.81 全 <0.95） |
| 图 | `fig1`–`fig6` | 窄版全套 **+** `fig7`（patch）、`fig8`（梯度对齐） |
| 代价 | 故事小，但每条都硬 | 机制证据**只覆盖残差流**，而头条效应在**注意力头**——这个缺口已写进 Limitations (0) |

**为什么分两版**：机制那半篇的证据强度够不上头条那半篇。窄版把"哪里能读到相关性"
做成一条跨 8 模型无例外的定量结论；宽版加"为什么输出端读不出"的机制解释，
但那条解释是用残差流探针做的，而最强效应是注意力头——审稿人会问两个 reader 的关系，
目前答不上来，所以宽版把它写进 Limitations 而不是掩盖。

两个分支用 **git worktree** 共存（共享一个 `.git`），所以磁盘上有两个目录：

```bash
git worktree list
# /root/autodl-tmp/my_paper_project  [paper-A]    ← 窄版（本目录）
# /root/autodl-tmp/pw_wide           [paper-wide] ← 宽版
```

**注意两边 `paper/main.tex` 同名但内容不同**，改论文前先确认目录。
详见 `STATUS.md` §0.1。

## 现在的状态

| 项 | 值 |
|---|---|
| 本分支 | `paper-wide`（宽版论文在 `paper/main.tex`） |
| 窄版位置 | 分支 `paper-A`；本分支的归档副本在 `archive/paper_narrow/` |
| 权威状态文档 | **[`STATUS.md`](STATUS.md)**（§0.1 有两版论文的对照表） |
| 旧文档 | 全部在 **[`archive/`](archive/)**（`git mv` 移入，历史保留，不再维护） |
| 取数单一来源 | `results/PAPER_NUMBERS.json`（每个值带 provenance） |

## 核心结果（8 模型 × 1500 query，val 选层，配对 bootstrap + BH 校正）

| 模型 | judge | final | best-resid | **attn（ours）** | attn − final |
|---|---|---|---|---|---|
| LLaMA-3.2-3B | 0.545 | 0.536 | 0.613 (L9) | **0.684** | **+0.148** |
| LLaMA-3.1-8B | 0.574 | 0.562 | 0.621 (L8) | **0.677** | **+0.114** |
| Mistral-7B-v0.3 | 0.579 | 0.583 | 0.606 (L16) | **0.684** | **+0.101** |
| LLaMA-3.2-3B-Instruct | 0.614 | 0.612 | 0.638 (L18) | **0.712** | **+0.100** |
| LLaMA-3.1-8B-Instruct | 0.630 | 0.596 | 0.609 (L12) | **0.725** | **+0.129** |
| Mistral-7B-Instruct | 0.646 | 0.596 | 0.634 (L13) | **0.713** | **+0.117** |
| Qwen2.5-7B | 0.652 | 0.592 | 0.655 (L22) | **0.680** | **+0.088** |
| Qwen2.5-7B-Instruct | 0.660 | 0.610 | 0.630 (L23) | **0.669** | **+0.059** |

**`attn − final` 是唯一在 8/8 模型上全部通过 BH 校正的效应**（逐模型 q<0.001）。
四条 edge 的存活数：`attn−final` **8/8**、`internal_edge` 3/8、`resid−judge` 1/8、`train_edge` 0/8。

### 机制侧（宽版特有的证据）

**patch v2**（跨样本替换残差激活，无尺度混淆）：判决是否跟随移植内容**是模型依赖的**——
Qwen 两模型最强（Δgap +0.176 / +0.228），两个 LLaMA base 几乎为零（+0.006 / +0.003）。
**范数匹配与不匹配结果相同**（8/8 模型差 ≤0.0002）→ **"移植后崩塌"不是尺度伪影**。

**梯度对齐**（$\cos(\nabla_{h_{L^\ast}}(\text{yes}-\text{no}), \hat w_{\text{probe}})$
vs 16 个随机方向）：**8/8 模型的探针方向都不是离群者**（分位 0.06–0.81，判据是 ≥0.95）。
这条**不编辑任何激活**，所以不受"分布外"质疑；而且**在判决确实被 $L^\ast$ 驱动的 Qwen 上也成立**。

## 怎么复现

```bash
# 1) 结构自检 + 证据纪律检查（纯 CPU，秒级）
python scripts/selfcheck.py
python scripts/build_status.py --check      # 同一量在不同 JSON 里不一致会报警

# 2) 论文取数（纯 CPU）
python scripts/collect_paper_numbers.py --n 1500    # → results/PAPER_NUMBERS.json
python scripts/aggregate_significance.py --suffix _q1500   # BH 校正

# 3) 出图（窄版全套 + 宽版机制图，全部纯 CPU，从 JSON 直出）
python scripts/plot_paper_figures.py          # fig1–fig6 + 每模型 fig3_depth_<model>
python scripts/plot_mechanism_figures.py      # fig7（patch）、fig8（梯度对齐）

# 4) 编译论文（需先开代理：tectonic 缓存不自足，见 STATUS §7.2 第 17 条）
source /etc/network_turbo
cd paper && ../paper/tools/tectonic main.tex && ls -la main.pdf   # 用 mtime 确认真的写出
```

## 需要 GPU 的部分

激活提取、judge、因果 steering、patch、梯度对齐都要加载模型（`extract_instruct.py`、
`llm_judge.py`、`causal_steer.py`、`patch_internal_v2.py`、`gradient_alignment.py` 等）。
GPU 阶段**必须串行**——32 GB 显存下两个 7B 进程会 OOM（已实测）。
队列脚本支持 `WAIT_PID=<pid>` 串接。

## 阅读顺序

1. **[`STATUS.md`](STATUS.md)** —— 权威状态文档（§0.1 两版论文对照；§2 实测数据自动生成）
2. **[`archive/README.md`](archive/README.md)** —— 旧文档清单与它们现在的作用
3. `paper/main.tex` —— 本分支（宽版）论文正文
4. `results/figures/README.md` —— 图的来源与重生成方式
5. `docs/` —— 缓存命名规范、模型实测 config、文献调研（仍然有效）
