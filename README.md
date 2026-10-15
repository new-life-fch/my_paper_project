# Internals Know More Than the Model Can Say

> **一句话**：同一个模型、同一个 prompt、同一个 answer token 上，
> 「这段文档是否与 query 相关」这个信号，从**注意力头激活**里读出来，
> 明显比从**输出层表示**（最终层 hidden）里读出来更准——**跨 8 个模型无一例外**。

这是一篇**可解释性 / 探针（probing）**方向的论文。它**不是**「做一个更好的 RAG reranker」，
也**不主张**探针在重排精度上打赢 cross-encoder（那条路在本项目早期已被自己的实验证伪）。

## 现在的状态

| 项 | 值 |
|---|---|
| 分支 | `paper-A`（窄版论文）／`paper-B`（宽版：机制叙事） |
| 权威状态文档 | **[`STATUS.md`](STATUS.md)** —— 实测数据由脚本从 `results/*.json` 生成 |
| 旧文档 | 全部在 **[`archive/`](archive/)**（`git mv` 移入，历史保留，不再维护） |
| 论文 | `paper/main.tex` + `paper/main.pdf` |
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

## 怎么复现

```bash
# 1) 结构自检 + 证据纪律检查（纯 CPU，秒级）
python scripts/selfcheck.py
python scripts/build_status.py --check      # 同一量在不同 JSON 里不一致会报警

# 2) 论文取数（纯 CPU）
python scripts/collect_paper_numbers.py --n 1500    # → results/PAPER_NUMBERS.json
python scripts/aggregate_significance.py --suffix _q1500   # BH 校正 → significance_summary

# 3) 复现核心结果（纯 CPU，需要 results/cache/ 的 .pt）
python scripts/internals_vs_output.py --cache results/cache/llama32_3b_q1500_instruct
python scripts/significance.py --cache results/cache/llama32_3b_q1500_instruct \
    --judge-cache results/cache/llama32_3b_q1500_judge --tag llama32_3b --n-boot 5000

# 4) 出图（主图三连 + 因果/机制图）
python scripts/plot_layer_signal.py --cache results/cache/llama32_3b_q1500_instruct \
    --judge-cache results/cache/llama32_3b_q1500_judge --tag llama32_3b
python scripts/plot_causal.py --json results/causal_steer_llama32_3b_q1500.json --tag llama32_3b

# 5) 编译论文（需先开代理：tectonic 缓存不自足，见 STATUS §7.2 第 17 条）
source /etc/network_turbo
cd paper && ../paper/tools/tectonic main.tex && ls -la main.pdf   # 用 mtime 确认真的写出
```

## 需要 GPU 的部分

激活提取、judge、因果 steering、patch、梯度对齐都要加载模型（`extract_instruct.py`、
`llm_judge.py`、`causal_steer.py`、`patch_internal_v2.py`、`gradient_alignment.py` 等）。
GPU 阶段**必须串行**——32 GB 显存下两个 7B 进程会 OOM（已实测）。
队列脚本支持 `WAIT_PID=<pid>` 串接。

## 阅读顺序

1. **[`STATUS.md`](STATUS.md)** —— 唯一权威状态文档（实测数据自动生成，含判据与更正记录）
2. **[`archive/README.md`](archive/README.md)** —— 旧文档清单与它们现在的作用
3. `paper/main.tex` —— 论文正文
4. `results/figures/README.md` —— 图的来源与重生成方式
5. `docs/` —— 缓存命名规范、模型实测 config、文献调研（仍然有效）
