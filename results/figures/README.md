# results/figures/ — 图的来源与重生成方式

> **2026-10-14 重写**。本文档此前的内容已失效（它描述的是旧主机上"图只存在于 PDF 里"
> 的状态；换到 AutoDL 新实例后所有图都从缓存重新生成过了）。旧版本留在 git 历史里。

## 现在这里有什么

**8 个模型 × 2 类图**，全部由本机当前代码从 `results/cache/` 重新生成：

| 文件 | 生成脚本 | 论文里用哪张 |
|---|---|---|
| `<tag>_layer_signal.png` + `.json` | `scripts/plot_layer_signal.py` | **Fig. 1**（深度定位三联图）用 `llama32_3b_layer_signal.png` |
| `causal_steer_<tag>.png` | `scripts/plot_causal.py` | **Fig. 2**（因果 steering）用 `causal_steer_llama32_3b.png` |

`<tag>` ∈ `llama32_3b`、`llama32_3b_instruct`、`llama31_8b`、`llama31_8b_instruct`、
`mistral_7b`、`mistral_7b_instruct`、`qwen25_7b`、`qwen25_7b_instruct`（共 8 个）。

`.json` 是同一张图的数字转储（逐层 AUC、top-k 头所在层等），**论文里任何引用图内数值的
句子都应指向它，而不是指向 PNG**。

## 怎么重新生成

```bash
# 深度定位图（需要 results/cache/<tag>_q1500_instruct/ 的 .pt 缓存，纯 CPU）
python scripts/plot_layer_signal.py --cache results/cache/llama32_3b_q1500_instruct \
    --judge-cache results/cache/llama32_3b_q1500_judge --tag llama32_3b

# 因果图（只读 JSON，连缓存都不需要）
python scripts/plot_causal.py --json results/causal_steer_llama32_3b_q1500.json \
    --tag llama32_3b
```

⚠️ `plot_layer_signal.py` 会**重新计算**逐层 AUC 与 per-head AUC，所以它产出的 `.json`
是这些数字的唯一可复现来源。图与 `.json` 必须成对提交，否则以后无法核对图里的数。

## 历史：这些图曾经丢过一次（2026-10-12）

当时 `.gitignore` 里同时有 `results/` 和 `*.png`，一张图只有被 `git add -f` 过才能活下来。
两张论文图从未被强制提交，于是从所有副本里消失，而 `main.tex` 还引用着它们 → 编译失败。
它们是从 `paper/main.pdf` 里恢复的（LaTeX 把每个 `\includegraphics` 作为图像 XObject
嵌在 PDF 中，原始像素还在）。脚本：`scripts/extract_figures_from_pdf.py`。

**已修**：`.gitignore` 收窄为 `results/cache/` + `results/**/*.pt` + `results/logs/`，
图与结果 JSON 现在默认入库，同样的路走不通了。

> 从 PDF 恢复出来的那两张是**旧主机、旧代码口径**（test-argmax 选层）的产物。
> 现在目录里的同名文件**已经是在新主机上用 val 选层重生成的版本**，不要混淆。
