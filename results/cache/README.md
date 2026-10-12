# results/cache/ — 目录命名对照（旧 → 新）

> 这个目录里的**激活缓存不进 git**（每个模型 1–3 GB），所以你在 clone 里通常看不到它们。
> 它们宿在 GPU 主机的 `/root/shared-nvme/paper_A_internals/results/cache/`（且常是软链）。
> 本文件存在的唯一目的：**让你在照着旧文档找文件时不会以为文件丢了。**

## 现行规范（2026-10-12 起）

```
{tag}_q{N}_instruct     激活缓存：attn_{train,val,test}.pt + resid_*.pt + meta.json
{tag}_q{N}_judge        judge 缓存：judge_{train,val,test}.pt + meta.json
{tag}_q{N}_judge_fs{K}  few-shot judge 变体
```

完整说明见 [`docs/CACHE_NAMING.md`](../../docs/CACHE_NAMING.md)。

## 旧名 → 新名

| 旧目录名（2026-10-12 之前） | 规范名 |
|---|---|
| `q500_instruct` / `q500_judge` | `llama32_3b_q500_instruct` / `llama32_3b_q500_judge` |
| `llama32_3b_instruct` / `llama32_3b_judge` | `llama32_3b_q500_instruct` / `llama32_3b_q500_judge` |
| `llama31_8b_instruct` / `llama31_8b_judge` | `llama31_8b_q500_instruct` / `llama31_8b_q500_judge` |
| `mistral_7b_instruct` / `mistral_7b_judge` | `mistral_7b_q500_instruct` / `mistral_7b_q500_judge` |
| `llama32_3b_instruct_instruct` / `..._judge` | `llama32_3b_instruct_q500_instruct` / `..._judge` |
| `llama31_8b_instruct_instruct` / `..._judge` | `llama31_8b_instruct_q500_instruct` / `..._judge` |
| `mistral_7b_instruct_instruct` / `..._judge` | `mistral_7b_instruct_q500_instruct` / `..._judge` |
| `llama32_3b_fiqa_instruct` / `..._judge` | `llama32_3b_fiqa_q1500_instruct` / `..._judge` |

`{tag}_q1500_instruct` / `{tag}_q1500_judge` 这两类**本来就是合规的**，没有改名。

## 三件事你要知道

1. **找不到文件不代表丢了。** 所有读缓存的脚本都会自动回退到旧名并打印应执行的 `mv`
   （别名表在 [`src/cache_paths.py`](../../src/cache_paths.py) 的 `LEGACY_ALIASES`，
   是唯一事实来源，`scripts/migrate_cache_names.sh` 也从它读）。
2. **要真正改名**：`bash scripts/migrate_cache_names.sh`（默认只列出，`--apply` 才动手）。
3. **别靠目录名判断内容** —— 读 `meta.json` 的 `model` / `n_queries` / `prompt` 字段，
   那才是权威。`mistral_7b_instruct` 这种旧名从目录名上就无法判断装的是 base 还是 Instruct 模型。
