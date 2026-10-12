# 缓存目录命名规范（`results/cache/`）

> 建立于 **2026-10-12**。此前缓存目录名是历史上自然长出来的，同一个 500q 缓存
> 在不同模型上叫法不同，代价是真实的：调试时要靠猜、文档里的"快速开始"命令直接跑不通。
> 本文是唯一权威约定；`README.md`、`CLAUDE.md`、所有 `scripts/*` 的默认值都以此为准。

---

## 一、规范

```
results/cache/{tag}_q{N}_instruct     # 激活缓存：attn_{train,val,test}.pt + resid_*.pt + meta.json
results/cache/{tag}_q{N}_judge        # judge 缓存：judge_{train,val,test}.pt + meta.json
results/cache/{tag}_q{N}_judge_fs{K}  # few-shot judge 变体（K = exemplar 数）
```

- `{tag}` — 与结果文件名 `results/*_{tag}*.json` 完全一致的模型标签。
- `{N}` — query 数，**必须写出来**（`500` / `1500`）。这是本次规范化的核心：
  旧命名里 500q 缓存不带 query 数，于是 `llama32_3b_instruct` 既可能指
  "LLaMA-3.2-3B **base** 的 instruct-prompt 缓存"，也可能被误读成
  "**LLaMA-3.2-3B-Instruct** 这个模型的缓存"。
- 数据集不是 MS MARCO 时，把数据集并进 `{tag}`（如 `llama32_3b_fiqa`），
  不新增字段——保持"目录名 = tag + 查询数 + 产物类型"这一条规则。

**例子**（都是真实存在的缓存）：

| 规范名 | 模型 | 数据 | query |
|---|---|---|---|
| `llama32_3b_q500_instruct` | LLaMA-3.2-3B (base) | MS MARCO | 500 |
| `llama32_3b_q1500_instruct` | LLaMA-3.2-3B (base) | MS MARCO | 1500 |
| `llama32_3b_instruct_q1500_instruct` | LLaMA-3.2-3B-**Instruct** | MS MARCO | 1500 |
| `mistral_7b_q500_judge` | Mistral-7B-v0.3 (base) | MS MARCO | 500 |
| `llama32_3b_fiqa_q1500_instruct` | LLaMA-3.2-3B (base) | BeIR FiQA | 1500 |
| `llama32_3b_q1500_judge_fs4` | LLaMA-3.2-3B (base) | MS MARCO | 1500, 4-shot |

> `llama32_3b_instruct_q1500_instruct` 看着啰嗦，但它是**机械可推导**的：
> 前面那个 `instruct` 属于模型名，后面那个属于 prompt 类型。宁可啰嗦也不要歧义。

---

## 二、`meta.json` 才是权威，目录名只是标签

任何缓存目录里都有一份 `meta.json`，记录 **`model` / `n_queries` / `prompt` / `seed` / `n_layers` / `n_heads`**。
怀疑某个目录是什么的时候，直接读它，不要靠目录名猜：

```bash
python -c "import json,glob; [print(p, json.load(open(p))['model'], json.load(open(p))['n_queries']) \
  for p in sorted(glob.glob('results/cache/*/meta.json'))]"
```

这比任何命名约定都可靠——迁移 / 重命名时请连同 `meta.json` 一起移动。

---

## 三、旧 → 新 对照与迁移

**1500q 的缓存不用动**：旧名 `{tag}_q1500_instruct` 本来就符合规范。
需要迁移的是 500q 那批（还有 FiQA 那一个）：

| 旧目录名 | 规范名 |
|---|---|
| `q500_instruct` / `q500_judge` | `llama32_3b_q500_instruct` / `llama32_3b_q500_judge` |
| `llama31_8b_instruct` / `llama31_8b_judge` | `llama31_8b_q500_instruct` / `llama31_8b_q500_judge` |
| `mistral_7b_instruct` / `mistral_7b_judge` | `mistral_7b_q500_instruct` / `mistral_7b_q500_judge` |
| `llama32_3b_instruct_instruct` / `..._judge` | `llama32_3b_instruct_q500_instruct` / `..._judge` |
| `llama31_8b_instruct_instruct` / `..._judge` | `llama31_8b_instruct_q500_instruct` / `..._judge` |
| `mistral_7b_instruct_instruct` / `..._judge` | `mistral_7b_instruct_q500_instruct` / `..._judge` |
| `llama32_3b_fiqa_instruct` / `..._judge` | `llama32_3b_fiqa_q1500_instruct` / `..._judge` |

迁移（**先在 GPU 主机上核对左边这几个目录确实存在**再执行）：

```bash
cd /root/shared-nvme/paper_A_internals     # 或任一持有 results/cache 的工作目录
mv results/cache/q500_instruct              results/cache/llama32_3b_q500_instruct
mv results/cache/q500_judge                 results/cache/llama32_3b_q500_judge
mv results/cache/llama31_8b_instruct        results/cache/llama31_8b_q500_instruct
mv results/cache/llama31_8b_judge           results/cache/llama31_8b_q500_judge
# ... 其余照上表类推
```

⚠️ **两个必须注意的点**：

1. **不要 `mv` 整个 `results/cache`**。按 `CLAUDE.md` §七第 5 条，`results/cache` 是**软链**，
   指向主仓库的真实目录；对着软链递归操作很容易误伤主缓存。逐个目录 `mv`，或先 `ls -l` 确认。
2. `mistral_7b_instruct` 这个旧名**有歧义**：按旧 `run_model.sh` 的逻辑，
   tag `mistral_7b` 产生 `mistral_7b_instruct`，而 tag `mistral_7b_instruct`
   产生 `mistral_7b_instruct_instruct`。所以 `mistral_7b_instruct` 到底装的是
   base 模型的缓存还是 Instruct 模型的缓存，**只能靠读它的 `meta.json` 判定**（见第二节）。

---

## 四、各脚本读写哪些目录

| 脚本 | 作用 | 默认目录 |
|---|---|---|
| `extract_instruct.py` | 写激活缓存 | `--out results/cache/llama32_3b_q500_instruct` |
| `llm_judge.py` | 写 judge 缓存 | `--out results/cache/llama32_3b_q500_judge` |
| `llm_judge_fewshot.py` | 写 few-shot judge | `--out results/cache/llama32_3b_q1500_judge_fs4` |
| `internals_vs_output.py` | 读两种缓存 | `--cache` / `--judge-cache`（同上 500q 规范名） |
| `plot_layer_signal.py` | 读两种缓存，出主图 | `--cache` / `--judge-cache`（同上） |
| `causal_steer.py` | 读激活缓存 | `--cache results/cache/llama32_3b_q500_instruct` |
| `significance.py` | 读两种缓存 | `--suffix _q500`（默认）或 `_q1500`，拼成 `{tag}{suffix}_{instruct,judge}` |
| `run_model.sh` | 500q 全流程驱动 | 写 `{tag}_q500_{instruct,judge}` |
| `run_scaleup.sh` | 1500q 全流程驱动 | 写 `{tag}_q1500_{instruct,judge}` |
| `run_fiqa.sh` | FiQA 全流程驱动 | 写 `{tag}_q1500_{instruct,judge}` |
| `run_queue.sh` | 队列驱动 | 调 `run_model.sh`（500q）后出图 |

脚本默认值里的 `llama32_3b` 不是硬性约定——只是因为 `extract_instruct.py` /
`llm_judge.py` / `causal_steer.py` 的默认 `--model` 就是 `meta-llama/Llama-3.2-3B`，
这样默认值自洽。换模型时显式传 `--out` 即可。

### `significance.py` 的旧名回退

历史上 500q 的 3B-base 缓存叫 `q500_*`，而其他模型叫 `{tag}_*`。
`significance.py::resolve_caches()` 仍**优先用规范名，找不到才回退旧名**，
并在回退时打印出应该执行的 `mv` 命令：

```
  ! llama32_3b: using LEGACY cache name results/cache/q500_instruct
    canonical would be  results/cache/llama32_3b_q500_instruct
    rename with:  mv results/cache/q500_instruct results/cache/llama32_3b_q500_instruct
```

这个回退只是为了**不让主机上尚未重命名的缓存失效**，新跑的流程一律用规范名。

---

## 五、历史遗留的其它命名（与本次规范无关，仅备注）

- `results/internals_vs_output_{tag}.json` / `results/internals_vs_output_{tag}_q1500.json`
  —— 结果文件名没有统一带 query 数，同理有歧义。**本次未改动**（会牵动论文引用与文档），
  记在 `TODO.md` P3-3。
- `results/significance.json` / `_q1500.json` / `_qwen_q1500.json` / `_fiqa.json`
  —— 四个显著性结果的命名规则不统一，同样记在 `TODO.md` P3-3。
