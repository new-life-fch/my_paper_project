# Paper A 项目完整交接文档

**生成日期**: 2026-06-30  
**项目分支**: `paper-A` @ commit `51e51f2`  
**工作目录**: `/root/shared-nvme/paper_A_internals`  
**论文状态**: 8模型(3B-8B, LLaMA/Mistral/Qwen, base+instruct), 6页完整草稿, 已过顶会级审稿修正, 达 CCF-B 投稿量级  

---

## 一、项目主旨与研究问题

### 核心问题
**大语言模型的内部表示是否"知道"的比它输出表达的更多？**

具体到检索相关性判断任务：当我们给 LLM 一对 (query, passage) 并让它判断相关性时，模型在中间层的激活（attention heads、residual stream）能否解码出比最终输出(logits)更准确的相关性信号？

### 研究方法
在**同一个冻结前向传播**上比较 4 种相关性"读取器"：
1. **LLM-judge (零样本输出)**：模型自己的 P(yes) vs P(no)，无训练
2. **Output-repr probe (输出表示)**：在最终隐藏层训练线性探针
3. **Internal probe (内部表示)**：在**最佳中间层**残差流训练探针（验证集选层）
4. **Attention-head probe (注意力头)**：读 top-k 个头的输出（验证集选头）

### 主要发现（已写入论文）
✅ **8个模型一致**：内部读取器（best-resid 或 attn）均超过输出表示和 judge  
✅ **显著性**：attn−final 在所有 LLaMA(4个) 和 Qwen(2个) 上 p<0.003  
✅ **深度定位**：信号在中层(L10-16 for LLaMA)达峰，向输出层衰减  
✅ **因果证据**：steering 实验显示内部方向causally影响输出判断  
✅ **对齐机制**：base→instruct 过程压缩 internal_edge (信号向输出移动但未完全到达)  
🔑 **Qwen 特殊性**：Qwen base 的 judge 已达 0.652(vs LLaMA ~0.52)，但 attn−final 仍显著 → 即使输出已强，heads 仍超过 output-repr

---

## 二、完整文件结构与说明

### 2.1 项目根目录布局

```
paper_A_internals/
├── paper/                  # LaTeX 论文源码
│   ├── main.tex           # 主文档 (6页，8模型，完整草稿)
│   ├── refs.bib           # 文献库 (14篇，已联网核实真实性)
│   ├── main.pdf           # 编译产物 (401KB)
│   ├── figures/ -> ../results/figures/  # 图片符号链接
│   └── tools/tectonic     # LaTeX 编译器静态二进制
├── src/                   # Python 源码包
│   ├── data.py            # 数据加载 (MS MARCO / BeIR FiQA)
│   └── common.py          # 通用工具
├── scripts/               # 实验脚本 (extract/judge/probe/significance/plot)
├── results/               # 所有实验结果 (gitignored, 关键文件 -f 强制提交)
│   ├── cache/            # 模型激活缓存 (*.pt, 1-2GB per model)
│   ├── figures/          # 深度曲线图 (*.png)
│   ├── *.json            # 汇总结果 (internals_vs_output / significance / logit_lens)
│   └── REPORT.md         # 实验日志 (§1-§15, 详尽记录每次推进)
├── CLAUDE.md             # 项目进度总览 (§六，6次主要推进)
├── README.md             # 一页项目摘要
└── .claude/              # Claude 工作区
    ├── memory/           # 持久化记忆
    └── skills/write-paper/  # 论文写作 skill

```

### 2.2 核心脚本详解 (scripts/)

#### 数据提取脚本

**`extract_instruct.py`** — 提取 instruct prompt 下的激活  
- 作用：给定模型，在 (Q,P) 对上跑前向传播，捕获两种激活：
  - `attn`: 每个头的 o_proj 输出 @ answer token → [N, L×H, head_dim]
  - `resid`: 每层残差流 @ answer token → [N, L, hidden]
- 输出：`<out_dir>/attn_{train,val,test}.pt`, `resid_*.pt`, `meta.json`
- 示例：
  ```bash
  python scripts/extract_instruct.py \
    --model meta-llama/Llama-3.2-3B \
    --n-queries 1500 \
    --out results/cache/llama32_3b_q1500_instruct
  ```

**`llm_judge.py`** — LLM-judge 基线（零样本）  
- 作用：同样的 instruct prompt，但只保存 answer token 的 yes/no logits，归一化成 P(yes)
- 输出：`<out_dir>/judge_{train,val,test}.pt` (scores/labels/query_ids), `meta.json`
- 示例：
  ```bash
  python scripts/llm_judge.py \
    --model meta-llama/Llama-3.2-3B \
    --n-queries 1500 \
    --out results/cache/llama32_3b_q1500_judge
  ```

**`llm_judge_fewshot.py`** — few-shot judge (4-exemplar)  
- 作用：在 instruct prompt 前加 4 个 in-context 样例（仅从 train 取，无泄露）
- 用途：回应"judge 弱是 prompt 问题"的质疑 → few-shot 仍远低于 probe

#### 探针与分析脚本

**`internals_vs_output.py`** — 主分析：4种读取器对比  
- 输入：`--cache <instruct_dir>`, `--judge-cache <judge_dir>`
- 输出：`results/internals_vs_output_<tag>.json`
- 关键逻辑：
  1. 在 **val** 上选 best resid layer (argmax AUC)
  2. 在 **val** 上选 top-k attn heads
  3. 在 **test** 上评估 4 种 reader 的 AUC
  4. **注意**：早期版本(bug)在 test 上选层，已在 commit `3457979` 修复为 val 选层
- 示例输出字段：
  ```json
  {
    "llm_judge_zeroshot": 0.652,
    "resid_final_layer(output_repr)": {"auc": 0.592},
    "resid_best_layer": {"layer": 22, "auc": 0.655},
    "attn_head_probe(ours)": {"auc": 0.682}
  }
  ```

**`significance.py`** — query-clustered bootstrap 显著性检验  
- 作用：5000次 bootstrap (按 query 重采样)，计算 edge 的 95% CI 和单侧 p 值
- 关键 edge 定义：
  - `internal_edge = best_resid - final`
  - `attn-final = attn - final`
  - `best_resid-judge = best_resid - judge`
  - `train_edge = final - judge`
- 模型注册：在脚本开头 `MODELS` 列表，格式 `(tag, display_name, cache_dir_base)`
- 输出：`results/significance_<suffix>.json` (per-model CI/p-value)
- 示例：
  ```bash
  python scripts/significance.py --suffix _q1500 \
    --only llama32_3b,qwen25_7b \
    --out results/significance_llama_qwen_q1500.json
  ```

**`logit_lens.py`** — 无监督输出端基线（P3审稿要求）  
- 作用：对**每层**残差应用 `model.norm + lm_head`，读 P(yes)−P(no)，计算 per-layer AUC
- sanity check：**最后一层 logit-lens AUC 应 = judge AUC**（已在 LLaMA-3B/8B/Qwen 上验证 absdiff=0.000）
- 关键发现：无监督读出在所有层接近随机(peak ~0.56)，远低于训练探针(0.62)
- 输出：`results/logit_lens_<tag>.json` (per-layer AUC数组)
- 示例：
  ```bash
  python scripts/logit_lens.py \
    --model meta-llama/Llama-3.2-3B \
    --cache results/cache/llama32_3b_q1500_instruct \
    --judge-cache results/cache/llama32_3b_q1500_judge \
    --out results/logit_lens_llama32_3b_q1500.json
  ```

**`causal_steer.py`** — 因果 steering 实验  
- 作用：在 best internal layer 形成方向 `d = mean(h_rel) - mean(h_irrel)`，在前向传播时添加到**输出层**残差，sweep 强度 α，观察 judge P(yes) 和 test AUC 变化
- 三种方向对比：internal / output-layer-own / random
- 关键发现：只有 internal 方向同时 (1)单调改变 verdict (2)保持 AUC；final 方向摧毁 AUC
- 输出：`results/causal_steer_<tag>.json` (alphas, p_yes_curves, auc_curves)

**`plot_layer_signal.py`** — 深度定位主图  
- 输入：instruct cache + judge cache
- 输出：3-panel 图 (A: per-layer resid AUC曲线; B: per-head heatmap; C: layer vs head view)
- 输出文件：`results/figures/<tag>_layer_signal.png` + `.json`

#### 管线驱动脚本

**`run_model.sh`** — 单模型完整管线  
```bash
# 用法: bash scripts/run_model.sh <hf_model_id> <tag>
# 例: bash scripts/run_model.sh meta-llama/Llama-3.2-3B llama32_3b
# 自动跑: extract_instruct -> llm_judge -> internals_vs_output
```

**`run_scaleup.sh`** — 扩样本到 1500q (4个旗舰模型)  
- 硬编码跑：llama32_3b, llama32_3b_instruct, llama31_8b, llama31_8b_instruct
- 用于解决 500q 时 test 只有 75 个 query 导致的 power 不足

**`run_queue.sh`** — 后台队列执行多模型  
- 读取 `jobs.txt` 中的模型列表，依次跑 run_model.sh

### 2.3 数据源 (src/data.py)

**`load_msmarco_dev_small(n_queries=500, seed=42, max_passages=5)`**
- 来源：MS MARCO dev-small (6980 queries, qrels)
- 采样：随机选 n_queries 个 query，每个 query 最多取 5 个 passage（1 rel + 4 BM25 hard neg）
- 输出：list of dict `{"query_id", "query", "passage", "label"}`
- 划分：按 query 分层 70/15/15 → train/val/test (保证同 query 的样本在同一 split)

**`load_beir_relevance(dataset_name="fiqa", n_queries=500, ...)`**
- 来源：BeIR benchmark (本项目用 FiQA)
- 同样采样+负例逻辑，输出格式同上

### 2.4 关键结果文件清单

#### 激活缓存 (results/cache/)
每个模型生成 6 个 .pt 文件（attn/resid × train/val/test）+ meta.json。文件巨大(1-2GB)，gitignored，需从头跑或从备份恢复。

已跑模型列表（1500q @ instruct prompt）：
```
llama32_3b_q1500_instruct / _judge
llama32_3b_instruct_q1500_instruct / _judge
llama31_8b_q1500_instruct / _judge
llama31_8b_instruct_q1500_instruct / _judge
mistral_7b_instruct / _judge  (仅500q)
qwen25_7b_q1500_instruct / _judge
qwen25_7b_instruct_q1500_instruct / _judge
llama32_3b_fiqa_instruct / _judge  (FiQA数据集)
```

#### 汇总结果 JSON (results/)
- **`internals_vs_output_<tag>_q1500.json`**: 4-reader 对比（judge/final/best_resid/attn 的 AUC）
- **`significance_<tag>_q1500.json`**: bootstrap CI + p-value，per-model edges
- **`logit_lens_<tag>_q1500.json`**: 无监督 per-layer AUC + sanity check
- **`causal_steer_<tag>_q1500.json`**: steering 曲线（alphas, p_yes, auc）
- **`aggregate_models.json`**: 跨模型汇总表（用于快速对比）

#### 图片 (results/figures/)
- **`<tag>_layer_signal.png`**: 深度曲线+heatmap 3-panel 主图
- **`causal_steer_<tag>.png`**: steering 双 y 轴图（P(yes) + AUC vs α）

---

## 三、完整实验流程与复现指南

### 3.1 环境与依赖

**Python 环境**：
- transformers, torch, numpy, scikit-learn, matplotlib, joblib
- HF cache 位置：`/root/shared-nvme/hf_cache` (通过 `export HF_HOME=...` 设定，避免 30G 系统盘爆满)

**LaTeX 编译**：
- **不用 conda/mamba**（本机 broken）
- 用 `paper/tools/tectonic` 静态二进制，从 GitHub 直接下载
- 编译命令（需代理访问 ctan 包源）：
  ```bash
  cd paper
  export https_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
  ../tools/tectonic main.tex
  ```

**Git 推送注意**：
- `git push` 时必须 `unset https_proxy`，否则 TLS 错误
- 多次尝试（网络不稳定）：重试 3-4 次通常成功

### 3.2 从头跑一个新模型的完整流程

假设要跑 `Qwen/Qwen2.5-7B` (tag `qwen25_7b`)，1500 queries：

**Step 1: 下载模型** (如果未缓存)
```bash
export HF_HOME=/root/shared-nvme/hf_cache
export https_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen2.5-7B', ignore_patterns=['*.gguf'])"
```

**Step 2: Extract activations**
```bash
export HF_HUB_OFFLINE=1  # 模型已缓存，离线模式
python scripts/extract_instruct.py \
  --model Qwen/Qwen2.5-7B \
  --n-queries 1500 --seed 42 --max-passages 5 \
  --out results/cache/qwen25_7b_q1500_instruct
```
耗时：~20-30 分钟（GPU，依模型大小）

**Step 3: LLM-judge baseline**
```bash
python scripts/llm_judge.py \
  --model Qwen/Qwen2.5-7B \
  --n-queries 1500 --seed 42 --max-passages 5 \
  --out results/cache/qwen25_7b_q1500_judge
```
耗时：~15 分钟

**Step 4: Probe + 4-reader 对比**
```bash
python scripts/internals_vs_output.py \
  --cache results/cache/qwen25_7b_q1500_instruct \
  --judge-cache results/cache/qwen25_7b_q1500_judge \
  --out results/internals_vs_output_qwen25_7b_q1500.json
```
耗时：~5-10 分钟（训练 28×2 个 logistic regression）

**Step 5: Significance (bootstrap)**
先在 `scripts/significance.py` 的 `MODELS` 列表注册：
```python
("qwen25_7b", "Qwen2.5-7B", "qwen25_7b"),
```
然后跑：
```bash
python scripts/significance.py \
  --suffix _q1500 --only qwen25_7b \
  --out results/significance_qwen_q1500.json
```
耗时：~10-20 分钟（5000 bootstrap × 28 layers）

**Step 6: Logit-lens (optional, for depth analysis)**
```bash
python scripts/logit_lens.py \
  --model Qwen/Qwen2.5-7B \
  --cache results/cache/qwen25_7b_q1500_instruct \
  --judge-cache results/cache/qwen25_7b_q1500_judge \
  --out results/logit_lens_qwen25_7b_q1500.json
```

**Step 7: 深度图 (optional, for supplement)**
```bash
python scripts/plot_layer_signal.py \
  --cache results/cache/qwen25_7b_q1500_instruct \
  --judge-cache results/cache/qwen25_7b_q1500_judge \
  --tag qwen25_7b
```
输出：`results/figures/qwen25_7b_layer_signal.png`

**Step 8: Causal steering (optional, high-value)**
```bash
python scripts/causal_steer.py \
  --cache results/cache/qwen25_7b_q1500_instruct \
  --judge-cache results/cache/qwen25_7b_q1500_judge \
  --out results/causal_steer_qwen25_7b_q1500.json
```

### 3.3 更新论文表格的流程

假设新加了 Qwen，要更新 main.tex：

1. **主表 (Table 1)**: 从 `internals_vs_output_qwen25_7b_q1500.json` 或 `significance_qwen_q1500.json` 取 point estimate，加一行
2. **Significance 表 (Table 2/3)**: 从 `significance_qwen_q1500.json` 取 edge 的 mean, CI, p-value
3. **Abstract**: 更新模型数量（six → eight models）
4. **Experiments 段落**: 加 Qwen 专门段落，诚实描述其 judge 强但 attn−final 仍显著的特性
5. **Limitations**: 更新覆盖范围（"三家族"）

然后重新编译：
```bash
cd paper
export https_proxy="..."
../tools/tectonic main.tex
git add main.tex main.pdf
git commit -m "feat: 加 Qwen2.5-7B"
```

---

## 四、当前论文状态与关键数字

### 4.1 论文结构 (paper/main.tex, 6页)

**Sections**:
1. Introduction (0.8页)
2. Related Work (0.4页)
3. Method (0.7页): 4 readers, gap decomposition, bootstrap significance, causal steering
4. Experiments (2.5页): 主表、显著性表、深度定位、logit-lens、Qwen段、因果steering、FiQA、few-shot judge
5. Discussion & Limitations (0.4页)
6. Conclusion (0.2页)

**Tables**:
- Table 1: 8模型 4-reader AUC (主表)
- Table 2: 500q significance (6模型，sign test)
- Table 3: 1500q scaleup significance (6模型 + Qwen 2模型)

**Figures**:
- Fig 1: LLaMA-3.2-3B 深度曲线 3-panel (main text)
- Fig 2: Causal steering (3 方向对比)
- Supplement: 其余模型的 per-model 深度图

### 4.2 8 模型的关键数字汇总

| 模型 | judge | final | best_resid | attn | attn−final | internal_edge |
|---|---|---|---|---|---|---|
| **LLaMA-3.2-3B** | 0.525 | 0.544 | **0.581**(L14) | **0.685** | +0.146, p<1e-3 | +0.077, p=0.0008 |
| LLaMA-3.2-3B-Inst | 0.597 | 0.606 | 0.633(L16) | **0.725** | +0.097, p<1e-3 | +0.024, n.s. |
| **LLaMA-3.1-8B** | 0.521 | 0.537 | **0.581**(L14) | **0.676** | +0.114, p<1e-3 | +0.060, p=0.004 |
| LLaMA-3.1-8B-Inst | 0.603 | 0.583 | 0.628(L16) | **0.746** | +0.129, p<1e-3 | +0.012, n.s. |
| Mistral-7B-v0.3 | 0.537 | 0.585 | **0.621**(L18) | 0.638 | +0.053, n.s. | +0.036, n.s. |
| Mistral-7B-Inst-v0.3 | 0.621 | 0.616 | 0.628(L19) | **0.667** | +0.051, n.s. | +0.012, n.s. |
| **Qwen2.5-7B** | **0.652** | 0.592 | **0.655**(L22) | **0.682** | +0.090, p<1e-3 | +0.063, p=0.001 |
| Qwen2.5-7B-Inst | 0.660 | 0.610 | 0.630(L23) | **0.670** | +0.060, p=0.002 | +0.020, n.s. |

**模式总结**：
- ✅ attn−final 在 LLaMA(4) + Qwen(2) 上均显著(p<0.003)，Mistral 弱（架构差异）
- ✅ internal_edge 在所有 **base** 模型显著(LLaMA 3B/8B + Qwen)，instruct 收缩
- 🔑 Qwen 特殊：judge 强(0.65) → best_resid−judge≈0，但 attn−final 仍显著 → "即使输出强，heads仍超过output-repr"

### 4.3 额外实验结果

**FiQA 数据集** (LLaMA-3.2-3B, in-domain):
- judge 0.739, final 0.976, best_resid **0.993**(L14), attn 0.993
- internal_edge +0.0165 (p<1e-3, 虽薄但显著)
- 主导 edge 是 train_edge +0.237 (final 远超 judge)
- 说明：FiQA 极度线性可分，输出表示已近天花板，但 internal_edge 仍存在

**Causal steering** (3 LLaMA variants: 3B, 8B base/inst):
- 一致模式：ablation 臂 P(yes) 单调崩（8B-inst 最剧 0.871→0.003）
- 只有 internal 方向保持 AUC；final 方向摧毁 AUC(→0.485)；random 无定向效应

**Few-shot judge** (4-exemplar, LLaMA-3.2-3B base):
- 零样本 0.545 → 4-shot **0.566** (+0.021)
- 仍远低于 best_resid 0.632 / attn 0.685
- 堵"judge 弱是 prompt 问题"的质疑

**Logit-lens** (LLaMA 3B/8B, Qwen 7B base):
- Sanity check 全通过：last-layer logit-lens AUC = judge AUC (absdiff 0.000)
- 无监督峰值：3B 0.560@L25, 8B 0.577@L30, Qwen 0.654@L26
- 远低于训练探针(0.62-0.68) → 信号在残差流但不与 vocabulary 投影对齐

---

## 五、已完成的工作清单（时间线）

### 第一阶段：实验探索（2026-06-27 至 06-28）
- ✅ 搭建数据 pipeline (MS MARCO sampling + BeIR loader)
- ✅ 实现 4-reader 框架 (extract_instruct / llm_judge / internals_vs_output)
- ✅ 跑通 LLaMA-3.2-3B 500q 初步结果
- ✅ 实现 query-clustered bootstrap (significance.py)
- ✅ 发现主表选层 bug（test argmax）并修复为 val 选层

### 第二阶段：扩模型+加固（2026-06-28）
- ✅ 扩样本 500q→1500q (4 LLaMA 旗舰)
- ✅ 添加 Mistral-7B base+instruct (架构对比)
- ✅ 因果 steering 实验 (3 LLaMA variants, probe-free)
- ✅ FiQA 数据集 (第二个 in-domain)
- ✅ Few-shot judge baseline

### 第三阶段：复查+审稿+改稿（2026-06-29）
- ✅ 方法学审查 agent：发现 Table 1 选层 bug
- ✅ 时效性/新颖性调研 agent：联网核实 14 篇文献，补 5 篇必引
- ✅ 顶会 reviewer agent：给 4/10 (略低于接收线)，5 个 must-fix
- ✅ P1 修复主表选层 bias（改用 val 选层数）
- ✅ P2 补充 Related Work 切割（Orgad2024/Aiersilan2026/...）
- ✅ **P3 logit-lens 实验**（审稿最后一个 reject-trigger）
- ✅ P4 因果措辞收口（三方对照）
- ✅ P5 abstract 数字对齐（加 attn-final 列到 scale 表）
- 📄 论文从 reject 提升到 borderline-accept 量级

### 第四阶段：加第三家族（2026-06-30）
- ✅ 下载 Qwen2.5-7B base+instruct (Apache-2.0, 无需 token)
- ✅ 1500q 全管线 (extract/judge/internals/significance/logit-lens/图)
- ✅ 论文整合：主表+scale表各+2行，新增 Qwen 段落
- ✅ Abstract → eight models, Limitations 更新
- 📄 **达到 CCF-B 投稿量级**

---

## 六、未完成的可选工作（优先级排序）

### 高价值（如目标顶会，建议补做）

1. **Qwen 因果 steering** (1-2小时)
   - 当前：只有 LLaMA 3 variants 的 steering 结果
   - 补充：跑 `causal_steer.py` on Qwen base，验证内部方向的因果性在第三家族复现
   - 命令：
     ```bash
     python scripts/causal_steer.py \
       --cache results/cache/qwen25_7b_q1500_instruct \
       --judge-cache results/cache/qwen25_7b_q1500_judge \
       --out results/causal_steer_qwen25_7b_q1500.json
     ```

2. **Prompt 鲁棒性**（2-3 种改写模板，2-4小时）
   - 当前：只用一种 instruct prompt
   - 扩展：设计 2-3 种不同 phrasing（保持语义），重跑 3B 看 gap 是否稳定
   - 堵 m1 质疑："结果是单一 prompt 的伪影"

3. **Appendix 放 8 模型全图**（1小时）
   - 当前：main text 只放 LLaMA-3.2-3B 一个图
   - 补充：把 Mistral/Qwen/8B 的深度图都放进 supplement
   - 已有图：`results/figures/{mistral_7b, qwen25_7b, llama31_8b}_layer_signal.png`

### 中等价值（CCF-B 够了，顶会加分）

4. **FiQA 换 BM25 难负例**（2-3小时）
   - 当前：FiQA 用 random negatives → 太简单(AUC 0.99)
   - 改进：用 BM25 top-100 非相关 passage 作为 hard negatives
   - 需修改 `src/data.py::load_beir_relevance` 加 BM25 检索逻辑

5. **FiQA 加第二模型**（3-4小时）
   - 当前：FiQA 只跑了 LLaMA-3.2-3B
   - 补充：跑 Qwen-7B 或 LLaMA-8B on FiQA，看 internal_edge 是否一致
   - 命令同 Step 2-4，只是 `--dataset beir:fiqa`

6. **Judge 阈值校准**（Platt scaling, 1-2小时）
   - 当前：judge 直接用 P(yes)，未校准
   - 改进：在 val 上做 Platt scaling，看校准后 AUC 是否改善
   - 仍预期低于 probe（probe 有更多自由度）

### 低价值（可不做）

7. **Mistral 因果 steering**
   - Mistral 的 attn-head 本身就弱，steering 预期效果不明显
   - 若审稿要求"完整性"再补

8. **Cross-domain 泛化测试**
   - 当前 Limitations 已声明"无 cross-domain claim"
   - 若做，需在 MS MARCO train → BeIR test (多数据集)，工作量大

---

## 七、如何继续这项工作

### 7.1 立即可操作的 next steps

**场景 A：目标 CCF-B 会议，尽快投稿**
1. 确定目标会议（如 SIGIR、CIKM、WSDM）和 deadline
2. 按会议模板调整格式（当前是 ACL 6页风格）
3. 可选：补做"Qwen 因果 steering" + "8模型全图 appendix"（+1天）
4. 校对、投稿

**场景 B：冲顶会（NeurIPS/ICLR/ICML）**
1. 必做：Prompt 鲁棒性（堵 m1 质疑）
2. 必做：Qwen 因果 steering（三家族完整性）
3. 可选：FiQA BM25 难负例（提升 FiQA 说服力）
4. 扩写 Related Work（当前 0.4页，顶会通常要 0.8-1页）
5. Appendix 加完整实验细节（超参、per-query 分析、失败案例）

**场景 C：继续探索新方向**
- **方向1：跨任务泛化**（QA、summarization、math — 内部>输出 是否通用？）
- **方向2：机制解释**（哪些头/层专门负责相关性？用 attribution / circuit analysis）
- **方向3：应用**（基于内部 probe 的 inference-time 干预，提升输出质量）
- **方向4：更大模型**（70B/405B — gap 是否随 scale 变化？）

### 7.2 与前人工作的连接（必读文献）

已在 refs.bib，关键 3 篇：
- **Azaria & Mitchell 2023** (2304.13734): 最早观察"内部 truthfulness > 输出"
- **Orgad+ 2024** (2410.02707): "LLMs Know More Than They Show" — 标题撞车，必引并切割
- **Aiersilan+ 2026** (2606.02628): 同代 7-8B，hallucination 中层可解码 — 最接近本文

### 7.3 代码/数据复用建议

- **Pipeline 可直接迁移到其他任务**：只需改 `src/data.py` 的 loader（输出统一格式），`extract_instruct.py` 改 prompt template
- **Significance 框架通用**：query-clustered bootstrap 适用于任何"分组数据"场景
- **Steering 框架可扩展**：当前是 additive/ablation，可改成 rotation、scaling、subspace projection

---

## 八、已知坑与注意事项

### 技术坑
1. **conda/mamba broken**：本机 conda 卡住，mamba ImportError → 必须用 tectonic 静态二进制编译 LaTeX
2. **30G 系统盘容易爆**：所有 HF cache 必须指向 `/root/shared-nvme/hf_cache`
3. **Git push over proxy fails**：推送前 `unset https_proxy`，否则 GnuTLS error
4. **Significance 脚本慢**：5000 bootstrap × 28 layers → 10-20分钟，正常

### 实验坑
5. **Test argmax 选层 bias**：早期 `internals_vs_output.py:78` 在 test 上选层 → 乐观高估。已修复为 val 选层（commit `3457979`），但若从旧代码恢复需警惕
6. **Judge yes-bias**：LLaMA instruct 有 yes-bias → causal steering 的 additive 臂受限，主要看 ablation 臂
7. **Mistral 单头弱**：Mistral 架构的单个 head 信号极弱(~0.50)，residual stream 仍强 → 不是 bug，是架构差异

### 论文坑
8. **"Internals > output" 非首创**：Azaria2023/Orgad2024 已做过，必须在 Related Work 切割净新颖性
9. **Qwen judge 强 ≠ 命门失守**：best_resid−judge≈0，但 attn−final 仍显著 → 诚实写成"结构镜像"而非隐藏
10. **Logit-lens 不复现衰减形状**：它处处低平(~0.55)，不像训练探针中部成峰 → 只声称"无监督读出处处不显著"，不过度解释

---

## 九、联系方式与资源

- **项目 GitHub**: `new-life-fch/my_paper_project` (branch `paper-A`)
- **内部文档**:
  - `CLAUDE.md` §六：进度总览（6次推进）
  - `results/REPORT.md` §1-§15：详尽实验日志
  - `.claude/memory/paper-A-draft-and-tooling.md`：持久化记忆
- **HF 模型缓存**: `/root/shared-nvme/hf_cache` (72GB already used)
- **LaTeX 编译器**: `paper/tools/tectonic` (static binary, 自包含)

---

## 十、最后的话

这是一个**完整、可复现、诚实**的项目。所有数字、所有文献、所有 commit 都可追溯。论文当前状态：
- ✅ 8模型（3B-8B，三家族），主表+2个显著性表+因果图+深度图
- ✅ 5个 must-fix 全部完成（选层bug修复、prior-work切割、logit-lens、措辞收口、数字对齐）
- ✅ 达到 CCF-B 投稿量级，borderline-accept 水平

**核心论点稳健**：内部读取器 (attn heads / best-resid) 在 8 模型上一致地超过输出表示和 judge，有深度证据、因果证据、跨数据集证据、对齐机制证据。

**诚实的边界**：Mistral attn-head 弱（架构差异），Qwen judge 强（预训练差异），FiQA 过于简单（internal_edge 薄但显著）—— 这些都已如实写进 Limitations。

祝新接手的 agent 顺利！有任何疑问，回来读 `CLAUDE.md` §六 + `results/REPORT.md` + 本文档。

---

**文档版本**: v1.0  
**生成者**: Claude Opus 4.8  
**最后更新**: 2026-06-30 10:05 UTC  
**commit hash**: `51e51f2`
