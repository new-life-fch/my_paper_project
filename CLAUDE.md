# CLAUDE.md — 工作区 A：Internals > Output（探针主线）

> 本工作区是 git worktree（分支 `paper-A`），与工作区 B（`paper-B`，cheap-adaptation）**物理隔离**。
> 主仓库 `/root/shared-nvme/my_paper_project`（main 分支）保留全部历史，请勿在此改动它。
> 语言：回复与文档优先用中文。状态：主线已验证，进入论文化实验阶段。

---

## 〇、这个工作区是干什么的（一句话）

证明并刻画一个发现：**LLM 内部激活编码了「检索文档是否与 query 相关」的信号，但这个信号在传向输出的过程中被衰减，导致模型「嘴上」（输出 logits）表达不出来**。一句话口号：**Internals know more than the model can say.**

这是一篇**可解释性 / 探针（probing）方向**的论文，不是「做一个更好的 RAG reranker」。务必记住：**本主线不声称探针在重排精度上打赢 reranker（那个已被证伪，见第四节废弃方案）。** 本主线的价值在于「内部 vs 输出」的科学发现与机制刻画。

---

## 一、会话规则（每次新会话必读）

1. **会话压缩后恢复**：立即重读本文件全文，根据第六节「当前进度 / 下一步」恢复上下文继续。
2. **会话压缩前保存**：在上下文接近上限前，主动把进度更新到第六节（标注「第几次压缩」，精简旧内容）。本工作区**不用单独的 WORK_STATUS.md**，状态直接维护在本文件第六节。
3. **禁止胡编**：所有文献信息、技术结论必须可溯源，不确定的标「未确认 / 推断」。
4. **诚实优先**：这个项目的全部价值建立在诚实归因上。实验若推翻当前结论，如实记录并更正本文件与 memory，不得粉饰（前期正是靠这个原则发现「探针打不过 reranker」）。

---

## 二、作者授予的权限与工作方式（沿用主仓库）

- **完全自主权**：已获授权自主迭代，不要因小决策频繁打断用户；里程碑处提交 git。
- **可下载**：所需 skills / 数据集 / 模型可下载，但**不下载超大模型**（磁盘未扩容；当前已缓存 LLaMA-3.2-3B，够用；8B 级别需先确认磁盘）。
- **可用后台任务 / 并行**：长实验用后台跑；独立任务可并行。
- **HF token**：用户已提供（如失效向用户索取）。`huggingface-cli` 在本环境损坏，用 Python `huggingface_hub.login()` API。
- **提交规范**：里程碑式提交，message 写清「做了什么 + 结论」，带 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`。提交前留意不要把 `results/`（已 gitignore）或密钥提交进去。

---

## 三、敲定的方案：主线论点与已验证的核心结果

### 3.1 核心论点（三段式）

1. **存在性**：用线性探针读 LLaMA-3.2-3B 在 answer-token 处的注意力头激活，能判断 (query, passage) 相关性，AUC 0.685，**显著高于**让模型直接输出判断的 LLM-judge（0.535，≈随机）。
2. **归因（关键对照，已通过）**：这个优势主要来自「读内部」而非「被训练」。把 +0.15 的总 gap 拆解：
   - 训练一个**只读输出表示**（最终层 hidden）的探针只到 **0.593**（训练贡献 +0.058）；
   - 读**内部**（中层残差 / 注意力头）到 0.632 / 0.685（读内部额外贡献 +0.092，是训练贡献的 1.6 倍）。
3. **机制**：相关性信号沿网络深度的可解码性**中部（L12）最高（0.632），向输出层单调衰减（0.593）** → 信号在内部已形成，但在抵达 logits 前被稀释/抑制。

### 3.2 已验证结果（截至 worktree 创建，commit a296676）

| 读取位置 | 训练 | test AUC | 数据来源 |
|---|---|---|---|
| 输出 logits（LLM-judge）| 否 | 0.535 | `results/cache/q500_judge` |
| 最终层 hidden（输出表示）| 是 | 0.593 | `internals_vs_output.py` |
| 中层残差 L12（内部最优）| 是 | 0.632 | 同上 |
| 注意力头探针（本方法）| 是 | 0.685 | 同上 |

数据集：MS MARCO v1.1，500 queries，instruct prompt + answer-token，按 query 70/15/15 切分（防泄露）。
模型：LLaMA-3.2-3B（28 层 / 24 头 / head_dim 128 / 共 672 头）。

### 3.3 prompt 与提取约定（固定，勿随意改）

- instruct prompt：`"Document: {passage}\nQuestion: {query}\nIs the document relevant to the question? Answer:"`，取 **answer token（末位）** 激活。
- 注意力头激活：nnsight `model.trace()` 内逐层取 `model.model.layers[i].self_attn.o_proj.output`，reshape `[B,S,n_heads,head_dim]` 取末位 token。
- 残差流：`resid` 缓存为每层 decoder layer 输出 `[N, 28, 3072]`。
- transformers 5.x：`o_proj.output` 已是 `[B,S,H]`，**不要再 `[0]` 索引**；nnsight 变量声明放 `with model.trace()` 块外、用 `.save()`。

---

## 四、方案探索史与**废弃方案**（重要：避免重走弯路）

本项目最初定位是「基于注意力头探针的 RAG 检索优化，目标打赢 cross-encoder reranker」。经系统实验，**原定位被证伪**，逐条记录已死的路，**A 分支不要再碰**：

1. **❌「探针重排精度优于 reranker」**：同域 MS MARCO 探针 0.685 < 强 reranker `bge-reranker-v2-m3` 0.753。**调参无用**（top-k 5~200、StandardScaler、whole-L2、mean-probe 全扫过，最优仍 0.685；且 StandardScaler 在注意力头探针上反而有害）。**加数据无用**（数据效率曲线单调缓升，外推 2000q≈0.72 仍 <0.753）。→ 不要再试图用探针在重排榜上赢 reranker。
2. **❌「探针天生跨域鲁棒」**：零样本迁移（MS MARCO 探针→新域）SciFact 0.795 / FiQA 0.627，**两域都输**给 reranker 零样本（0.921 / 0.789）。强 reranker 跨域不降反升。→ 「跨域天生强」已死。**跨域 / 适配相关的一切归 B 分支**，A 分支不做跨域。
3. **❌「数据效率高」**：曲线显示探针要更多数据才涨，不是少样本优势。→ 不作卖点。
4. **⚠️ 曾经的假结论**：早期 SciFact 探针 0.964「反超」reranker，是**不公平对比**（探针在 SciFact 重训、reranker 零适配）造成的假象，已被 zeroshot_transfer 推翻。**勿引用 0.964 作为 A 的证据。**

**A 分支唯一保留并深耕的，是「internals > output」这一可解释性发现**（第三节），它与重排精度、跨域都无关，是独立成立的科学结论。

> 与 B 分支的边界：B 做「廉价领域适配 vs LoRA 微调 reranker」的实用价值论证。**A 不碰跨域、不碰 reranker 对比、不碰适配**。

---

## 五、脚本说明（本工作区保留的脚本，各自干什么）

| 脚本 | 作用 | 备注 |
|---|---|---|
| `scripts/setup_env.sh` | 环境安装 + 验证（torch/nnsight/transformers/sklearn/datasets 版本与 CUDA） | 新容器首次必跑 |
| `scripts/extract_instruct.py` | **激活提取器**：用 instruct prompt 在 answer-token 处提取 attn（per-head）+ resid（每层残差）双位点激活，缓存为 float16 .pt。输出 `results/cache/q500_instruct/{attn,resid}_{train,val,test}.pt` + `meta.json` | A 的数据来源生产者 |
| `scripts/llm_judge.py` | **输出端基线**：同模型同 prompt，读 answer-token 处 `P("yes")` vs `P("no")` 输出概率，零训练。输出 `results/cache/q500_judge/judge_*.pt`（scores/labels/query_ids） | internals>output 的「输出端、零训练」对照 |
| `scripts/internals_vs_output.py` | **A 的核心实验**：在同一缓存上比较 ①LLM-judge(输出,0训练) ②最终层 hidden 探针(输出表示,训练) ③逐层残差探针(找信号最强层) ④注意力头探针(本方法)。输出逐层 AUC 曲线 + 四者对比，落 `results/internals_vs_output.json` | 跑它复现第 3.2 节结果 |
| `scripts/run_model.sh` | **多模型驱动**：`bash scripts/run_model.sh <hf_model_id> <tag>` 串联 extract_instruct→llm_judge→internals_vs_output，输出 `results/cache/{tag}_instruct`、`{tag}_judge`、`internals_vs_output_{tag}.json`。模型无关（脚本动态读 n_layers/n_heads） | 多模型复现 gap 用 |
| `scripts/plot_layer_signal.py` | **论文主图生成器**：从 `{tag}_instruct` 缓存计算并出三联图（A 逐层 resid AUC 深度曲线+judge 基线+峰值/输出层标注；B per-head AUC 热力图 layer×head；C head 视角 vs layer 视角+top-k head 层分布直方图）。落 `results/figures/{tag}_layer_signal.{png,json}`。`--cache --judge-cache --tag --topk` | 跑它出信号层定位图 |
| `scripts/aggregate_models.py` | **多模型汇总**：扫描所有 `internals_vs_output_<tag>.json`，输出对比表（judge/final/best_resid/attn + peak_frac + internal_edge=best_internal−final + train_edge=final−judge + holds 判定），落 `results/multimodel_summary.json` | 多模型实验汇总用 |
| `scripts/sweep_topk.py` | 探针超参敏感性扫描（top-k∈{5..200}×raw/scaled×whole-L2×mean-probe），证明结论对调参稳健 | 已证调参不改变结论，按需复用 |
| `scripts/compare_probes.py` | 6 种探针方法论对比（baseline 选头 / StandardScaler / CV 选头 / whole-L2 / stacking / mean-probe），数据驱动选最优探针架构 | 探针架构消融 |
| `src/data.py` | `load_ms_marco()` 造 (query,passage,label) 三元组；`split_by_query()` 按 query_id 70/15/15 切（防泄露） | |
| `src/activations.py` | nnsight 激活提取底层（last_token / pooling 两方案，o_proj.output reshape 成 per-head） | |
| `src/probes.py` | 两阶段探针：Stage1 per-head L1 逻辑回归选头，Stage2 top-k 拼接 L2 ensemble | |
| `src/evaluation.py` | 可视化：per-head ROC-AUC 热力图、top-k 曲线 | 画论文图用 |
| `initial_validation.py` | 早期 M1+M2 主入口（编排 load_model→sanity_check→提激活→训探针→方案对比），含 `sanity_check()` 验证 nnsight 激活 shape | 参考用，新实验优先用 scripts/ 下专用脚本 |
| `scripts/download_model.sh` | classic-HTTP 断点续传下载 LLaMA-3.2-3B（`HF_HUB_DISABLE_XET=1`，hf_xet 须卸载） | 模型已缓存则跳过 |

> 已删除的脚本（属于 B 分支或废弃方案，A 不需要）：`build_ood*.py`、`compare_ood.py`、`zeroshot_transfer.py`、`strong_reranker.py`、`compare_methods.py`、`data_efficiency.py`、`extract_and_cache.py`（旧 plain prompt 提取器，已被 extract_instruct 取代）。

---

## 六、当前进度 / 下一步（实时维护，压缩前必更新）

### 当前进度（第 1 次会话推进，2026-06-27）

- ✅ 主线敲定为 internals>output；核心对照实验已通过（commit a296676，结果见 3.2）。
- ✅ 本工作区复现核心结果无误：judge 0.535 → final-layer 0.593 → L12 0.632 → attn-head 0.685。
- ✅ **论文主图已出**：`scripts/plot_layer_signal.py` 三联图（A 逐层 resid 深度曲线+judge 基线+峰值/输出标注；B per-head 热力图；C head vs layer 视角）。3B top-20 heads 集中 **L10-16**，与 resid 峰值 L12 一致。
- ✅ **磁盘修复**（重要）：系统盘仅 30G，HF 缓存(21G)已搬到数据盘 `/root/shared-nvme/hf_cache` 并软链回 `~/.cache/huggingface`；`HF_HOME` 固化进 `run_model.sh`/`run_queue.sh`。系统盘 72%→6%。已写记忆 `disk-layout-server.md`。
- ✅ **5 模型队列已就绪并自主跑**：`scripts/run_queue.sh`（串行 base+instruct，单模型失败不阻塞，每个完成后自动 aggregate+出图）。检查点 cron（每小时:37）自主分析。

#### 多模型结果（截至本次，命门检验 + 机制对照）

| 模型 | L | judge | final(输出表示) | best_resid(内部) | attn(本方法) | peak层占比 | internal_edge | train_edge | holds |
|---|---|---|---|---|---|---|---|---|---|
| LLaMA-3.2-3B (base) | 28 | 0.535 | 0.593 | 0.632(L12) | 0.685 | 0.44 | **+0.092** | +0.058 | ✅ |
| LLaMA-3.1-8B (base) | 32 | 0.514 | 0.552 | 0.634(L5) | 0.680 | 0.16 | **+0.128** | +0.038 | ✅ |
| LLaMA-3.2-3B-Inst | 28 | 0.597 | 0.607 | 0.704(L12) | 0.725 | 0.44 | **+0.118** | +0.010 | ✅ |

**命门安全**：3/3 模型 internal_edge > train_edge，且 internal_edge 在 8B 上(+0.128)比 3B 更大——内部效应是训练效应的 ~3.4 倍，主线随规模增大反而更强，未削弱。

**机制对照（base vs instruct，关键发现，3B 同尺寸对比）**：
- judge 0.535→**0.597**（instruct 嘴上判断变强，对齐确实提升输出端表达）；
- train_edge 0.058→**0.010**（几乎归零：instruct 上"训练读输出表示"已无增量，对齐已把信号推进输出表示）；
- **但 internal_edge 仍高 +0.118，best_resid 飙到 0.704**——内部信号依然远超输出能表达的。
- → 论点比原假设更强：**不是"对齐压制输出"，而是无论是否对齐，内部始终比输出知道得多**。对齐改善了输出端、缩小了"训练 gap"，却没消除"内部 gap"。

🔄 队列进行中：Mistral-7B-v0.3 (base) 下载中(~5G/14G)，之后 8B-Inst、Mistral-Inst。系统盘稳定 6%。

### 下一步（论文化，按优先级）

1. 🔄 **多模型复现 gap**（验证普适性，进行中）：在 LLaMA-3.1-8B、Mistral-7B-v0.3 上复现「内部探针 > 输出端探针 > judge」的层级。工具链已就绪（`run_model.sh`/`aggregate_models.py`）。8B 后台跑中，完成后接 Mistral。**关键看命门对照线**（见关键风险）：8B 上 final-layer 探针是否仍显著低于 attn-head 探针。
2. ✅ **信号层定位图**（论文主图，已完成）：`scripts/plot_layer_signal.py`。3B 已出图，多模型缓存就绪后对每个模型出同款图即可。
3. **机制解释实验**：为什么信号到输出被衰减？候选假设——(a) RLHF/对齐使模型倾向保守拒答；(b) 相关性信号未被路由到输出 token。可设计干预（如对 judge prompt 变体、base vs instruct 模型对比）验证。注：本主线用的 LLaMA-3.2-3B 与 3.1-8B 均为 **base** 模型，若再加 instruct 版对比可直接验证假设 (a)。
4. **gap 的鲁棒性**：多 prompt 框架（换 instruct 模板）下复现 gap，排除「特定 prompt 造成 judge 偏弱」的质疑。可调一个更强的 judge（few-shot / 校准阈值）作为更公平的输出端上界。

### 预期实验结果（写论文前的假设，需实验证实）

- 多模型上「内部 > 输出端 > judge」层级**应稳定复现**（gap 量级可能随模型变化）。
- 信号峰值层**应在网络中后段**；输出层解码能力**应低于峰值层**（这是机制论点的关键，若不成立则论点削弱）。
- 若某模型上输出端探针追平内部探针 → 如实记录，可能说明该结论与模型对齐方式有关。

### 关键风险

- 若「输出端探针（最终层 hidden）」在更强训练 / 更多数据下追平注意力头探针，则「internals>output」会退化为「训练>不训练」，主线受损。**这是 A 分支的命门**，多模型实验要重点盯这条对照线。

---

## 七、环境（新容器必看，会复现的坑）

| 项 | 值 |
|---|---|
| 工作区目录 | `/root/shared-nvme/paper_A_internals`（分支 paper-A） |
| GPU | RTX 3090 24GB | Python 3.12 | torch 2.7 | nnsight 0.7.0 | transformers 5.x | datasets 3.6.0 |

**会复现的坑：**
1. **新容器依赖需重装**：仅 torch/sklearn/numpy/matplotlib 预装，需装 transformers/datasets/sentence-transformers/nnsight/accelerate/seaborn。
2. **dill pip 约束**：`/etc/pip/constraint.txt` 把 dill 钉死 0.3.9，直接 `pip install datasets` 会静默装成远古 1.1.1。解法：`PIP_CONSTRAINT="" pip install 'datasets>=3.0,<4'`。
3. **hf_xet 与代理不兼容**：下载大文件卡死。解法：`pip uninstall hf_xet` + `export HF_HUB_DISABLE_XET=1`。
4. **代理**（下载 HF/PyPI 慢时）：
   ```bash
   export https_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
   export http_proxy="$https_proxy"
   export no_proxy="127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.paracloud.com,*.paratera.com,*.blsc.cn"
   export HF_HUB_DISABLE_XET=1
   ```
5. **缓存是软链**：`results/cache` → 主仓库真实目录。删 worktree 前勿 `rm -rf` 跟随软链误删主缓存。

**常用命令：**
```bash
cd /root/shared-nvme/paper_A_internals
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1   # 模型已缓存
python scripts/internals_vs_output.py --cache results/cache/q500_instruct   # 复现核心结果
```
