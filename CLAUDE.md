# CLAUDE.md — 工作区 B：Cheap Adaptation（廉价领域适配）

> 本工作区是 git worktree（分支 `paper-B`），与工作区 A（`paper-A`，internals>output）**物理隔离**。
> 主仓库 `/root/shared-nvme/my_paper_project`（main 分支）保留全部历史，请勿在此改动它。
> 语言：回复与文档优先用中文。状态：现象已观测（两域），核心公平对照实验**待做**。

---

## 〇、这个工作区是干什么的（一句话）

论证一个**实用价值**命题：**面对新领域、只有少量目标域标注 + 有限算力时，把"冻结通用 LLM 上的线性探针"廉价适配过去，其性价比优于微调一个专用 reranker。**

口号：**Adapt a linear probe in minutes, not fine-tune a transformer for hours.**

这是一篇**RAG 检索 / 高效适配方向**的论文。**核心卖点是"性价比（精度/算力/标注代价）"，不是单纯精度。** 务必记住：零样本下探针打不过 reranker（已证伪），本主线只在"给定目标域少量标注 + 公平算力预算"的设定下立论。

---

## 一、会话规则（每次新会话必读）

1. **会话压缩后恢复**：立即重读本文件全文，根据第六节「当前进度 / 下一步」恢复上下文继续。
2. **会话压缩前保存**：在上下文接近上限前，主动把进度更新到第六节（标注「第几次压缩」，精简旧内容）。本工作区**不用单独的 WORK_STATUS.md**，状态直接维护在本文件第六节。
3. **禁止胡编**：所有文献信息、技术结论必须可溯源，不确定的标「未确认 / 推断」。
4. **诚实优先（B 分支的命门）**：当前"探针适配 0.964/0.860 > reranker 零样本 0.921/0.789"是**不公平对比**（探针看了目标域标注、reranker 没看）。**B 分支存在的全部意义就是把这个对比做公平**（给 reranker 同等适配机会）。若公平对比后探针输了，必须如实记录、调整论点，不得粉饰。

---

## 二、作者授予的权限与工作方式（沿用主仓库）

- **完全自主权**：已获授权自主迭代，不要因小决策频繁打断用户；里程碑处提交 git。
- **可下载**：所需 skills / 数据集 / 模型可下载，但**不下载超大模型**（磁盘未扩容；当前已缓存 LLaMA-3.2-3B 与 bge-reranker-v2-m3，够用）。
- **可用后台任务 / 并行**：长实验（尤其 LoRA 微调）用后台跑；独立任务可并行。
- **HF token**：用户已提供（如失效向用户索取）。`huggingface-cli` 损坏，用 Python `huggingface_hub.login()` API。
- **提交规范**：里程碑式提交，message 写清「做了什么 + 结论」，带 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`。提交前留意不要把 `results/`（已 gitignore）或密钥提交进去。

---

## 三、敲定的方案：主线论点与已观测的现象

### 3.1 核心论点（性价比框架）

不是问「谁零样本强」（reranker 赢），而是问：**给定目标域 N 条标注 + 算力预算 C，谁的检索质量更高？**
- **探针适配** = 在目标域激活上重训一层逻辑回归（CPU 几分钟、几百条标注、模型权重全冻结、零额外部署）。
- **reranker 适配** = LoRA / 全量微调一个 transformer（GPU、更多时间、需调参）。
论点：探针适配以**低一个数量级的代价**，达到接近甚至超过 reranker 微调的检索质量。

### 3.2 已观测现象（两域一致，commit a296676 时点）

| 设置 | SciFact(科学) | FiQA(金融) |
|---|---|---|
| 探针 零样本（MS MARCO→目标域）| 0.795 | 0.627 |
| 强 reranker bge 零样本 | 0.921 | 0.789 |
| **探针 适配**（目标域 train 重训）| **0.964** | **0.860** |
| 老 CE (MiniLM) 零样本 | 0.858 | 0.725 |

**适配增益大且普遍**：SciFact +0.169、FiQA +0.233（零样本→适配）。**但适配后探针 vs reranker 零样本是不公平对比**——这正是第六节要补的实验。

### 3.3 数据与提取约定（固定）

- OOD 数据集：BEIR 系列（`BeIR/scifact`、`BeIR/fiqa`、可扩 `BeIR/nfcorpus`、`BeIR/trec-covid`）。
- 候选池：每 query 用 BM25 取 top-20 构成重排池（模拟真实 RAG 检索器输出，含难负例）；金标若未被 BM25 召回则注入替换最差候选。
- 三元组 (query, passage, label)，按 query 50/20/30 切分（防泄露）。
- instruct prompt：`"Document: {passage}\nQuestion: {query}\nIs the document relevant to the question? Answer:"`，取 answer token attn 激活。
- 探针：Stage1 per-head L1 选 top-k（按 val AUC），Stage2 top-k 拼接 L2 ensemble。
- 模型：LLaMA-3.2-3B（672 头）；reranker：`BAAI/bge-reranker-v2-m3`（当前最强开源通用 reranker，XLM-RoBERTa backbone，已缓存）。
- 池构造无偏差已验证（passage 长度单独 AUC≈0.49，非表面特征捷径）。

---

## 四、方案探索史与**废弃方案**（重要：避免重走弯路）

项目最初定位「探针打赢 cross-encoder reranker」，经实验大幅修正。逐条记录已死的路：

1. **❌「探针重排精度优于 reranker」（同域）**：MS MARCO 探针 0.685 < 强 reranker 0.753，调参/加数据均不翻盘。→ 不在同域重排榜上跟 reranker 拼精度。
2. **❌「探针天生跨域鲁棒（零样本）」**：MS MARCO 探针零样本迁移到 SciFact 0.795 / FiQA 0.627，**两域都输**给 reranker 零样本（0.921/0.789）。强 reranker 跨域不降反升。→ **不要声称零样本跨域强**。B 分支的论点建立在"有少量目标域标注"的前提上，不是零样本。
3. **❌「数据效率高 / 少样本优势」**：数据效率曲线单调缓升，探针要更多数据才涨。→ 不作"少样本"卖点；但"适配只需重训线性层"的**低算力代价**仍是 B 的合法卖点（区别：省的是算力/部署，不是标注量）。
4. **⚠️ 曾经的假结论**：早期把 SciFact 探针 0.964「反超」reranker 当作"跨域鲁棒"证据——错。它是"适配 vs 零适配"的不公平对比。**B 分支要做的就是修正这个不公平**，而不是继续引用它当胜利。

> 与 A 分支的边界：A 做「internals>output」可解释性发现（同域、读内部 vs 读输出）。**B 不碰 internals>output、不碰 LLM-judge 对照、不碰逐层信号定位**；B 只管"廉价适配的性价比"。

---

## 五、脚本说明（本工作区保留的脚本，各自干什么）

| 脚本 | 作用 | 备注 |
|---|---|---|
| `scripts/setup_env.sh` | 环境安装 + 验证 | 新容器首次必跑 |
| `scripts/download_model.sh` | classic-HTTP 断点续传下载模型（`HF_HUB_DISABLE_XET=1`） | 模型已缓存则跳过 |
| `scripts/build_ood.py` | **OOD 数据生产者（通用）**：对任意 `BeIR/<name>` 数据集，BM25 建候选池→造三元组→按 query 切分→提取 LLaMA instruct attn 激活→缓存。用法 `--dataset BeIR/fiqa --out results/cache/fiqa_ood`。输出 `triples.json` + `attn_{train,val,test}.pt` + `meta.json` | 扩新域就跑它 |
| `scripts/extract_instruct.py` | instruct prompt + answer-token 激活提取器（attn+resid 双位点）。MS MARCO 同域缓存的生产者（探针源域训练用） | 源域探针来自这里 |
| `scripts/zeroshot_transfer.py` | **零样本迁移评估**：探针只在源域(MS MARCO)选头+训练，零适配打分目标域 test；对比 reranker 零样本、BM25。用法 `--src results/cache/q500_instruct --tgt results/cache/<dom>_ood` | 产出"探针零样本输"那一行 |
| `scripts/compare_ood.py` | **适配探针评估**：探针在目标域 train 重训（看目标域标注），打分目标域 test；对比 cross-encoder、BM25，输出 per-query MRR/NDCG/Recall。用法 `--cache results/cache/<dom>_ood` | 产出"探针适配"那一行 |
| `scripts/strong_reranker.py` | **强 reranker 基线**：用 `bge-reranker-v2-m3` 打分。`--which msmarco/scifact/both` 或 `--which cache --cache results/cache/<dom>_ood` 打分任意 OOD 缓存的 test | reranker 零样本基线 |
| `src/data.py` | `load_ms_marco()` + `split_by_query()`（按 query 防泄露切分） | |
| `src/activations.py` | nnsight 激活提取底层 | |
| `src/probes.py` | 两阶段探针（per-head L1 选头 + L2 ensemble） | |
| `src/evaluation.py` | 可视化 | 画论文图用 |
| `initial_validation.py` | 早期 M1+M2 主入口（参考用） | 新实验优先用 scripts/ 专用脚本 |

> 已删除的脚本（属于 A 分支或废弃方案）：`internals_vs_output.py`、`llm_judge.py`、`compare_probes.py`、`sweep_topk.py`（A 用）；`compare_methods.py`、`data_efficiency.py`、`build_ood_scifact.py`（被 build_ood.py 取代）、`extract_and_cache.py`（旧 plain）。

---

## 六、当前进度 / 下一步（实时维护，压缩前必更新）

### 当前进度（worktree 初始化时，第 0 次压缩）

- ✅ 现象在两个领域观测到：适配增益大且普遍（SciFact +0.169，FiQA +0.233）。
- ✅ 缓存就绪并软链共享：`results/cache/`→ 主仓库（q500_instruct 源域、scifact_ood、fiqa_ood）。
- ✅ 强 reranker 基线就位（bge-reranker-v2-m3 已缓存，零样本数已测）。
- ✅ 池构造偏差已排除（长度单独 AUC≈0.49）。
- ❌ **关键缺口**：缺"适配后 reranker"——当前对比不公平。

### 下一步（按优先级）

1. **【最高优先 / 决定论文成败】LoRA 微调 reranker 作公平对手**：
   - 用 `bge-reranker-v2-m3` 在目标域（SciFact / FiQA）的 **同一 train 三元组**上做 LoRA 微调，在同一 test 上评估。
   - 与"探针适配"在**同等目标域标注**下正面对比，并各自记录**适配代价**（GPU 时间、可训练参数量、显存、wall-clock）。
   - 需新写脚本（建议 `scripts/lora_finetune_reranker.py`）：peft + LoRA on XLM-RoBERTa cross-encoder，pairwise/pointwise loss。需装 `peft`。GPU 微调，后台跑。
2. **性价比曲线（核心论文图）**：固定算力预算，横轴=目标域标注量 N（如 50/100/200/全部），纵轴=test AUC/NDCG，两条线（探针适配 vs reranker LoRA）。看探针在小 N / 低算力区是否占优。
3. **扩到第 3 个领域**（NFCorpus 医学 或 TREC-COVID）确认普适：`python scripts/build_ood.py --dataset BeIR/nfcorpus --out results/cache/nfcorpus_ood` 后跑 zeroshot/compare_ood/strong_reranker。
4. **端到端代价对比表**：探针（重训 LR，CPU 分钟级）vs reranker LoRA（GPU 小时级）的部署/适配成本量化。

### 预期实验结果（写论文前的假设，需实验证实）

- LoRA 微调后的 reranker **大概率会反超**探针适配的绝对精度（它参数多、表达力强）。**因此 B 的论点必须落在"性价比/帕累托前沿"**：探针以远低的代价达到"足够接近"的精度，在小标注 / 低算力场景占优。
- 若 LoRA reranker 在**很少标注**下就大幅超过探针 → B 论点受损，需退守到更窄场景（极低算力 / 无 GPU 部署）或重新评估 B 是否成立。
- 适配增益的普适性应在第 3 个领域复现。

### 关键风险

- **B 的命门**：一旦给 reranker 公平的适配机会，"探针更好"可能不成立。**必须诚实面对**——B 的价值是"性价比"而非"绝对精度"，实验设计要紧扣算力/标注代价的量化，否则论文站不住。

---

## 七、环境（新容器必看，会复现的坑）

| 项 | 值 |
|---|---|
| 工作区目录 | `/root/shared-nvme/paper_B_adaptation`（分支 paper-B） |
| GPU | RTX 3090 24GB | Python 3.12 | torch 2.7 | nnsight 0.7.0 | transformers 5.x | datasets 3.6.0 | sentence-transformers 5.6.0 |

**会复现的坑：**
1. **新容器依赖需重装**：仅 torch/sklearn/numpy/matplotlib 预装，需装 transformers/datasets/sentence-transformers/nnsight/accelerate/seaborn（LoRA 还需 `peft`）。
2. **dill pip 约束**：`/etc/pip/constraint.txt` 钉死 dill 0.3.9，直接装 datasets 会静默退化到 1.1.1。解法：`PIP_CONSTRAINT="" pip install 'datasets>=3.0,<4'`。
3. **hf_xet 与代理不兼容**：下载大文件卡死。解法：`pip uninstall hf_xet` + `export HF_HUB_DISABLE_XET=1`。
4. **跑 reranker / 下数据集需联网代理**：
   ```bash
   export https_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
   export http_proxy="$https_proxy"
   export no_proxy="127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.paracloud.com,*.paratera.com,*.blsc.cn"
   export HF_HUB_DISABLE_XET=1
   ```
5. **缓存是软链**：`results/cache` → 主仓库真实目录。删 worktree 前勿 `rm -rf` 跟随软链误删主缓存。

**常用命令：**
```bash
cd /root/shared-nvme/paper_B_adaptation
# 复现两域已有对比：
python scripts/zeroshot_transfer.py --src results/cache/q500_instruct --tgt results/cache/fiqa_ood --topk 20 --out results/zeroshot_fiqa.json
python scripts/compare_ood.py --cache results/cache/fiqa_ood --topk 20 --out results/ood_fiqa.json
python scripts/strong_reranker.py --which cache --cache results/cache/fiqa_ood --out results/strong_reranker_fiqa.json
# 扩新域：
python scripts/build_ood.py --dataset BeIR/nfcorpus --out results/cache/nfcorpus_ood --max-queries 300
```
