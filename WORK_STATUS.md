# Attention-Probe-RAG 工作状态

> 本文件记录项目的实时工作状态，用于跨会话恢复上下文。
> 每次会话压缩后必须读入本文件（流程见 CLAUDE.md 第一节）。

---

## 最后更新

- 日期：2026-06-27
- 第几次压缩：第 5 次
- 会话摘要（精炼）：完成「方法不泛化」的全套修复 + 跨方法对比 + 跨域实验。核心剧情：
  (1) 同域 MS MARCO 探针**输给** cross-encoder（0.685 < 0.759），raw 同域精度路堵死；
  (2) 数据效率曲线单调上升无平台 → 数据效率是**弱论点**；
  (3) 跨域 SciFact 探针**反超**（0.964 > 0.858 > BM25 0.669）→ 论点重定位为「**跨域鲁棒性**」；
  (4) 关键发现：prompt 框架（instruction + answer-token）比探针架构更重要（0.634→0.685）；
  (5) EvidITI 已核实为本人前作（fengchenhui@nuaa），引用合规。
- ⚠️ **复查发现的公平性漏洞**（本次会话）：跨域实验里探针在 SciFact 上重训过、而 cross-encoder 零适配 → 0.964 混淆了「迁移能力强」与「多看了目标域标注」。审稿人必揪。→ 下一步用「零样本迁移」补干净对比（见下）。
- 历史摘要（前 4 次压缩，已提炼）：环境全部就绪（Python 3.12.3, torch 2.7.0, CUDA 12.8, nnsight 0.7.0, transformers 5.x）。LLaMA-3.2-3B license 已审批、模型已缓存（xet 卸载、HF_HUB_DISABLE_XET=1 后 classic-HTTP 下载成功）。执行过一次 /init 项目整理（归档 docs/、重写 README/CLAUDE、核对三处冲突）。`torch_dtype`→`dtype` 已修复。

---

## 环境信息

| 项目 | 值 |
|------|-----|
| 项目目录 | /root/shared-nvme/my_paper_project |
| Git 仓库 | https://github.com/new-life-fch/my_paper_project.git |
| GPU | RTX 3090 24GB |
| Python | 3.12.3 |
| PyTorch | 2.7.0a0 (NVIDIA 构建版, CUDA 可用) |
| Transformers | 5.3.0 |
| datasets | 3.6.0（必须 3.x：1.x 缺 IterableDataset、4.x 移除 trust_remote_code）|
| nnsight | 0.7.0 |
| sentence-transformers | 5.6.0 |
| seaborn | 0.13.2 |

> 2026-06-27 实测：当前容器为**全新环境**，仅 torch/sklearn/numpy/matplotlib 预装。
> transformers/datasets/sentence-transformers/nnsight/accelerate/seaborn 均为本次新装。
> （故 WORK_STATUS 历史「依赖已就绪」对新容器不成立，每次新容器都需重装。）

---

## ⚠️ 环境踩坑（会在新容器复现，重要）

**系统级 pip 约束把 dill 钉死在 0.3.9 → 直接 `pip install datasets` 会装成远古的 1.1.1。**

- 容器有 `PIP_CONSTRAINT=/etc/pip/constraint.txt`，其中 `dill==0.3.9`。
- 现代 `datasets`（2.x/3.x）要求 `dill<0.3.9`，pip 解析器因此**静默回退**到 2020 年的
  `datasets 1.1.1`（仅 147KB），它缺 `IterableDataset`，连带 `sentence-transformers` 都无法 import，且无法加载 MS MARCO。
- **解法**：装 datasets 时临时清空约束 —
  ```bash
  PIP_CONSTRAINT="" pip install 'datasets>=3.0,<4'
  ```
  这会让 dill 降到 0.3.8、multiprocess 降到 0.70.16。仅 lightning-thunder（PyTorch 编译工具，
  本项目不用）受影响，对本项目无副作用。
- 安装其它包用代理即可（见下）：
  ```bash
  export https_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
  export http_proxy="$https_proxy"
  pip install transformers sentence-transformers nnsight accelerate seaborn
  ```

---

## 整理后项目结构

```
my_paper_project/
├── CLAUDE.md                # 开发/协作完整上下文（必读）
├── WORK_STATUS.md           # 本文件：实时工作状态
├── README.md                # 项目说明（精炼）
├── requirements.txt
├── initial_validation.py    # M1+M2 主入口
├── run_experiment.sh        # 服务器运行脚本（含代理 + HF_HUB_DISABLE_XET）
├── src/                     # data / activations / probes / evaluation
├── scripts/setup_env.sh     # 环境安装与验证
├── docs/                    # 三份详细源文档（归档参考）
│   ├── 文献调研报告.md（17 篇论文分析）
│   ├── 实现计划-Attention-Probe-RAG.md（完整伪代码，含校正说明）
│   └── 讨论总结-研究思路梳理.md
├── PDF论文知识库/            # 参考论文 PDF
└── nnsight/                 # nnsight 框架参考
```

---

## 实现计划 vs 代码 — 三处冲突的决策（2026-06-27）

整理时核对 `docs/实现计划` 与 `src/` 代码，确认三处差异并定夺（详见实现计划顶部校正说明）：

1. **Prompt 模板**：以**代码**为准 → `"Q: {query}\nP: {passage}"`（实现计划的 `"Question:/Passage:"` 已过时）。
2. **数据规模/正负比**：以**代码**为准 → `src/data.py` 每 query 上限 5 passages，因 MS MARCO v1.1 正样本有限，实际产出仍接近 1:4 不均衡，由 `class_weight="balanced"` + ROC-AUC 主指标处理。规模目标（Phase 1=500、Phase 2=2000 queries）不变。
3. **Stage 1 选头指标**：**两种都保留以供对比**。代码当前用验证集 **ROC-AUC** 选头；实现计划用验证集 **accuracy**（`clf.score`）。→ **待补充实现**：增加 accuracy 选头方案，作为消融对比项。

---

## 方法论关键决策（稳定）

- **两阶段探针**：Stage 1 per-head L1 逻辑回归选 top-k 头（参考 ITI，L1 为项目选择）；Stage 2 top-k 头激活拼接 → L2 逻辑回归做最终分类（ITI 无此步，本项目筛选需统一相关性分数）。
- **激活提取**：Scheme A（last_token）`o_proj.output` 末位 token；Scheme B（pooling）passage token 范围 mean pooling。用 nnsight `model.trace()` 逐层按 forward order 提取。
- **模型**：Phase 1 = LLaMA-3.2-3B（28 层，24 头/层，head_dim=128，共 672 头）。
- **数据集**：MS MARCO v1.1，按 query 维度 70/15/15 切分（防泄露）。

---

## 当前任务 / 下一步

### 📊 已建成的实验资产（缓存 + 脚本 + 结果）

**磁盘缓存**（`results/cache/`，float16 .pt，提取一次秒级迭代探针）：
- `q500/`：原始 prompt `"Q:/P:"`，scheme A/B，500 queries。
- `q500_instruct/`：instruct prompt + answer-token，**attn + resid 双位点**，672 头。← 同域主用
- `q500_judge/`：LLM-judge 的 P(yes)/P(no) logits。
- `scifact_ood/`：BeIR/SciFact，BM25 top-20 候选池，3000/1200/1800 triples，instruct-attn。

**脚本**（`scripts/`）：
- `extract_and_cache.py` / `extract_instruct.py`：抽激活缓存（后者 instruct + 双位点）。
- `compare_probes.py`：6 种探针方法论对比（baseline/scaler/CV选头/whole-L2/stacking/mean）。
- `llm_judge.py` + `compare_methods.py`：probe vs llm_judge vs cross_encoder vs bm25（同域，per-query MRR/NDCG/Recall）。
- `data_efficiency.py`：探针 test AUC vs #train-queries 曲线。
- `build_ood_scifact.py` + `compare_ood.py`：跨域 SciFact 实验。

**关键结果**（已提交）：
| 实验 | probe | cross_encoder | bm25 | llm_judge | 结论 |
|------|-------|---------------|------|-----------|------|
| 同域 MS MARCO (AUC) | 0.685 | **0.759** | 0.543 | 0.535 | 探针**输**（对手主场）|
| 跨域 SciFact (AUC) | **0.964** | 0.858 | 0.669 | — | 探针**反超**（但有漏洞，见下）|

### 🎯 论点重定位（稳定）

四点优势里：raw 同域精度（输）、数据效率（弱，曲线无平台）已被排除；
**唯一硬贡献 = 跨域鲁棒性**：通用 LLM 上的线性探针 vs MS-MARCO 专用 reranker，换领域时探针更稳。
工程属性（零部署/非侵入）作辅助卖点，不作主实验。

### ⏭️ 下一步：补「零样本迁移」干净对比（进行中）

**动机**：跨域 0.964 里探针在 SciFact 重训过、cross-encoder 没有 → 不公平，混淆「迁移」与「适配」。
**做法**：新脚本 `scripts/zeroshot_transfer.py` —
- 探针：在 **MS MARCO**（q500_instruct）上选头 + 训 ensemble，**零适配**直接打分 SciFact test。
- cross-encoder：同样零适配（已有 0.858）。
- 若「MS MARCO 训的探针零样本套 SciFact 仍 ≥ cross-encoder」→ 公平性无懈可击，这才是能写进论文的主结果。
- 特征对齐：两缓存同模型/同 prompt/同 672 头，训一边测一边即可。
- ⚠️ 注意 train/test 来自不同缓存，需重新 fit；选头仍只用 MS MARCO 的 val（不碰 SciFact 任何标注）。

**之后**（按优先级）：
1. 查 SciFact 候选池构造偏差（正例=真摘要 vs 负例=BM25召回，会不会学到表面文本特征）。
2. 第 2 个领域复现（FiQA / NFCorpus），确认不是 SciFact 巧合。
3. 论文初稿。

### 历史命令（重跑实验）
```bash
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1   # 模型已缓存；跑 cross-encoder 需临时联网代理
cd /root/shared-nvme/my_paper_project
python scripts/compare_ood.py --cache results/cache/scifact_ood --topk 20
```

---

## 待办事项

- [ ] 重跑 initial_validation.py 完成 M1+M2（带 `HF_HUB_DISABLE_XET=1`）
- [ ] **新增 accuracy 选头方案**（实现计划冲突 3 决策：与 ROC-AUC 选头做对比消融）
- [ ] `initial_validation.py:88` `load_model` 仍传 `torch_dtype=`，transformers 5.x 已 deprecated，改用 `dtype=`
- [ ] M3 方案 A vs B 对比（随 M2 产出）
- [ ] M4 端到端 RAG + cross-encoder baseline（需补 `rerank_with_probe` 端到端代码）
- [ ] （安全，暂缓）`run_experiment.sh` 硬编码代理凭证已提交公开仓库，建议改为环境变量/.env 并轮换密码

---

## 已完成里程碑

| 日期 | 里程碑 | 状态 |
|------|--------|------|
| 2026-06-03 | 服务器连接 + git 初始化 + 代码编写上传 | ✅ |
| 2026-06-04 | ITI 方法论提取 + CLAUDE.md 校正 + Bug 修复 + 环境搭建 | ✅ |
| 2026-06-05 | LLaMA license 审批 + 依赖安装 + 文件同步 | ✅ |
| 2026-06-27 | 项目整理：归档 docs/、清理冗余、重写 README/CLAUDE、核对冲突 | ✅ |
| 2026-06-27 | 磁盘缓存基础设施 + 6 种探针方法论对比（修复不泛化） | ✅ |
| 2026-06-27 | instruct-prompt + 双位点抽取（prompt 框架是关键增益） | ✅ |
| 2026-06-27 | 跨方法对比 + LLM-judge baseline（同域探针输 reranker） | ✅ |
| 2026-06-27 | 数据效率曲线（弱论点）+ 跨域 SciFact（探针反超，待补公平对比） | ✅ |
| 2026-06-27 | EvidITI 核实为本人前作，引用合规 | ✅ |
