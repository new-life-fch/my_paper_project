# Attention-Probe-RAG 工作状态

> 本文件记录项目的实时工作状态，用于跨会话恢复上下文。
> 每次会话压缩后必须读入本文件（流程见 CLAUDE.md 第一节）。

---

## 最后更新

- 日期：2026-06-27
- 第几次压缩：第 4 次（整理为基准）
- 会话摘要（精炼）：执行了一次项目整理 /init。归档三份中文源文档到 `docs/`，删除冗余/垃圾文件（`.project-memory.md` 旧副本、空文件 `0`、`experiment.log`、`.ipynb_checkpoints/`），重写 `README.md` 与 `CLAUDE.md` 为精炼准确版本并指向 `docs/`。核对了实现计划与代码的三处冲突并做出决策（见下）。
- 历史摘要（前 3 次压缩，已提炼）：环境全部就绪（Python 3.12.3, torch 2.7.0, CUDA 12.8, nnsight 0.7.0, transformers 5.10.1）。LLaMA-3.2-3B 为 gated 模型、license 已审批；首次实验运行因模型下载（xet 协议与代理不兼容）被 `Terminated`，需用 `HF_HUB_DISABLE_XET=1` 重跑。requirements.txt 已放宽版本约束。`torch_dtype` 在 transformers 5.x 已 deprecated，应改用 `dtype`（代码中仍传 torch_dtype，见待办）。

---

## 环境信息

| 项目 | 值 |
|------|-----|
| 项目目录 | /root/shared-nvme/my_paper_project |
| Git 仓库 | https://github.com/new-life-fch/my_paper_project.git |
| GPU | RTX 3090 24GB |
| Python | 3.12.3 |
| PyTorch | 2.7.0 (CUDA 12.8) |
| Transformers | 5.10.1 |
| nnsight | 0.7.0 |

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

### 下一步：运行 M1+M2 初始验证实验

模型下载曾因 xet 协议被中断，重跑命令：
```bash
export https_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
export http_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
export HF_HUB_DISABLE_XET=1
cd /root/shared-nvme/my_paper_project
python -u initial_validation.py --n-queries 50 --scheme both
```
验证标准（M2，最关键）：探针准确率 / ROC-AUC 显著高于随机基线。若不显著，核心假设不成立，需换方向（MLP 层激活 / 残差流）。

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
