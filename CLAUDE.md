# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> 项目：Attention-Probe-RAG | 语言：回复与文档优先用中文 | 状态：进行中（Phase 1 可行性验证）

---

## 一、会话规则（每次新会话必读）

### 1.1 会话压缩后恢复流程（强制）

每次会话被压缩后，**立即执行**，不得跳过：
1. 重新读入本文件 `CLAUDE.md`（全文）
2. 读入 `WORK_STATUS.md`（实时工作状态）
3. 根据 `WORK_STATUS.md` 的「当前任务/下一步」恢复上下文，继续未完成的工作

### 1.2 会话压缩前保存状态（强制）

在上下文接近上限或预判将被压缩前，**主动更新 `WORK_STATUS.md`**：当前任务及进度、下一步具体操作（含命令/路径）、重要中间结论与踩坑、环境状态。每次写入时把上一次的内容提炼精简并标注「第几次压缩」，避免新旧信息混淆。

### 1.3 禁止胡编乱造

所有文献信息、技术结论必须可溯源。不确定的信息标注「未确认」或「推断」。知识库 PDF 中 EvidITI 仅据标题推断（未获全文）；CrAM、ADR 来自外部检索而非知识库。

---

## 二、项目目标与思路

**研究方向：** RAG 检索阶段优化 —— 基于注意力头激活的文档相关性探针。

**核心假设（最关键，待 M2 验证）：** LLM 注意力头激活能编码「检索片段是否与 query 相关」的信号。若 M2 探针准确率不显著高于随机基线，核心假设不成立，需换方向（如 MLP 层激活 / 残差流）。

**方法流程：**
```
检索片段 → (query, passage) 拼接 → LLM 前向推理(仅 forward，不 generate)
        → 提取注意力头激活 → 探针二分类 → 保留相关片段
```

**方法论来源：** ITI（Inference-Time Intervention, Li et al. 2023, NeurIPS Spotlight）。借鉴其「提取激活 → per-head 探针 → 选头」范式，但做了三处迁移：目标 truthfulness→**文档相关性**；应用 干预生成→**检索后筛选**；新增 **ensemble 分类阶段**（产出统一相关性分数，ITI 无此步）。灵感论文 EvidITI 将 ITI 用于 RAG 干预。

**差异化（论文创新点）：** 语义同源（探针与生成器共享表示空间，无外部模型语义鸿沟）、数据效率高（数百条标注即可训逻辑回归）、零额外模型部署、非侵入式（只读激活不改权重）。核心优势论证点是这四点，**不是计算量优势**。

**期望产出：** 一篇学术论文 + 开源代码仓库。预期结果：探针分类指标（accuracy/F1/ROC-AUC）显著高于随机，且端到端 Recall@k 不劣于（理想情况优于）cross-encoder baseline（`ms-marco-MiniLM-L-6-v2`），同时论证同源/数据效率优势。

---

## 三、常用命令

```bash
# 环境安装 + 验证（torch / nnsight / transformers / sklearn / datasets 版本与 CUDA）
bash scripts/setup_env.sh
huggingface-cli login            # LLaMA 为 gated 模型，需先在 HF 接受 license

# M1+M2 主验证实验（本地直接跑）
python initial_validation.py --model meta-llama/Llama-3.2-3B --n-queries 50 --scheme both

# 服务器运行（封装了代理 + HF_HUB_DISABLE_XET=1，下载大文件必须）
bash run_experiment.sh
```

主入口参数：`--n-queries`、`--scheme {last_token,pooling,both}`、`--top-k`（ensemble 选头数，可多值如 `10 20 50 100`）、`--dtype {float16,bfloat16,float32}`、`--max-length`、`--output-dir`、`--seed`。

无独立测试套件；`initial_validation.py` 内置 `sanity_check()` 作为 nnsight 联调自检（验证激活 shape）。结果输出到 `results/initial_validation/`：`validation.log`、`config.json`、`per_head_results_*.csv`、`*.png`。

**网络代理（下载 HF/GitHub/PyPI 慢或超时时按需开启）：**
```bash
export https_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
export http_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
export no_proxy="127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.paracloud.com,*.paratera.com,*.blsc.cn"
# HF 新版默认 xet 协议下载与代理不兼容，大文件下载必须：
export HF_HUB_DISABLE_XET=1
```

---

## 四、代码架构

数据从 `initial_validation.py` 串起四个 `src/` 模块，理解大局需看它如何编排（这是单文件读不出的「big picture」）：

```
initial_validation.py   # 编排者：load_model → sanity_check → 数据 → 按 scheme 循环{提激活 → 训探针 → 评估} → 方案对比
src/data.py             # load_ms_marco(): MS MARCO v1.1 → (query,passage,label) 三元组
                        # split_by_query(): 按 query_id 切 70/15/15，同一 query 不跨 split（防泄露）
src/activations.py      # extract_activations_last_token() / _pooling()：用 nnsight model.trace() 逐层取
                        #   o_proj.output，reshape [B,S,n_heads,head_dim]。返回 [N,n_layers,n_heads,head_dim]
                        # reshape_for_probes(): 压成 [N, n_total_heads, head_dim] 供探针用
src/probes.py           # 两阶段探针（核心）：
                        #   Stage 1 train_per_head_probes(): 每个头独立 L1 逻辑回归
                        #   Stage 2 train_ensemble_probe(): top-k 头激活拼接 → L2 逻辑回归 → 最终分类
src/evaluation.py       # 可视化：per-head ROC-AUC 热力图、top-k 曲线、方案 A/B 对比
```

**关键约束 / 易错点：**
- **激活提取依赖 nnsight 的 forward-pass 访问顺序**：必须按 `layers[0..L]` 顺序在 `model.trace()` 内访问 `o_proj.output`，否则 nnsight 报错。
- **两阶段探针的指标流向**：Stage 1 per-head 探针在 train 上训练，在 **val** 上评估并据此排序选 top-k 头（避免用 train 指标选头造成乐观偏差）；Stage 2 在 train 上训 ensemble，在 val/test 上评估。改动选头逻辑时注意 train/val 指标不要混用。
- **类别不均衡**：MS MARCO 正样本天然偏少，实际接近 1:4，靠 `class_weight="balanced"` + 以 ROC-AUC（非 accuracy）为主指标处理。
- **prompt 模板**：`"Q: {query}\nP: {passage}"`（见 `src/activations.py:_build_prompt`）。
- 探针训练在 CPU 即可（逻辑回归），仅激活提取需 GPU。

**框架选型：** 主用 nnsight 0.7.0（`model.trace()` + `.save()`，见 `nnsight/CLAUDE.md` 使用指南）；fallback 为 transformers `output_hidden_states`（仅 hidden states，拿不到 head 级激活）；保底参考 ITI 原仓库 honest_llama（仅适配 LLaMA-1，需改写）。

---

## 五、模型与数据集

| 阶段 | 模型 | 配置 | VRAM | 用途 |
|------|------|------|------|------|
| Phase 1 | LLaMA-3.2-3B | 28 层 / 24 头 / head_dim 128 / 共 672 头 | ~6GB | 可行性验证 + 方案 A/B 对比 |
| Phase 2 | LLaMA-3.1-8B | 32 层 / 32 头 / head_dim 128 / 共 1024 头 | ~16GB | 主实验 |
| Phase 2 | Mistral-7B-v0.3 | 32 层 / 32 头 / 共 1024 头 | ~15GB | 跨架构泛化 |

实验环境：本机单卡 RTX 3090 24GB。数据集：MS MARCO v1.1（`microsoft/ms_marco`，`is_selected` 1/0 为天然正负标注）。Phase 1 = 500 queries（冒烟用 50），Phase 2 = 2000 queries，按 query 维度 70/15/15 切分。

---

## 六、里程碑

| M | 内容 | 验证标准 |
|---|------|---------|
| M1 | 环境 + 数据准备 | nnsight 与 LLaMA-3.2-3B 联调通过（打印激活 shape） |
| **M2** | 激活提取 + 探针训练 | **探针准确率/ROC-AUC 显著 > 随机（最关键验证点）** |
| M3 | 方案 A（末位 token）vs B（池化）对比 | 确定最终激活提取方案 |
| M4 | 端到端 RAG + cross-encoder 对比 | Recall@k 对比表（需补 `rerank_with_probe` 端到端代码） |
| M5 | 扩展 8B + 跨模型 | 泛化性验证 |

---

## 七、参考文档（`docs/`）

详细内容已归档，需要时查阅，**不必每次通读**：
- `docs/实现计划-Attention-Probe-RAG.md` —— 完整伪代码 + 实验设计（顶部有 2026-06-27 校正说明，标注了与现行代码的三处差异）
- `docs/RAG检索优化文献调研报告.md` —— 17 篇论文逐篇分析 + 数据集/模型调研
- `docs/讨论总结-研究思路梳理.md` —— 最初的研究思路与优势论述
- `nnsight/CLAUDE.md` —— nnsight 框架使用指南

> ⚠️ **文档与代码冲突时以代码为源真相**。已知三处（详见实现计划顶部校正说明 + `WORK_STATUS.md`）：prompt 模板、数据正负比、Stage 1 选头指标（ROC-AUC 为现行，accuracy 方案待补充以做对比）。发现新冲突时向用户提出由其定夺。
