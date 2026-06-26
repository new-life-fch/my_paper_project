# Attention-Probe-RAG

利用 LLM 自身的注意力头激活训练二元探针，在 RAG 检索阶段自主精筛检索片段的相关性。

## 研究思路

RAG 系统检索阶段常返回混杂噪声的文档片段。传统重排序方法依赖外部 cross-encoder 或排序模型，与生成 LLM 的表示空间存在语义鸿沟。本项目改为**利用 LLM 自身的内部表示**判断文档相关性：

```
检索片段 → (query, passage) 拼接 → LLM 前向推理 → 提取注意力头激活 → 探针二分类 → 保留相关片段
```

方法论改编自 ITI（Inference-Time Intervention, Li et al. 2023），但将探针目标从「真实性」迁移到「文档相关性」，将应用从「干预生成」改为「检索后筛选」，并增加 ensemble 分类阶段产出统一的相关性分数。

**核心优势：** 语义同源（探针与生成器共享表示空间）、数据效率高（数百条标注即可训练逻辑回归探针）、零额外模型部署（仅需一次前向推理）、非侵入式（只读激活，不改权重）。

## 环境配置

需 Python 3.10+ 与单卡 GPU（Phase 1 用 LLaMA-3.2-3B，约 6GB VRAM）。

```bash
bash scripts/setup_env.sh        # 安装依赖 + 验证 torch/nnsight/transformers
huggingface-cli login            # LLaMA 为 gated 模型，需先在 HF 接受 license
```

关键依赖：`nnsight`（激活提取主框架）、`transformers`、`torch`、`scikit-learn`、`sentence-transformers`、`datasets`。

## 运行实验

```bash
# Milestone 1 + 2：环境验证 + 核心假设验证（探针准确率是否显著高于随机）
python initial_validation.py --model meta-llama/Llama-3.2-3B --n-queries 50 --scheme both

# 服务器环境（含代理 + 禁用 xet 下载协议）
bash run_experiment.sh
```

主要参数：`--n-queries`（query 数）、`--scheme`（`last_token` / `pooling` / `both`）、`--top-k`（ensemble 选头数）、`--dtype`。结果输出到 `results/initial_validation/`（log、config、per-head CSV、可视化 PNG）。

## 项目结构

```
initial_validation.py   # M1+M2 主入口：加载模型→sanity check→提数据→提激活→训探针→评估→可视化
src/
  data.py               # MS MARCO 加载与按 query 维度切分（防数据泄露）
  activations.py        # nnsight 激活提取（Scheme A: 末位token / Scheme B: 平均池化）
  probes.py             # 两阶段探针：Stage 1 per-head L1 选头 + Stage 2 ensemble L2 分类
  evaluation.py         # 可视化（热力图、top-k 曲线、方案对比）
scripts/setup_env.sh    # 环境安装与验证
docs/                   # 详细研究文档（文献调研、实现计划、讨论总结）
PDF论文知识库/           # 参考论文 PDF
nnsight/                # nnsight 框架参考（含 CLAUDE.md 使用指南）
```

`CLAUDE.md` 是面向开发者/AI 协作者的完整上下文文档（目标、方法论决策、模型/数据集方案、命令、架构）。`WORK_STATUS.md` 记录实时工作进度。

## 实验里程碑

| Milestone | 内容 | 验证标准 |
|-----------|------|---------|
| M1 | 环境搭建 + 数据准备 | nnsight 与 LLaMA-3.2-3B 联调通过 |
| M2 | 激活提取 + 探针训练 | **探针准确率显著 > 随机基线（最关键验证点）** |
| M3 | 方案 A vs B 对比 | 确定最终激活提取方案 |
| M4 | 端到端 RAG + cross-encoder 对比 | Recall@k 对比表 |
| M5 | 扩展到 8B + 跨模型验证 | 泛化性验证 |

## License

学术研究用途。
