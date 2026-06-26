# Attention-Probe-RAG 工作状态

> 本文件记录项目的实时工作状态，用于跨会话恢复上下文。
> 每次会话压缩后必须读入本文件。

---

## 最后更新

- 日期：2026-06-05
- 第几次压缩：第3次
- 会话摘要（已修正）：环境全部就绪（Python 3.12.3, torch 2.7.0, CUDA 12.8, nnsight 0.7.0）。LLaMA-3.2-3B shard-1 已下载(4.96GB)，shard-2 下载中断(1.24GB .incomplete)，shard-1的snapshot symlink 不完整需重命名。requirements.txt 版本约束已放宽。Qoder CLI 安装到服务器。

---

## 环境信息

| 项目 | 值 |
|------|-----|
| 项目目录 | /root/shared-nvme/my_paper_project |
| Git仓库 | https://github.com/new-life-fch/my_paper_project.git |
| GPU | RTX 3090 24GB |
| Python | 3.12.3 |
| PyTorch | 2.7.0 CUDA 12.8 |
| Transformers | 5.10.1 |
| nnsight | 0.7.0 |

---

## 代码状态

### Git 提交记录
```
bcaec06 fix: correct F1 dead code and val/train metric inconsistency
609afb5 feat: add initial validation pipeline (M1+M2)
63c6455 Initial commit: add README.md
```

### 项目文件结构
```
/root/shared-nvme/my_paper_project/
├── CLAUDE.md                          # 项目上下文（必读）
├── WORK_STATUS.md                     # 本文件
├── nnsight/                           # nnsight 框架参考
├── PDF论文知识库/                       # 论文 PDF
├── RAG检索优化文献调研报告.md
├── 实现计划-Attention-Probe-RAG.md
├── 讨论总结-研究思路梳理.md
├── initial_validation.py              # 主验证入口（M1+M2）
├── requirements.txt                   # 依赖列表
├── scripts/
│   ├── setup_env.sh
│   └── run_experiment.sh              # 含代理 + HF_HUB_DISABLE_XET=1
├── src/
│   ├── activations.py                 # nnsight激活提取（Scheme A/B）
│   ├── data.py                        # MS MARCO数据加载与切分
│   ├── evaluation.py                  # 可视化（热力图、top-k、方案对比）
│   └── probes.py                      # 两阶段探针训练（per-head L1 + ensemble L2）
└── results/                           # 实验输出（git忽略）
```

---

## 方法论关键决策

### 两阶段探针训练（已校正，2026-06-04）
- **Stage 1（参考ITI）**：per-head L1逻辑回归 → 验证集排序 → 选top-k heads
  - ITI原文没有L1正则化（用无正则化的线性探针），L1是项目选择
- **Stage 2（项目适配）**：top-k heads激活拼接 → L2逻辑回归 → 最终分类器
  - ITI没有这一步（ITI的探针只用于选头和干预方向，不做独立分类）
  - 本项目需要统一的相关性分数来筛选文档，ensemble是必要的适配

### 激活提取
- Scheme A（last_token）：`o_proj.output[0].view(B,S,n_heads,head_dim)[0,-1,:,:]`
- Scheme B（pooling）：passage token范围内mean pooling
- 使用nnsight `model.trace()` 逐层按forward-pass order提取

### 模型
- Phase 1：LLaMA-3.2-3B（28层，24头/层，head_dim=128，总672头）

### 数据集
- MS MARCO v1.1，Phase 1用50 queries，按query维度70/15/15切分

---

## 当前任务

### 进行中：运行初始验证实验（M1+M2）

**环境状态（全部就绪）：**
- 所有依赖已安装（torch 2.7.0, nnsight 0.7.0, transformers 5.10.1）
- LLaMA-3.2-3B license 已审批通过
- 模型下载中（约6GB）
- HF token 已配置（~/.huggingface/token）

**运行实验命令：**
```bash
export https_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
export http_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
export HF_HUB_DISABLE_XET=1
cd /root/shared-nvme/my_paper_project
python -u initial_validation.py --n-queries 50 --scheme both
```

### 踩坑记录
- requirements.txt 原来有严格版本约束，在 Python 3.12 + torch 2.7.0 下导致 pip ResolutionImpossible，已放宽为无版本约束
- HuggingFace 新版 hub 默认用 xet 协议下载大文件，代理不兼容，需 `HF_HUB_DISABLE_XET=1`
- `torch_dtype` 参数在 transformers 5.x 已 deprecated，改用 `dtype`

---

## 已完成里程碑

| 日期 | 里程碑 | 状态 |
|------|--------|------|
| 2026-06-03 | 服务器连接 + git初始化 + README | ✅ 完成 |
| 2026-06-03 | 代码编写 + 上传 + git推送 | ✅ 完成 |
| 2026-06-04 | ITI论文方法论提取 + CLAUDE.md方法论校正 | ✅ 完成 |
| 2026-06-04 | Bug修复（F1死代码 + val/train排序不一致） | ✅ 完成 |
| 2026-06-04 | 新容器环境搭建 + 依赖安装 | ✅ 完成 |
| 2026-06-05 | LLaMA license 审批 + Qoder CLI 安装 + 文件同步 | ✅ 完成 |

---

## 待完成里程碑

| 里程碑 | 备注 |
|--------|------|
| M1 环境验证 | 模型下载完成后 sanity check |
| M2 核心假设验证 | 运行 initial_validation.py，探针准确率>50% |
| M3 方案A vs B对比 | 随M2一起产出 |
| M4 端到端RAG + baseline | 需额外编写 rerank 代码 |
