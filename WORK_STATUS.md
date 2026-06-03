# Attention-Probe-RAG 工作状态

> 本文件记录项目的实时工作状态，用于跨会话恢复上下文。
> 每次会话压缩后必须读入本文件。

---

## 最后更新

- 日期：2026-06-04
- 第几次压缩：第1次
- 会话摘要：旧容器torch版本过低(2.2.0)，用户正在创建新容器。网络代理配置已记录到CLAUDE.md。

---

## 服务器信息

| 项目 | 值 |
|------|-----|
| 主机 | ssh.zw1.paratera.com:2222 |
| 旧容器用户名 | root@ackcs-00gjh2st（已弃用，torch版本过低） |
| 新容器用户名 | **待用户提供** |
| SSH别名 | paratera-01（已注册在ssh-skill中，新容器需更新） |
| 项目目录 | /root/shared-nvme/my_paper_project |
| Git仓库 | https://github.com/new-life-fch/my_paper_project.git |
| GitHub Token | REDACTED |
| GPU | RTX 3090 24GB（旧容器确认） |

### 旧容器踩坑记录
- torch 2.2.0a0+81ea7a4 (CUDA 12.3) 太旧，nnsight要求>=2.4，transformers 5.9也要求>=2.4
- 尝试安装torch 2.5.1 cu124 但因包太大(~2GB) SSH连接反复超时
- nnsight 0.7.0 可以用 `pip install nnsight --no-deps` 装上，但依赖的torch版本不匹配
- 结论：需要新容器自带较新的torch

---

## 代码状态

### 本地代码位置
`D:\solo-paper\attention-probe-rag`

### 远程 Git 提交记录
```
bcaec06 fix: correct F1 dead code and val/train metric inconsistency
609afb5 feat: add initial validation pipeline (M1+M2)
63c6455 Initial commit: add README.md
```

### 项目文件结构
```
attention-probe-rag/
├── initial_validation.py      # 主验证入口（M1+M2完整流程）
├── requirements.txt           # 依赖列表
├── scripts/setup_env.sh       # 环境安装脚本
├── src/
│   ├── __init__.py
│   ├── activations.py         # nnsight激活提取（Scheme A/B）
│   ├── data.py                # MS MARCO数据加载与切分
│   ├── evaluation.py          # 可视化（热力图、top-k、方案对比）
│   └── probes.py              # 两阶段探针训练（per-head L1 + ensemble L2）
└── results/                   # 实验输出（git忽略）
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

### 阻塞中：等待用户创建新容器

**用户正在做的事情：**
- 创建一个新容器，选一个自带较新torch的镜像（需要torch>=2.4 + CUDA兼容）

**新容器就绪后，我需要执行的操作（按顺序）：**

1. 获取新容器的SSH用户名和密码
2. 注册新SSH别名（更新ssh-skill配置）
3. 配置git凭证：
   ```bash
   git config --global credential.helper store
   # 使用token: REDACTED
   ```
4. 克隆代码：
   ```bash
   cd /root/shared-nvme
   git clone https://new-life-fch:REDACTED@github.com/new-life-fch/my_paper_project.git
   ```
5. 检查环境：`python --version && nvidia-smi && python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
6. 如遇网络问题，先开启代理（见CLAUDE.md 1.4节）
7. 安装依赖：`cd my_paper_project && bash scripts/setup_env.sh && pip install nnsight`
8. HuggingFace登录：`huggingface-cli login`（LLaMA-3.2-3B是gated model）
9. 运行实验：`python initial_validation.py --n-queries 50 --scheme both`

### 潜在阻塞点
- 新容器的torch版本是否>=2.4
- HuggingFace登录：用户可能需要在HF网站先接受LLaMA-3.2的license
- 网络问题：用代理解决（配置已记录在CLAUDE.md 1.4节）
- nnsight与torch版本兼容性

---

## 已完成里程碑

| 日期 | 里程碑 | 状态 |
|------|--------|------|
| 2026-06-03 | 服务器连接 + git初始化 + README | ✅ 完成 |
| 2026-06-03 | 代码编写 + 上传 + git推送 | ✅ 完成 |
| 2026-06-04 | ITI论文方法论提取 + CLAUDE.md方法论校正 | ✅ 完成 |
| 2026-06-04 | Bug修复（F1死代码 + val/train排序不一致） | ✅ 完成 |
| 2026-06-04 | CLAUDE.md会话压缩规则 + 网络配置 + 工作状态文件 | ✅ 完成 |

---

## 待完成里程碑

| 里程碑 | 预计 | 备注 |
|--------|------|------|
| M1 环境搭建 | 等新容器 | 新容器需自带torch>=2.4 + CUDA |
| M2 核心假设验证 | 待定 | 运行initial_validation.py，探针准确率>50% |
| M3 方案A vs B对比 | 随M2一起 | 确定最终提取方案 |
| M4 端到端RAG + baseline | 待定 | 需要额外编写rerank代码 |
