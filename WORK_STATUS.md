# Attention-Probe-RAG 工作状态

> 本文件记录项目的实时工作状态，用于跨会话恢复上下文。
> 每次会话压缩后必须读入本文件。

---

## 最后更新

- 日期：2026-06-04
- 会话摘要：完成代码编写、方法论校正（ITI论文对照）、bug修复、服务器环境检查

---

## 服务器信息

| 项目 | 值 |
|------|-----|
| 主机 | ssh.zw1.paratera.com:2222 |
| 用户名 | root@ackcs-00gjh2st |
| SSH别名 | paratera-01（已注册在ssh-skill中） |
| 项目目录 | /root/shared-nvme/my_paper_project |
| Git仓库 | https://github.com/new-life-fch/my_paper_project.git |
| GPU | 待确认（首次连接时未检查） |

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

### 正在进行：服务器环境搭建 + 运行实验

**下一步操作：**
1. 在服务器上运行 `bash scripts/setup_env.sh` 安装Python依赖
2. 运行 `huggingface-cli login` 登录（LLaMA-3.2是gated model，需要先接受license）
3. 检查GPU状态：`nvidia-smi`
4. 运行初期验证：`python initial_validation.py --n-queries 50 --scheme both`
5. 观察输出，确认核心假设是否成立

### 潜在阻塞点
- HuggingFace登录需要用户手动提供token（如果服务器上未登录）
- LLaMA-3.2-3B需要在HuggingFace上先接受模型license
- 网络问题可能导致模型下载失败

---

## 已完成里程碑

| 日期 | 里程碑 | 状态 |
|------|--------|------|
| 2026-06-03 | 服务器连接 + git初始化 + README | ✅ 完成 |
| 2026-06-03 | 代码编写 + 上传 + git推送 | ✅ 完成 |
| 2026-06-04 | ITI论文方法论提取 + CLAUDE.md方法论校正 | ✅ 完成 |
| 2026-06-04 | Bug修复（F1死代码 + val/train排序不一致） | ✅ 完成 |
| 2026-06-04 | CLAUDE.md添加会话压缩恢复规则 + 工作状态文件 | ✅ 完成 |

---

## 待完成里程碑

| 里程碑 | 预计 | 备注 |
|--------|------|------|
| M1 环境搭建 | 进行中 | 安装依赖、登录HF、下载模型 |
| M2 核心假设验证 | 待定 | 运行initial_validation.py，探针准确率>50% |
| M3 方案A vs B对比 | 随M2一起 | 确定最终提取方案 |
| M4 端到端RAG + baseline | 待定 | 需要额外编写rerank代码 |
