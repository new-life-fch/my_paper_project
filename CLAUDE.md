# Attention-Probe-RAG 项目 — 跨会话上下文

> 创建日期：2026-05-31 | 最后更新：2026-06-03 | 状态：进行中
> 本文档整合了项目的全局规则、研究思路、文献调研、实现计划，用于跨会话传递上下文。

---

## 一、全局会话规则

> 每个新会话开始时必读本节。

### 1.0 会话压缩后恢复流程（强制）

每次会话被压缩（上下文窗口重置或大幅截断）后，**必须立即执行以下步骤**，不得跳过：

1. 重新读入本文件 `D:\solo-paper\CLAUDE.md`（全文）
2. 读入工作状态文件 `D:\solo-paper\WORK_STATUS.md`
3. 根据工作状态文件中的"当前任务"和"下一步"恢复工作上下文，继续未完成的工作

如果没有工作状态文件，从本文件的"待办事项"（第十四节）中找到最近的待办项继续。

**会话压缩前保存状态（强制）：** 在会话即将被压缩之前（例如上下文接近上限、用户提示要压缩、或你预判到需要压缩时），**必须主动将以下信息写入 `WORK_STATUS.md`**：

1. 当前正在执行的任务及其进度
2. 下一步需要执行的操作（尽可能具体，包含命令或文件路径）
3. 你认为重要的中间结论、发现或注意事项（例如踩过的坑、已确认/未确认的假设）
4. 服务器/环境的当前状态（已安装的依赖、正在运行的进程等）
5. 每次写入信息时，将上一次压缩前写入的信息总结、提炼、精简，并标注第几次压缩，不要使Agent弄混

### 1.1 禁止命令

禁止在本机运行下载pytorch和模型等命令

### 1.2 禁止胡编乱造

所有文献信息、技术结论必须有可溯源依据。不确定的信息必须标注"未确认"或"推断"。

### 1.3 相关文件夹

相关框架nnsight:
`D:\solo-paper\nnsight`，该目录下有CLAUDE.md文件，引导你使用框架

知识库:
`D:\solo-paper\PDF论文知识库`，论文为PDF格式，内容较长，不推荐阅读

项目代码:
`D:\solo-paper\attention-probe-rag`，所有代码编写和修改都在这里进行，临时脚本也写在此目录。禁止在临时工作目录中写项目代码。

远程服务器项目目录:
`/root/shared-nvme/my_paper_project`，代码通过 git 推送到 https://github.com/new-life-fch/my_paper_project.git

### 1.4 服务器网络配置

服务器访问外网（GitHub、HuggingFace、PyPI）时可能需要配置代理加速。

**代理加速设置（按需开启）：**
```bash
export https_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
export http_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
export no_proxy="127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.paracloud.com,*.paratera.com,*.blsc.cn"
```

**取消代理：**
```bash
unset http_proxy https_proxy no_proxy
```

**加速覆盖域名：** .github.com, .githubusercontent.com, .huggingface.co, .pypi.org, .pythonhosted.org 等。

**pip 国内源（备用）：**
```bash
pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
pip config set global.trusted-host mirrors.bfsu.edu.cn
```

**注意：** 遇到网络超时/下载慢时优先尝试开启代理，再考虑换 pip 源。

---

## 二、项目定位

**研究方向：** RAG 检索阶段优化 — 基于注意力头激活的文档相关性探针

**核心思路：** 利用 LLM 自身的注意力头激活训练二元探针，自主判断检索片段的相关性，实现大模型自主的检索片段精筛。

流程：`检索片段` -> `(query, passage) 拼接` -> `LLM 前向推理` -> `提取注意力头激活` -> `探针二分类` -> `保留相关片段`

**目标产出：** 一篇学术论文 + 开源代码仓库

**核心理念：** 利用 LLM 自身的内部表示空间（representation space）来判断检索文档的相关性，而非依赖外部的评分模型或重排序器。通过提取注意力头级别的激活信息，捕捉模型在处理特定文档-查询对时的深层语义响应模式，构建一个轻量级、可解释性强、且与主模型同源的筛选机制。

---

## 三、方案优势与差异化

| 维度 | 优势 |
|------|------|
| **语义同源** | 探针直接在 LLM 自身的表示空间上训练，筛选器的打分标准与生成模型的内在偏好一致，不存在外部模型语义隔阂 |
| **数据效率高** | 仅需数百条标注片段的正反激活对即可训练探针（逻辑回归等线性分类器） |
| **零额外模型部署** | 不需要额外加载交叉编码器或外部排序模型，仅需一次前向推理获取激活值（多个片段可 batch 并行处理） |
| **通用性** | 不依赖特定检索策略，可与任何检索器搭配使用 |
| **非侵入式** | 仅读取注意力头激活值，不修改模型权重或激活值，保持模型完整性 |

**与现有方法的五维差异化：** 方法论（相关性探针 vs ITI的真实性探针）、场景应用（RAG筛选 vs EvidITI的RAG干预）、干预方式（非侵入式 vs CrAM/ADR的侵入式修改）、筛选依据（模型自身表示空间 vs FlashRank的外部模型）、以及筛选范式（筛选文档 vs 修改模型行为）。

---

## 四、已敲定的方法论决策

| 决策项 | 选择 | 敲定日期 |
|--------|------|---------|
| 方法论来源 | ITI (Inference-Time Intervention, Li等 2023) | 2026-05-18 |
| 灵感论文 | EvidITI (Enhancing RAG Robustness through ITI) | 2026-05-18 |
| 激活提取方案 | 方案A (passage末位token) + 方案B (passage范围平均池化) 并行对比 | 2026-05-31 |
| Batch 策略 | Phase 1 逐一推理 -> Phase 2 batch 优化 | 2026-05-31 |
| 对比 Baseline | Cross-encoder reranker (ms-marco-MiniLM-L-6-v2) | 2026-05-31 |
| 探针类型 | 两阶段：Stage 1 per-head L1逻辑回归(选头，参考ITI) + Stage 2 ensemble L2逻辑回归(分类，项目适配) | 2026-05-31 |
| 是否需要生成 | 否，仅前向传播 (model.forward, 不调用 model.generate) | 2026-05-31 |
| 核心优势论证点 | 语义同源 + 数据效率高 + 零额外模型部署 (非计算量优势) | 2026-05-31 |

---

## 五、核心文献摘要

### 5.1 四篇最相关论文

**A.1 ITI — Li等, 2023 (NeurIPS 2023 Spotlight, arXiv: 2306.03341, 引用1109)**
最直接的方法论来源。在包含真实/虚假陈述的数据集上对 LLaMA 做前向推理，提取注意力头激活值（最后token），对每个头独立训练无正则化的线性探针 `p(x)=sigmoid(<θ,x>)`，用验证集准确率排序选出 top-K 敏感头（K=48/256），然后在推理时沿这些头的"真实方向"偏移激活值。**关键：探针仅用于选头和确定干预方向，没有 ensemble 阶段。** 核心发现：注意力头可以编码高级语义属性。用户方案借鉴其"提取激活 -> per-head 探针 -> 选头"的范式，但目标从"真实性"改为"文档相关性"，应用从"干预生成"改为"筛选文档"，并增加了 ensemble 分类阶段（因为筛选任务需要统一的相关性分数）。

**A.2 EvidITI (知识库PDF，未获取全文)**
核心灵感论文。将 ITI 方法论从通用场景拓展到 RAG 场景，识别"证据搜索型"注意力头并进行推理时干预。首次证明注意力头激活数据可以用于 RAG 文档质量评估。EvidITI 聚焦于"干预"（改变模型行为），用户方案聚焦于"筛选"（文档级决策）。

> 注意：分析基于标题推断，建议在原型实现前获取原文全文验证。

**A.3 CrAM — Deng等, 2024 (AAAI 2024, arXiv: 2406.11497, 引用21)** `[补充文献]`
可信度感知注意力修改机制。识别对 RAG 性能有显著影响的注意力头，根据文档可信度动态调节注意力权重。证明了"注意力头激活可以反映文档质量"的假设，为用户方案提供理论支持。区别：CrAM 修改注意力权重（侵入式），用户方案用探针筛选文档（非侵入式）。

**A.4 ADR — Wang等, 2025 (Information期刊)** `[补充文献]`
检测"RAG抑制头"并学习权重系数在推理时衰减其输出。提供了"注意力头激活模式可以反映 RAG 质量"的进一步证据。区别：ADR 检测并抑制坏头，用户方案筛选文档而非修改模型。

### 5.2 四篇核心论文对比

| 维度 | ITI | EvidITI | CrAM | ADR | **用户方案** |
|:----:|:----:|:--------:|:----:|:---:|:------------:|
| **目标** | 真实性增强 | RAG鲁棒性增强 | 可信文档优先 | 抑制RAG坏头 | 文档相关性筛选 |
| **操作对象** | 注意力头激活值 | 注意力头激活值 | 注意力权重 | 头部输出权重 | 检索文档列表 |
| **操作方式** | 推理时偏移 | 推理时干预 | 动态调节权重 | 推理时衰减 | 探针分类筛选 |
| **是否需要探针** | 是(truthfulness) | 是(evidence-seeking) | 否 | 否 | **是(relevance)** |
| **干预时机** | 推理中 | 推理中 | 推理中 | 推理中 | **检索后，推理前** |
| **互补性/角色** | 方法论来源 | 场景迁移验证 | 假设支撑 | 激进性佐证 | — |

### 5.3 其他论文分类（关联度较低）

- **混合检索/系统优化：** AH-RAG (自适应混合检索, Recall@20+15.3%), Zhou等(多策略RAG优化), CARROT (代价约束检索优化, arXiv: 2411.00744) — 均为检索策略/工程优化，不涉及注意力头分析
- **查询/文档扩展：** LLM-QE (RL优化查询扩展, arXiv: 2502.17057), Doc2Query++ (主题覆盖文档扩展, arXiv: 2510.09557) — 与注意力头分析无直接关联
- **分块优化：** MoC (Mixture-of-Chunkers, arXiv: 2503.09600), Growing Window — 文档预处理优化
- **图增强/动态检索：** GFM-RAG (图基础模型检索器, arXiv: 2502.01113), Context-Guided Dynamic Retrieval (arXiv: 2504.19436)
- **两阶段检索：** FlashRank (两阶段重排序, arXiv: 2601.03258, 关联度★★★☆☆ — 同样关注检索后筛选但使用外部模型), Setty等(金融RAG管线)
- **综述：** Zhao等2026a (RAG技术全景), Zhao等2026b (归因技术缓解幻觉, 与注意力分析有相通之处)

### 5.4 研究缺口（创新性论证）

1. **ITI 未应用于 RAG 文档筛选：** ITI 的方法论潜力尚未被拓展到 RAG 文档筛选场景。用户方案将探针目标从"真实性"迁移到"检索片段相关性"。
2. **EvidITI 聚焦干预而非筛选：** 目前尚未有工作将"从注意力头激活判断文档质量"独立出来用于检索后的文档筛选/重排序阶段。用户方案提出"筛选而非干预"的新范式，更轻量、更模块化。
3. **CrAM 和 ADR 修改注意力权重而非筛选文档：** 侵入式修改模型内部机制工程复杂且可能影响通用能力。用户方案采用非侵入式筛选策略，保持模型完整性。
4. **现有重排序方法依赖外部模型：** Cross-encoder 等外部模型与生成 LLM 表示空间不一致。用户方案实现筛选器和生成器的完全同源设计。

---

## 六、模型路线

### 6.1 实验模型选择

| 阶段 | 模型 | 参数量 | GPU需求 | 用途 |
|------|------|--------|---------|------|
| Phase 1 可行性验证 | LLaMA-3.2-3B | 3B | 单卡4090 (~6GB VRAM) | 验证核心假设 + 方案A/B对比 |
| Phase 2 主实验 | LLaMA-3.1-8B | 8B | 单卡4090 (~16GB VRAM) | 完整实验 |
| Phase 2 交叉验证 | Mistral-7B-v0.3 | 7.3B | 单卡4090 (~15GB VRAM) | 跨架构泛化性 |
| Phase 3 (可选) | Qwen2.5-7B | 7.6B | 单卡4090 | 中文RAG场景 |

**实验执行环境：** 远程 GPU 服务器（非本地笔记本）

### 6.2 Attention Head 配置速查

| 模型 | 层数 | 每层头数 | 头维度 | 总头数 | 总激活维度 |
|------|------|---------|--------|--------|-----------|
| Llama-3.2-1B | 16 | 32 | 64 | 512 | 32,768 |
| Llama-3.2-3B | 28 | 24 | 128 | 672 | 86,016 |
| Llama-3.1-8B | 32 | 32 | 128 | 1024 | 131,072 |
| Mistral-7B-v0.3 | 32 | 32 | 128 | 1024 | 131,072 |

### 6.3 硬件需求估算

**Phase 1（单卡 4090）：** 模型~6GB + 中间激活~2GB = ~8GB VRAM；12,500条激活~4.3GB内存可存；单条推理~20ms，总提取~4min。

**Phase 2（单/双卡 4090）：** 模型~16GB + 中间激活~3GB = ~19GB（单卡勉强够）；50,000条激活~22GB需双卡或分批存盘；batch=8时总提取~10min。

---

## 七、数据集方案

### 7.1 首选：MS MARCO Passage Ranking

| 属性 | 值 |
|------|-----|
| 来源 | Microsoft, HuggingFace: `microsoft/ms_marco` v1.1 |
| 规模 | ~532K queries (train), ~6.9K (dev) |
| 标注 | `is_selected`: 1=relevant, 0=not relevant |
| 领域 | Web 搜索（Bing 真实查询） |
| 推荐理由 | 天然正负标注，最符合探针训练需求 |

### 7.2 备选：Natural Questions

| 属性 | 值 |
|------|-----|
| 来源 | Google, HuggingFace: `natural_questions` |
| 规模 | ~307K (train), ~7.8K (dev) |
| 标注 | gold passage（正样本），负样本需自行构建 |
| 推荐理由 | 广泛使用，便于与 AH-RAG 等方法对比 |

### 7.3 数据规模与切分

```
Phase 1: 500 queries, 正负比~1:4, 总~12,500条激活对
Phase 2: 2,000 queries, 总~50,000条
切分: train 70% / val 15% / test 15% (按 query 维度，避免数据泄露)
```

### 7.4 其他可用数据集

TriviaQA (95K, evidence标注, ★★★★☆), HotpotQA (113K, 多跳场景, ★★★★☆), ASQA (歧义场景, ★★★★☆), BEIR (多数据集泛化评估, ★★★☆☆), KILT (多任务, ★★★☆☆), FiQA (金融领域, ★★★☆☆)。

---

## 八、代码框架选型

| 优先级 | 框架 | 说明 |
|--------|------|------|
| 主方案 | **nnsight v0.7.0** | 现代API，`model.trace()` 上下文管理，`.save()` 标记，支持 LLaMA-3/Mistral/Qwen，活跃维护至 2026-05 |
| Fallback | transformers `output_hidden_states` | 仅能获取 hidden states，不能获取 head 级激活 |
| 保底 | honest_llama 代码参考 | ITI原仓库，仅适配 LLaMA-1，需改写 |

**编程语言：** Python 3.10+
**关键依赖：** nnsight, transformers, torch, scikit-learn, sentence-transformers, accelerate, datasets

**nnsight 核心用法示例：**

```python
model = LanguageModel("meta-llama/Llama-3.2-3B", device_map="auto", dispatch=True)
with model.trace(prompt):
    attn_out = model.model.layers[15].self_attn.o_proj.output[0, -1, :].save()
# attn_out.value 直接可用
```

---

## 九、实现计划摘要

### 9.1 方法论流程图

```
标注数据集 (query, passage, label)
        |
构造 Prompt "Q: {query}\nP: {passage}"
        |
LLM 前向推理 (forward, 不生成token)
        |
提取各层注意力头输出激活
  - 方案A: passage末位token
  - 方案B: passage范围平均池化
        |
训练逻辑回归探针 (per-head + top-k选择)
        |
推理时: 逐passage forward -> 提取 -> 探针打分 -> 筛选
```

### 9.2 五阶段伪代码要点

**Phase 1 数据构建：** 加载 MS MARCO -> 按 query 构建 (query, passage, label) 三元组 -> 按 query 维度 70/15/15 切分（防止泄露）。

**Phase 2 激活提取：**
- 方案A：取 `o_proj.output[0, -1, :]`（序列末位 token = passage 末位 token）
- 方案B：先 tokenize 定位 passage 起止位置，再对 `o_proj.output[0, start:end, :]` 做 mean pooling
- Phase 1 逐条推理，Phase 2 改 batch

**Phase 3 探针训练（两阶段）：**
- Stage 1（参考ITI）：每个 attention head 独立训练 L1 正则化逻辑回归 -> 验证集评估 -> 按验证集 ROC-AUC 排序选 top-k heads。注意：ITI 原文中 per-head 探针仅用于选头和提供干预方向，不使用正则化，也没有 ensemble 阶段。此处加 L1 是项目选择，用于头内特征稀疏化。
- Stage 2（项目适配）：将 top-k heads 的激活拼接为统一特征向量 -> 训练 L2 正则化逻辑回归作为最终分类器。ITI 不做这步，因为 ITI 的目标是干预生成而非独立分类。本项目需要统一的相关性分数来筛选文档，因此 ensemble 是必要的适配。
- 消融实验：对比 Stage 1 only（per-head 预测取均值/投票）vs Stage 1+2（ensemble），验证 ensemble 是否带来增益。

**Phase 4 评估：** accuracy, F1, ROC-AUC；方案A vs 方案B 对比。

**Phase 5 端到端 RAG：** `rerank_with_probe()` 对候选 passage 列表打分取 top_n -> 计算 Recall@k (k=1,3,5,10) -> 与 cross-encoder baseline (`ms-marco-MiniLM-L-6-v2`) 对比。

### 9.3 激活提取方案对比实验矩阵

```
因子:
  - 模型:   [Llama-3.2-3B, Llama-3.1-8B]
  - 提取方案: [last_token, pooling]
  - 数据集大小: [500 queries, 1000 queries]
  - top_k heads: [50, 100, 200, all]

评估指标: 探针分类准确率/F1/ROC-AUC, 端到端 Recall@k, 训练时间
```

**方案对比要点：**

| 对比项 | 方案A: 末位token | 方案B: 平均池化 |
|--------|------------------|----------------|
| 提取计算量 | O(1) 一次索引 | O(L_passage) 均值计算 |
| 信号聚焦 | 精炼，可能丢失早期token信息 | 平滑，可能稀释关键token信号 |
| 实现复杂度 | 低 | 中（需额外tokenize定位） |
| ITI验证 | 已验证 | 未被ITI验证 |

---

## 十、实验里程碑

| Milestone | 内容 | 预计耗时 | 验证标准 |
|-----------|------|---------|---------|
| M1 | 环境搭建 + 数据准备 | 1-2天 | nnsight与Llama-3.2-3B联调通过 |
| M2 | 方案A激活提取 + 探针训练 | 1-2天 | 探针准确率 > 50% (随机基线) |
| M3 | 方案A vs 方案B 对比 | 1天 | 确定最终提取方案 |
| M4 | 端到端RAG + cross-encoder对比 | 2天 | Recall@k 对比表 |
| M5 | 扩展到8B模型 + 跨模型验证 | 2-3天 | 泛化性验证 |

**最关键验证点：M2** — 若探针准确率不显著高于随机，核心假设不成立，需调整方向（如改为 MLP 层激活或残差流）。

### M1 检查清单

- 安装 nnsight + transformers + torch
- 下载 Llama-3.2-3B 到服务器
- 验证 nnsight + Llama-3.2-3B 联调通过（打印任意层激活 shape）
- 下载 MS MARCO 数据集
- 运行数据构建脚本，产出 500 queries 的三元组
- 验证数据集切分正确（train/val/test 无 query 泄露）

---

## 十一、潜在挑战与应对

| 挑战 | 风险 | 应对策略 |
|------|------|---------|
| 注意力头激活无法编码文档相关性 | 中 | M2最早验证。若不显著高于随机，换方向（MLP层激活/残差流） |
| nnsight与新模型不兼容 | 低 | 降级到transformers hooks或raw PyTorch hooks |
| 激活维度高导致内存/过拟合 | 中 | top-k head选择 + L1正则化 + PCA降维 |
| MS MARCO正负样本不平衡 | 低 | 1:4比例可控，调整class_weight |
| 单卡4090 VRAM不够存50K条激活 | 中 | 分批提取->存盘->离线训练探针（逻辑回归CPU可跑） |
| 探针跨领域/跨数据集泛化 | 中 | 在BEIR等多样化基准上系统评估 |

---

## 十二、知识库信息


**文献数据来源说明：** ITI 和 EvidITI 直接来源于PDF论文知识库；CrAM 和 ADR 来源于 Semantic Scholar 外部补充搜索（以"RAG + attention heads"为关键词），不在知识库中。B~G 类论文均来源于知识库。

---

## 十三、关键注意事项

1. **实验必须在远程GPU服务器上进行**，笔记本配置不足以运行任何模型
2. **禁止胡编乱造**，所有文献信息需可溯源
3. 文献报告中的 CrAM 和 ADR 是外部补充搜索发现，非知识库论文
4. EvidITI 分析基于文件名+标题推断，未获取原文全文
5. 幻觉检测已从当前方案中移除，仅聚焦文档相关性探针
6. 核心优势论证点是语义同源 + 数据效率 + 零额外模型部署，不是计算量优势

---

## 十四、待办事项

- [ ] 配置远程GPU服务器SSH连接
- [ ] 在服务器上安装Python环境 + nnsight + transformers
- [ ] 下载 MS MARCO数据集 + LLaMA-3.2-3B 模型权重
- [ ] 运行 Milestone 1 环境验证
- [ ] 运行 Milestone 2 核心假设验证
- [ ] 获取 EvidITI 论文原文全文（如可获取）

---

## 十五、原始文件索引

本文档整合自以下四个文件，如需查看完整原文可参考：

| 文件 | 原始路径 | 说明 |
|------|----------|------|
| 项目记忆 | `D:\solo-paper\.project-memory.md` | 项目决策与进度追踪 |
| 文献调研报告 | `D:\solo-paper\RAG检索优化文献调研报告.md` | 17篇论文完整分析 + 数据集/模型详细调研 |
| 实现计划 | `D:\solo-paper\实现计划-Attention-Probe-RAG.md` | 完整伪代码 + 实验设计 |
| 讨论总结 | `D:\solo-paper\讨论总结-研究思路梳理.md` | 核心思路与优势论述 |

> 如需查看完整伪代码实现，请参见原始文件 `实现计划-Attention-Probe-RAG.md`。
> 如需查看17篇论文的逐篇详细分析，请参见原始文件 `RAG检索优化文献调研报告.md`。
