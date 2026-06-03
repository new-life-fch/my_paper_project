# RAG检索阶段优化论文精选（提升召回率方向）

本文档精选了10篇在 RAG（检索增强生成）系统检索阶段进行优化以提升召回率（Recall）的代表性论文，涵盖混合检索、查询扩展、文档扩展、分块优化、图增强检索、动态检索等多个技术方向。

## 一、论文列表

### 1\. AH\-RAG: Adaptive Hybrid Retrieval\-Augmented Generation with Context\-Aware Confidence Control

- **作者：** Yiran Sun \| **年份：** 2025 \| **引用数：** 1

- **核心方法：** 动态融合稀疏检索（BM25）与稠密检索（DPR），并在生成阶段引入上下文置信度控制机制。在 Natural Questions 数据集上 Recall@20 提升 15\.3%，EM 提升 0\.6%。

- **获取：✅ 已上传到本知识库**

### 2\. A Method for Improving Retrieval Effectiveness in RAG

- **作者：** Xuewen Zhou, Zhuowei Li, Rong Jiang, Chaojun Xu, Chengcheng Shao \| **年份：** 2025 \| **引用数：** 0

- **核心方法：** 混合检索（BM25 \+ DPR）联合网格搜索参数优化与递归层次化检索。混合检索相比单一稠密检索召回率提升超 20%，递归检索提升精确度 4\.61%。

- **获取：✅ 已上传到本知识库**

### 3\. Improving Retrieval for RAG based QA Models on Financial Documents

- **作者：** Spurthi Setty, Katherine Jijo, Eden Chung, Natan Vidra \| **年份：** 2024 \| **引用数：** 65

- **核心方法：** 系统性改进 RAG 检索管线，包括语义分块策略、查询扩展（Query Expansion）、元数据标注、重排序算法（Re\-ranking）和 Embedding 微调。

- **PDF：** ✅ 已上传到本知识库 \| arXiv: [https://arxiv\.org/abs/2404\.07221](https://arxiv.org/abs/2404.07221)

### 4\. CARROT: A Learned Cost\-Constrained Retrieval Optimization System for RAG

- **作者：** Ziting Wang, Haitao Yuan, Wei Dong, Gao Cong, Feifei Li \| **年份：** 2024 \| **引用数：** 7

- **核心方法：** 基于学习的代价约束检索优化系统，在给定预算约束下自动选择最优检索策略，平衡检索质量与计算成本。

- **PDF：** ✅ 已上传到本知识库 \| arXiv: [https://arxiv\.org/abs/2411\.00744](https://arxiv.org/abs/2411.00744)

### 5\. Optimising Retrieval Performance in RAG Systems: Growing Window Semantic Chunking

- **作者：** Antonio Moreno\-Cediel, Eva Garcia\-Lopez, Antonio Garcia\-Cabot, David de\-Fitero\-Dominguez \| **年份：** 2025 \| **引用数：** 0

- **核心方法：** 针对弱语义边界问题提出增长窗口语义分块策略，通过动态扩加分块窗口提升检索召回率。

- **获取：** ✅ 已上传到本知识库

### 6\. MoC: Mixtures of Text Chunking Learners for RAG

- **作者：** Jihao Zhao, Zhiyuan Ji, Zhaoxin Fan, Hanyu Wang, Simin Niu, Bo Tang, Feiyu Xiong, Zhiyu Li \| **年份：** 2025 \| **引用数：** 21

- **核心方法：** 提出 Mixture\-of\-Chunkers（MoC）框架，引入粒度感知的多分块器混合机制，结合 LLM 进行智能分块正则表达式生成。

- **PDF：** ✅ 已上传到本知识库 \| arXiv: [https://arxiv\.org/abs/2503\.09600](https://arxiv.org/abs/2503.09600)

### 7\. GFM\-RAG: Graph Foundation Model for Retrieval Augmented Generation

- **作者：** Linhao Luo, Zicheng Zhao, Gholamreza Haffari, Dinh Phung, Chen Gong, Shirui Pan \| **年份：** 2025 \| **引用数：** 28

- **核心方法：** 首个可用于未见数据集的图基础模型（GFM）检索器。在 14M\+ 三元组和 700K 文档上训练 8M 参数 GNN，无需微调即可泛化到新数据集。

- **PDF：** ✅ 已上传到本知识库 \| arXiv: [https://arxiv\.org/abs/2502\.01113](https://arxiv.org/abs/2502.01113)

### 8\. Context\-Guided Dynamic Retrieval for Improving Generation Quality in RAG

- **作者：** Jacky He, Guiran Liu, Binrong Zhu, Hanlu Zhang, Hongye Zheng, Xiaokai Wang \| **年份：** 2025 \| **引用数：** 12

- **核心方法：** 状态感知的动态知识检索机制，通过多层次感知检索向量构建和可微分文档匹配路径，实现检索与生成模块的端到端联合训练。

- **PDF：** ✅ 已上传到本知识库 \| arXiv: [https://arxiv\.org/abs/2504\.19436](https://arxiv.org/abs/2504.19436)

### 9\. LLM\-QE: Improving Query Expansion by Aligning LLMs with Ranking Preferences

- **作者：** Sijia Yao, Pengcheng Huang, Zhenghao Liu, Yu Gu, Yukun Yan, Shi Yu, Ge Yu \| **年份：** 2025 \| **引用数：** 4

- **核心方法：** 通过将 LLM 与排序偏好对齐排序偏好来优化查询扩展质量，使用强化学习训练更具判别性的查询扩展词。

- **PDF：** ✅ 已上传到本知识库 \| arXiv: [https://arxiv\.org/abs/2502\.17057](https://arxiv.org/abs/2502.17057)

### 10\. Doc2Query\+\+: Topic\-Coverage based Document Expansion for Dense Retrieval

- **作者：** Tzu\-Lin Kuo, Wei\-Ning Chiu, Wei\-Yun Ma, Pu\-Jen Cheng \| **年份：** 2025 \| **引用数：** 1

- **核心方法：** 基于主题覆盖度的文档扩展方法，结合双索引融合（Dual\-Index Fusion）策略提升稠密检索性能。

- **PDF：** ✅ 已上传到本知识库 \| arXiv: [https://arxiv\.org/abs/2510\.09557](https://arxiv.org/abs/2510.09557)

---

## 二、技术方向分类

### 混合检索

论文 1（AH\-RAG）、2、8 通过稀疏与稠密检索的动态融合提升召回率。

### 查询扩展与文档扩展

论文 9（LLM\-QE）优化查询侧扩展；论文 10（Doc2Query\+\+）优化文档侧表征。

### 分块策略优化

论文 5（Growing Window Chunking）、6（MoC）从文本分块粒度入手提升检索命中质量。

### 图增强检索

论文 7（GFM\-RAG）利用图神经网络建模知识间复杂关系，提升多跳推理场景下的检索准确性。

### 系统级管线优化

论文 3（Financial RAG）、4（CARROT）从系统角度综合优化检索管线各组件。

---

## 三、关键指标汇总

|论文|关键指标|提升幅度|数据集|
|---|---|---|---|
|AH\-RAG|Recall@20|\+15\.3%|Natural Questions|
|混合检索|Recall|大于20%|Amnesty\_qa|
|动态检索|BLEU/ROUGE\-L|显著提升|Natural Questions|
|GFM\-RAG|Multi\-hop QA|SOTA|多跳与领域数据集|

---

**更新日期：** 2025\-05\-18 \| **检索平台：** Semantic Scholar

> (注：内容由 AI 生成，请谨慎参考）
