# Attention-Probe-RAG：实现计划

> 版本：v1.0 | 日期：2026-05-31 | 状态：待执行

---

## 目录

- [一、背景与目标](#一背景与目标)
- [二、方法论总览](#二方法论总览)
- [三、详细伪代码](#三详细伪代码)
  - [Phase 0：环境搭建](#phase-0环境搭建)
  - [Phase 1：数据集构建](#phase-1数据集构建)
  - [Phase 2：激活向量提取](#phase-2激活向量提取)
  - [Phase 3：探针训练](#phase-3探针训练)
  - [Phase 4：探针评估](#phase-4探针评估)
  - [Phase 5：端到端 RAG 筛选](#phase-5端到端-rag-筛选)
- [四、激活提取方案对比实验设计](#四激活提取方案对比实验设计)
- [五、数据集方案](#五数据集方案)
- [六、模型选型](#六模型选型)
- [七、代码框架选型](#七代码框架选型)
- [八、硬件需求估算](#八硬件需求估算)
- [九、实验路线与里程碑](#九实验路线与里程碑)
- [十、潜在挑战与应对](#十潜在挑战与应对)

---

## 一、背景与目标

### 1.1 问题定义

RAG 系统检索阶段常返回大量文档片段，其中混杂噪声文档。传统重排序方法依赖外部 cross-encoder 或小型排序模型，存在以下局限：

- 需要额外模型加载和推理开销
- 外部模型与生成 LLM 的表示空间不一致（语义隔阂）
- 无法利用 LLM 自身的推理信号参与判断

### 1.2 核心思路

**利用 LLM 自身的注意力头激活训练二元探针，自主判断检索片段的相关性。**

流程：`检索片段` → `(query, passage) 拼接` → `LLM 前向推理` → `提取注意力头激活` → `探针二分类` → `保留相关片段`

### 1.3 实施目标

1. **可行性验证（单卡 4090）**：在小参数量模型上验证"注意力头激活能编码文档相关性信号"这一核心假设
2. **激活提取方案对比**：对比"passage 末位 token" vs "passage 范围平均池化"两种激活提取策略
3. **与 Cross-Encoder Reranker 对比**：证明同源探针的优势（数据效率、语义对齐）

---

## 二、方法论总览

```
                              ┌──────────────────────────┐
                              │   标注 RAG 数据集          │
                              │  (query, passage, label)  │
                              └────────────┬─────────────┘
                                           │
                              ┌────────────▼─────────────┐
                              │   构造 Prompt              │
                              │   "Q: {query}\nP: {passage}"│
                              └────────────┬─────────────┘
                                           │
                              ┌────────────▼─────────────┐
                              │   LLM 前向推理 (forward)   │
                              │   不生成 token             │
                              └────────────┬─────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
           ┌────────▼────────┐   ┌────────▼────────┐   ┌────────▼────────┐
           │ Layer 0 Heads   │   │ Layer 1 Heads   │   │ Layer L Heads   │
           │ [激活向量]       │   │ [激活向量]       │   │ [激活向量]       │
           └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
                    │                      │                      │
                    └──────────────────────┼──────────────────────┘
                                           │
                              ┌────────────▼─────────────┐
                              │   提取目标 token 的激活    │
                              │   方案A: passage末位token  │
                              │   方案B: passage范围池化   │
                              └────────────┬─────────────┘
                                           │
                              ┌────────────▼─────────────┐
                              │   训练逻辑回归探针         │
                              │   正样本: relevant=1      │
                              │   负样本: irrelevant=0    │
                              └────────────┬─────────────┘
                                           │
                              ┌────────────▼─────────────┐
                              │   推理时：逐个 passage     │
                              │   前向→提取→探针打分→筛选  │
                              └──────────────────────────┘
```

### 关键设计决策（已敲定）

| 决策点 | 选择 | 说明 |
|--------|------|------|
| 激活提取方案 | **方案A + 方案B 并行对比** | 方案A: passage末位token；方案B: passage范围平均池化 |
| Batch 策略 | **Phase 1 逐一推理，Phase 2 改为 batch** | 先验证可行性，再优化效率 |
| 对比 Baseline | **Cross-encoder reranker** | 论文论证核心对比对象 |
| 探针类型 | **逻辑回归 (logistic regression)** | 与 ITI 一致，可解释性强，过拟合风险低 |
| 是否需要生成 | **否，仅前向传播** | 不调用 `model.generate()` |

---

## 三、详细伪代码

### Phase 0：环境搭建

```python
# ======== 环境安装 ========
# pip install nnsight==0.7.0
# pip install transformers torch accelerate

# ======== 导入 ========
import torch
import numpy as np
from nnsight import LanguageModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from datasets import load_dataset
from typing import List, Tuple, Dict
```

### Phase 1：数据集构建

#### 1a. 加载 MS MARCO

```python
# ==========================================
# Phase 1a: 加载 MS MARCO Passage Ranking
# ==========================================

def load_msmarco_data(
    num_queries: int = 1000,
    positives_per_query: int = 5,
    negatives_per_query: int = 20,
    seed: int = 42
) -> Tuple[List[str], List[str], List[int]]:
    """
    从 MS MARCO 构建 (query, passage, label) 三元组

    Args:
        num_queries:    使用的 query 数量
        positives_per_query: 每个 query 保留的正样本数（有标注的相关 passage）
        negatives_per_query: 每个 query 采样的负样本数（从候选池中随机采样）

    Returns:
        queries:    List[str], 长度 = num_queries * (pos + neg)
        passages:   List[str], 同上
        labels:     List[int], 1=相关, 0=不相关
    """
    rng = np.random.RandomState(seed)

    # ---- 加载 MS MARCO 数据集 ----
    # HuggingFace: microsoft/ms_marco, 配置: v1.1, split: train
    # 每条数据: {
    #   "query": str,
    #   "passages": {"passage_text": [...], "is_selected": [...]}
    # }
    dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train")

    # ---- 打乱并采样 query ----
    indices = rng.permutation(len(dataset))[:num_queries]
    sampled = dataset.select(indices)

    queries, passages, labels = [], [], []

    for item in sampled:
        query = item["query"]
        passage_texts = item["passages"]["passage_text"]
        is_selected = item["passages"]["is_selected"]  # 1 = relevant

        # ---- 正样本：is_selected == 1 的 passage ----
        positive_indices = [i for i, sel in enumerate(is_selected) if sel == 1]
        if len(positive_indices) == 0:
            continue  # 跳过无正样本的 query

        selected_pos = positive_indices[:positives_per_query]

        for pi in selected_pos:
            queries.append(query)
            passages.append(passage_texts[pi])
            labels.append(1)

        # ---- 负样本：is_selected == 0 中随机采样 ----
        negative_indices = [i for i, sel in enumerate(is_selected) if sel == 0]
        if len(negative_indices) == 0:
            continue

        n_neg = min(negatives_per_query, len(negative_indices))
        selected_neg = rng.choice(negative_indices, size=n_neg, replace=False)

        for ni in selected_neg:
            queries.append(query)
            passages.append(passage_texts[ni])
            labels.append(0)

    return queries, passages, labels
```

#### 1b. 数据集切分

```python
def split_dataset(
    queries: List[str],
    passages: List[str],
    labels: List[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, Tuple]:
    """
    按 query 维度切分，保证同一 query 的所有 (passage, label) 对
    进入同一个 split，避免数据泄露。
    """
    rng = np.random.RandomState(seed)

    # 按 query 分组
    query_groups = {}
    for q, p, l in zip(queries, passages, labels):
        query_groups.setdefault(q, []).append((p, l))

    unique_queries = list(query_groups.keys())
    rng.shuffle(unique_queries)

    n = len(unique_queries)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_queries = set(unique_queries[:n_train])
    val_queries = set(unique_queries[n_train:n_train + n_val])
    test_queries = set(unique_queries[n_train + n_val:])

    def build_split(q_set):
        qs, ps, ls = [], [], []
        for q in q_set:
            for p, l in query_groups[q]:
                qs.append(q); ps.append(p); ls.append(l)
        return qs, ps, ls

    return {
        "train": build_split(train_queries),
        "val":   build_split(val_queries),
        "test":  build_split(test_queries)
    }
```

### Phase 2：激活向量提取

#### 2a. 定义 Prompt 模板

```python
PROMPT_TEMPLATE = """\
Question: {question}

Passage: {passage}"""

def build_prompt(question: str, passage: str) -> str:
    return PROMPT_TEMPLATE.format(question=question, passage=passage)
```

#### 2b. 提取激活 — 方案A：passage 末位 token

```python
# ==========================================
# Phase 2b (方案A): 提取 passage 最后一个 token 的激活
# 技巧: 由于 prompt 拼法是 "Q: ...\n\nP: ..."，
#       序列末位 token = passage 末位 token
# ==========================================

def extract_activation_last_token(
    model: LanguageModel,
    query: str,
    passage: str,
    layers_to_extract: List[int] = None
) -> np.ndarray:
    """
    提取各层注意力头的输出激活 — 取序列最后一个 token 位置。

    Returns:
        activation: shape = [total_heads * head_dim]
                    其中 total_heads = n_layers * n_heads_per_layer
    """
    prompt = build_prompt(query, passage)

    all_activations = []

    with model.trace(prompt) as tracer:
        # 遍历指定的层，提取 attention output
        # nnsight 中: model.model.layers[i].self_attn.o_proj.output
        # 即 attention 输出投影后的 hidden states
        for layer_idx in layers_to_extract:
            # attention output: shape [batch=1, seq_len, hidden_dim]
            attn_output = model.model.layers[layer_idx].self_attn.o_proj.output
            # 取最后一个 token: shape [1, hidden_dim]
            last_token = attn_output[0, -1, :]
            all_activations.append(last_token.save())

    # 拼接所有层的激活
    # shape: [n_layers, hidden_dim]
    # flatten 后: [n_layers * hidden_dim]
    result = np.concatenate([
        act.value.detach().cpu().float().numpy().flatten()
        for act in all_activations
    ])

    return result
```

#### 2c. 提取激活 — 方案B：passage 范围平均池化

```python
# ==========================================
# Phase 2c (方案B): 对 passage 覆盖的所有 token 做 mean pooling
# ==========================================

def extract_activation_pooling(
    model: LanguageModel,
    query: str,
    passage: str,
    layers_to_extract: List[int] = None
) -> np.ndarray:
    """
    提取各层注意力头的输出激活 — 对 passage 范围内的 token 做 mean pooling。

    实现思路:
    1. 对 prompt 做 tokenize, 定位 passage 的起止 token 位置
    2. 前向推理, 提取每层 attention output
    3. 对 passage token 范围做 mean pooling
    """
    prompt = build_prompt(query, passage)

    # ---- 步骤 1: 定位 passage token 范围 ----
    # 分别 tokenize question 部分和完整 prompt, 用差异定位
    question_prefix = PROMPT_TEMPLATE.split("{passage}")[0].format(question=query)

    q_ids = model.tokenizer(question_prefix, return_tensors="pt")["input_ids"][0]
    full_ids = model.tokenizer(prompt, return_tensors="pt")["input_ids"][0]

    # passage 起始 = question_prefix 的 token 数
    passage_start = len(q_ids)
    passage_end = len(full_ids)  # 末位 token = passage 最后一个 token

    all_activations = []

    with model.trace(prompt) as tracer:
        for layer_idx in layers_to_extract:
            attn_output = model.model.layers[layer_idx].self_attn.o_proj.output
            # 对 passage 范围做 mean pooling: [1, hidden_dim]
            pooled = attn_output[0, passage_start:passage_end, :].mean(dim=0)
            all_activations.append(pooled.save())

    result = np.concatenate([
        act.value.detach().cpu().float().numpy().flatten()
        for act in all_activations
    ])

    return result
```

#### 2d. 批量提取激活（Phase 1：逐一推理）

```python
def build_activation_dataset(
    model: LanguageModel,
    queries: List[str],
    passages: List[str],
    labels: List[int],
    extraction_method: str = "last_token",
    layers_to_extract: List[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对每条 (query, passage) 逐一做前向推理，提取激活向量。

    注意: Phase 1 使用逐一推理（每条独立一次 forward），
    未来 Phase 2 优化时改为 batch 推理。

    Args:
        extraction_method: "last_token" | "pooling"

    Returns:
        X: np.ndarray, shape [n_samples, activation_dim]
        y: np.ndarray, shape [n_samples]
    """
    X_list, y_list = [], []

    for q, p, l in zip(queries, passages, labels):
        if extraction_method == "last_token":
            act = extract_activation_last_token(model, q, p, layers_to_extract)
        elif extraction_method == "pooling":
            act = extract_activation_pooling(model, q, p, layers_to_extract)
        else:
            raise ValueError(f"Unknown method: {extraction_method}")

        X_list.append(act)
        y_list.append(l)

    return np.array(X_list), np.array(y_list)
```

### Phase 3：探针训练

```python
# ==========================================
# Phase 3: 训练逻辑回归探针
# 参考 ITI 的做法: 对每个 attention head 独立训练探针,
# 然后选择 top-k 个最"有用"的头
# ==========================================

def train_probe_per_head(
    X_train: np.ndarray,   # shape: [n_samples, n_heads * head_dim]
    y_train: np.ndarray,   # shape: [n_samples]
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_heads: int,
    head_dim: int,
    top_k: int = 100,      # 保留前 k 个最有判别力的头
    C: float = 1.0         # L2 正则化强度
) -> Tuple[LogisticRegression, np.ndarray, List[int]]:
    """
    为每个 attention head 独立训练逻辑回归探针,
    根据验证集准确率选择 top-k 个 head。

    Returns:
        probe:            合并后的 LogisticRegression（使用选择的 head）
        head_accuracies:  每个 head 的验证集准确率
        selected_heads:   被选中的 head 索引列表
    """
    rng = np.random.RandomState(42)

    # ---- 为每个 head 训练独立探针, 记录验证集准确率 ----
    head_accuracies = np.zeros(n_heads)

    for head_idx in range(n_heads):
        start = head_idx * head_dim
        end = (head_idx + 1) * head_dim
        X_head_train = X_train[:, start:end]
        X_head_val = X_val[:, start:end]

        clf = LogisticRegression(
            penalty="l1",          # L1 正则化—进一步进行特征选择
            C=C,
            solver="saga",
            max_iter=1000,
            random_state=42
        )
        clf.fit(X_head_train, y_train)
        head_accuracies[head_idx] = clf.score(X_head_val, y_val)

    # ---- 选择 top-k 个 head ----
    selected_heads = np.argsort(head_accuracies)[::-1][:top_k]

    # ---- 用选中的 head 的激活拼接, 训练最终探针 ----
    X_train_selected = np.concatenate([
        X_train[:, h * head_dim:(h + 1) * head_dim]
        for h in selected_heads
    ], axis=1)

    final_probe = LogisticRegression(
        penalty="l2",
        C=C,
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )
    final_probe.fit(X_train_selected, y_train)

    return final_probe, head_accuracies, selected_heads


def probe_predict(
    probe: LogisticRegression,
    activation: np.ndarray,
    selected_heads: List[int],
    head_dim: int
) -> float:
    """
    用训练好的探针对单条激活向量打分。

    Returns:
        score: float, 该 passage 是 "relevant" 的概率
    """
    # 提取选中 head 的激活
    features = np.concatenate([
        activation[h * head_dim:(h + 1) * head_dim]
        for h in selected_heads
    ]).reshape(1, -1)

    proba = probe.predict_proba(features)[0, 1]  # P(label=1)
    return float(proba)
```

### Phase 4：探针评估

```python
# ==========================================
# Phase 4: 探针性能评估
# ==========================================

def evaluate_probe(
    probe: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
    selected_heads: List[int],
    head_dim: int
) -> Dict[str, float]:
    """评估探针的二分类性能"""
    # 构建选中 head 的特征矩阵
    X_selected = np.concatenate([
        X_test[:, h * head_dim:(h + 1) * head_dim]
        for h in selected_heads
    ], axis=1)

    y_pred = probe.predict(X_selected)
    y_proba = probe.predict_proba(X_selected)[:, 1]

    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_proba),
    }


def compare_extraction_methods(
    model: LanguageModel,
    data_splits: Dict,
    layers_to_extract: List[int],
    n_heads: int,
    head_dim: int
) -> Dict[str, Dict]:
    """
    对比方案A（last_token）和方案B（pooling）的探针性能。
    """
    results = {}
    train_q, train_p, train_l = data_splits["train"]
    val_q, val_p, val_l = data_splits["val"]
    test_q, test_p, test_l = data_splits["test"]

    for method_name in ["last_token", "pooling"]:
        print(f"=== 方案: {method_name} ===")

        # 提取激活
        X_train, y_train = build_activation_dataset(
            model, train_q, train_p, train_l,
            extraction_method=method_name,
            layers_to_extract=layers_to_extract
        )
        X_val, y_val = build_activation_dataset(
            model, val_q, val_p, val_l,
            extraction_method=method_name,
            layers_to_extract=layers_to_extract
        )
        X_test, y_test = build_activation_dataset(
            model, test_q, test_p, test_l,
            extraction_method=method_name,
            layers_to_extract=layers_to_extract
        )

        # 训练探针
        probe, head_accs, selected = train_probe_per_head(
            X_train, y_train, X_val, y_val,
            n_heads=n_heads, head_dim=head_dim
        )

        # 评估
        metrics = evaluate_probe(
            probe, X_test, y_test,
            selected_heads=selected, head_dim=head_dim
        )

        results[method_name] = {
            "metrics": metrics,
            "top_heads": selected[:20],  # 保存 top-20 head 供分析
            "head_accuracies": head_accs
        }

    return results
```

### Phase 5：端到端 RAG 筛选

```python
# ==========================================
# Phase 5: 端到端对比 — RAG 管线中的探针筛选
# ==========================================

def rerank_with_probe(
    model: LanguageModel,
    probe: LogisticRegression,
    query: str,
    candidate_passages: List[str],
    selected_heads: List[int],
    head_dim: int,
    extraction_method: str = "last_token",
    layers_to_extract: List[int] = None,
    top_n: int = 5
) -> List[Tuple[str, float]]:
    """
    用训练好的探针对候选 passage 列表打分，返回 top_n 个高分 passage。
    """
    scored = []

    for passage in candidate_passages:
        if extraction_method == "last_token":
            act = extract_activation_last_token(model, query, passage, layers_to_extract)
        else:
            act = extract_activation_pooling(model, query, passage, layers_to_extract)

        score = probe_predict(probe, act, selected_heads, head_dim)
        scored.append((passage, score))

    # 按分数降序排列, 取 top_n
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def evaluate_rag_end_to_end(
    model: LanguageModel,
    probe: LogisticRegression,
    test_queries: List[str],
    test_gold_passages: List[str],      # 每个 query 的 gold passage 列表
    test_candidate_pools: List[List[str]],  # 每个 query 的候选 passage 池
    selected_heads: List[int],
    head_dim: int,
    extraction_method: str,
    layers_to_extract: List[int]
) -> Dict[str, float]:
    """
    端到端评估：探针筛选后的 RAG 生成质量。
    指标: Recall@k（筛选出的 top-k passage 包含 gold passage 的比例）
    """
    recall_at = {1: [], 3: [], 5: [], 10: []}

    for query, gold_list, pool in zip(test_queries, test_gold_passages, test_candidate_pools):
        ranked = rerank_with_probe(
            model, probe, query, pool,
            selected_heads, head_dim,
            extraction_method, layers_to_extract
        )
        selected_texts = [r[0] for r in ranked]

        for k in recall_at:
            hits = sum(1 for g in gold_list if g in selected_texts[:k])
            recall_at[k].append(hits / len(gold_list) if gold_list else 0)

    return {f"recall@{k}": np.mean(v) for k, v in recall_at.items()}
```

#### 5b. Cross-encoder baseline

```python
def evaluate_cross_encoder_baseline(
    test_queries: List[str],
    test_gold_passages: List[str],
    test_candidate_pools: List[List[str]],
    ce_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
) -> Dict[str, float]:
    """
    使用 HuggingFace cross-encoder 做重排序作为 baseline。
    """
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(ce_model_name)

    recall_at = {1: [], 3: [], 5: [], 10: []}

    for query, gold_list, pool in zip(test_queries, test_gold_passages, test_candidate_pools):
        pairs = [[query, p] for p in pool]
        scores = ce.predict(pairs)
        ranked_idx = np.argsort(scores)[::-1]

        for k in recall_at:
            top_texts = [pool[i] for i in ranked_idx[:k]]
            hits = sum(1 for g in gold_list if g in top_texts)
            recall_at[k].append(hits / len(gold_list) if gold_list else 0)

    return {f"recall@{k}": np.mean(v) for k, v in recall_at.items()}
```

---

## 四、激活提取方案对比实验设计

### 对比维度

| 对比项 | 方案A: 末位 token | 方案B: 平均池化 |
|--------|------------------|----------------|
| 提取计算量 | O(1) 一次索引 | O(L_passage) 均值计算 |
| 信号聚焦 | 精炼，可能丢失早期 token 信息 | 平滑，可能稀释关键 token 信号 |
| 实现复杂度 | 低—直接用 nnsight 取 [-1] | 中—需要额外 tokenize 定位 passage 范围 |
| ITI 验证 | ✅ 已验证 | ❌ 未被 ITI 验证 |

### 实验矩阵

```
因子:
  - 模型:   [Llama-3.2-3B, Llama-3.1-8B]
  - 提取方案: [last_token, pooling]
  - 数据集大小: [500 queries, 1000 queries]
  - top_k heads: [50, 100, 200, all]

评估指标:
  - 探针分类准确率、F1、ROC-AUC
  - 端到端 Recall@k (k=1,3,5,10)
  - 训练时间（样本数 → 激活提取耗时）

输出:
  → 方案对比表 + 消融曲线（top_k vs accuracy）
```

---

## 五、数据集方案

### 5.1 首选：MS MARCO Passage Ranking

| 属性 | 值 |
|------|-----|
| 来源 | Microsoft, HuggingFace: `microsoft/ms_marco` |
| 规模 | ~532K queries (train), ~6.9K (dev) |
| 标注格式 | `is_selected`: 1=relevant passage, 0=not relevant |
| 候选数/query | 平均 ~8 个 passage, 需自行扩充候选池 |
| 语料领域 | Web 搜索（Bing 真实查询） |
| 推荐理由 | 天然正负标注，最符合探针训练需求 |

### 5.2 备选：Natural Questions

| 属性 | 值 |
|------|-----|
| 来源 | Google, HuggingFace: `natural_questions` |
| 规模 | ~307K (train), ~7.8K (dev) |
| 标注格式 | gold passage（正样本），负样本需自行从 Wikipedia 采样 |
| 推荐理由 | 广泛使用，便于与 AH-RAG 等方法对比 |

### 5.3 数据规模建议

```
Phase 1 (可行性验证):
  - MS MARCO: 500 queries
    * 正样本: ~500 × 5  = 2,500
    * 负样本: ~500 × 20 = 10,000
    * 总计: ~12,500 条 (query, passage) 对

Phase 2 (完整实验):
  - MS MARCO: 2,000 queries
    * 正样本: ~2,000 × 5  = 10,000
    * 负样本: ~2,000 × 20 = 40,000
    * 总计: ~50,000 条

切分: train 70% / val 15% / test 15% (按 query 维度)
```

---

## 六、模型选型

### 6.1 硬件约束

| GPU | VRAM | 可用场景 |
|-----|------|---------|
| 单卡 RTX 4090 | 24 GB | Phase 1 可行性验证 |
| 双卡 RTX 4090 | 48 GB (total) | Phase 2+ 完整实验 |

### 6.2 模型候选

| 模型 | 参数量 | FP16 VRAM | 注意力头配置 | 单卡 4090 | 推荐等级 |
|------|--------|-----------|-------------|-----------|---------|
| **Llama-3.2-3B** | 3B | ~6 GB | 28层×?头 | ✅ 非常充裕 | ★★★★★ Phase 1 首选 |
| **Llama-3.1-8B** | 8B | ~16 GB | 32层×32头 | ✅ 刚好 | ★★★★★ 主实验 |
| **Mistral-7B-v0.3** | 7.3B | ~15 GB | 32层×32头 | ✅ 刚好 | ★★★★☆ 交叉验证 |
| **Qwen2.5-7B** | 7.6B | ~15 GB | 28层×?头 | ✅ 刚好 | ★★★☆☆ 中文场景 |
| **Gemma-2-2B** | 2.6B | ~5 GB | 26层×?头 | ✅ 非常充裕 | ★★★☆☆ 快速实验 |
| **Llama-3.2-1B** | 1.2B | ~2.5 GB | 16层×?头 | ✅ 极其充裕 | ★★☆☆☆ 调试用 |

### 6.3 推荐路线

```
Phase 1 可行性验证（先用最小成本跑通全流程）:
  ┌─────────────────────────────────────────┐
  │ 主模型: Llama-3.2-3B (≈6 GB VRAM)     │
  │  GPU:   单卡 4090                        │
  │  层数:   28 层                           │
  │  激活维度估算: 28 × head_dim × n_heads  │
  │                                        │
  │  目标: 验证核心假设 + 对比方案A vs B    │
  └─────────────────────────────────────────┘

Phase 2 完整实验（换更大模型验证泛化性）:
  ┌─────────────────────────────────────────┐
  │ 主模型:   Llama-3.1-8B (≈16 GB VRAM)   │
  │ 交叉验证: Mistral-7B-v0.3               │
  │  GPU:     单卡 4090（逐个推理时）        │
  │          或双卡 4090（batch 推理时）     │
  │                                        │
  │  目标: 完整实验结果 + cross-encoder对比  │
  └─────────────────────────────────────────┘
```

### 6.4 为什么要从小模型开始？

1. **LLaMA-3.2-3B 单卡推理时 VRAM 占用仅 ~6 GB**，剩余 18 GB 可存储批量激活向量，无需频繁磁盘 I/O
2. **前向推理速度快**（3B 模型单条 forward ≈ 10-50ms），Phase 1 跑 12,500 条仅需数分钟
3. **快速试错**：探针代码逻辑、激活提取 bug、数据集格式问题都能在小模型上快速定位
4. **ITI 用了 LLaMA-7B/13B**，3B 的注意力头数量虽然少但不会改变核心结论

---

## 七、代码框架选型

### 7.1 框架对比

| 框架 | 版本 | 维护状态 | 优势 | 劣势 |
|------|------|---------|------|------|
| **nnsight** | v0.7.0 (2026-05) | ✅ 活跃 | 现代 Pythonic API，原生支持 HuggingFace/Llama，支持 batching + remote execution，MIT 协议 | 依赖抽象层，debug 不如 raw hooks |
| **honest_llama** (ITI) | 2023 | ❌ 停止维护 | 已验证 ITI 方法可行 | 老旧，用 raw PyTorch hooks，只适配 LLaMA-1，不兼容 LLaMA-3/Mistral |
| **TransformerLens** | 持续更新 | ✅ 活跃 | 专门的机械可解释性工具链，内置可视化 | 仅支持 GPT-style 模型（需手动转换），抽象层较重 |
| **Raw PyTorch Hooks** | — | — | 完全可控，无依赖 | 代码冗长，容易出错，需手动管理 hook 生命周期 |

### 7.2 推荐方案：nnsight（主方案）+ raw hooks（fallback）+ honest_llama（保底）

**主方案：nnsight（推荐指数 ★★★★★）**

```python
# 示例：nnsight 的三行核心代码
model = LanguageModel("meta-llama/Llama-3.2-3B", device_map="auto", dispatch=True)

with model.trace(prompt):
    # 提取任意层的输出，代码即注释
    attn_out = model.model.layers[15].self_attn.o_proj.output[0, -1, :].save()

# attn_out.value 直接可用
```

优势：
- `model.trace()` 上下文管理器自动管理 hook 注入/移除
- `.save()` 标记自动持久化需要的 tensor，其余自动释放
- 支持 `.invoke()` 实现多输入 batch 推理
- 文档完善，有针对 Claude Code 的 agent skill
- 适配 LLaMA-3、Mistral、Qwen、Gemma 等主流模型

**Fallback：直接使用 transformers library 的 output_hidden_states**

```python
# Fallback 方案：如果 nnsight 遇到兼容性问题
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    torch_dtype=torch.float16,
    device_map="auto",
    output_hidden_states=True   # ⚠️ 开启此选项会返回所有层的 hidden states
)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model(**inputs)
    # outputs.hidden_states: tuple of (n_layers+1) tensors
    # 每层 shape: [1, seq_len, hidden_dim]
    # 取最后一层最后一个 token:
    last_hidden = outputs.hidden_states[-1][0, -1, :]
```

⚠️ 但这个 fallback 只能获取 hidden states，不能直接获取 attention head 级别的输出（`o_proj.output`）。需要更细粒度的 attention head 激活时仍需使用 hook 或 nnsight。

**保底方案：honest_llama 代码参考**

如果 nnsight 和 raw hooks 都有问题，可参考 ITI 的 `honest_llama` 仓库中的 hook 注册模式。但其代码需修改以适配新模型（模型结构路径不同）：

```python
# ITI honest_llama 的核心思路（需改写路径适配 LLaMA-3）
# 原路径 (LLaMA-1): model.model.layers[layer].self_attn.o_proj
# LLaMA-3 路径相同, 但嵌套层级可能不同

def get_activation_hook(layer_idx, storage_dict):
    def hook_fn(module, input, output):
        # output shape: [batch, seq_len, hidden_dim]
        storage_dict[layer_idx] = output.detach().cpu()
    return hook_fn
```

---

## 八、硬件需求估算

### 8.1 Phase 1 可行性验证（单卡 4090）

| 项目 | 估算 | 说明 |
|------|------|------|
| 模型 VRAM | ~6 GB | Llama-3.2-3B FP16 |
| 模型 + 中间激活 VRAM | ~8 GB | forward pass 中间值 |
| 剩余 VRAM | ~16 GB | 可用于存储激活向量 |
| 单条激活大小 | ~86 KB (float32) | 估算: 28层 × 3072维 ≈ 86K × 4 bytes = 344 KB |
| 12,500条激活 | ~4.3 GB | 内存中完全可存 |
| 前向推理时间/条 | ~20 ms | 3B model, short prompt |
| 总提取时间 | ~4 min | 12,500 × 20ms |

### 8.2 Phase 2 完整实验（单/双卡 4090）

| 项目 | 估算 | 说明 |
|------|------|------|
| 模型 VRAM | ~16 GB | Llama-3.1-8B FP16 |
| 模型 + 中间激活 | ~19 GB | 单卡勉强够 |
| Batch 推理优化后 | ~20+ GB | 可能的 VRAM 压力，需考虑梯度检查点或 4-bit 量化 |
| 50,000条激活 | ~22 GB | 双卡 4090 轻松; 单卡需要分批存盘 |
| 前向推理时间/batch | ~100 ms | 8B model, batch=8 |
| 总提取时间 | ~10 min | 50,000 / 8 × 100ms |

---

## 九、实验路线与里程碑

### Milestone 1：环境与数据准备（预计 1-2 天）

```
□ [ ] 安装 nnsight + transformers + torch
□ [ ] 下载 Llama-3.2-3B 到本地
□ [ ] 验证 nnsight + Llama-3.2-3B 联调通过（打印任意层激活 shape）
□ [ ] 下载 MS MARCO 数据集
□ [ ] 运行数据构建脚本, 产出 500 queries 的 (query, passage, label) 三元组
□ [ ] 验证数据集切分正确（train/val/test 无 query 泄露）
```

### Milestone 2：激活提取方案A验证（预计 1-2 天）

```
□ [ ] 实现 extract_activation_last_token()
□ [ ] 对全部 train 数据提取激活（验证速度）
□ [ ] 训练探针, 输出 top-20 注意力头
□ [ ] 评估: 二分类准确率是否显著高于 50%（随机基线）
□ [ ] 消融: top_k heads 数量 vs 准确率
```

### Milestone 3：方案A vs 方案B 对比（预计 1 天）

```
□ [ ] 实现 extract_activation_pooling()
□ [ ] 两个方案用相同 train/val/test 跑完整 pipeline
□ [ ] 输出对比表: accuracy / F1 / ROC-AUC
□ [ ] 确定最终采用的激活提取方案
```

### Milestone 4：端到端 RAG + Cross-Encoder 对比（预计 2 天）

```
□ [ ] 实现 rerank_with_probe() 端到端管线
□ [ ] 实现 cross-encoder baseline
□ [ ] 在 test 上对比 Recall@k
□ [ ] 输出对比表 + 分析（探针 vs cross-encoder 的优劣势场景）
```

### Milestone 5：扩展到 Llama-3.1-8B + Mistral-7B（预计 2-3 天）

```
□ [ ] 在 Llama-3.1-8B 上复现 Milestone 2-4
□ [ ] 在 Mistral-7B 上复现 Milestone 2-4
□ [ ] 跨模型对比: 探针的泛化性分析
□ [ ] 数据规模扩展: 500 → 2000 queries
```

---

## 十、潜在挑战与应对

| 挑战 | 风险等级 | 应对策略 |
|------|---------|---------|
| 假设不成立：注意力头激活无法编码文档相关性 | 中 | Milestone 2 是最早的验证点。若准确率不显著高于随机，换方向——如改为 MLP 层激活或残差流 |
| nnsight 与新模型版本不兼容 | 低 | nnsight 2026年5月仍在更新, 且支持主流模型。若遇兼容问题，降级到 transformers hooks |
| 激活维度高导致内存/过拟合 | 中 | top-k head 选择（参考 ITI）+ L1 正则化；必要时对激活做 PCA 降维 |
| MS MARCO 正负样本不平衡 | 低 | 1:4 (正:负) 比例可控。若探针偏向负类，调整 class_weight |
| 单卡 4090 VRAM 不够存 50K 条激活 | 中 | 分批提取 → 存盘 → 离线训练探针。探针训练不需要 GPU（逻辑回归 CPU 可跑） |

---

## 附录 A：模型 attention head 数量速查

| 模型 | 层数 | 每层头数 | 头维度 | 总头数 | 总激活维度 |
|------|------|---------|--------|--------|-----------|
| Llama-3.2-1B | 16 | 32 | 64 | 512 | 32,768 |
| Llama-3.2-3B | 28 | 24 | 128 | 672 | 86,016 |
| Llama-3.1-8B | 32 | 32 | 128 | 1024 | 131,072 |
| Mistral-7B-v0.3 | 32 | 32 | 128 | 1024 | 131,072 |

## 附录 B：相关文件索引

| 文件 | 路径 | 说明 |
|------|------|------|
| 文献调研报告 | `d:\solo-paper\RAG检索优化文献调研报告.md` | 17篇论文分析 |
| 研究思路总结 | `d:\solo-paper\讨论总结-研究思路梳理.md` | 核心方法论 |
| 本文档 | `d:\solo-paper\实现计划-Attention-Probe-RAG.md` | 实现计划 |