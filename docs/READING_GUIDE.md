# 阅读指引 — 怎么在 15 分钟内接手这个项目

> 建立于 **2026-10-12**。目标读者：**下一个接手的人或 agent**（包括在云容器里继续做实验的你）。
> 这个项目有 6+ 份 Markdown，且**它们不是同一时间写的**，口径会打架。本文告诉你：
> 先读什么、后读什么、每份文档有多可信、进展记在哪、待办在哪。

---

## 一、30 秒版：这是什么项目

**一句话**：LLM 的内部激活编码了「检索文档是否与 query 相关」的信号，但这个信号在传向输出 logits
的过程中**要么被衰减、要么与读出方向错位**，导致模型"嘴上"（直接问它）说不准。
口号：**Internals know more than the model can say.**

**做法**：同一次冻结前向传播上，比较 4 个「相关性读取器」——

| 读取器 | 读哪里 | 训练 |
|---|---|---|
| LLM-judge | 输出 logits（P(yes) vs P(no)） | ❌ 零训练 |
| final / output-repr probe | 最终层 hidden | ✅ |
| best-resid probe | 残差流最优层（val 选层） | ✅ |
| attn-head probe | top-k 注意力头（val 选头）拼一起 | ✅ |

外加：logit-lens（无监督、逐层、走模型自己的 unembedding）、因果 steering（把内部相关性方向注入输出层）、
第二个数据集（FiQA）、更强 judge 上界（4-shot）、第三模型家族（Qwen2.5）。

**规模**：8 个模型（3B–8B，LLaMA / Mistral / Qwen，base + instruct），2 个 in-domain 数据集。

**结论强度（诚实版）**：
- ✅ **最稳**：`attn − final` 在 LLaMA 4/4 与 Qwen 2/2 上单模型显著（p≤0.002）。
- ✅ **较强**：`best_resid − judge` 在 base 模型上单模型显著；跨模型符号检验 6/6（p=0.016）。
- ⚠️ **要小心**：`best_resid − final`（**唯一与 final 同维度、同正则的干净对照**）只在
  **base 模型**显著；instruct 上不显著且 edge 缩小。论文把它讲成"对齐把信号推进了输出表示"的
  机制证据——这个解读是自洽的，但它同时也是**全文最大的逻辑软肋**（见 `TODO.md` R-1）。
- ⚠️ **注意**：`attn − final` 这个头条主张**不是 capacity-matched** 的对照（见 `TODO.md` R-1）。
- 🚫 **本项目不声称**：探针打赢 reranker、跨域鲁棒、数据效率高——这三条都已被证伪（见第四节）。

---

## 二、阅读顺序

### 路线 A — 只想搞懂「这项目在干什么、做到哪了」（约 15 分钟）

1. **`README.md`** — 一页摘要。⚠️ 里面的 AUC 数字是 **P1 修复前的旧口径**（见 `TODO.md` P1-1），
   当作"项目定位"读，别当数字读。
2. **`CLAUDE.md` §〇 + §三** — 一句话定位 + 核心论点三段式（存在性 / 归因 / 机制）。
   §3.2 顶部有 ⚠️ 加注说明哪些数字是旧口径。
3. **`CLAUDE.md` §六** — **项目进展的唯一权威流水账**。读法见本文第四节。
4. **`TODO.md`** — 还没做什么、已知什么问题、优先级。

### 路线 B — 要改论文

1. 上面路线 A 全部。
2. **`paper/main.tex`**（424 行，8 页）——正文。
3. **`.claude/skills/write-paper/SKILL.md`** — 写作方法论 + 诚实铁律 + 取数速查表。
   ⚠️ **这份已过时**（还写着 6 模型、因果限 3B、单数据集，且指向一个不存在的
   `results/multimodel_summary.json`）。当"方法论"读，别当"当前状态"读。见 `TODO.md` P1-5。
4. **`TODO.md` 的 P0 段** —— 三个投稿前必须处理的论文问题（图表口径不一致、Table 1 混样本量、
   正文引用不存在的 supplement）。
5. **`results/REPORT.md`** —— 需要某个具体数字的出处时查它。

### 路线 C — 要跑实验 / 重算 / 重画图

1. 路线 A 全部。
2. **`docs/CACHE_NAMING.md`** —— **先读这个**。缓存目录命名规范 + 旧名对照表。
   不知道缓存叫什么，后面的脚本都会找不到文件。
3. **`CLAUDE.md` §五**（脚本说明表）+ **§七**（环境与坑，含"哪些脚本纯 CPU 可跑"）。
4. **`TODO.md` P2** —— 哪些结果文件已经丢了、怎么从 GPU 主机打包取回。
5. `scripts/run_model.sh` / `run_scaleup.sh` / `run_fiqa.sh` —— 三条驱动管线，照抄即可。

### 路线 D — 要了解历史上走过哪些弯路（避免重走）

1. **`CLAUDE.md` §四** —— 废弃方案清单（探针打 reranker / 跨域 / 数据效率，全部证伪）。
   **必读**，否则很容易重新发明一条已经死掉的路。
2. **`HANDOVER.md`** —— 2026-06-30 的完整交接文档。⚠️ **是历史快照**：数字是旧口径，
   文件树里的 `src/common.py`、`paper/tools/tectonic`、`.claude/memory` **都不存在了**，
   还声称数据来自 "MS MARCO dev-small (6980 queries, qrels)"，而代码实际加载的是
   `split="train"` 并用 `is_selected` 造标签。见 `TODO.md` P1-3 / P1-6。
3. **`docs/RAG检索优化文献调研报告.md`** —— ⚠️ 写于 **2026-05-18**，描述的是**转向之前的**定位
   （"探针做 reranking/filtering，替代外部 reranker"）。这正是后来被证伪的那条路。
   把它当**文献综述**读（17 篇论文的分类整理仍然有用），**不要**当项目当前定位读。

---

## 三、文档可信度速查表

口径会打架，这张表说明每份文档的**时效性**与**该拿它当什么用**：

| 文档 | 生成/更新 | 时效性 | 当什么用 |
|---|---|---|---|
| `README.md` | 早期 | ⚠️ 数字旧 | 项目定位（数字别信） |
| `CLAUDE.md` §〇–§五 | 早期，2026-10-12 加注 | ⚠️ §3.1 数字旧 | 论点、脚本说明、废弃方案 |
| `CLAUDE.md` §六 | 持续更新到 2026-10-12 | ✅ **最新** | **进展权威流水账** |
| `CLAUDE.md` §七 | 2026-10-12 更新 | ✅ 最新 | 环境、坑、能力边界 |
| `HANDOVER.md` | 2026-06-30 | ❌ 历史快照 | 历史背景、文件结构意图 |
| `results/REPORT.md` | 到 2026-06-30 | ✅ 数字可用 | **实验日志与数字出处** |
| `TODO.md` | 2026-10-12 | ✅ 最新 | 待办与已知问题 |
| `docs/CACHE_NAMING.md` | 2026-10-12 | ✅ 最新 | 缓存命名规范 |
| `.claude/skills/write-paper/SKILL.md` | 早期 | ⚠️ 已过时 | 写作方法论 |
| `docs/RAG检索优化文献调研报告.md` | 2026-05-18 | ❌ 转向前的 | 文献综述 |

**数字的权威顺序**（冲突时以此为准）：
`results/*.json`（原始数据） > `results/REPORT.md` > `CLAUDE.md` §六 > 论文表格 > `README.md` / `HANDOVER.md`。

---

## 四、项目进展怎么读（`CLAUDE.md` §六 的体例）

§六 是**流水账**，不是摘要。按时间顺序记了 **7 次推进**，每次一个引用块。读的时候注意三件事：

1. **标了日期和"第 N 次推进"**，越靠后越新。最新的是 **2026-10-12 第 7 次推进**（在 §六 末尾附近）。
2. **🔴 / 🟢 / ⚠️ / 🔍 这类符号是"当时的更正"**，不是装饰：
   - 🔴 = 发现之前的结论**过于乐观**，必须收紧（例：2026-06-28 补做 bootstrap 后，
     `internal_edge` 从"6/6 通过"降级为"单模型不显著"）。
   - 🟢 = 后续实验把结论**重新加固**（例：扩样本到 1500q 后 base 模型恢复显著）。
   - ⚠️ = 跨架构的诚实记录（Mistral 的 attn-head 弱是真实差异，不是 bug）。
   → **同一个数字在不同段落可能不同，那是口径变化（500q → 1500q、点估计 → val 选层），
   不是笔误。** 看到冲突先查它是哪一次推进的。
3. **末尾那份"下一步"编号清单已部分过时**（第 1 条标着"进行中"其实早完成了；
   2026-10-12 已修正，但整段仍建议以 TODO.md 为准）。

**实验日志在 `results/REPORT.md`**，§1–§15 编号固定，速查：

| 节 | 内容 |
|---|---|
| §1–§2 | 实验设计、6 模型主表（点估计） |
| §3–§4 | 命门检验、base↔instruct 机制对照 |
| §5 | 信号的深度定位（论文主图） |
| §9 | **显著性（500q，val 选层）** ← 论文 Table 2 的出处 |
| §10 | **扩样本 1500q 加固** ← 论文 Table 3 的出处 |
| §11 / §11.1 | 因果 steering（3B / 8B base+instruct） |
| §12 | FiQA 第二数据集 |
| §13 | few-shot judge 上界 |
| §14 | logit-lens 无监督基线 |
| §15 | Qwen2.5 第三家族 |

---

## 五、待办

**统一看 `TODO.md`**，按优先级分档：

- **S** 安全问题（用户已决定暂缓，但**轮换代理口令仍是最小成本的正解**）
- **P0** 投稿前必须：3 个论文层面的问题
- **P1** 文档与事实不符（README/SKILL/HANDOVER/requirements）
- **P2** 数据缺口（6 个结果 JSON + 7 张图，只能回 GPU 主机取；含打包命令）
- **P3** 工程（LaTeX 工具链、vendored nnsight、结果文件命名遗留）
- **P4** 论文可选加分项
- **R** 本次通读发现的**研究逻辑层面**的问题（比 bug 更值得先想清楚）

---

## 六、在云容器里从零开始

```bash
git clone https://github.com/new-life-fch/my_paper_project.git
cd my_paper_project && git checkout paper-A

# 1. 环境
bash scripts/setup_env.sh            # 注意：脚本里 pip install nnsight，仓库里另有一份
                                     # 未被 import 的 nnsight/ 副本（321 文件，见 TODO P3-2）
export HF_HOME=/root/shared-nvme/hf_cache   # 系统盘只有 30G，务必改 HF 缓存位置

# 2. 代理（凭据不要写进仓库！）
export https_proxy="http://<user>:<pass>@<proxy-host>:<port>"

# 3. 先确认手上有什么
python -c "import json,glob; [print(p, json.load(open(p))['model'], json.load(open(p))['n_queries']) \
  for p in sorted(glob.glob('results/cache/*/meta.json'))]"
```

**关键认知：仓库里没有激活缓存**（`.pt` 是 gitignore 的，每模型 1–3 GB）。
所以 clone 下来只能读代码和文档，**任何"重算/重画"都需要先把缓存从原主机拷过来**。
详见 `TODO.md` P2 与 `docs/CACHE_NAMING.md`。

- ✅ 纯 CPU 可跑（只要有缓存）：`plot_layer_signal.py`、`internals_vs_output.py`、`significance.py`
- ❌ 必须 GPU：`extract_instruct.py`、`llm_judge*.py`、`causal_steer.py`、`logit_lens.py`

---

## 七、别在这些地方浪费时间

1. **不要试图让探针在重排精度上打赢 reranker** —— 已系统证伪（`CLAUDE.md` §四第 1 条）。
2. **不要做跨域迁移** —— 属 `paper-B` 分支；A 分支的 FiQA 实验是**同域**重训重测，不是跨域。
3. **不要引用 0.964 这个数**（早期 SciFact 假结论，对比不公平）。
4. **不要以为 clone 能拿回丢失的图/JSON** —— 实测过了，它们从未进过 git。
5. **不要在没读 `docs/CACHE_NAMING.md` 的情况下猜缓存目录名** —— 历史上同一个 500q 缓存
   在不同模型上叫法不同，`mistral_7b_instruct` 这种旧名甚至无法从名字判断它属于哪个模型；
   以 `meta.json` 的 `model` 字段为准。
6. **不要相信 `internals_vs_output_*.json` 里的 `resid_best_layer`** —— 它是**在 test 上**
   取 argmax 得到的（`internals_vs_output.py:78`），有乐观偏差。以 `significance*.json`
   的 val 选层值为准。见 `TODO.md` R-2。
