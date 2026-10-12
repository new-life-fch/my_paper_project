# TODO — 待办与已知问题清单

> 建立于 **2026-10-12**，同日通读全项目后大幅扩充。
> 本文件只记录**尚未处理**的事项，按优先级排列；处理完改成 ✅ 并注明日期。
> 项目主线结论与实验状态看 `CLAUDE.md` §6 与 `results/REPORT.md`；
> **接手时先读 `docs/READING_GUIDE.md`**（阅读顺序 + 每份文档的时效性）。
>
> 分档：**S** 安全 / **P0** 投稿前必须 / **P1** 文档失真 / **P2** 数据缺口 /
> **P3** 工程 / **P4** 可选加分项 / **R** 研究逻辑（比 bug 更该先想清楚）

---

## S — 安全（用户已决定暂缓）

### S-1. 公开仓库曾泄露真实代理口令 🔴

- **事实**：`run_experiment.sh`、`scripts/download_model.sh`、`CLAUDE.md` §七、`HANDOVER.md`
  曾硬编码 `http://<user>:<pass>@10.255.128.102:3128`。仓库 `private: false`（**公开**），
  该口令出现在 **8 个提交**里，最早可追到 `init`。
- **已做**（2026-10-12，commit `7a7b634`）：4 个文件的工作区内容改为从环境注入；
  `git grep` 确认工作区无残留。
- **未做**：**git 历史里仍在**。
- **最小成本的正解**：**轮换该代理口令**。换了之后历史里那串自动失效，不必动历史。
- **风险量级（诚实评估）**：代理是 `10.255.x.x` 私有地址，仅在那张内网内可用，危害有限；
  但公网仓库的泄露主要面对**自动扫描机器人**，不是人工查看。
- **副作用**：`download_model.sh` / `run_experiment.sh` 现在**要求先 `export https_proxy`**，否则 `exit 1`。

---

## P0 — 投稿前必须处理（影响论文正确性）

### P0-1. Fig. 1 与 Table 1 不一致，且有**两个**混淆原因（不只是选层口径）

- **现象**：`results/figures/llama32_3b_layer_signal.png` 标 **peak L12 = 0.632**；
  `main.tex` Table 1 的同一格是 **best-resid 0.628**。
- **原因一（已知）**：图用 **test 曲线 argmax** 选峰（`plot_layer_signal.py` 的 `resid_peak`
  对 `resid_auc` 取 `np.argmax`，而 `resid_auc` 是 test 上的逐层 AUC）→ selection-on-test；
  Table 1 用 val 选层。
- **原因二（本次通读新发现）**：**两者的正则强度不同**。
  `plot_layer_signal.py` 的逐层残差探针用 **`lr(C=0.5)`**；
  而 `significance.py`（Table 1/3 的数字来源）用 **`lr()` 即 C=1.0**。
  → 所以 0.632 与 0.628 的差是**选层口径 + 正则强度**两个因素叠加，不能只归因于前者。
- **怎么修**：把 `plot_layer_signal.py` 改成就地用 val 选层、并把 C 对齐到 1.0，再重跑。
  **需要 `results/cache/llama32_3b_q500_instruct` + `..._judge`**（缓存拷到本机后是**纯 CPU** 的活）。
- **附带**：`internals_vs_output.py:78` 的 `resid_best_layer` 也是 test-argmax，
  所以仓库里那 3 个 `internals_vs_output_*.json` 的 `resid_best_layer` 都带乐观偏差（见 R-2）。

### P0-2. Table 1 混用了两个样本量，且与 Table 3 自相矛盾

- **现象（已逐格核对）**：Table 1 八行**与各自的来源 JSON 完全吻合**，但
  **前 6 行来自 `significance.json`（500q）**，**Qwen 两行来自 `significance_qwen_q1500.json`（1500q）**。
  caption 只说 "test AUC, point estimates; val-selected layers and heads"，**没交代样本量不同**。
- **与 Table 3 冲突**：Table 3 用 1500q 报 LLaMA-3.2-3B base 的 final 0.536 / best-resid 0.614，
  Table 1 同一行是 0.592 / 0.628。同一模型同一 reader 两套数。
- **怎么修**（都不需要跑实验）：
  - **A（推荐）**：Table 1 统一到 1500q。LLaMA 4 个 + Qwen 2 个已有 1500q
    （`significance_q1500.json` + `significance_qwen_q1500.json`），**只缺 Mistral 两个**
    （需回主机补跑，或在行内标注 Mistral 为 500q）。
  - **B（最省事）**：保留现状，caption 写明 "LLaMA/Mistral rows: 500 queries; Qwen rows: 1500 queries"。

### P0-3. 正文引用了不存在的 supplement

- `main.tex` Experiments 的 Depth localization 段写
  "Figure~\ref{fig:depth} (**per model in the supplement**) shows ..."，
  但论文**没有 appendix / supplement 章节**，也没有任何 per-model 深度图。
- **怎么修**：删掉该括号，或真补 appendix（那些图见 P2-3，重画需缓存）。
- **注意**：本地若无 LaTeX 工具链（P3-1），改完 `main.tex` 无法编译验证。

### P0-4. 🔴 论文 Table 2 / Table 3 / Abstract 的 CI 与 p 值**无法从已提交的结果文件复现**

> **本次通读最重要的发现**。投稿前必须解决：审稿人或 AC 一旦索要 artifact 就会撞上。

- **证据（逐格核对）**：
  - **Table 2（500q）**：6 个模型 × 3 条 edge 的 CI，**几乎每一格都比
    `results/significance.json` 更宽**。例：
    - Mistral-7B `internal_edge`：论文 `[-.07,.10]`，JSON `[-0.046, 0.081]`
    - LLaMA-3.2-3B `attn−final`：论文 `[.01,.18]`，JSON `[0.012, 0.169]`
    - LLaMA-3.1-8B `internal_edge`：论文 `[-.06,.11]`，JSON `[-0.058, 0.117]`
  - **Table 3（1500q）**：`attn−final` 各行全部吻合，但 **`internal_edge` 有 3 行不吻合**：
    - 3B：论文 `p=0.0008`，JSON `p_le0 = 0.0`（即 <2e-4）；CI 下界论文 `.03` vs JSON `0.035`（应四舍五入为 .04）
    - 3B-Inst：论文 `p=0.15`，JSON `0.112`
    - 8B-Inst：论文 `p=0.32`，JSON `0.2854`；CI 下界论文 `-.04` vs JSON `-0.031`
  - **`REPORT.md` §9/§10 与 `CLAUDE.md` §6 的同一批数字**（如 3B `+0.036 [-0.045,0.117] p=0.193`）
    **也与 JSON 不符**，而与**论文**一致 —— 说明论文表格是从 REPORT 誊抄的，而 REPORT 没跟着 JSON 更新。
- **为什么这不可能是"bootstrap 随机性"**：`significance.py:128` 是
  `rng = np.random.default_rng(42)` —— **bootstrap 种子写死**。同一份缓存 + 同一份代码
  必然给出逐位相同的结果。所以论文与 JSON 不可能同时正确。
- **根因（fact-check 给出的结构性解释，比"没同步"更精确）**：这个仓库里**同时存在两代 bootstrap 产物**，
  而论文**跨在两代之间**：

  | 代 | bootstrap 方式 | 产生于 | 现在这些数字住在哪 |
  |---|---|---|---|
  | **Gen-1（已被取代）** | **非配对**，每个 reader 各自重采样 | `0f58593`(500q) / `3476b90`(1500q) | git 历史 + `REPORT.md` §9/§10 的表格 |
  | **Gen-2（现行 artifact）** | **配对**，共享重采样索引 | `4c8be8a`(500q) / `25d878b`(1500q) | `results/significance.json`、`significance_q1500.json` |

  **论文 Table 2 精确复现 Gen-1（18/18 格）；Table 3 的 LLaMA 四行也精确复现 Gen-1；
  只有两个 Qwen 行（2026-06-30 才写）复现 Gen-2。** 也就是说论文的 LLaMA 列
  **无法从 `results/` 里任何现存文件复现**——只能从 git 历史或 REPORT 复现。
  这和 `CLAUDE.md` §6 里"改成配对后**结论完全不变**"的说法也对不上：配对确实改了 CI 与 p。
- **方向是良性的**：论文的数字比 artifact **更保守**（CI 更宽、p 更大），
  这不像"夸大结果"，更像"引用了旧版本"。但**必须解决**。
- **怎么修（2026-10-12 更正：比原先设想的便宜得多）**：
  **改论文，不需要重跑。** 独立审计用 git diff 证明：`4c8be8a`/`25d878b` 那两个提交里
  **被替换掉的旧值就是论文现在写的值**（逐格吻合），而 `results/` 里的是修正后的配对版本。
  所以仓库里的 JSON **才是对的**，论文引的是旧版。
  → 只要把 `main.tex` Table 2/3 + Abstract 的 CI/p **改成引用 `results/significance*.json`**，
  再同步 `REPORT.md` §9/§10 与 `CLAUDE.md` §6 即可。**不需要 GPU、不需要缓存。**
  （这一点现在尤其重要：**原主机已被回收**，见 P2，所以"重跑后再对齐"这条路已经走不通了。）
  - 顺带：改完之后 **加粗/显著性判定不会翻转**（旧值更保守），所以论文的结论不受影响。
  - 但有一个边界格子会变：Mistral-7B-Inst 的 `attn−final` 由 `p=0.1476` → `0.0526`，
    仍不显著，但已贴到 0.05 边缘，值得在文中留意。
- **独立复核（一次专项 fact-check，逐格比对 main.tex 与 results/）的结论**：
  - `Table 2`：**18 个 edge 格子，0 个与现行 artifact 吻合**；全部 18 个都精确复现
    **被取代的旧版本**（13 个在论文精度下逐位相同，5 个仅末位差 1）。
  - `Table 3`：6 行里 **2 行吻合**（两个 Qwen，2026-06-30 才写的），**4 个 LLaMA 行不吻合**。
  - `Table 1`：**42/42 全部吻合** ✓（问题只在 Table 2/3 与 Abstract 的 p 值）。
  - 论文 Table 3 的 **500q 列**也一起旧了：3B-Inst 论文 `p=0.28` vs 文件 `0.2254`；
    8B-Inst 论文 `p=0.16` vs 文件 `0.0742`。
  - **误差方向是混合的，别一句"更保守"带过**：
    1500q base 的 `internal_edge` 上论文**低估**了显著性（报 0.0008，实际 <2e-4）；
    但 500q instruct 列上论文**高估**了 p（报 0.16，实际 0.074）。
    所以这不能简单说成"论文更保守就没事"——**必须逐格重算重对齐**。
  - `sign test 6/6 (p=.016)` 一行与加粗判定在两种版本下都成立 ✓。
  - 另有一个**边界格子会因重算而翻转**：Mistral-7B-Inst 的 `attn−final`，
    论文（旧版）`p=0.1476` 未加粗，现行 artifact `p=0.0526` —— 仍不显著，但已贴到 0.05 边缘。
- **附带（已核对无问题）**：FiQA 的 `resid_best_layer=0.993` 与 Abstract 的
  "0.993 / +0.253" **吻合**；Abstract 的因果、logit-lens 数字也**全部吻合**。
  本问题集中在 bootstrap 那两张表。

---

### P0-5. 正文自相矛盾："internal edge 在 LLaMA 上变宽" vs "在 instruct 上收缩"

- `main.tex` L325（Mechanism 段）写 "the internal edge persists (**and widens on LLaMA**)"，
  而 L244–246（Scaling 段）写 "**shrinks and loses significance on instruct models**"。
- **实测两者不能同真**：现行 artifact 下，1500q 的 `internal_edge` 在 LLaMA **两对都变窄**
  （3B `0.0779 → 0.0249`；8B `0.0601 → 0.0129`）；500q 下 3B 也是变窄（`0.0351 → 0.0269`），
  只有 8B 变宽（`0.0289 → 0.0436`）。
- **来源**：这句"widens"是从 `REPORT.md` §4 抄来的，而 §4 用的是**已被取代的 v1（test-argmax）数字**。
  这是"论文从旧 REPORT 誊抄"的又一例证，与 P0-4 同源。
- **怎么修**：改成"internal edge 在 base 上显著、在 instruct 上收缩"，并在重算后统一口径（与 P0-4 一起做）。

---

### P0-6. `logit_lens.py` 也在 test 上选层 —— 而论文**明确否认**了这件事 🔴

- `scripts/logit_lens.py:112`：`best = int(np.argmax(layer_auc))`，而 `layer_auc` 是在
  **被评估的 split**（`--split`，默认 `test`）上算的 → 又一次 selection-on-test。
  论文引的峰值（3B `0.560@L25`、8B `0.577@L30`）就出自这一行。
- **严重性在于论文的相反声明**：`main.tex:144-147` 写 logit lens
  "**no training and no label-based layer selection**"，`CLAUDE.md` §6 也重复
  "零训练**无选层**"。**这两句按字面都是错的**——零训练✔，但**有**基于标签的选层。
  这是"论文与自己的代码矛盾"，审稿人查代码就能发现。
- 同一类问题还有两处（全部已加注释标记，避免未来 Agent 当成新 bug 各自"修"）：
  - `internals_vs_output.py` 的 `resid_best_layer`（test-argmax）→ 仓库里 3 个
    `internals_vs_output_*.json` 的该字段都带乐观偏差（R-2）。
  - `aggregate_models.py` 的 `internal_edge = max(best_resid_test, attn_test) − final`，
    **与论文的 `best_resid − final` 不是同一个量**（3B：+0.093 vs +0.036）。
    `multimodel_summary.json` 的 `intEdge` 列因此不可与 Table 2 对比。
- **另外，被测的量本身噪声很大**：同一个 3B 模型，val 选层在 500q 是 **L13**、
  在 1500q 是 **L9**（差 4 层）。所以 ±1 层的差异没有意义，但 +0.01~0.02 的 AUC 乐观偏差是系统性的。
- **怎么修**（三处一起改，并同步论文措辞）：改成 val 选层；`aggregate_models.py` 的
  `internal_edge` 要么改名、要么改成论文的定义。**改代码会变动论文数字，必须同一次改完。**

### P0-7. 因果 steering 的**随机对照表现和实验组一样** 🔴 这条动摇论文的因果论断

- 论文（`main.tex:40-44`、`336-341`）与 `REPORT.md` §11 的核心对照是：
  "唯独 internal 方向**既**改变判断**又**保住 AUC；final 改判断但摧毁 AUC；
  **random 不产生定向效应**"，据此得出"输出通路本可表达该信号，只是被衰减"。
- **但 artifact 自己不支持"random 不产生定向效应"**。`results/causal_steer_llama32_3b_q1500.json`：

  | 方向 | mean P(yes) 摆动 | AUC 范围 |
  |---|---|---|
  | internal | 0.793 → 0.260（**0.53**），随 α 单调 | 0.541–0.553（平） |
  | **random** | 0.789 → 0.363（**0.43**），**同样随 α 单调** | 0.540–0.546（**也平**） |
  | final | 0.793 → 0.995/0.010 | 崩到 0.485 |

  也就是说 **random 方向同样大幅改变判断、同样保住 AUC**，它与 internal 的唯一差别是
  internal 在 α>0 时 AUC +0.007、random −0.006 —— **远在 ±0.05 的 CI 之内**。
- **根因**：`causal_steer.py` 只记录了 mean P(yes) 与 AUC，**没有按真实标签分组**，
  所以"relevance-aligned"（相关性对齐的摆动）**在这个 artifact 里根本无法测量**。
  又只用**单个**随机方向（`default_rng(0)`），无法排除"任何方向都能这样"。
- **怎么修**：①在 `causal_steer.py` 里按标签分组记录 `mean P(yes) | relevant` 与 `| irrelevant`
  per α（这样"定向"才可测）；②用 **多个**随机方向（如 ≥10 个）给出随机基线分布；
  ③在那之前，论文的因果措辞应收紧到"消融臂单调崩塌"这一条（它仍然成立），
  去掉"random 无定向效应"和"唯独 internal 两者兼具"这两个过强表述。

---

## P1 — 文档与事实不符

### P1-1. `README.md` 数字是 P1 修复前的旧口径

第 5–10 行 `0.593`（final）、`0.632`（L12 最强）是 **test-argmax** 口径；
现行 val 选层应为 **0.592 / 0.628 (L13)**。且只给了 3B 单模型数字，没提 8 模型规模。

### P1-2. `CLAUDE.md` §3.1 仍是旧口径

§3.1 三段式论点里的 `0.593` / `0.632` / `L12` / 「训练贡献 +0.058 / 读内部 +0.092」都是旧数。
§3.2 顶部已加 ⚠️ 加注，**但 §3.1 正文尚未改**。

### P1-3. `HANDOVER.md` 有**事实性错误**，不只是数字旧

除数字口径旧（test-argmax）外，本次核对发现它描述的结构与代码不符：

| HANDOVER 声称 | 实际 |
|---|---|
| `src/common.py` | **不存在** |
| `load_ms_marco_dev_small(n_queries, seed, max_passages)` | 实际函数名是 **`load_ms_marco`** |
| 数据来自 "MS MARCO **dev-small** (6980 queries, **qrels**)" | 代码加载 **`split="train"`**，用 **`is_selected`** 造标签，**不用 qrels** |
| `.claude/memory/` | **不存在** |
| `paper/tools/tectonic` | **不存在** |
| `results/multimodel_summary.json` | **不存在**（`aggregate_models.py` 的产物从未提交） |

- **建议**：在文件开头加一条"历史快照"横幅，逐条列出上述失效引用。
  HANDOVER 生成于 2026-06-30，作为历史文档保留合理，但**不能让它继续冒充当前结构**。

### P1-4. `.claude/skills/write-paper/SKILL.md` 已过时，会误导下一个写作 agent

- 写"**6 模型**主表" → 实际 **8 模型**。
- 写"局限：**因果限 3B**" → 实际已复现 3 个变体（3B / 8B base / 8B instruct）。
- 写"**in-domain 单数据集**" → 实际 2 个（MS MARCO + FiQA）。
- 取数速查表指向 **`results/multimodel_summary.json`**（不存在）。
- 缺本次新增的三块证据：logit-lens（REPORT §14）、Qwen 第三家族（§15）、few-shot judge（§13）。
- 结尾写"figures 按需 `-f`" → 2026-10-12 起 `.gitignore` 已收窄，图**默认入库**。
- 编译段用硬编码主机路径 `/root/shared-nvme/paper_A_internals/paper`。
- **这份文档的作用是"保证只写已验证的数字"，它本身失准会直接导致论文引用错数**，优先级不低。

### P1-5. 同一个量在论文里出现**两个不同值**（未披露样本量切换的直接后果）

这是 P0-2 的"可见症状"——即使不重算数字，**审稿人现在就能看到同一模型同一量有两个值**：

| 量 | Table 1（500q） | 正文别处（1500q） | 位置 |
|---|---|---|---|
| judge, LLaMA-3.1-8B | `0.514` | `0.574` | logit-lens 段 L302 |
| judge, LLaMA-3.2-3B | `0.535` | `0.545` | L302 与 few-shot 段 L385 |

- logit-lens 那段的 sanity check 说"末层 logit-lens 复现 judge 到 <0.001"，
  用的是 **1500q** 的 `0.545 / 0.574`；而读者刚在 Table 1 看到同两个模型的 judge 是 `0.535 / 0.514`。
- **怎么修**：统一 Table 1 到 1500q（即 P0-2 的选项 A），或在两处都写明 query 数。

### P1-6. "除 Qwen 外每个模型都超过零样本 judge" 对 Qwen-base 不成立

- L175–176 写 "on every model except the Qwen pair it also exceeds the zero-shot judge"。
- **实测**：Qwen2.5-7B **base** 的 best-resid `0.655 > judge 0.652`、attn `0.682 > 0.652`，
  **两者都超过 judge**；只有 Qwen-**Instruct** 的 best-resid `0.630 < judge 0.660`。
- "except the Qwen pair" 过度概括，把一个成立的结论说弱了（对本文不利）。
- **怎么修**：改成 "except Qwen2.5-7B-Instruct"。

### P1-7. Limitations 的 "attn 在 Mistral 上弱" 只对 Mistral-**base** 成立

- L406–408（与 Intro L89–90）写 attn reader "weak on Mistral"。
- **实测**：Mistral-7B-v0.3 base 确实弱（attn `0.603` < best-resid `0.621`）；
  但 **Mistral-7B-Instruct-v0.3 上 attn `0.667` > best-resid `0.628`，attn 就是最强的内部读取器**。
- **怎么修**：限定为 "Mistral-7B-v0.3 (base)"。

### P1-8. "judge rising from 0.52 to 0.60" 无精确出处

- L283 的 `0.52 → 0.60` 与任何单模型对或均值都不精确吻合
  （最近的：8B `0.514→0.603`；三对均值 `0.527→0.607`）。
- **怎么修**：改成具体某一对（如 8B 的 `0.51→0.60`）或写"三对均值 0.53→0.61"。

### P1-9. FiQA 的 final-layer AUC 在相邻两句里是 `0.976` 与 `0.977`

- L374 用 `0.976`（来自 `significance_fiqa.json`），L376 用 `0.977`
  （来自 `internals_vs_output_llama32_3b_fiqa.json`）。**两个文件对同一个量不一致**，
  根因还是 val 选层 vs test-argmax 两套口径（同 P0-1）。
- **怎么修**：统一取 `significance_fiqa.json` 的值（val 选层，"L14 0.993 vs L28 0.977"这句要一起改）。

### P1-10. 层号约定不统一：同一输出层被写成 `L28` 和 `L27`

- L376 写 FiQA "L14 0.993 vs **L28** 0.977" —— 28 是 **1-based**（28 层模型的最后一层）；
  而 Fig. 1 把同一个输出层标成 **"final L27"** —— **0-based 索引**。
- 正文其它地方（如 "L12"、"L10–16"）用的是 0-based。
- **怎么修**：全文统一为 0-based（或统一 1-based），并在第一次出现时说明约定。

### P1-11. "four flagship models at 1500 queries" 与同表的 2 个 Qwen 1500q 行不符

- L240 说 1500q 只跑了四个旗舰模型，但 Table 3 里还有两个 Qwen 的 1500q 行。
- **怎么修**：改为 "four LLaMA models (plus the two Qwen models, below)"。

### P1-12. 论文从未给出 `top-$k$` heads 的 `k`

- Method 写 "top-$k$ heads chosen on validation"，但**全文没给 k 的值**；
  代码里是硬编码 **20**（`significance.py` 的 `sel = ... [:20]`，`internals_vs_output.py --topk` 默认 20）。
- 属于可复现性缺口（不是数字错误）。**怎么修**：在 Method 里写明 k=20。

### P1-13. `requirements.txt` 与实际依赖不符

- 缺 **`nnsight`**（`src/activations.py` 直接 import；由 `setup_env.sh` 单独 `pip install`）。
- 缺 **`joblib`**（`plot_layer_signal.py`、`significance.py` 用 `joblib.Parallel`）。
- 缺 **`pypdf`**、**`pillow`**（新增的 `scripts/extract_figures_from_pdf.py` 需要）。

### P1-14. `scripts/setup_env.sh` 有几处问题

- `pip install nnsight` 与仓库里那份 vendored `nnsight/` 并存且无说明（见 P3-2；已核实**不会**遮蔽）。
- `torch.cuda.get_device_properties(0).total_mem` —— **疑似属性名错误**（现代 PyTorch 是
  `total_memory`）。⚠️ **未能验证**（本机无 GPU），需在有卡的机器上跑一次确认；
  若确实错误，会因 `set -e` 让脚本在 `[3/4]` 步直接失败。
- 提示用 `huggingface-cli login`，与 `CLAUDE.md` §2「`huggingface-cli` 在本环境损坏，
  请用 Python `huggingface_hub.login()`」**直接矛盾**。
- 没有设置 `HF_HOME`（30G 系统盘的坑只在 `run_*.sh` 里防了，setup 阶段没防）。

---

### P1-15. 两个缓存的"对齐"只是口头约定，唯一那句断言抓不到真正的失效

- 全仓库**只有** `significance.py` 有一句跨缓存校验：
  `assert np.array_equal(jd["labels"], yte)`（只看 test、只比 labels）。
  `internals_vs_output.py`、`logit_lens.py`、`plot_layer_signal.py` **一句都没有** ——
  judge 的分数与激活缓存的标签是**纯按位置**拼起来的。
- **并且那句断言抓不到最现实的失效**。审计实测：用真实 `src/data.py`、同样的 10 个段落但
  **把 query 内顺序打乱**，结果 **labels 完全相同、(query,passage) 对完全不同**，
  断言照样通过。因为 `load_ms_marco` 是先 append 正例再 append 负例、再取 `negatives[:n_neg]`，
  所以只要 HF 数据集 revision / `trust_remote_code` 脚本 / `--max-passages` / `--seed` 任一改变，
  就会得到"标签一样、文本不同"的两个缓存 → judge 与探针**静默错配**，
  而 `train_edge`、`best_resid−judge` 以及整张 4-reader 表都会失效。
- **好消息**：两个缓存其实都存了 `query_ids`（`extract_instruct.py`、`llm_judge.py`），
  `significance.py` 甚至已经 load 了它，只是**从没拿来比对**。
- **怎么修**：在**每个**同时消费两种缓存的脚本里，加一句按 `query_id` + label 的 join 校验
  （不只是 labels）：`assert np.array_equal(jd["query_ids"], qte)`。
  一行代码，消除一类静默失效。对 FiQA 尤其重要——见 P1-16。

### P1-16. BeIR 的 query id 哈希依赖进程 → 两个脚本可能造出**不同的切分**

- `src/data.py`：`except ValueError: qid_num = abs(hash(q)) % 10**9`。
  Python 的 `hash(str)` 受 `PYTHONHASHSEED` 随机化，审计实测**4 个进程给出 4 组不同的 id
  与 4 套不同的 query→split 分配**。
- 而 `extract_instruct.py` 与 `llm_judge.py` 是**两个独立进程**，所以 FiQA 实验可能让
  judge 与探针在不同的 query 集合上训练/测试；`run_fiqa.sh` 调的是
  `internals_vs_output.py`，而它**没有对齐断言**，不会发现。
- 目前 FiQA 的 id 都是数值、走 `int(q)` 分支，所以是**潜在**地雷（未能联网确认全部 id 都是数值）。
- **怎么修**：把回退分支换成稳定的哈希（`hashlib` + 固定盐），或直接改成顺序编号；
  并顺手加上 P1-15 的 query_id 校验。

### P1-17. bootstrap 的随机流是**跨模型共享**的 → `--only` 会改变其它模型的 CI

- `significance.py` 的 `rng = np.random.default_rng(42)` 建在**模型循环之外**、在循环内被消耗。
  被 `--only` 跳过、或缓存缺失而 `continue` 的模型**不消耗**随机数。
  → 所以 `--only X`、或某台机器少一个缓存目录，都会**改变其后所有模型的 CI 与 p**。
  再加上 `default_rng` 的流本身不保证跨 numpy 版本稳定，**已发表的 CI 不是逐模型可复现的**。
- JSON 里原先也没有 `n_test`/`n_boot`/缓存路径/版本 —— 这正是 P0-2（Table 1 混样本量）
  **无法从 artifact 审计**的原因。
- **已部分修复（2026-10-12）**：`significance.py` 现在会记录
  `n_test / n_boot / cache / judge_cache / env`。
- **仍需修**：改成按模型派生子流（`SeedSequence(42).spawn(...)`），让每个模型的 CI
  与"跑了哪些模型"无关。

### P1-18. `compare_probes.py` 用未分组的 StratifiedKFold → train 内部按段落泄露

- `per_head_cv_auc` 与 M4 的 stacking OOF 都在**段落层面**切折，于是"留出折"里的样本，
  其同 query 的兄弟段落进了训练集。`load_split` 连 `query_ids` 都没读，所以想用 GroupKFold
  得先改它。
- **后果**：M2 用来选头的排序、以及 M4 的元特征都偏乐观，且与 val/test 的元特征分布不一致；
  `corr(val_auc, test_auc)=0.028` 那个诊断也被混淆。M0/M1/M3/M5 是干净的。
- **怎么修**：换成 `GroupKFold(groups=query_id)`。

### P1-19. 因果 steering 的三处协议偏差

- **方向层是按 train 选的**：`causal_steer.py` 取 train 投影 AUC 的 argmax（排除最后一层）→
  q1500 的 3B 跑出 **L13**，而论文 val 选层是 **L9**（`significance_q1500.json`）。
  不是泄露，但论文 `main.tex:162` 写 "At the best internal layer"，与事实不符。
- **截断不一致**：`causal_steer.py` 用 `truncation=True, max_length=512`（右侧截断会**切掉
  "Question … Answer:" 后缀**），而 nnsight 提取器**从不截断**
  （`max_length` 在 `extract_instruct.py` 里其实是**死参数**）。
  → 对超过 512 token 的 prompt，steering 读的 P(yes) 在文档中间的 token 上，不是 answer token。
- **yes/no token 集漂移**：`causal_steer.py` 用 4 个变体，`llm_judge.py`/`logit_lens.py` 用 5 个
  （多 " YES"/" NO"），尽管前者 docstring 声称一致。实测影响 ≈0.001 AUC（α=0 基线 0.546 vs judge 0.545），
  属文档/代码漂移，非效应。
- **怎么修**：方向层改用 val 选层；去掉截断或让提取器也截断（两者取一致）；统一 token 集。

---

## P2 — 数据缺口（⚠️ 2026-10-12 性质变更：**原主机已被回收，这些不是"待取回"而是"已永久丢失"**）

> **重要**：原先这一档写的是"回 GPU 主机 `/root/shared-nvme/paper_A_internals` 打包取回"。
> **该主机已被回收**，`results/cache/`（几十 GB 激活缓存）随之消失。
> 所以 P2-1~P2-5 列出的文件**除非另有一份备份，否则已不可恢复**，
> 只能在新主机上**重新跑管线生成**（见 `docs/READING_GUIDE.md` §七）。
>
> **如果有旧备份**（tar/网盘/另一台机），打包命令仍然适用（**别 tar 整个 `results/`**，里面有 `.pt`）：
> ```bash
> cd /root/shared-nvme/paper_A_internals
> tar czf ~/paperA_snapshot_$(date +%F).tgz \
>     paper/main.tex paper/refs.bib paper/main.pdf \
>     results/*.json results/figures results/cache/*/meta.json \
>     scripts src README.md CLAUDE.md HANDOVER.md TODO.md docs .gitignore
> ```
>
> **这也意味着两件事**：①新的跑法会给出**复现版**数字而非原件，注意第 3 位小数的漂移
> （`requirements.txt` 目前**零钉版本**，已在 `src/provenance.py` 里开始记录版本与 revision）；
> ②P0-1 的"重画 Fig. 1"与 R-2/R-4 等需要缓存的事项，全部要先在新主机上重跑提取。
> 好消息是**纯 CPU 的分析脚本**（`plot_layer_signal` / `internals_vs_output` / `significance`）
> 只要拿到 `.pt` 就不需要显卡。

### P2-1. 缺 4 个 `internals_vs_output_*.json`

`internals_vs_output_llama32_3b_q1500.json`、`..._llama32_3b_instruct_q1500.json`、
`..._llama31_8b_q1500.json`、`..._llama31_8b_instruct_q1500.json`。

- **最关键**：`internals_vs_output_llama32_3b_q1500.json` 是 **Fig. 1 panel A 逐层 AUC 数组**的来源。
  它丢了 ⇒ **Fig. 1 的逐层曲线与 panel B 的 per-head 热力图数据，现在只存在于那张恢复出来的 PNG 里**
  （以及 PDF 内嵌副本）。这也是 P0-1 无法就地修复的直接原因。
- 存活 3 个：`..._llama32_3b_fiqa.json`、`..._qwen25_7b_q1500.json`、`..._qwen25_7b_instruct_q1500.json`。

### P2-2. 缺 2 个 `causal_steer_*.json`

`causal_steer_llama31_8b_q1500.json`、`causal_steer_llama31_8b_instruct_q1500.json`。
`REPORT.md` §11.1 与论文 Causality 段都声称"三个变体复现"，但仓库里**只剩 3B 一个的数值**。
好消息：`plot_causal.py` **只读 JSON、不依赖缓存**，JSON 一拿回来就能立刻出图。

### P2-3. 缺 7 张图

`llama31_8b_layer_signal.png`、`llama31_8b_instruct_layer_signal.png`、
`llama32_3b_instruct_layer_signal.png`、`mistral_7b_layer_signal.png`、
`mistral_7b_instruct_layer_signal.png`、`causal_steer_8b_base.png`、`causal_steer_8b_inst.png`。
- 前 5 张要 `.pt` 缓存（回主机重跑，或拷缓存到本机纯 CPU 跑）。
- 后 2 张只需 P2-2 的 JSON。
- 目前 `results/figures/` 只有 3 张（`llama32_3b_layer_signal` / `causal_steer_3b` 已恢复、
  `qwen25_7b_layer_signal` 一直存活）。

### P2-4. 所有 `*_layer_signal.json` 数字转储都缺

`plot_layer_signal.py` 本来会落盘 `results/figures/<tag>_layer_signal.json`，
但**一张图的 JSON 都没留下**（连存活的 qwen 图也没有）。回主机时一并带上。

### P2-5. `multimodel_summary.json` / `aggregate_models.json` 缺失

`aggregate_models.py` 的产物，被 `CLAUDE.md` §5 / `REPORT.md` §7 / `SKILL.md` 引用。
可重生成，但它扫描的正是 P2-1 里缺的那些 JSON，**必须先补 P2-1**。

---

### P2-6. 因文件缺失而**完全不可验证**的具体主张（清单）

上面 P2-1~P2-5 是"文件缺"，这里列"因此哪几句话在仓库里查不到依据"，
投稿前一并想清楚：要不要在文中降级措辞，或把文件补回来。

| 论文主张 | 依据文件 | 现状 |
|---|---|---|
| L385 few-shot judge `0.545 → 0.566` | `results/cache/llama32_3b_q1500_judge_fs4/meta.json` | **文件缺失**（在 `results/cache/` 里，被 gitignore），仅 REPORT §13/CLAUDE/HANDOVER 有文字 |
| L343 "only ~22% of test pairs are relevant" | 无任何 artifact | **仓库里没有出处**，仅散文 |
| L347–349 "因果在三个变体上复现 / 消融臂每次都崩塌 / 唯独 internal 保 AUC" | `causal_steer_llama31_8b{,_instruct}_q1500.json` | **两个文件都缺**，仅 REPORT §11.1 的表格佐证 |
| L350 "instruct 上崩塌最剧 `0.87 → 0.003`" | 同上 | **文件缺失**，仅 REPORT |
| L290 "LLaMA 最强头集中 L10–16" | `figures/<tag>_layer_signal.json` | **全部缺失**，仅 REPORT §5/CLAUDE 有文字 |
| L291–292 "Mistral 早层单头 AUC ≈0.50" | 同上 | **全部缺失**，仅 REPORT §6/CLAUDE |
| Fig. 1 三个 panel 的全部数值 | `internals_vs_output_llama32_3b_q1500.json` + `*_layer_signal.json` | **都缺**，只剩恢复出的那张 PNG |

---

## P3 — 工程与复现环境

### P3-1. LaTeX 工具链不在仓库里 → `main.tex` 改了无法验证

`tectonic`/`pdflatex`/`xelatex`/`latexmk` 均不在；`paper/tools/tectonic` 不存在
（`tools/` 在 `.gitignore` 里，编译器没随项目走）。
**建议**：下载 tectonic 静态二进制（约 30MB）放 `paper/tools/`，并加 `.gitignore` 白名单
（`!paper/tools/tectonic`）或单独备份。首次编译需联网拉 CTAN 宏包。
**这是处理 P0-3 的前置条件。**

### P3-2. vendored `nnsight/`（321 文件 / 3.7MB）是**惰性死重**（本次已核实）

- 它在 **`init` 提交里一次性加入，之后再没被改动过**（`git log -- nnsight` 只有 1 个提交）。
- 它的包在 `nnsight/src/nnsight/`（src-layout），而脚本把**仓库根**插到 `sys.path[0]`
  —— **仓库根上并没有可导入的 `nnsight` 包**，所以它**不会**遮蔽 `pip install nnsight` 装的那份。
  （我原本怀疑它会遮蔽，**核实后不成立**，特此记录以免重复排查。）
- 真正提供 `import nnsight` 的是 `setup_env.sh` 里的 pip 安装。
- **所以它的用途不明**：既没被 import、也没有任何相对上游的改动记录。
- **建议**：要么删掉（省 3.7MB 与困惑），要么说明"这是我们打了什么补丁的副本"并真正接进
  `sys.path`。现状是最糟的组合：占地方，又不能确定它和 PyPI 版本的差别。

### P3-3. 结果文件名的命名遗留（本次刻意未动）

- `internals_vs_output_{tag}.json`（500q/FiQA）vs `internals_vs_output_{tag}_q1500.json`（1500q）
  —— 前者不带 query 数。
- `significance.json` / `_q1500.json` / `_qwen_q1500.json` / `_fiqa.json` —— 四种后缀风格。
- **建议**：统一为 `{kind}_{tag}_q{N}.json`，一次性改掉论文与文档里的引用。

### P3-5. `src/probes.py` 的宽 except 与 docstring 不符（审计 P2 级）

- `probes.py:165-173`：`except Exception` 在 `roc_auc_score` 抛错时把 **accuracy 与 f1 一起置 0.0**
  （AUC 置 0.5），**静默**污染该 split 的指标。当前两个 loader 下每个 query 都同时贡献正负例，
  所以触发不了，属**潜在**地雷。
- `probes.py:94-105`：任何 fit 错误都变成 "AUC 0.5 + probe None"，而那个 `error` 键没人读。
- `probes.py:107-108`：按 **train** AUC 排序，而 docstring（第 38 行）说是 validation；
  `initial_validation.py:209` 在外面绕过了这个不一致。

### P3-6. `src/evaluation.py` / `initial_validation.py` 的图与标签错配（审计 P2 级）

- `evaluation.py:81-83`：`aucs[k]` 把第 (k+1) 个值标成 "top-k" —— **off-by-one**。
- 同一处在一份**按 val 排序**的列表上画 **train** AUC，于是图的顺序与数值/轴标签互相矛盾。
- `initial_validation.py:429-438`：用 **mean train AUC** 宣布哪个方案是 "Winner"。
- `evaluation.py:164-169`：标题写 "Ensemble Probe Performance"，画的却是 train 的 accuracy/f1
  （而 MS MARCO 是 1:4 不平衡，accuracy/F1 本就没有意义——见"已核实"第 5 条）。
- **注意**：这条只影响 `initial_validation.py` 那条**早期**管线，不影响论文的任何数字。

### P3-7. `scripts/` 的三处卫生问题（审计 P2 级）

- **`run_queue.sh`**：`:43,47` 用 `|| true` 吞掉 aggregate 与出图的失败；`:35` 对
  **JSON 已存在就 SKIP** —— 由于仓库里那些 `internals_vs_output_*.json` 是
  **test-argmax 的旧产物**（P0-6），队列会**永远复用它们、不再重生成**。
  修：把 SKIP 条件换成检查产物是否由当前代码版本生成（例如比对 `env`/时间戳）。
- **`run_scaleup.sh`**：`:4,7` 只有 `set -u` **没有 `-e`**，且硬编码
  `cd /root/shared-nvme/paper_A_internals` —— 在新主机上 `cd` 失败后，
  整个 scale-up 会**在错误的 CWD 下静默跑下去**。修：`set -euo pipefail` +
  `cd "$(dirname "$0")/.."`（其他驱动脚本已经是这个写法，只有它没跟上）。
- **`plot_causal.py:24`**：把 `0.793` 作为水平基线**硬编码**，这只对 3B 那张图成立；
  画 8B / instruct 的因果图时基线是错的。修：从传入的 JSON 里读 `sweep.internal.mean_pyes[α=0]`。

### P3-8. `llm_judge_fewshot.py` 仍然评了 train split（审计 P2 级）

- 它从 **train** 抽 exemplar（`:46-58`，无 val/test 泄露 ✔），但随后**也评了 train split**
  （`:117-128`），而 exemplar 就来自那批行 → 那个 train AUC **没有意义**。
- 引用 few-shot 数字时**只引 test**（`0.566`）。
- 另外 `:95-96` 用文本匹配 `ex_qids`，属元数据、不影响结果。

### P3-4. 残留文件

- `.ipynb_checkpoints/README-checkpoint.md`（已被 gitignore，不影响提交），可删。
- 上级目录若还有 `my_paper_project-paper-A.zip` 之类旧快照，注意**它不含这两张丢失的图**。

---

## R — 研究逻辑层面（比 bug 更值得先想清楚；本次通读新增）

> 这些不是代码错误，是**论证结构上的软肋**。投稿前值得逐条想清楚怎么回应，
> 因为审稿人最可能从这些地方进攻。

### R-1. 最强的主张建立在**混淆最多**的对照上，而**无混淆的对照最弱** ⚠️ 最大软肋

- `best_resid − final` 是**唯一干净**的对照：同为残差流、同为 3072 维、
  在 `significance.py` 里**同为 `C=1.0` + StandardScaler**——只差"读哪一层"。
  但它在 **instruct 模型上不显著**，只在 base 模型显著。
- 论文的头条主张 `attn − final` 在 instruct 上也显著，但它**不是 capacity-matched**：
  top-20 头 × head_dim（2560 维）vs 最终层残差（3072 维），特征空间不同、还有"在 val 上选头"
  这一步搜索。在 `internals_vs_output.py` 里两者连正则都不同（attn 是 C=1.0，残差是 C=0.5）。
- **后果**：一个 hostile reviewer 会说"你的效应是 base 模型 + 未对齐架构的性质；
  在人们真正部署的 instruct 模型上，那个干净的对照消失了"。
  论文把 instruct 上的收缩讲成"对齐把信号推进输出"的机制证据（自洽、也确实有意思），
  但**没有正面承认它同时是 thesis 的边界**。
- **建议**：①在 Limitations 里明确写"干净对照只在 base 上成立"；②补 capacity-matched 对照（R-4）。

### R-2. `internals_vs_output.py` **至今仍在 test 上选层**，且缺少样本对齐断言

`internals_vs_output.py:78`：`resid_best_layer = argmax(layer_aucs)`，而 `layer_aucs` 是 **test** AUC。
P1 的修复只改了**论文表格**（改用 `significance.json` 的 val 选层值），**没有改这个脚本**。
`plot_layer_signal.py` 有同样的毛病（P0-1）。

- **后果**：仓库里 3 个 `internals_vs_output_*.json` 的 `resid_best_layer` 都带乐观偏差；
  任何未来跑这个脚本的人都会拿到偏高的数。
- **另一个隐患**：`internals_vs_output.py` **缺少** `significance.py` 里那句
  `assert np.array_equal(jd["labels"], yte)` —— 它把 judge AUC 与探针 AUC 分开算，
  **没有任何机制保证两个缓存描述的是同一批样本**（`--n-queries`/`--seed`/`--dataset` 不一致会静默错配）。
- **建议**：把这两个脚本都改成 val 选层（与 `significance.py` 对齐），并补上该断言。

### R-3. "attenuation"（衰减）这个标题词，证据其实更支持"misalignment"（错位）

- 论文实测：①中层线性可解码性 > 最终层；②**logit-lens 在每一层都接近随机**
  （3B 峰值仅 0.560、8B 0.577），远低于训练探针的 0.62；③把内部方向注入输出层能改变判断。
- ①是"线性可达性随深度变化"；②说明信号**与 unembedding 方向不对齐**——
  论文在 logit-lens 那节自己也写了 "misaligned with the output readout"。
  但标题与 Abstract 用的是 **attenuated**（衰减），暗示"信号本来会到达输出、只是被削弱了"。
- **没有任何证据表明中层信号曾经"可以"到达输出**。严格说，②恰恰是对"存在一条被衰减通路"的**反证**：
  如果只是衰减，模型自己的读出至少应该在**某个**深度显示出部分信号。
- **建议**：要么把主张改写为"信号可解码但与输出读出方向错位"，
  要么补一条真正测"衰减"的实验（例如把**第 L 层的真实激活** patch 到输出层，
  看输出能否恢复判别力——§11 的 steering 做了一半，但注入的是 diff-of-means 方向而不是真实激活）。
  审稿人一定会问这个区别。

### R-4. 缺 capacity-matched / 非线性对照（最可能被要求补的实验）

标准攻击："你的 attn 探针有 2560 维 + 选头搜索，final 探针只是 3072 维线性——
给 final 也上同等自由度（MLP、或多层拼接、或多头读出），gap 就没了。"

- 论文**没有**跑这个对照。`sweep_topk.py` 扫的是 top-k 与归一化，不是探针表达能力。
- **建议**（按性价比）：①在 final 残差上跑 MLP 探针（1 隐藏层）；
  ②把 final 层也拆成多头式读出以匹配归纳偏置；
  ③至少报告 top-k 变化时 attn−final 是否稳定。
  若 gap 在非线性 final 探针下仍在，thesis 会强很多；若消失，就得改写论点——**早知道比被审稿人指出好**。

### R-5. 因果 steering 的替代解释没有被排除

`results/causal_steer_llama32_3b_q1500.json` 自己记录了
`proj_auc_internal = 0.602` vs `proj_auc_final = 0.577`。

- 也就是说：**内部方向本身就是一个更好的相关性方向**。那么"internal 方向保持 AUC、
  final 方向摧毁 AUC"这个结果，**一部分可以用"注入的方向质量不同"解释**，
  而不必然是"输出通路本可表达内部信号"。
- 消融臂（抽掉内部方向 → P(yes) 崩塌）仍然是有意义的因果证据，但**AUC 对照这条论据偏弱**。
- **建议**：补一个把 `proj_auc` 匹配过的对照，或在文中明确承认这一替代解释。

### R-6. 多重比较未校正

8 模型 × 4 条 edge，单模型 p 值未做 multiple-comparison 校正。
跨模型符号检验（p=0.016）稳健，但逐模型 p 值建议说明或加 BH 校正。

### R-7. 数据构造的次要问题（诚实记录，非致命）

- `load_beir_relevance` 的负例用 `rng.choice` 且**只排除正例、不排除已选负例**
  → 同一 query 可能出现**重复负例**（轻微重复加权）。
- 同函数的 `qid_num = abs(hash(q)) % 10**9` 回退分支：Python 的 `hash(str)` 受
  `PYTHONHASHSEED` 随机化，**若出现非数值 query id，切分将不可复现**。
  目前 FiQA 的 id 都是数值、走 `int(q)` 分支，所以只是**潜在**地雷。
- 文档多处写 MS MARCO 负例是 "BM25 hard negatives"，代码取的是该记录里 `is_selected == 0`
  的段落——**这确实是 BM25 检索到的未选中段落**，所以说法**成立**（已核实，记录以免重复怀疑）。
- 每 query 的正负比并非固定 "1 正 4 负"：`n_pos = min(len(pos), 5//2)`，
  所以有 2 个正例的 query 会变成 2 正 3 负。文档里的 "1 rel + 4 hard neg" 是**近似**。

---

## P4 — 论文尚未做的可选工作（沿用原计划，只做索引）

| # | 事项 | 价值 | 需要什么 |
|---|---|---|---|
| 1 | Qwen 因果 steering（第三家族因果复现） | 高 | GPU + qwen 缓存 |
| 2 | Prompt 改写鲁棒性（2–3 种模板） | 高（堵 prompt-specific 质疑） | GPU + 重跑 extract/judge |
| 3 | Appendix 放 8 模型全图 | 中（正面回应 P0-3） | 缓存 + `plot_layer_signal.py` |
| 4 | FiQA 改 BM25 难负例 + 加第二模型 | 中（FiQA 现太易，AUC 0.99） | GPU + 改 `src/data.py` |
| 5 | Judge 阈值校准（Platt scaling） | 中 | GPU + 重跑 judge |
| 6 | Mistral 因果 steering | 低 | GPU |
| 7 | 投稿目标会议定稿（格式/篇幅/deadline） | —— | 决策，非工程 |

---

## 已处理

- ✅ **2026-10-12 两张丢失的论文图已恢复**（`llama32_3b_layer_signal.png`、`causal_steer_3b.png`）：
  从 `paper/main.pdf` 的内嵌图像 XObject 无损导出，脚本 `scripts/extract_figures_from_pdf.py`，
  来源说明 `results/figures/README.md`。commit `2ce9363`。
- ✅ **2026-10-12 `.gitignore` 已收窄**：`results/` + `*.png` + `*.csv`
  → `results/cache/` + `results/**/*.pt` + `results/logs/`。commit `2ce9363`。
- ✅ **2026-10-12 缓存命名已规范化**：见 `docs/CACHE_NAMING.md`；覆盖
  `run_model.sh`/`run_queue.sh`/`run_fiqa.sh`/`significance.py` + 7 个脚本默认值。commit `9af7d6a`。
- ✅ **2026-10-12 硬编码代理口令已从工作区移除**（历史未清）。commit `7a7b634`。见 S-1。
- ✅ **2026-10-12 `CLAUDE.md` 已纠偏**：§2 加"提交前查密钥"、§3.2 加 P1 口径加注、
  §5 加命名规范指针、§6 修正已完成的"进行中"条目并新增第 7 次推进、§七加非 GPU 机器能力边界。
- ✅ **2026-10-12 新增 `docs/READING_GUIDE.md`**：阅读顺序（4 条路线）、每份文档的时效性与
  权威性速查表、§6 流水账读法、云容器从零开始步骤、6 条"别浪费时间"。
- ✅ **2026-10-12 已核实无误的项**（记录以免重复排查）：`paper/refs.bib` 14 条全部被引用、
  无悬挂 key 也无未用条目；Table 1 八行与各自来源 JSON **逐格吻合**；
  Abstract 的因果 / FiQA / logit-lens 数字全部与 JSON 吻合；
  vendored `nnsight/` **不会**遮蔽 pip 版本（见 P3-2）；
  MS MARCO 负例确为 BM25 未选中段落（"hard negative" 说法成立）。
- ✅ **2026-10-12 完成第二次独立审计（代码正确性）**，结论已并入 P0-6/P0-7/P1-15~P1-19。
  它**执行了**两处复现（用真实 `src/data.py` 跑 stub 数据集；多进程 `PYTHONHASHSEED` 跑切分逻辑），
  并明确列出**已排除**的怀疑（记录以免重复排查）：
  1. **四个脚本的 instruct prompt 逐字节相同**（逐个 AST 提取并实际调用过 builder）；
     `src/activations.py` 里那个 "Q:/P:" 旧格式只被 `initial_validation.py` 用到，属另一套 regime。
  2. **种子一致**（处处 42），整数 query_id 下数据管线**确定性**（跨 PYTHONHASHSEED 实测相同）；
     只有 BeIR 的哈希回退路径会破（P1-16）。
  3. **切分确实是按 query 的**，且**验证集从未被用于拟合**（选头/选层在 val，探针 fit 在 train，
     没有任何 scaler 在 val/test 上 fit）。对 test 的唯一接触就是 P0-6 那些 argmax。
  4. **bootstrap 形状是对的**：确实是 query 聚类 + **配对**（共享重采样索引），
     `p_le0` 是正确方向的单侧（用 Qwen 那个负的 train_edge 验过，p=0.9996）。
     唯一保留意见：没有 `(1+count)/(1+B)` 平滑，所以 `p=0.0` 实为 <2e-4。
  5. **`class_weight="balanced"` 不偏 AUC**（AUC 是秩统计量、四个 reader 评在同一批行上）。
     正负比确为 1:4（用真实 loader 复现出 20% 正例率，与论文"~22%"一致）——
     对 AUC 无碍，但 `initial_validation` 里那些 accuracy/F1 图因此**没有意义**。
  6. **MS MARCO 负例是 hard 而非随机**；随机采样只存在于 BeIR 加载器。
  7. **logit-lens 的实现位置正确**（对每层 decoder 输出施加 `norm` + `lm_head`），
     末层逐位复现 judge（3B/8B/Qwen 的 `sanity_lastlayer_vs_judge_absdiff = 0.0`）。
     这同时证明**残差缓存末层就是 judge logits 的来源向量**（所以 `train_edge` 是干净的
     "纯训练"对照），也证明那三次运行的 judge 与激活缓存**是对齐的**。
  8. **头索引一致**（layer-major，提取 / `probes.py` / `plot_layer_signal.py` 三处一致）；
     同一缓存内 attn 与 resid 由同一循环产出，必然对齐。
  9. **没有裸 `except:`**；唯一的 `|| true` 在 `run_queue.sh`（已记录）。
- ✅ **2026-10-12 完成一次独立交叉复核**（专项 fact-check，逐格比对 `main.tex` 与 `results/`，
  结论已并入 P0-4 / P0-5 / P1-7~P1-14 / P2-6）。核心结论：**Table 1 全对（42/42）；
  Table 2 全错（0/18，全部是被取代的旧版本）；Table 3 对 2 错 4**。
  这不是零散笔误，而是"artifact 重算后文档/论文没同步"的**系统性**问题。
