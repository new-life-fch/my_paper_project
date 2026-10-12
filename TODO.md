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
- **最可能的解释（诚实标注为推断）**：git 显示 `results/significance.json` 最后一次被写是
  `4c8be8a`「显著性改用**配对** bootstrap」这个提交本身；而 `REPORT.md` §9 是更早的
  `0f58593` 写的（**非配对**版本，CI 偏宽、p 偏大）。也就是说**artifact 在修方法学时被重算了，
  但 REPORT 与论文表格没有同步更新**。
- **方向是良性的**：论文的数字比 artifact **更保守**（CI 更宽、p 更大），
  这不像"夸大结果"，更像"引用了旧版本"。但**必须解决**。
- **怎么修**：回 GPU 主机（或拿到缓存后在 CPU 上）用**当前代码**重跑
  `significance.py`（500q 与 1500q 两轮），然后把
  `REPORT.md` §9/§10、`CLAUDE.md` §6、`main.tex` Table 2/3 + Abstract **全部对齐到新 JSON**。
  在解决之前，建议在 `REPORT.md` 顶部加一条醒目提示，避免后续继续引用错数。
- **附带（已核对无问题）**：FiQA 的 `resid_best_layer=0.993` 与 Abstract 的
  "0.993 / +0.253" **吻合**；Abstract 的因果、logit-lens 数字也**全部吻合**。
  本问题集中在 bootstrap 那两张表。

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

### P1-5. `requirements.txt` 与实际依赖不符

- 缺 **`nnsight`**（`src/activations.py` 直接 import；由 `setup_env.sh` 单独 `pip install`）。
- 缺 **`joblib`**（`plot_layer_signal.py`、`significance.py` 用 `joblib.Parallel`）。
- 缺 **`pypdf`**、**`pillow`**（新增的 `scripts/extract_figures_from_pdf.py` 需要）。

### P1-6. `scripts/setup_env.sh` 有几处问题

- `pip install nnsight` 与仓库里那份 vendored `nnsight/` 并存且无说明（见 P3-2；已核实**不会**遮蔽）。
- `torch.cuda.get_device_properties(0).total_mem` —— **疑似属性名错误**（现代 PyTorch 是
  `total_memory`）。⚠️ **未能验证**（本机无 GPU），需在有卡的机器上跑一次确认；
  若确实错误，会因 `set -e` 让脚本在 `[3/4]` 步直接失败。
- 提示用 `huggingface-cli login`，与 `CLAUDE.md` §2「`huggingface-cli` 在本环境损坏，
  请用 Python `huggingface_hub.login()`」**直接矛盾**。
- 没有设置 `HF_HOME`（30G 系统盘的坑只在 `run_*.sh` 里防了，setup 阶段没防）。

---

## P2 — 数据缺口（只能回 GPU 主机 `/root/shared-nvme/paper_A_internals` 取）

> `.gitignore` 原先把 `results/` 整个忽略，这些文件从未进入任何 clone（已实测确认：
> clone 出来的 `results/` 只有 13 个文件）。2026-10-12 已把 `.gitignore` 收窄为
> 只忽略 `results/cache/`、`results/**/*.pt`、`results/logs/`，**以后的结果会自动入库**；
> 但**已经丢的只能回主机取**。
>
> 一次性打包（**别 tar 整个 `results/`**，里面有几十 GB 的 `.pt`）：
> ```bash
> cd /root/shared-nvme/paper_A_internals
> tar czf ~/paperA_snapshot_$(date +%F).tgz \
>     paper/main.tex paper/refs.bib paper/main.pdf \
>     results/*.json results/figures results/cache/*/meta.json \
>     scripts src README.md CLAUDE.md HANDOVER.md TODO.md docs .gitignore
> ```

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
