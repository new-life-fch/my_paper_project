# TODO — 待办与已知问题清单

> 建立于 **2026-10-12**。本文件只记录**尚未处理**的事项，按优先级排列。
> 处理完一条就改成 ✅ 并注明日期。项目主线结论与实验状态看 `CLAUDE.md` §6 与 `results/REPORT.md`。
> 缓存目录命名规范见 `docs/CACHE_NAMING.md`。

---

## S — 安全（用户已决定暂缓）

### S-1. 公开仓库泄露了真实代理口令 🔴（用户 2026-10-12 决定「暂时先不管」）

- **事实**：`run_experiment.sh`、`scripts/download_model.sh`、`CLAUDE.md` §七、`HANDOVER.md`
  里硬编码了 `http://<user>:<pass>@10.255.128.102:3128`。仓库 `private: false`（公开），
  该口令出现在 **8 个提交**里，最早可追到 `init`。
- **已做**（2026-10-12）：4 个文件的工作区内容已改为从环境注入，`git grep` 已确认工作区无残留。
- **未做**：**git 历史里仍在**。要清掉需要 `git filter-repo`（会改写全部 commit hash，需 force push，
  协作方可复现问题）。
- **最小成本的正解**：**轮换该代理口令**。口令一旦换了，历史里那串就自动失效，
  不需要动历史。这是唯一能真正止损的动作。
- **风险量级（诚实评估）**：代理是 `10.255.x.x` 私有地址，仅在那张内网内可用，
  所以危害有限；但公网仓库的泄露主要面对**自动扫描机器人**而非人工查看。
- **操作副作用（注意）**：`scripts/download_model.sh` 与 `run_experiment.sh` 现在**要求先
  `export https_proxy=...`**，否则直接 `exit 1`。在 GPU 主机上跑之前记得设一次。

---

## P0 — 影响论文正确性，投稿前必须处理

### P0-1. Fig. 1 与 Table 1 的选层口径不一致（P1 修复的尾巴）

- **现象**：`results/figures/llama32_3b_layer_signal.png` 图内标注 **peak L12 = 0.632**；
  而 `paper/main.tex` Table 1 的 LLaMA-3.2-3B 行是 **best-resid 0.628**。
- **原因**：这张图是在 **P1 修复（test 选层 → val 选层）之前**生成的。
  `scripts/plot_layer_signal.py` 里 `resid_peak` 用的是 `np.argmax(resid_auc)`，
  而 `resid_auc` 是 **test** 曲线 → selection-on-test。Table 1 后来统一改成 val 选层
  （`results/significance.json`：`best_layer_val=13`, `best_resid=0.628`）。**图没跟着重画。**
  代码里已加 `NOTE (known bias...)` 注释标出该行。
- **影响**：审稿人对照图与表就会发现差异——而且恰好是 P1 那个「选层乐观偏差」问题本身。
- **怎么修**：把 `plot_layer_signal.py` 改为在 val 上选层，再重跑。**需要
  `results/cache/llama32_3b_q500_instruct` + `..._judge`**（注意按 `docs/CACHE_NAMING.md` 可能是旧名
  `q500_instruct`）。缓存拷到本机后是**纯 CPU** 的活，不需要 GPU。
- **备注**：同源问题还有 final 的小数差（图 0.593 / 表 0.592），量级无害，一起重画最干净。

### P0-2. Table 1 混用了两个样本量

- **现象**：Table 1 八行里，前 6 行（LLaMA×4、Mistral×2）取自 `results/significance.json`（**500q**），
  Qwen 两行取自 `results/significance_qwen_q1500.json`（**1500q**）——因为 Qwen 只跑过 1500q。
- **影响**：两层。
  1. caption 只写 "test AUC, point estimates; val-selected layers and heads"，**没交代样本量不同**。
  2. 与 **Table 3 自相矛盾**：Table 3 用 1500q 报 LLaMA-3.2-3B base 的 final 0.536 / best-resid 0.614，
     而 Table 1 同一行是 0.592 / 0.628。同一模型同一 reader 两套数，审稿人很容易追问。
- **怎么修**（二选一，都不需要跑实验）：
  - **A（推荐）**：Table 1 统一到 1500q。LLaMA 4 个 + Qwen 2 个已有 1500q 数据
    （`significance_q1500.json` + `significance_qwen_q1500.json`），**只缺 Mistral 两个的 1500q**
    （需回主机补跑，或把 Mistral 行显式标注为 500q）。
  - **B（最省事）**：保留现状，但在 caption 里写明 "LLaMA/Mistral rows: 500 queries; Qwen rows: 1500 queries"，
    并说明 Table 3 是 1500q 重测。诚实，但读起来略别扭。

### P0-3. 正文引用了不存在的 supplement

- **现象**：`main.tex` Experiments 的 Depth localization 段写
  "Figure~\ref{fig:depth} (**per model in the supplement**) shows ..."，
  但论文里**没有 appendix / supplement 章节**，也没有任何 per-model 深度图。
- **影响**：dangling reference，审稿人会去找 supplement 然后找不到。
- **怎么修**：要么删掉 "(per model in the supplement)"，要么真补一个 appendix ——
  但那些图（8B / Mistral / instruct 的 `*_layer_signal.png`）现在全都不在（见 P2-3），
  重画同样需要缓存。
- **注意**：本地若没有 LaTeX 工具链（见 P3-1），改完 `main.tex` **无法编译验证**。

---

## P1 — 文档与事实不符

### P1-1. `README.md` 仍是 P1 修复前的旧数

第 5–10 行：`0.593`（final）、`0.632`（L12 最强）。其中 final/peak 是 test-argmax 口径，
现行应为 **0.592 / 0.628 (L13)**。另外 README 只给了 3B 单模型的数，没提 8 模型规模。

### P1-2. `CLAUDE.md` §3.1 也仍是旧数

§3.1 三段式论点里的 `0.593` / `0.632` / `L12` / 「训练贡献 +0.058 / 读内部 +0.092」都是旧口径。
§3.2 顶部已加 ⚠️ 加注指向新数，**但 §3.1 正文还没改**（担心改动过大影响历史叙述）。

### P1-3. `HANDOVER.md` 的 8 模型表是旧口径，且引用了已不存在的路径

- §4.2 的表用 **test-argmax** 数（3B 0.581(L14)、8B 0.581(L14)、Mistral 0.621(L18)…），
  与 `main.tex` Table 1 的 val 选层数不同。
- §2.1 的文件树还写着 `paper/tools/tectonic`（已不存在）。
- **说明**：HANDOVER 生成于 2026-06-30、标为 v1.0，作为**历史交接文档**保留是合理的。
- **建议**：在文件开头加一句「本文件是 2026-06-30 的历史快照，数字口径可能不同于当前 `main.tex`，
  以 `CLAUDE.md` §6 与 `results/*.json` 为准」。既保留历史又不误导。

### P1-4. `requirements.txt` 与实际依赖不符

- 缺 **`nnsight`**（`src/activations.py` 直接 import；项目里 vendor 了一份，见 P3-2）。
- 缺 **`joblib`**（`scripts/plot_layer_signal.py`、`significance.py` 用了 `joblib.Parallel`）。
- **建议**：补上这两个，或明确注释「nnsight 用 vendored 副本」。

---

## P2 — 数据缺口（只能回 GPU 主机 `/root/shared-nvme/paper_A_internals` 取）

> 背景：`.gitignore` 原先把 `results/` 整个忽略了，所以这些文件从未进入任何 clone。
> 2026-10-12 已把 `.gitignore` 收窄为只忽略 `results/cache/`、`results/logs/`、`*.pt`，
> **以后的结果会自动入库**；但**已经丢的只能回主机取**。
>
> 在主机上一次性打包（**别 tar 整个 `results/`**，里面有几十 GB 的 `.pt`）：
> ```bash
> cd /root/shared-nvme/paper_A_internals
> tar czf ~/paperA_snapshot_$(date +%F).tgz \
>     paper/main.tex paper/refs.bib paper/main.pdf \
>     results/*.json results/figures results/cache/*/meta.json \
>     scripts src README.md CLAUDE.md HANDOVER.md .gitignore
> ```

### P2-1. 缺 4 个 `internals_vs_output_*.json`

`internals_vs_output_llama32_3b_q1500.json`、`..._llama32_3b_instruct_q1500.json`、
`..._llama31_8b_q1500.json`、`..._llama31_8b_instruct_q1500.json`。

- **最重要的一条**：`internals_vs_output_llama32_3b_q1500.json` 是 **Fig. 1 panel A 逐层 AUC 数组**的来源。
  它丢了，意味着 **Fig. 1 的逐层曲线与 panel B 的 per-head 热力图数据，现在只存在于那张恢复出来的 PNG 里**
  （以及 PDF 里的内嵌副本）。这也是 P0-1 无法就地修复的直接原因。
- 存活下来的只有 3 个：`..._llama32_3b_fiqa.json`、`..._qwen25_7b_q1500.json`、`..._qwen25_7b_instruct_q1500.json`。

### P2-2. 缺 2 个 `causal_steer_*.json`

`causal_steer_llama31_8b_q1500.json`、`causal_steer_llama31_8b_instruct_q1500.json`。

- `REPORT.md` §11.1 与 `main.tex` 的 Causality 段都声称「同一协议在三个变体
  （3B / 8B base / 8B instruct）上复现」，但仓库里**只剩 3B 一个的数值**，无法核对 8B 的两个。
- 好消息：`plot_causal.py` **只依赖这些 JSON、不依赖缓存**——JSON 一拿回来就能立刻出图。

### P2-3. 缺 7 张图

`llama31_8b_layer_signal.png`、`llama31_8b_instruct_layer_signal.png`、
`llama32_3b_instruct_layer_signal.png`、`mistral_7b_layer_signal.png`、
`mistral_7b_instruct_layer_signal.png`、`causal_steer_8b_base.png`、`causal_steer_8b_inst.png`。

- 前 5 张需要 `.pt` 缓存 → 回主机重跑 `plot_layer_signal.py`（或把缓存拷到本机，纯 CPU）。
- 后 2 张只需 `causal_steer_llama31_8b*.json`（= P2-2）。
- 目前 `results/figures/` 只有 3 张：`llama32_3b_layer_signal.png`（已恢复）、
  `causal_steer_3b.png`（已恢复）、`qwen25_7b_layer_signal.png`（一直存活）。

### P2-4. 所有 `*_layer_signal.json` 数字转储都缺

`plot_layer_signal.py` 本来会把底层数字落成 `results/figures/<tag>_layer_signal.json`，
但**一张图的 JSON 都没留下**（连存活的 qwen 图也没有对应 JSON）。回主机时一并带上。

### P2-5. `multimodel_summary.json` / `aggregate_models.json` 缺失

`scripts/aggregate_models.py` 的产物，`CLAUDE.md`/`HANDOVER.md` 都提到过。
用现有 JSON 可重生成，但它扫描的是 P2-1 里缺的那些 `internals_vs_output_*.json`，**必须先补 P2-1**。

---

## P3 — 工程与复现环境

### P3-1. LaTeX 工具链不在仓库里 → `main.tex` 改了无法验证

`tectonic` / `pdflatex` / `xelatex` / `latexmk` 都不在；`paper/tools/tectonic` 不存在
（`tools/` 在 `.gitignore` 里，编译器没随项目走）。
**建议**：下载 tectonic 静态二进制（GitHub releases，约 30MB）放 `paper/tools/`，
并在 `.gitignore` 加白名单（`!paper/tools/tectonic`）或单独备份。
注意 tectonic 首次编译要联网拉 CTAN 宏包（`HANDOVER.md` 提到需要代理）。

### P3-2. `nnsight/` 整库 vendored 进项目（3.7MB / 321 个文件）

- 它是**被提交进仓库的**（`git ls-files nnsight | wc -l` = 321），所以不是误拷贝的残留，是**有意的**。
- **但没有任何记录说明它相对上游改了什么、基于哪个版本**——这才是真问题。
- **建议**：要么记录 patch / 固定上游 commit（如 `nnsight @ git+...`），要么删掉改用 `pip install nnsight`。
  否则以后无法判断「本地这份和 PyPI 上那份差在哪」。

### P3-3. 结果文件名的命名遗留（本次未动）

`results/` 下的结果 JSON 命名规则不统一，同理有歧义，但改动会牵动论文引用与多处文档，**本次刻意未动**：

- `internals_vs_output_{tag}.json`（500q/FiQA）vs `internals_vs_output_{tag}_q1500.json`（1500q）
  —— 前者不带 query 数。
- `significance.json` / `significance_q1500.json` / `significance_qwen_q1500.json` / `significance_fiqa.json`
  —— 四个文件四种后缀风格（有 `_qwen_` 这种夹在中间的，也有 `_fiqa` 这种纯数据集后缀）。
- **建议**：统一为 `{kind}_{tag}_q{N}.json`，并把论文与文档里的引用一次性改掉（一次性成本，长期收益）。

### P3-4. 残留文件

- `.ipynb_checkpoints/README-checkpoint.md` —— Jupyter 残留（已被 gitignore，不影响提交），可删。
- 上级目录若还有 `my_paper_project-paper-A.zip` 之类的旧快照，注意**它不含这两张丢失的图**，
  容易误导后来人以为「备份里有」。建议删掉或改名标注日期。

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
  来源说明 `results/figures/README.md`。详见 `CLAUDE.md` §6 第 7 次推进。
- ✅ **2026-10-12 `.gitignore` 已收窄**：`results/` + `*.png` + `*.csv`
  → `results/cache/` + `results/**/*.pt` + `results/logs/`。
- ✅ **2026-10-12 缓存命名已规范化**：见 `docs/CACHE_NAMING.md`；
  改动覆盖 `run_model.sh` / `run_queue.sh` / `run_fiqa.sh` / `significance.py` + 7 个脚本的默认值。
- ✅ **2026-10-12 `CLAUDE.md` 已纠偏**：§2 加「提交前查密钥」、§3.2 加 P1 口径加注、
  §5 加命名规范指针、§6 修正早已完成的「进行中」条目并新增第 7 次推进、§七加非 GPU 机器能力边界。
