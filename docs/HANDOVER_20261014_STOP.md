
# 停机扩容前的断点记录 + 本次会话完整工作日志（2026-10-13 ~ 10-14）


> ⚠️ **本文档已于 2026-10-14 归档，不再维护。**
> 项目状态、实测数据、判据与规划**一律以 [`STATUS.md`](STATUS.md) 为准**。
> 保留本文档只为查历史（它记录的当时口径与路径多为旧主机，**不要再据此操作**）。
> 归档原因：七份分散文档对同一个量给出过不同数值（其中一次写进了论文同一张表），
> 手工维护的状态文档会腐烂；`STATUS.md` 的实测部分由
> `python scripts/build_status.py` 从 `results/*.json` 直接生成。

> 用户于 2026-10-14 下午**主动停机以扩容数据盘**，并说明**扩容后会压缩 agent 的上下文**。
> 因此本文件被写成**自足**的：即使上下文被压缩、甚至只读这一个文件，也能知道
> ①机器与环境前提 ②做过什么、结论是什么 ③踩过哪些坑 ④停机时精确状态 ⑤接下来做什么。
>
> 恢复时建议顺序：**本文件 → `CLAUDE.md` §六（第 8 次推进）→ `TODO.md`（E 档与已决定）→
> `docs/CCFB_READINESS.md`（投稿就绪度）**。

---

## 〇、一句话摘要

原 GPU 主机已回收，本次在**新租的 AutoDL 实例**上从零重建环境、重下 8 个模型（104 GB）、
修掉约 20 个「新主机必炸」或「会让结论失真」的 bug，然后跑完 **8 模型 × 1500q 全流水线 +
四项补充实验 + 500q 受控复算 + FiQA 换难负例重做**，共 **24 个提交 / 119 个文件**。
结果是：**头条结论复现且更硬（`attn−final` 8/8 过 BH 校正），但因果章节被自己的新判据推翻，
论文必须改论证结构**。停机时正跑到因果 4 变体重跑的第 2 个。

---

## 一、机器与环境前提（**变了，别照旧文档操作**）

| 项 | 现行值 |
|---|---|
| 机器 | **AutoDL 实例，RTX 4080 SUPER（32G 显存档）**；系统盘 30G、数据盘 150G |
| 仓库 | `/root/autodl-tmp/my_paper_project`（**不是** `/root/shared-nvme/paper_A_internals`） |
| Python/torch | 3.12.3 / **torch 2.8.0+cu128（服务器自带，勿重装）** / CUDA 12.8 |
| 关键依赖 | transformers 5.19.0 · datasets 5.1.0 · nnsight 0.7.0 · sklearn 1.9.1 · numpy 2.3.2 · rank_bm25 0.2.2（已钉进 `requirements.txt`） |
| 缓存 | `HF_HOME=/root/autodl-tmp/hf_cache`（`~/.autodl_cache_env.sh` + `/etc/profile.d` + `BASH_ENV` 三处兜底） |
| 下载通道 | **hf-mirror.com 直连**（`HF_ENDPOINT`）；走 AutoDL 代理会停滞 |
| LaTeX | `paper/tools/tectonic`（0.15.0 静态二进制，**已入库**）；缓存 `/root/autodl-tmp/tectonic_cache` |
| 模型 | 8 个（LLaMA-3.2-3B/-Inst、LLaMA-3.1-8B/-Inst、Mistral-7B/-Inst、Qwen2.5-7B/-Inst）= 104 GB |
| 数据集 | MS MARCO v1.1（train 82,326 行）+ BeIR FiQA（corpus 57,638 段）+ FiQA qrels |

---

## 二、做过什么（按时间顺序，含结论）

### 阶段 A：环境与数据（对应提交 `9b20e7f` `599a6ec` `5147d18` `fc453f2`）

1. **装依赖**（torch 用服务器自带的，未重装）；**卸载 hf_xet**（与代理不兼容会卡死，
   注意装 nnsight 会把它带回来，要再卸一次）。
2. **缓存全部指向数据盘**，并保证交互/登录/非交互三种 shell 都生效。
3. **下载 8 个模型**：huggingface.co 走代理拉大分片会停滞（首个 3B 靠超时续传磨了 6.9 分钟），
   改 **hf-mirror 直连**稳定 3–5 MB/s；8 个模型逐个**真实加载 + 前向**验证通过。
4. **删掉 Mistral 的重复权重**（`consolidated.safetensors` 与分片格式是同一份，各省 14.5 GB）。
5. **数据集缓存**（MS MARCO 只下 v1.1 的 parquet，217 MB，避免 40 GB arrow 缓存）。
6. **LaTeX 工具链**：tectonic 0.15.0 入库 + `.gitignore` 加白名单；首次编译经代理拉 CTAN 包，
   之后**离线可复现**已实测。

### 阶段 B：修 B 组前置 bug（`4336584`）——不修就会把偏差固化进新缓存

- **选层口径全面改 val**：`internals_vs_output` / `logit_lens` / `plot_layer_signal`
  （后者连正则 C 也从 0.5 对齐到 1.0，因为论文取数的 `significance.py` 用的是 C=1.0）。
- **`query_id` 对齐断言**加到 6 个脚本（原先只比 labels，抓不到"标签相同、文本不同"的错配）。
- **bootstrap 逐模型 RNG 子流**（原实现共享一个流 → `--only X` 会改变其它模型的 CI）。
- **因果协议四项**：按真实标签分组、随机方向 1→12、方向层 val 选、去掉截断、统一 token 集。

### 阶段 C：8 小时无人值守全量实验（`365d61a`）

主流水线 8 模型 × 1500q（extract → judge → internals_vs_output → significance →
logit_lens → plot_layer_signal），加上第二轮：
**capacity（8 模型）、patch（4 变体）、causal（旧协议 4 变体）、prompt 鲁棒性（3 模板 × 2 模型）、
500q 受控复算（4 个 7B 模型）**。

### 阶段 D：批判性复核 TODO + 删旧 artifact（`7f792ff` `2fef18a` `8796bd5` `85c3178`）

- 对 TODO 做**批判性复核**（用户要求扮理智角色）：把已完成的、判定降级/不做的逐条标注；
  发现 `plot_causal.py` 硬编码基线 0.793 且 random 变 12 条后会 KeyError、`run_second_round.sh`
  调用参数名不符（`--in` vs `--json`）会静默失败。
- 按用户决定**删掉 9 个原主机 artifact**（先用**内容特征**而非 mtime 区分新旧，
  并在仓库外留参考副本 `/root/autodl-tmp/_old_host_artifacts_20261014/`，md5 校验一致）。
- 新增 **「⚠️ 证据纪律」**：写进论文的每个数字都必须能指到 `results/` 里一个现存且当前代码
  产出的 JSON；`scripts/collect_paper_numbers.py` → `results/PAPER_NUMBERS.json` 是取数单一来源。

### 阶段 E：四项补充实验（E-1~E-4）+ BH 校正（E-7）（`96f9cff` `df2a9fc`）

- **E-2 capacity/readout 对照（8 模型）**：见下面"关键结论"。
- **E-3 中层真实激活 patch（4 变体）**：见下面"关键结论"。
- **E-4 prompt 鲁棒性**：3 模板 × 2 模型，两个 edge **6/6 全部同号**。
- **E-7 BH 校正**：`scripts/aggregate_significance.py`，**拒收旧格式 artifact**。
- **E-8** 因果脚本补三方向 `proj_auc`（原缺 random 的）。

### 阶段 F：FiQA 换 BM25 难负例（`defe27d` `3e37d87` `5930c97`）

- **自研向量化 BM25**：`rank_bm25.get_scores` 是纯 Python 遍历词表，实测 **~2 分钟/query**
  （FiQA 57,638 段 / 235,223 词表），1500 query 要几十小时；改为**倒排 posting list + numpy**，
  **11.4 秒**完成 1500 query 排名（快约 600 倍）。
- **等价性对拍** `scripts/verify_bm25.py`：与库版逐查询比对，**最大绝对偏差 1.08e-06、
  top-20 排名逐位一致**。
- 修两个真 bug（见下"踩过的坑"）。

### 阶段 G：投稿就绪度评估（`df2a9fc`）

`docs/CCFB_READINESS.md`：逐项过审稿标准，**结论是目前不足以支撑 CCF-B 投稿，
差距不在数据量而在因果与机制论证**。

---

## 三、实验结论（现行权威口径，全部来自本机当前代码产出的 JSON）

### ✅ 站得住

| 结论 | 证据 |
|---|---|
| **`attn−final` 是 8/8 全模型唯一过 BH 校正的效应**（q<0.001） | `results/significance_summary_q1500.json` |
| 深度可解码性中部最高、向输出衰减 | 逐层曲线 8 模型一致（3B：L9 0.616 → L26 0.569 → final 0.538） |
| 零训练读出（logit-lens）处处近随机 | 3B 0.560、8B 0.577，远低于训练探针 |
| **读出自由度不足以解释优势**：给最终层**完全相同**的读出流程后 `attn − 选20块` 仍 8/8 显著 | `results/capacity_control_*` |
| prompt 鲁棒（3 模板 6/6 同号） | `results/significance_*terse*` / `*verbose*` |
| **FiQA 换难负例后第二数据集由弱变强** | `internal_edge +0.099 CI[0.066,0.13] p<2e-4`；final 从 0.976（天花板）降到 0.724 |
| 500q 对 +0.05 量级效应**无功效**（跨家族系统性假阴性） | Mistral-7B：500q +0.0069 p=0.407 → 1500q +0.101 p<2e-4 |

### ⚠️ 被削弱

- **`internal_edge` 只在 base 模型存活（3/8）**，instruct 上不显著 → 强形式不能作通用主张。
- **"读出方式"的贡献与"读哪层"相当**：换逐块 LR ensemble 后
  **LLaMA-3.2-3B-Instruct 上 `attn − headmean = +0.0071 [−0.015,+0.029]` 不显著**。

### ❌ 站不住（真正的障碍）

- **因果章节的定向性判据为 null**：改成按标签分组的 `alignment_gap` + 12 随机方向零分布后，
  internal 排不进前 5%；"internal 保持 AUC"在 3B 上与 random 完全重叠、在 Qwen 上反而崩到 0.427。
  唯一仍成立的是**消融臂单调崩塌**（3B 0.814→0.452、8B 0.886→0.307、8B-Inst 0.922→0.002）。
- **标题词 "attenuated" 被 patch 实验反向证伪**：把 val 选层的**真实**中层激活移植到输出层，
  4 个变体里 **3 个判别力反而低于 judge**（0.511<0.545、0.492<0.563、0.617<0.652）。
  证据指向 **misalignment（错位）**。

---

## 四、踩过的坑（都会复现，已修/已记入 `CLAUDE.md` §七）

| # | 坑 | 后果 / 修法 |
|---|---|---|
| 1 | `datasets` 5.x **移除 `trust_remote_code` 且是报错而非忽略** | 老代码直接崩；已去掉该参数（MS MARCO v1.1 是纯 Parquet，无脚本，口径不变） |
| 2 | **hf_xet 被 nnsight 带回来** | 与代理不兼容会卡死；装完 nnsight 要再卸一次 |
| 3 | **别用代理下载** | huggingface.co 走代理拉大分片停滞；hf-mirror 直连稳定，带代理反而卡死 |
| 4 | **GPU 阶段必须串行** | 手工并发两个实验 → CUDA OOM（7B 两进程各占 ~15-16 GB）。已踩两次 |
| 5 | **`pgrep -f` 会匹配到自己的 shell** | `pkill -f <脚本名>` 会把自己杀掉。**本次会话踩了三次**。用 PID 或 `scrip[t].py` 括号写法 |
| 6 | **`du -sm` 取整骗人** | 估算进度请用 `stat -c%s` |
| 7 | `setup_env.sh` 的 `total_mem` **属性不存在** | 会因 `set -e` 直接失败；已改 `total_memory` |
| 8 | BeIR 的 `abs(hash(q))` **逐进程随机** | FiQA 的 judge 与探针可能落在不同 query 集；已改 `zlib.crc32` |
| 9 | 四个驱动脚本**硬编码旧主机路径** | `run_scaleup.sh` 会直接 cd 失败；已改脚本相对定位 |
| 10 | **BM25 排名缓存不校验覆盖范围**（我引入的） | 50-query 缓存被 1500-query 运行命中 → 绝大多数 query 0 负例；已加覆盖校验 + 合并写入 |
| 11 | **FiQA 标签失衡（真 bug，原实验也受影响）** | `pos_by_q[q][:n_pos_per]` 静默丢弃多余正例；FiQA 36.3% 的 query 有 >2 个相关段 → 构出 2454 正/169 负（94% 正例）。已改为只从"相关段数∈[1,2]"的 query 构建 |
| 12 | **因果输出写了非法 JSON**（Python 默认写裸 `NaN`） | 实测 18 个 `NaN` 令牌，`jq` 等严格解析器拒绝；已加 `_finite()` 清洗 + `allow_nan=False` |
| 13 | **因果注入在 Qwen 上概率饱和** | α=±1 就 P(yes)∈{0,1} → gap/AUC 塌成常数，是**假象不是测量**；已改为**对数几率空间标定**（先测每单位方向平移多少 logit，再按目标位移反解 α） |
| 14 | `plot_causal.py` 硬编码基线 0.793、random 变 12 条后 KeyError | 已重写（基线从 JSON 读、random 画均值+min-max 带、新增 alignment-gap panel） |
| 15 | `run_second_round.sh` 调 `plot_causal.py --in` 而脚本收 `--json` | 参数名不符且那行带 `\|\| true`，会**静默失败**；已修 |
| 16 | `run_queue.sh` 产物存在就 SKIP | 会永远复用 test-argmax 旧产物；已改为检查"是否当前代码产出" |
| 17 | **误删了仓库里恢复出来的论文图** | 烟雾测试清理时连带删掉 `llama32_3b_layer_signal.png`；已从 git 恢复（教训：清理前先确认哪些文件在 git 里） |
| 18 | commit message 声称的改动**漏 add** | 一条 message 写了"补 rank_bm25"但 `requirements.txt` 没被 add；已单独补提交。教训：提交前用 `git diff --cached --stat` 对账 |
| 19 | 我自己引入过 TODO 悬挂引用（P4-6） | `selfcheck` 第 4 项抓到；已修。**改完文档必须跑 selfcheck** |
| 20 | 一条 commit 因 `-m "…中文…"` 引号被 shell 拆错而失败 | 改用 `git commit -F -` 从 stdin 读 message |

---

## 五、停机瞬间的精确状态

### 正在跑（已被主动终止）

`scripts/run_second_round.sh causal` —— E-1/E-11 的**因果饱和修正重跑**，队列 4 变体：

| 变体 | 停机时状态 | 重启后需做什么 |
|---|---|---|
| `llama32_3b` | 新协议**已算完**，但 JSON 含 18 个 `NaN`（汇总行 bug） | **删掉重跑**（约 41 分钟） |
| `llama31_8b` | 跑到 37 分钟被终止 | 重跑（约 60 分钟） |
| `llama32_3b_instruct` | 排队未开始 | 重跑（约 45 分钟） |
| `qwen25_7b` | 排队未开始 | 重跑（约 50 分钟） |

**必须用已修好的代码重跑**（停机那次全是旧代码）。已终止的东西：
因果计算、`run_second_round.sh`、`run_causal_after_fiqa.sh` 链式脚本、
`wake_causal_done.sh` 唤醒任务、定时提醒（都已撤掉）。**GPU 已释放（1 MiB / 0%）**。
**无半成品或损坏文件**（全部 `results/*.json` 逐个验证过解析）。

### 未能完成、且重启前无法做的

- 因果 4 变体的最终判读（等重跑）。
- 论文 `main.tex` **目前编译不过**：它引用的 `figures/causal_steer_3b.png` 已随旧 artifact 删除，
  新图是 `figures/causal_steer_llama32_3b.png`（含 alignment-gap panel）。改引用即可修好。

---

## 六、为什么停机

数据盘 `/root/autodl-tmp` 150 GB **已用 141 GB（94%），仅余 9.5 GB**。
余下计划本身只需 0–18 GB，但 9.5 GB 不足以安全运行（随时可能写失败），
且**无法为 FiQA 增加模型**（每个 +2.5 GB）。

**扩容建议**：

| 方案 | 总容量 | 效果 |
|---|---|---|
| 现状 | 150 GB（余 9.5 GB） | 跑不动新实验，随时可能写失败 |
| **建议下限** | **200 GB**（+50） | 余约 59 GB：够把论文改到可投状态（含 FiQA 扩到 8 模型，+18 GB） |
| **推荐** | **250 GB**（+100） | 余约 109 GB：投稿期间不必再管磁盘 |

磁盘构成（停机实测）：模型权重 98 GB + 激活缓存 38.6 GB + 数据集 0.5 GB + DSH harness 3 GB。
激活缓存可省：prompt 模板缓存 10.6 GB（分析已跑完，结果 JSON 已落盘）。

---

## 七、恢复步骤（扩容重启后照此执行）

```bash
# 1) 确认新盘与缓存完好
cd /root/autodl-tmp/my_paper_project
df -h /root/autodl-tmp
python scripts/selfcheck.py                        # 六项自检，应全绿
python scripts/collect_paper_numbers.py --n 1500   # 论文取数是否齐（应 8/8）

# 2) 重跑因果（先删掉含 NaN 的 3B 产物，否则会被判为"已完成"而跳过）
rm -f results/causal_steer_llama32_3b_q1500.json
bash scripts/run_second_round.sh causal            # 约 3.5 小时 GPU，串行

# 3) 因果跑完 → 按 TODO.md E-11 的**事先写死**的判读标准出结论
#    （判据：internal 的 alignment_gap 是否排进 12 个随机方向的 95% 分位）
#    → 更新 docs/CCFB_READINESS.md 与 TODO.md E-11

# 4) 再按 docs/CCFB_READINESS.md 第三节推进入：
#    必做 = E-9（论证结构改写：attenuation → misalignment）、
#           E-10（patch 扩 8 模型 + path patching）、E-12（FiQA 扩模型）
#    然后全文数字对齐（PAPER_NUMBERS.json 是唯一取数来源）+ 重编译 PDF
```

**下一步的最高优先**：因果 4 变体重跑 → E-9 论证改写。
因为"标题说 A、实验支持 B"是**必须解决才能投**的问题，不是加实验能绕过去的。
