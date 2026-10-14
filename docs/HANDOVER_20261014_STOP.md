# 停机扩容前的断点记录（2026-10-14 停机）

> 用户于 2026-10-14 下午**主动停机以扩容数据盘**。本文件记录停机瞬间的精确状态，
> 供扩容重启后接着做。恢复时先读本文件，再读 `CLAUDE.md` §六 与 `TODO.md`。

## 一、为什么停机

数据盘 `/root/autodl-tmp` 150 GB **已用 141 GB（94%），仅余 9.5 GB**。
余下计划本身只需 0–18 GB，但 9.5 GB 不足以安全运行（随时可能写失败），
且**无法为 FiQA 增加模型**（每个 +2.5 GB）。用户决定扩容。

**扩容建议（详见对话与 `docs/CCFB_READINESS.md` 的估算）**：
- **建议下限 200 GB**（+50）：余约 59 GB，够把论文改到可投状态
- **推荐 250 GB**（+100）：余约 109 GB，投稿期间不必再管磁盘

## 二、停机时正在跑什么（已被主动终止，重启后需重跑）

`scripts/run_second_round.sh causal` —— E-1/E-11 的**因果饱和修正重跑**（对数几率标定协议）。
队列 4 个变体，被终止时进度：

| 变体 | 状态（停机时） | 重启后需做什么 |
|---|---|---|
| `llama32_3b` | 新协议**已算完**，但 JSON 里有 18 个 `NaN` 字面量（汇总行 bug） | **需重跑**（约 41 分钟），用修好 NaN 的代码 |
| `llama31_8b` | 跑到 37 分钟时被 kill | 重跑（约 60 分钟） |
| `llama32_3b_instruct` | 排队未开始 | 重跑（约 45 分钟） |
| `qwen25_7b` | 排队未开始 | 重跑（约 50 分钟） |

**关键：必须用代码里已修好的版本重跑**（停机前那次全队列都是旧代码）。已修的两处：
1. `causal_steer.py` 汇总时过滤掉"全部行饱和 → gap 为 0/0=NaN"的随机方向，
   并记录 `n_random_usable/n_random_total`；
2. 落盘前用 `_finite()` 清洗 + `allow_nan=False`，**保证输出是合法 JSON**
   （Python 默认会写裸 `NaN` 令牌，`jq` 等严格解析器会直接拒绝）。

重跑命令（幂等，会自动跳过已完成的）：
```bash
cd /root/autodl-tmp/my_paper_project
export HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
# 注意：run_second_round.sh 的 causal 分支只跳过"含 calibration 字段"的产物，
# 3B 那个旧产物含 calibration 但含 NaN，所以要先删掉它强制重算：
rm -f results/causal_steer_llama32_3b_q1500.json
bash scripts/run_second_round.sh causal
```

## 三、停机时**已完成**、不受影响的工作

- ✅ **8 模型 × 1500q 全流水线**（extract / judge / internals_vs_output / significance /
  logit_lens / plot_layer_signal），结果全部入库（commit `365d61a`）。
- ✅ **四项补充实验**：capacity（8 模型）、patch（4 变体）、causal（旧协议 4 变体）、
  prompt 鲁棒性（3 模板 × 2 模型）。
- ✅ **500q 受控复算**（4 个 7B 模型），坐实"Mistral 上 attn 弱"是**欠采样**。
- ✅ **FiQA 换 BM25 难负例**（停机前不久跑完）：
  `internal_edge +0.099 CI[0.066,0.13] p<2e-4`、`attn−final +0.154`；
  best_resid 0.822 / final 0.724（原随机负例下 final 0.976 贴天花板、edge 仅 +0.0165）。
  连同修掉的两个真 bug（静默丢弃多余正例 → 标签失衡到 2454:169；BM25 排名缓存不校验覆盖）。
- ✅ **BH 多重比较校正**：`attn−final` 是 8/8 唯一存活（`results/significance_summary_q1500.json`）。
- ✅ **LaTeX 工具链**（tectonic 0.15.0 已入库）、**论文取数单一来源**
  （`results/PAPER_NUMBERS.json`）、**投稿就绪度评估**（`docs/CCFB_READINESS.md`）。

## 四、停机瞬间的"未做/待做"清单

| 事项 | 说明 |
|---|---|
| 因果 4 变体重跑 | 见上，约 3.5 小时 GPU，是**当前最高优先** |
| E-9 论证结构改写 | 把 "attenuation" 改成 "misalignment"，**决定论文能否投** |
| E-10 patch 扩到 8 模型 + path patching | 替代"衰减"叙事的唯一正面机制证据 |
| E-12 FiQA 扩模型 | 需扩容后方可（每个 +2.5 GB） |
| 论文全文数字对齐 + 重编译 | `paper/main.tex` 目前**编译不过**（引用的 `figures/causal_steer_3b.png` 已删，新图是 `causal_steer_llama32_3b.png`） |

## 五、已完成的 3B 标定结果（判读用，虽是待重跑的版本但趋势可信）

internal 方向的消融臂把 P(yes) 从 **0.814 单调压到 0.393**，
alignment gap 从 **+0.0121 单调收缩到 +0.0012**，但 **AUC 全程平**（0.522–0.546）。
即：steering 确实起作用、gap 随之后移，但**没有出现"越推越对齐"的定向性**。
按事先写死的判读标准（TODO E-11），这落在**"排不进随机方向 95% 分位"**一侧
→ 因果章节只保留**消融臂单调崩塌**。最终判读待 4 变体齐后给。

## 六、恢复步骤（扩容重启后照此执行）

```bash
# 1) 确认新盘与缓存完好
cd /root/autodl-tmp/my_paper_project
df -h /root/autodl-tmp
python scripts/selfcheck.py                    # 六项自检
python scripts/collect_paper_numbers.py --n 1500   # 论文取数是否齐

# 2) 重跑因果（先删掉含 NaN 的 3B 产物）
rm -f results/causal_steer_llama32_3b_q1500.json
bash scripts/run_second_round.sh causal

# 3) 因果跑完后：判读 4 变体 → 更新 docs/CCFB_READINESS.md 与 TODO E-11
# 4) 再按 CCFB_READINESS.md 第三节的必做项推进（E-9 改写、E-10 patch 扩容）
```
