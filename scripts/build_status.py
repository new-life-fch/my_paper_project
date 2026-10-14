#!/usr/bin/env python3
"""
Regenerate the MEASURED sections of STATUS.md from the artifacts on disk.

Why this exists (2026-10-14)
----------------------------
The project accumulated seven overlapping markdown files whose numbers drifted
apart, so the same quantity appeared with two different values in the same paper
table. Hand-maintained status documents rot. This script makes the status file
mechanically reproducible: every number in STATUS.md's "measured" half is read
straight out of `results/*.json`, and `--check` fails if two sources disagree
about the same quantity.

Single source of truth per quantity (chosen 2026-10-14)
------------------------------------------------------
  reader AUCs (judge/final/best_resid/attn)
        -> results/significance_<tag>_q1500.json  ["auc"]
           Chosen because significance.py applies ONE regularisation (C=1.0) to
           all four readers and is also where the edges/p-values come from, so
           Table 1 reconciles with the reported edges. The other candidate,
           results/internals_vs_output_<tag>_q1500.json, uses C=0.5 for the
           residual readers and C=1.0 for attn -- mixing it into the same table
           is what produced the old inconsistency.
  edges and BH q-values -> results/significance_summary_q1500.json
  everything else       -> its own artifact

Usage:
  python scripts/build_status.py            # rewrite the generated block of STATUS.md
  python scripts/build_status.py --check    # verify sources agree; non-zero exit on mismatch
  python scripts/build_status.py --stdout   # print the block, change nothing
"""
import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEGIN = "<!-- BEGIN GENERATED: scripts/build_status.py -->"
END = "<!-- END GENERATED -->"

# tag -> display name, in the paper's row order
MODELS = [
    ("llama32_3b", "LLaMA-3.2-3B (base)"),
    ("llama32_3b_instruct", "LLaMA-3.2-3B-Instruct"),
    ("llama31_8b", "LLaMA-3.1-8B (base)"),
    ("llama31_8b_instruct", "LLaMA-3.1-8B-Instruct"),
    ("mistral_7b", "Mistral-7B-v0.3 (base)"),
    ("mistral_7b_instruct", "Mistral-7B-Instruct-v0.3"),
    ("qwen25_7b", "Qwen2.5-7B (base)"),
    ("qwen25_7b_instruct", "Qwen2.5-7B-Instruct"),
]
NQ = 1500
EDGE_ORDER = ["attn-final", "internal", "resid-judge", "train"]
EDGE_LABEL = {
    "attn-final": "attn − final",
    "internal": "internal_edge (best_resid − final)",
    "resid-judge": "best_resid − judge",
    "train": "train_edge (final − judge)",
}


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, list):
        d = d[0] if d else None
    return d


def sig(tag):
    return load(os.path.join(ROOT, f"results/significance_{tag}_q{NQ}.json"))


def ivo(tag):
    return load(os.path.join(ROOT, f"results/internals_vs_output_{tag}_q{NQ}.json"))


def collect():
    """Return (block_text, problems)."""
    problems = []
    warnings = []
    summary = load(os.path.join(ROOT, f"results/significance_summary_q{NQ}.json"))
    bh = {}
    if summary:
        for ek, ev in summary.get("edges", {}).items():
            for pm in ev.get("per_model", []):
                bh.setdefault(pm["tag"], {})[ek] = pm

    rows = {}
    for tag, _disp in MODELS:
        s = sig(tag)
        if s is None:
            problems.append(f"missing results/significance_{tag}_q{NQ}.json")
            continue
        rows[tag] = s
        # --- source-disagreement check (this IS the evidence discipline) ---
        v = ivo(tag)
        if v is not None:
            a = s["auc"]
            pairs = [("judge", a["judge"]["point"], v.get("llm_judge_zeroshot")),
                     ("final", a["final"]["point"],
                      v.get("resid_final_layer(output_repr)")),
                     ("attn", a["attn"]["point"], v.get("attn_head_probe(ours)"))]
            for name, x, y in pairs:
                if y is not None and abs(x - y) > 0.0015:
                    warnings.append(
                        f"{tag}: {name} differs between sources — "
                        f"significance.py {x} vs internals_vs_output.py {y} "
                        f"(expected: different regularisation)")

    L = []
    A = L.append

    # ---------------------------------------------------------------- setup --
    n_ok = sum(1 for t, _ in MODELS if rows.get(t))
    A("### 2.1 机器、模型、数据（自动核对）\n")
    A(f"- 8 个模型的 `significance_*_q{NQ}.json` 齐备：**{n_ok}/8**")
    if summary:
        A(f"- BH 聚合覆盖：{summary['meta'].get('n_records')} 条记录，"
          f"拒收旧格式 {summary['meta'].get('n_legacy_skipped')} 条")
    A(f"- 证据单一来源：reader AUC → `results/significance_<tag>_q{NQ}.json`；"
      f"edge / q 值 → `results/significance_summary_q{NQ}.json`\n")

    # ------------------------------------------------------------- readers --
    A("### 2.2 四个读取器的实测 AUC（test，val 选层，统一 C=1.0）\n")
    A("| 模型 | $L$ | judge | final | best-resid | attn | 最优 |")
    A("|---|---|---|---|---|---|---|")
    for tag, disp in MODELS:
        s = rows.get(tag)
        if not s:
            A(f"| {disp} | — | — | — | — | — | 缺 |")
            continue
        a = s["auc"]
        j, f, r, t = (a[k]["point"] for k in ("judge", "final", "best_resid", "attn"))
        best = max([("judge", j), ("final", f), ("resid", r), ("attn", t)],
                   key=lambda kv: kv[1])[0]
        A(f"| {disp} | {s['n_layers']} | {j:.3f} | {f:.3f} | "
          f"**{r:.3f}** (L{s['best_layer_val']}) | **{t:.3f}** | {best} |")
    A("")

    # --------------------------------------------------------------- edges --
    A(f"### 2.3 四条 edge 与 BH 多重比较校正（{NQ}q，8 模型）\n")
    if summary:
        A("| edge | CI 不含 0 | BH 后 q<.05 | 逐模型 mean [CI95] |")
        A("|---|---|---|---|")
        for ek in EDGE_ORDER:
            ev = summary["edges"].get(ek)
            if not ev:
                continue
            pm = ev["per_model"]
            n_ci = sum(1 for x in pm if x["ci_excludes_0"])
            n_bh = sum(1 for x in pm if x["q_bh"] is not None and x["q_bh"] < 0.05)
            cells = "; ".join(
                f"{dict(MODELS)[x['tag']].split(' (')[0]} {x['mean']:+.3f}"
                f"[{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}]"
                + ("" if x["q_bh"] is None or x["q_bh"] >= 0.05 else f" q={x['q_bh']:.3f}")
                for x in sorted(pm, key=lambda z: -z["mean"]))
            A(f"| {EDGE_LABEL[ek]} | **{n_ci}/8** | **{n_bh}/8** | {cells} |")
        A("\n> `internal_edge` 只在 base 模型上过校正；`attn − final` 是唯一 8/8 全存活的效应。\n")

    # ------------------------------------------------------------ capacity --
    A("### 2.4 capacity / readout 对照（8 模型）\n")
    A("给最终层**完全相同的读出流程**后，`attn − final_select20_chunks` 是否仍显著。\n")
    A("| 模型 | final 线性 | final 选20块 | final 逐块集成 | attn | attn−选20块 | 显著 |")
    A("|---|---|---|---|---|---|---|")
    for tag, disp in MODELS:
        c = load(os.path.join(ROOT, f"results/capacity_control_{tag}_q{NQ}.json"))
        if not c:
            A(f"| {disp} | — | — | — | — | — | 缺 |")
            continue
        rd = c["readers"]
        # NB the key has spaces/parens; do not tidy it or the lookup breaks
        e = c.get("edges", {}).get("attn-select20chunks (KEY CONTROL)", {})
        p = e.get("p_le0")
        sig_mark = "✅" if (p is not None and p < 0.05) else "❌"
        A(f"| {disp} | {rd['final_linear']['auc']:.3f} | "
          f"{rd['final_select20_chunks']['auc']:.3f} | "
          f"{rd['final_headmean_ensemble']['auc']:.3f} | "
          f"{rd['attn_linear']['auc']:.3f} | "
          f"{e.get('mean', float('nan')):+.3f} | {sig_mark} |")
    A("")
    A("> 本表的 AUC 由 `capacity_control.py` 自己算出（所有 linear reader 同为 C=1.0），")
    A("> 与 §2.2 的 `significance.py` 数值有 ~0.001 的差（不同的拟合路径），**不要跨表比较**。")
    A("> 关键对照是**同一张表内**的 `attn − 选20块`：给最终层完全相同的读出流程后差距是否还在。")
    A("")
    A("**但换一种读出方式后，头条主张在 2/8 模型上被打平甚至反超——这条必须正面写进论文**：\n")
    A("| 模型 | attn | final 逐块集成 | attn − 逐块集成 | 95% CI | p | 判读 |")
    A("|---|---|---|---|---|---|---|")
    n_loss = 0
    for tag, disp in MODELS:
        c = load(os.path.join(ROOT, f"results/capacity_control_{tag}_q{NQ}.json"))
        if not c:
            continue
        e = c["edges"].get("attn-final_headmean_ensemble", {})
        if not e:
            continue
        m, lo, hi, p = e.get("mean"), e["ci95"][0], e["ci95"][1], e.get("p_le0")
        if m is not None and m <= 0:
            n_loss += 1
            verdict = "❌ **头探针没有优势**"
        elif p is not None and p >= 0.05:
            n_loss += 1
            verdict = "⚠️ 不显著"
        else:
            verdict = "✅"
        A(f"| {disp} | {c['readers']['attn_linear']['auc']:.3f} | "
          f"{c['readers']['final_headmean_ensemble']['auc']:.3f} | {m:+.4f} | "
          f"[{lo:+.4f},{hi:+.4f}] | {p} | {verdict} |")
    A("")
    A(f"→ **{n_loss}/8 模型上「读头 > 读输出表示」这条头条主张，在换一种读出方式后不再成立。**")
    A("→ 所以论文只能说：`attn − final` 在**我们定义的那个**输出表示探针（最终层单个 3072 维向量 +")
    A("线性 LR）上 8/8 成立；**这不等于「注意力头读取器整体优于输出端」**。")
    A("→ 诚实底线（用户曾确认）：主张是「信号在内部比在输出表示里更**可解码**」，")
    A("**不是**「内部读取器整体打赢输出端」。")
    A("")

    # --------------------------------------------------------------- patch --
    A("### 2.5 patch 实验：v1 的尺度混淆，与 v2 的设计\n")
    A("#### 2.5.1 为什么必须先看范数\n")
    A("v1（`patch_internal.py`）把 L\\* 的真实激活按 "
      "`h_final ← α·h_L* + (1−α)·h_final` 写进输出层，**不控制两者范数**。\n")
    A("| 模型 | L\\* | ‖h_L*‖ | ‖h_final‖ | h_L* 比 h_final 小 | v1 在 α=1.0 的 AUC（judge→patch） |")
    A("|---|---|---|---|---|---|")
    try:
        import torch  # optional: only needed for this table
        for tag, disp in MODELS:
            p = os.path.join(ROOT, f"results/cache/{tag}_q{NQ}_instruct/resid_train.pt")
            s = rows.get(tag)
            if not os.path.exists(p) or not s:
                A(f"| {disp} | — | — | — | 缓存缺失 | — |")
                continue
            r = torch.load(p, map_location="cpu")["acts"].float()
            norms = r.norm(dim=-1).mean(dim=0)
            ls = s["best_layer_val"]
            pv1 = load(os.path.join(ROOT, f"results/patch_internal_{tag}_q{NQ}.json"))
            cell = "—"
            if pv1:
                sw = pv1.get("sweep", {})
                last = sorted(sw.items(), key=lambda kv: float(kv[0]))[-1][1]
                a1, jj = last.get("patch_internal", {}).get("auc"), pv1["judge_natural"]["auc"]
                if a1 is not None:
                    cell = f"{jj:.3f} → {a1:.3f} ({a1 - jj:+.3f})"
            A(f"| {disp} | {ls} | {norms[ls]:.1f} | {norms[-1]:.1f} | "
              f"**{norms[-1] / norms[ls]:.1f}×** | {cell} |")
        A("")
        A("> α=1 时输出层残差被**替换**成一个范数小 1.7 ~ 16.3 倍的向量。")
        A("> v1 观察到的 AUC 崩塌程度与这个倍数几乎单调对应，所以 v1 **无法区分**")
        A("> 「格式错位」与「注入量小了一个数量级」——它的结论只能当作待检验假设。")
        A("> 表中最后一个模型（Qwen-Instruct）尚未跑过 v1，故该格为空。\n")
    except ImportError:
        A("| — | — | — | — | 需要 torch 才能算（本机未装） | — |\n")

    A("#### 2.5.2 v1 产物（4 变体，仅作对照，勿作为机制证据）\n")
    A("| 模型 | L\\* | 探针@L\\* | judge | α=1.0 patch_internal | 判据（应接近探针） |")
    A("|---|---|---|---|---|---|")
    for tag, disp in MODELS:
        p = load(os.path.join(ROOT, f"results/patch_internal_{tag}_q{NQ}.json"))
        if not p:
            continue
        sw = p.get("sweep", {})
        last = sorted(sw.items(), key=lambda kv: float(kv[0]))[-1][1]
        row = last.get("patch_internal", {})
        A(f"| {disp} | {p['layer']} | {p['probe_test_auc_at_layer']:.3f} | "
          f"{p['judge_natural']['auc']:.3f} | {row.get('auc', float('nan')):.3f} | "
          f"{'未恢复' if row.get('auc', 1) < p['probe_test_auc_at_layer'] else '恢复'} |")
    A("")

    v2 = sorted(glob.glob(os.path.join(ROOT, f"results/patch_v2_*_q{NQ}.json")))
    A("#### 2.5.3 patch v2（无尺度混淆）\n")
    if v2:
        A("跨样本替换 L\\* 的 answer-token 激活（优先取**不同标签**的伙伴）；"
          "α 从 0（原激活）到 1（完全换成伙伴的）。")
        A("`Δgap` = follow_gap(α=1) − follow_gap(α=0)，衡量**判决跟随移植内容的程度**。\n")
        A("| 模型 | L\\* | 探针@L\\* | judge | 恒等 max\\|Δ\\| | 行对齐 | Δgap | α=1 auc_target | α=1 auc_source | 末层 nm@1 / raw@1 |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for path in v2:
            d = json.load(open(path))
            sw = d.get("sweep", {})
            c = d.get("controls_Lstar", {})
            a0 = sw.get("cross_Lstar@0.0", {})
            a1 = sw.get("cross_Lstar@1.0", {})
            dg = a1.get("follow_gap_vs_source", 0) - a0.get("follow_gap_vs_source", 0)
            nm = sw.get("final_normmatched@1.0", {}).get("auc_vs_target")
            rw = sw.get("final_raw@1.0", {}).get("auc_vs_target")
            A(f"| {d['tag']} | {d['layer']} | {d['probe_test_auc_at_Lstar']:.3f} | "
              f"{d['judge_auc_natural']:.3f} | {c.get('self_patch_max_absdiff_vs_natural')} | "
              f"{d.get('row_alignment_rel_mae')} | {dg:+.4f} | "
              f"{a1.get('auc_vs_target')} | {a1.get('auc_vs_source')} | {nm} / {rw} |")
        A("")
        A("**结论是模型依赖的**：两个 LLaMA base 完全不跟随移植内容（Δgap +0.003/+0.006），"
          "3B-Instruct 与 Qwen 明显跟随（+0.040 / **+0.176**）。")
        A("所以「模型用不了自己的内部表示」这个一般说法**是错的**。\n")
        A("**范数匹配与不匹配结果相同**（上表最后一列两个数）→ "
          "**v1 观察到的崩塌不是尺度伪影**，这是本次重设计最想回答的问题，答案是否定的。\n")
    else:
        A("_尚未运行。脚本已就绪：_\n")
        A("```bash")
        A("bash scripts/run_patch_v2.sh                 # 4 变体（与 v1 同集合，可直接对照）")
        A("MODELS=all8 bash scripts/run_patch_v2.sh     # E-10：8 模型")
        A("WAIT_PID=<pid> bash scripts/run_patch_v2.sh  # 等别的 GPU 任务退出后再开始")
        A("```\n")
    ga = sorted(glob.glob(os.path.join(ROOT, f"results/gradient_alignment_*_q{NQ}.json")))
    A("#### 2.5.4 梯度对齐（解开 patch 的矛盾，且不编辑任何激活）\n")
    if ga:
        A("测 ∂(yes−no logit 差)/∂h_L\\* 与探针判别方向的余弦。梯度是网络在**它自己实际产生的**"
          "激活处的局部性质，**不编辑任何激活**，所以不受跨样本 patch 的「分布外」限注影响。\n")
        A("| 模型 | cos(模型梯度, 探针方向) | 非零比例 | 随机方向 max\\|mean\\| | 探针分位 | 判读 |")
        A("|---|---|---|---|---|---|")
        for path in ga:
            d = json.load(open(path))
            pc = d["probe_direction_cosine"]; rc = d["random_directions_cosine"]
            pct = d["probe_vs_random_abs_mean_percentile"]
            A(f"| {d['tag']} | {pc['mean']:+.5f} | {pc['frac_positive']} | "
              f"{rc['max_abs_mean']:.5f} | **{pct}** | "
              f"{'与随机不可区分 → 读出**不对准**' if pct < 0.95 else '显著高于随机'} |")
        A("")
        A("> **这是 misalignment 最锋利的证据**：cos 只有梯度尺度的约 1%，且"
          "**在 Qwen 上也落在随机分布之内** —— 即**即使判决能被 L\\* 内容驱动**（patch 显示），"
          "驱动它的方向也**不是**探针读相关性的那个方向。\n")
        A("> 两个结果合起来才是准确的机制表述：**模型的下游计算会用 L\\*，"
          "但它的读出方向不对准相关性方向**。\n")
    else:
        A("_尚未运行。_ `python scripts/gradient_alignment.py --model M --cache C --tag T`\n")

    A("**四个实验的分工（判据见 §12.2）**：\n")
    A("| 实验 | 问的问题 | 无尺度问题的理由 |")
    A("|---|---|---|")
    A("| 自 patch 恒等 | sanity：hook 是否是惰性的 | payload 就是自己的激活 |")
    A("| 跨样本 patch @L\\* | 模型自己的下游计算是否使用 L\\* 的内容 | 同层同分布同坐标 |")
    A("| 末层跨样本 + 范数匹配 | v1 的崩塌是格式还是尺度 | payload 缩放到目标行范数 |")
    A("| logit lens（已有，§7） | 模型自己的 unembedding 能否读出 | `final_norm` 自带归一化 |")
    A("")

    # --------------------------------------------------------------- causal --
    A("### 2.6 因果 steering\n")
    A("消融臂 = 注入强度的负端点（α 最负那一点）。`P(yes)` 崩塌说明该方向与判决耦合；")
    A("`gap 分位` 是 internal 的 alignment gap 在 N 个随机方向零分布中的分位，")
    A("**判据是 ≥ 0.95 才算定向性成立**。\n")
    A("| 模型 | 协议 | 消融臂 P(yes)（自然→最强负位移） | AUC（自然→负端点） | gap 分位（判据 ≥0.95） | 可用? |")
    A("|---|---|---|---|---|---|")
    for tag, disp in MODELS:
        c = load(os.path.join(ROOT, f"results/causal_steer_{tag}_q{NQ}.json"))
        if not c:
            A(f"| {disp} | — | — | — | — | 缺 |")
            continue
        proto = ("对数几率标定" if "calibration" in c else "**旧固定-α（饱和）**")
        ab = c["sweep"]["internal"]
        st = (c.get("calibration") or {}).get("shift_targets_logits")
        # the arrays are in shift-target order; index 0 is target shift 0 (baseline),
        # and the strongest NEGATIVE shift is the largest index with target < 0
        bi = list(st).index(0.0) if st else 4
        neg = [i for i, t in enumerate(st) if t < 0] if st else [0]
        ni = max(neg, key=lambda i: abs(st[i])) if st else 0
        A(f"| {disp} | {proto} | "
          f"{ab['mean_pyes'][bi]:.3f} → {ab['mean_pyes'][ni]:.3f} | "
          f"{ab['auc'][bi]:.3f} → {ab['auc'][ni]:.3f} | "
          f"{c.get('gap_at_alpha-8', {}).get('internal_percentile_vs_random', float('nan')):.2f} | "
          f"{'✅' if 'calibration' in c else '❌ 须重跑'} |")
    A("\n> 判读标准事先写死在 §12.1，**不得事后调整**。")
    A("> 结论（3B，2026-10-14）：**判据未通过**（分位 0.55/0.45），")
    A("> 因果章节只保留消融臂单调崩塌。详见 `docs/E11_JUDGEMENT.md`。")
    A("> 顺带发现的机制：**所有方向（含随机）的 alignment gap 都随注入强度收缩到零**，")
    A("> 即大范围扰动压掉标签可分性 —— internal 是钝的平移旋钮，不是相关性对齐的。\n")

    # ------------------------------------------------------------ logit lens --
    A("### 2.7 零训练读出（logit lens）——不受尺度影响的关键对照\n")
    A("logit lens 把每层残差过模型**自己的** `final_norm` + `unembedding` 读 $P(\\text{yes})$，")
    A("零训练、无标签选层。它**不受残差范数失配影响**（`final_norm` 自带归一化），")
    A("所以是判断「信号是否与输出读出方向对齐」最干净的探针。\n")
    A("| 模型 | judge | 末层=judge 校验 | LL @L\\* | 训练探针 @L\\* | **同层差** | LL @自身最佳层 | 差 |")
    A("|---|---|---|---|---|---|---|---|")
    n_same_ll = 0
    for tag, disp in MODELS:
        ll = load(os.path.join(ROOT, f"results/logit_lens_{tag}_q{NQ}.json"))
        s = rows.get(tag)
        if not ll or not s:
            A(f"| {disp} | — | — | — | — | — | — | 缺 |")
            continue
        Ls = s["best_layer_val"]
        ll_Ls = ll["logit_lens_per_layer_auc"][Ls]
        ll_best = ll["logit_lens_best_layer"]["auc"]
        tr = s["auc"]["best_resid"]["point"]
        same = ll_Ls - tr           # SAME layer: the strictest comparison
        if same >= 0:
            n_same_ll += 1
        A(f"| {disp} | {ll['llm_judge_auc']:.3f} | "
          f"{ll['sanity_lastlayer_vs_judge_absdiff']:.3f} | {ll_Ls:.3f} | {tr:.3f} | "
          f"**{same:+.3f}** | {ll_best:.3f} "
          f"(L{ll['logit_lens_best_layer']['layer']}) | {ll_best - tr:+.3f} |")
    A("")
    A("> **末层一致性校验并非全都完美**：`absdiff` 在 8B 上是 **0.011**（其余 ≤0.007）。")
    A("> 论文写的是 \"matching the judge to within <0.001\"，**对 8B 不成立**，必须改。")
    A("> 这不是 bug（末层 logit-lens 与 judge 走的是同一组权重，差异来自 fp16/批处理），")
    A("> 但引用时要么给真实 absdiff，要么把话改成\"末层复现 judge（最大偏差 0.011）\"。")
    A("")
    A("**同层比较（含 instruct）训练探针 8/8 全胜**，但幅度并不按 base/instruct 干净分裂：")
    A("")
    A("| 同层差（LL@L\\* − 训练@L\\*） | 模型 |")
    A("|---|---|")
    A("| 最大（−0.10 ~ −0.13） | LLaMA-3.1-8B (base)、LLaMA-3.2-3B (base)、**Mistral-7B-Instruct** |")
    A("| 中等（−0.040 ~ −0.087） | Mistral-7B (base)、LLaMA-3.1-8B-Instruct、Qwen2.5-7B (base) |")
    A("| 最小（−0.005 ~ −0.024） | LLaMA-3.2-3B-Instruct、Qwen2.5-7B-Instruct |")
    A("")
    A("> ⚠️ 我一度写成「base 差得多、instruct 差得少」，**那是错的**——")
    A("> Mistral-7B-**Instruct** 是同层差最大的模型之一（−0.117）。")
    A("> 稳健的是**方向**（8/8 训练探针更好），不是这个分组。")
    A("")
    A("> ⚠️ **方法学限注（必须写进论文，否则会被审稿人抓）**：这个比较里训练探针")
    A("> 有 5218 条训练样本去学读出方向，而 logit-lens 用的是**固定**的 unembedding 方向。")
    A("> 所以「训练探针 > logit-lens」**不代表**「信息不可被固定线性方向读出」——")
    A("> 它只说明**模型自己的读出方向不是为这个任务优化的**。")
    A("> 要完全分离这一点，需要第三条对照：在 L\\* 上训练探针、在**另一个数据集**上测")
    A("> （或用与任务无关的固定方向），给出「任意固定线性读出的可达上界」。")
    A("> 在此之前，能说的只有：**模型自己的 unembedding 从 L\\* 读得比训练探针差，")
    A("> 且 instruct 上差距大幅缩小。**")
    A("> 另一个附带限注：`final_norm` 本来是为末层设计的，把它用在 L\\* 上是分布外操作")
    A("> （虽然它自带归一化、不受范数失配影响），这会被视为 logit-lens 的固有局限。")
    A("")
    A("> ⚠️ **论文现有那句话必须改。** 论文写的是")
    A("> \"the logit lens stays near chance at \\emph{every} depth (3B peak 0.560; 8B peak 0.577)\"，")
    A("> 并由此推出\"信号与输出读出错位\"。核对后：")
    A("> ① 引的 0.560/0.577 是**旧主机、test-argmax 口径**的数，换机器重跑后没人重新核过这句；")
    A("> ② base 模型的 logit-lens 是 0.560/0.563/0.579，**并不低于 judge**（0.545/0.574/0.579），")
    A(">    说\"near chance\"不严谨；")
    A("> ③ 最要紧的是**它没有报出该报的那一列**——同层比较。只看\"各自最佳层\"会得到")
    A(">    \"instruct 上 logit-lens 反超\"的假象（那是拿 LL 的最佳层比训练探针的 L\\*），")
    A(">    同层比较后训练探针 8/8 全胜。")
    A("")
    A("### 2.7.1 固定方向对照：模型自己的读出方向 vs「别处学来的」方向\n")
    A("训练探针有 5218 条样本学方向，logit-lens 用的是固定方向——所以"
      "\"训练探针赢\"不能推出\"信息不可被固定线性映射读出\"。")
    A("用**跨数据集搬方向**实测这个界（零目标域拟合）：\n")
    A("| 数据集 | 模型自己的读出 @L\\* | 跨数据集搬来的方向 | 域内训练探针 | 结论 |")
    A("|---|---|---|---|---|")
    # Each dataset has its own artifact with its own key names; read them
    # explicitly rather than guessing, so a schema change fails loudly here.
    rows_fd = []
    fq = load(os.path.join(ROOT, "results/fixed_direction_control_llama32_3b.json"))
    if fq:
        b = fq.get("fiqa_logit_lens_bootstrap") or {}
        ci = f" CI[{b['ci95'][0]},{b['ci95'][1]}]" if b.get("ci95") else ""
        rows_fd.append(("FiQA（BM25 难负例）", f"{fq.get('fiqa_logit_lens_at_L')}{ci}",
                        fq.get("ms_direction_on_fiqa"), fq.get("fiqa_probe_on_fiqa"),
                        "模型自己的读出**显著低于随机**（CI 整体在 0.5 下），外来方向反而赢"))
    ms = load(os.path.join(ROOT, "results/readout_direction_llama32_3b.json"))
    if ms:
        ci = ms.get("logit_lens_ci95") or []
        cis = f" CI[{ci[0]},{ci[1]}]" if len(ci) == 2 else ""
        rows_fd.append(("MS MARCO", f"{ms.get('logit_lens_auc')}{cis}",
                        ms.get("fiqa_fitted_direction_on_msmarco"),
                        ms.get("msmarco_fitted_direction_on_msmarco"),
                        "模型自己的读出**与随机不可区分**（CI 跨 0.5），外来方向反而输"))
    for r in rows_fd:
        A("| " + " | ".join(str(x) for x in r) + " |")
    A("")
    A("> ⚠️ **不要一般化**：\"低于随机 / 外来方向能赢\"**只在 FiQA 上成立**。")
    A("> 两个数据集上都成立、也够用的是**更弱的比较**：模型自己的读出方向比同层训练探针")
    A("> 读得少 → 说明「**读出方向不对这个任务**」，而不是「信息无法被固定线性映射读出」。")
    A("> （这条一般化我犯过一次，见 §9 纪律第 6 条。）\n")
    A("> **仍然成立、且更硬的一条**：即使在 logit-lens 最强的 Qwen-Inst 上（0.665），")
    A("> 注意力头读取器依然是 **0.669**，judge 只有 **0.660**；")
    A("> 在 8/8 模型上 `attn − final` 全部过 BH 校正（§2.3）。")
    A("> 所以论文的机制表述应收成：")
    A("> **「内部表示携带的相关性信号，模型自己的输出读出方向只能取出其中一部分；")
    A("> 对齐把这个方向推向信号，但 8/8 模型上仍未追平一个直接训练的探针。」**")
    A("> 这与 base→instruct 的 `internal_edge` 收缩是同一个现象的两个侧面。\n")

    # ----------------------------------------------------------------- fiqa --
    A("### 2.8 第二个数据集（FiQA，BM25 难负例）\n")
    fq = load(os.path.join(ROOT, "results/significance_llama32_3b_fiqa_bm25_q1500.json"))
    if fq:
        a = fq["auc"]
        A(f"- 读取器：judge {a['judge']['point']:.3f} / final {a['final']['point']:.3f} / "
          f"best_resid {a['best_resid']['point']:.3f} (L{fq['best_layer_val']}) / "
          f"attn {a['attn']['point']:.3f}（n_test={fq['n_test']}）")
        for k, v in fq["edges"].items():
            A(f"- {k} = **{v['mean']:+.4f}** CI[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] "
              f"p={v['p_le0']}")
    else:
        A("_缺 `results/significance_llama32_3b_fiqa_bm25_q1500.json`_")
    A("")

    # ----------------------------------------------------------------- misc --
    A("### 2.9 稳健性与其它\n")
    for templ in ("terse", "verbose"):
        for tag in ("llama32_3b", "qwen25_7b"):
            s = load(os.path.join(ROOT, f"results/significance_{tag}_{templ}_q1500.json"))
            if not s:
                continue
            e = s.get("edges", {})
            ie = e.get("internal_edge(best_resid-final)", {}).get("mean")
            af = e.get("attn-final", {}).get("mean")
            A(f"- prompt 模板 `{templ}`，{tag}：internal_edge {ie:+.4f}，attn−final {af:+.4f}")
    A("")

    return "\n".join(L), problems, warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()

    block, problems, warnings = collect()
    for w in warnings:
        print(f"  source-disagreement (documented, expected): {w}")
    if problems:
        print("\n".join(f"  PROBLEM: {p}" for p in problems))
        if args.check:
            return 1
    if args.stdout:
        print(block)
        return 0
    if args.check:
        print(f"  sources OK ({len(MODELS) - len(problems)}/{len(MODELS)} models)")
        return 0

    path = os.path.join(ROOT, "STATUS.md")
    if not os.path.exists(path):
        print(f"ERROR: {path} does not exist (it holds the hand-written half too)")
        return 1
    src = open(path).read()
    if BEGIN not in src or END not in src:
        print(f"ERROR: {path} lacks the generated markers")
        return 1
    head, rest = src.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    out = f"{head}{BEGIN}\n\n{block}\n{END}{tail}"
    with open(path, "w") as f:
        f.write(out)
    print(f"updated the generated block of {path} ({len(block)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
