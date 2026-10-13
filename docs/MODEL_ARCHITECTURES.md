# 模型架构实测表（8 个论文模型）

> 建立于 **2026-10-13**。数据来源：本机 HF 缓存里每个仓库的 `config.json`
> （`/root/autodl-tmp/hf_cache/hub/models--*/snapshots/*/config.json`），
> **不是**从文档誊抄的——原主机已回收，此前多处架构数字只存在于散文里。
> 用途：写论文 / 回答审稿意见 / 校准"探针容量"这类论证时**以此表为准**。

## 一、汇总表

| 模型 | 层 L | query 头 | hidden | head_dim | top-20 头特征 | 残差维度 | attn/final | 词表 | 权重 |
|---|---|---|---|---|---|---|---|---|---|
| LLaMA-3.2-3B | 28 | 24 | 3072 | 128 | 2560 | 3072 | 0.83× | 128256 | 6.44 GB |
| LLaMA-3.2-3B-Instruct | 28 | 24 | 3072 | 128 | 2560 | 3072 | 0.83× | 128256 | 6.44 GB |
| LLaMA-3.1-8B | 32 | 32 | 4096 | 128 | 2560 | 4096 | 0.62× | 128256 | 16.07 GB |
| LLaMA-3.1-8B-Instruct | 32 | 32 | 4096 | 128 | 2560 | 4096 | 0.62× | 128256 | 16.07 GB |
| Mistral-7B-v0.3 | 32 | 32 | 4096 | 128 | 2560 | 4096 | 0.62× | 32768 | 14.50 GB |
| Mistral-7B-Instruct-v0.3 | 32 | 32 | 4096 | 128 | 2560 | 4096 | 0.62× | 32768 | 14.50 GB |
| Qwen2.5-7B | 28 | 28 | 3584 | 128 | 2560 | 3584 | 0.71× | 152064 | 15.24 GB |
| Qwen2.5-7B-Instruct | 28 | 28 | 3584 | 128 | 2560 | 3584 | 0.71× | 152064 | 15.24 GB |

**参数量**（`sum(p.numel())`，实测加载）：3B=3.21B、8B=8.03B、Mistral-7B=7.25B、Qwen2.5-7B=7.62B。

## 二、三条必须记住的结论

1. **head_dim 在全部 8 个模型上都是 128**，`hidden = n_query_heads × 128` 对每个模型都成立。
   → **"head_dim 因架构而异（例如 Mistral 是 16 而 LLaMA 是 128）"是错的**，
   不要用它解释任何东西（`TODO.md` R-4 原先就是这个错法，已于 2026-10-13 更正）。

2. **attn 探针的特征数恒小于 final 探针**：top-20 头 × 128 = 2560，
   而最终层残差是 3072 / 3584 / 4096。
   → 所以「attn 探针赢是因为参数/维度更多」**方向就是反的**：它是**用更少特征**赢的。
   真正的自由度差异是 ①在 val 上**从 672/784 个头里选 20 个**的搜索步骤，
   ②20 块拼接 vs 单个向量的归纳偏置。攻击要按这两条组织，不要按"参数更多"。

3. **Mistral 是 GQA**（`num_key_value_heads=8`），但 query 头仍是 32 个、head_dim 仍是 128，
   `o_proj` 输入维度是 `n_heads × head_dim = 4096`（与 hidden 相同）。
   → 本项目取的是 `o_proj.output`，**每头 128 维的切分对 GQA 同样成立**，
   与 LLaMA 的 MHA 在"per-head 激活"这件事上可比。

## 三、复现方式

```bash
export HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_OFFLINE=1
python - <<'EOF'
import glob, json, os
for d in sorted(glob.glob(os.environ["HF_HOME"]+"/hub/models--*")):
    s = glob.glob(d+"/snapshots/*/")
    if not s: continue
    c = json.load(open(os.path.join(s[0], "config.json")))
    name = os.path.basename(d).replace("models--","").replace("--","/")
    h, hid = c["num_attention_heads"], c["hidden_size"]
    print(f'{name:36s} L={c["num_hidden_layers"]:3d} heads={h:3d} hidden={hid:5d} '
          f'head_dim={hid//h} kv_heads={c.get("num_key_value_heads")}')
EOF
```

实测输出（2026-10-13）：

```
Qwen/Qwen2.5-7B                      L= 28 heads= 28 hidden= 3584 head_dim=128 kv_heads=4
Qwen/Qwen2.5-7B-Instruct             L= 28 heads= 28 hidden= 3584 head_dim=128 kv_heads=4
meta-llama/Llama-3.1-8B              L= 32 heads= 32 hidden= 4096 head_dim=128 kv_heads=8
meta-llama/Llama-3.1-8B-Instruct     L= 32 heads= 32 hidden= 4096 head_dim=128 kv_heads=8
meta-llama/Llama-3.2-3B              L= 28 heads= 24 hidden= 3072 head_dim=128 kv_heads=8
meta-llama/Llama-3.2-3B-Instruct     L= 28 heads= 24 hidden= 3072 head_dim=128 kv_heads=8
mistralai/Mistral-7B-Instruct-v0.3   L= 32 heads= 32 hidden= 4096 head_dim=128 kv_heads=8
mistralai/Mistral-7B-v0.3            L= 32 heads= 32 hidden= 4096 head_dim=128 kv_heads=8
```

（注意 LLaMA 与 Qwen 也是 GQA：kv_heads 8 / 4。全部 8 个模型都只有 **query 头**是本项目
`o_proj.output` 切分的对象，数量即上表 `heads` 列。）
