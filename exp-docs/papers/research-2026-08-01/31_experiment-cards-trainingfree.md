# 把 LatentMAS 搬到 VLM 多智能体：可执行实验方案

> **写给谁**：准备把 LatentMAS 迁到多模态 MAS 上做 baseline 的人。
> **环境事实**（本机实测）：8× RTX 6000 Ada 48GB / transformers 4.57.1 / torch 2.9.1+cu128 / vllm 0.18.0；HF 缓存里已有 `lmms-lab--POPE / GQA / MME / MMVet / ScienceQA / HallusionBench`、`DogNeverSleep--AMBER`（多数只有 metadata，需重新拉数据）；**没有** Qwen3-VL 权重，有 llava-1.5-7b 的 lock 文件与 Qwen2/2.5-VL-7B 的 SENTINEL 微调版目录。
> **前置结论（来自你已有的 `/home/yilin/LatentMAS/exp-docs/code-to-paper-mapping.md`，直接复用不重做）**：`--latent_space_realign` 默认关（$W_a=I$）、tied embedding 下 $W_a\approx I$、hierarchical 实为串行、vLLM 路径根本不传 KV、`padding_side=right` 会取到 pad 位隐状态。这些坑在多模态版里**全部继承**，下面每张卡都假定你已经绕开它们。

---

## 0. 先把命题说清楚：LatentMAS 是两个可分离的机制

很多人把 LatentMAS 当一个整体，这会让实验设计糊掉。它其实是两件互不依赖的事：

| | 机制 | 代码位置 | 迁到 VLM 的风险 |
|---|---|---|---|
| **M1** | agent **内部**潜在思维：$h \to W_a \to$ `inputs_embeds`，自回归 $m$ 步不解码 | `/home/yilin/LatentMAS/models.py:321-349` | **中**。$W_a$ 的闭式解只对文本词表成立，但它做的事是「把 $h$ 拉回输入嵌入空间」——在 VLM 里这个空间被视觉 projector 输出**扩张**过了，$W_a$ 会把视觉信息强行压回文本流形 |
| **M2** | agent **之间**通信：逐层 KV 前置（`past_key_values` 从不清空） | `/home/yilin/LatentMAS/methods/latent_mas.py:130-135` | **高，但坑不在你以为的地方**。KV 是 post-embedding 的东西，视觉 token 进了 LLM 就是普通位置，机制上「应该」直接可用；真正会炸的是 **①位置编码（Qwen2-VL/2.5-VL 的 mRoPE 是 3D 的）②视觉 token 数量导致 KV 链爆炸** |

**关键判断**：你最关心的 $W_a$ 理论断点（M1）**其实不是这个项目的主要风险**——因为 repo 里 $W_a$ 默认就是单位阵，论文的主要收益来自 M2。**真正决定项目生死的是 M2 在多模态下能不能传得动视觉信息**。所以下面的假设排序把 M2 放在前面。

另一个必须先摆上桌的事实：**这块地已经被占掉大半**。L²-VMAS（ICML 2026 已接收）占了 framing、MACF 占了「潜在通信优于文本通信」的定量证据、VisionWormhole 占了「VLM 视觉口当通信信道」。**你唯一确定还空着的地是 training-free**——四篇全部需要训练。所以整套实验的设计目标不是「证明多模态 latent MAS 有用」（已被证明），而是**「证明 training-free 的多模态 latent MAS 可行 / 或者干净地证伪它并解释为什么」**。后者同样能成文，见 §E。

---

## A. 假设拆解：6 条可单独证伪的子假设

每条写成「若 H 成立，则观测量 X 应 >/< 阈值」。阈值都是可测的，不是形容词。

### H1 — $W_a$ 在 VLM 上非平凡，且视觉嵌入与文本嵌入的几何错位是可量化的

> **若 H1 成立**：$\|W_a - I\|_F / \|I\|_F > 0.1$，**且**视觉 token 嵌入（projector 输出）落在「文本嵌入表 top-256 主成分子空间」中的能量占比 $< 3\times$ 随机基线（$3 \times 256/d$）。
>
> **若 H1 被证伪**（两种方向，含义完全不同）：
> - (a) $\|W_a-I\|_F/\|I\|_F < 0.01$（tied embedding）→ **$W_a$ 这个理论断点是个伪问题**，因为原版就是恒等映射，「$W_a$ 不适用于视觉」这句话没有可攻击的对象。你的论文不能拿它当卖点。
> - (b) 视觉嵌入能量占比很高（$>0.8$）→ **VLM 的 projector 在训练时已经把视觉特征对齐到文本嵌入流形上了**，$W_a$ 照搬过来无害。这是好消息（training-free 可行性 +1），但同时意味着「$W_a$ 的多模态替代」这个改进方向没有增量。

### H2 — 逐层 KV 前置在 VLM 上仍然数值等价（Theorem 3.3 的多模态版）

> **若 H2 成立**：对同一序列，「一次性前向 `[A1 tokens][A2 tokens]`」与「A1 前向拿 past → A2 带 past 前向」在 A2 位置上的 logits 满足 $\max|\Delta \text{logit}| < 0.05$（bf16 口径），top-1 token 一致率 $> 99\%$。
>
> **若被证伪**：说明 KV 前置在 VLM 上**不是无损的**，最可能的根因是位置编码（见 H3）。这是硬阻塞——H2 不过，后面所有卡的结果都无法归因。

### H3 — mRoPE / 3D 位置编码是多模态特有的失败点，且必须显式接管

> **若 H3 成立（即「确实是个坑」）**：在 Qwen2.5-VL 上**不显式传 `position_ids`** 时，H2 的 $\max|\Delta\text{logit}| > 1.0$；**显式接管后**降回 $< 0.05$。而在 LLaVA-1.5（1D RoPE，图像 token 只是 576 个普通位置）上，两种做法差异 $< 0.05$。
>
> **若被证伪**（HF 的 `rope_deltas` 机制恰好处理对了）：省一大块工程量，直接上 Qwen2.5-VL。
>
> **为什么单列一条**：LatentMAS 原文/原码是纯文本，`past_key_values` 跨 agent 拼接时位置是自然连续的。Qwen2-VL 系的 `get_rope_index` 依赖 `image_grid_thw` 重算 3D 位置，且 `self.rope_deltas` 是**跨调用缓存**的实例状态——跨 agent 复用时它记的是上一个 agent 的 delta。这个坑四篇论文没有一篇写过（MACF 用「输入嵌入层通信」绕过了它，VisionWormhole 直接放弃 KV 路线），**你把它写清楚本身就是贡献**。

### H4 — 视觉信息真的经 KV 通道传到了下游 agent（★项目生死线★）

> 构造「必须看图才能答」的任务，A1 看图，A2 **不看图**只接 A1 的 KV。四条对照：
> - **U**（上界）：A2 直接看图作答
> - **L**（下界）：A2 只看问题、不看图、不接 KV
> - **T**（文本）：A1 看图 → 输出自然语言 → A2 读文字作答
> - **X**（潜在）：A1 看图 + $m$ 步 latent → A2 接 `past_kv`
>
> **若 H4 成立**：$X - L > 5\text{pp}$，且配对 McNemar 检验 $p<0.01$；同时 $X \ge T - 2\text{pp}$。
>
> **若被证伪**：$X - L < 2\text{pp}$ → **KV 通道没传过去视觉信息**，training-free 路线原地死亡，转 §E 的备选方向。

### H5 — 收益来自「传了信息」而不是「多占了几个位置 / 多算了几次前向」

> 同一 pipeline，只换 KV 来源：真 KV / **别的题的 KV（shuffle）** / 高斯随机 KV（匹配逐层均值方差）/ 全零 KV / 只保留 latent 那 $m$ 步的 KV。
>
> **若 H5 成立**：真 KV $-$ shuffle KV $> 4\text{pp}$，且 shuffle KV $\approx$ 随机 KV $\approx$ L（下界）。
>
> **若被证伪**：shuffle 与真 KV 差 $< 1.5\text{pp}$ → **整个 latent 通信是安慰剂**，收益来自「前缀让模型进入了某种更好的解码状态」而非信息传递。
>
> **这条便宜、致命、且四篇论文一篇都没做**。无论正反都能单独成一节。

### H6 — 多模态下 latent 通信仍然省成本

> **若 H6 成立**：在**同时报三个口径**（生成 token 数 / 端到端墙钟 / 峰值显存）下，Latent-MAS 相对 Text-MAS 墙钟 $<0.85\times$、显存 $<1.5\times$；**且相对 single-agent 的墙钟 $<3\times$**。
>
> **若被证伪**：视觉 token 的 KV 链让 prefill 成本超过省下的解码成本 → 效率论证反噬。
>
> **必须同时给 single-agent 列**——L²-VMAS 报的「−21.3~−44.8% token」只相对文本 VMAS，相对单智能体仍贵 ~5.4×；VisionWormhole 的 Table 1 有 7 个配置是**负加速**（0.57×~0.96×）。别重蹈。

### H7（可选，低优先）— 异构 backbone 间的 training-free KV 传递可行

> **若成立**：Qwen2.5-VL-3B → Qwen2.5-VL-7B（同族异尺寸）的 X 仍满足 $X-L>5\text{pp}$。
>
> **VisionWormhole Appendix F 已经替你证伪了跨厂商的情形**（Qwen+Gemma GSM8K 全程 ≤0.5%），引用即可，别重跑。同族异尺寸是它没测的空白。

---

## B. 实验卡（按性价比排序）

### 卡 1 — $W_a$ 退化检查 + 视觉/文本嵌入几何测距 ⏱ **1 小时 / 1 卡 / ~25 行**

对应 **H1**。这是全项目第一件事，因为它决定「$W_a$ 断点」这个你最关心的理论问题**到底存不存在**。

**输入**
- 模型：`llava-hf/llava-1.5-7b-hf`（1D RoPE，576 个固定图像 token，Vicuna backbone，untied）+ `Qwen/Qwen2.5-VL-7B-Instruct`（mRoPE，动态 token 数）
- 数据：任意 3 张图（COCO 随便抓），不需要 benchmark

**操作**

```python
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
mid = "llava-hf/llava-1.5-7b-hf"
m = LlavaForConditionalGeneration.from_pretrained(mid, dtype=torch.float32, device_map="cuda:0")
proc = AutoProcessor.from_pretrained(mid)

Win  = m.get_input_embeddings().weight.float()      # [V, d]
Wout = m.get_output_embeddings().weight.float()     # [V, d]
G  = Wout.T @ Wout + 1e-5*torch.eye(Wout.shape[1], device=Wout.device)
Wa = torch.linalg.solve(G, Wout.T @ Win)
I  = torch.eye(Wa.shape[0], device=Wa.device)
print("rel||Wa-I||_F =", ((Wa-I).norm()/I.norm()).item())
print("tied =", m.config.text_config.tie_word_embeddings)

# 视觉侧：projector 输出（= 真正进 inputs_embeds 的视觉向量）
px = proc(images=img, text="<image>", return_tensors="pt").to("cuda:0")
vis = m.get_image_features(px["pixel_values"])[0].float()      # [576, d]

print("norm  vis=%.3f  text=%.3f" % (vis.norm(dim=-1).mean(), Win.norm(dim=1).mean()))
mu = Win.mean(0)
_,_,V = torch.pca_lowrank(Win-mu, q=256)                        # 文本嵌入 top-256 主子空间
e_in  = ((vis-mu) @ V).pow(2).sum() / (vis-mu).pow(2).sum()
print("energy of vis in text-top256 subspace = %.3f (random baseline %.3f)"
      % (e_in, 256/Win.shape[1]))
# 关键：真实 h 经 Wa 之后落在哪
h = m(**px, output_hidden_states=True).hidden_states[-1][0,-1].float()
a = h @ Wa; a = a * (Win.norm(dim=1).mean()/a.norm())
print("cos to nearest text-emb  =", torch.nn.functional.cosine_similarity(a[None], Win).max().item())
print("cos to nearest vis-token =", torch.nn.functional.cosine_similarity(a[None], vis).max().item())
```

**期望观测**
| 量 | 若「$W_a$ 断点真实」 | 若「断点是伪问题」 |
|---|---|---|
| `rel||Wa-I||_F` | $>0.1$ | $<0.01$（tied） |
| 视觉能量占比 | $<0.19$（$3\times 0.0625$） | $>0.8$ |
| $\cos$ to nearest vis-token | $\ll$ to nearest text-emb | 两者相当 |

**判决标准**
- `rel||Wa-I||_F < 0.01` → **$W_a$ 在这个 backbone 上就是恒等映射**，「$W_a$ 不适用于视觉」不能当论文卖点，改把重心全放在 M2（KV 通信）。
- 视觉能量占比 $>0.8$ → projector 已把视觉对齐到文本流形，$W_a$ 照搬无害，**training-free 可行性 +1**，同时「$W_a$ 多模态替代」这个改进方向失去动机（D2 降级）。
- 视觉能量占比 $<0.19$ 且 `rel||Wa-I||_F > 0.1` → 断点真实，D2 是有价值的改进方向。

**成本**：1 GPU × 1h（主要是下权重）。**这张卡不跑完不要写任何一行 pipeline 代码。**

---

### 卡 2 — KV 前置数值等价性 + mRoPE 接管 ⏱ **3 小时 / 1 卡**

对应 **H2 + H3**。这是硬阻塞检查。

**输入**：LLaVA-1.5-7B 与 Qwen2.5-VL-7B-Instruct 各一份；构造 `seq_A =` 含图 prompt（约 600 token），`seq_B =` 纯文本 prompt（约 80 token）。

**操作**
1. **路径 P1**（金标准）：把 A、B 的 `inputs_embeds` 沿序列维拼成一条，一次前向，取 B 段的 logits。
2. **路径 P2**（LatentMAS 机制）：A 前向拿 `past_key_values` → B 带 past 前向，取 logits。
3. 对 Qwen2.5-VL 跑 **三个变体**：(a) 什么都不管（依赖 HF 的 `rope_deltas`）；(b) 手动传连续 `position_ids = arange(past_len, past_len+L_B)`；(c) 手动用 `get_rope_index` 算 B 段自身 3D 位置再整体偏移 `past_len`。
4. 报 `max|Δlogit|`、top-1 一致率、以及 B 段每个位置的 $\Delta$ 曲线（看是否随距离衰减）。

**期望观测**
- LLaVA-1.5：P1 vs P2 `max|Δ| < 0.05`（bf16 噪声量级），top-1 一致 $>99\%$。
- Qwen2.5-VL 变体 (a)：`max|Δ| > 1.0`，top-1 一致率显著掉。变体 (b)/(c) 之一回到 $<0.05$。

**判决标准**
- LLaVA 上 P1≈P2 → **M2 机制在 1D RoPE 的 VLM 上原生可用**，立刻用 LLaVA 当主实验台。
- Qwen2.5-VL 变体 (a) 不等价、(c) 等价 → **H3 证实，mRoPE 必须显式接管**；把这段写进论文的 method（这是四篇都没覆盖的技术点）。
- 三个变体全不等价 → mRoPE 系 VLM 在 training-free KV 前置下**结构性不兼容**，这本身是一个可发表的负面结论，同时意味着主实验台必须锁死在 LLaVA/1D-RoPE 系。

**成本**：1 GPU × 3h（含调 position_ids 的试错）。

---

### 卡 3 — 盲 agent 视觉信息可达性探针 ⏱ **1 卡 × 8h + 2 人天工程** ★核心★

对应 **H4**。这是决定项目生死的一张卡，也是**整套方案里唯一必须自己搭 pipeline 的部分**。

**输入**
- 模型：LLaVA-1.5-7B（主）；若卡 2 通过则加 Qwen2.5-VL-7B
- 数据：**POPE**（`lmms-lab/POPE`，yes/no 存在性判断，chance = 50%，n 取 1000）+ **GQA** 平衡子集（`lmms-lab/GQA`，开放短答，n 取 500）
- 关键设计：**问题本身不能泄露答案**。POPE 天然满足（"Is there a dog in the image?"）。GQA 要过滤掉可以靠常识猜的题。

**操作**：两 agent 的最小 MAS（不是 4 个，先把机制测干净）

| 臂 | A1 | A2 | 说明 |
|---|---|---|---|
| **U** | — | 看图 + 问题 | 上界 |
| **L** | — | 只有问题 | 下界（盲猜） |
| **T** | 看图，输出 ≤128 token 描述 | 问题 + A1 文本 | Text-MAS |
| **X** | 看图 + $m{=}10$ 步 latent | 问题 + A1 的 `past_kv`，**自己不带图** | Latent-MAS |
| **X-noW** | 同上但 $W_a=I$（默认配置） | 同上 | 隔离 $W_a$ 的贡献 |

$m \in \{0, 5, 10, 20\}$ 扫一遍（$m=0$ 是 repo 默认，会**静默退化成单 agent**，见 `run.py:102`）。

**期望观测**
- U ≈ 87%（POPE 上 LLaVA-1.5 的公开量级），L ≈ 50–60%（POPE 的 label 有先验偏置，实测再定）
- 若 KV 通道有效：X ∈ [70%, 85%]
- T ≈ 75–82%

**判决标准**（POPE n=1000，配对 McNemar）
| 结果 | 判决 |
|---|---|
| $X-L > 5\text{pp}$，$p<0.01$ | **H4 证实** → 继续卡 4/5，项目成立 |
| $2\text{pp} \le X-L \le 5\text{pp}$ | 边缘 → 先上 D1（共享视觉前缀 KV）再复测一次，仍不过则按证伪处理 |
| $X-L < 2\text{pp}$ | **H4 证伪** → 触发 §E 止损线一 |
| $X > T$ | 强结论，可作主卖点（且能正面回应 MACF 的 LatentMAS* 负结果） |

**必须同时记录**：A2 的输出里有没有出现「我看不到图像」类拒答（这会直接暴露 KV 里没有视觉证据），以及 A2 输出的 PPL（抄 VisionWormhole 的健康度诊断手段）。

**成本**：1 GPU × 8h 推理；工程 2 人天（见 §C）。

---

### 卡 4 — 破坏性/安慰剂对照 ⏱ **1 卡 × 6h（复用卡 3 pipeline，几乎零工程）**

对应 **H5**。改一个函数就能跑，性价比极高，且**四篇论文一篇都没做**。

**操作**：卡 3 的 X 臂，只替换 `past_kv` 的来源：
1. 真 KV（基准）
2. **shuffle KV**：把 batch 内**别的题**的 A1 KV 传给 A2（位置数、长度完全一致）
3. **随机 KV**：逐层用匹配真 KV 均值/方差的高斯噪声
4. **全零 KV**
5. **latent-only KV**：只保留 A1 那 $m$ 步的 KV（丢掉 prompt 段）——注意 repo 里这个逻辑**已经写好了**（`methods/latent_mas.py:40-44` + `_truncate_past`，`:61-79`），只是没在 argparse 注册，加两行即可
6. **prompt-only KV**：只保留 A1 的 prompt 段，丢掉 latent 步

**期望观测**：真 KV $\gg$ shuffle ≈ 随机 ≈ 全零 ≈ L

**判决标准**
- 真 $-$ shuffle $> 4\text{pp}$ → H5 成立，收益确实来自信息传递
- 真 $-$ shuffle $< 1.5\text{pp}$ → **安慰剂**，立刻触发 §E 止损线二
- latent-only ≈ 真 KV → 说明有效载荷全在那 $m$ 步 latent 上，**这是 D3（选择性传输）最强的动机**，也是最省显存的配置
- prompt-only ≈ 真 KV 而 latent-only ≈ L → 说明 $m$ 步 latent 思维是无效的，M1 可以整个砍掉

**成本**：1 GPU × 6h，工程 < 半天。

---

### 卡 5 — 完整 4-agent MAS 主表 ⏱ **4 卡 × 2 天**

对应 **H4/H6 的规模化验证**。**卡 1–4 全部通过之前不要开这张卡。**

**输入**
- Backbone：LLaVA-1.5-7B + Qwen2.5-VL-7B（若卡 2 允许）+ Qwen2.5-VL-3B（做规模趋势）
- Benchmark：POPE / MME / MMVet / ScienceQA-IMG / HallusionBench（全在本机 HF 缓存的清单里）。**别把 MMBench 当主评测集**——L²-VMAS 在上面出现 3 处负增益。
- 方法臂：Single / Text-MAS / Latent-MAS($W_a$=I) / Latent-MAS($W_a$) / Latent-MAS+D1 / Latent-MAS+D3
- 拓扑：sequential（**注意 hierarchical 在 latent 路径下实为串行**，`methods/latent_mas.py:91`，两栏不可直接对比）

**期望观测**：Latent-MAS $>$ Single $\ge$ Text-MAS（L²-VMAS 已证文本 VMAS 在多轮下会跌破单智能体）

**判决标准**
- Latent-MAS $-$ Single $> 2\text{pp}$ 且跨 $\ge3$ 个 benchmark 一致 → 主结论成立
- Latent-MAS $<$ Single → 见 §E 止损线三

**必须做**：3 个 seed，报均值±std。四篇论文全部单次运行无误差棒，你带上这个就有硬优势。

**成本**：4 GPU × 2 天（6 臂 × 5 benchmark × 3 seed × 3 backbone）。

---

### 卡 6 — 效率三口径 + KV 规模压力 ⏱ **1 卡 × 4h（与卡 5 同跑）**

对应 **H6**。

**操作**：记录 ①生成 token 数 ②端到端墙钟（计时起点打在模型加载之后，`run.py:129` 已经是对的；但**要把沙箱执行时间剔出去**）③峰值显存 ④KV 链长度随 agent 数的增长曲线。扫 agent 数 $N\in\{1,2,4,6,8\}$。

**期望观测**：KV 链长度 $\approx \sum_i (t_i + n_{\text{vis},i} + m)$，其中 $n_{\text{vis}}$ 对 LLaVA 是 576/agent。$N=8$ 时视觉 token 就占 4608 个位置。

**判决标准**：$N=8$ 下 OOM 或墙钟 $>3\times$ single → D1（共享视觉前缀）从「改进」升级为「必需品」。

---

### 卡 7 — 同族异尺寸异构（可选，最后做）⏱ **2 卡 × 3 天**

Qwen2.5-VL-3B → 7B。跨厂商不用做，**VisionWormhole Appendix F 已经证伪**（引用即可，且要注明它证伪的是「词表配对的闭式算子」这一具体做法，不是 LatentMAS 思想本身；而且那份实现 `methods/latent_mas_hybird.py:63/76/81` 还有最小二乘两处 X 矩阵不一致的 bug）。

---

## C. 最小可行起点：拿哪个 repo、改哪几个文件

### 骨架选择：**`/home/yilin/LatentMAS` 原版，不是 fork**

理由：
- `heterogeneous-latent-mas` **开箱即崩**（`run.py:20-36` 无条件 import 6 个不存在的模块），而且它**完全放弃了 KV 路线**（只写一层输入嵌入），9 个 benchmark 全是纯文本，唯一的「图」是纯白 224×224。它对你的价值只有两个文件级的零件，抄过来就行。
- `ViF` 更糟：无 `__init__.py`、无 setup.py、核心「视觉流」一行未实现、`base_stub.py:12` 是 `torch.randn`。当伪代码看。
- LatentMAS 原版 1651 行、机制只有两处（`models.py` 的 12 行循环 + `latent_mas.py` 那个不清空的 `past_kv`），**改动面可控**。

### 要抄的零件（两个文件，共 ~200 行）

| 抄什么 | 从哪抄 | 用途 |
|---|---|---|
| `_find_image_positions` + `_infer_special_token_ids` | `/home/yilin/tmp/mm-latent-repos/heterogeneous-latent-mas/methods/vision_latent_mas_codec_new.py:669-806` | 跨 6 种 VLM 定位 image-token span，含 fallback。这是绕不开的脏活，别自己写 |
| `_load_model_without_meta_init` | 同 repo `models.py:162-204` | transformers 5.x 的 meta-init/tied-weight 补丁。**与你 memory 里「meta 加载 non-persistent buffer → loss=nan」同源** |
| Key-Norm 显著性代理 `torch.norm(K, dim=-1)` | `/home/yilin/tmp/mm-latent-repos/ViF/vif/utils/selection.py:9-11` | D3 选择性传输用。FlashAttention 下拿不到 attention 矩阵时的替代，与你的 LMCache/blend 基础设施天然兼容 |

### 要改的文件（5 个）

1. **`/home/yilin/LatentMAS/models.py`** — 主要工作量
   - `ModelWrapper.__init__` 加 VLM 分支：`AutoProcessor` + `LlavaForConditionalGeneration` / `Qwen2_5_VLForConditionalGeneration`
   - `prepare_chat_batch` → `processor.apply_chat_template(...)` + `processor(images=..., text=...)`
   - **`generate_latent_batch`（`:290-349`）是核心改动点**：首次前向必须带 `pixel_values`（+ Qwen 系还要 `image_grid_thw`）；**后续 latent step 只能传 `inputs_embeds`，绝对不能再传 `pixel_values`**（会重复插入图像特征）。这一点原码天然满足（`:340` 已经是 `inputs_embeds=latent_embed`），只需保证首次调用分流。
   - **显式接管 `position_ids`**（卡 2 的产物），别依赖 `self.rope_deltas`
   - **先把 `tokenizer.padding_side = "left"`**（`models.py:314` 的 `hidden_states[-1][:, -1, :]` 在 right padding 下取的是 pad 位）

2. **`/home/yilin/LatentMAS/data.py`** — 加 mm loader，统一成 `{question, image, gold}`；先只接 POPE 和 GQA

3. **`/home/yilin/LatentMAS/prompts.py`** — 删掉 `:7` 的 `assert "qwen" in args.model_name.lower()`；加 VLM 版角色 prompt（把「plan 以 latent KV 格式提供」那句改成视觉版）

4. **`/home/yilin/LatentMAS/methods/latent_mas.py`** — 加开关：`--vision_agents {first,all}`（只有 A1 带图 / 每个 agent 都带图）；把 `latent_only` / `sequential_info_only` 注册进 argparse（逻辑 `:40-44` + `:61-79` 已存在，白捡）；加 `--kv_source {real,shuffle,random,zero}` 给卡 4

5. **`/home/yilin/LatentMAS/run.py`** — `:91` 的 `choices` 放开（现在写死三个 Qwen3 且 4B 重复、8B 缺失）；`--latent_steps` 默认值从 0 改成报错（避免静默退化）

### Milestone

| # | 目标 | 判定 | 预计 |
|---|---|---|---|
| **M0** | 卡 1 跑完，知道 $W_a$ 是不是恒等、视觉嵌入离文本流形多远 | 两个数字进文档 | 1 天 |
| **M1** ★ | **卡 2 通过：`max|Δlogit| < 0.05`** | 这是「多模态 LatentMAS 机制成立」的第一个硬证据 | 3 天 |
| **M2** ★★ | **卡 3 的四条数字（U / L / T / X）全部落表** | 项目生死判决点 | +1 周 |
| **M3** | 卡 4 的六条 KV 来源对照落表 | 决定 D3 是否值得做 | +3 天 |
| **M4** | 卡 5 主表 | 论文骨架 | +2 周 |

**M2 是唯一真正重要的里程碑**。M0/M1 是两天的事，M2 之前不要写任何主表代码。

### 一个强烈建议：**主实验台先锁 LLaVA-1.5-7B**

- 1D RoPE，图像 token 就是序列里 576 个普通位置 → **完全绕开 mRoPE 混淆**
- untied embedding → $W_a$ 非平凡（卡 1 会确认）
- POPE/MME/GQA 全是它的标准评测集，公开数字满地都是，你的 U 上界有外部锚点
- 机制在它上面测干净了，再迁 Qwen2.5-VL 就只剩位置编码一个变量

四篇论文没有一篇用 LLaVA-1.5 当主台（都追新 backbone），这反而让你的机制实验更容易被审稿人相信——**变量少**。

---

## D. 算法改进方向（5 个）

### D1 — 共享视觉前缀 KV（Shared Visual Prefix KV）★优先级最高★

**改什么**：图像 token 的 KV **只算一次**，作为所有 agent 共享的前缀；每个 agent 只把自己的 role prompt + $m$ 步 latent 追加到共享前缀之后。序列布局从

```
[A1_prompt+img][A1_latent][A2_prompt+img][A2_latent][A3_prompt+img]...
```
改成
```
[IMG_KV 共享]  [A1_role][A1_latent][A2_role][A2_latent][A3_role]...
```

**解决哪个失败机制**
- **H4 失败的最可能原因**：下游 agent 拿到的 KV 里视觉证据被 A1 的 prompt/latent 稀释了。共享前缀让**每个 agent 都直接对原始视觉 KV 做注意力**，视觉证据不再需要「经过 A1 转述」。
- **H6/卡 6 的 KV 爆炸**：LatentMAS 原码里题面被重复编码 4 次（`methods/latent_mas.py:93-107`，`context` 恒为 `""`），多模态下变成**图被重复编码 4 次**，$N=8$ 时 4608 个视觉位置。共享前缀把它压回 576。

**为什么这样改能行**：这正是 prefix caching 的标准做法，你手上就有 LMCache/blend 的基础设施（`/home/yilin/LMCache`），APC 常态开启的配置已经跑通（见你的 blend-APC 兼容项目）。工程上是熟路。

**怎么验证**：卡 3 的 X 臂加一个 X+D1 臂，直接比 $X_{D1}-X$；同时卡 6 报 $N=8$ 下的显存/墙钟。**判决**：$X_{D1} - X > 3\text{pp}$ 或显存降 $>50\%$ 即成立。

**风险**：共享前缀意味着所有 agent 的图像位置编码相同（都是 0..575），而它们各自的 role prompt 起始位置不同——需要确认 attention mask 与 position_ids 的构造正确（用卡 2 的等价性测试当参照系；**注意 memory 里「参照系必须独立」那条教训，别用同一套假设自检**）。

---

### D2 — $W_a$ 的多模态替代：对齐目标选谁

**改什么**：原版 $W_a=(W_{out}^\top W_{out}+\lambda I)^{-1}W_{out}^\top W_{in}$ 的目标是文本嵌入表 $W_{in}$。三个候选替代：

| 变体 | 目标矩阵 | 是否 training-free |
|---|---|---|
| **D2-a**（对照） | $W_{in}$（原版） | ✅ |
| **D2-b 样本级视觉锚点** | 把 $W_{in}$ 换成 $[W_{in}; V_{\text{img}}]$，其中 $V_{\text{img}}$ 是**当前样本的 576 个 projector 输出向量**，重解最小二乘 | ✅（每样本一次 $d\times d$ solve，$d$=4096 时约 0.1s，可接受） |
| **D2-c 子空间投影** | 把 $h$ 分解成「文本嵌入主子空间分量 + 残差」，两部分各自归一后加权重组 | ✅ |
| **D2-d 全不做** | $W_a=I$ + 模长归一（repo 默认行为） | ✅ |

**解决哪个失败机制**：H1 若显示视觉嵌入能量占比 $<0.19$（错位真实），则原版 $W_a$ 会在每一步 latent 把视觉信息**压回文本流形**，$m$ 步之后视觉证据被磨光——这直接解释 H4 失败。

**为什么这样改能行**：D2-b 的核心洞察是——**「输入嵌入空间」在 VLM 里不是 $W_{in}$ 的行空间，而是 $W_{in} \cup \text{proj}(\text{ViT})$ 的并**。闭式最小二乘对「扩张后的目标集合」照样成立，只是要把视觉锚点加进去。这是对 $W_a$ 理论断点**最直接的 training-free 修补**，而且四篇论文一篇都没讨论过对齐算子（L²-VMAS 完全没碰，MACF 用学出来的 MLP 绕过，VisionWormhole 是回避）。

**怎么验证**：
1. **中间量**：每步 latent 后测 $\cos(\text{aligned}, \text{nearest visual token})$ 随 $m$ 的衰减曲线。D2-a 应该单调衰减，D2-b 应该保持。
2. **下游**：卡 3 的 X 臂四个变体全跑。**判决**：D2-b $-$ D2-a $> 3\text{pp}$ 即成立。
3. **前置条件**：卡 1 若显示 `rel||Wa-I||_F < 0.01`（tied）或视觉能量占比 $>0.8$，**D2 整个降级为附录消融**，不值得当主卖点。

---

### D3 — 选择性 KV 传输：只传该传的那 ~2%

**改什么**：不把全部视觉位置的 KV 往下传，用 **Key-Norm 显著性代理**（$\|K\|_2$，`ViF/vif/utils/selection.py:9-11`）选 top-$p\%$ 视觉位置的 KV 传给下游，其余丢弃。$p\in\{1,5,20,100\}$。**丢弃时保留原 position_ids，不重排**（序列里留「洞」，这和 KVCOMM/blend 的做法一致，你有经验）。

**解决哪个失败机制**：H6 的带宽/显存爆炸。ViF 已经替你测过：中间层丢掉 50% 的 **inactive** 视觉 token 只掉 **2.3 分**，而丢 random 掉 19.1、丢 unimodal 掉 40.7 → **视觉信息高度集中在约 2% 的 token 上**。把 97% 的带宽花在低价值 token 上是纯浪费。

**为什么这样改能行**：Key-Norm 是从 KV cache 里**现成就有**的量，不用改 kernel、不用 attention 矩阵、FlashAttention/vLLM 下都能拿。**零训练**。这是整套改进里最「training-free 友好」的一条。

**怎么验证**：
1. 卡 4 的 latent-only 臂若 $\approx$ 真 KV，D3 的动机就已经被自己的数据支持了。
2. 画准确率–显存曲线，$p$ 扫四档；**必须对照随机丢同样比例**（这是 ViF 做过的正确对照）。
3. **判决**：$p=5\%$ 时准确率损失 $<2\text{pp}$ 而显存降 $>80\%$ → 成立。
4. **坑**：ViF 的 `selection.py:2-8` 里那个 omega 是 **no-op**（z-score + sigmoid 都是单调变换，top-k 不变；omega=0.01 时 sigmoid 饱和还会让 top-k 退化成按索引连号取）。别照抄那段，只抄 `norm(K)` 那一行。

---

### D4 — Delta 而非绝对状态

**改什么**：两个层次，从便宜到贵：
- **D4-a（零成本）**：传「$m$ 步 latent 的 KV」而不是「整段 prompt+latent 的 KV」。repo 逻辑已写好（`methods/latent_mas.py:40-44` 的 `latent_only` + `_truncate_past`），**只差两行 argparse**。
- **D4-b**：传残差 KV——同一 agent 分别跑「带图」和「不带图（纯白图或 dummy）」两次前向，把两者 KV 之差传下去，接收端加到自己的基线 KV 上。这是 VisionWormhole 残差注入思想（`vision_latent_mas_codec_new.py:1620`，`inputs_embeds[b,pos,:] = base_img + g*dlt`）的 **training-free、KV 层**版本。

**解决哪个失败机制**：H5 的安慰剂问题。若 shuffle KV ≈ 真 KV，说明收益来自「有个前缀」而非「前缀里的内容」；**残差形式天然剔除了「共同部分」**，只传差异，能把「信息」和「占位」在机制上分开。

**为什么这样改能行**：残差相对固定基线，保证注入落在流形附近（VisionWormhole 的原始动机），且 delta 的模长本身就是「这个 agent 到底看到了多少新东西」的天然度量——可以拿来做自适应触发（省掉 L²-VMAS 那套需要训练的熵门控）。

**怎么验证**：卡 4 直接加两臂。**判决**：D4-b 的 shuffle-gap（真 $-$ shuffle）显著大于 D4-a 的 shuffle-gap → 残差形式确实提高了信息密度。
**风险**：D4-b 要跑两次前向，成本翻倍，会打击 H6。必须同时报效率。

---

### D5 — 免训练的跨模型对齐：锚点法

**改什么**（仅在 H7 需要时启用，优先级最低）：
- **D5-a 相对表示（Relative Representations）**：准备 $M$ 条共享锚点（同一批图 + 同一批文本，$M\approx300$，`heterogeneous-latent-mas/data/vision_codec_anchor_text/mixed_cose_ocr_prm800k.jsonl` 有现成 3000 条，VisionWormhole 的 Table 2 证明 **90 条就够**）。两个模型各自对锚点前向取 hidden，把待传的 $h$ 表示成「对 $M$ 个锚点的余弦相似度向量」——这个 $M$ 维表示是模型无关的，接收端用自己的锚点集反解回自己的空间。
- **D5-b KVCOMM 式偏移锚点**：直接对 KV 做——用锚点上的 KV 偏移量估计当前 KV 的修正量。你 memory 里有 KVCOMM 的一手经验（`nips2025-KVCOMM`：$W=\sum(a\times\Delta v)$，**该重算的是中段内容 + 中后层，sink/浅层/前缀安全**）。这条经验可以直接指导「哪些层/位置的 KV 需要修正、哪些可以裸传」。

**解决哪个失败机制**：H7。同时它绕开了 VisionWormhole 最深的设计缺陷——**hub-and-spoke 对齐隐含「槽位对应」假设却没有任何机制强制它**（`merge_vision_codec_checkpoints.py:345,350` 用 `reshape(-1,D)` 摊平后按行配对，但 `q_sem` 是各模型独立 `randn*0.02` 初始化独立训练的，`vision_latent_mas_codec_new.py:557`）。**锚点法天生解决槽位对应**：锚点是共享的、有序的、语义确定的。

**怎么验证**：先在同族异尺寸（3B→7B）上做，扫锚点数 $M\in\{30,90,300,1000\}$，用**外部裁判 PPL** 当健康度指标（抄 VisionWormhole 的诊断手段，那个手段本身是好东西）。**判决**：$M=90$ 时 $X-L>5\text{pp}$ 且外部裁判 PPL $<20$ → 成立。

---

### 改进方向的依赖关系

```
卡1 ──► D2 是否值得做
卡3 ──► D1（若 X-L 边缘，D1 是救命稻草）
卡4 ──► D3（若 latent-only≈真KV）、D4（若 shuffle≈真KV）
卡6 ──► D1 升级为必需品
卡7 ──► D5
```

**推荐顺序：D1 → D3 → D4-a → D2 → D4-b → D5**。D1 和 D3 是纯工程、零训练、且直接对着 H6 的效率论点；D2 是理论上最漂亮但依赖卡 1 的结果；D5 最贵最后做。

---

## E. 止损线

### 止损线一：视觉信息传不过去（卡 3）

**触发条件**：卡 3 中 $X - L < 2\text{pp}$，**且**上了 D1（共享视觉前缀 KV）之后复测仍 $< 3\text{pp}$。

**判决**：training-free 的多模态 latent 通信不成立。这与 MACF 报的「朴素移植 LatentMAS 收益≈0」一致（LatentMAS* 55.7/48.5/33.2/42.3 vs 单模型 55.9/50.7/33.2/41.5，一胜两负一平）。

**换成什么**（两条都是能成文的）：
- **(a) 转诊断/证伪论文**：MACF 那个负面结果**配置疑似错配**（角色串行 KV 拼接硬套分段并行；很可能跑在 `--latent_space_realign` 默认关即 $W_a=I$ 的配置下）。你手上有卡 1–4 的完整机制分解（$W_a$ 退化量、KV 等价性、mRoPE 失败点、shuffle 对照、六种 KV 来源分解），**能把「为什么不成立」讲到机制层**，这比 MACF 一句「≈无收益」扎实得多。标题方向：《Why Training-Free Latent Communication Fails in Vision-Language Multi-Agent Systems》。
- **(b) 转纯效率系统论文**：D1（共享视觉前缀 KV）+ D3（Key-Norm 选择性 KV 传输）作为**多模态 MAS 推理加速系统**，不谈准确率增益，只谈「同准确率下省多少显存/延迟」。你有 LMCache/blend/APC 的现成基础设施（`/home/yilin/LMCache`，blend-APC 兼容已完结），这条路的工程门槛对你最低。

### 止损线二：安慰剂（卡 4）

**触发条件**：真 KV $-$ shuffle KV $< 1.5\text{pp}$。

**判决**：立刻停，不要再往下做任何主表。这说明观测到的收益是「前缀存在」的副作用，不是通信。**这个发现本身价值极高**——四篇论文没一篇做过这个对照，你可以直接反过来质疑它们的结论（尤其是没有任何 latent 基线正面对比的 L²-VMAS）。转向止损线一的 (a)。

### 止损线三：多智能体本身在这个规模上无意义（卡 5）

**触发条件**：Latent-MAS 在 $\ge3$ 个 benchmark 上低于 single-agent。

**判决**：问题不在 latent，在 MAS。L²-VMAS 的 Table 2 已经显示 **Qwen3-VL-32B-Thinking 的文本 VMAS（75.1）跑输单智能体（75.6）**，且相对单智能体的增益随规模坍缩（2B +7.1 → 4B +5.8 → 8B +5.2 → 32B +2.5）。

**换成什么**：换任务，而不是换方法。选 MAS 真正有增益的任务——多图/长视频推理、需要分工的组合任务（MACF 的分段视频路线）——而不是 POPE/MMBench 这类单图饱和 benchmark。

### 止损线四：mRoPE 系结构性不兼容（卡 2）

**触发条件**：Qwen2.5-VL 上三个 position_ids 变体全部不等价。

**判决**：不是止损，是**收缩范围**。把主实验台锁死 LLaVA/1D-RoPE 系，并把「mRoPE 与跨 agent KV 前置结构性冲突」单独写成一节——这是四篇都没覆盖的技术发现。

### 止损线五：时间

**4 周内卡 1–4 没有全部跑完并给出判决，就不要开卡 5。** 卡 1–2 是 4 天的事，卡 3 的工程是 2 人天，卡 4 几乎零工程。若 4 周还卡在 pipeline 上，说明工程路径选错了（大概率是选了 Qwen2.5-VL 而非 LLaVA 当起点，或者试图从 fork 起步）。

### 止损线六：新颖性

**触发条件**：卡 3/4 显示 training-free 不可行（必须训才能work）。

**判决**：**放弃这个方向**。因为 training-free 是你相对四篇论文唯一确定的空地：L²-VMAS 要三阶段 PPO（8×H200，230k steps）、MACF 要双侧 LoRA + 三阶段课程、ViF 要两阶段训练还解冻 LLM、VisionWormhole 要每模型训 41M codec。**一旦你也要训，你就在跟 ICML 2026 已接收的 L²-VMAS 正面撞车，且在实验规模上（5 backbone × 8 benchmark × 4 尺寸 × 6 拓扑）没有胜算。**

---

## 附：立项前必须做的三件小事

1. **精读 L²-VMAS 全文**（ICML 2026 已接收，arXiv 2602.00471）。它的 Appendix A.1 对 LatentMAS 的批评是**一句话、零实验、零分析**（"not directly transferable to VMAS ... perceptual-cognitive information conflation"）。**你的卡 3+卡 4 就是证真/证伪这句话的实验**，这本身可以当一篇论文的实验核心。注意它的 "key-value pairs" **不是** KV cache（$K\in\mathbb{R}^{d}$ 是单个检索键，$V\in\mathbb{R}^{l\times d}$ 是单层 hidden，注入方式是 8 个伪 token 的 embedding），写 related work 说它「也传 KV cache」是错的。
2. **别引 MACF §4.2 的「vs LatentMAS 领先 9.3/14.1/10.5/10.3」**——与它自己的 Table 1 全部对不上，真值是 +4.7/+8.3/+7.0/+6.9。引 Table 1 原始格子。
3. **别引 `https://github.com/YU-deep/L2-VMAS`**——已验证 HTTP 404，是幻觉链接。L²-VMAS 无公开代码。

---

## 附：一句话路线图

**先花 1 小时跑卡 1 弄清 $W_a$ 是不是恒等 → 花 3 小时跑卡 2 确认 KV 前置在 VLM 上数值等价（顺手抓 mRoPE 坑）→ 花 1 周搭 LLaVA-1.5 的两 agent 盲探针跑卡 3，看 $X-L$ 有没有超过 5pp → 立刻加卡 4 的 shuffle 对照排除安慰剂 → 过了才开主表，没过就转诊断论文或纯效率系统论文。**


---

## 总判决

建议按「卡1（1小时测W_a是否恒等）→ 卡2（3小时测KV前置数值等价+mRoPE坑）→ 卡3（盲agent视觉可达性探针，X−L>5pp 为生死线）→ 卡4（shuffle KV安慰剂对照）」的顺序推进，主实验台锁 LLaVA-1.5-7B 以绕开 mRoPE 混淆，骨架用 /home/yilin/LatentMAS 原版只抄 heterogeneous-latent-mas 的两个零件；核心判断是 W_a 断点其实是次要风险（repo 里默认就是单位阵），真正决定生死的是 KV 通道能否传动视觉信息，而 training-free 是相对四篇已发表工作唯一剩下的空地，一旦证明必须训就应立刻放弃转诊断/效率论文。
