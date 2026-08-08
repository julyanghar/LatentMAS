> **我的角色声明**：以下全部按魔鬼代言人写，即**只列反对理由，不做平衡**。我会明确标注每条的证据级别（**实测** / **静态读码** / **已发表结论** / **我的推断**），以及推翻它需要跑什么。最后的总判决是我在把这些反对都摆出来之后的诚实结论，不受"必须唱反调"的约束。
>
> **本轮新增的一手证据**（不是从那 4 篇里抄的，是我刚在本机跑出来的）：
> - `/home/yilin/tmp/devil-advocate-mm-latentmas/mrope_probe.py`
> - `/home/yilin/tmp/devil-advocate-mm-latentmas/mrope_latentstep.py`
>
> 环境：transformers 4.57.1，Qwen3-VL 架构（随机初始化的极小 config；**position_ids 的计算与权重无关，纯索引逻辑，因此结论一字不差地迁移到真实 7B/8B**）。

---

# 反对意见 0（先说结论级的那一击）

**你想做的事——"把 LatentMAS 当 baseline 搬到多模态 MAS 上跑，看效果"——已经被别人做过两次，两次都是负结果，而且都已经写进论文了。**

| 谁做的 | 怎么做的 | 结果 | 出处 |
|---|---|---|---|
| MACF（arXiv 2605.00444，ICML 2026 投稿格式） | `LatentMAS*` 作为 Table 1 的正式基线，长视频分段多 agent | 55.7 / 48.5 / 33.2 / 42.3 **vs** 单模型 Qwen3-VL-8B 55.9 / 50.7 / 33.2 / 41.5 —— **一胜两负一平，净收益 ≈ 0** | 已发表结论 |
| VisionWormhole（arXiv 2602.15382）Appendix F | LatentMAS 式跨模型潜在传输，12 个 latent-step 设定系统扫描 + 外部裁判 PPL | Qwen3-VL-2B + Gemma-3-4B 在 GSM8K **全程 ≤0.5% 准确率**；Qwen+LFM2.5 在 256 步阶跃崩溃到 PPL 8.1e5 | 已发表结论 |
| L²-VMAS（arXiv 2602.00471，**ICML 2026 已接收**）Appendix A.1 | 一句话断言 LatentMAS "not directly transferable to VMAS" | 无实验，纯断言 | 已发表断言 |

**失败机制（对你的项目而言）**：这不是技术失败，是**贡献失效**。你跑完之后能写的最好的一句话是"我们确认了 MACF Table 1 里那一行"。审稿人第一个问题就是 "How does this differ from MACF's LatentMAS\* baseline and VisionWormhole's Appendix F?"，而你没有答案，因为你做的确实就是那件事。

**最小证伪实验**：不需要跑实验——去读 MACF Table 1 那一行和 VisionWormhole Appendix F 的扫描表，写一段 200 字说明"我做的和他们做的**在哪个具体维度上不同**"。写不出这段话，项目当场停。**耗时 2 小时。** 如果你的答案是"他们配置可能错了，我要做对的复现"——那这是**反对意见 1** 的内容，见下，那条我反而认为是唯一的活路。

---

# 反对意见 1 ★最硬的一条：M-RoPE 下，你连"跑对"都做不到，而且不报错

## 1.1 失败机制（精确到张量）

LatentMAS 的两处核心调用（[/home/yilin/LatentMAS/models.py:303](file:///home/yilin/LatentMAS/models.py) 和 `:339`）长这样：

```python
outputs = self.model(input_ids=..., attention_mask=..., past_key_values=past_key_values, use_cache=True, ...)   # agent prefill
outputs = self.model(inputs_embeds=latent_embed, attention_mask=latent_mask, past_key_values=past, use_cache=True, ...)  # latent 自回归 m 步
```

**两处都不传 `cache_position`，也不传 `position_ids`。** 对纯文本 Qwen3 这没问题：`Qwen3Model.forward` 里 `if cache_position is None: cache_position = arange(past_seen, past_seen+L)`，再 `position_ids = cache_position.unsqueeze(0)`，位置自动接续。

**但 Qwen-VL 家族多了一层外壳，这层外壳在把活交给 text model 之前就把 `position_ids` 算好了**（[transformers/models/qwen3_vl/modeling_qwen3_vl.py:1196-1221](file:///home/yilin/anaconda3/envs/gpt-deep/lib/python3.11/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py)）：

```python
prefill_noncompiled_stage = (cache_position is not None and cache_position[0] == 0) \
                            or (past_key_values is None or past_key_values.get_seq_length() == 0)
if prefill_noncompiled_stage or self.rope_deltas is None:
    position_ids, rope_deltas = self.get_rope_index(...)   # 真正的 3D M-RoPE
    self.rope_deltas = rope_deltas
else:
    delta = (cache_position[0] + self.rope_deltas) if cache_position is not None else 0   # ← 关键
    position_ids = arange(seq_length).view(1,-1).expand(batch,-1).add(delta)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)                            # ← 三轴塌成同一行
```

第二个 agent 带着非空 `past_key_values` 进来 → `prefill_noncompiled_stage = False`，`self.rope_deltas` 又是上一个 agent 留下的非 None → **走 else 分支**。而 `cache_position is None` → **`delta = 0`**。

于是 `Qwen3VLTextModel` 里那句本来能救场的 `if position_ids is None: position_ids = cache_position.view(1,1,-1).expand(3,...)`（`:823-824`）**永远不会触发**，因为外壳已经塞了一个错的 `position_ids` 进来。

## 1.2 实测结果（**实测**，本机 transformers 4.57.1）

`mrope_probe.py` 输出：

```
AGENT1 position_ids (t/h/w 三轴):
 [0, 1, 2, 3, 4, 4, 4, 4, 6, 7, 8, 9]     ← t 轴
 [0, 1, 2, 3, 4, 4, 5, 5, 6, 7, 8, 9]     ← h 轴（图像 2×2 网格）
 [0, 1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 9]     ← w 轴
past len after agent1: 12

AGENT2 position_ids（LatentMAS 的调用签名，KV 接着传）:
 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

**两个独立的灾难同时发生：**

1. **位置从 0 重启**。Agent 2 的每一个 token 的 RoPE 位置和 cache 里 Agent 1 的某个 token **完全重合**。对 RoPE 来说这意味着"agent 2 的第 3 个 token 和 agent 1 的第 3 个 token 是同一个时刻"，因果链彻底乱掉。
2. **三个 M-RoPE 分段塌成同一个 1D arange**。Agent 2 的图像 token 的 (h, w) 二维空间结构**被完全抹掉**——对比 agent 1 那三行明显不同的 (4,4,4,4)/(4,4,5,5)/(4,5,4,5)。也就是说，**从第二个 agent 起，模型看图是"一串没有空间关系的序列"**。

`mrope_latentstep.py` 输出更狠——latent 自回归那 m 步：

```
latent step 0: pos=[[0], [0], [0]]
latent step 1: pos=[[0], [0], [0]]
latent step 2: pos=[[0], [0], [0]]
latent step 3: pos=[[0], [0], [0]]
```

**所有 m 步"潜在思维"全部落在位置 0**，和 BOS 同一个位置，彼此位置上完全不可区分。LatentMAS 论文的整个 §3.1（"隐空间自回归 m 步"）在 Qwen-VL 上执行出来的东西，**位置编码层面是 m 个叠在原点上的向量**。

## 1.3 补一句更难受的：连"修一半"都不够

我加测了"规范地传 `cache_position`"的版本（很多人会这么修）：位置变成单调递增了，但**三行仍然完全相同** —— else 分支的 `expand(3,-1,-1)` 是无条件的。要真修，你必须**自己对每个 agent 调 `get_rope_index`，再决定 h/w 两个轴怎么和前一个 agent 的 h/w 轴拼**（前一张图占了 h∈[4,5]，这张图从哪开始？重叠还是偏移？两张不同的图共享一个 h/w 坐标系在语义上是什么意思？）。**这个问题没有标准答案，L²-VMAS / ViF / MACF / VisionWormhole 四篇里没有一篇碰过它——因为它们全都走 inputs_embeds 通道，压根不做 KV 拼接。**

## 1.4 证据级别

- position_ids 的具体数值：**实测**（权重无关的索引逻辑，可 1 分钟复跑）
- Qwen2.5-VL 同一 bug 模式：**静态读码**（`modeling_qwen2_5_vl.py:1285-1299` 与 Qwen3-VL 逐行同构）
- LLaVA / InternVL 系（1D RoPE，无 `rope_deltas`）**不受此影响**：**静态读码**（grep 零命中）
- "这会导致准确率崩到什么程度"：**我的推断**
- **我的强推断**：MACF 报的 `LatentMAS*` ≈ 0 收益、VisionWormhole App.F 的 ≤0.5%，**很可能相当一部分就是这个 bug**，而不是"latent 通信思想不行"。

## 1.5 最小证伪实验（★这条是你唯一的机会，优先做★）

```
模型：Qwen2.5-VL-7B-Instruct（或 Qwen3-VL-8B）；数据：MMStar 500 题；4 agent sequential，m=10
三臂：
 A. 原样 LatentMAS 端口（position_ids 走 else 分支，即上面那个坏路径）
 B. 修 offset：每个 agent 手动传 cache_position（位置单调，但三轴仍相同）
 C. 全修：每个 agent 手动 get_rope_index，t 轴接续、h/w 轴各自独立复位
对照：单 agent baseline、TextMAS
看的数字：四臂准确率差；以及每个 agent 前向后 h_t 的 ‖·‖ 和与 agent1 h_t 的 cos
```
**耗时估计**：单卡 RTX 6000 Ada，7B bf16，500 题 × 4 臂 ≈ **6–10 小时**。
**判决标准**：如果 A ≈ C，说明位置编码根本不是瓶颈，反对意见 1 被推翻，但同时**你也证明了 MACF 的负结果是真的**（更糟）。如果 C ≫ A 且 C > 单 agent，**你就拿到了一个真结果：已发表的两个负结果是实现 artifact**——这是能发论文的东西。

---

# 反对意见 2：$W_a$ 的问题不是"无定义"，是"定义得刚好把视觉信息压掉"

## 2.1 先纠正一个提法

你的设想里写"$W_a$ 在 VLM 上无定义，因为视觉输入不过 $W_{in}$"。**严格说这是不对的，而不对的方式让问题更糟。**

在 LatentMAS 里，$W_a$ **从来不作用在视觉 token 上**。它只作用在 latent 步的 $h_t$——即"最后一个位置的最后一层隐状态"（[models.py:314](file:///home/yilin/LatentMAS/models.py)）。VLM 的 `get_input_embeddings()` 照样返回一张 $|V|\times d$ 的文本词嵌入表，`lm_head` 照样在，所以 $W_a=(W_{out}^\top W_{out}+\lambda I)^{-1}W_{out}^\top W_{in}$ **算得出来，代码一行不用改，跑起来一点都不报错**。

真正的问题是：**这个 $h_t$ 已经注意过图像了，它的视觉相关分量正好落在 $W_a$ 要压掉的方向上。**

## 2.2 失败机制

$W_a$ 本质是"对 $W_{out}$ 做正则化最小二乘伪逆，再右乘 $W_{in}$"。它的值域是 $\mathrm{span}(W_{in}$ 的行$)$ 附近——**即"能被某个文本 token 的嵌入表达的东西"**。$h_t$ 里凡是无法被任何文本 token 表达的成分（"这块纹理""这两个物体的相对位置""这个色块的具体色调"——ViF 和 MACF 两篇都把这类叫做"文本表达不了的细粒度视觉线索"），**在这一步被投影掉**。

再叠加 [models.py:212](file:///home/yilin/LatentMAS/models.py) 那句无条件执行的模长硬拉：

```python
aligned = aligned * (target_norm / aligned_norm)   # ‖e‖ ≡ avg‖W_in,x‖，恒定
```

**幅度信息也被扔掉，只剩方向。** 所以 LatentMAS 在 VLM 上的"潜在思维"，本质是：把一个含视觉信息的隐状态，**投影到文本词表张成的子空间上，再把模长归一化成文本嵌入的平均模长**。这是一个**结构性的"把视觉压成文本"的算子**——而你搬 LatentMAS 到多模态的全部动机，恰恰是"不要把视觉压成文本"。

**这是全篇最讽刺的一点：LatentMAS 的 latent 通道，在 VLM 上就是一个隐式的、有损的、无法学习的 image-captioning。**

## 2.3 补救之后还算不算 training-free？

不算。逐条数：

| 补救方案 | 需要训吗 | 谁已经这么干了 |
|---|---|---|
| 把 $W_{in}$ 换成"文本词嵌入 ∪ 视觉投影器输出的经验分布"，重解最小二乘 | 不用梯度，但**需要一批图跑一遍视觉塔采样**，且解出来的 $W_a$ **依赖数据集** → 不再是"闭式、data-free" | 无人做过（这是空地，但很小） |
| 学一个投影/Transformer 块 | 要训 | ViF 的 $f$（2519 万参数，Stage2 连 LLM 都解冻）；MACF 的 2 层 MLP adapter；VisionWormhole 的 41M codec |
| 复用 VLM 原生冻结 projector 做对齐 | **不用训 projector，但外挂的压缩器要三阶段 PPU** | **L²-VMAS 附录 C.2 Eq.13-14 已经写出来了** |

第三行是最痛的：**"绕开 $W_a$ 视觉断点"的最优雅解法（复用 VLM 自带的冻结 projector）已经被 L²-VMAS 写出来并接收了。** 你留下的空间是"连压缩器也不训"，而那正是 §2.2 说的会被投影掉视觉信息的方案。

- 证据级别：$W_a$ 的形式与模长归一化 = **静态读码**（[code-to-paper-mapping.md §3.2/§3.3](../../code-to-paper-mapping.md)）；"视觉分量被压掉"= **我的推断**。

## 2.4 最小证伪实验

```
Qwen2.5-VL-7B。取 200 张差异极大的图 × 同一个问题（"Describe what you see."）
对每张图取 prompt 末位的 h_t，算 e = h_t·W_a·(target_norm/‖·‖)
指标 1：pairwise cosine 的分布 —— 比较 {h_t} 与 {e} 的均值/方差
指标 2：把 e 过 lm_head 取 top-5 token（logit lens），看 200 张图的 top token 集合大小
判决：若 cos(e_i,e_j) 的均值从 h 的 ~0.4 涨到 ~0.95，或 200 张图的 top token 收敛到十几个通用词
     → W_a 确实抹平了视觉差异，这条反对成立
```
**耗时：< 1 小时（200 次单图前向 + 一个矩阵乘）。这是全清单里性价比最高的实验，先做这个。**

---

# 反对意见 3：视觉 hidden state 的分布问题在多模态上只会更严重

## 3.1 失败机制

你提到的 SDE 那篇的结论（"直接注入原始 hidden state 会伤害接收方，传增量才有用"）在这里有**两个独立的旁证**，而且都指向同一方向：

**旁证 A（代码级，最有力）**：VisionWormhole 的注入不是直接写，是**相对固定 dummy 图嵌入做残差 + 门控**：
`methods/vision_latent_mas_codec_new.py:1620` → `inputs_embeds[b,pos,:] = base_img + g * dlt`，而门控 bias 初始化在 `:615` 是 **`-4.0`，即 σ(−4) ≈ 0.018**。翻译一下：**作者在训练开始时把注入强度调到 1.8%，然后让模型慢慢学着敢用它。** 这是"直接注入会炸"的最直白的工程自白。而且他们还在 `train_..._codec_new.py:1715-1777` 挂了全套数值守卫（非有限 loss 跳步、梯度 NaN 置零、连续 5 步坏梯度熔断）+ clip（logits ±80 / latent ±50 / **injection ±20**）——**论文里一个字没写**。

**旁证 B（论文级）**：VisionWormhole 自己在 §1 就承认 "sidestepping the off-manifold problem that breaks text-only LLMs under arbitrary continuous inputs"。它们的整个方法论基础是"纯文本 LLM 对任意连续输入是脆的，所以我们改走 VLM 的视觉口"。

## 3.2 多模态放大机制

VLM 的视觉 token 隐状态在**范数**上就与文本 token 不同一个量级（massive activations / attention sink 现象在 VLM 上被反复观察到）。而 LatentMAS 的处理是 `aligned * (target_norm/aligned_norm)`——**把模长硬拉到文本嵌入的平均模长**。对一个视觉主导的 $h_t$，这不是"归一化"，是**把它按文本的尺度重新解释**。

- 证据级别：VisionWormhole 的 gate bias 与 clip 参数 = **代码级已核**（第三方精读一手结论）；"VLM 视觉 token 范数异于文本" = **广为报道的技术事实，但你必须自测**；"因此 LatentMAS 的模长归一在 VLM 上有害" = **我的推断**。

## 3.3 最小证伪实验

```
Qwen2.5-VL-7B，同一 prompt 模板，两组输入：(a) 纯文本问题 (b) 图+同一问题
测：末位 h_t 的 L2 范数分布、以及每层视觉 span vs 文本 span 的 per-token ‖h‖ 直方图
再测：target_norm = mean‖W_in‖ 这个标量，和 (b) 的 ‖h_t‖ 差几倍
判决：若 ‖h_t^{img}‖ / target_norm 与 ‖h_t^{text}‖ / target_norm 差 >2×，模长硬拉就是在做模态相关的畸变
```
**耗时：< 30 分钟。可与反对意见 2 的实验合并跑。**

---

# 反对意见 4：显存不是"会不会爆"的问题，是"效率叙事直接反转"

## 4.1 先把账算清楚（**我的核算**，假设已标注）

Qwen2.5-VL-7B：28 层 / 4 个 KV head / head_dim 128 / bf16
→ 每 token KV = 2 × 28 × 4 × 128 × 2 B = **57,344 B ≈ 56 KiB**

单图 4-agent 场景（图 ≈ 1300 视觉 token，每 agent prompt ≈ 250 文本 token，m=10，文本 MAS 每 agent 输出 ≈ 300 token）：

| 方案 | 峰值 KV 长度 | 峰值 KV 显存 / 样本 | batch=15 |
|---|---|---|---|
| **LatentMAS（KV 累积，从不清空）** | Σ(1300+250+10) × 4 = **6,240 tok** | **341 MiB** | **5.0 GiB** |
| TextMAS（每 agent 独立前向，峰值 = 最后一个） | 1300+250+3×300 = **2,450 tok** | 134 MiB | 2.0 GiB |
| 单 agent | 1,550 tok | 85 MiB | 1.2 GiB |

## 4.2 失败机制：**在多模态下，latent MAS 比 text MAS 更吃 KV，不是更省**

原因很朴素但很致命：**LatentMAS 让每个 agent 重新编码整个 prompt（含整张图），并把这些 KV 永久留在链上**（[code-to-paper-mapping.md §2](../../code-to-paper-mapping.md) 第 1 条：题面被重复编码 4 次，`context` 参数恒为 `""`）。在纯文本里，重复的是 200 个 token 的题面，KV 累积仍小于 text MAS 累积的长篇输出。**在多模态里，重复的是 1300 个视觉 token，一下就反超了。**

于是：

- LatentMAS 在文本上的卖点是"省 token 又省显存" → **多模态上只剩"省 decode 步数"，显存直接输给 baseline**。
- 你论文里那张"效率对比表"会出现"我方 KV 峰值 2.5×  baseline"这一行，而 L²-VMAS 恰恰**只报 token 数、不报墙钟/显存**（它的坑 #10）——你一旦老实报显存，就把它没暴露的东西暴露在自己身上。

## 4.3 更要命的：scaling 实验做不了

L²-VMAS 的整个 framing 是 "scaling wall"，$N_A$ 扫到 10；ViF 的 circular 拓扑扫到 20 turn。

- $N_A=10$：15,600 tok → 0.85 GiB/样本 → 48 GiB 卡去掉 15 GiB 权重后 batch ≤ ~25，还要算 attention 的 $O(L^2)$
- $N_A=20$：31,200 tok → 1.7 GiB/样本 → **batch ≤ 12**
- 长视频（MACF M=6，每 agent 4096 视觉 token）：26,136 tok → **1.4 GiB/样本，batch=1**

**你没法在 scaling 维度上和 L²-VMAS 正面比，而 scaling 正是这块地的主战场。**

- 证据级别：56 KiB/token = **我的核算**（Qwen2.5-VL-7B 公开 config，需你自己 `AutoConfig` 核一遍）；Qwen3-VL-8B 的层数/KV head 我**未核**，跑之前先 `python -c "from transformers import AutoConfig;print(AutoConfig.from_pretrained('Qwen/Qwen3-VL-8B-Instruct').text_config)"`。

## 4.4 最小证伪实验

**耗时 20 分钟**：拿 `torch.cuda.max_memory_allocated()` 在 LatentMAS / TextMAS / 单 agent 三臂上各跑 20 个样本，$N_A \in \{2,4,6,8,10\}$，画 KV 峰值 vs $N_A$。若 LatentMAS 曲线始终在 TextMAS 之下，这条被推翻。

---

# 反对意见 5：给不给下游 agent 看图 —— 这是个两难，两边都已经被证过是坏的

这是个纯逻辑上的死角，我认为你无法回避。

**分支 A：每个 agent 都重新喂图（LatentMAS 原样端口的行为）**
→ 下游 agent 的 KV 里有 $N$ 份**同一张图的近乎重复的视觉 KV**，位于不同 RoPE 偏移。注意力被摊薄到 $N$ 份重复内容上。而 ViF 的 token 维消融已经证明：**视觉证据高度集中在约 1.22% 的 token 上**（Turn 1；Turn 20 只剩 0.10%），丢掉 inactive token 只掉 2.3 分，丢掉 unimodal token 掉 40.7 分。**你把 97%+ 的 KV 带宽/显存花在低价值 token 的重复副本上。**（已发表结论）

**分支 B：只有第一个 agent 看图，后面靠 KV 继承**
→ 图在位置 0–1300，第 4 个 agent 的 query 在位置 6000+。ViF 的轮次维实证：视觉 token 平均注意力 Turn 1 = 0.165 → Turn 10 = 0.099 → Turn 20 = 0.063（**−62%**），中层视觉注意力峰随轮次**变平**。**这正是 ViF 命名为"幻觉雪球"的那条曲线**，而 ViF 的整篇论文就是为了修它。（已发表结论）

**你在 B 分支上得到的最好结果，就是 ViF 在没有 ViF 的情况下测出的那条衰减曲线。** 也就是说：**你的 baseline 的失败模式，别人已经命名过、量化过、并且发表了修它的方法。**

- 证据级别：ViF 的两组数字 = **已发表结论**（注：ViF 的表格数字是 HTML 二次转录，引用前必须回 PDF 核）；"分支 A 会注意力稀释" = **我的推断**。
- **最小证伪实验**：4-agent，两分支各跑，记录**每个 agent 的中层视觉 span 注意力占比**（或 FlashAttention 下用 ViF 的 Key-Norm 代理，`vif/utils/selection.py:9-11` 的 `torch.norm(K,dim=-1)`，不用改 kernel）。看占比是否随 agent 单调下降。**耗时 3–4 小时。**

---

# 反对意见 6：Qwen3-VL 的 deepstack —— inputs_embeds 通道在结构上够不着视觉

**这条是新的，那 4 篇里没有一篇提过，但它直接打掉一大类补救方案。**

`modeling_qwen3_vl.py:862-867`：

```python
for layer_idx, decoder_layer in enumerate(self.layers):
    hidden_states = decoder_layer(...)
    if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
        hidden_states = self._deepstack_process(hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx])
```

**Qwen3-VL 把视觉特征注入到前若干层的残差流里，不是只在第 0 层。** 后果：

1. 任何"把消息当伪 token 塞进 `inputs_embeds`"的方案（LatentMAS 的 latent 步、L²-VMAS 的 8 个伪 token、ViF 的 relay token、MACF 的通信 token、VisionWormhole 的残差注入——**五篇全部走这条路**）在结构上只能触达第 0 层。**它们复现不出真实视觉 token 的多层条件化。**
2. 反过来，这是 LatentMAS 的 **KV 直传路线唯一的理论优势**：KV 是逐层的，agent 1 的第 $l$ 层 KV 里**已经含 deepstack 在第 $l$ 层注入的效果**。也就是说 KV 传递是**唯一能把 deepstack 的多层视觉条件化原样递给下游**的机制。

**为什么这仍然是"反对意见"而不是"机会"**：因为要吃到这个优势，你必须先跨过反对意见 1 的 M-RoPE 坑；而且这条优势**只对 deepstack 架构成立**（Qwen3-VL 系），LLaVA / InternVL 系没有 deepstack，这条优势就不存在——**即你最好的卖点是架构特定的，泛化性论证会被打**。

- 证据级别：deepstack 的多层注入 = **静态读码，已核**；"KV 路线能保留它" = **我的推断，需实验**。
- **最小证伪实验**：Qwen3-VL-8B，agent 1 前向后取 KV；agent 2 分别用 (a) 继承 KV (b) 只继承第 0 层等价的 inputs_embeds 伪 token。测两者在需要细粒度空间定位的题上（RealWorldQA / MMStar 的 perception 子集）的差。**耗时 4 小时。**

---

# 反对意见 7：地基不稳 —— LatentMAS 的实现坑在 VLM 上被系统性放大

逐条对照 [/home/yilin/LatentMAS/exp-docs/code-to-paper-mapping.md](../../code-to-paper-mapping.md)，看每个坑在多模态下变成什么：

| 原坑（文本） | 多模态下的放大 | 级别 |
|---|---|---|
| **§4.6 padding_side=right，`hidden_states[-1][:,-1,:]` 取到 pad 位** | 文本 prompt 长度批内差异 ~2×；**Qwen-VL 动态分辨率下视觉 token 数差异 4–16×**（同一 batch 里一张 336² 图 144 token、一张 1024² 图 1300 token）→ 短样本的"最后一列"深深埋在 pad 里。**批内几乎每一行的 $h_t$ 都取错位置。** | **静态读码 + 推断** |
| **§4.6 `latent_mask = ones(B, past_len+1)` 把 past 的 pad 位重新对注意力放开** | 在 VLM 里这些 pad 位夹在**视觉 span 中间**，等于让模型对着一段乱码做视觉推理 | **静态读码** |
| **§4.2 tied embedding ⇒ $W_a \approx I$** | L²-VMAS 主打 Qwen3-VL-**2B/4B**。Qwen 系小模型按惯例 tied → **你在小模型上的 $W_a$ 消融毫无意义**，只剩模长归一 | **待核**（1 分钟可查：`AutoConfig.from_pretrained(...).tie_word_embeddings`） |
| **§4.1 `--latent_space_realign` 默认关，且"关"≠论文的"无对齐"** | 你极可能在完全没开 $W_a$ 的情况下跑了整套多模态实验，然后得出"$W_a$ 在多模态上无所谓"的结论 | **静态读码** |
| **§4.8 `--latent_steps` 默认 0，静默退化成单 agent** | repo 自带 example log 就踩过（`Latent Steps: 0`）。你多模态跑一周，最后发现测的是单 agent | **实测（原文档）** |
| **§4.3 hierarchical 实为串行** | 想做多模态并行拓扑，必须解决"三段独立 KV 沿序列维拼接的位置编码怎么排"——**在 M-RoPE 下这个问题从难变成没有定义**（h/w 两个轴怎么拼？） | **静态读码 + 推断** |
| **§4.4 vLLM 路径根本不传 KV，换了机制** | 想用 vLLM 加速多模态实验 → 你测的不是 KV 传递；且 `B=1` 直接崩（`squeeze(0)` + 三元解包） | **静态读码** |
| **§4.7 argparse choices 只有 Qwen + `prompts.py:7` 硬 assert "qwen"** | 想跨 backbone 泛化（reviewer 必问）→ 先重写 prompt 系统 | **静态读码** |
| **新增：`self.rope_deltas` 是模型级可变属性** | multi-agent 循环里被上一个 agent 污染；batch>1 时 `delta.repeat_interleave(batch_size // delta.shape[0])` 有形状假设 | **实测**（我的 probe 里 agent2 用的就是 agent1 留下的 `[[-2]]`） |

**总结这条**：你要搬的不是一个"稳定的 training-free 方法"，是**一份只在 Qwen 文本、batch=1、显式开对齐、m>0 的窄条件下才是论文所描述那个东西的评测脚本**。往上叠多模态，等于在这些坑之上再叠 M-RoPE、动态分辨率、deepstack 三层新变量。**当结果不好时，你分不清是"思想不行"还是"第 7 个坑"。** 这是最消耗人的失败方式：不是失败，是**无法归因**。

**最小证伪实验**：不跑实验，先做 [§5 的 7 条检查清单](../../code-to-paper-mapping.md)，再加 3 条多模态专属：(8) `padding_side` 设 left 了吗？(9) 每个 agent 的 position_ids 打印出来看过吗？(10) `tie_word_embeddings` 是什么？**耗时 1 小时，能省掉后面一周的错误归因。**

---

# 反对意见 8：新颖性 —— L²-VMAS 占了大半，剩下的一半被 V2X 占了六年

## 8.1 L²-VMAS 那一侧（**已发表**）

它是 **ICML 2026 已接收**，占掉的地包括：

- 「把 latent 通信搬到 VLM MAS」这个 framing 本身
- VMAS scaling wall 的实证刻画（第 3 轮见顶、第 6 轮跌破单 agent、第 10 轮低 2.6%、token 30 倍）
- full-content 文本传输是四种传输里**最差**的（第 10 轮净损 3.8%）
- 感知/思考记忆解耦，且有子任务级硬证据
- **复用 VLM 原生冻结 projector 绕开视觉断点**（附录 C.2 Eq.13-14）——**这就是你最想要的那个解法**
- 实验矩阵门槛：5 backbone × 8 benchmark × 4 尺寸 × 6 拓扑

它明确留白的、你能占的只有：training-free 版本、**真正的逐层 KV 直传**（它全篇没碰 KV）、$W_a$ 在视觉上的理论讨论、多步 latent 视觉思考、异构 backbone、**与任意 latent 基线的正面对比**、真实墙钟/显存效率。

**注意这个空地的形状**：它们全部是"**证伪/补齐/做对比**"型，不是"**提出新方法**"型。你能写的是一篇**实证/分析论文**，不是一篇 method 论文。这不一定不好，但你要提前接受。

## 8.2 V2X / 协同感知那一侧（**这条我认为你严重低估了**）

"多个 agent 之间传中间神经特征而不是文本/决策"这件事，在协同感知领域是**教科书级的既有范式**，术语叫 **intermediate fusion**（对 early fusion / late fusion）：

- When2com / Who2com（CVPR 2020 / ICRA 2020）—— **学习"什么时候、跟谁"通信**，即"按需触发"
- V2VNet（ECCV 2020）、DiscoNet（NeurIPS 2021）、V2X-ViT（ECCV 2022）、CoBEVT
- **Where2comm（NeurIPS 2022）—— 空间置信图做带宽感知的特征稀疏化**，即"只传高价值区域"，和 ViF 的"只传 ~2% unimodal token"是同一个 idea
- **HEAL（ICLR 2024）—— 异构协同感知的 backbone 对齐**，即"不用逐对翻译器就能把不同模型的中间特征对齐"，和 VisionWormhole 的 hub-and-spoke $O(N)$ 是同一个 idea
- 更早：**DIAL（Foerster et al., NeurIPS 2016）—— agent 之间的可微连续消息信道**。**"latent communication between agents" 这个概念，2016 年就有了。**

**失败机制（评审侧）**：一个 CVPR/ICCV 背景的审稿人读到"我们提出 agent 之间传连续潜在表示而非文本，这是新方向"，第一反应是 *"intermediate fusion, 2020"*；读到"带宽 $M\times K$ 可控"，第一反应是 *"Where2comm 的 bandwidth-accuracy Pareto 曲线在哪"*；读到"跨模型对齐 $O(N)$"，第一反应是 *"HEAL"*。**而这三篇都不会出现在你的 related work 里，因为你的 related work 是 LLM MAS 那一支。**

这不是"我猜他会这么想"，这是这个领域的标准评审反射。LatentMAS 之所以能中 ICML spotlight，是因为它的新颖性来自 **training-free + 逐层 KV + LLM agent 三者的交集**——**你一旦把 training-free 拿掉（见反对意见 2），或者把 KV 换成 embedding 注入（见反对意见 6 的所有替代方案），你就掉回 2020 年的先验里。**

- 证据级别：上述工作的存在与年份 = **公开技术事实**（我按记忆列出，**投稿前必须逐篇核实并读原文**，别照抄我这段）；"审稿人会这么反应" = **我的推断，但我给它很高置信度**。
- **最小证伪实验**：花 4 小时读 Where2comm + HEAL 两篇，然后写一段"我的方法 vs intermediate fusion 的本质区别"。如果你能写出一个**不是"我们用的是 LLM"** 的区别，这条被推翻。我的预测是你写不出来。

---

# 反对意见 9：评测设计上，你要测的信号比混淆项还小

## 9.1 失败机制

L²-VMAS 自己的数据（**已发表**）：

- Qwen3-VL-**32B**-Thinking 上，文本 VMAS **75.1 < 单 agent 75.6** —— **多智能体本身在大模型上就是负收益**
- 相对单 agent 的增益随规模坍缩：2B +7.1 → 4B +5.8 → 8B +5.2 → 32B +2.5
- MMBench 上出现 3 处负增益
- **全部数字单次运行，无 seed 无方差**

再叠上样本量：RealWorldQA 765 题、MMVet 218 题。765 题上一个臂的标准误约 ±1.8pp，两臂差的标准误约 ±2.5pp。**而你要测的效应量本身就是 2–5pp。**

**所以你的实验会长这样**：LatentMAS 端口比单 agent 低 3 分。你无法区分以下五种解释：
(1) latent KV 传递在多模态上不 work；(2) M-RoPE bug（反对意见 1）；(3) padding 取错 $h_t$（反对意见 7）；(4) **这个 benchmark 上多智能体本身就是负收益**（L²-VMAS 已证）；(5) 单次运行噪声。

**这不是"结果不好"，是"结果不可解释"。** 而 (4) 这一项是最恶心的：**你的 baseline 在没有任何 bug 的情况下也应该输给单 agent**，因为 text MAS 也输。

## 9.2 最小证伪实验

```
先不跑 latent，只跑：单 agent vs TextMAS，3 个 seed，N_A ∈ {1,2,4,6}
在你打算用的每一个 benchmark 上做
判决：找出至少 2 个 benchmark，满足 "TextMAS(N_A=4) − 单agent > 2×标准误"
      —— 这些才是你有资格测 latent 的地方
```
**耗时：7B 模型，4 个 benchmark × 4 个 $N_A$ × 3 seed ≈ 2 天。**
**这一步必须在跑任何 latent 实验之前做。** 找不到这样的 benchmark，整个项目当场停 —— **因为没有可测的信号。**

---

# 反对意见 10：就算全部跑对，你能写出的最好的论文是什么？

诚实地推演一下最好情况：

1. 你修好了 M-RoPE（反对意见 1 的 C 臂），修好了 padding，开了 $W_a$，选了 tie=False 的 backbone。
2. 结果比单 agent 高 2–4 点，比 TextMAS 高 1–2 点，token 省 30%，但 **KV 显存 2.5×**（反对意见 4）。
3. 你写：《我们证明 training-free 的 KV 级 latent 通信在 VLM MAS 上是可行的，此前报告的负结果源于位置编码实现缺陷》。

这篇论文的命运：

- **优点**：它推翻了两篇已发表工作的负结果（MACF Table 1 的 `LatentMAS*`、VisionWormhole App.F），并且直接证伪了 L²-VMAS Appendix A.1 那句零证据断言。**这是真贡献。**
- **风险 1**：审稿人说"这是 bug fix + reproduction，不是新方法"。
- **风险 2**：L²-VMAS 已经用**训练**的方法拿到了 +2.7–5.4%（相对），你 training-free 拿到差不多的数 → 审稿人问"那为什么不直接用 L²-VMAS"。你的答案只能是"我们不用训"，而这在 8×H200 训 230k step 这件事上确实是个卖点，**但它是效率卖点，而你的显存输了 2.5×**。
- **风险 3**：CVPR 线审稿人拿 intermediate fusion 打你（反对意见 8）。

**结论**：这条路的天花板是一篇**扎实的实证/勘误型论文**，投 EMNLP Findings / NeurIPS D&B track / ACL 的分析 track 有戏，投 CVPR/ICML 主会作为 method paper 很危险。

---

# 反对意见 11（补充，短）：几条零散但真实的钉子

1. **闭源不可用**：KV 直传要求完整白盒 `past_key_values`。任何 API 模型直接出局。L²-VMAS 的熵触发也一样（需完整 logits）。你的方法的适用面 = "你能拿到权重的开源 VLM"。
2. **异构完全不成立**：LatentMAS 的 KV 直传要求收发双方**层数/head 数/head_dim/tokenizer 全兼容**。VisionWormhole §2 已经点名批评了这一点。而多模态 MAS 的实际卖点之一恰恰是"不同 VLM 各有所长"。**你的方法只能同构自我对话**，而 L²-VMAS 的 "model-agnostic" 也只是"换 backbone 重训一遍"，不是跨 backbone —— **这块地空着是因为它很难，不是因为没人想到。**
3. **prompt 层的约定没有任何机制保证**：`prompts.py:23` 那句 "The plan information is provided in latent KV representation format" 是**用自然语言告诉模型"你的上文是隐式的"**。模型从未在这种输入上训练过。文本上它勉强 work，是因为 KV 是"类文本状态"；换成视觉主导的 KV，这个未经训练的行为的可靠性是纯赌博。（**静态读码 + 推断**）
4. **MACF 的负结果配置可能错配，但你无法证明**：我同意"MACF 的 `LatentMAS*` 疑似跑在 `--latent_space_realign` 默认关的配置上，且把角色串行 KV 拼接硬套到分段并行"——但 **MACF 承诺的 Appendix 不存在**（12 页 PDF 里 "appendix" 只有 2 次前向引用），你没有任何办法核实它的实现。所以你"推翻它"的说服力有天然上限：**你只能说"我们的实现得到了不同结论"，不能说"它们错在哪"。**

---

# 总判决

## **(B) 值得做，但必须换主张 —— 而且换掉之后，它就不是你现在设想的那个项目了**

**你现在的主张**（"把 LatentMAS 当 baseline 搬到多模态 MAS 上跑，看效果"）我判**不值得做**，理由是反对意见 0：这件事已经被 MACF 和 VisionWormhole 各做过一次，都是负结果，都已发表。再做一次是复现。

**但反对意见 1 意外地打开了唯一一条活路**，而这条路我认为是真的：

> **主张改成：已发表的两个"多模态 latent MAS 无效"的负结果，是 M-RoPE 实现缺陷造成的 artifact，而非机制本身的失效。**

支撑这个主张的东西我今天已经拿到一半了：

- **实测**：Qwen3-VL 架构下，第二个 agent 起 position_ids 从 0 重启且三个 M-RoPE 轴塌成同一 arange；latent 自回归的 m 步**全部位于位置 0**。这是 transformers 4.57.1 主线代码在 LatentMAS 的调用签名下的确定性行为，**不报错、不警告、静默产出垃圾**。
- **静态读码**：Qwen2.5-VL 同一 bug 模式；LLaVA/InternVL 系（1D RoPE）不受影响 —— **这个"受影响 / 不受影响"的对照本身就是一个漂亮的实验设计**。
- **已发表**：L²-VMAS 全篇没碰 KV（它是 embedding 层伪 token），ViF/MACF/VisionWormhole 也全是 embedding 通道 —— **五篇里没有一篇真正实现过多模态的逐层 KV 传递**，所以这个 bug 没人查过。
- **静态读码**：Qwen3-VL 的 deepstack 多层视觉注入，给了 KV 路线一个 embedding 路线**结构上做不到**的理论优势。

换成这个主张之后，项目变成："**多模态 latent MAS 的负结果复核 + KV 通道的正确实现 + 与 embedding 通道的机制对照**"。它是分析型论文，不是 method 论文，天花板在 Findings / D&B / 分析 track。你要提前接受这一点。

## 推翻我这个判决所需要的证据

**要把判决从 (B) 推到 (A) 不值得做**，只需要下面**任意一条**：

1. 反对意见 1 的**最小实验**跑出 **A ≈ C**（修不修位置编码没差别）→ 那么已发表的负结果是真的，机制本身不 work，这个方向死。**这是最快的判决实验，6–10 小时。**
2. 反对意见 9 的**前置实验**找不到任何一个 benchmark 满足 "TextMAS 显著优于单 agent" → 没有可测信号，方向死。**2 天。**
3. 反对意见 2 的实验显示 $W_a$ 把 200 张不同图的 $e$ 压到 cos > 0.95 → latent 通道在 VLM 上本质是有损 captioning，training-free 版本没有理论出路。**< 1 小时。**

**要把判决从 (B) 推到 (C) 主张成立**，需要下面**全部**：

1. C 臂（正确 M-RoPE）显著优于 A 臂（naive 端口），差距 > 3× 标准误，3 个 seed；
2. C 臂 **同时**优于单 agent **和** TextMAS，在至少 2 个已验证有 MAS 信号的 benchmark 上；
3. 你能给出一个不依赖"我们用的是 LLM"的、相对 intermediate fusion（Where2comm / HEAL）的本质区别陈述；
4. 你能老实报出 KV 显存对照表，并且有一个说得过去的效率论证（比如"我们省的是 decode 步数和墙钟，不是显存"，并给出实测墙钟）。

## 我最想让你先做的三件事（按顺序，总计约 1 天）

1. **1 小时**：跑 `/home/yilin/tmp/devil-advocate-mm-latentmas/mrope_probe.py` 和 `mrope_latentstep.py`，亲眼看到那三行相同的 arange 和那四个 `[[0],[0],[0]]`。同时 `AutoConfig` 查一遍你要用的 backbone 的 `tie_word_embeddings` / 层数 / KV head 数。
2. **1 小时**：反对意见 2 的 $W_a$ 塌缩实验（200 张图，cos 分布 + logit lens）。这是全清单里最便宜、判决力最强的一个。
3. **2 小时**：读 MACF Table 1 的 `LatentMAS*` 那一行 + VisionWormhole Appendix F 的扫描表，写 200 字回答"我做的和他们做的在哪个具体维度上不同"。

这三件做完，你就知道该走 (A) 还是 (B) 了 —— **而且花的是一天，不是一个月。**


---

## 总判决

判 (B) 值得做但必须换主张：原设想「把 LatentMAS 当 baseline 搬到多模态 MAS 跑」已被 MACF 与 VisionWormhole 各做过一次且都是已发表负结果，再做即复现；唯一活路是我本机实测到的新证据——Qwen-VL 系 M-RoPE 下 LatentMAS 的调用签名会让第二个 agent 起 position_ids 从 0 重启、三个 M-RoPE 轴塌成同一 arange、latent 自回归 m 步全部落在位置 0（静默无报错），因此主张应改为「已发表的多模态 latent MAS 负结果是位置编码实现 artifact 而非机制失效」，做成分析/勘误型论文而非 method 论文。
