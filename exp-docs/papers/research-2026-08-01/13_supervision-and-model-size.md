# 训练式 latent 通信模块：监督信号从哪来 + 小模型该多小

> 标注约定：`[原文]` = 论文明确写的（附 arXiv 号）；`[计算]` = 我从公开架构维度算的；`[推断]` = 我的判断，未经论文背书；`[未查到]` = 检索失败，不猜。

---

## 0. 先给结论

你的新设想（训一个模块做 agent 间 latent 通信）在监督信号上**不是没有出路，而是出路已经被人踩出来了三条半**。真正的坏消息不是「没有平行数据」——是这块地已经很挤，且验收标准比你想的严。

| 监督路径 | 谁做过 | 判决 | 建议角色 |
|---|---|---|---|
| ① 文本通道当教师 | Vision Wormhole、MACF Stage1 | 有现成公式，但**天花板 = 文本通道** | Stage-1 冷启动，**不能当主 loss** |
| ② 自监督重建 | KV-CAR、Interlat 的 ℒ_geom | 保信息不保有用 | **只当正则项**（小权重） |
| ③ 任务奖励 RL | L²-VMAS、LatentMem、Mem-W Stage2 | **无人纯 RL 冷启动过** | 最后 5-10% 打磨 |
| ④ 上界蒸馏 | **Mem-W Stage1**、Interlat、DiscoNet | **性价比最高，已在 LLM agent 上验证** | **主监督** |
| ⑤ 合成平行数据 | LatentQA / Activation Oracles | 造诊断集 > 造训练集 | 审计与消融 |

---

## 一、五条路逐条拆解

### 路 ①：以文本通道当教师

**唯一给出完整公式的是 Vision Wormhole。** `[原文, arXiv:2602.15382]`

$$\mathcal{L}_{\text{codec}} = \lambda_h\|h_{\text{vis}} - \text{stopgrad}(h_{\text{text}})\|_2^2 + \lambda_{kl}\,\tau^2\,\mathrm{KL}\!\left(\text{softmax}(\ell_{\text{text}}/\tau)\,\|\,\text{softmax}(\ell_{\text{vis}}/\tau)\right) + \lambda_{rms}\left(\mathrm{RMS}(\Delta_{\text{inj}}) - \mathrm{RMS}(\bar{X}_{\text{img}})\right)^2$$

三项各司其职：

- **第一项（表示匹配）**：让走视觉通道后接收方的 hidden 贴住走文本通道时的 hidden。`stopgrad` 在教师侧——教师不动。
- **第二项（行为匹配）**：温度 τ 下的 logits KL。这才是「拟合同样的下游行为」的落点。
- **第三项（RMS 正则）**：把注入扰动的幅度压到跟视觉 embedding 的原生 norm 同量级。**这一项很容易被忽略但是致命的**——注入信号如果幅度失配，要么被 LayerNorm 洗掉，要么盖过真实视觉输入。

注入方式是**残差写进 image token embedding**：$X^{(i)}_{\text{img}} = \bar{X}^{(i)}_{\text{img}} + g_i \cdot \text{Resample}(\Delta_i; L^{(i)}_{\text{img}})$，其中 $\bar{X}^{(i)}_{\text{img}}$ 是 **dummy image** 的 baseline `[原文]`。接收方实测用了 Qwen3-VL-2B-Thinking / Gemma-3-4B-IT / SmolVLM2-2.2B / LFM2.5-VL-1.6B `[原文]`。

**数据构造是零成本的**：跑一遍 text-MAS 就同时拿到了 $h_{\text{text}}$ 和 $\ell_{\text{text}}$，不需要任何人工标注。Vision Wormhole 甚至走到极端——弱监督变体的原话是 **"We train codecs using fewer than 100 anchor texts"**（Sec 4.3）`[原文]`。

**但必须看它的真实数字。** v2 Table 1：GSM8K 文本 MAS 80.8%（27.3s）vs Vision Wormhole 76.2%（26.7s），**−4.6pp，1.02× 加速** `[原文]`；HumanEval-Plus 最好能到 +26.2pp `[原文]`。弱监督变体 Table 2 更分裂：GSM8K 在 SmolVLM2/Qwen 组合上 +12.7pp，在 Gemma/Qwen 上 −3.2pp `[原文]`。

`[推断]` 这不是实现问题，是结构性上限：**你在教 latent 通道模仿文本通道的行为，最好情况是打平**。而 latent 通道的全部卖点本该是「传文本传不了的东西」。用蒸馏做主 loss，等于亲手把自己的卖点砍掉。

**还有人这么做吗？有，而且给出了最有价值的消融。** MACF `[原文, arXiv:2605.00444]` 的三阶段课程：

| 阶段 | 损失 | 监督来源 | 数据 |
|---|---|---|---|
| ① 语义对齐 | $\mathrm{CE}(C, A_0(c))$ | ground-truth caption | LLaVA-Video-178K（0-30s 子集） |
| ② 证据摘要 | $\mathrm{CE}(y, A_0([c; \mathrm{Emb}(q)]))$ | image-QA 对 | Video-R1 的 image 部分 |
| ③ 跨 agent 协作 | $\mathrm{CE}(y, A_0([c^{(1)};\dots;c^{(M)}; \mathrm{Emb}(q)]))$ | video-QA ground truth | Video-R1 video + Molmo2 子集 |

消融（MLVU-Test）`[原文]`：

| 配置 | 准确率 | 相对全量 |
|---|---|---|
| 仅 Stage 1 | 36.3% | −13.9 |
| Stage 1+2 | 46.7% | −3.0 |
| Stage 1+3 | 36.5% | −12.5 |
| 全部三阶段 | **49.2%** | — |

**这张表是本次调研里信息量最大的一张。** 读法：

- Stage 2（证据摘要，+10.4pp）是真正的杠杆；
- **Stage 1+3 = 36.5%，几乎等于 Stage 1 单干的 36.3%** —— 跳过「学会摘要证据」直接上「跨 agent 协作」的端任务 loss，**增益接近零**；
- 而 Stage 1（caption CE，本质就是「文本当教师」）虽然自己分数最低，去掉它后面全塌。

`[推断]` 合起来的意思：**「文本当教师」是必需的地基但不能是屋顶；直接拿跨 agent 端任务 loss 训通信模块（也就是你最可能第一反应去做的事）会失败。**

---

### 路 ②：自监督重建

**存在证明有，但都是配角。**

- **KV-CAR** `[原文, arXiv:2512.06727]`：轻量 autoencoder 沿 embedding 维压缩 K/V，存进 cache 前压、取出时还原。纯重建监督。
- **Interlat 的 ℒ_geom** `[原文, arXiv:2511.09149]`：对逐步平均后的 latent 方向施加余弦惩罚，保住几何朝向。
- **Q-KVComm** `[原文, arXiv:2512.17914]`：按 sensitivity profiling 做自适应逐层量化分配 bit-width，5-6× 压缩比。

**为什么只能当正则？** emergent communication 领域早有定论：用监督重建 loss 把通信 ground 在观测空间**确实能提升样本效率**，但**输入侧目标很容易被满足，模型仍然需要多得多的样本才能探索出任务特化的通信空间** `[原文，来自 emergent-comm 文献综述，最接近的出处是 arXiv:2302.14276 一线的工作]`。

`[推断]` 直白说：重建 loss 逼你把有限带宽花在「还原得像」上，而不是「对下游有用」上。你的 K=32 个 token 会被塞满冗余细节。**正确用法是小权重防坍缩**——防止 gate 学成常数、防止 message 塌到一个点。

---

### 路 ③：任务奖励 RL

三个走这条路的工作，全部细节：

**L²-VMAS** `[原文, arXiv:2602.00471]`
- Stage I：随机激活 latent memory，优化 memory 构建与更新机制
- Stage II：冻结 memory synthesis 组件，单独训 memory orchestration 组件提升召回
- Stage III：全部外部 memory 组件解锁，端到端联合训练
- 数据：GQA（明确无测试基准暴露）；硬件：8× H200 141G
- base VLM **全冻结**，只训外部 memory 模块（压缩模块 + 可学门控）
- 结果：平均准确率 +2.7~5.4%，token 用量 −21.3~44.8%
- **`[未查到]` 论文没有披露：损失函数形式、reward 定义、算法名（PPO/GRPO/其他）、每阶段样本量、wall-clock、模块参数量、样本效率讨论。**

**LatentMem** `[原文, arXiv:2602.03036]`
- LMPO = Latent Memory Policy Optimization，「把任务级优化信号通过 latent memory 传播回 composer」
- 两阶段：训练集上收集原始轨迹入 experience bank → 用 LMPO 训 memory composer
- **memory composer 的底座是完整的 Qwen3-4B-Instruct-2507**（HF: `Kana-s/LatentMem-Qwen3-4B`）`[原文/模型卡]` —— 这里的「小模型」是 4B，一点也不小
- 最高 +19.36%
- `[未查到]` LMPO 的 advantage estimator、是否 GRPO 式、GPU 数量（PDF 文本抽取失败）

**Mem-W** `[原文, arXiv:2605.09317]`
- Stage 2 才是 RL：**RLOO**（Leave-One-Out），二值终局奖励（成功/失败），policy gradient 按 trajectory-level advantage 加权 + KL 正则回 stage-1 参考策略
- 数据：web 11,176 条成功轨迹（CoMEM，13 domains，memory bank 22,346）；mobile 2,489 条（GUI-Odyssey，6 类，memory bank 4,972）
- 硬件：8× A800 80GB，bf16，DeepSpeed ZeRO-3，per-GPU batch 2

**判决** `[推断]`：**三篇全都不是纯 RL 冷启动。** Mem-W 明确先自蒸馏再 RLOO；L²-VMAS 第一阶段是无奖励的构建优化；LatentMem 先离线收轨迹。这不是巧合——连续消息 + 稀疏二值奖励的信用分配在 MARL 里就是老大难：REINFORCE 类估计器噪声大、需要任务特定调参、样本效率低 `[原文，emergent-comm 综述共识]`；让梯度穿过通信通道虽然能提性能，但**实际上把多个 agent 建模成了单一实体** `[原文，同上]`——这对「异构 agent 通信」的设定是自相矛盾的。

**RL 值得放在最后，不值得当主监督。**

---

### 路 ④：上界蒸馏 —— 我推荐的主监督

**DiscoNet 的原始配方** `[原文, arXiv:2111.00643, NeurIPS'21]`：teacher 用 **early collaboration + holistic-view 输入**，student 用 **intermediate collaboration + single-view 输入**，训练时约束 student 的 post-collaboration feature map 匹配 teacher 的对应位置。另有 matrix-valued edge weight，每个元素反映特定空间区域上的 inter-agent attention。

**关键发现：这个配方已经被搬到 LLM agent 上了，而且没人把它当卖点讲。**

**Mem-W Stage 1** `[原文, arXiv:2605.09317]` 的原话结构是：
- teacher = **同一个冻结的 GUI agent**，但喂它 **extended raw context window（L' >> L 步）**
- student = 只看 latent-augmented 输入
- loss = action token 序列上的标准 next-token 交叉熵 **+** teacher 分布到压缩表示的 KL 散度

`[推断]` 这就是 DiscoNet 一比一的 LLM 版：**上界 = 看得见全部原始上下文的自己**。

**Interlat** 是同一形状的另一变体 `[原文, arXiv:2511.09149]`，三项等权（λ=1）复合 loss：
- $\mathcal{L}_{\text{task}}$：冻结 actor 预测上的交叉熵
- $\mathcal{L}_{\text{pref}}$：全长 latent 路径 vs 压缩路径穿过同一 actor 的 KL，**按相对基线的熵减做不确定性加权**
- $\mathcal{L}_{\text{geom}}$：逐步平均 latent 方向的余弦惩罚

监督目标明确写着来自「从固定 instruction-tuned 模型抽取的 full-length latent communications」`[原文]`——即上界模型的输出就是标签。

**为什么这条路最好** `[推断]`：

1. **上界是可构造的**，不需要标注：把所有 agent 的原始输入（图 + 文 + 问题）拼给单个 agent，它就是上界。
2. **监督是稠密的**：token 级 CE + 分布级 KL，每个位置都有梯度，与 RL 的稀疏终局奖励差了几个数量级。
3. **天花板高于路 ①**：上界模型（看得见全部）严格强于文本 MAS（信息已经被文字量化损失过一遍）。**这是路 ④ 相对路 ① 唯一但决定性的优势。**
4. **完全 label-free**。
5. **有先例但未被作为方法卖点** —— Mem-W 把它写成一个不起眼的 stage，你可以正面做。

**VLM MAS 上的具体构造建议** `[推断]`：
- **teacher**：单个 VLM 同时吃 sender 的图像 + receiver 的图像 + 问题 + 完整文本上下文
- **student**：receiver 只吃自己的图像 + sender 传来的 K 个 latent token
- **loss**：$\mathrm{CE}(y, \text{student}) + \beta\,\mathrm{KL}(\text{teacher} \| \text{student})$，可选加 Interlat 式的熵减加权（教师越确定的位置权重越大）
- **关键设计点**：teacher 和 student 是**同一个模型**，只是输入不同 —— 这样分布天然对齐，不需要跨模型 logits 对齐这道额外的坎

---

### 路 ⑤：合成平行数据

**LatentQA 给了唯一真正可用的配方** `[原文, arXiv:2412.08686, ICLR'26]`：

> Latent Interpretation Tuning (Lit)：微调一个 decoder LLM，把 stimulus 产生的 activation **patch 进去**，在 QA 对上最小化 loss——「类似视觉指令微调在图像关联的 QA 对上训练」。

数据构造的精髓：**在 stimuli 前面 prepend 「controls」（规定期望输出属性的片段）→ 用这些 prompt 喂 target LLM 拿 activation → 围绕这些 controls 自动生成 QA 对** `[原文]`。

`[推断]` 为什么这招管用：**你人为知道这个 hidden state 里该有什么，因为是你亲手放进去的。** 这就是凭空造出平行数据的方法——不是去猜「A 的 hidden 对应 B 该收到什么」，而是**反过来控制 A 的输入使得答案已知**。

**Activation Oracles** `[原文, arXiv:2512.15674]` 是它的泛化版，结论对你直接有用：
- **数据多样性是关键变量**，不是数据量
- AO 能恢复「微调进模型、但在输入文本里根本没出现」的信息（传记知识、恶意倾向），**尽管从来没在微调过的模型的 activation 上训练过**
- 数据足够多样时，可以匹配或超过领域特化的白盒可解释性 pipeline

**反例：纯无监督对齐这条路在规模上是死的。** "It's a (Blind) Match!" `[原文, arXiv:2503.24129]` 用 QAP + factorized Hahn-Grant solver（内存 O(N⁴)→O(N³)，时间仍 O(N⁵)）做无平行数据的视觉-语言匹配：
- N=10 时 DINOv2 在 CIFAR-10 上 80%、CINIC-10 上 100%
- **N>10 就开始掉；N>40 时 Gurobi 超 1.5 小时时限；N>100 全局最优不可行**
- 无监督分类在 CIFAR-10 上只有 ~51%，「明显劣于监督模型」

`[推断]` 含义：**别指望 Relative Representations 式的免训练锚点对齐能扛住 VLM 规模。** 它在几十个概念的量级上是漂亮的，在你需要的规模上不成立。这反过来正是「要训练」这个新设想的最强论据。

**路 ⑤ 的正确定位** `[推断]`：用来**造诊断集 / 审计集**比当主训练信号更值。因为你可以精确控制「sender 知道 X，receiver 不知道 X」，从而直接测量下面要讲的 CAG 指标。

---

## 二、参数量对比表

### 已发表工作的通信模块

| 模块 | 论文 | 参数量 | 来源可信度 |
|---|---|---|---|
| **C2C Fuser** | arXiv:2510.03215 | **478M**（可训练参数，Table 6） | `[原文]` 转述，与下行独立吻合 |
| C2C Fuser（1.1B 模型对） | 由 arXiv:2605.22863 报告 | **477.8M / 956 MB** | `[原文]` |
| C2C 对照：直接微调 receiver | arXiv:2510.03215 Table 6 | 596M | `[原文]` |
| C2C 对照：同模型自融合 | 同上 | 529M | `[原文]` |
| **LCF-128** | arXiv:2605.22863 | **19.4M / 39 MB**（24.6× 小） | `[原文]` |
| LCF-256 | 同上 | 53.4M / 107 MB | `[原文]` |
| **LCF-128 剪枝至 9 层** | 同上 | **6.24M / 13 MB**（76.7× 小） | `[原文]` |
| Vision Wormhole codec | arXiv:2602.15382 | **`[未查到]`** 论文只说 "lightweight"；首次 v1 HTML 抓取曾报「∼0.05B」但两次复核未复现该句 | ⚠️ 低置信，**不要引用** |
| MACF 通信 adapter | arXiv:2605.00444 | 2-layer MLP，K∈{16,32,48}（主结果 K=32）；**参数量未给** | `[原文]` 结构 / `[未查到]` 数字 |
| Mem-W 压缩器 | arXiv:2605.09317 | Q-Former 式，K=8 learned queries + 投影头 + outcome embedding；**参数量未给** | `[原文]` 结构 / `[未查到]` 数字 |
| L²-VMAS memory 模块 | arXiv:2602.00471 | **`[未查到]`** | — |
| **Interlat 压缩器 $M_\phi$** | arXiv:2511.09149 | **和 base 同架构的完整 Qwen2.5-7B / LLaMA3.1-8B**，AdamW 全参更新 | `[原文]` |
| **LatentMem memory composer** | arXiv:2602.03036 | **完整 Qwen3-4B-Instruct-2507** | `[原文]`+模型卡 |
| MoT router + interaction layers | arXiv:2509.21164 | **`[未查到]`**（只说 "lightweight"，全文未抓到） | — |

### 视觉-语言连接器参照系

| 模块 | 参数量 | 来源 |
|---|---|---|
| **BLIP-2 Q-Former** | **188M**（32 queries × 768 dim） | `[原文, arXiv:2301.12597]` |
| **Flamingo Perceiver Resampler** | **约 200M**（三种模型规模下固定不变） | `[原文, arXiv:2204.14198 Table 5]`，经搜索结果转述，我未直接读表 → 中置信 |
| Flamingo GATED XATTN-DENSE | 1.4B (3B) / 1.8B (9B) / 10B (80B) | 同上 |
| LLaVA-1.0 linear projector (1024→4096) | **4.20M** | `[计算]` |
| **LLaVA-1.5 projector (7B)**：2-layer MLP, 1024→4096→4096 | **20.98M** | `[计算]` |
| LLaVA-1.5 projector (13B)：1024→5120→5120 | 31.47M | `[计算]` |
| Qwen2-VL 式 patch merger：2×2 concat (5120)→5120→3584 | 44.57M | `[计算]` |

### 从维度直接推参数量（我的计算，供你设计用）

一个标准 transformer block ≈ $12d^2$（attn $4d^2$ + FFN $8d^2$）：

| 工作维度 d | 单 block | 2 block | 4 block | 6 block |
|---|---|---|---|---|
| 768 | 7.09M | 14.18M | ~28M | 42.53M |
| **1024** | **12.60M** | 25.19M | **~50M** | 75.58M |
| 2048 | 50.36M | 100.72M | ~201M | 302.15M |
| **4096**（= receiver 原生维度） | **201.38M** | 402.76M | ~805M | 1208.28M |

learned query latents（$K \times d$）：K=32, d=4096 → **0.13M**；K=8, d=1024 → **0.008M**。

**`[推断]` 这张表的读法，也是整节的核心结论：**

> **决定参数量的是工作维度 $d$，不是 token 数 $K$。**

K 的贡献完全可以忽略（0.1M 量级）。**省参数的唯一手段是降到 bottleneck 维度做，而不是减少传几个 token。** 这精确解释了 C2C 为什么会 478M——LCF 的批评说得很准 `[原文, arXiv:2605.22863]`：C2C「把通信保持在 receiver 的高维 cache 空间」，K/V 各需要一条 **full-width fuser**，还逐层做。LCF 把瓶颈压到 d=128 后直接掉到 19.4M，性能保持。

---

## 三、「到底该多小」：我的判断

### 推荐区间：**10M – 60M**（bottleneck d=512~1024，4 层 cross-attention，K=8~32）`[推断]`

**下界依据**：LCF 剪枝到 6.24M 仍然能工作（在 1.1B 模型对上）`[原文]`。所以 6M 不是不可能，但那是文本对上的结果，VLM 场景我不建议低于 10M。

**上界依据（硬红线）**：`[推断]` **必须 ≤ receiver 参数量的 1%**。理由是 C2C Table 6 里最刺眼的一组对照：

- 直接微调 receiver（不用 sharer）：**596M**
- C2C fuser：**478M**

两者只差 20%。这意味着 C2C 的 fuser 成本已经逼近「干脆把这些参数拿去微调 receiver」。**审稿人一定会问这个问题**，你必须能回答「用同样参数预算 LoRA 微调 receiver」这个 baseline。对 8B VLM，1% = 80M —— 10~60M 正好落在红线内且留有余量。

**最强的正面先例** `[推断]`：LLaVA-1.5 的 projector 只有 21M `[计算]`，而它只需要 558K 图文对做预训练就够用。**「轻量接口 + 少量数据」在 VLM 上是被验证过的路线**，Q-Former 的 188M 之所以大，是因为它要从零学视觉-语言对齐，任务重得多——而你的模块是在两个都已经会看会说的模型之间搭桥，任务轻得多。

### 单卡 / 少卡可行性

各家实际用量 `[全部原文]`：

| 工作 | 硬件 | 时长 | 训了什么 |
|---|---|---|---|
| **LCF** | **单张 A100 80GB** | **4-5 小时 / run**，300 optimizer steps，~52s/step，effective batch 260 | 19.4M adapter，1.1B 模型对 |
| C2C | 单 A100（论文只写了推理） | 1,929 steps，batch 256，seq 2048 | 478M fuser |
| MACF | 4× A100 80GB | 未给 | 端到端微调 Qwen3-VL-8B（非仅 adapter） |
| Mem-W | 8× A800 80GB | 未给 | Q-Former 压缩器 + RLOO |
| L²-VMAS | 8× H200 141G | 未给 | 外部 memory 模块（base 冻结） |
| **Interlat** | **64× A100 80GB**（压缩训练），8× A100（actor） | 未给 | **整个 7B/8B 压缩器全参更新** ← 反面教材 |

**我的判断** `[推断]`：

- **单卡 24GB（4090）**：只能训 0.5B~2B 的收发对，模块 <30M，必须 gradient checkpointing + 离线缓存 sender。能跑通 proof-of-concept，不够发论文。
- **单卡 80GB（A100/H100）**：**这是甜点区**。2B~4B VLM 对 + 10~60M 模块。参照 LCF 的 4-5 小时（1.1B 文本对），VLM 场景乘 3~5 倍 ≈ 一天一个 run。可以跑消融。
- **4× 80GB**：对得上 MACF 的规模，但注意 MACF 是**端到端微调 8B VLM**，不是只训 adapter——你只训 adapter 的话 4 卡很宽裕。
- **8× 80GB**：L²-VMAS / Mem-W 的量级 = 加了 RL 阶段。
- **64 卡（Interlat）**：因为它训的是整个 7B 压缩器。**这是你要主动避开的设计选择。**

### 一个能砍掉一半显存的工程点 `[推断]`

**sender 侧完全冻结且不需要梯度** —— 所以 sender 的 hidden / KV 可以**离线预计算存盘**，训练时只跑 receiver 的前向 + 反传。这做三件事：

1. 显存需求砍掉一半以上（大模型只剩一个在图里）
2. 「合成平行数据」变成一次性预处理，可以复用于所有消融
3. 让 batch size 能提上去（LCF 的 effective batch 260 就是这么来的）

**VLM 的额外代价** `[推断]`：视觉 token 数量巨大（Qwen2.5-VL 动态分辨率下可达几千个），缓存 sender KV 的盘开销 = 层数 × 2 × 头数 × head_dim × token 数，比纯文本贵一个数量级。**建议只缓存被选中的注入层，或只缓存最后一层 hidden**（SDE 路线）。

---

## 四、三条会咬你的红线

### 红线 1：端任务 loss 会骗你 —— causal audit 的数字

`[原文, arXiv:2607.26773]` 的四种消息设置：**无消息 / 当前样本的消息 / 别的样本的消息（长度匹配）/ receiver 自产消息（算力匹配）**；五个指标：

- **PS**（Positive Signaling）：sender 信息在消息里的可编码性，用互信息下界估
- **PL**（Positive Listening）：消息**存在与否**是否改变 receiver 预测分布
- **CIC**（Causal Influence of Communication）：消息**身份**是否影响预测（当前样本 vs 别的样本）
- **CAG**（Content-Attributable Gain）：样本特异内容带来的任务价值增益
- **SSG**（Self-Substitution Gap）：独立 agent 相对 receiver 自产消息的额外价值

**实测数字**（审计对象是 LatentMAS λH）`[原文]`：

| 设置 | 总效应 | 「别的样本消息」效应 | CAG |
|---|---|---|---|
| GSM8K / Qwen3-4B (n=100) | **−1.00pp** | **−6.17pp** | **+5.17pp**（两分量区间都不含 0） |
| GSM8K / Qwen3-8B (n=60) | +1.67pp | +3.96pp | −2.29pp（**方向完全反转**） |
| MATH-500 / Qwen3-4B (n=60) | +15.00pp | +8.33pp | +6.67pp [0.42, 12.50] |
| MATH-500 / Qwen3-8B (n=40) | +10.00pp | +8.13pp | +1.88pp（区间跨 0） |
| ARC-C | 近零 | 分量方向相反 | — |

**读法** `[推断]`：GSM8K/Qwen3-4B 那一行是全场最毒的——**总效应是负的（−1.00pp），但里面藏着一个 +5.17pp 的真实内容增益和一个 −6.17pp 的消息存在性伤害**。只看总效应你会得出「没用，放弃」；只看 CAG 你会得出「很有用」。而且同一任务换个模型规模（4B→8B）**两个分量的方向全都反过来**。

对训练者的直接含义：**如果你只用端任务 loss 训，你无法区分自己学到的是「通信」还是「一个恰好有用的提示式前缀」**。必须在训练监控和评测里都加入「错配消息」对照组（用别的样本的 message 喂进去），把 CAG 单独报出来。这应该写进你的实验设计，不是事后补的分析。

### 红线 2：地已经很挤

| 已被占的坑 | 论文 |
|---|---|
| VLM 多 agent + latent 通信 + 课程训练 | MACF, arXiv:2605.00444 |
| 两阶段（自蒸馏 + RLOO）训 Q-Former 压缩器 | Mem-W, arXiv:2605.09317 |
| 视觉 MAS + 三阶段 RL + latent memory | L²-VMAS, arXiv:2602.00471 |
| 异构 VLM + 视觉端口注入 + 文本教师蒸馏 | Vision Wormhole, arXiv:2602.15382 |
| 学习式 KV fusion + 逐层门控 | C2C, arXiv:2510.03215 |
| 压缩瓶颈 + 层剪枝 + 跨上下文 agent | LCF, arXiv:2605.22863 |

**但有一条缝** `[推断]`：上面所有工作，sender 传的都是**推理 rollout 产生的 hidden**。Vision Wormhole 的 sender 名义上是 VLM，但论文**不讨论 sender 的视觉输入如何被处理**——它抽的是 sender 的推理状态，不是图像状态。

也就是说：你地形图里「$W_a$ 只对文本词表成立、VLM 视觉输入不过 $W_{\text{in}}$」这个理论断点，在训练路线下**变形而非消失**：训练能绕开 $W_a$（学一个投影就行），但**「sender 看到的视觉证据本身如何跨模型传给 receiver」仍然没有被任何一篇工作正面解决**。这可能是真正的空位——而且它天然需要训练（Blind Match 的结果已经证明免训练对齐扛不住规模）。

### 红线 3：别把「小模型」做成大模型

Interlat 用 **64× A100** 训一个和 base 同架构的 7B 压缩器 `[原文]`；LatentMem 的 composer 是完整 4B `[原文]`。这两个都不是「小模块」，是「第二个大模型」。`[推断]` 如果你的方案滑向这个方向，你的贡献就从「通信协议」变成了「多花一个模型」，审稿人会直接把你的 baseline 换成「同参数量的更大单模型」。

---

## 五、我建议的落地配方 `[全部推断，但每步都有论文先例]`

```
Stage 0  离线预计算
  sender 全冻结、不需梯度 → 把 sender 的 hidden/KV 存盘
  先例：C2C/LCF 都是 sender frozen
  收益：显存减半，batch 提上去，消融可复用

Stage A  语义对齐（路 ① + 路 ⑤）           ← 不能省
  loss: CE(caption, receiver(message))
  或 Vision Wormhole 三项式（含 RMS 正则，别漏）
  先例：MACF Stage1（去掉它后面全塌：1+3 = 36.5% ≈ 1 only 36.3%）

Stage B  上界蒸馏（路 ④）                  ← 主监督
  teacher = 单个 VLM 吃全部 agent 的图 + 文 + 问题
  student = receiver 只吃自己的图 + K 个 latent token
  loss = CE(y, student) + β·KL(teacher ‖ student)
  可选：Interlat 式熵减加权（教师越确定的位置权重越大）
  先例：Mem-W Stage1、Interlat ℒ_task+ℒ_pref、DiscoNet

Stage C  重建正则（路 ②）                  ← 小权重
  加一个方向保持项（Interlat ℒ_geom 式余弦惩罚）
  目的：防 message 坍缩、防 gate 学成常数
  权重要小，否则挤占带宽

Stage D  RL 打磨（路 ③）                   ← 可选，最后
  RLOO 二值终局奖励 + KL 正则回 Stage B 参考策略
  先例：Mem-W Stage2
  预期收益：最后 5-10%，不是主力

全程评测  causal audit 四设置
  必报「错配消息」对照 + CAG 单独拆出
  否则你不知道训出的是通信还是提示效应
```

**规模参数**：bottleneck d=512~1024，4 层，K=8~32，总参数 10~60M `[计算 + 推断]`。单张 80GB 卡起步。

---

## 六、检索覆盖与诚实的缺口

本次跑了 16 轮不同措辞的 WebSearch，抓取了 15+ 篇论文原文/HTML。

**确认查不到的（不要让我编）**：
- Vision Wormhole codec 的硬参数量（论文只说 "lightweight"；v1 HTML 首次抓取曾出现「∼0.05B」，两次定向复核均未复现该句，**判为不可引用**）
- Vision Wormhole 的训练硬件与时长（论文写在 Appendix B.2，正文抓取不到）
- L²-VMAS 的 loss 形式、reward 定义、算法名、每阶段样本量、wall-clock、模块参数量 —— **这是该论文自身的复现性缺口，不是我的检索问题**
- MoT (arXiv:2509.21164) 的 joint objective 具体形式、参数量、数据、硬件（abstract 无，全文未抓到）
- LatentMem 的 LMPO 目标函数与 advantage estimator（PDF 文本抽取失败）
- MACF 的 adapter 参数量、训练时长、各阶段样本数

**本次新发现、原地形图里没有的论文**：

| 论文 | arXiv | 为什么对你重要 |
|---|---|---|
| **Mem-W: Latent Memory-Native GUI Agents** | 2605.09317 | **最完整的两阶段监督模板**（长上下文自蒸馏 + RLOO），DiscoNet 配方的 LLM 版 |
| **Latent Cache Flow** | 2605.22863 | 参数量下界的实证（19.4M → 剪枝 6.24M），单 A100 4-5 小时，且专门批评 C2C 不适合上下文不同的 agent |
| **Beyond Tokens（综述）** | 2606.05711 | 18 篇方法的统一分类（WHAT/WHICH/HOW 三轴），明确 13/18 是 training-free |
| **LatentQA** | 2412.08686 (ICLR'26) | **合成平行数据的唯一可用配方**（prepend controls → 生成 QA） |
| **Activation Oracles** | 2512.15674 | LatentQA 泛化版，结论：数据多样性 > 数据量 |
| **KV-CAR** | 2512.06727 | 纯重建监督的存在证明（KV autoencoder） |
| **Q-KVComm** | 2512.17914 | 自适应逐层量化 + 异构模型校准 |
| **It's a (Blind) Match!** | 2503.24129 | **免训练对齐的死刑判决书**（N>40 不可行）→ 你「要训练」的最强论据 |
| **Awesome-Latent-Communication** | github.com/enochliu98/... | 18 篇带标签的持续维护列表 |

---

## 来源

- [Vision Wormhole (2602.15382)](https://arxiv.org/html/2602.15382v1) · [v2](https://arxiv.org/html/2602.15382v2)
- [Cache-to-Cache (2510.03215)](https://arxiv.org/html/2510.03215v2) · [GitHub](https://github.com/thu-nics/C2C)
- [Latent Cache Flow (2605.22863)](https://arxiv.org/html/2605.22863v2)
- [L²-VMAS / Dual Latent Memory (2602.00471)](https://arxiv.org/html/2602.00471v2)
- [MACF (2605.00444)](https://arxiv.org/html/2605.00444)
- [Mem-W (2605.09317)](https://arxiv.org/html/2605.09317)
- [Interlat (2511.09149)](https://arxiv.org/html/2511.09149)
- [LatentMem (2602.03036)](https://arxiv.org/abs/2602.03036) · [模型卡](https://huggingface.co/Kana-s/LatentMem-Qwen3-4B)
- [Mixture of Thoughts (2509.21164)](https://arxiv.org/abs/2509.21164)
- [Causal Audit (2607.26773)](https://arxiv.org/html/2607.26773v1)
- [SDE (2506.19209)](https://arxiv.org/abs/2506.19209)
- [LatentMAS (2511.20639)](https://arxiv.org/abs/2511.20639)
- [Beyond Tokens 综述 (2606.05711)](https://arxiv.org/html/2606.05711v3)
- [LatentQA (2412.08686)](https://arxiv.org/abs/2412.08686)
- [Activation Oracles (2512.15674)](https://arxiv.org/pdf/2512.15674)
- [KV-CAR (2512.06727)](https://arxiv.org/abs/2512.06727)
- [Q-KVComm (2512.17914)](https://arxiv.org/html/2512.17914)
- [DiscoNet (2111.00643)](https://arxiv.org/abs/2111.00643) · [GitHub](https://github.com/ai4ce/DiscoNet)
- [It's a (Blind) Match! (2503.24129)](https://arxiv.org/html/2503.24129)
- [BLIP-2 (2301.12597)](https://arxiv.org/html/2301.12597) · [Flamingo (2204.14198)](https://arxiv.org/pdf/2204.14198)
- [Awesome-Latent-Communication](https://github.com/enochliu98/Awesome-Latent-Communication)

---

## 附：关键发现速览

1. **「上界蒸馏」是五条路里唯一已被 LLM agent 场景验证过、且性价比最高的主监督信号** —— 不是我的推测：Mem-W (arXiv:2605.09317) 的 Stage 1 就是把 DiscoNet 配方原样搬了过来：teacher = 同一个冻结 agent 但喂它超长原始上下文 (L' >> L)，student = 只看压缩 latent，loss = action next-token CE + KL(teacher‖student)。Interlat (arXiv:2511.09149) 的 ℒ_task + ℒ_pref 也是同一形状（全长 latent 路径当 target）。上界可构造、监督稠密、完全 label-free、天花板高于「文本通道当教师」。

2. **「文本通道当教师」有现成公式但天花板被锁死，只能当 stage-1 冷启动**。Vision Wormhole 的 loss 是 ℒ_codec = λ_h‖h_vis − stopgrad(h_text)‖² + λ_kl τ² KL(softmax(ℓ_text/τ)‖softmax(ℓ_vis/τ)) + λ_rms(RMS(Δ_inj) − RMS(X̄_img))²（arXiv:2602.15382 原文）。但它 v2 Table 1 的 GSM8K 实际数字是 text 80.8%/27.3s vs VW 76.2%/26.7s = **−4.6pp、1.02× 加速**——学生打不过老师正是这条路的结构性上限。MACF (arXiv:2605.00444) 的消融给了同一结论的正面版：caption-CE 对齐阶段单独只有 36.3%，但去掉它后面全塌（1+3 = 36.5%，等于白加）。

3. **纯 RL 冷启动没有任何一篇论文真的做过**。L²-VMAS (8×H200)、LatentMem (LMPO)、Mem-W (RLOO) 三个用 RL 的工作全都是「先有稠密监督阶段，RL 放最后」。而且 L²-VMAS 论文**完全没有披露 loss 形式、reward 定义、算法名 (PPO/GRPO)、每阶段样本量、wall-clock、模块参数量**——样本效率无从判断，这是复现性缺口不是设计选择。RL 只值得当最后 5-10% 打磨。

4. **参数量的决定因素是工作维度 d，不是 token 数 K**。K×d 的 learned query 参数完全可忽略（K=32,d=4096 只有 0.13M，我的计算）；一个 transformer block ≈ 12d²，d=4096 时 201M/block，d=1024 时 12.6M/block，**16× 差距**。C2C fuser 之所以 478M，就是因为它在 receiver 全宽 KV 空间里、K/V 各一条 pipeline、逐层做；Latent Cache Flow (arXiv:2605.22863) 把瓶颈压到 d=128 后降到 19.4M（24.6× 小），再剪层到 6.24M（76.7×）仍然工作。**推荐区间 10M–60M（bottleneck d=512~1024，4 层）**，硬红线是「必须 ≤ receiver 的 1%」——因为 C2C 自己 Table 6 里「直接微调 receiver」是 596M vs C2C fuser 478M，两者已经逼近，审稿人必问。

5. **占地已经很挤，但缝在「视觉证据本身怎么过去」**。MACF 占了「VLM 多 agent + latent 通信 + 课程训练」，Mem-W 占了「两阶段自蒸馏 + RLOO 训 Q-Former 压缩器」，L²-VMAS 占了「视觉 MAS + RL」。但上述所有工作传的都是 sender 的**推理 rollout hidden**——Vision Wormhole 的 sender 虽是 VLM，论文却不讨论 sender 的视觉输入如何处理。原设想里的 W_a 理论断点在训练路线下变形为：训练能绕开 W_a，但「跨模型传输视觉证据」仍无人正面解决。另需警惕 causal audit (arXiv:2607.26773) 的毒性发现：GSM8K/Qwen3-4B 上「别的样本的消息」效应 −6.17pp 与 CAG +5.17pp 方向相反、总效应 −1.00pp——只看端任务 loss 你分不清训出的是通信还是提示效应。
