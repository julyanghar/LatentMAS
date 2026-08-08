# 先例审计：这个 idea 有没有被做过

> **日期**：2026-08-05 · **问题原述**：「针对现有的多模态 latent collaboration 研究，**特别要关注实验部分**，有没有人已经做过了与我们的 idea 类似的工作/实验」
>
> **方法**：7 个子智能体（3 路 hunt + 2 路逐篇 dissect + 2 路**对抗验证**）。对抗验证的任务是**推翻**结论、默认判 refuted。
>
> **物证**：原始结构化输出 `/tmp/claude-1008/-home-yilin/dc3155e2-adbb-49a4-8fc6-fcefd9b89eda/scratchpad/priorart_raw.txt`（205 KB）· 下载的 PDF/HTML 在同目录及 `/home/yilin/tmp/priorart/`
>
> **本文档里我亲手复核过的**：NavGPT-2 · Sterner et al.（见 [02](02-navgpt2-and-sterner-explained.md)）· **§5/§5.5 那八篇的 PDF 现已全部下载逐一读过**（见 [03](03-eight-risk-papers-explained.md)）。**全部 11 篇 PDF 归档在 [papers/pdfs/](../papers/pdfs/README.md)。**
>
> 🔴 **2026-08-05 重要**：逐一核对后**发现 §5 表里两处事实错误**（Mem-W 的引句是伪造的、SportMV-Agent 的模型与数字都错），已就地标注纠错。**命中率 25%（8 篇错 2 篇）——凡标【调研】未复核的条目，引用前必须回原文核。**
>
> **证据分级**：`【实测】`本项目 · `【复核】`我亲自读原文/原表 · `【论文】`论文报告值 · `【调研】`子智能体报告未二次核对 · `【推断】`判断。

---

## TL;DR（三句话）

1. ⚠️ **我们原来准备写进 paper 的两条"没人做过"，都被推翻了。** 而且不是被 2026 年的新论文推翻，是被 **2024 年的两篇旧论文**推翻：NavGPT-2（ECCV'24）和 Sterner et al.（Cambridge, 2024）。
2. ⚠️ **更坏的消息是方向**：在唯一一篇视觉领域跑过"caption 对 latent，其余全固定，接收方是纯文本 LLM"的论文里，**caption 赢了 4.5 分**。这不是"没人做过"，是"做过而且结果对我们不利"。
3. ✅ **好消息**：真空确实还在，但**位置比我们以为的窄得多**——只剩下「**接收方自己决定 viewer 下一步看什么**」这一个合取项，而且 NavGPT-2 顺手给了这条一个**已发表的负结果**（把决策权交给冻结文本 LLM，成功率从 67.52 掉到 21.46）。

---

## §1 先把 idea 写成能判真假的形式

要问"有没有人做过"，必须先把 idea 拆成能逐条打勾的性质。全文用这五个代号：

| 代号 | 内容 | 白话 |
|---|---|---|
| **P1** | 发送方看过像素 | viewer 是真的过了视觉编码器，不是只读文字 |
| **P2** | 通道是连续向量，不是文字 | 传 hidden state / KV / soft token，不传 caption |
| **P3** | 接收方是**独立挑选的、冻结的纯文本推理 LLM** | 没有视觉塔、不是跟 viewer 一起训出来的、forward 图没被改 |
| **P4** | 多轮循环，且**接收方决定 viewer 下一步看什么** | reasoner 说"去看 340–420 帧，高分辨率" |
| **P5** | 有一个训出来的跨架构 adapter | 因为两边空间不通 |

我们的 idea = **P1+P2+P3+P4+P5 全部同时成立**。

> 📌 **想看每条的判据和手把手示例**（拿 Mem-W / NavGPT-2 逐格走一遍）→ [03 §0.5](03-eight-risk-papers-explained.md)。

**为什么必须拆这么细**：下面你会看到，P1+P2+P3 这三条的组合**五年前就是标准做法**（BLIP-2 那一套），P2+P5 在 2026 年已经是成熟工程（C2C 一系）。真正没人占的只有 P4，而且要加限定词。

---

## §2 ⚠️ 主张一被推翻：NavGPT-2（ECCV 2024）

**原主张**："没有已发表工作让一个看过像素的发送方，把 latent 状态交给一个独立挑选的纯文本推理 LLM，在多轮 agentic 循环里用。"

**推翻它的论文**：[NavGPT-2: Unleashing Navigational Reasoning Capability for Large Vision-Language Models](https://arxiv.org/abs/2407.12366)（ECCV 2024，Adelaide/Adobe/UNC/UCSC）。

【复核】我下载了 PDF 逐条对：

| 性质 | NavGPT-2 的原文 | 判定 |
|---|---|---|
| **P1** | "for a candidate view image o_i, we incorporate a **frozen ViT-g/14 from EVA-CLIP** as the vision encoder to extract visual feature" —— 真实 RGB 全景视图，**每一步都重新编码** | ✅ |
| **P2** | "These visual features are later cross-attended with **32 learnable queries embedding Q_i ∈ R^{32×768}** ... These queries are **fed to the LLM after a linear projection W as the image tokens**" | ✅ |
| **P3** | "**all parameters of the vision encoder and LLMs are kept frozen during the entire training process**"；四个纯文本 LLM 互换：**FlanT5-XL (3B) / FlanT5-XXL (11B) / Vicuna-7B / Vicuna-13B**，Val-Unseen SR 分别 **67.52 / 71.31 / 53.77 / 48.28**（Table 6） | ✅ |
| **P4**（只按"多轮循环"这个字面） | 每一步在当前视点重新编码周围视图 → LLM 重新被调用 → 策略选下一个视点 → 新观测。R2R 一整段 episode | ✅ |
| **P5** | 只训 Q-former + projection（stage 1），LLM 和 ViT 全冻 | ✅ |

> ⚠️ **P3 这一格特别致命**：它不是"用了一个纯文本 LLM"，而是**四个模型来回换、全程冻结**。这比我们打算做的"独立挑选"证明得更强——**"独立性"不能当我们的区分点**。

**我们原来的辩解站不住**：我们的说法是"循环是由一个单独训练的导航策略闭合的，不是 LLM 自己选看哪里"。这话是对的，**但这个条件不在原主张里**——原主张只说"在多轮 agentic 循环里"。**我们的 TL;DR 悄悄把它写成了"receiver-steered 多轮循环"，两种措辞之间的落差就是全部的新颖性余量。**

**另外两个旁证（弱一些）**：

- **PaLM-E**（ICML 2023）：ViT 特征经 learned affine 投进 LM embedding 空间，与文字 token 交织成前缀；"In all experiments, the LLM is frozen"（TAMP）；且**LLM 自己闭环**——"integrated into a control-loop... **PaLM-E is able to replan**"。【调研】P3 是按 arm 而定（我没确认闭环 rollout 用的是冻结版还是共训版）。**它最危险的地方是 steering 那半：reasoner 真的在闭环。它只是不能选视角——相机是被动的。**
- **RoboFlamingo**（ICLR 2024）：冻结纯文本 LLM + Perceiver latent 走 K/V，CALVIN 闭环 5 连续子任务。P3 弱一些——Flamingo 式在接收方**内部插了可训练层**，推理时它已经不是 stock 模型。【调研】

### 2.1 ⭐ NavGPT-2 顺手给了我们一个已发表的负结果（子智能体没报，我自己读出来的）

【复核】Table 5 有一个我们必须知道的 arm。他们把导航策略网络**拿掉**，"we **force the LLM to take over** the visual-textual based decision-making to exploit its navigational capability"：

| Val-Unseen | SR ↑ | SPL ↑ | NE ↓ |
|---|---|---|---|
| NavGPT-2 完整（策略网络在） | **67.52** | 56.01 | 3.37 |
| **w/o policy model（冻结 LLM 自己决策）** | **21.46** | 10.23 | 8.03 |

**SR 掉 46.06 分。** 论文原话：**"a frozen LLM is incapable of inferring effective representations that indicate a correct action."**

> ⚠️ **这条直接打在我们计划的心脏上。** 我们的 P4 就是"让冻结纯文本 reasoner 从 latent 视觉状态里决定下一步看什么"。NavGPT-2 已经测过最接近的版本，**结论是不行**。
>
> **能保留的差别有三个，必须写清**：(a) 他们要的输出是**离散动作**（选哪个视点），我们要的是**语言化的取帧请求**——后者恰好是 LLM 的强项；(b) ⚠️ ~~他们是输入层 soft token 不是逐层 KV~~ **此条作废**——我们的载体也是输入层伪 token（[03 §0.5](03-eight-risk-papers-explained.md)）；真差别只剩 adapter 的**读料**（他们读 ViT 特征、我们读前 4~10 层 KV），属机制细节；(c) 他们没训过任何"让 LLM 学会读这种 latent 然后发指令"的东西——Q-former 只在 stage 1 用**推理文本**监督训过。
>
> **但这是一个"必须先证伪的对照"，不是可以绕过去的细节。** 计划 §3 的判死点必须加一条：**如果我们的 reasoner 在 latent 上的取帧决策明显差于它在 caption 上的取帧决策，整条路就是 NavGPT-2 Table 5 的重演。**

> ⚠️ **2026-08-05 修订（本节上面那句"结论是不行"说重了，收回一半）**：Table 5 测的是"**用一个几乎线性的读出头，从冻结 LLM 隐状态里解码离散动作**"，**不是**"冻结 LLM 用语言决策"。三角定位（详见 [02 §C.4.3.1](02-navgpt2-and-sterner-explained.md)）：
>
> | 谁决策 | 看到什么 | SR |
> |---|---|---|
> | LLM 用语言 | caption（GPT-4） | 34 / 39 / **43** |
> | ⭐ **LLM 用语言** | **latent** | **❓ 无人测量 = 我们的格子** |
> | 浅层读出头 | LLM 隐状态 | **21.46** |
> | 策略网络 | LLM 隐状态 | 67.52 ~ 74 |
>
> **这个格子既没被证伪也没被支持，但两个邻居是 43 和 21.46——先验不乐观。** 同时它的人类评分（Accuracy **1.66/3**、Rationality **1.78/3**，vs GPT-4V 2.31/2.34）说明"读懂"只有五六成，**救不了我们**。
> **判死点 D0 因此拆成三级（D0-a 复述 / D0-b 示范敏感度 / D0-c 训练单调性），见 [02 §C.4.5–C.4.6](02-navgpt2-and-sterner-explained.md)。**

---

## §3 ⚠️ 主张二被推翻，而且方向对我们不利

**原主张**："没有多模态论文跑过'caption 通道 对 latent 通道，其余全固定，且接收方是纯文本模型'这个 arm。"

**推翻它的论文**：[Few-Shot VQA with Frozen LLMs: A Tale of Two Approaches](https://arxiv.org/abs/2403.11317)（Sterner, Lin, Chen, Byrne；剑桥；2024-03-17）。

【复核】我把全文抽出来读了。**这篇论文存在的唯一目的就是跑这个 arm**：

- 摘要开头：*"Two approaches have emerged to input images into LLMs. The first is to **caption** images into natural language. The second is to **map image feature embeddings** into the domain of the LLM and pass the mapped embeddings directly."*
- 对照比我们打算做的**更严**：文字臂的 caption 不是另一个 captioner 产的，而是**同一个冻结 LLM 从同一份 latent 解码出来的**。原文：*"The only difference is that in caption-based VQA, the embeddings are **first passed to the LLM alone to generate a caption**, before being concatenated with the question."* 又：*"This is a surprising result, because **the same visual representation is used in both approaches**."*
- 接收方：**Flan-T5 XL（3B，encoder-decoder，纯文本，全程冻结）**；发送方：raw pixels → 冻结 CLIP ViT-G → 单隐层 mapping net（1280-D 入，20×2048-D 出），在 Conceptual Captions 2.7M 上**穿过冻结 LLM 反传**训出来。
- 判分：VQAv2 官方指标，配对置换检验（p<0.01）。

**Table 1（VQAv2，shots 0/1/2/4；R=随机选例，Q=按问题相似度，Q+I=问题+图像相似度）**：

| 通道 | 选例 | 0-shot | 1 | 2 | 4 |
|---|---|---|---|---|---|
| **Caption** | R | **45.9** | 45.3 | 45.1 | 45.3 |
| **Caption** | Q | **45.9** | **47.6** | **48.0** | **48.5** |
| **Caption** | Q+I | **45.9** | 50.0 | 50.3 | 50.8 |
| **Latent** | R | 41.4 | 44.5 | 43.6 | 42.5 |
| **Latent** | Q | 41.4 | 40.6 | 45.7 | 47.3 |
| **Latent** | Q+I | 41.4 | **50.5** | **51.8** | **52.4** |

**逐格做减法（latent − caption）**：

```
0-shot（三种选例等价）：      41.4 − 45.9 = −4.5     ← caption 赢，p<0.01
R  选例 1/2/4 shot：         −0.8 / −1.5 / −2.8      ← caption 全赢
Q  选例 1/2/4 shot：         −7.0 / −2.3 / −1.2      ← caption 全赢
Q+I 选例 1/2/4 shot：        +0.5 / +1.5 / +1.6      ← latent 只在这一行赢，且幅度小
```

论文自己的结论：*"connecting visual embeddings directly to the LLM embedding space **does not necessarily improve performance** compared to using image captions. We find that **the selection of relevant in-context examples is far more important**."*

而且它**连"这个对照被前人漏了"这个观察本身都已经发表了**：*"It is therefore a natural baseline to compare their embedding-based VQA approaches with that of using image captions generated by the same system. However, **none of the aforementioned systems report such results**."*

**第二个独立反例（不同模态，且方向相反）**：Kang & Roy（MIT，**Interspeech 2024**）。语音波形 → HuBERT-Large → 4× 平均池化 → 线性投到 3072-D 连续 token → 冻结 MiniChat-3B。文字臂用**同一个 HuBERT-Large** 做 ASR 出转写。【调研】ROUGE-1/2/L：cascade 49.18/25.56/34.86 vs latent **51.12/27.50/37.48** —— **这里 latent 赢了**。

> ⚠️ **两篇已发表的同款实验结论相反。** 所以我们既不能说"我们第一个问这个问题"，也不能把"latent 比文字准"当前提——**它是待检验的假设**。
>
> 再叠上 **ViF Table 11**（同一个 MAS、同样 prompt）：**mean-pooling 的 latent 在 5 个库里输给纯文字 4 个**（MMHal 42.5 vs 47.9 = −5.4；HallBench 46.7 vs 53.1 = −6.4）。**这是第三个"latent 输给 caption"的已发表实例，而且它同时说明差别在于 latent 选得好不好。**

---

## §4 修订后还能说的话（每个合取项都在承重）

【调研+复核】对抗验证给出的可存活表述，我按自己复核后的措辞重写：

> **没有已发表工作，让 (a) 一个自己会推理、独立部署的 VLM agent（它看过像素），(b) 把 latent 状态而非文字，(c) 交给一个从未与它共训、forward 图未被改动的冻结现成纯文本推理 LLM，(d) 并且在多轮循环里由这个纯文本接收方决定 viewer 下一步看什么（哪些帧/区域、什么分辨率）。**

**哪个合取项杀掉谁**（这张表就是 related work 的骨架）：

| 合取项 | 杀掉的先例 | 理由 |
|---|---|---|
| **(a) 发送方是 agent，不是接收方自己的感知前端** | NavGPT-2 · PaLM-E · RoboFlamingo · CogRad · Frozen/BLIP-2/LiMBeR/eP-ALM/LLaMA-Adapter 全谱系 | 它们的"发送方"是一个裸编码器模块，服务于接收方本身，没有自己的推理回路 |
| **(c) 冻结、现成、未共训、架构未改** | **MACF**（coordinator 是 Qwen3-VL-8B，且经 3 阶段课程与 sender 共训）· **CogRad**（LLaMA-2-7B 被 LoRA 微调过）· Flamingo/MAGMA/RoboFlamingo（在接收方内部插了层） | |
| **(d) 接收方决定看什么** | **NavGPT-2**（动作由单独训练的拓扑策略网络选，环境供下一视图）· **PaLM-E**（LLM 会 replan，但相机被动，它选不了视角）· **MACF**（严格前馈，无反向通道） | |

**两条必须记住的定位警告**：

1. ⚠️ **不要再引用 Vision Wormhole 的 "off-manifold / 纯文本接收方会崩" 那段当"领域认为不可能"的证据。** Frozen（2021）就把 2 个来自从头训练的像素编码器的连续向量塞进冻结 7B 纯文本 LM 并成功了；NavGPT-2 在循环里对四个不同冻结文本 LLM 都做成了。**懂这条谱系的审稿人会把这个 framing 读成稻草人。这个反对意见五年前就被实验解决了。**
2. ⚠️ **不要让"独立挑选"承担新颖性。** NavGPT-2 的 LLM 全程冻结、四模型互换，独立性比我们大多数设计都更强；而我们的 adapter 也会**按接收方逐个训**，跟他们的 Q-former 一样。**独立性不能当区分点，只有 (a) 和 (d) 能。**

---

## §5 新发现的危险论文（原计划 §8 没有的）

| 论文 | 满足几条 | 危险在哪 | 我们的差别 |
|---|---|---|---|
| ⚠️⚠️ **The Latent Bridge**（[2606.24470](https://arxiv.org/abs/2606.24470)，Jun 2026） | **P1+P2+P5** | Qwen3-VL-8B-Thinking 把 residual 经**训好的 bridge** 投进第二个模型的 input-embedding 空间；**带一条显式的 Text Bridge 基线并声称在每个域都赢**（7 个 Atari + MetaDrive）。**这就是我们的 F/T/L 三臂消融结构** | 接收方 **MiniCPM-o 4.5 是 9B VLM**（P3 挂）；实时反应式控制，没有 receiver 重派 sender 的轮次（P4 挂）。⚠️ **~~通道是 embedding 级不是逐层 KV~~ 此条作废**——我们的载体也是输入层伪 token，**这是相同点** |
| ⚠️ **Mem-W**（[2605.09317](https://arxiv.org/abs/2605.09317)，May 2026） | **P1+P2+P4+P5 = 4/5** | GUI agent 全程冻结、只训 compressor；K=8 soft memory token 喂回**同一个**冻结 encoder；多轮导航，最高 +30.0。**"给看像素的 agent 做 latent 记忆"这个 framing 被占了** | ⚠️ **2026-08-05 纠错**：审计初版引的 "no separate text-only LLM is involved" **在原文中不存在**（全文 `text-only` 出现 0 次，`K=8` 0 次）——**子智能体伪造的引句**。结论不变但理由改为：**写记忆和读记忆是同一个冻结 VLM**（Qwen3-VL-4B / UI-Venus-1.5-8B），无模型边界（P3 挂）。**我们的差别就是那条边界** |
| ⚠️ **SportMV-Agent**（[2607.11844](https://arxiv.org/abs/2607.11844)，Jul 2026） | ⚠️ **P1+P4**（**非** P1+P3+P4，见纠错） | 迭代主动选视角 + 感知工具。**68.35 vs GPT-4.1 基线 59.12**（Table 2）。**我们的 workflow 形状被占了** | ⚠️ **2026-08-05 纠错两处**：(a) 数字应为 **68.35 / 59.12**（初版 68.31/59.68 错）；(b) **GPT-4.1 在此是当多模态模型用**——原文 *"whenever a view switch occurs, the video segment from the newly selected view is integrated into the agent's observation context"*，**不是纯文本 orchestrator，P3 不成立**。→ ✅ **P1+P3+P4 那个角仍只被 SeeingEye/BeMyEyes 占**。它当强基线原样跑 |
| **ChronoStitch**（[2607.19547](https://arxiv.org/abs/2607.19547)） | P2 | **免训练**把独立缓存的视觉 KV 拼起来：全局三轴 mRoPE 重定基 + 选择性重算，比全量重 prefill 快 3.3×。**这正是我们跨比较表里列为障碍 (a) 的那个机制** | 模型内，不跨模型边界。**mRoPE 重定基不能当我们的贡献，只能引为"模型内先例，我们跨边界推广"** |
| **LinguDistill**（[2604.00829](https://arxiv.org/abs/2604.00829)） | P1+P2(KV)+P3 | 目前找到的**唯一**真把逐层 KV 从像素模型交给冻结纯文本 LM 的论文 | 同架构（nanoVLM-460M ↔ SmolLM2-360M，**KV 形状必须一样才行**）、**只在训练时存在**（推理是普通 VLM）、冻结 LM 是蒸馏 teacher 不是 reasoner、无循环。**但必须引，不能说"没人把 VLM KV 交给冻结文本 LM"** |
| **两篇 7 月的怀疑派** | 都是纯文本 | [2607.26773](https://arxiv.org/abs/2607.26773) 因果审计：**总准确率识别不出 latent 消息干了什么**，GSM8K/Qwen3-4B 上 −1.00pp 拆成方向相反的两块，**4B 和 8B 结论反转**。[2607.14103](https://arxiv.org/abs/2607.14103)：latent 通道**在测过的任务上不超过文字**；文字序列化毁掉 88% 的 SAE 特征，**但丢掉的主要编码表层形式，不是任务语义** | 不是先例，是**评测设计约束**：单报准确率差会被质疑没识别机制；"文字有损"这个 motivation 会被这两篇打。**反过来这也是我们的开口——它们的零结果都是文字发送方，序列化只丢表层；像素发送方才是丢语义的那个 case，两篇都没测** |

---

## §5.5 ⭐ 两个最高危口子已读完（2026-08-05 补，我亲自读的 PDF）

原 §10 排第 1、第 2 的两个"只有摘要"的口子已闭合。PDF 在 `/home/yilin/tmp/priorart/2606.07512.pdf` 和 `2602.06566.pdf`。

### 5.5.1 MemDreamer（[2606.07512](https://arxiv.org/abs/2606.07512)，v2 2026-06-24）—— **不是先例，但抽掉了我们的安全前提**

【复核】**判决：P2 = NO。通道是"purely textual"图记忆。** 全文 `latent` / `hidden state` / `KV` / `key-value` 出现次数 **全部为 0**，`caption` 只出现 1 次。原文两句写死：

> "a perception model P first processes the visual inputs in a streaming fashion to construct a structured, **purely textual** Hierarchical Graph Memory, denoted as G."
>
> "**During inference, R operates only on this pure text memory, never raw video information.**"

**而且它主动拒绝回看原始帧，并把这当卖点**：

> "Unlike previous leading methods that **require additional perception models to revisit raw video frames during retrieval**, our approach relies solely on interactions with text memory."

→ **我们的 P4（接收方让 viewer 重新感知）它是故意不做的**，跟 L²-VMAS 一样是"从静态存储里检索"。P3 也挂：reasoner 是 Gemini-2.5-Pro / Gemini-3.1-Pro / Qwen3-VL-235B-A22B-Thinking，**全是 VLM**（推理时只吃文字，与 MACF coordinator 同一个失败模式）。没有 caption-vs-latent 对照。

⚠️⚠️ **但它有一组对我们 motivation 极不利的数（LVBench，Table 3+4）**：

| 角色 | 模型 | 上下文 | LVBench |
|---|---|---|---|
| 端到端（整段视频喂进去） | Gemini-3.1-Pro | 265K | 78.2 |
| 端到端 | Gemini-2.5-Pro | 784K | 72.0 |
| 端到端 | Qwen3-VL-235B-A22B-Thinking | 240K | 63.6 |
| MemDreamer 感知侧 | Gemini-3.1-Pro | 40.3K | — |
| **MemDreamer 推理侧** | Gemini-3.1-Pro | **6.2K** | **90.7（+12.5）** |
| MemDreamer 推理侧 | Gemini-2.5-Pro | 6.3K | 80.7（+8.7） |
| MemDreamer 推理侧 | Qwen3-VL-235B | 5.9K | **84.8（+21.2）** |

> ⚠️ **一个纯文字通道，在小时级视频上，比"把像素全喂进去"高 8.7 ~ 21.2 分，上下文还小 41~124 倍。**（比值我自己算过：265/6.2=42.7、784/6.3=124.4、240/5.9=40.7。）**如果文字通道是主要瓶颈，这不可能发生。**

Table 5 更狠：端到端下"模型的 AIME2025 推理能力"与 LVBench 成绩的相关性只有 **R=0.702, p=0.052（不显著）**；接上 MemDreamer 后 **R=0.897, p<0.01**。原文：*"raw video ingestion creates **perceptual barriers** that hinder models from leveraging their reasoning capacity."*

✅ **同一篇论文的 Table 6 却强力支持我们的另一半主张（"坚持文本 reasoner"）**：

| 换什么 | 幅度 |
|---|---|
| 换**感知**模型（reasoner 固定） | **0.4 / 1.4 / 2.5 pp** ⚠️ 论文正文只引了前两个 |
| 换**推理**模型（perception 固定） | 90.7−80.7 = **10.0**；90.3−78.2 = **12.1** |

→ **reasoner 比 perceiver 重要约 4~30 倍**，独立复现了 VideoAgent 的 +11.4 vs +0.6（≈19×）。**我们计划 §1"为什么坚持文本 reasoner"这一条得到第二份独立证据。**

⭐ **我自己的判读（不要照抄论文口径）**：那个 +12.5 **混淆了两件事**——(i) **通道**（文字 vs 像素）与 (ii) **上下文预算与结构**（6K 结构化 vs 240K 平铺）。端到端臂是"整段视频像素一次性平铺进 240–784K 窗口"；MemDreamer 感知侧是"10 分钟滑窗、流式、逐段处理"。**两臂的视觉吞吐量和上下文结构都不同——这正是我们自己的门槛 E4(d)（帧数/视野混淆）。所以它不是干净的通道对照。**

> ⭐ **而我们计划 §3 的 C 臂恰好是那个干净版本**：给 reasoner 的是**同一批被选中帧**的像素（不是整段视频），文字臂是**同一批帧**的 caption。**MemDreamer 没跑这个对照 → 它没有抢先我们的实验。**
> **但它确实抽掉了"文字有损，所以要换 latent"这个安全前提。** 修订后的 motivation 只能是：**"文字丢的那部分，在把上下文预算控住之后还剩多少"** —— 这正是 C−D 要量的东西，判死点不变，但**先验概率往不利方向移了**。

### 5.5.2 SPARC（[2602.06566](https://arxiv.org/abs/2602.06566)，**ICML 2026 已录用**）—— 命名冲撞 + 效率对手

【复核】**判决：P2 = NO，P3 = NO。** 是**同一个 Qwen3-VL**（对 vision 与 language 两侧都加了 LoRA），**两阶段而非循环**（`loop` 全文 0 次），接口是 **crop / 区域**（`crop` 出现 70 次；`latent` 只 2 次，且都是 "latent visual relevance" 这种形容词用法）。原文：*"the model first performs explicit visual search to localize question-relevant regions, then conditions its reasoning on those regions"* + *"running global search at lower image resolutions and allocating **high-resolution processing only to selected regions**"*。

**它属于我们早已列出的"原生 crop/zoom 一族"，但两件事变了**：

1. ⚠️ **它是 ICML 2026 已录用**，不是 preprint。这条威胁的分量升级了。
2. ⚠️⚠️ **它直接打我们的效率论证**。我们的说法是"重编码要付 N 次视觉 prefill，KV 不用"。SPARC 的做法是"低分辨率全局搜 + 只给选中区域上高分辨率"，**同样不付 N 次全量 prefill**，而且已经拿到：Qwen3VL-4B 在 **V\* VQA +6.7**；OOD 上比 "thinking with images" 高 4.6 分而 **token 预算低 200×**；附录 Table 8：**SPARC@256 分辨率就超过 thinking-with-images@全分辨率，TTFT 快 200×、E2E 快 50×**。
3. **命名冲撞**：它叫 "**Separating Perception And Reasoning** Circuits"。**这个 framing 不能当我们的新颖点。**

> ⭐ **对计划的直接影响**：§7.2 的效率账**必须把 SPARC 这条线当对手**，不能只跟"全量重编码"比。我们唯一还站得住的效率差别是：**crop/zoom 一族每次重看都要重新过视觉编码器，而 KV 复用不过**——这个差必须量出来（Kamera 那个 230ms 视觉塔编码 vs ~5ms KV replay 的数就是干这个用的）。

---

## §6 ✅ 好消息：真空确实存在，而且有三份独立证据

1. ⭐ **原计划标为"最危险抢先"的 Latent Cache Flow（[2605.22863](https://arxiv.org/abs/2605.22863)）没有发生。** 【调研】Semantic Scholar 引用数 **0**；v2 全文（6 Jun 2026）里 **vision / VLM / image / video / multimodal / pixel 出现次数全为 0**。它的 limitations 把"richer agentic workflows with longer contexts, tool outputs, heterogeneous roles, and multi-turn communication"列为 future work——**它认领了我们的 P4，但完全没有往视觉动**。
2. ⭐ **穷举扫描：2026-05-01 至 2026-09-01，arXiv 摘要里同时含 "KV cache" 和 "visual" 的论文共 39 篇，全部是模型内**（压缩/淘汰/复用）。【调研】**没有一例把视觉 KV 或视觉 hidden state 交给另一个模型。** 这比"关键词碰运气"强——它是在时间窗内的枚举。
3. ⭐ **第三方 survey 背书**：[2606.05711](https://arxiv.org/abs/2606.05711)（"Beyond Tokens"，截止 2026-07-15）编目 18 个 latent communication 方法，**只有 Vision Wormhole 一个碰视觉**，而它九个 benchmark 全是文字题；survey 的 Open Problems 一节**根本没有"多模态/视觉发送方"这一项**；"bidirectional / feedback / multi-round / multi-turn" 在全文出现 **0 次**，原文断言 *"No method is described as inherently iterative with bidirectional feedback loops."*
   ⚠️ **但这份 survey 漏了 MACF（5 月）和 L²-VMAS（2 月），两篇都在它截止日之前。** 所以它对多模态那半的覆盖不完整，**"0 篇"要当"至少还有 1–2 篇没被索引到"来防**。
4. **C2C（ICLR'26）自己把两半都写成 future work**：*"C2C may serve as a better communication primitive for real-world agentic tasks with complex multi-round reasoning, coding, and tool use"* 和 *"fusing caches among vision-language models (VLMs) and vision-language-action (VLA) models may enable richer multi-modal collaboration."* 【调研】**这是"这一步被公认是下一步"的引文。**
5. **没有任何多模态 latent-MAS 有可跑的公开代码。** 【调研】L²-VMAS 仓库 404（网上流传的链接是编出来的）；MACF 无代码且正文指向的附录在 12 页 PDF 里不存在；ViF 仓库是空壳（`hidden_states` 是 `torch.randn`，`vif/multiagent/` 从不 import ViF 模块，`agent.py` 是单轮文字多数投票）。→ **所有多模态 latent 基线都得我们自己搭**（成本），**但也没人能抢先复现我们**（利）。

---

## §7 五篇多模态 latent MAS 的通道对照汇总（数字，供 related work 直接用）

【调研】全部为"其余固定、只换通道"的 arm。⚠️ 引用请用**表**不要用正文（见下）。

| 论文 | 文字臂 | latent 臂 | 差 | 关键 caveat |
|---|---|---|---|---|
| **MACF** [2605.00444](https://arxiv.org/abs/2605.00444)（视频，同 Qwen3-VL-8B 引擎，同 16×224×224 预算） | MapReduce\* **46.7 / 38.2 / 30.5 / 33.3**<br>(Video-MME/LongVideoBench/LVBench/MLVU) | 裸 KV (LatentMAS\*) 55.7/48.5/33.2/42.3<br>训好的 token (MACF) **60.4/56.8/40.2/49.2** | 文字→MACF **+13.7/+18.6/+9.7/+15.9**<br>裸KV→MACF +4.7/+8.3/+7.0/+6.9 | ⚠️ **正文写 "20.3%" 和 "9.3/14.1/10.5/10.3"，跟自己的表算不上**。只引表 |
| **L²-VMAS** [2602.00471](https://arxiv.org/abs/2602.00471)（5 个 VLM backbone） | VMAS 文字传递，如 Qwen3-VL-8B-Inst 均 71.4 @2480 tok | L²-VMAS 双 latent memory **74.1 @1889 tok** | **+2.7 ~ +5.4 分，token −21.3% ~ −44.8%** | ⚠️ **混淆**：它同时加了检索、触发、感知/思考解耦、3 阶段 RL。**不是纯换通道** |
| **ViF** [2509.21789](https://arxiv.org/abs/2509.21789)（Table 11，LLaVA-NeXT-7B，circular） | 纯文字流 CHAIR 43.0 / POPE 91.0 / AMBER 89.4 / MMHal 47.9 / HallBench 53.1 | unimodal latent relay **41.2 / 93.3 / 92.7 / 51.1 / 55.7** | −1.8(越低越好)/+2.3/+3.3/+3.2/+2.6 | ⭐ **同表里 mean-pooling latent 输给文字 4/5**（MMHal 42.5、HallBench 46.7）；MLP 压缩基本打平。**"latent 赢文字"不自动成立** |
| **LACO** [2605.22504](https://arxiv.org/abs/2605.22504)（5 个驾驶 VLA，权重不动，CARLA 闭环） | Language ORION V0 DS 29.34 | Visual-token 31.48 → **KV latent 35.48**（Noncollab 26.68） | 5 个 backbone 里 4 个是单调 none<language<visual<latent；LMDrive-Vicuna 上 **language 比不通信还差**（24.79 vs 25.16） | 唯一带**线路带宽**账的：language 1.8–2.5KB 但 1300–8509ms；visual 52–95ms 但 896–8208KB；latent 203–430ms / 103–4881KB |
| **Vision Wormhole** [2602.15382](https://arxiv.org/abs/2602.15382)（对照最干净：*"the channel is the only changed factor"*，54 个配对格） | TextMAS macro 52.7 / 45.2 | VW macro **54.5 / 52.1** | 宏平均正，**逐格噪声大**：最好 +26.2（HumanEval+），最差 −10.0（AIME2024） | ⚠️ **P1 挂**：九个 benchmark 全是文字题，"图"是固定 dummy，token 段被解码器输出覆盖。**发送方名义上是 VLM，实际从不是 viewer** |
| **CogRad** [2607.03853](https://arxiv.org/abs/2607.03853) | —— | —— | —— | **完全没有这种对照**。两行消融都是删辅助 loss，visual prefix 在每个 arm 里都在。**它断言 latent 优于文字但从不测** |

**两条从这张表里读出来的、能直接当 motivation 的数**：

- ⭐ **多模态里文字通道不只是次优，是有害到掉破单模型地板**。MACF 同预算单模型 Qwen3-VL-8B = 55.9/50.7/33.2/41.5，而文字多智能体臂 46.7/38.2/30.5/33.3 —— **四个库全输，−2.7 ~ −12.5**。L²-VMAS 独立复现：GLM-4.1V-9B 单模型 66.7 vs 文字 VMAS 66.6，**token 从 596 涨到 6061（10×）换来 −0.1**。
- ⭐ **Vision Wormhole 的 OCR 臂顺手替我们杀掉一个必被问的反对**："直接把 caption 渲染成图片、让接收方用自己的视觉塔读呢？" 实测 macro **39.1 / 34.2**，比纯文字差 **−13.6 / −11.0**，比 latent 差 **−15.4 / −17.9**，而且更慢。

---

## §8 ⭐ 六条直接改我们设计的硬约束

这一节是本次调研最实用的产出。每条都来自一个已发表的负结果。

| # | 约束 | 证据 | 怎么改设计 |
|---|---|---|---|
| **1** | **训过的 latent 通道全面吊打免训练的**，四篇同时测两者的论文无一例外 | MACF：免训练 LatentMAS\* 55.7/48.5/33.2/42.3 vs 训过 60.4/56.8/40.2/49.2 · DiffMAS：LatentMAS AIME24 56.7 vs DiffMAS **76.7** · HyLaT：免训练基线"severe performance drop" · VW 弱监督版（<100 anchor）在 9 个库里 5 个转负 | **P5 是决定变量，不是实现细节。** 免训练版只该当**故意报告的负结果**跑一次（1 小时判死，见计划 §3.2），不要投入更多 |
| **2** | ⚠️ **纯 latent 通道在多轮里会漂**，这正打在 P4 上 | **HyLaT** [2605.25421](https://arxiv.org/abs/2605.25421)：同一个训好的模型只翻通道，8 库均分 **56.93 → 50.28（−6.65）**，格式错误率 3.28% → 6.12%（约翻倍）。原文归因：*"the intermediate text channel serves as a **semantic anchor** that maintains contextual coherence across rounds."* 另 LACO Figure 6：latent 推理步数过长→"semantic drift / over-reasoning" | **我们的系统必须**设计**成混合**：**保留一条细的文字控制通道**（reasoner 的取帧请求、最终答案），**只把感知载荷放进 latent**。而且要在审稿人说之前自己说 |
| **3** | ⚠️ **全深度 KV 融合是最差设定**（"agent identity confusion"） | **LACO** Table 3（SSKD 深度）：ORION V1 全深度 **27.61** vs 10% 深度 **32.65**——**全传比只传浅层差 5.04**。机制：接收方过度注意发送方的内部状态，自己的策略被劫持 | ⚠️ **范围注记（2026-08-05）**：此规则仅适用于**载体=直接 KV 注入**的路线（计划 §5.2 的 (i)，已弃）。我们的载体是输入层伪 token、adapter 主动读取，**不存在劫持病理**；读料层深的证据反而偏**中层**（ViF drop study + Latent Bridge 读 24/36 层，见 [00-plan §5.3](../VideoAgent-Plan/00-plan.md)） |
| **4** | **latent 通道必须用文字监督引导** | **MACF** Table 4：三阶段课程里 **Stage 1（caption 监督语义对齐）是最重要的一个**——去掉它 Video-MME −14.9、MLVU −12.9 | adapter 的第一阶段损失就该是"从 latent 解出 caption"。⭐ **顺带一个乐观推论**：我们的接收方**本来就母语说文字**，可能比 VLM 接收方**更容易**被引导，不是更难 |
| **5** | ⚠️ **文字基线必须是强基线** | **BeMyEyes** [2511.19417](https://arxiv.org/abs/2511.19417)（Qwen2.5-VL-7B perceiver + DeepSeek-R1）：R1 单独 54.8/36.8/37.1/42.6 → +BeMyEyes **67.4/57.2/72.7/48.5** · **SeeingEye** [2510.25092](https://arxiv.org/abs/2510.25092) 外循环 1→2→3 轮：**34.21 → 36.84 → 44.62（+10.41）** · SportMV-Agent 68.31 vs GPT-4.1 59.68 | **不能拿 caption map-reduce 当对照**（那是稻草人）。文字臂必须是 SeeingEye / SportMV-Agent 强度，**而且原样跑，不重写** |
| **6** | ⚠️ **只报准确率差会被质疑没识别机制** | [2607.26773](https://arxiv.org/abs/2607.26773)：总准确率识别不出 latent 消息的作用；GSM8K/Qwen3-4B 上 −1.00pp 拆成方向相反两块；**4B 与 8B 结论反转** | 计划 §7.2 的 **CAG（content attribution gain）对照必须进主表**，不能当附录。并且**至少两个尺寸的接收方**，否则"结论会反转"这条会被直接拿来打 |

### 8.1 ⭐ 一个白送的评测切片

**BeMyEyes 的数据流水线会丢弃**那些"produce no conversations that lead to the correct answer... **typically because the visual information is too complex or abstract to be effectively conveyed through text-based communication**"的样本。【调研】

> **他们量到了文字瓶颈，然后把难题扔了，而不是修它。那批被扔掉的样本，就是我们最好的评测切片。**

---

## §9 两条"别用"的论证（会被反打）

1. ❌ **别用 Vision Wormhole 的 off-manifold 段落**当"领域认为纯文本接收方不可能"。见 §4 警告 1——Frozen(2021)/NavGPT-2 已经做成了。
2. ❌ **别把 "latent 比文字更保真" 当前提**。已发表的三个反例：Sterner et al.（视觉，caption 赢 4.5）、ViF Table 11（pooled latent 输 4/5）、HyLaT（多轮里 latent 输 6.65）。**它是待检验假设。** 我们自己的 C1 也同向：一旦下游能看图，latent 相对文字只剩 +1.2 / −4.7 / +9.9 / −0.5（见 [phase2-results.md](../VMAS-latent-collab/phase2-results.md) `C1-rev`）。

---

## §10 没闭合的口子（按优先级）

| # | 口子 | 为什么要紧 |
|---|---|---|
| 1 | **MemDreamer**（[2606.07512](https://arxiv.org/abs/2606.07512)，"Decoupling Perception and Reasoning for Long Video Understanding"）只读到摘要，无 HTML | **标题是所有搜到的东西里最像 P1+P3+P4 的**。通道只被描述为 "Hierarchical Graph Memory"，推断是符号/文字但**未核实** |
| 2 | **SPARC**（[2602.06566](https://arxiv.org/abs/2602.06566)）只到摘要 | "Separating Perception And Reasoning Circuits" 就是我们的 framing，2026 年的论文。未确认感知与推理是不是两个**模型**、接口是 latent 还是区域裁剪、有没有循环 |
| 3 | **全文检索做不到**：arXiv 搜索只匹配标题/摘要，API 从本机反复超时 | **MACF 的 `LatentMAS*` 那种"只出现在表体里的基线臂"抓不到**。已对 7 篇最高危读了全文 HTML 兜底，但残余风险集中在"摘要读起来像纯文字流水线"的 agentic 视频论文 |
| 4 | **具身/VLA/自动驾驶文献没搜** | 那些领域天天跑连续状态的闭环感知-动作。**如果审稿人来自机器人方向，这是最可能出意外先例的地方**——本次两个反例（NavGPT-2、PaLM-E）就都来自这里，说明这个担心是真的 |
| 5 | **C2C 附录 A.5.3**（agentic-flow 案例）只看到一行摘要 | 如果那个附录真跑了 KV 的多轮循环，我们的 P4 措辞得改成"带**视觉**发送方的 agentic 循环" |
| 6 | 8 月窗口基本空的（今天 2026-08-05），survey 截止 2026-07-15 | 有效覆盖到 7 月底。ICLR 2027 投稿前后的 preprint 一篇没出现 |
| 7 | **2606.09290 Visual Para-Thinker++**（"single-policy multi-agent framework for visual reasoning"，共享 visual prefix + KV 复用）只按摘要判为模型内 | 39 篇视觉 KV 扫描里**最像多智能体视觉 KV 的一篇**，值得单独读 |

---

## §11 一句话总结

**真空还在，但从"整个 P1+P2+P3+P4 组合没人做"缩小到"只有 (d) 接收方决定看什么 + (a) 发送方是 agent 这两个合取项没人占"；而 (d) 这一条已经有 NavGPT-2 Table 5 的一个负结果（冻结 LLM 自己决策 SR 67.52→21.46）。计划的判死点必须先撞这一条。**
