# 推翻我们两条主张的那两篇论文，到底做了什么

> **缘起**：[01-prior-art-audit.md](01-prior-art-audit.md) 判定我们准备写进 paper 的两条"没人做过"都被推翻了，推翻者是 **NavGPT-2（ECCV 2024）** 和 **Sterner et al.（剑桥 2024）**。审计文档只列了判决和数字，没讲这两篇**到底在做什么**。这份补上。
>
> **读者起点**：知道我们的 idea（viewer 看帧 → latent → 纯文本 reasoner → 循环取帧），知道什么是 LLM、什么是 KV cache。**其余术语（VLN / R2R / Q-Former / VQA / in-context learning / SR / OSR）全部从零讲。**
>
> 📌 **P1–P5 速查**（完整定义 + 判据 + 手把手示例见 [03 §0.5](03-eight-risk-papers-explained.md)）：**P1** 发送方看过像素 · **P2** 通道传连续向量而非文字 · **P3** 接收方是独立挑选、冻结、无视觉塔的纯文本 LLM · **P4** 多轮循环且**接收方决定 viewer 下一步看什么** · **P5** 有训出来的跨架构 adapter。**我们的 idea = 五条同时成立。**
>
> ⚠️ **P2 只管"是不是文字"，不管"是什么形式的向量"。** 载体形式（输入层伪 token vs 逐层 KV）是**另一条正交轴**，我们选的是**输入层伪 token**——跟 NavGPT-2 / Latent Bridge / BLIP-2 谱系**同款**。见 [03 §0.5](03-eight-risk-papers-explained.md)。
>
> **读法**：§A 讲 NavGPT-2（做机器人室内导航的）· §B 讲 Sterner（做看图问答的）· §C 两篇合起来对我们意味着什么 · §D 追问区 · §E 术语表。**§A 和 §B 互不依赖，可以分开读。**
>
> **数字来源**：两篇的 PDF 我都逐表读过，**原件已归档**（含 pypdf 抽好的 `.txt` 便于 grep）：
> - [2407.12366_NavGPT-2.pdf](../papers/pdfs/2407.12366_NavGPT-2.pdf)（26 页）
> - [2403.11317_Sterner_TaleOfTwoApproaches.pdf](../papers/pdfs/2403.11317_Sterner_TaleOfTwoApproaches.pdf)（7 页）
> - 目录说明：[papers/pdfs/README.md](../papers/pdfs/README.md)
>
> **本文所有数字都是我从原表抄的，减法都是我自己按的**；四处英文引句已在下载后的 PDF 上重新核过一遍。

---

## TL;DR

| | NavGPT-2 | Sterner et al. |
|---|---|---|
| **一句话** | 让**冻结的纯文本 LLM** 当室内导航机器人的"眼睛+嘴"，但**方向盘交给一个单独训练的小策略网络** | 严格对照：把图变成**文字**喂给纯文本 LLM，和把图变成**向量**喂给它，哪个好？ |
| **为什么威胁我们** | 它把 P1+P2+P3+P4 五条里的四条**字面上全占了**，而且用了**四个不同的冻结纯文本 LLM 来回换** | 它**就是**我们打算跑的那个对照实验，而且**对照做得比我们还严** |
| ⭐ **最要命的一个数** | 把策略网络拿掉、逼 LLM 自己决策 → 成功率 **67.52 → 21.46** | 0-shot：文字 **45.9** vs 向量 **41.4**，**文字赢 4.5 分**（p<0.01） |

---

# §A NavGPT-2：让 LLM 当导航员，但不让它开车

论文：[NavGPT-2: Unleashing Navigational Reasoning Capability for Large Vision-Language Models](https://arxiv.org/abs/2407.12366)，ECCV 2024，Adelaide / Adobe / UNC / UCSC。[代码](https://github.com/GengzeZhou/NavGPT-2)。

## A.1 先讲清任务：VLN 是什么

**VLN = Vision-and-Language Navigation，视觉语言导航。**

给机器人一句自然语言指令，让它在一栋**没见过的房子**里走到指定地点。用的数据集叫 **R2R（Room-to-Room）**，房子来自 Matterport3D——61 栋真实房屋的全景扫描。

**一条真实的指令长这样**（论文 Figure 1 的例子）：

> *"Walk towards the fireplace then turn right and proceed to the next room. Turn right at the dining room and enter the hallway. Wait near the phone."*
>
> （朝壁炉走，然后右转进下一个房间。在餐厅右转进走廊。在电话旁边等。）

**机器人不是自由移动的**，房子被建模成一张**图**（graph）：

```
     ●───────●           ● = 可以站的位置（节点）
     │       │           ─ = 可以走的通路（边）
     ●───────●───────●
             │
             ●   ← 机器人现在站这
```

**每一步机器人要做的事**：站在当前节点，往四周看，看到 N 个"下一步可以去的地方"（每个方向一张 RGB 图 + 一个角度），**从中选一个**。走过去，再看，再选。直到它决定"到了，停"。

**怎么算成功**：最后停下的位置离目标点**小于 3 米**就算成功。这个比例叫 **SR（Success Rate）**。

> 📌 **桥句（后文反复用）**：**导航的每一步，本质是一道"看图选择题"——N 个候选视角，选一个。**

## A.2 之前两条路都不行

论文开篇就把前人分成两派，各自有病：

| 路线 | 做法 | 病在哪 | 数字 |
|---|---|---|---|
| **零样本派**（NavGPT、MapGPT、DiscussNav） | 用 BLIP-2 之类的模型**把每张图翻译成一句话**，把历史也总结成文字，全部塞进 GPT-4 的 prompt，让它选方向 | 提示词工程又长又脆；每步都要重新塞一遍，越来越贵；**caption 和总结都在丢信息** | NavGPT(GPT-4) 在 val-unseen 上 SR 只有 **34**，而专门训练的 VLN 模型有 **72** —— **差约 40 分** |
| **微调派**（LangNav、NavCoT、NaviLLM） | 直接拿 VLN 数据微调一个 LLaMA-7B | 数据太少；而且**微调完 LLM 就不会说人话了**——变成只会吐动作的黑盒，引入 LLM 的意义被抹掉 | NaviLLM(Vicuna-7B) test SR **68**，仍低于最好的专门模型 |

> **NavGPT-2 的动机就是"我全都要"**：既要专门模型的成功率，又要 LLM 会解释自己在干什么。
>
> ⚠️ **注意零样本派的病灶——"caption 在丢信息"——正是我们项目的 motivation。** 这篇论文和我们的出发点是**同一个**。区别在于它们给出的解法不是"换 latent 通道"，而是"**别让 LLM 做决策**"。

## A.3 结构：四个角色，先认人再看图

| 角色 | 是谁 | 干什么 | 训不训 |
|---|---|---|---|
| 👁 **眼睛** | ViT-g/14（EVA-CLIP） | 把一张 RGB 图变成一堆视觉特征 | ❄️ **全程冻结** |
| 🔤 **翻译官** | **Q-Former**（32 个可学 query） | 把一张图的特征压缩成 **32 个向量**，并且**读过指令**再压（"instruction-aware"） | 🔥 **第一阶段训它** |
| 🧠 **解说员** | LLM（FlanT5-XL/XXL、Vicuna-7B/13B 四选一） | 读 32 个向量 + 指令，**说出**"我看到什么、我打算往哪走" | ❄️ **全程冻结** |
| 🚗 **司机** | 拓扑图策略网络（graph policy） | 真正**选**下一个节点 | 🔥 **第二阶段训它** |

### Q-Former 是什么（一句话）

**一个可学的"摘要器"**：它有 32 个可训练的向量（叫 query），这 32 个向量去 cross-attention 地"看" ViT 吐出来的那一大堆图像特征，把它们吸收进自己。输出就是 32 个 768 维向量。再过一个线性层投到 LLM 的词嵌入维度，**当成 32 个"词"塞进 LLM 的 prompt**。

来自 BLIP-2 / InstructBLIP。**它是把冻结视觉编码器接到冻结 LLM 上的标准零件。**

> ⚠️ 注意"**instruction-aware**"：这 32 个 query 会**先**和指令的文本嵌入做 self-attention，**然后**才去看图。所以同一张图，指令不同，压出来的 32 个向量就不同。**摘要是问题条件化的。**

## A.4 手把手：一步之内发生什么

假设指令是 *"Walk towards the fireplace then turn right"*，机器人站在客厅，周围有 **3 个**可去的方向。

### 第 ① 步：三张图各自过眼睛 + 翻译官

```
候选 1（正前方，30°）：壁炉那面墙的照片  → ViT → Q-Former → 32 个向量
候选 2（右侧，120°）：通向餐厅的门        → ViT → Q-Former → 32 个向量
候选 3（左后，250°）：来时的走廊          → ViT → Q-Former → 32 个向量
```

### 第 ② 步：拼成一段 prompt（论文 Figure 3 原文格式）

```
You are navigating in an indoor environment given the instruction:
<INST>Walk towards the fireplace then turn right...</INST>;
The navigable locations are listed below: {
  "Candidate 1, facing 30 degree, front" : <IMG>[32 个向量]</IMG>;
  "Candidate 2, facing 120 degree, right": <IMG>[32 个向量]</IMG>;
  "Candidate 3, facing 250 degree, left" : <IMG>[32 个向量]</IMG>;
};
Please choose the next direction.
```

> 📌 **这里就是"latent 通道"**：`<IMG>` 和 `</IMG>` 之间**不是文字**，是 32 个连续向量，直接占据 LLM 输入嵌入的位置。
>
> ⚠️ **而角度、方位（"front"/"right"）是文字**——它是个**混合通道**，视觉走 latent、结构走文字。（顺带印证了我们审计 §8 约束 2："系统应该设计成混合"。）

### 第 ③ 步：LLM 前向一遍，同时产出两样东西

这是最容易看混的地方，用"单步显微镜"拆开：

| | 产物 | 谁用 | 用途 |
|---|---|---|---|
| **③a 解码出来的文字** | *"I am currently positioned in a spacious indoor environment with a clear view of a fireplace at the front. To my left, there is a dining room… Based on the given instructions, I have already walked towards the fireplace. My next step is to turn right and proceed to the next room."* | **人看** | 可解释性、跟人对话 |
| ⭐ **③b 内部的 hidden state** | LLM 编码器最后一层里，那 32×3 个 image token 位置上的隐状态 `H′v`，以及指令 token 的隐状态 `H′l` | **司机（策略网络）** | **真正拿来决策** |

**③b 还要再压一次**：每个视角的 32 个隐状态 → 一个 MLP → **压成 1 个向量**。所以 3 个候选 = 3 个向量。

> ⚠️⚠️ **这是理解整篇论文（和 §A.6 那个负结果）的关键：**
>
> **LLM 说的那段话，对导航决策一点用都没有。** 它是纯粹的"解说词"。真正被下游消费的，是 LLM **内部**的表示。
>
> 换个比方：**LLM 是副驾上的解说员，一边看窗外一边描述路况；方向盘在司机手里。副驾说的话不接方向盘，但司机会读副驾的脑电波。**

### 第 ④ 步：司机选节点

策略网络维护一张**拓扑图**（走过的节点 + 相邻还没走的节点）：

- 每个节点的表示 = 从这个节点看到的各视角向量（③b 的产物）取平均 + 方向嵌入 + 步数嵌入
- 节点之间过 **GASA**（graph-aware self-attention）——注意力里加了一项**节点之间的 L2 距离矩阵**，让它懂空间关系
- 和指令做 cross-attention
- 每个节点打一个分，**选最高的**，走最短路过去（已访问过的节点分数被 mask 掉，鼓励探索）

> **为什么要拓扑图而不是让 LLM 记历史**：论文明说 LLM 对空间结构的理解和长程记忆能力不行。图记忆能让它**回溯**——走错了可以退回某个没探索过的岔路口。

### 第 ⑤ 步：走过去，回到 ①

## A.5 训练：两阶段，LLM 一次都没被训过

| 阶段 | 训谁 | 数据 | 目标 |
|---|---|---|---|
| **一** | 🔥 只训 **Q-Former + 投影层** | **10K 条**从 R2R 训练集抽的中间步，**用 GPT-4V 自动生成"导航推理"文字**（描述周围 + 说下一步往哪走） | 让冻结 LLM 会**说**导航推理。200K 步，batch 8 |
| **二** | 🔥 只训 **策略网络** | R2R + PREVALENT 合成数据，行为克隆 + DAgger | 让司机学会从 ③b 那些向量里选节点。batch 2 |

原文一句话钉死：

> **"all parameters of the vision encoder and LLMs are kept frozen during the entire training process."**

**全部实验在一张 A100 上跑完。**

## A.6 结果

### 主表（R2R，SR 越高越好）

| 方法 | LLM 冻结？ | Val-Unseen SR | Test-Unseen SR |
|---|---|---|---|
| 人类 | — | — | 86 |
| **DUET**（VLN 专门模型基线） | — | 72 | 69 |
| NavGPT (GPT-4)，零样本派 | ✓ | **34** | — |
| MapGPT (GPT-4)，零样本派 | ✓ | 39 | — |
| NaviLLM (Vicuna-7B)，微调派 | ✗ | 67 | 68 |
| **NavGPT-2 FlanT5-XL (1.5B)** | ✓ | 68 | — |
| ↳ w/ PREVALENT | ✓ | 70 | 71 |
| **NavGPT-2 FlanT5-XXL (5B)** | ✓ | 71 | — |
| ↳ w/ PREVALENT | ✓ | **74** | **72** |

自己按一遍：**72 − 68 = +4**（打赢微调派 NaviLLM）；**72 − 69 = +3**（打赢 VLN 专门模型 DUET）；**68 − 34 = 34 分**（碾压零样本派）。

### 顺带两个有用的数

- **数据效率**：NavGPT-2 只用 **50%** R2R 数据 → val-unseen SR **63.30**；DUET 用 **100%** → **63.90**。**用一半数据打平。**
- **人类评分**（30 个样本，10 个志愿者，满分 3）：NavGPT-2 的解说词得 1.66 / 1.93 / 1.78（准确性/信息量/合理性），GPT-4V 得 2.31 / 2.95 / 2.34。**不如 GPT-4V，但可用。**

### ⭐ Table 6：四个纯文本 LLM 来回换

| LLM | 参数 | Val-Unseen SR |
|---|---|---|
| FlanT5-XL | 3B | 67.52 |
| **FlanT5-XXL** | 11B | **71.31** |
| Vicuna-7B | 7B | 53.77 |
| Vicuna-13B | 13B | 48.28 |

> ⚠️ **这一格对我们最致命。** 它不是"用了一个纯文本 LLM"，而是**换了四个、全程冻结、还有排名分析**（FlanT5 系是 encoder-decoder，全注意力更适合对齐任务；Vicuna 是 decoder-only 因果注意力，反而更差，与模型大小无关）。
>
> **我们本来想拿"接收方是独立挑选的"当区分点——这条已经被占死了。**

## A.7 ⭐⭐ Table 5：把司机开除，会发生什么

这是子智能体没报、我自己读出来的一格，也是**对我们计划最重要的一个数**。

**他们做的事**：把 Q-Former 和策略网络里所有的 visual-language cross-attention 层都**删掉**，只留**一层图自注意力 + 一层前馈**来打分。用论文原话——

> **"By doing so, we force the LLM to take over the visual-textual based decision-making to exploit its navigational capability."**
>
> （这样做，我们**逼 LLM 自己接管**基于视觉-文本的决策。）

**结果**：

| Val-Unseen | TL（走了多远，米） | NE↓（终点离目标多远，米） | OSR↑ | **SR↑** | SPL↑ |
|---|---|---|---|---|---|
| 完整（司机在） | 13.68 | 3.37 | 74.37 | **67.52** | 56.01 |
| **w/o policy model** | **26.70** | **8.03** | 69.60 | **21.46** | 10.23 |

**逐个按一遍**：

```
SR   67.52 → 21.46   = −46.06   ← 成功率崩了三分之二
TL   13.68 → 26.70   = ×1.95    ← 路程几乎翻倍
NE    3.37 →  8.03   = ×2.38    ← 最后停的地方离目标远了 2.4 倍
OSR  74.37 → 69.60   = −4.77    ← ⭐ 这个几乎没掉
```

论文的结论一句话：**"a frozen LLM is incapable of inferring effective representations that indicate a correct action."**（冻结 LLM 推不出能指示正确动作的有效表示。）

### ⭐ OSR 那一格值得单独讲

**OSR（Oracle Success Rate）= "假设它有一个完美的停止策略"时的成功率**——换句话说，**它这一路上有没有经过目标点 3 米以内**。

- OSR 69.60，SR 21.46 → **它有 69.6% 的概率走到过目标附近，但只有 21.46% 的时候停在那。**
- 但它同时**走了两倍远**（26.70m vs 13.68m）。

**诚实的读法**：这两件事纠缠在一起——瞎走得更久，自然更容易路过目标。所以不能简单说"它知道往哪走、只是不知道在哪停"。**能说的是：把决策权交给冻结 LLM 之后，它的行为退化成"到处乱逛"，而不是"完全瘫痪"。**

> ⚠️⚠️ **这为什么正打在我们心脏上**
>
> 我们的 **P4** 就是"**让冻结的纯文本 reasoner 从 latent 视觉状态里，决定下一步看什么**"。NavGPT-2 测的是最接近的版本——"让冻结纯文本 LLM 从 latent 视觉状态里，决定下一步**走哪**"——**结论是不行**。
>
> **我们能保留的差别有三个，写 paper 时必须逐条讲清**：
>
> | # | 差别 | 为什么可能救我们 |
> |---|---|---|
> | (a) | 他们要的输出是**离散动作**（选第几个候选节点）；我们要的是**语言化的取帧请求**（"我需要看清水槽附近发生了什么"） | 后者恰好是 LLM 的**强项**——它本来就是生成语言的。前者要求它在隐空间里编码一个精确的离散决策，那是策略网络擅长的事 |
> | (b) | ⚠️ ~~他们是输入层 soft token，我们传逐层 KV~~ **作废（2026-08-05）**：我们的载体**也是**输入层伪 token（[00-plan §5.3](00-plan.md)），KV 只是 adapter 的读料。**这是相同点，不是差别。** 真差别只剩：他们的 adapter 读 **ViT 特征**，我们读 **前 4~10 层 KV** —— 属机制细节，不承担新颖性。详见 [03 §0.5](03-eight-risk-papers-explained.md) | ❌ 不成立 |
> | (c) | 他们**从没训过**"让 LLM 学会读这种 latent 然后发指令"这件事——Q-Former 只在阶段一用**解说文本**监督训过 | 我们的 adapter 会**专门为这件事训**（审计 §8 约束 4） |
>
> **但这不是可以绕过去的细节，是必须先撞的判死点。** 已写进 [00-plan.md §3.0 的 D0](00-plan.md)：**如果我们的 reasoner 在 latent 上的取帧决策明显差于它在 caption 上的取帧决策，整条路就是 NavGPT-2 Table 5 的重演。**

---

# §B Sterner et al.：一场"文字 vs 向量"的干净对决

论文：[Few-Shot VQA with Frozen LLMs: A Tale of Two Approaches](https://arxiv.org/abs/2403.11317)，Igor Sterner / Weizhe Lin / Jinghong Chen / Bill Byrne，剑桥大学工程系，2024-03-17。

**这是一篇短论文（arXiv-only，未见正式发表），但它存在的唯一目的就是跑我们计划要跑的那个对照。**

## B.1 先讲清任务：VQA 是什么

**VQA = Visual Question Answering，看图问答。** 给一张图 + 一个关于图的问题，输出答案。论文 Figure 1 的三个例子：

| 图 | 问题 | 答案 |
|---|---|---|
| 一头大象背上坐着人 | *How many people are riding on the elephant?* | `2` |
| 一辆公交车 | *What color is the bus?* | `blue and white` |
| 一列火车 | *Does this train look to be in good working order?* | `no` |

数据集是 **VQAv2**，验证集 **214K** 条〈图，问题，答案〉。判分用官方指标：每题有 **10 个人类标注的答案**，模型的答案**至少和其中 3 个完全一致**才得分。

## B.2 把图喂给纯文本 LLM，历来有两条路

**大前提**：LLM（这里是 Flan-T5 XL，3B）和视觉模型（CLIP ViT-G）是**分开训练的**，两边的表示互不相通。要让 LLM "看见"图，得搭个桥。历来两条路：

```
路 A（embedding-based / 向量路）
   图 ──CLIP──► 特征 ──mapping net──► k 个向量 ──┐
                                                ├─► LLM ──► 答案
   问题 ──────────────────► 词嵌入 ─────────────┘

路 B（caption-based / 文字路）
   图 ──CLIP──► 特征 ──mapping net──► k 个向量 ──► LLM ──► "一辆蓝白色的公交车停在街边"
                                                                    │
   问题 ────────────────────────────────────────────────────────────┤
                                                                    ▼
                                                                   LLM ──► 答案
```

**路 A** 的代表：Frozen（2021）、LiMBeR（2023）、BLIP-2、MAGMA。
**路 B** 的代表：PICa、Img2LLM 那一系——先 caption，再当成普通的阅读理解题。

论文抓住的漏洞（原文）：

> *"The above systems begin by learning an image captioning task. It is therefore a natural baseline to compare their embedding-based VQA approaches with that of using image captions generated by the same system. **However, none of the aforementioned systems report such results.**"*

**翻译**：这些"向量路"的系统，桥本来就是**用图像描述任务训出来的**——那你顺手让它生成一句 caption、走文字路对比一下，是最自然不过的基线。**但没有一个系统报过这个数。**

## B.3 ⭐ 这篇的关键设计：两条路共用一切

这是整篇论文最漂亮的地方，也是它**比我们计划要跑的对照更严**的原因。

**两条路的前半段完全相同**：

```
raw pixels
   ↓ 冻结 CLIP ViT-G
1280 维特征
   ↓ 单隐层 mapping network（1280 → 隐层 20×2048/2 → 20 个 2048 维向量）
20 个向量  ← ⭐ 到这里两条路一模一样
```

这个 mapping network 是**唯一被训练的东西**，训法：在 Conceptual Captions（2.7M 图文对）上做图像描述，**梯度穿过冻结的 Flan-T5 反传**回来。两个 epoch，batch 32，AdamW。

**分叉只在后半段**：

| | 路 A（向量） | 路 B（文字） |
|---|---|---|
| 做法 | 20 个向量 + 问题的词嵌入 → **拼起来** → LLM → 答案 | **先只把 20 个向量喂给 LLM**，让它生成一句 caption → 把这句**文字** + 问题 → **再喂一次同一个 LLM** → 答案 |

原文：

> *"**The only difference is that in caption-based VQA, the embeddings are first passed to the LLM alone to generate a caption**, before being concatenated with the question."*

> ⭐⭐ **为什么这个设计比我们的严**
>
> 我们计划里的文字臂，caption 是 **viewer 模型**写的，latent 是 **viewer 模型**产的——两边至少还差一层"谁写的"。
>
> 而这里，**caption 是同一个冻结 LLM，从同一份 latent，用同一套权重解出来的**。所以：
>
> **文字臂拿到的信息，在数学上必然是向量臂拿到的信息的一个子集**（它是从那份向量里"读"出来的）。
>
> **按理说路 A 不可能输。** 论文自己也这么想——所以他们把结果称为 *"a surprising result, because **the same visual representation is used in both approaches**"*。

## B.4 还有一个变量：few-shot 怎么选例子

**In-context learning（上下文学习）**：在 prompt 里先塞几道**做好的例题**（图/caption + 问题 + 正确答案），再问真正要答的题。塞几道就叫"几 shot"。

**例题从哪来是个选择**。他们比了三种（用 CLIP 嵌入 + 内积相似度，FAISS 检索）：

| 记号 | 选例方式 |
|---|---|
| **R** | 随机选 |
| **Q** | 只按**问题**相似度选 |
| **Q+I** | 按**问题 50% + 图像 50%** 的联合相似度选 |

> 📌 0-shot 时没有例题，所以三种记号在 0-shot 那一列是**同一个数**。

## B.5 结果（Table 1，VQAv2）

| 通道 | 选例 | 0-shot | 1 | 2 | 4 |
|---|---|---|---|---|---|
| *参照：Frozen (2021)* | | *29.5* | *35.7* | *–* | *38.2* |
| *参照：Linear mapping (LiMBeR)* | | *33.3* | *39.9* | *40.8* | *40.3* |
| *参照：MAGMA* | | *36.9* | *42.2* | *43.8* | *45.4* |
| **文字路** | R | **45.9** | 45.3 | 45.1 | 45.3 |
| **文字路** | Q | **45.9** | **47.6** | **48.0** | **48.5** |
| **文字路** | Q+I | **45.9** | 50.0 | 50.3 | 50.8 |
| **向量路** | R | 41.4 | 44.5 | 43.6 | 42.5 |
| **向量路** | Q | 41.4 | 40.6 | 45.7 | 47.3 |
| **向量路** | Q+I | 41.4 | **50.5** | **51.8** | **52.4** |

**先做健全性检查**：他们两条路的最好成绩（45.9→50.8 和 41.4→52.4）在**所有 shot 数上都超过三个参照系统**。所以实验设置本身没问题。

**然后逐格做减法（向量 − 文字）**：

```
0-shot（三种选例等价）：   41.4 − 45.9 = −4.5    ← 文字赢，p<0.01
R  选例，1/2/4 shot：      −0.8 / −1.5 / −2.8     ← 文字全赢，而且越来越赢
Q  选例，1/2/4 shot：      −7.0 / −2.3 / −1.2     ← 文字全赢
Q+I 选例，1/2/4 shot：     +0.5 / +1.5 / +1.6     ← 向量只在这一行赢，且幅度小
```

**12 个可比的格子里，向量路只赢了 3 个，而且都赢得很小。**

论文结论：

> *"connecting visual embeddings directly to the LLM embedding space **does not necessarily improve performance** compared to using image captions. We find that **the selection of relevant in-context examples is far more important**."*

## B.6 ⭐ 三条线索指向一个机制（论文没明说，我拼的）

论文只报现象没给机制，但表里有三条线索可以拼起来：

**线索 1——向量路对"示范"的依赖强得多**：

```
0 → 1 shot（Q+I 选例）：
   文字路：45.9 → 50.0   = +4.1
   向量路：41.4 → 50.5   = +9.1   ← 跳幅是文字路的 2.2 倍
```

**线索 2——选例方式决定谁赢**：只按**问题**相似度选例（Q），文字路**全程赢**；一旦例题也按**图像**相似度选（Q+I），向量路**全程赢**。

**线索 3——细粒度分析（论文正文，4-shot）**：向量路在**需要看清细节**的题上更强——**颜色 +4.2%、计数 +2.7%**；在**简单 yes/no** 题上略输 **−0.5%**；在"why"/"where"这类两边都弱的题上没差别。

> ⭐ **拼起来的读法（我的推断，非论文原话）**：
>
> **向量确实带了更多细节信息**（颜色、计数这种"caption 一句话装不下"的东西，向量路赢）。**但 LLM 不会自动知道该拿这 20 个向量干什么。** 它需要 in-context 例题来"解锁"——而且例题得是**图像也相似**的，才教得会。
>
> **文字则自带使用说明书**：LLM 一辈子都在读文字，一句 caption 进来它立刻知道怎么用。所以文字路 0-shot 就有 45.9，而且对例题不敏感。
>
> **一句话：latent 的问题不是"装得少"，是"接收方不会用"。**

这条推断对我们**直接有用**：它说明 [审计 §8 约束 4](01-prior-art-audit.md)（adapter 必须用文字监督引导）不是可选项——**"教会接收方怎么用这堆向量"本身就是主要工作量**。

### 论文自己承认的局限（我们可以用）

> *"Our findings are applicable for the **Flan-T5 XL** LLM, which is **not as large as other closed-source LLMs**. Our tale may have been different for these models. In addition, we learn the mapping between image and text space using a **non-linear network**, rather than other methods such as that presented by Li et al. (2023) [BLIP-2 的 Q-Former]."*

**两个口子**：(a) 接收方只有 3B，换大的可能不一样；(b) 桥只是个单隐层 MLP，不是 Q-Former 那种强桥。**这是我们能站的地方，但只能站这么大——不能说"没人做过"。**

---

# §C 两篇合起来，对我们意味着什么

## C.1 各自杀掉我们哪条主张

| 我们的主张 | 谁杀的 | 怎么杀的 |
|---|---|---|
| "没人让像素发送方把 latent 交给独立挑选的冻结纯文本 LLM，在多轮 agentic 循环里用" | **NavGPT-2** | 字面上全中：冻结 ViT → 32 个 Q-Former token → **四个冻结纯文本 LLM 互换** → 每步循环 |
| "没有多模态论文跑过 caption vs latent 其余全固定 + 纯文本接收方" | **Sterner et al.** | 整篇论文就是这个 arm，**对照比我们还严**，而且**文字赢 4.5 分** |

## C.2 但它们各自也留了口子

| 口子 | NavGPT-2 | Sterner |
|---|---|---|
| **发送方不是 agent** | ViT+Q-Former 是**接收方自己的感知前端**，没有独立推理回路 | 同上（CLIP + MLP） |
| **接收方不决定看什么** | 下一个视角由**策略网络**选，LLM 只解说 | 完全单次前向，没有循环 |
| **通道是输入层 soft token** | 32 个/视图 | 20 个/图 |
| **没为"读 latent 然后发指令"训过** | Q-Former 只用解说文本训 | mapping net 只用 caption 训 |

**这四条的交集，就是我们剩下的全部空间。**（→ [审计 §4](01-prior-art-audit.md) 的四合取项表述。）

## C.3 ⭐ 三条会改我们设计的东西

**1. NavGPT-2 Table 5 = 判死点 D0。** 我们必须先证明"冻结纯文本 reasoner 能从 latent 里发出好的取帧指令"，再谈别的。这个实验很便宜（不用训 adapter，先用现成的 soft-prompt 版本就能试）。

**2. Sterner 的细粒度结论和我们自己的 C1 同向。** 他发现 latent 在**颜色/计数**这类细节题上赢、在 **yes/no** 上输。我们的 C1 发现：Qwen3-VL 在 **GQA**（大量 yes/no 和简单计数感知题）上，多轮讨论臂掉 **−17.6**；而在 **MMStar**（更需要综合推理）上打平甚至微赢。

> 📌 **两处独立证据指向同一件事：通道的价值取决于题型。简单感知题上，多绕一道（不管是多轮讨论还是换通道）都容易帮倒忙。**
>
> → **我们的评测必须按题型分层报**，不能只报总分。否则一个 −17.6 就能把整张表拖垮，而且看不出原因。

**3. Sterner 的"latent 需要示范才会用"= 我们的 adapter 训练方案的直接依据。** 审计 §8 约束 4 说"第一阶段损失就该是从 latent 解出 caption"——Sterner 的三条线索给了这条一个**机制性的解释**，而不只是"MACF 消融说它重要"。

---

## C.4 ⭐⭐ 两篇合起来指向同一个病：**不是看不懂，是不会用**（2026-08-05 追问回填）

> **追问原话**："这两篇论文似乎都可能指向一个致命缺陷：文本 LLM 不知道怎么用 latent —— 或者说看不懂。"

**判断是对的，但"看不懂"和"不会用"是两个不同的病，而且这两篇正好给了能把它们分开的证据。**

### C.4.1 先把三种可能分开

| # | 病 | 意思 | 药方 |
|---|---|---|---|
| ① | **通道没传到** | 信息压根不在那些向量里 | 换通道设计 |
| ② | **看不懂** | 信息在，但接收方解不出来 | 修桥 |
| ③ | **不会用** | 解得出来，但不知道该拿它干什么 | 训用法 |

### C.4.2 两篇都给了"看得懂"的正面证据

**Sterner 三条**：

1. 它**能从那 20 个向量里写出 caption**，且拿这句 caption 答题得 **45.9 分** → 内容确实被读出来了
2. 4-shot 时向量臂在**颜色 +4.2%、计数 +2.7%** 上赢文字臂 → **读不出来的东西赢不了**
3. **一道例题补回 +9.1**（41.4 → 50.5）→ **不存在的信息，示范补不回来**

**NavGPT-2 两条**：

1. 人类评分：冻结 LLM 从 32 个向量生成的场景描述，准确性 **1.66/3**、信息量 **1.93/3**（GPT-4V 直接看真图是 2.31/2.95）→ **它对着一堆向量把房间里有什么描述对了**
2. 完整系统 SR **74**，超过 VLN 专门模型 → 视觉信息显然流过去了

> ⚠️ **一个诚实的打折**：NavGPT-2 的 Q-Former 就是专门为"让冻结 LLM 说出好的导航推理"训的，所以"它能描述"有循环论证成分。**但这恰好印证另一半：为某个用法训过就能读，没训过的用法就抓瞎。**

### C.4.3 ⭐ 我上次在审计里把 NavGPT-2 Table 5 说重了，收回一点

[01-prior-art-audit.md §2.1](01-prior-art-audit.md) 我写的是"NavGPT-2 已经测过最接近的版本，结论是不行"。**过头了。** 重读消融原文：

> *"we remove all the visual-language cross-attention layers in the Q-former and policy network and use only **a single graph-aware self-attention layer followed by a single feed-forward layer** to predict the action"*

它测的是：**能不能用一个几乎线性的读出头，从冻结 LLM 的隐状态里解码出正确动作。** 答案是不能。

**它没测的是：让冻结 LLM 用语言说出该走哪个候选。** 完整系统里 LLM 本来就在说 *"My next step is to turn right and proceed to the next room"*，但**这段话没接方向盘**。

⚠️ **不过"它看懂了也说对了"这个说法要打折**（这是我第一版的过度表述，收回）：

| 人类评分（30 样本 / 10 人 / 满分 3） | NavGPT-2 | GPT-4V（直接看真图） |
|---|---|---|
| Accuracy（描述准不准） | **1.66 = 55%** | 2.31 = 77% |
| Informativeness（信息全不全） | 1.93 = 64% | 2.95 = 98% |
| **Rationality**（论文定义：*"the correctness of the **action planned**"*） | **1.78 = 59%** | 2.34 = 78% |

**1.66/3 是"读懂个大概"，不是"读懂了"**，比直接看真图差 22 个百分点。**Rationality 1.78 更是直接给"LLM 语言级动作决策对不对"打的分——59%。**

### C.4.3.1 ⭐ 三角定位：那个格子在两个方向上都是空的

| 谁做决策 | 看到什么 | Val-Unseen SR |
|---|---|---|
| **LLM 用语言** | caption 文字（GPT-4） | **34**（NavGPT）/ 39（MapGPT）/ **43**（DiscussNav，零样本最好） |
| ⭐ **LLM 用语言** | **latent** | **❓ 没人测过** ← **我们的格子** |
| 浅层读出头 | LLM 隐状态（源自 latent） | **21.46**（Table 5） |
| **策略网络** | LLM 隐状态（源自 latent） | **67.52 ~ 74** |

**两头都要看**：

- 往下：把决策交给冻结 LLM 的**隐状态** → **21.46**，已发表负结果。
- 往上：把决策交给 GPT-4 的**语言 + caption** → **最好只有 43**，而专门模型 72。→ ⚠️ **在这个任务上语言级决策本身就弱，跟通道无关。**

> 📌 **所以不能说"系统没利用上，用上就行"。** 上面那一格已经被测过了，只有 43。我们的格子真空，但**两个邻居是 43 和 21.46——先验不乐观**。

### C.4.3.2 但 VLN 的决策和我们的决策，难在完全不同的地方

| | NavGPT-2 要 LLM 做的 | 我们要 reasoner 做的 |
|---|---|---|
| 决策内容 | "去 Candidate 2"——从 N 个候选里选一个**几何位置** | "我想看清 340–420 帧发生了什么"——一句**语义检索请求** |
| 需要的能力 | 空间结构理解、长程记忆、回溯 | 语言表达、知道自己缺什么 |
| LLM 天生 | ❌ 弱（论文明说这正是它引入策略网络的原因） | ✅ 强 |
| 还要多做一步 | 把 "turn right and proceed to the next room" 映射回"第 2 个候选"，且无图记忆不能回溯 | 无——请求本身就是最终产物 |

**而且我们有正面证据**：VideoAgent 的循环**已经用文字跑通了**——**72.2%** 的题至少触发一次重新取帧、**73.3%** 的自评说"当前报告不够用"，而且循环是涨分的。**"纯文本 LLM 能不能用语言指挥取帧"，VideoAgent 已经证明能，只是它用的是 caption。**

📌 **准确表述**：NavGPT-2 Table 5 杀掉的是"**从冻结 LLM 隐状态浅层读出离散动作**"，那不是我们要做的事；它的人类评分**也救不了我们**（1.66/3 说明读得不够准，且语言路的端到端分数无人测量）。**这个格子既没被证伪，也没被支持——它是空的，而且先验不乐观。**

### C.4.4 ②「看不懂」不是不存在——取决于桥好不好

| 桥的质量 | 结果 |
|---|---|
| **好桥**（Sterner：2.7M 图文对、梯度穿过冻结 LLM 反传） | 看得懂，病在 ③ |
| **烂桥**（ViF 的 mean-pooling） | **真·看不懂**——5 个库里输给纯文字 4 个（MMHal 42.5 vs 47.9、HallBench 46.7 vs 53.1） |
| **没桥**（免训练直接塞） | Vision Wormhole 预言 generation collapse；我们的漂移探针（[00-plan.md §4.1](00-plan.md)，第 0 层误差 31.9%）指向同一处 |

> **一句话：桥不好 → ②；桥好了 → ③。两关都要过，顺序不能反。**

### C.4.5 ⭐ 一个不用训 adapter 的判别实验

```
探针 1（会不会读）：让 reasoner 只凭 latent 复述内容
                    "这几帧里有什么？谁在做什么？"
   → 复述得对  = 看得懂，病在 ③ → 继续
   → 复述不出  = ② 看不懂 → 先修桥，别谈循环

探针 2（示范敏感度）：同一批题，0-shot vs 给 1~2 道 in-context 示范
   → 示范能补上大半（Sterner 那种 +9.1）= 信息在，只是不会用 → 可训练解决
   → 示范补不动                          = ① 信息真没传到 → 止损
```

**Sterner 已经把探针 2 替我们跑过一遍，答案是"示范能补"。我们只需在自己的模型对上确认。**

### C.4.6 对计划的两处修改

**(1) 判死点 D0 从二元判据拆成三级**（[00-plan.md §3.0](00-plan.md) 已同步）：

| 级 | 问题 | 不过怎么办 |
|---|---|---|
| **D0-a** | reasoner 能凭 latent 复述帧内容吗？ | 不能 → ② → 桥设计错了，回 §5 |
| **D0-b** | 给示范后，latent 臂追上 caption 臂了吗？ | 追不上 → ① → 止损 |
| **D0-c** | gap 随 adapter 训练单调缩小吗？ | 不缩小 → 训练目标选错 → 止损 |

**a 和 b 完全不用训 adapter。**

**(2) adapter 训练课程必须两段。** Sterner 的桥**只训过 captioning**，然后在 VQA 上输了 → **纯重建/对齐损失（MACF Stage 1）只治 ②，不治 ③**。第二段必须直接拿"取帧决策对不对"当信号（MACF Stage 2/3）。

⚠️ **代价要认**：如果"用法"必须逐接收方训，"**接收方是独立挑选的现成模型**"这个卖点会进一步缩水——每换一个 reasoner 就得重训。**这正是 NavGPT-2 做的（每个 LLM 一个 Q-Former）。** 论文里只能写"adapter 很小、训一次几 GPU 小时"，**不能写"即插即用"**。

### C.4.7 自洽性交叉验证：我们自己的 C1 同向

如果病因是"**跨模型边界**时接收方不会用"，那么发送方与接收方是**同一个模型**时这病就不该出现。C1 正是那个设定：

| | Single | **Latent (m=0)** | 差 |
|---|---|---|---|
| LLaVA-OV / GQA | 59.4 | 58.8 | **−0.6** |
| Qwen3-VL / MMStar | 61.8 | 62.2 | **+0.4** |
| LLaVA-OV / MMStar | 53.9 | 50.1 | −3.8 |
| Qwen3-VL / GQA | 63.2 | 55.5 | −7.8 |

**同模型时 KV 通道基本无损（两格 ≈ 0）** → 与"病在跨边界的用法、不在通道本身"自洽。掉分的两格是那个模型在该库上的弱项，不是通道问题（详见 [phase2-results.md](../VMAS-latent-collab/phase2-results.md) `C1-rev`）。

---

# §D 追问区（预判的坑）

**Q1：NavGPT-2 的 LLM 说的那段话，真的一点都不用来决策吗？**
**是的。** 决策链是：LLM 内部隐状态 `H′v` → MLP 压成每视角 1 个向量 → 拓扑图策略网络打分 → 选最高分节点。**解码出来的文字不在这条链上。** 它的作用是可解释性和人机交互（论文 Figure 1 右半边全是"用户中途干预/纠错/求助/具身问答"的例子）。

**Q2：那 Table 5 到底改了什么？为什么说是"逼 LLM 自己决策"？**
他们把 Q-Former 和策略网络里所有 visual-language cross-attention 层删掉，只留**一层图自注意力 + 一层前馈**。也就是说，**下游几乎不再做任何加工**，能不能选对全靠 LLM 内部那些隐状态本身**已经**编码了正确动作。结果是不能。

**Q3：Sterner 的两条路，为什么文字路要跑两遍 LLM？那不是更慢吗？**
是更慢（多一次生成 caption 的解码）。但这篇论文**不关心效率，只关心准确率**。⚠️ **这反而是我们的机会**：我们的效率论证（省 N−1 次视觉编码、零中间 token）在这篇里根本没被测过。**但也别高兴太早——我们的准确率论证被它打了。**

**Q4："20 个向量" vs "32 个向量"，这个数重要吗？**
它是**信息带宽**的上限。20×2048 和 32×768 都是几万个浮点数，比一句 7 词的 caption 大好几个量级。**所以"latent 输给 caption"绝不是带宽问题**——这正是两篇论文都觉得意外的原因，也是 [审计 §9](01-prior-art-audit.md) 里"别把 latent 更保真当前提"的根据。

**Q5：SR 和 OSR 差在哪？为什么 Table 5 里一个崩了一个没崩？**
SR = 最终停下的位置离目标 < 3 米的比例。OSR = **假设有一个完美的停止策略**时的成功率，也就是"这条轨迹上有没有任何一点离目标 < 3 米"。Table 5 里 OSR 只掉 4.77 而 SR 掉 46.06，**但它同时走了两倍远**——走得久自然更容易路过目标。所以不能推出"它知道往哪走只是不知道在哪停"，只能说"**它退化成到处乱逛，而不是完全瘫痪**"。

**Q6：Sterner 是 arXiv-only，能算数吗？**
**能，而且必须当数。** 审稿人不会因为它没发表就不认；相反，"你们声称第一个做这个对照，但 2024 年就有人做过且结论相反"是最难回答的意见之一。**我们唯一的正确做法是主动引用它、承认它、说清我们的设定差在哪。**

**Q7：那我们是不是该放弃？**
不是。**结论从"填一个空格"变成"在一个有负结果的方向上做出正结果"。** 这更难写，但也更有价值——前提是我们**先撞 D0**，撞不过就止损。这就是 [00-plan.md](00-plan.md) 把判死点排在所有开发之前的原因。

---

# §E 术语表

| 词 | 意思 |
|---|---|
| **VLN** | Vision-and-Language Navigation，按自然语言指令在陌生房子里导航 |
| **R2R** | Room-to-Room，最常用的 VLN 数据集，房子来自 Matterport3D 的 61 个真实场景 |
| **Val Seen / Val Unseen** | 验证集分两半：**训练时见过的房子** / **完全没见过的房子**。后者才是真本事 |
| **SR** | Success Rate，最终停下的位置离目标 < 3 米的比例 |
| **OSR** | Oracle SR，假设有完美停止策略时的成功率 = "路上有没有经过目标附近" |
| **NE** | Navigation Error，终点离目标的平均距离（米），越低越好 |
| **TL** | Trajectory Length，平均走了多远（米） |
| **SPL** | 用路径长度惩罚过的成功率——绕远路会扣分 |
| **DUET** | 一个 VLN 专门模型（0.18B），本文的主要对照基线 |
| **Q-Former** | BLIP-2 提出的"摘要器"：一组可学 query 向量去 cross-attention 地吸收视觉特征，输出固定条数的向量。InstructBLIP 版还会先读指令（instruction-aware） |
| **InstructBLIP** | BLIP-2 的指令微调版，NavGPT-2 的起点 |
| **FlanT5 / Vicuna** | 两类纯文本 LLM。FlanT5 是 encoder-decoder（全注意力），Vicuna 是 decoder-only（因果注意力） |
| **soft token / soft prompt** | 直接塞进 LLM 输入嵌入层的连续向量，不对应任何真实词 |
| **VQA / VQAv2** | 看图问答 / 最常用的数据集，验证集 214K 条 |
| **in-context learning / shot** | 在 prompt 里塞几道做好的例题再问正题；塞几道叫几 shot |
| **DAgger** | 一种模仿学习训练法：让模型自己走，然后用"从当前位置到目标的最短路"当伪标签纠正它 |
| **GASA** | Graph-Aware Self-Attention，注意力分数里额外加了一项节点间 L2 距离矩阵 |
| **P1–P5** | 我们把自己 idea 拆成的五个可判定性质，定义见 [01-prior-art-audit.md §1](01-prior-art-audit.md) |

---

## 相关文档

- [01-prior-art-audit.md](01-prior-art-audit.md) —— 完整先例审计（这两篇的判决出处，加另外十几篇）
- [00-plan.md](00-plan.md) —— 路线计划，§3 的诊断体系源自本文两篇的判决
- [../Q&A/06-latent-collab-design-space-and-brain-tool-edge.md](../Q&A/06-latent-collab-design-space-and-brain-tool-edge.md) —— latent 协作的设计空间与六条正交轴
