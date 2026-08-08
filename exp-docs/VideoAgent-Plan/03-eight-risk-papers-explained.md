# 审计 §5 / §5.5 那八篇"危险论文"，分别做了什么

> **缘起**：[01-prior-art-audit.md](01-prior-art-audit.md) 的 §5 和 §5.5 用表格列了八篇新发现的威胁论文，只给了判决和一句话概括。这份把每篇**从零讲清楚它做了什么**，格式同 [02-navgpt2-and-sterner-explained.md](02-navgpt2-and-sterner-explained.md)。
>
> **读者起点**：知道我们的 idea（viewer 看帧 → latent → 纯文本 reasoner → 循环取帧）。**P1–P5 五个代号在 §0.5 完整定义，其余术语从零讲。**
>
> **读法**：八篇分成三组——**A 组「有人做了很像的东西」**（Latent Bridge / Mem-W / SportMV-Agent）· **B 组「有人解决了我们以为要自己解决的机制」**（ChronoStitch / LinguDistill）· **C 组「有人在质疑我们的立论本身」**（MemDreamer / 两篇怀疑派 / SPARC）。**每组独立，可跳读。**
>
> ⚠️ **这一版我把八篇的 PDF 全下载下来逐一读过了**，PDF + 抽好的 `.txt` 归档在 [papers/pdfs/](../papers/pdfs/README.md)。**结果发现审计表里有两处事实错误**（子智能体报告未经二次核对造成），已在 §0 列出并已回改审计文档。

---

## §0.5 术语先行：P1–P5 是什么

> 本文和 [01](01-prior-art-audit.md) / [02](02-navgpt2-and-sterner-explained.md) 通篇用这五个代号。**这里是完整定义，读这三份文档任何一份都不用回头翻。**

我们把自己的 idea 拆成**五个能逐条打勾的性质**：

| 代号 | 一句话 | 具体到我们的系统 | 怎么对一篇论文判 yes/no |
|---|---|---|---|
| **P1** | **发送方看过像素** | viewer（Qwen3-VL）真把视频帧过了视觉编码器 | 有没有真图/真视频进模型？还是纯文字任务？ |
| **P2** | **通道传连续向量，不是文字** | 传 hidden state / KV cache / **伪 token**，**不生成 caption**。⚠️ **形式不限，判据只有"经没经过文字离散化"**（见下方「两条正交轴」） | 两个模块之间那条边上流的是浮点数组还是字符串？ |
| **P3** | **接收方是独立挑选的、冻结的纯文本推理 LLM** | reasoner（Qwen3-8B-Thinking）**没有视觉塔**、**没跟 viewer 共训**、**forward 图未被改动** | 接收方有视觉编码器吗？被共训/微调过吗？有没有往它内部插层？ |
| **P4** | **多轮循环，且接收方决定 viewer 下一步看什么** | reasoner 说"我要看 340–420 帧，高分辨率"，viewer 照办 | 有没有一条从接收方**指回**发送方的反向边？ |
| **P5** | **有一个训出来的跨架构 adapter** | 两边表示空间不通，得训个翻译器 | 中间那座桥是训出来的，还是免训练/根本不存在？ |

**我们的 idea = P1+P2+P3+P4+P5 同时成立。**

### 为什么非拆这么细

**因为每一条单独看都早有人做过，只有全部同时成立那一格才可能是空的**：

```
P1+P2+P3      → BLIP-2 / Frozen / LiMBeR 那一套，2021 年就是标准做法
P2+P5         → Cache-to-Cache 一系，2026 年已是成熟工程
P1+P2+P4+P5   → Mem-W（本文 §A.2，五占四）
P1+P3+P4      → SeeingEye / BeMyEyes（但通道是文字，P2 反）
```

**每去掉一个合取项，就立刻冒出一篇已发表的反例。这五条是被反例一条条逼出来的，不是随便列的。**

### 手把手：拿两篇走一遍

| | **Mem-W**（§A.2） | **NavGPT-2**（见 [02](02-navgpt2-and-sterner-explained.md)） |
|---|---|---|
| **P1** | ✅ 真在看屏幕截图 | ✅ 冻结 ViT-g/14 编码真实 RGB 视角 |
| **P2** | ✅ 记忆 token 是连续向量 | ✅ 32 个 Q-Former query 向量 |
| **P3** | ❌ **写和读是同一个冻结 VLM**，无模型边界 | ✅ **四个纯文本 LLM 互换、全程冻结**（这格它比我们还狠） |
| **P4** | ✅ 多轮 GUI 导航 + 主动调取记忆 | ⚠️ 有多轮循环，**但下一个视点是策略网络选的，不是 LLM 选的** |
| **P5** | ✅ 训了轨迹→latent 压缩器 | ✅ 训了 Q-Former + 投影层 |
| **结论** | **五占四**，差的 P3 正是我们的全部差别 | **只有 P4 的后半句（"接收方决定看什么"）救了我们** |

---


### ⚠️ P2 之外还有两条正交轴（2026-08-05 补，原先被我混进 P2 了）

**P2 只管"是不是文字"，不管"是什么形式的向量"。** 后者是两条独立的设计选择：

| 轴 | 问什么 | 我们计划的选择（[00-plan §5.2–§5.4](00-plan.md)） |
|---|---|---|
| **载体形式** | 以什么形式、注入接收方哪一层 | ⭐ **输入层伪 token**（m≈32），**不是**逐层 KV |
| **读料来源** | 从发送方**取什么**当 adapter 的输入 | ⚠️ **待扫描**（原"前 4~10 层 KV"的 LACO 依据被识别为错置——那是载体结论非读料结论；证据偏**中层**，且 h 信息上不亏于 KV。详见 [00-plan §5.3](00-plan.md)） |

按这两条轴重排先例：

| 论文 | 读料 | 载体 | 跟我们 |
|---|---|---|---|
| **Latent Bridge**（§A.1） | 第 24 层 residual | 8 个伪 token @ 输入层 | ⚠️ **几乎同款** |
| **NavGPT-2**（[02](02-navgpt2-and-sterner-explained.md)） | ViT 特征 | 32 个 Q-Former token @ 输入层 | ⚠️ **几乎同款** |
| **Mem-W**（§A.2） | 轨迹 | K 个记忆 token @ 输入层 | ⚠️ 同款 |
| MACF | 末层 hidden | 32 token @ 共享空间 | ⚠️ 同款 |
| BLIP-2 / Frozen / LiMBeR 谱系 | ViT 特征 | soft prompt @ 输入层 | ⚠️ 同款 |
| **LinguDistill**（§B.2） | **逐层 KV** | **逐层 KV** | ⭐ **形式跟我们不同的是它** |
| LatentMAS / LACO / C2C | 逐层 KV | 逐层 KV | 形式不同 |

⚠️⚠️ **两个必须认的后果**：

1. **载体形式承担不了任何新颖性**——输入层伪 token 是这个领域最主流、验证最多的做法。凡是把"我们传 KV、他们传 embedding"写成差别的地方**全部作废**（本文初版、[01 §2.1(b)](01-prior-art-audit.md)、[02 §A.7(b)](02-navgpt2-and-sterner-explained.md) 都犯过，已改）。
2. ⚠️ **我们不再填 [Q&A/06 §4.2](../Q&A/06-latent-collab-design-space-and-brain-tool-edge.md) 那个"最大空格"**——那格写的是"没有任何论文**传过** VLM 的逐层 KV"。**我们读 KV 但不传 KV，所以这个卖点得撤。**

**唯一还剩一点特色的是读料轴**：多数先例读末层 hidden 或 ViT 特征，我们读**逐层 KV**。这是真差别，**但属于机制细节，不是新颖性**。

### ⭐ 由此引出一个几乎零成本的判死实验：反解探针（D0-d）

**伪 token 可以用接收方的词嵌入矩阵反解回 token**（`argmax_v cos(e, E[v])`，或过 `lm_head` 看词表分布）。这给了一个直接的证伪实验：

```
同一份伪 token 走三臂喂回 reasoner（2026-08-05 由两臂升级，源自用户提问「e·Eᵀ 得到的非 one-hot 软分布有意义吗」）：

hard:  E[argmax(e·Eᵀ)]         最近词，one-hot
soft:  softmax(e·Eᵀ/τ)·E       投回词嵌入凸包 = CIPHER 式软叠加
raw:   e                        原伪 token

raw ≈ soft ≈ hard   →  文字就装得下 → 通道优势是假的 → 止损
raw ≈ soft ≫ hard   →  信息在「叠加权重」里 → 优势真，⚠️ 但 CIPHER 级简单通道可能就够
                        ——审稿人会问"要 21M adapter 干嘛"，半好半坏
raw ≫ soft          →  信息在词嵌入凸包之外 → 「超语言」的最强证据
```

📌 **几何澄清**：词表 151936 ≫ 维度 4096，**E 的行张满整个空间**——任何向量都在"张成空间内"，`e·Eᵀ` 永远给得出分数。所以 off-manifold ≠ 出span，而是**范数异常/离所有具体词嵌入都远/分数谱平坦**；有信息量的正是**软分布的形状（熵）**。soft 臂就是 CIPHER 的通信载体（不采样、发 softmax 加权的嵌入和）——**非 one-hot 不是 bug，是一种已发表的通信格式**。

**比 Sterner 的对照还干净**：他是"同一份 latent，一条解成 caption 一条直接喂"；这个是"**同一份伪 token，一条投到最近邻词一条不投**"，中间连生成过程都没有，**唯一变量就是离散化本身**。

⚠️ **三个坑**：(a) 反解有损且会误导——一个向量可以是多个词嵌入的加权和（CIPHER 一系正是靠这个），**"人眼看不通"不能当结论，必须靠喂回去的分数判**；(b) 先查 Qwen3 的 `embed_tokens` 与 `lm_head` 是否 tied，**输入嵌入和输出嵌入几何不同**；(c) ⚠️ **别把"靠近词表流形"当训练目标**——靠得越近越接近 caption、信息量越少。**这个 trade-off 本身值得画一张图**（横轴反解距离 / 纵轴任务分数）。

**顺带一个便宜的健康指标**：反解的最近邻**余弦相似度**。太低说明伪 token 落在词嵌入流形之外——正是 Vision Wormhole 那段 off-manifold 警告说的情形（它自己的解法里有个 **NormMatch** 组件专门把伪 token 拉回嵌入范数流形）。

> ⭐⭐ **2026-08-05 升级（源自用户追问）**：soft 那臂不只是探针，**可以直接当 adapter 的输出格式**——让 bridge 天生只能输出 `softmax(h·Eᵀ/τ)·E`（E=接收方冻结词嵌入，零新增参数，端到端可训）= **trained cross-modal CIPHER**（格式同 CIPHER，但 p 由训过的 adapter 从视觉 latent 算出、落在**接收方**词表上，跨模型）。
> **关键区别**：D0-d 的事后投影是这个格式的**下界**（训练后量化）；公平比较要**穿过投影训练**的臂（量化感知训练）。→ 实验矩阵改为**两个训练臂**（软词瓶颈 vs 自由向量）+ 事后投影当诊断。**A≈B 选软词（可解释性白送、off-manifold 异议按构造消失）；A≫B 则差值=超语言视觉信息的实测大小，本身是主图。双赢。** 已进 [00-plan §5.4](00-plan.md)。

---

## §0 ⚠️ 先纠错：审计表里两条经不起核对

| 出错处 | 审计原文 | **PDF 实际** |
|---|---|---|
| **Mem-W** | 「原文写死 "**no separate text-only LLM is involved**"」 | ❌ **这句话不存在。** 全文 `text-only` 出现 **0 次**，`no separate` **0 次**，`K=8` **0 次**。→ **这是子智能体编的引句。** 结论（P3 挂）仍然成立，但理由要换成"读和写是同一个冻结 VLM"，不能引这句 |
| **SportMV-Agent** | 「**GPT-4.1 纯文本 orchestrator**；68.31% vs GPT-4.1 自己 59.68%」 | ❌ **两处都错。** (1) 数字应为 **68.35 vs 59.12**（Table 2）。(2) **GPT-4.1 在这篇里是当多模态模型用的**——它出现在 Table 2 的 MLLM 基线里直接看视频，且原文写 *"whenever a view switch occurs, **the video segment from the newly selected view is integrated into the agent's observation context**"*。**它不是纯文本 orchestrator，P3 不成立** |

> ⭐ **SportMV-Agent 那条纠错对我们是好消息**：它**没有**占住 P1+P3+P4 那个角。占那个角的仍然只有 SeeingEye / BeMyEyes（图 → 文字翻译器 → 纯文本 reasoner）。

**教训**（已同步 [claim-evidence-discipline]）：**子智能体给的"原文引句"必须回原文核**。八篇里核出两篇有问题，命中率 25%。

---

# A 组：有人做了很像的东西

## A.1 The Latent Bridge —— 唯一一篇真跑了我们那套三臂消融的论文

**论文**：[The Latent Bridge: A Continuous Slow–Fast Channel for Real-Time Game Agents](https://arxiv.org/abs/2606.24470)（Pine AI + UW，2026-06-23）· [本地 PDF](../papers/pdfs/2606.24470_LatentBridge.pdf) · [有代码](https://github.com/19PINE-AI/latent-bridge-games)

### 它要解决什么

**打电子游戏时，"反应快"和"想得深"是矛盾的。**

```
Atari 游戏 15 Hz 控制 → 每 67 毫秒必须出一个动作
一个会推理的 VLM（Qwen3-VL-8B-Thinking）想一次要 1.5 秒 = 晚了 22 帧
一个反应快的 VLM（MiniCPM-o 4.5，9B）33 毫秒就能出动作，但不会规划
```

**它的方案**：两个模型都冻住，一个当"快脑"（15 Hz 打游戏），一个当"慢脑"（1 Hz 在后台想）。**唯一可训练的东西就是它们之间的通道。**

### ⭐ 三条臂，正是我们计划的 F/T/L 结构

| 臂 | 快脑收到什么 |
|---|---|
| **F（Fast-Only）** | 什么都不给，自己打 |
| **T（Text Bridge）** | 慢脑写的一段文字（中位数 302 字符），当 prompt 后缀贴上去 |
| **L（Latent Bridge）** | 慢脑第 24 层最后 8 个位置的 residual → 一个 **33M 参数的 MLP** → 投到快脑 4096 维输入嵌入空间 → **prepend 8 个 latent token** |

**两个 backbone 都冻结、规模相当（9B 快 / 8B 慢），所以通道是唯一变量。** 这就是我们要做的对照。

### ⭐⭐ 三个必须知道的结果

**① 审计说它"在每个域都赢"——不准确。真实情况：**

| 游戏 | F | T | L | L vs T |
|---|---|---|---|---|
| MsPacman | 273 | 401 | **628** | **+57%** (p=0.01) ✅ |
| RoadRunner | 0 | 475 | **608** | **+28%** (p=0.00) ✅ |
| River Raid | **994** | 639 | 566 | −11% (p=0.35) 平 |
| Seaquest | 57 | **143** | 125 | −13% 平 |
| Q\*bert | 65 | **185** | 146 | −21% 平 |
| Enduro | 4 | 3 | 2 | −33% 平 |
| SpaceInvaders | 135 | **162** | 142 | −12% 平 |

**7 个游戏里显著赢 2 个、平 5 个、从不显著输。** 论文自己的措辞是 *"never significantly worse ... a safe-or-better drop-in"*——**是"不输"，不是"每个域都赢"。**

⚠️ **而且 River Raid 和 SpaceInvaders 上，Fast-Only 打赢了两条桥**——原文：*"whether the slow–fast architecture beats Fast-Only at all is a separate, game-dependent question."* **接了慢脑反而更差。**

**② ⭐⭐ 作者明确拒绝"文字带宽不够"这个故事**

> *"We frame it this way rather than as a **'text is bandwidth-limited' story, which our own ablations do not support**."*

他们的解释是另一个：**桥有没有用，是任务的属性，不是通道的属性。** L−F 和 T−F 的相关系数 **r = 0.93**（8 个域）——**慢推理本来就有用的地方，桥才有用；慢推理没用的地方，换什么通道都白搭。**

MetaDrive 是他们特意设的阴性对照：把训好的 latent 换成**全零或随机向量**，分数不变（L ≈ L_zero ≈ L_random ≈ F）→ **证明桥在那里是惰性的**。

> ⭐ **这个"桥替换对照"（zero / random）我们必须抄。** 它比 CAG 更便宜，能一眼看出 latent 到底有没有在起作用，还是只是个 prompt 扰动。

**③ 两条通道一起用会互相打架**

同时给文字后缀 + latent token：**从不超过更好的那条单通道，在 3 个游戏上显著干扰（RoadRunner −96%）**。原因：动作头是在"一次只有一种条件信号"上训的。

> ⚠️ **这跟审计 §8 约束 2（HyLaT 说"要保留一条细文字通道当语义锚"）直接冲突。**
> **怎么调和**：Latent Bridge 的冲突是**训练造成的**（动作头没见过双通道），不是原理性的。**但它提醒我们：如果做混合通道，两条通道必须在训练时就一起出现，不能训完单通道再拼。**

### ⭐ 一个白送的架构教训（对我们 §5.2 的注入点选择）

> *"A first attempt (**v1, 256-d cross-attention at 2 of 36 layers**) converged to **KL=0.004 offline yet failed at deployment**; the working **v2 is the LLaVA pattern**—project slow residuals into the fast LLM's 4096-d input-embedding space and prepend, so all layers attend via standard causal attention."*

**翻译**：他们先试了"在 36 层里挑 2 层做 cross-attention"，**离线指标好得不得了（KL=0.004），一上线就崩**。换成"投到输入嵌入层、prepend"才work。

> ⭐⭐ **这是对我们 [00-plan.md §5.2](00-plan.md) 选 (iii) 输入层注入的第三方实验背书**，而且顺带给了一个警告：**离线重建指标好 ≠ 部署能用。** 我们的验收必须有端到端臂，不能只看 KL / cos。

### 对我们的净影响

| | |
|---|---|
| **占了什么** | P1（Qwen3-VL 编码游戏画面）+ P2（连续 residual）+ P5（训好的 33M bridge），**以及我们的三臂消融结构** |
| **没占什么** | **只有 P3 和 P4**。P3 挂：接收方 MiniCPM-o 4.5 是 9B VLM，有自己的视觉塔。P4 挂：实时反应式控制，快脑从不回头指挥慢脑。<br>⚠️ **本文初版写的"通道是 embedding 级不是逐层 KV"是错的**——我们的载体也是输入层伪 token，**这是相同点不是差别**（见 §0.5） |
| **必须引** | 是。而且不能装作它没做过 F/T/L 消融 |
| ⭐ **反而帮我们的** | 它自己说 bandwidth story 不成立 → **我们的 motivation 不能写"文字带宽不够"**，只能写"文字丢的是**问题相关的细节**"（这是不同的主张，见 §C.1 ChronoStitch 那句话） |

---

## A.2 Mem-W —— 五条占四条，但差的那条正好是我们的全部

**论文**：[Mem-W: Latent Memory-Native GUI Agents](https://arxiv.org/abs/2605.09317)（NUS & NTU，2026-05-10）· [本地 PDF](../papers/pdfs/2605.09317_Mem-W.pdf)

### 它要解决什么

**GUI agent**（自动操作网页/手机 App 的 agent）干长任务时的记忆问题：

```
任务："在购物网站上找到评分 4 星以上、价格低于 200 的蓝牙耳机，加入购物车"
→ 要走十几个屏幕
→ "4 星以上、200 以下"这个约束是十屏之前说的
→ 但当前屏幕上没有它
```

**现有做法**：把历史**总结成文字或结构化记录**存起来，要用的时候再塞回 prompt，让模型**重新编码一遍**。

**Mem-W 的批评**：*"This creates a mismatch between the representational form in which experience is stored and the latent embedding sequence over which modern GUI policies actually act."* —— **存的形式和用的形式不一致，中间白白转了两道。**

### 它怎么做

```
过去的成功/失败轨迹（4,972 条记忆库）──┐
                                      ├─→ 共享的「轨迹→latent 压缩器」C_ϕ ─→ K 个记忆 token
本次会话里已经过期的片段 ─────────────┘                                      │
                                                                            ▼
                                              和当前屏幕观测 + 局部上下文「编织」成一条连续嵌入序列
                                                                            ▼
                                                              ❄️ 冻结的 GUI agent policy → 动作
```

**训练**：GUI agent **全程冻结**，只训压缩器 `C_ϕ`。两阶段——**Stage I 自蒸馏**（学会把轨迹压成 latent）+ **Stage II RLOO 强化学习**（用"任务最后成没成功"这个二元奖励，把记忆筛向真正有用的证据）。

### 结果（我从 Table 2 逐格核的）

| Backbone | AC-v2-High Pass@1 Acc | 提升 |
|---|---|---|
| Qwen3-VL-4B | 49.30 | — |
| **↳ Mem-W-4B** | **63.07** | **+13.77** |
| UI-Venus-1.5-8B | 61.06 | — |
| **↳ Mem-W-8B** | **68.59** | **+7.53** |

⚠️ 摘要称"gains of up to **+30.0**"，我在 Table 2 里核到的最大单格是 **+13.77**。那个 +30.0 应在另一张表（Mind2Web / MMINA），**我没逐格核，引用前要查**。

### 对我们的净影响

| | |
|---|---|
| **占了什么** | **P1+P2+P4+P5 = 五占四**。这是所有先例里占得最多的一篇。而且"**给看像素的 agent 做 latent 记忆**"这个 framing 被它拿走了 |
| **差在哪** | ⭐ **P3**：**写记忆和读记忆是同一个冻结 VLM**（Qwen3-VL-4B / UI-Venus-1.5-8B）。**没有跨模型边界，所以它根本不需要 adapter 去翻译，也不会遇到"接收方看不懂"的问题** |
| ⚠️ **纠错** | 审计引的那句 "no separate text-only LLM is involved" **在原文中不存在**（§0）。结论不变，但理由要改成上面这条 |
| **我们的差别** | **就是那条模型边界，仅此而已。** 这话必须在 related work 里自己说，不能等审稿人说 |
| ⭐ **可借鉴的** | **Stage II 用任务成败当奖励去筛记忆** —— 这正是 [02 §C.4.6](02-navgpt2-and-sterner-explained.md) 说的"第二段训练必须拿最终任务当信号"，Mem-W 给了一个可抄的具体做法（RLOO + 二元成败奖励） |

---

## A.3 SportMV-Agent —— 我们的 workflow 槽位被占了，但没被占死

**论文**：[Beyond the Single Camera: Agentic Multi-View Reasoning in Sports Video Understanding](https://arxiv.org/abs/2607.11844)（浙大 + MSRA，2026-07）· [本地 PDF](../papers/pdfs/2607.11844_SportMV-Agent.pdf)

### 它要解决什么

**体育视频里，一个机位经常看不清。**

```
(a) View 1 和 View 2 都被挡住 → 判"7 号出界"（错）
    View 3 角度好          → 真相是"23 号出界"
(b) View 1 显示防守方在禁区内接触了进攻方
    View 2 显示防守方没碰到球
    → 只有合起来看，才判得出：进攻方犯规，不是点球
```

现实里裁判就是靠 VAR 多机位。但**没有任何 benchmark 测过 MLLM 的多机位理解**——所以他们先做了 **SportMV-Bench**（1022 个多机位视频包、3015 道题、10 种运动、三个难度层）。

### ⭐ 它先做了一组很有用的诊断

在提出方法之前，他们先查"MLLM 到底卡在哪"：

| 加什么 | 效果 |
|---|---|
| **CoT（思维链）** | ❌ **反而掉分**（Qwen3-VL-30B）。原文：*"sequential reasoning without reliable visual grounding **amplifies hallucination** rather than improving decision quality"* |
| **体育领域知识**（GPT-5 生成的判罚要点） | ❌ 提升可忽略 → **知识不是瓶颈** |
| **细粒度感知**（直接给标注好的动作和接触） | ✅ **提升巨大** → **感知才是瓶颈** |

> 📌 **这和我们 C1 的发现同向**：多轮讨论/CoT 在感知题上会帮倒忙（我们 Qwen3-VL/GQA 掉 17.6）。**这是第三份独立证据。**

### 它的循环

```
while 轮次 ≤ 上限:
    orchestrator 读【问题 + 当前激活的视角 + 已积累的证据】
    → 生成一段"思考"H_t + 一个操作 o_t
    → 如果 H_t 的自信度够 → 直接出答案，结束
    → 否则二选一：
        · ViewSelect：换机位（新机位的视频段直接进 agent 的观测上下文）
        · ToolExec：调感知工具（动作识别 / 接触检测 / 接触部位检测）
    → 证据累积，回到开头
```

**感知工具返回的是「带置信度的排序列表」这类结构化字符串。**

### 结果

| 模型 | PAR | REI | ADR | **总体** |
|---|---|---|---|---|
| Qwen2.5-VL-72B | 45.31 | 53.99 | 35.75 | 45.34 |
| GPT-4o | 52.25 | 59.55 | 41.80 | 52.31 |
| **GPT-4.1（最强基线）** | 58.23 | 65.99 | 49.09 | **59.12** |
| **SportMV-Agent** | 68.57 | 72.56 | 62.22 | **68.35** |

**+9.23 绝对分 / +15.61% 相对。**

### ⭐ 消融里有一条对我们极有用

| 换什么 | 变化 |
|---|---|
| **orchestrator** 从 GPT-4.1 换成 GPT-4o | **−10.99%** |
| **感知工具**从 Qwen 换成 GPT-4.1 | **+0.41%**（几乎没用） |
| 关掉主动换机位 | −7.72% |

> ⭐⭐ **"reasoner 比 perceiver 重要得多"的第三份独立证据**（前两份：VideoAgent +11.4 vs +0.6；MemDreamer 10.0~12.1 vs 0.4~2.5）。**我们计划 §1 "为什么坚持文本 reasoner"这一条，现在有三份互相独立的支撑。**

### 对我们的净影响

| | |
|---|---|
| **占了什么** | **P1+P4**（真的在迭代主动选视角）。**我们的 workflow 形状被占了** |
| ⚠️ **纠错** | 审计说它是 **P1+P3+P4**——**P3 不成立**。GPT-4.1 在这里是当多模态模型用的，**换机位后新视频段直接进它的观测上下文**。所以它**不是**"纯文本 reasoner + 感知工具"，而是"**多模态 reasoner + 感知工具**" |
| **好消息** | → **P1+P3+P4 那个角仍然只被 SeeingEye / BeMyEyes 占着**，而它们是图像任务不是视频 |
| **怎么用** | 当**强文字/结构化通道基线**，原样跑，不重写 |

---

# B 组：有人解决了我们以为要自己解决的机制

## B.1 ChronoStitch —— 把独立缓存的视觉 KV 拼起来，免训练

**论文**：[ChronoStitch: Training-Free Composition of Visual KV Memories for Long-Horizon Temporal Reasoning](https://arxiv.org/abs/2607.19547)（KGraph AI，印度班加罗尔，2026-07-21，6 页短文）· [本地 PDF](../papers/pdfs/2607.19547_ChronoStitch.pdf)

### ⭐ 它开篇那句话，就是我们的 motivation

> *"**Text captions are useful but lossy: a caption written before the question is asked may omit precisely the detail needed later.** Storing the model's KV cache offers a richer alternative because it preserves internal visual activations rather than a fixed natural-language summary."*

**这正是我们计划 §3 的 A′ 臂（先知 caption 上界）在论证的东西**——只不过人家已经把它写成论文的第一段了。

### 它要解决什么

长视频问答，你想把视频**切成块、各自缓存 KV**，用的时候取出来拼。**但拼不起来**：

```
每一块都是从「本块的位置 0」开始 prefill 的
→ 直接首尾相接，两块的时间相位撞在一起
→ "谁先发生""发生了几次""什么变了"这类问题全崩
```

**为什么不能简单地把位置号往后挪**：Qwen2.5-VL 这类模型的视觉 token 用的是**三轴 mRoPE（时间 t / 高 h / 宽 w）**。如果你把视觉 token 压成一维连续位置号，**同一帧里不同位置的 patch 会拿到不同的"时间"坐标** —— 帧内的空间顺序被错当成时间先后。

### 它怎么做（两步）

1. **三轴重定基**：把存好的**旋转后的 key** 直接再转一个角度，从"本块坐标"搬到"全局视频坐标"。⭐ **不需要恢复原始未旋转的 key，也不用重训。**
2. **选择性重算**：位置修好了还不够——后面的块在原本编码时**从没注意过前面的块**。所以挑出"受影响最大"的那部分 token（用一个无 oracle 的代理：浅层重算一次，看哪些 token 的第 1 层 key 移动最大），**只把这批 token 重新过全部层**，让它们注意到拼好的完整 cache。

### ⭐⭐ 结果里有一条对我们是当头一棒

**Table 1（key 重建保真度，相对全量联合 prefill）**：

| 层 | 朴素拼接 相对 MSE | 一维标量重定基 | **三轴重定基** |
|---|---|---|---|
| 0 | 9.7e-3 | 1.3e-2 | **3.6e-8** ← 机器精度 |
| 9 | 6.6e-1 | 8.0e-1 | 2.1e-2 |
| 35 | 3.6e-1 | 4.3e-1 | 8.4e-2 |

**Table 2（受控的顺序探针：把 A–B 和 B–A 两种拼接顺序都问一遍"哪个先"，正确的记忆应该换顺序就换答案）**：

| 策略 | 准确率 | 6 对里翻转了几对 |
|---|---|---|
| 全量联合 prefill（上界） | 100.0% | 6/6 |
| 朴素拼接 | 50.0% | 0/6 |
| 一维标量重定基 | 58.3% | 1/6 |
| ⚠️ **只做三轴重定基** | **41.7%** | **0/6** |
| 重算 ρ=0.15 | 66.7% | — |
| **重算 ρ=0.35** | **100.0%** | **6/6** |

> ⚠️⚠️ **看第 4 行：位置修到机器精度（3.6e-8），顺序探针反而从 50% 掉到 41.7%，比什么都不修还差。**
>
> 论文的结论一句话：**"positions are necessary, but not sufficient."**

**Table 3（TempCompass 时间子集，N=590）**：联合 prefill 上界 63.9%，ChronoStitch **54.1%**，只做三轴 49.8%，一维 49.5%，朴素 49.3%。→ **ChronoStitch 比朴素 +4.8，事件排序上 +7.0**，但**离上界还差 9.8 分**。

**Table 4（效率）**：2411ms → **748ms = 3.26×**，重算约占拼好 cache 的 **25%**。

### 对我们的净影响

| | |
|---|---|
| **它占了什么** | **mRoPE 三轴重定基这个机制**——这是我们跨比较表里列为障碍 (a) 的东西。**不能当我们的贡献了**，只能引为"模型内先例，我们跨边界推广" |
| ⭐⭐ **它推翻了我们的一个假设** | [00-plan.md 旧版 §2.1（现 §4.2 已修订）](00-plan.md) 说"位置能算准（③是确定性修正，**不用学**）"，并把它当成已解决。**ChronoStitch 证明：即使在同一个模型内、位置修到机器精度，内容仍然对不上，还需要重算 25~35% 的 token。** 我们是**跨模型**，只会更糟 |
| **要改什么** | 计划 §5.1 的三件事（形状/位置/内容）里，**"位置"那一栏不能再标 ✅ 免费**。必须补一句：**位置修正是必要非充分，后面还要接内容修复** |
| **可借鉴的** | 它那个**无 oracle 的选点代理**（浅层重算一次，看第 1 层 key 移动多少）——比我们 attn-guided 那套便宜，值得对比 |

---

## B.2 LinguDistill —— 唯一真把逐层 KV 交给"纯文本 LM"的论文，但方向完全不同

**论文**：[LINGUDISTILL: Recovering Linguistic Ability in Vision-Language Models via Selective Cross-Modal Distillation](https://arxiv.org/abs/2604.00829)（MBZUAI，2026-04）· [本地 PDF](../papers/pdfs/2604.00829_LinguDistill.pdf)

### 它要解决的问题，跟我们完全不是一回事

**现象**：拿一个纯文本 LM 去改造成 VLM，**它的语言能力会掉**。

论文实测（nanoVLM 原版 vs 多模态微调 4000 步后）：

| 基准 | 原版 | 微调后 | 变化 |
|---|---|---|---|
| COCO captioning | 0.800 | 0.673 | **−15.9%** |
| MME Cognition | 302 | 229 | **−24.2%** |
| HellaSwag | 0.405 | 0.326 | **−19.5%** |
| DocVQA（视觉题） | 0.769 | 0.767 | −0.1% |
| MMStar（视觉题） | 0.360 | 0.359 | −0.3% |

**掉的全是语言题，视觉题没掉。** 而且论文说，**再拿语言数据微调也补不回来**。

### ⭐ 它的做法：让"没被改造过的原版 LM"当老师

```
学生 = nanoVLM-460M-8k（已经被改造成 VLM 的那个）
老师 = SmolLM-360M-Instruct  ← ⭐ 就是这个 VLM 改造之前的语言主干本身，冻结

问题：老师是纯文本的，看不见图，怎么给"看图条件下"的监督？
答案：⭐ 逐层 KV cache 共享 —— 老师直接复用学生的 KV，
      于是老师"看到"了学生的多模态上下文，可以给出 vision-aware 的监督信号

训完：老师丢掉。推理时就是一个普通 VLM，零额外参数、零额外开销。
```

**为什么 KV 能直接共享**：原文一句话说白了——*"This is **straightforward since the teacher and student share the same LM architecture**."* **老师就是学生的前身，形状和语义天然兼容。**

### 结果

| | nanoVLM-full（多模态微调） | **LinguDistill** |
|---|---|---|
| 语言题平均 | 0.471 | **0.564（+19.7%）** |
| 文档/OCR 平均 | 0.592 | 0.557（−5.9%） |
| MME 平均 | 628.0 | 736.5（+17.3%） |

**语言能力捞回来一大截，视觉能力小幅让步。**

### 对我们的净影响

| | |
|---|---|
| ⚠️ **必须引** | 它确实是"**逐层 KV 从一个看过像素的模型 → 一个冻结的纯文本 LM**"。**我们不能说"没人这么做过"** |
| **但差得远** | ① **老师是学生自己的前身**（同架构同血统），不是独立挑选的模型 —— 这比 NavGPT-2 的"独立性"弱得多；② **KV 路径只在训练时存在**，推理时整条路被丢掉；③ 冻结 LM 是**蒸馏老师**，它的输出是 next-token 分布用来当监督，**不是一个会做决策的 reasoner**；④ **没有循环** |
| ⭐ **反过来是正面证据** | **一个冻结的纯文本 LM，靠着共享 KV，真的能产出有用的 vision-aware 监督信号。** 如果它完全"看不懂"这些 KV，整个蒸馏就不可能work。→ 这支持 [02 §C.4](02-navgpt2-and-sterner-explained.md) 的判断：**病在"不会用"，不在"看不懂"**——**前提是两边血统足够近** |
| **给我们的警告** | 它的可行性**建立在"同架构"上**。我们要跨 Qwen3-VL ↔ Qwen3，血统近但不同（漂移 15–27%），**中间那道坎正好落在它绕开的地方** |
| ⭐ **载体轴上它反而是异类** | 八篇里**只有它真的以"逐层 KV"当载体**；我们和其余七篇一样走**输入层伪 token**（§0.5）。所以"我们传 KV"这个说法从来不成立 |

---

# C 组：有人在质疑我们的立论本身

## C.1 MemDreamer —— 纯文字通道打赢了"把像素全喂进去"

**论文**：[MemDreamer: Decoupling Perception and Reasoning for Long Video Understanding](https://arxiv.org/abs/2606.07512)（2026-06-24 v2）· [本地 PDF](../papers/pdfs/2606.07512_MemDreamer.pdf)

**判决**：**不是先例**（P2 挂），**但它抽掉了我们的安全前提。**

### 它做了什么

把长视频理解拆成两段，中间**只用文字**：

> *"a perception model P first processes the visual inputs in a streaming fashion to construct a structured, **purely textual** Hierarchical Graph Memory"*
>
> *"**During inference, R operates only on this pure text memory, never raw video information.**"*

而且它**主动拒绝回看原始帧，并把这当卖点**：

> *"Unlike previous leading methods that **require additional perception models to revisit raw video frames during retrieval**, our approach relies solely on interactions with text memory."*

→ **我们的 P4（接收方让 viewer 重新感知）是它故意不做的。** 全文 `latent`/`hidden state`/`KV` 出现 **0 次**。

### ⚠️ 但它的数对我们的 motivation 极不利

| 角色 | 模型 | 上下文 | LVBench |
|---|---|---|---|
| 端到端（整段视频喂进去） | Gemini-3.1-Pro | 265K | 78.2 |
| 端到端 | Gemini-2.5-Pro | 784K | 72.0 |
| 端到端 | Qwen3-VL-235B-A22B | 240K | 63.6 |
| **MemDreamer 推理侧** | Gemini-3.1-Pro | **6.2K** | **90.7（+12.5）** |
| MemDreamer 推理侧 | Gemini-2.5-Pro | 6.3K | 80.7（+8.7） |
| MemDreamer 推理侧 | Qwen3-VL-235B | 5.9K | **84.8（+21.2）** |

**一个纯文字通道，在小时级视频上，比"把像素全喂进去"高 8.7 ~ 21.2 分，上下文还小 41~124 倍。**

> ⭐ **我的判读（不要照抄论文口径）**：那个 +12.5 **混淆了两件事**——(i) **通道**（文字 vs 像素）和 (ii) **上下文预算与结构**（6K 结构化 vs 240K 平铺）。端到端臂是"整段视频一次性平铺进 240–784K 窗口"，MemDreamer 感知侧是"10 分钟滑窗、流式逐段"。**两臂的视觉吞吐量和上下文结构都不同——这正是我们自己的门槛 E4(d)（帧数/视野混淆）。所以它不是干净的通道对照。**
>
> ⭐ **而我们计划 §3 的 C 臂恰好是那个干净版本**：给 reasoner 的是**同一批被选中帧**的像素，文字臂是**同一批帧**的 caption。**MemDreamer 没跑这个对照 → 没有抢先我们的实验。**
>
> **但它确实抽掉了"文字有损，所以要换 latent"这个安全前提。** 修订后的 motivation 只能是：**"把上下文预算控住之后，文字丢的那部分还剩多少"**——这正是 C−D 要量的东西。判死点不变，**先验往不利方向移了**。

### ✅ 同一篇的另一半却强力支持我们

| 换什么 | 幅度 |
|---|---|
| 换**感知**模型（reasoner 固定） | **0.4 / 1.4 / 2.5 pp** |
| 换**推理**模型（perception 固定） | **10.0 / 12.1 pp** |

**reasoner 比 perceiver 重要约 4~30 倍。** 另外 Table 5：端到端时"模型的 AIME2025 推理能力"与 LVBench 成绩相关性只有 **R=0.702, p=0.052（不显著）**；接上 MemDreamer 后 **R=0.897, p<0.01**。原文：*"raw video ingestion creates **perceptual barriers** that hinder models from leveraging their reasoning capacity."*

---

## C.2 两篇怀疑派 —— 不是先例，是评测设计约束

### C.2.1 因果审计：**总准确率根本识别不出 latent 消息干了什么**

**论文**：[Do Latent Channels Actually Communicate? A Causal Audit of Latent Multi-Agent LLM Communication](https://arxiv.org/abs/2607.26773)（Georgia Tech + Memorial University，AAAI'27 投稿格式）· [本地 PDF](../papers/pdfs/2607.26773_CausalAudit.pdf)

**它的核心质问**：你报了一个准确率差，怎么知道这个差是"接收方真的用了发送方关于**这道题**的信息"，而不是——

- 消息**存在**这件事本身（多了一段前缀，扰动了分布）
- 通信过程带来的**额外计算**
- 上下文复用
- 冗余的推理轨迹

**它的做法**：在"发送方表示进入接收方"的那个边界上做**受控的消息替换**——四种消息设定（空消息 / **别的题的消息** / 自己生成的 / 当前题的），支持五种测量。

**结果（对我们最要命的那一格）**：

```
GSM8K / Qwen3-4B   总效应 −1.00pp
                    = −6.17（换成"别的题的消息"仍然保留的部分）
                    + +5.17（真正归因于"本题特定内容"的部分）
                    ⭐ 两个分量方向相反，而且在 8B 上双双反转

MATH-500 / Qwen3-4B 总增益 +15.00pp
                    = +8.33（"别的题的消息"也能拿到）
                    + +6.67（本题特定内容）
                    ⭐ 8B 上增益几乎全部来自前者
```

> ⚠️⚠️ **读懂第二条**：一个 **+15 分**的漂亮结果里，**有 8.33 分是"随便给它一条别的题的 latent 消息也能拿到"**。**那部分跟"发送方说了什么"毫无关系。**
>
> → **我们只报"latent 臂比文字臂高多少"会被直接打回。** 计划 §7.2 的 CAG 对照**必须进主表**，而且要**至少两个尺寸的接收方**（因为这篇的结论在 4B/8B 之间会反转）。

### C.2.2 文字到底丢了什么：**丢的主要是表面形式，不是任务语义**

**论文**：[Latent Communication Between Language Model Agents: Channels, Alignment, and the Limits of Text](https://arxiv.org/abs/2607.14103)（Constructor University）· [本地 PDF](../papers/pdfs/2607.14103_LimitsOfText.pdf)

**它的假设**（跟我们一样）：LLM 内部的世界模型可能**超出文字的表达能力**。

**它怎么测**：造三条通道，用 **SAE（稀疏自编码器）特征分析**量"概念区分信息"还剩多少。

> **SAE 是什么**：一个训好的稀疏自编码器，把 LLM 的隐状态拆成一组**可解释的特征**（"这是在讲颜色"、"这是在讲否定"…）。**用它可以数：一段信息里有哪些特征，过一道通道之后还剩哪些。**

**结果**：

| 通道 | 探针准确率 |
|---|---|
| 稠密 latent（直接注入隐状态） | 基准 |
| **SAE-稀疏**（28 倍压缩） | **99.4%** |
| **文字** | **80.4%** |

**跨架构对齐**：Llama ↔ Mistral 之间做一个**简单的 Procrustes 线性对齐**，top-1 检索就有 **92%**。（Procrustes = 找一个正交矩阵把一组向量最好地旋转到另一组上。）

**⭐ 文字往返（序列化再重编码）的特征存活分析**：

```
文字序列化毁掉 88% 的 SAE 特征
⭐ 而且是「身份替换」，不是「衰减」—— 旧特征没了，换上了一组不同的特征
⭐⭐ 但：丢掉的特征「主要或完全编码表面形式，不是任务相关的语义」
```

**任务层面**：latent 通道在跨语言概念任务上**打平文字，从不超过**。文字里再补充 latent 特征：**没有任何好处**。

**作者自己的结论**：*"leading us to **negative conclusions for the initial hypothesis**."*

> ⚠️ **这篇比上一篇更直接地打我们的 motivation。** "文字丢了 88% 的特征"听起来很惊人，**但它同时证明了那 88% 不重要。**
>
> ⭐ **但它自己留了两个口子，写进原文**：
> 1. *"representational convergence between architectures is **real and potentially exploitable**"* ← **支持我们做跨模型 adapter 是可行的**
> 2. *"To pinpoint the practical advantage of latent communication over a text channel, **deeper tasks eliciting complex concepts** and a corresponding analysis framework **are needed which are both beyond our current approach**."* ← **它自己承认任务太浅**
>
> **而且它和上一篇的发送方全都是纯文本 LLM。** 文字发送方序列化的是"一段推理"，丢表面形式合理；**像素发送方序列化的是"我看见了什么"，丢的可能就是语义。两篇都没测这个 case——这是我们的开口。**

---

## C.3 SPARC —— 命名冲撞 + 效率对手（ICML 2026 已录用）

**论文**：SPARC（**S**eparating **P**erception **A**nd **R**easoning **C**ircuits），[2602.06566](https://arxiv.org/abs/2602.06566) · [本地 PDF](../papers/pdfs/2602.06566_SPARC.pdf)

**判决**：P2 = NO，P3 = NO。**不是先例，但是两个麻烦。**

### 它做什么

**同一个 Qwen3-VL**（对 vision 和 language 两侧都加了 LoRA），**两阶段而非循环**（`loop` 全文 0 次）：

> *"the model **first performs explicit visual search to localize question-relevant regions**, then conditions its reasoning on those regions"*
>
> *"running **global search at lower image resolutions** and allocating **high-resolution processing only to selected regions**"*

**接口是 crop / 区域**（`crop` 出现 70 次；`latent` 只 2 次，且都是 "latent visual relevance" 这种形容词用法）。→ 它属于我们早就列出的"**原生 crop/zoom 一族**"。

### ⚠️ 但两件事变了

1. **它是 ICML 2026 已录用**，不是 preprint。**这条威胁的分量升级了。**
2. ⚠️⚠️ **它直接打我们的效率论证。** 我们的说法是"重编码要付 N 次视觉 prefill，KV 复用不用"。SPARC 的做法是"**低分辨率全局搜 + 只给选中区域上高分辨率**"，**同样不付 N 次全量 prefill**，而且已经拿到：
   - Qwen3VL-4B 在 **V\* VQA +6.7**
   - OOD 上比 "thinking with images" 高 4.6 分，而 **token 预算低 200×**
   - 附录 Table 8：**SPARC@256 分辨率就超过 thinking-with-images@全分辨率，TTFT 快 200×、E2E 快 50×**
3. **命名冲撞**：**"Separating Perception And Reasoning" 这个 framing 不能当我们的新颖点了。**

> ⭐ **对计划的直接影响**：[00-plan.md §7.2](00-plan.md) 的效率账**必须把 SPARC 这条线当对手**，不能只跟"全量重编码"比。**我们唯一还站得住的效率差别是：crop/zoom 一族每次重看都要重新过视觉编码器，而 KV 复用不过。** 这个差必须量出来（Kamera 那个 **230ms 视觉塔编码 vs ~5ms KV replay** 就是干这个用的）。

---

# §D 八篇合起来，改了我们什么

| # | 改动 | 来源 |
|---|---|---|
| **1** | ⭐⭐ **"位置修正是免费的"这个假设作废。** 计划 §5.1 三件事里"位置"那栏不能标 ✅ ——位置修到机器精度之后，**同一个模型内**顺序探针仍然只有 41.7%（比不修还差），要重算 25~35% token 才回到上界 | **ChronoStitch** Table 2 |
| **2** | ⭐ **motivation 不能写"文字带宽不够"。** 一篇跑了 F/T/L 三臂的论文自己说 *"a 'text is bandwidth-limited' story, which our own ablations do not support"*；另一篇量出文字丢 88% 特征**但那 88% 不重要** | **Latent Bridge** + **LimitsOfText** |
| **3** | ⭐ **必须抄"桥替换对照"**：把训好的 latent 换成**全零/随机向量**，看分数变不变。比 CAG 便宜，能一眼看出 latent 是不是只是个 prompt 扰动 | **Latent Bridge**（MetaDrive 阴性对照） |
| **4** | ⭐ **CAG 必须进主表 + 至少两个尺寸的接收方。** 一个 +15 分的结果里可能有 8.33 分是"给别的题的消息也能拿到"，而且 4B/8B 结论会反转 | **CausalAudit** |
| **5** | **注入点选输入层（而非几层 cross-attention）得到第三方背书**，同时警告：**离线 KL/cos 好 ≠ 部署能用**（v1 KL=0.004 却上线崩） | **Latent Bridge** v1→v2 |
| **6** | **混合通道要慎重**：两条通道一起喂会互相打架（RoadRunner −96%）。若做混合，**必须在训练时就双通道同时在场** | **Latent Bridge** §4.5 |
| **7** | **"reasoner 比 perceiver 重要"拿到第三份独立证据**（换 orchestrator −10.99% vs 换感知工具 +0.41%） | **SportMV-Agent** 消融 |
| **8** | **"CoT/多轮在感知题上帮倒忙"拿到第三份独立证据**，与我们 C1 的 −17.6 同向 | **SportMV-Agent** 诊断 |
| **9** | **adapter 第二阶段的具体做法有可抄的了**：RLOO + 二元任务成败奖励筛记忆 | **Mem-W** Stage II |
| **10** | ⚠️ **效率账的对手换人**：不是"全量重编码"，是 SPARC 那条"低分辨率搜 + 选区上高分辨率"的线（TTFT 200×） | **SPARC** |
| **11** | ✅ **P1+P3+P4 那个角没被 SportMV-Agent 占**（纠错后），仍然只有 SeeingEye/BeMyEyes，且都是图像不是视频 | 本次纠错 |

---

# §E 术语表（本篇新增）

| 词 | 意思 |
|---|---|
| **Atari / MetaDrive** | 经典街机游戏环境 / 一个开源自动驾驶模拟器。都是"每几十毫秒要出一个动作"的实时控制任务 |
| **residual（残差流）** | Transformer 每一层的主干信号。Latent Bridge 取的是第 24 层最后 8 个位置的 residual |
| **GUI agent** | 自动操作网页/手机/桌面界面的 agent，输入是屏幕截图，输出是点击/输入/滚动等动作 |
| **RLOO** | REINFORCE Leave-One-Out，一种强化学习优势估计法：同一道题采 G 条轨迹，用其余 G−1 条的平均当基线 |
| **mRoPE（三轴）** | 多模态旋转位置编码。视觉 token 的位置是 (时间 t, 高 h, 宽 w) 三元组，而不是一个标量序号 |
| **prefill** | 把 prompt 一次性过一遍模型、生成 KV cache 的阶段（对应 decode = 逐 token 生成） |
| **SAE（稀疏自编码器）** | 把 LLM 隐状态拆成一组可解释稀疏特征的工具，常用来问"这段表示里有哪些概念" |
| **Procrustes 对齐** | 找一个正交矩阵，把一组向量最好地旋转到另一组上。最简单的跨模型表示对齐方法 |
| **蒸馏（KD）/ 教师-学生** | 用一个模型（教师）的输出分布当监督信号去训另一个模型（学生） |
| **V\* / VQAv2 / TempCompass / LVBench** | 分别是：高分辨率细节搜索基准 / 看图问答基准 / 视频时间理解基准 / 长视频理解基准 |
| **TTFT** | Time To First Token，出第一个 token 要多久。衡量 prefill 阶段开销 |

---

## 相关文档

- [01-prior-art-audit.md](01-prior-art-audit.md) —— 完整先例审计（本篇是它 §5/§5.5 的展开）
- [02-navgpt2-and-sterner-explained.md](02-navgpt2-and-sterner-explained.md) —— 推翻我们两条主张的那两篇
- [00-plan.md](00-plan.md) —— 路线计划
- [papers/pdfs/README.md](../papers/pdfs/README.md) —— 本地论文原件目录
