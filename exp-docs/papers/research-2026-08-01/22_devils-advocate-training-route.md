# 魔鬼代言人：为什么「训一个小模型做 VLM 多智能体 latent 通信」大概率是个坏赌注

> **标注约定**
> - `[本次核实]` = 我这次会话亲自 WebFetch / WebSearch / 读本地文件核过的
> - `[四路调研]` = 来自你给我的四份调研（objectives / evaluation / alignment / supervision），我未重复核验
> - `[技术事实]` = 不依赖某篇论文的通用工程/数学事实
> - `[我的推断]` = 我的判断，没有论文背书
>
> 我这次核实的原始材料：`/tmp/claude-1008/-home-yilin/e4c133de-4904-4f07-bf05-a54b12262277/scratchpad/l2vmas.txt`（L²-VMAS 全文）、同目录 `sde.txt`（SDE 全文，Table 4 数字逐个核过）、以及 WebFetch 到的 arXiv 2510.00494 全文。**部分 arXiv 号（2602.\*/2605.\*/2606.\*/2607.\*）来自你的调研包，我没有独立复核其存在性——引用时标 `[四路调研]`。**

---

## 反对意见 1：你不是在开辟赛道，你是在 L²-VMAS 的主场用 1/16 的算力打它

**失败机制（具体到哪一步）**
不是某个张量算错，而是**审稿人的比较表**。你的方法一旦"要训练"，它就自动进入 L²-VMAS 的对照表。而 L²-VMAS 已经把这块地圈死了：

`[本次核实，l2vmas.txt:339/902/904/906/912]` 原文逐条：
- 训练：三阶段 RL，**base VLM 全冻结，只训外部 memory 组件**（Stage I 随机激活 latent memory 优化构建/更新 → Stage II 冻 synthesis 只训 orchestration → Stage III 全解锁端到端）
- 硬件：**8× NVIDIA H200 141G**
- backbone：GLM-4.1V-Thinking、InternVL-3.5-8B、LLaVA-OV-1.5-8B、Qwen3-VL-8B-Thinking/Instruct（5 个）+ 4 个尺寸（2B/4B/8B/32B）+ **6 种拓扑**（linear/layered/centralized/random/complete/dynamic）
- benchmark：MMBench、MMStar、RealWorldQA、SimpleVQA + MuirBench、BLINK、MVBench、LVBench（**8 个**）
- 训练数据：GQA，且明确声明"no exposure to the test benchmarks"
- 结果：平均准确率 **+2.7~5.4%**，token 用量 **−21.3~44.8%**

**这个配置意味着什么** `[我的推断]`：它不只是把结果做出来了，它把**"model-agnostic"和"结构无关"这两个防御性论点也做出来了**。你想说"我的方法更通用"——它已经有 6 拓扑 × 4 尺寸 × 5 backbone 的证据；你想说"我更省"——它已经报了 −44.8% token。你在单卡/少卡上能做的，最多是 1~2 个 backbone × 2~3 个 benchmark，而这恰好落进"coincidental model-specific adaptation"这个它已经预先在正文里点名反驳过的坑（`l2vmas.txt:914` 原话：*"does not rely on coincidental model-, size-, or structure-specific adaptations"*）。

**更毒的一点**：L²-VMAS 的复现性缺口（`[四路调研]`：未披露 loss 形式、reward 定义、算法名、样本量、模块参数量）**对你是负资产不是机会**。因为你没法复现它 → 你没法把它当 baseline 跑 → 你的表里它只能引用它自己报的数字 → 审稿人会说"你和它不可比"。你想通过"我更便宜"来赢，但便宜不是它主张的维度，赢了也不算赢。

- **已实证还是推断**：占地事实 `[本次核实]`；"你打不过"是 `[我的推断]`。
- **最小证伪实验**：在**同一个** backbone（Qwen3-VL-8B-Thinking）、**同一个** benchmark（RealWorldQA，L²-VMAS 报增益最大的那个）、**同一个**拓扑（dynamic）上，用你的单卡训法跑出一个数字，直接和它 Table 1 的格子对齐比。**如果你在 1 张卡上做不出 ≥ 它一半的增益（即 ≥ +1.4%），这条反对成立。** 成本：一周内可判。这个实验必须**在写任何 method 之前做**。

---

## 反对意见 2：四条监督路各自有一个致命弱点，而且它们的弱点不能互相抵消

这是整条路最硬的结构性问题。逐条拆：

### 2a. 文本教师（Vision Wormhole 路线）：天花板 = 老师，而老师正是你要打败的对象

`[四路调研，arXiv:2602.15382]` VW 的 loss：
$$\mathcal{L}_{codec}=\lambda_h\|h_{vis}-\text{sg}(h_{text})\|_2^2+\lambda_{kl}\tau^2\text{KL}(\text{sm}(\ell_{text}/\tau)\|\text{sm}(\ell_{vis}/\tau))+\lambda_{rms}(\text{RMS}(\Delta_{inj})-\text{RMS}(\bar X_{img}))^2$$

它的实际数字：GSM8K 文本 MAS **80.8% / 27.3s** vs VW **76.2% / 26.7s** = **−4.6pp、1.02× 加速**。

**失败机制** `[我的推断]`：第一项是 `stopgrad` 的 MSE，第二项是 KL 到教师 logits。这两项**在数学上的最优解就是"完全复刻文本通道"**。你训得越好，越接近 80.8%，永远不会超过。而 latent 通信的全部卖点是"传文本传不了的东西"——你用一个把文本当上界的 loss，等于在目标函数里亲手把卖点删掉了。1.02× 的加速换 −4.6pp，这不是一个可以卖的 trade-off。

- **最小证伪实验**：训完之后，把 sender 侧输入里一段**文本无法表达的信息**（例如图上一个精确的像素坐标 / 一个细粒度纹理差异）做成一道题，比较「latent 通道」vs「文本通道」。**如果 latent 通道在这类题上不能显著超过文本通道，路 ① 就只是个更慢更贵的文本通道压缩器。** 40 题就够（配对设计）。

### 2b. 端任务 CE（C2C 路线）：MACF 的消融已经证明它单独训通信模块几乎零增益

`[四路调研，arXiv:2605.00444]` MLVU-Test 消融：仅 Stage1 = 36.3%；**Stage1+3 = 36.5%**；Stage1+2 = 46.7%；全三阶段 = 49.2%。

**读法** `[我的推断]`：Stage 3 就是"跨 agent 端任务 CE"。**加上它只涨 0.2pp。** 也就是说，你最可能第一反应去做的事——"我搭个通信模块，用下游任务 CE 端到端训"——在一篇已发表的多模态多 agent 论文里被证明**增益接近零**。它需要 Stage 2（证据摘要，有明确中间监督）才有 +10.4pp。你要复现 Stage 2，就要有 image-QA 中间监督数据，成本回到路 ⑤。

### 2c. RL：没有一篇纯 RL 冷启动成功过，而你多半只买得起纯 RL

`[四路调研]` L²-VMAS / LatentMem / Mem-W 三个用 RL 的，**全部先有稠密监督阶段**。`[本次核实，l2vmas.txt:339]` L²-VMAS 的 Stage I 是"随机激活 latent memory 优化构建与更新机制"，不是从零 RL。

**失败机制** `[技术事实 + 我的推断]`：连续消息 + 稀疏二值奖励 = MARL 里最经典的信用分配灾难。梯度要穿过「sender 生成 latent → 压缩模块 → receiver 全部层 → 采样出的 token 序列 → 二值 reward」。中间的采样是不可微的，只能靠 policy gradient 的高方差估计器。而通信通道刚初始化时对 reward 的边际贡献 ≈ 0，**"忽略通道"是一个稳定的局部最优**（见反对意见 5）。

### 2d. 上界蒸馏（我认为你最可能选的那条）：上界本身可能不存在

`[四路调研]` 建议的构造是：teacher = 单个 VLM 同时吃 sender 的图 + receiver 的图 + 问题；student = receiver 只吃自己的图 + K 个 latent token。

**失败机制** `[我的推断，但有强旁证]`：这个构造有一个隐藏前提——**"把所有 agent 的输入拼给一个 agent"确实是上界**。但 L²-VMAS 全文的动机恰恰是反的：`[本次核实，l2vmas.txt:194]` 原文实测 —— 随 agent 轮数增加，"accuracy 在第 3 轮见顶（84.8→86.6），此后持续下降，**第 6 轮起低于单 agent baseline**，第 10 轮低 2.6%"，token 从 557 涨到 16,840（30×）。**长上下文塞更多信息会让 VLM 更差，不是更好。** 那么你的"上界 teacher"（吃两张图 + 全部文本上下文）很可能**本身就不如 receiver 单独干**。你会去蒸馏一个比 baseline 弱的东西。

DiscoNet 的上界之所以成立，是因为它的 teacher 是**几何上确定更优的**（holistic-view 传感器融合，仿真器给的 privileged info）。VLM 的"看全部"不是 privileged info，是 **distraction**。

- **最小证伪实验（这条必须最先做，成本最低）**：不训任何东西。构造 100 道你的目标任务题，跑三个配置：(i) receiver 单独；(ii) 你设想的 teacher（拼全部输入的单 VLM）；(iii) 文本 MAS。**如果 (ii) 不显著高于 (i) 和 (iii)，路 ④ 直接死，上界蒸馏无标签可用。** 这个实验一天能做完，零训练成本。**我强烈建议你在做任何其他事之前先做它。**

---

## 反对意见 3：训完之后，LatentMAS 的理论包袱不是"还剩多少价值"，而是**变成负资产**

**失败机制**：Theorem 3.3 的内容是"latent working memory transfer 保证信息保真度等价于显式输入交换"`[本次核实，2511.20639 检索摘要 + 官方 repo 描述]`。这个"无损"结论**唯一的来源就是 KV 没有经过任何有损变换**——它是 training-free 的直接副产品。

**你一旦插入一个学出来的压缩/投影模块：**
1. **Thm 3.3 立即失效**，且不可修补——你的模块是 K 个 token 的瓶颈，信息论上必然有损，你无法证明任何保真度下界。
2. **Thm 3.1 / 3.4（表达力 / 复杂度）你仍然继承**，但那不是你的贡献，是 LatentMAS 的。你在 related work 里引用它们，不能放进 contribution。
3. **$W_a$ 的理论断点被"绕开"而不是"解决"** `[我的推断]`。绕开的方式是"我学一个投影"。但"学一个跨模型/跨模态的 KV 投影 + 逐层门控"**就是 C2C（2510.03215）的定义**。你的 novelty 收缩成：**C2C 换成 VLM**。

**审稿人的一句话杀伤** `[我的推断]`：
> "这与 C2C 的区别是把 LLM 换成 VLM；与 L²-VMAS 的区别是少了两个训练阶段；与 Vision Wormhole 的区别是把闭式 ridge 换成学出来的映射。请说明方法论上的新贡献。"

你现在没有一句话能回答这个问题。**这是最该在开题前就解决的问题，而不是等实验做完。**

**另一个被忽略的成本**：training-free 版本有一个训练版**永远拿不回来的**性质——**对任意新 backbone 对、新任务零成本**。你训完之后要额外证明「GQA 上训的模块能迁移到 MMBench/BLINK/LVBench」，而 L²-VMAS 花了 8×H200 才敢报这张泛化表（`l2vmas.txt:1223` Table 10）。

- **最小证伪实验**：写一段 150 字的"与 C2C 的方法论差异"，拿给三个不做这个方向的博士生读，让他们复述。**如果三个人里有两个复述成"就是 C2C 用在 VLM 上"，这条反对成立。** 零成本，今天就能做。

---

## 反对意见 4：你训出来的增益，最可能是「参数量」和「消息存在」的功劳，不是通信的功劳 —— 而且已经有论文在同一形状的系统上实测到了

这是我认为**最致命**的一条，因为它有一个几乎完全同构的实证反例。

### 4a. 参数量混淆：已有的直接反例

`[本次核实，arXiv:2510.00494 全文 WebFetch]` 这篇做的是 LLM 内部的 System1↔System2 latent 通信（Base + Coprocessor 双模型，通过 latent token 通信）。它做了你**必须做但极可能不敢做**的对照：

> "The parameter-matched single model (Soft embeddings 250M) attain[s] **lower validation perplexity** than the Hypothesis 2 dual-system."（Sec. 10 / Figure 13）

- 参数配平方式：单个 254M GPT-2 ≈ (124M Base + 124M Coprocessor)
- 更狠的是 soft-embedding baseline：**同样的 latent budget（$N_L=16$）、一半的参数**，就"nearly matched the best dual-model variant"
- GPT-2 上双系统只比这个简单 baseline 高 **+0.4pp**（Table 1：HellaSwag 31.2 (+0.5)、ARC-Easy 55.8 (+1.6)、Social IQA 38.2 (+0.4)、PIQA 65.3 (+0.8)、Winogrande 52.1 (+1.6)）
- 而且"scaling the latent budget beyond small values fails to improve robustness"，在 GSM8K/ProsQA/Countdown 上"reasoning accuracy remained largely flat as $N_L$ increases and **can dip at larger $N_L$**"

**这对你意味着什么** `[我的推断]`：这是一个**已经发表的、结构和你设想同构的系统，在做了参数配平对照之后，通信收益基本消失**。你的对照组必须包括：把同等参数量直接加给 receiver（LoRA / prefix / 额外 soft token），**不做任何 agent 间通信**。C2C 自己的 Table 6 已经埋了这颗雷：`[四路调研]` fuser 478M vs 直接微调 receiver 596M —— **两者已经在同一量级**，审稿人必问。

### 4b. 消息存在 vs 消息内容：Causal Audit 的分解

`[四路调研，arXiv:2607.26773]` $\text{OPE}=\text{OME}+\text{CAG}$，且实测出了两个方向相反的例子：
- Qwen3-4B / GSM8K：OPE **−1.00pp** = OME −6.17pp + CAG **+5.17pp**（内容有用，注入方式有害）
- Qwen3-8B / GSM8K：OPE **+1.67pp** = OME +3.96pp + **CAG −2.29pp**（**内容有害，涨分全靠"有消息"**）

**失败机制，具体到张量** `[我的推断]`：训练路线**主动优化**了产生 OME 型假阳性的能力。receiver 冻结、只有通信模块可训时，梯度最容易找到的解是：让注入的 $\Delta$KV 在某几层形成一个**与输入无关的固定偏置**（相当于一个学出来的 soft prompt / 模式切换开关），把 receiver 推进一个"更谨慎/更长推理"的模式。这个偏置对**任何**消息都一样 → CAG ≈ 0，OME 大。而你的 loss（端任务 CE）**完全奖励这个解**，因为它确实涨分。

training-free 的 LatentMAS 反而不太会有这个问题——它没有可优化的自由度去学这个开关。**你换成训练路线，等于把这个失败模式从"可能"变成"梯度下降的首选"。**

- **最小证伪实验**：训完后跑四条件（$M_0$ / $M_{cur}$ / $M_{oth}$ 长度匹配的他样本消息 / $M_{self}$ 算力配平的自生成消息），报 OPE / OME / **CAG** / SSG。同题配对、固定 seed。**如果 CAG 的 bootstrap 区间跨零，你训的不是通信模块。** 再加一条参数配平臂：同参数量的 receiver-only LoRA。**如果它打平，你训的是个 adapter。** 这两条实验总成本 ≈ 5 组推理 × 200 题，一天内可跑完。

---

## 反对意见 5：连续消息通道的三个病理，训练路线全中；十年前就有人踩过

**病理 A：发送方"作弊编码"，指标虚高**
`[本次核实检索确认，arXiv:1903.05168]` Lowe et al. AAMAS'19 的核心反例：**即使消息在被观察前就被打乱（scrambled），agent 依然表现出高 Speaker Consistency**。原因是动作输出头与通信输出头**共享网络特征**，网络按"打算做的动作"分离表征——**即使通信参数完全没训练**。原文结论："agents can exhibit positive signaling without positive listening"。

`[我的推断]` 对你的直接后果：你的 sender 是 VLM 本体的 hidden，天然与 sender 的答案高度相关。**你的 PS/互信息指标一定好看，而它什么也不证明。不要拿它当卖点。**

**病理 B：接收方学会忽略消息（且这是稳定均衡）**
`[本次核实检索]` emergent-comm 文献共识：存在一个均衡——"speaker produces random symbols and the listener's policy is independent of communication"；"frozen senders produce largely collapsed messages with constant symbols across positions"。

`[四路调研]` C2C 源码里 `script/train/SFT_train.py:443-449` 有一段**被注释掉的**门控稀疏正则 `loss += 0.0025 * gate` —— 说明作者撞过门控行为的问题但没解决。而 C2C 论文**没有报告"门到底开没开"**。

`[我的推断]` 你的 gate（Gumbel-sigmoid，推理时硬二值）有一个平凡解：**全部关掉，退化成 receiver 单模型**。而如果你的 baseline 就是 receiver 单模型，这个解在 loss 上是"不比 baseline 差"，梯度没有动力离开它。

**病理 C：表示坍缩（已有量化证据）**
`[本次核实，2510.00494]` 它的可解释性分析：large-scale pretraining 时 latent 之间 **mean off-diagonal capture $\bar H_{off}=0.9873$**，silhouette $s=-0.1694$ —— 也就是**16 个 latent token 的子空间几乎完全重叠、聚类结构为负**。K=32 个 latent token 里可能只有 1~2 个方向是有效的。

`[四路调研]` 唯一显式防这个的是 Interlat（$\mathcal{L}_{sep}$ 加权 JS 散度 + $\mathcal{L}_{align}$），去掉 $\mathcal{L}_{sep}$ 就出现 "shortcut behavior，模型学会直接忽略 latent 通信"（论文原话）。**而 Interlat 是唯一一篇全量 SFT actor 的——它靠"动 backbone"才拿到这个效果，你冻结 backbone 时未必成立。**

- **最小证伪实验（三合一，都很便宜）**：
  1. **门开度审计**：训完后打印每层 gate 的硬二值结果。**若 >70% 的层关闭，或 gate 与输入无关（不同样本同一模式），病理 B 成立。**
  2. **消息秩审计**：收集 500 个样本的 K 个 latent token，做 SVD。**若 90% 能量落在 ≤3 个方向，病理 C 成立。**
  3. **随机通道对照**：把训好的模块的输出替换成**从同一分布采的随机向量**（保 norm）。**若准确率不掉，通道没在传信息。**

---

## 反对意见 6：视觉 hidden state 的分布问题让训练难一个量级，而且难在你的 loss 项上

**失败机制，具体到张量维度**

`[本次核实检索确认，arXiv:2402.17762]` Massive Activations：LLM 的 hidden state 里有**极少数特征维度**出现数量级更大的值，**input-agnostic**，深度方向"突然出现在某一层之后"；在 LLaMA2-7B 上**把 4 个 massive activation 置零就会导致灾难性崩溃**。

`[本次核实检索确认，arXiv:2503.03321]` Visual Attention Sink：在 LMM 中，**irrelevant visual tokens 在特定维度上有高激活，而这些维度与 BOS sink token 的维度完全相同**；论文明确提到 visual over-representation 和 **modality gap**（文本 token 与视觉 token 在 latent 空间的明确分布差）。

**这三件事合起来对你的 loss 做了什么** `[我的推断，但机制是确定的]`：
1. **MSE 项被 massive activation 维度独占梯度**。$\|h_{vis}-h_{text}\|_2^2$ 里，如果某 4 个维度的值比其余大 100×，梯度的 $10^4$ 倍权重都在那 4 个维度上。**你训了半天，学到的是"复刻 sink 维度的常数"**——而那 4 个维度恰恰是 input-agnostic 的，即**零信息量**。这直接产生反对意见 4 的 OME 型假阳性：注入的 $\Delta$ 主要是个常数偏置。
2. **RMS 对齐项（VW 的第三项）在视觉端更危险**。视觉 token 的 RMS 本身就被 sink 维度支配且 token 间差异巨大，把 $\text{RMS}(\Delta_{inj})$ 对齐到 $\text{RMS}(\bar X_{img})$ 相当于对齐到一个被少数维度决定的量。
3. **视觉 KV 的层内分布是异质的**：图像 token 与文本 token 混在同一序列里，你的门控是**逐层标量**（C2C 的做法），它无法区分"这一层的视觉 token 位置"和"文本 token 位置"。
4. **数值稳定性的旁证**：`[四路调研]` VW 源码里有 latent clip ±50、logit clip ±80、injection clip ±20、`nan_to_num(posinf=1e4)`、非有限损失检测跳步、`all_bad_grad_streak` 计数器，**InternVL 上还要专门开关 `--vision_codec_internvl_disable_kl` 直接关掉 KL 项**。这个补丁密度就是"这个目标函数在视觉端很脆"的证据。

**另一个层次的证据** `[本次核实，sde.txt Table 4 逐格核过]`：SDE（2506.19209）的 "w/o delta"（注入原始 hidden state）在 4 个设置里：
- Q-7B：NL 0.3050 / w/o delta **0.2950** / SDE 0.3150
- L-8B：NL 0.3250 / w/o delta **0.2967** / SDE 0.3517（另一列 L-8B NL 0.2850 / w/o delta 0.2750 / SDE 0.3050）

**注入原始 hidden state 掉到纯文本 baseline 以下**，而且 SDE 还发现"改所有层会显著掉点，只能改 top-1~3 层"（`sde.txt:1889/1898`）。这是纯文本场景的结论。视觉端只会更糟。

- **最小证伪实验**：**零训练**。抓 Qwen3-VL-8B 上 200 个样本的 hidden state，分别统计视觉 token 和文本 token 的 per-dim 方差分布，画 Pareto 曲线。**若 top-4 维度占 >50% 的 L2 能量，你的 MSE loss 就是在拟合 sink，这条反对成立**，你必须改成 per-dim 标准化后的 MSE 或换 loss。一小时能做完。

---

## 反对意见 7：单卡显存做不了这件事，而"冻结 backbone 省显存"是个误解

**失败机制，逐项算账** `[计算 + 技术事实]`

假设 Qwen3-VL-8B 级别，两个 agent：
- **权重常驻**：sender VLM + receiver VLM，bf16 各 ~16GB = **32GB**（即使同一个模型两份角色，也至少 16GB）
- **KV 张量本身**：按 36 层 / GQA 8 KV head × 128 dim = 1024 维估：每 token 每层 $2\times1024\times2\text{B}=4$KB，×36 层 = **147KB/token**。1024 个视觉 token ≈ **151MB**（bf16）。多轮多 agent ×N。
- **⚠️ 关键误解**：**冻结 receiver 不省激活显存**。梯度必须从 receiver 的输出 loss 一路反传到你插在第 $\ell$ 层的注入点，**中间所有层的前向激活都必须保留**。冻结只省优化器状态（Adam 的 2× fp32 momentum），**不省 activation**。8B 模型在 2k 序列长度下 activation 轻松几十 GB。
- **结论**：**24GB / 48GB 卡基本出局**。单张 80GB A100/H100 上，batch=1 + gradient checkpointing + 只在浅层注入，勉强能跑，但你会被迫做出一系列**削弱方法本身**的妥协（短序列、少层注入、tiny batch）。

**对比成本坐标** `[四路调研]`：
- C2C：500k 样本 × 1 epoch，有效 batch 256（4×8×8 卡）
- L²-VMAS：8×H200，三阶段 RL `[本次核实]`
- Vision Wormhole：**400 步 / batch 2 / A6000**，总共 800 次 anchor 抽样（0.27× 数据覆盖率）

**这里有一个残酷的相关性** `[我的推断]`：**最便宜的那个（VW）恰好是天花板最低的那个（GSM8K −4.6pp）**。成本和天花板在这个地形里是正相关的。你想用 VW 的成本拿 L²-VMAS 的结果，没有先例。

- **最小证伪实验**：不写方法，先写一个 dummy 训练脚本：两个 8B VLM 常驻 + 在 receiver 第 18 层注入一个随机 $\Delta$KV + 对最终 loss 反传。**跑通并测峰值显存和单步耗时。** 若单步 >5s 或 OOM，把"训练 8B 级 VLM 通信"这个设想直接降级到 2B。半天可判。

---

## 反对意见 8：即使一切成功，你可能只是造了一个「test-time compute 分配器」

**失败机制**：`[四路调研，arXiv:2607.26773]` GSM8K / Qwen3-4B 上 **CAG = +5.17pp 但 SSG = −2.00pp（区间跨零）**，论文据此说 "example-specific content and other-agent value are distinct"。

翻译：**消息内容确实有用，但 receiver 自己花同样算力想一遍也能得到。**

`[我的推断]` 你的训练路线让这个风险更高，因为：你训的模块是从 sender 的 hidden 里提取信息 —— 但 sender 和 receiver 用的是**同一个或同族 VLM，看的是同一个任务**。你提取的"信息"很可能只是"多跑了一遍前向"的结果。$M_{self}$（算力配平的自生成消息）一测就穿。

这不是完全没价值——"latent 形式的 test-time compute 更省 token"仍然是一个可发表的主张。但**它不是"agent 间通信"**，而你的整个故事、整个 related work、整个 L²-VMAS 对比表都是按"通信"写的。**这是一个必须在写作前就诚实决定的定位问题。**

- **最小证伪实验**：$M_{self}$ 臂。让 receiver 用**完全相同的算力预算**（同样的 latent 步数 $m$、同样的额外前向次数）自己生成消息注入自己。**若 SSG 区间跨零，你的贡献是 compute 分配不是通信。**

---

## 反对意见 9：评估成本本身就会压垮单人项目

`[我的推断，基于前面各条的实验清单]` 一个能过审的训练版通信模块，最少需要：

| 对照臂 | 为什么必须有 |
|---|---|
| receiver 单独 | baseline |
| 文本 MAS | 你要打败的对象 |
| LatentMAS（training-free） | 你要证明训练有意义 |
| **参数配平的 receiver-only LoRA** | 反对意见 4a |
| $M_0$ / $M_{oth}$ / $M_{self}$ / $M_{xbench}$ | 反对意见 4b、8 |
| 随机消息（保 norm） | 反对意见 5 |
| L²-VMAS | 同赛道 SOTA（**你复现不了**） |

= **至少 9 条臂 × N 个 benchmark × K 个 backbone × 多 seed**。而 `[四路调研]` causal audit 自己只用了 40–100 题、多个区间跨零。你要压住方差，配对设计 + 固定 seed 是必须的，样本量还得再上。**这个评估矩阵的推理成本很可能超过训练成本。**

- **最小证伪实验**：把上面 9 条臂 × 200 题 × 3 seed 的**推理 GPU 小时**先算一遍。若 >你总预算的 40%，说明训练根本不是瓶颈，评估是。

---

## 反对意见 10：时间窗口

`[本次核实 + 四路调研]` 2025-11 LatentMAS → 2026-02 Vision Wormhole / L²-VMAS → 2026-05 MACF / Mem-W → 2026-07 Causal Audit。**这个方向从"新"到"有审计论文"只用了 9 个月。** 审计论文出现是一个赛道成熟的标志——它意味着**下一批投稿会被用审计标准要求**。你现在（2026-08）入场，做完训练 + 9 条臂评估最快也要 4–6 个月，那时你面对的是「已经有 3 篇更早的工作 + 一套审稿人都知道的审计协议」。

- **最小证伪实验**：查一下你目标会场的下一个 deadline，倒推。**若留给你的时间 < 6 个月，反对意见 1/9 会自动杀死这个项目。**

---

## 综合账本

| 反对 | 严重度 | 可通过实验快速证伪？ | 成本 |
|---|---|---|---|
| 1 竞争面（L²-VMAS） | 🔴 致命 | 能（对齐单格子） | 1 周 |
| 2d 上界不存在 | 🔴 致命 | **能，零训练** | **1 天** |
| 3 丢卖点 = 变 adapter | 🔴 致命 | 能（150 字复述测试） | **0** |
| 4 参数量/OME 假阳性 | 🔴 致命 | 能（配平臂 + CAG） | 1 天推理 |
| 6 视觉分布 | 🟠 高 | **能，零训练** | **1 小时** |
| 7 显存 | 🟠 高 | 能（dummy 脚本） | 半天 |
| 5 坍缩 | 🟡 中 | 只能训完后测 | 训练后 |
| 8 test-time compute | 🟡 中 | 能（$M_{self}$ 臂） | 1 天 |
| 9 评估成本 | 🟡 中 | 能（算账） | 0 |
| 10 时间窗口 | 🟡 中 | 能（查 ddl） | 0 |

**注意最右列**：**四条致命反对里有三条可以在一周内、几乎零训练成本地判定。** 这是这份报告最实用的部分——你不需要先训一个模型再来发现路走不通。

---

## 总判决：**(B) 训，但必须换一个主张、换一个小得多的目标**

不选 (A)，因为 `[四路调研，2503.24129 / 2209.15430 / 2311.00664]` 免训练对齐的证据链在规模上是死的：Blind Match 在 N>100 类就急剧衰减、无监督分类 51.1%；相对表示跨架构 stitch 的标准差 ±21.14；文本侧 Procrustes 保真只有 85–91%。**而且所有这些证据都建立在"每样本一个向量"的 embedding 上，没有一篇验证过 per-token × per-layer × per-head 的 KV 张量满足同样的几何。** 纯免训练在 VLM KV 这一层没有任何正面证据支撑，坚持它是在赌一个未被验证的假设。

不选 (C)，因为反对意见 1/3/4 三条同时成立：**你在和 8×H200 抢同一个主张（"训练式 VLM latent 通信涨点"），而这个主张本身可能是假阳性，且你没有一句话能区分自己和 C2C。** 这三件事任何一件单独出现都能救，同时出现不能。

### (B) 具体是什么意思：三个可以活下来的主张，选一个

三个都满足：**不和 L²-VMAS 抢同一个格子、单卡可做、结论无论正负都能发。**

**B1 —「审计优先」：把 causal audit 协议扩到 VLM，顺带交付一个最小训练模块**
- 主张不是"我涨点"，是"**视觉 latent 通道到底传没传视觉信息**"。
- 关键是 causal audit 没有的、只有 VLM 才有的对照臂：**换图不换文 / 换文不换图 / 图文错配**。这直接对应 $W_a$ 理论断点——如果视觉信息压根没过去，CAG 在"换图"臂上会归零。
- `[四路调研]` VW（最接近的 VLM latent 通道工作）**完全没有任何通道级指标**，只有下游准确率和 wall-clock。这是一个确认存在的空档。
- 训练模块只作为"被审计对象之一"，可以很小、很弱，**结论为负也能发**。

**B2 —「低秩补丁 + 末层监督」：把参数量压两个数量级，用规模本身当主张**
- `[四路调研]` Kamera（2606.23581）证明 KV 重用丢掉的那部分**低秩（rank≈32 就够）且集中在深层**；Revisiting Model Stitching（2603.12433）证明**必须在接收方"最终输出层"做特征匹配（FFM），在拼接点匹配会失败**。
- 合起来推出一个具体设计：**per-layer rank-32 低秩 $\Delta$KV 补丁 + 只在 receiver 末层监督**。参数量对标 LCF-128 的 19.4M 甚至剪枝版 6.24M，而不是 C2C 的 478M。
- **主张变成"用 C2C 的 1/50 参数拿到可比效果"** —— 这个主张 L²-VMAS 没占，而且**参数量小本身就是反对意见 4a 的免疫**（审稿人问"是不是参数量的功劳"，你答"我只有 6M"）。

**B3 —「负面结论」：视觉证据到底能不能过 latent 通道**
- `[四路调研]` 所有现有工作传的都是 sender 的**推理 rollout hidden**，VW 的 sender 虽是 VLM 但论文不讨论 sender 的视觉输入如何处理。**"跨模型传输视觉证据"无人正面解决。**
- 设计一批**只有看图才能答**的题（细粒度定位、计数、纹理），测 latent 通道 vs 文本通道。**如果结论是"传不过去"，这是一篇有价值的负面论文**，且成本远低于正面工作。

**我的排序：B1 > B3 > B2。** B1 风险最低（评估型工作对算力最不敏感、且有确认的空档）、和 L²-VMAS 完全不冲突、还能顺手把反对意见 4/5/8 全部变成你的论文内容而不是你的软肋。

---

### 推翻我这个判决所需的证据

按"推翻力度"排序，**任何一条成立我就改判 (C)**：

1. **上界存在的证据（最关键）**：反对意见 2d 的一天实验，如果 teacher（拼全部输入的单 VLM）在你的目标任务上**显著高于** receiver 单独 **且**高于文本 MAS，那么路 ④ 活了，稠密监督有了，判决改为 (C)。**这是我认为你最该先做的一件事。**

2. **通信收益在参数配平后仍然存活**：跑一个 quick-and-dirty 版本（哪怕 2B backbone、一个 benchmark），报 CAG 和"同参数量 receiver-only LoRA"对照。**若 CAG 显著为正、区间不跨零，且显著高于配平臂**，反对意见 4 死，判决改为 (C)。

3. **视觉 hidden 的分布没那么糟**：反对意见 6 的一小时实验，若 top-4 维度能量占比 <20%、视觉与文本 token 的 per-dim 方差分布接近，反对意见 6 大幅减弱。

4. **单卡可行性证据**：反对意见 7 的 dummy 脚本在 48GB 卡上单步 <2s，则算力劣势没我说的那么大。

5. **一句话 novelty 通过复述测试**：如果你能写出一段让三个外行博士生都不复述成"C2C 换 VLM"的方法论差异，反对意见 3 死。

6. **L²-VMAS 单格子对齐胜出**：在 Qwen3-VL-8B-Thinking / RealWorldQA / dynamic 上你的单卡结果 ≥ +2.7%（它的下限），反对意见 1 死。

**反过来，会让我从 (B) 进一步退到 (A) 或"完全放弃"的证据只有一个**：如果 B1 的审计实验做出来发现，**即使是 training-free 的 LatentMAS 在 VLM 上的"换图"臂 CAG 也归零**——即视觉信息压根没通过 latent 通道传输过——那么整个"VLM latent 通信"的前提就不成立，训不训都没意义，此时应该把这个负面结论本身发出去（回到 B3），而不是继续做方法。

---

**Sources**（本次核实的外部来源）：
- [Massive Activations in Large Language Models (2402.17762)](https://arxiv.org/pdf/2402.17762)
- [See What You Are Told: Visual Attention Sink in Large Multimodal Models (2503.03321)](https://arxiv.org/pdf/2503.03321)
- [On the Pitfalls of Measuring Emergent Communication (1903.05168)](https://arxiv.org/pdf/1903.05168)
- [Exploring System 1 and 2 communication for latent reasoning in LLMs (2510.00494)](https://arxiv.org/html/2510.00494)
- [Latent Collaboration in Multi-Agent Systems / LatentMAS (2511.20639)](https://arxiv.org/abs/2511.20639) · [官方 repo](https://github.com/Gen-Verse/LatentMAS)
- [Do Latent Channels Actually Communicate? A Causal Audit (2607.26773)](https://arxiv.org/html/2607.26773v1)
- [Beyond Tokens: A Unified Framework for Latent Communication in LLM-based MAS (2606.05711)](https://arxiv.org/html/2606.05711v3)
- [Training LLMs to Reason in a Continuous Latent Space / Coconut (2412.06769)](https://arxiv.org/pdf/2412.06769)

**本地核实文件**：`/tmp/claude-1008/-home-yilin/e4c133de-4904-4f07-bf05-a54b12262277/scratchpad/l2vmas.txt`（L²-VMAS 全文，引用行号 194/339/902/904/906/912/914/1223）、`/tmp/claude-1008/-home-yilin/e4c133de-4904-4f07-bf05-a54b12262277/scratchpad/sde.txt`（SDE 全文，Table 4 数字与 1889/1898 行结论）


---

## 附：关键发现速览

1. **已有直接反例：参数配平后通信收益消失**。arXiv:2510.00494（我本次 WebFetch 全文核实）在一个结构同构的 latent 通信双系统上做了参数配平对照——254M 单模型（= 124M Base + 124M Coprocessor）验证困惑度**更低**；同 latent budget、一半参数的 soft-embedding baseline "nearly matched" 最好的双模型变体，GPT-2 上双系统只高 +0.4pp；且 latent 子空间坍缩量化为 H̄_off=0.9873、silhouette=−0.1694。这是"训出来的增益来自参数量而非通信"这一假阳性的**已实证**先例，不是推测。

2. **"上界蒸馏"这条最被推荐的监督路，其上界可能根本不存在**。L²-VMAS 原文实测（本地 l2vmas.txt:194）：VLM 多 agent 的准确率第 3 轮见顶后持续下降，**第 6 轮起低于单 agent baseline**，token 涨 30×。也就是"把所有 agent 输入拼给一个 VLM"很可能不如 receiver 单独干——DiscoNet 式 privileged teacher 在 VLM 长上下文场景不成立。**这个可以零训练、一天内证伪，我建议在做任何其他事之前先做。**

3. **视觉 hidden 的 massive activation 会让 MSE 类 loss 学到零信息量的常数**。2402.17762：少数 input-agnostic 维度值大数量级，LLaMA2-7B 上置零 4 个即灾难崩溃；2503.03321：LMM 中 irrelevant visual token 的高激活维度**与 BOS sink 维度完全相同**，且存在明确 modality gap。合起来：‖h_vis−h_text‖² 的梯度被 sink 维度独占，学出来的 Δ 主要是常数偏置——这恰好制造 Causal Audit 里的 OME 型假阳性。旁证是 Vision Wormhole 源码的补丁密度（clip ±50/±80/±20、非有限损失跳步、InternVL 要专门关掉 KL 项）。

4. **"冻结 backbone 省显存"是误解，单卡出局**。梯度必须从 receiver 输出反传到注入层，中间**所有层的前向激活都要保留**；冻结只省优化器状态。两个 8B VLM 权重 bf16 就 32GB，24/48GB 卡基本做不了。而成本坐标里有个残酷相关性：最便宜的方案（Vision Wormhole，400 步 A6000）恰好是天花板最低的那个（GSM8K −4.6pp、1.02× 加速）。

5. **判决 (B)：训，但必须换主张**。四条致命反对里有三条（上界不存在、novelty=C2C 换 VLM、参数量假阳性）可在一周内近零成本判定。推荐主张 B1「把 causal audit 协议扩到 VLM，加视觉专属对照臂（换图不换文/图文错配）」——Vision Wormhole 完全没有任何通道级指标，这是确认存在的空档，且结论为负也能发。改判 (C) 的唯一硬证据是：上界实验成立 + CAG 在参数配平臂之上显著为正。
