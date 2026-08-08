# 训练式 latent 通信模块：评估协议

> **标注约定**
> `【论文】` = 论文原文明确写的，附 arXiv / ACL 号
> `【本次核实】` = 我在本次会话中用 WebSearch/WebFetch 亲自查证过的
> `【上游】` = 来自本次四路调研报告（objectives / evaluation / alignment / supervision），报告方声明读过原文，我未逐条复核 PDF
> `【计算】` = 我自己算的，给出算式
> `【设计】` = 我的协议设计，无论文背书，是给你的建议
> `【未核实】` = 检索里见到但没读全文，不要引用

---

## 0. 这套协议要回答的三个独立问题

你的新设想是「训一个小模块做 agent 间 latent 通信」。评估协议的骨架不是「跑哪些 benchmark」，而是**把一个含混的问题拆成三个可以分别被证伪的问题**：

| | 问题 | 层 | 失败时你学到什么 |
|---|---|---|---|
| Q1 | 模块的输出里**有没有** sender 的信息？ | L1 通道层 | 编码器坏了 / 带宽被浪费了 |
| Q2 | receiver **用没用**，用的是**内容**还是"有消息"这件事？ | L2 协作层 | 训成了触发器，不是信道 |
| Q3 | 值不值？准确率、token、时延 | L3 任务层 | 有效但不划算 |

**为什么必须分开：** 【论文，2607.26773，经【上游】转述】在 GSM8K / Qwen3-4B 上，latent 通信相对无通信的**总效应是 −1.00pp**，但精确分解是 OME(消息存在效应) −6.17pp + CAG(内容归因增益) +5.17pp —— 通道内容实实在在帮了 5.17pp，只是"塞任何东西进去"这个动作伤了 6.17pp。反过来 Qwen3-8B / GSM8K 总效应 +1.67pp = OME +3.96pp + **CAG −2.29pp** —— 内容其实**有害**，涨分全靠"有消息"，但论文一律会写"我们的方法有效"。

`【设计】` 这两个例子对**训练**路线尤其致命。training-free 的 LatentMAS 没有可优化的自由度去学触发器；你一旦上梯度，**端到端 CE 最容易的下降方向就是让 receiver 对"消息存在"产生反应，而不是学会读消息内容**。这个失效在准确率上看起来完全像成功。所以：

> **本协议的主 endpoint 预注册为 CAG，不是 ΔAccuracy。** `【设计】`

---

## L0. 健全性套件（不过这关，后面所有数字都是噪声）

`【设计】` 这一层不是「指标」，是「你的注入代码有没有 bug」。latent 注入代码极易写错（层错位、位置索引偏一、mask 没对齐、TP rank 不一致），而这类 bug **不会崩溃**，只会让你测到一堆有意义外观的数字。

| 检查 | 做法 | 必须满足 | 来源 |
|---|---|---|---|
| **H1 掩码** | 把消息 mask 掉 | CIC 掉到数值地板 | 【论文，2607.26773】 |
| **H2 自替换恒等** | 用 $M_{cur}$ 去"替换" $M_{oth}$ 的位置（两边都是当前样本） | **CIC = CAG = 0，精确为 0** | 【论文，2607.26773】 |
| **H3 完全恢复** | 恢复全部组件 | NLD = 1 | 【论文，2607.26773】 |
| **H4 receiver 噪声地板** | 同一条 $M_{cur}$ 重跑 K=8 次，算 $\bar D_{cur,cur'}$ | 后续所有 CIC **必须显著大于**此地板 | `【设计】`，【论文原文提了 "receiver-instability" 但未给操作定义】 |
| **H5 形状匹配** | 消息 token 数、norm 分布做 KS 检验 | $M_{cur}/M_{oth}/M_{rand}$ 三组分布不可区分（p > 0.1） | `【设计】` |
| **H6 决定性** | 温度=0 重跑两次 | 逐 token 完全相同 | `【设计】` |

**H2 是全套里最毒的一条** `【设计】`：它是一个「本该平凡相等」的边界 case，任何注入路径的不对称（比如 $M_{cur}$ 走的是 in-place 写、$M_{oth}$ 走的是 concat）都会在这里露馅，而在正常实验里永远看不出来。**这一条必须进 CI。**

`【设计】` H4 是我给上游那篇补的洞。$T=0.6$ 的采样下 receiver 自身噪声地板不会小；不量出来，你报的 CIC 可能就是采样温度。**主实验一律 temperature=0**，采样只做鲁棒性附录。

---

## L1. 通道层：不看任务，只看消息里有什么

### L1.1 探针可解码性 —— 但**必须带控制任务**

`【设计】` 训一个线性探针，从消息 $M$ 预测 sender 侧的目标属性：图里的目标物体类别 / 物体计数 / sender 的中间结论 / 空间关系。

**光报探针准确率是废数。** 【论文，1909.03368，Hewitt & Liang, EMNLP 2019，本次核实】原文的核心发现是：探针（尤其是神经网络探针）**能独立于表示本身，记住大量标注决策**。因此必须同时构造 **control task**（把每个类型随机映射到随机输出，这个任务按构造只能靠探针自己学会），报告：

$$\text{Selectivity} = \text{Acc}_{\text{任务}} - \text{Acc}_{\text{控制任务}}$$

一个好探针必须是 **selective 的**：任务准确率高、控制任务准确率低。

### L1.2 更稳的替代：MDL 描述长度

【论文，2020.emnlp-main.14，Voita & Titov, EMNLP 2020，本次核实】原文指出：探针准确率的差异**不足以反映表示的差异**，对预训练表示相对随机初始化表示没有明显偏好，在真实语言标签和随机合成任务上甚至给出相近数字。他们的替代是把"训探针预测标签"重述为"教它高效传输数据"，测量的是 **给定表示后标签的描述长度**。两种估计：variational coding 与 online coding。

- 现成实现：`github.com/lena-voita/description-length-probing` 【本次核实】
- 报法：`【设计】` codelength (bits) + compression ratio（相对 uniform code）。比 accuracy 稳定，且天然惩罚"探针自己学会了"。

### L1.3 **不要报"消息里有几个 bit"**

`【上游】`【论文，1811.04251, McAllester & Stratos, AISTATS'20 / 1905.06922, Poole et al.】任何 distribution-free 高置信 MI 下界不超过 $O(\ln N)$，1000 个样本天花板约 10 bit；InfoNCE $\le \log K$。所以「hidden state ~40000 bits vs token ~15 bits」这类说法只是松上界，写进论文会被打。

**正确用法是相对量** `【设计】`：

$$\text{RelSep} = \text{AUC}\big[\ \text{判别}(M_i \leftrightarrow e_i)\ \text{vs}\ (M_j \leftrightarrow e_i)\ \big]$$

即"消息-样本配对是否可分"，等价于 InfoNCE 式 top-1 检索准确率。报的时候明确标注 $\le \log K$ 上界和 K 值。

### L1.4 消息几何健康度（这一组是训练监控的主力）

`【设计】` 四个都是 batch 内秒级可算，全部进 tensorboard：

| 量 | 定义 | 坍缩信号 |
|---|---|---|
| 有效秩 / 参与比 | $\text{PR} = (\sum_i \lambda_i)^2 / \sum_i \lambda_i^2$，$\lambda$ 为 batch 消息矩阵协方差特征值 | PR → 1 |
| batch 内平均 cosine | $\mathbb{E}_{i\ne j}\cos(M_i, M_j)$ | → 1 |
| 注入幅度比 | $\text{RMS}(\Delta_{\text{inj}}) / \text{RMS}(X_{\text{host}})$ | ≪ 1（被 LayerNorm 洗掉）或 ≫ 1（盖过真实输入） |
| 门控开启率 | 硬二值化后 gate=1 的层比例 | → 0（模型学会忽略通道） |

`【上游】`【论文，2602.15382 源码】Vision Wormhole 把幅度对齐直接写成损失项 $\lambda_{rms}(\text{RMS}(\Delta_{inj}) - \text{RMS}(\bar X_{img}))^2$，$\lambda_{rms}=0.1$ —— 这说明幅度失配在实践中是真问题，值得单独监控。
`【上游】`【论文，2511.09149】Interlat 给"模型学会直接忽略 latent 通信"起了名字：**shortcut behavior**，并用 $\mathcal{L}_{sep}$（错配 latent 的加权 JS 散度）专门防它。门控开启率就是它的免费监控版。

### L1.5 L1 需要的数据

`【设计】` 一批**带已知属性标签**的样本。若没有现成标注，用 LatentQA 式的合成控制：【论文，2412.08686，`【上游】`】在 stimuli 前 prepend "controls"（规定期望输出属性的片段）→ 用这些 prompt 喂模型拿 activation → 围绕这些 controls 自动生成 QA 对。**精髓是：你人为知道这个 hidden 里该有什么，因为是你亲手放进去的。** 造诊断集比造训练集划算。

---

## L2. 协作层：核心战场

### L2.1 五指标 + 两条恒等式（直接抄 2607.26773）

【论文，2607.26773，`【上游】`】记号：$\bar U_a = \mathbb{E}_e[U(Y_a)]$（$U$ = exact-match 或任务分），$\bar D_{a,b} = \mathbb{E}_e[D(P_a,P_b)]$（$D$ = Jensen–Shannon 散度）。

| 指标 | 全称 | 估计量 | 回答什么 |
|---|---|---|---|
| **PS** | Positive Signaling | $I(M_{cur};X)$ | 消息里编码了 sender 的信息吗 |
| **PL** | Positive Listening | $\bar D_{cur,0}$ | receiver 对"有消息"有反应吗 |
| **CIC** | Causal Influence of Communication | $\bar D_{cur,oth}$ | receiver 对**内容**有反应吗（预测层面） |
| **CAG** | Content-Attributable Gain | $\bar U_{cur} - \bar U_{oth}$ | **内容带来任务价值吗（主 endpoint）** |
| **SSG** | Self-Substitution Gap | $\bar U_{cur} - \bar U_{self}$ | 价值必须来自另一个 agent 吗 |

派生量：$\text{OPE} = \bar U_{cur} - \bar U_0$，$\text{OME} = \bar U_{oth} - \bar U_0$，$\text{DSC} = \bar U_{self} - \bar U_{oth}$，$\text{BME} = \bar U_{oth} - \bar U_{xbench}$。

**两条精确分解恒等式（全文最值钱的东西）：**

$$\boxed{\text{OPE} = \text{OME} + \text{CAG}}\qquad\qquad \boxed{\text{CAG} = \text{DSC} + \text{SSG}}$$

`【设计】` 这两条是**恒等的**（代入定义即得），不是经验发现 —— 所以**免费**。你只要多跑三组条件（$M_0, M_{oth}, M_{self}$），就能把论文里那个单一的 ΔAccuracy 拆成三块有独立解释的量。审稿成本极低，说服力提升极大。

**逐层定位工具 NLD**（Normalized Layer Difference）【论文，2607.26773】：

$$\text{NLD}(C) = \frac{\text{LD}_{\text{rest}}(C) - \text{LD}_{oth}}{\text{LD}_{cur} - \text{LD}_{oth}},\qquad \text{LD} = \text{teacher-forced 答案对数概率}$$

`【设计】` 直接拿来回答「该训哪几层」：只恢复层集合 $C$ 时恢复了多少比例的通道价值。**这对你尤其有用**，因为 `【上游】`【论文，2606.23581 Kamera】发现 KV 重用丢掉的信息**低秩（rank≈32 够）且集中在深层**，`【上游】`【论文，2506.19209 SDE】发现改所有层会显著掉点、只能改 top-1~3 层。NLD 让你用实测而不是拍脑袋决定作用层。

### L2.2 我给 VLM + 训练路线补的三个指标

`【设计】` 上面五个指标是通用的。你的场景有三个它们盖不住的洞：

**① VSG — Visual-Specific Gain（视觉专属增益）**

$$\text{VSG} = \bar U_{cur} - \bar U_{capt}$$

其中 $M_{capt}$ = sender 把自己看到的写成自然语言 caption 走文本通道传过去。

**这是你整篇论文的存在性证明。** 原设想的 $W_a$ 理论断点（$W_a$ 只对文本词表成立，视觉输入不过 $W_{in}$）在训练路线下被绕开了，但它变形成了一个**评估问题**：你必须证明通道传的是**文本传不了的视觉证据**。VSG ≤ 0 → 那就传文本，还可读可审计可 debug，没人需要你的模块。

`【上游】`Vision Wormhole (2602.15382) v2 Table 1：GSM8K 文本 MAS 80.8%(27.3s) vs VW 76.2%(26.7s)，**−4.6pp、1.02× 加速**。这就是 VSG < 0 的实例，而且它是这条路线的**结构性上限**——用文本通道当教师做蒸馏，最好情况是打平。

**② TFG — Training Gain over training-free**

$$\text{TFG} = \bar U_{cur} - \bar U_{tf}$$

$M_{tf}$ = 免训练 latent 通道（LatentMAS 原版 / Procrustes 闭式对齐）。**这是"训练算法好坏"最直接的落点**——你的模块必须打赢一个 SVD 闭式解。

`【上游】`【论文，2311.00664, NeurIPS'23】警告：在有平行锚点的前提下，闭式 Procrustes（`ortho`）**全面吊打**相对表示，CIFAR10 上 0.93 vs 0.80（no-stitch 上界 0.95）。别以为免训练基线很弱。

**③ AIG — Asymmetry-Induced Gain（信息不对称诱导增益）**

$$\text{AIG} = (\bar U_{cur} - \bar U_0)\big|_{\text{信息不对称}} - (\bar U_{cur} - \bar U_0)\big|_{\text{信息对称}}$$

`【设计】` 同一套题，跑两个版本：(a) sender 和 receiver 各看一半信息；(b) 两个 agent 都看全部信息。真通信应该在 (a) 上增益大得多。**若 AIG ≈ 0，你测到的不是"通信"，是"多跑了一遍推理"或"prompt 变了"。**

可用现成基准：
- **HiddenBench** 【论文，2505.11556，本次核实】65 个任务，基于 Hidden Profile 范式。关键数字：多智能体在信息分布式条件下 **30.1%**，单智能体给全信息 **80.7%**。这个 50pp 的缺口就是你的模块要填的空间，也是最理想的实验床。
- Sotopia-ToM 【2605.02307，`【本次核实·仅摘要】`】160 个人工审核场景、3–5 agent、public/private 双通道
- CalBench 【2605.09823，`【本次核实·仅摘要】`】
- VLM 侧：VS-Bench (2506.02387, CVPR'26 Oral)、AirCopBench (AAAI)、VIEW2SPACE (2603.16506) `【未核实·仅检索摘要】`

### L2.3 一个实操坑：JSD 算在哪

【论文，2607.26773】原文在**首个 completion token** 上算 JSD。

`【设计】` **thinking mode 下这会低估**，因为首 token 几乎恒定是 `<think>` 一类。建议同时报两个口径：(a) 首 token JSD（与原文可比）；(b) teacher-forced 全答案序列的**逐 token JSD 均值**。两者背离本身就是信息。**口径必须预注册，不可事后挑。**

---

## L3. 任务层

| 指标 | 怎么测 | 坑 |
|---|---|---|
| 准确率 | exact-match 优先；用 LLM judge 必须预注册 judge 模型且固定版本 | `【上游·你的 MEMORY】` judge 漂移可达 0.236，评委必须统一 |
| Token 节省 | 分开报 prefill / decode / 通信 token；**latent token 也要折算** | 只报"文本 token 少了"是耍流氓 |
| 时延 | 同一 serving 栈、同一 batch size、warmup 后、≥20 次取中位数与 p95 | Vision Wormhole 实测只有 **1.02×** `【上游】`——加速可能几乎不存在 |
| **Pareto** | 准确率 vs 总 FLOPs（或总 token），画曲线不画点 | 单点比较无法排除"多算了一遍" |
| 恢复率 | $\text{Recovery} = (\bar U_{cur} - \bar U_0)/(\bar U_{ub} - \bar U_0)$ | 定义你离上界多远，比裸准确率好读 |

---

## 2. 必备对照组清单

`【设计】`（第 1/3/5/7/9 条来自【论文，2607.26773】，其余是我补的）

| # | 对照组 | 构造 | 排除的替代解释 | 成本 |
|---|---|---|---|---|
| 1 | **$M_0$ 零消息** | 不通信 | 基线 | 低 |
| 2 | **$M_{rand}$ 随机消息** | 高斯噪声，**匹配一阶二阶统计与形状** | "注入任何东西都会扰动 receiver" | 低 |
| 3 | **$M_{oth}$ 置换消息** | 他样本消息，同 benchmark，长度匹配，每题 K=4 条 | **"有消息就好"（触发器效应）** ← 最重要 | 中 |
| 4 | **$M_{shuf}$ 内部打乱** | 同一条消息，token 顺序 / 层顺序打乱 | "只用了消息的聚合统计/边缘分布" | 低 |
| 5 | **$M_{self}$ 自生成** | receiver 自己走同一接口，**FLOPs 配平** | **"增益来自额外算力而非通信"** | 中 |
| 6 | **$M_{text}$ 文本消息** | sender 写自然语言 | "latent 并不比文本强" | 中 |
| 7 | **$M_{tf}$ 免训练 latent** | LatentMAS / Procrustes 闭式 | **"增益来自通道存在，而不是你的训练"** | 中 |
| 8 | **$M_{ub}$ 上界** | 单模型看全部信息 | 定天花板，算 recovery ratio | 低 |
| 9 | **$M_{xbench}$ 跨域消息** | 另一数据集的长度匹配消息 | 诊断"同域"本身值多少（BME） | 低 |
| 10 | **$M_{imgswap}$ 换图不换文** 🔴 | sender 的**图**换成无关图，文本上下文不变 | **"视觉证据根本没过去"** ← VLM 专属杀手锏 | 低 |
| 11 | **$M_{capt}$ caption 通道** 🔴 | sender 所见写成 caption 走文本 | **"文本能传的东西你用 latent 传"** | 中 |
| 12 | **$M_{randsender}$ 无关 sender** | sender 的输入整体换成无关样本 | **"模块自己在做题，不是在通信"** ← 上界蒸馏专属 | 低 |

**长度匹配是硬要求** 【论文，2607.26773】—— 否则你测到的是序列长度差异不是内容差异。`【设计】` 原文只写 "approximately length-matched" 没给容差，我建议：token 数完全相等（padding/truncate 到中位数长度），并用 H5 的 KS 检验验证 norm 分布不可区分。

**#10 和 #12 是我认为最被低估的两个** `【设计】`：
- #10 一次前向就能跑，却能直接判定你的 VLM 故事成不成立。`【上游】`确认 Vision Wormhole **没有做任何通道级对照**，视觉专属对照组目前是空白。
- #12 针对上界蒸馏路线（`【上游】`推荐的主监督）的特有塌陷：模块被蒸馏成一个小 solver 或通用先验注入器。若 sender 输入完全无关时 receiver 仍提升，那你训的是 prompt 增强不是通信。

---

## 3. 训练配方好坏的判别标准

### 3.1 记分卡（预注册，按优先级）

| 级别 | 量 | 说明 |
|---|---|---|
| **门槛（不过不比较）** | L0 六项健全性全过 | 有 bug 就不是配方差异 |
| **主 endpoint（唯一）** | **CAG** | 内容归因增益。单一预注册，避免多重比较 |
| 次要 1 | **SSG** | 是否真需要另一个 agent |
| 次要 2 | **VSG** | 视觉专属增益（VLM 必报） |
| 次要 3 | **TFG** | 是否打赢免训练闭式解 |
| 稳健性 1 | **CAG 的跨训练 seed 方差** | ≥5 seed |
| 稳健性 2 | **迁移矩阵**：3 sender × 3 receiver | 是否只在特定组合上成立 |
| 效率 1 | **CAG 的样本效率曲线**：CAG@100 / @500 / @2k / @10k 步 | **配方好坏最有区分度的轴** |
| 效率 2 | **训练稳定性**：每千步 non-finite loss 次数、是否需要 clip 补丁 | `【上游】`Vision Wormhole 源码里的 clip 补丁密度本身就是"这个目标函数很脆"的证据 |
| 成本 | 参数量（硬红线 ≤ receiver 的 1%）、GPU·小时 | `【上游】`C2C Fuser 478M vs "直接微调 receiver" 596M —— 两者已逼近，审稿人必问 |

`【设计】` **为什么样本效率曲线是最有区分度的轴**：`【上游】`Vision Wormhole 只用 **400 步、batch 2、共 800 次 anchor 抽样**（对 3000 条 anchor 只有 0.27× 覆盖），C2C 用 **50 万样本 × 1 epoch、有效 batch 256、8 卡**，L²-VMAS 用 **230k PPO 步 × 8×H200**。跨了四个数量级。**终点分数差不多的两个配方，样本效率差 100 倍时，谁好是显然的，而只报终点你看不见。**

### 3.2 统计判定

**（a）题目级：配对设计 + McNemar**

`【设计】` 同题、同 seed、**只换 message**，逐题记录二值正确性。配对差 $d_i \in \{-1,0,+1\}$。

- 正确检验是 **McNemar 精确检验**（只用 discordant pairs）。【本次核实】实践中 **discordant pairs > 25 用卡方版，否则用精确版**。
- **报告里必须给出 discordant pair 数量。** `【设计】` power 完全由它决定；**discordant < 20 时任何结论不可信**，这是最便宜的 sanity check。
- 效应量区间用 **paired bootstrap**（10k 次，按题重采样）。
- 【本次核实】配对设计所需样本量中位数比非配对公式小 **2.15 倍** —— 这就是为什么绝不能报两组独立均值之差。

**（b）样本量：我算给你** `【计算】`

设 discordant rate $p_+ + p_- = 0.20$，效应 $\delta = p_+ - p_-$，$\text{Var}(d) = (p_+ + p_-) - \delta^2$，$n = (z_{0.975}+z_{0.80})^2 \sigma_d^2/\delta^2 = 7.849\,\sigma_d^2/\delta^2$：

| 想检出的效应 | 需要题数 |
|---|---|
| 5 pp | **≈ 620 题** |
| 10 pp | **≈ 150 题** |
| n = 60（2607.26773 的规模） | 只能检出 **≈ 16 pp** |

`【设计】` 这解释了为什么 `【上游】`报告 2607.26773 的多个置信区间跨零（数据集仅 40–100 题）。**你的主实验最少 500 题，别用 60。** 快评可以用 150–200 题，但只能用来看符号、不能下结论。

**（c）训练随机性：≥5 seed + ASO**

`【设计】` **5 个训练 seed，不是 3。** 理由：`【上游】`【论文，2505.12540 vec2vec】同族模型 15 个种子里 14 个收敛，**异族模型只有 3/15 收敛** —— 训练式跨模型通道的 seed 方差极大，3 个 seed 看不出配方差异，只会看到运气。

比较两个配方跨 seed 的分数分布，用 **ASO（Almost Stochastic Order）**【论文，Dror et al., ACL 2019, aclanthology P19-1266，本次核实】，现成实现 `Kaleidophon/deep-significance`（pip `deepsig`）【本次核实】：

- 返回 $\varepsilon_{\min}$ = A 距离"显著优于 B"还有多远
- $\varepsilon_{\min} < 0.5$ → A 随机占优 B 的情况更多
- `【设计】` **实操阈值：$\varepsilon_{\min} \le 0.2$ 才敢写"稳定优于"**，0.2–0.5 写"略优"，> 0.5 写"不可区分"

**（d）多重比较**

`【设计】` 你会同时看 10 个指标。**预注册主 endpoint 只有 CAG 一个**，其余全部标为探索性（exploratory），或对次要指标做 Holm–Bonferroni 校正。这一条写进论文的 Experimental Setup，能挡掉一整类审稿意见。

### 3.3 配方 A vs 配方 B 的决策树 `【设计】`

```
L0 健全性任一不过 ────────────────► 修 bug，禁止比较
        │过
        ▼
ASO(CAG_A, CAG_B), ε_min > 0.5 ───► 不可区分 → 转看样本效率 + 成本 + 稳定性
        │ε_min ≤ 0.2
        ▼
CAG_A > CAG_B 且 SSG_A > 0 ───────► A 胜（真通信）
        │
CAG 相当，但 OME_A ≫ OME_B ───────► B 胜（A 的增益更依赖触发效应，换 receiver 就没了）
        │
CAG 相当，CAG_A@500步 ≫ B ────────► A 胜（样本效率）
        │
CAG_A > CAG_B 但 VSG_A ≤ 0 ───────► 两个都不该发（VLM 故事不成立）
```

---

## 4. 早期信号：几百步就能杀掉一个坏配方

`【设计】` 这一节是全文最省算力的部分。全部在训练循环内，额外成本 ≈ 一次 forward。

### 🥇 一号信号：错配消息 loss gap

$$\Delta L = \mathcal{L}\big(y \mid M_{oth}\big) - \mathcal{L}\big(y \mid M_{cur}\big)$$

每 N 步（比如 50 步）在同一个 held-out mini-batch 上多跑一次 forward，把消息换成错配的，记录 loss 差。

**为什么这是最好的早期信号：**
- 它是 **CAG 在 loss 层面的免费版本** —— CAG 要跑完整生成，$\Delta L$ 只要一次 teacher-forced forward
- 它直接测"模型有没有在读内容"，而 task loss 下降只说明"模型在学任务"
- `【上游】`【论文，2511.09149 Interlat】的 $\mathcal{L}_{sep}$（匹配/错配 latent 的加权 JS 散度）本质上就是把这个量当损失优化；去掉它，模型出现 "shortcut behavior，学会直接忽略 latent 通信"。**你即使不把它当损失，也必须把它当监控。**

**判决规则** `【设计】`：
- $\Delta L$ 在 500 步内**没有单调上升** → 杀掉配方，不用等下游
- $\Delta L \approx 0$ 而 task loss 降得很好 → **这是最危险的组合**：配方已经死了（训成了触发器），但所有常规指标看起来都在变好

### 🥈 二号信号：teacher-forced 答案 logprob gap

$$\text{LD}_{cur} - \text{LD}_{oth} > 0\ \text{且随步数增长}$$

`【设计】` 这是 NLD 的分母，也是 CAG 的连续版代理。比 $\Delta L$ 更贴近最终任务（因为只看答案 token 而不是全序列），但同样便宜。

### 🥉 三号信号：几何/门控四件套（L1.4）

- 有效秩 PR 塌到 1 → 消息坍缩，杀
- batch 内平均 cosine > 0.9 → 同上
- 门控开启率 → 0 → 模型在学 shortcut，杀
- 注入幅度比 $\text{RMS}(\Delta)/\text{RMS}(\text{host})$ 跑到 ≪0.05 或 ≫5 → 幅度失配，加 RMS 正则项

### 早期信号时间表 `【设计】`

| 训练步 | 跑什么 | 成本 | 杀掉的条件 |
|---|---|---|---|
| 0（训练前） | L0 六项健全性 | <1h | 任一不过 |
| 每 50 步 | $\Delta L$、LD gap、几何四件套 | ≈0 | PR→1 或 gate→0 |
| 500 | $\Delta L$ 趋势 | ≈0 | 非单调上升 |
| 500 | 消息探针 selectivity（线性探针，秒级） | 分钟 | selectivity ≈ 0 |
| 1000–2000 | 200 题 L2 快评（$M_0/M_{cur}/M_{oth}$） | 3–4h | CAG 点估计为负 |
| 1000–2000 | $M_{imgswap}$ 对照 | +1h | CAG 不变（视觉没过去） |

---

## 5. 反作弊检查：latent 通道最容易出的 12 种假阳性

`【设计】`（除标注外均为我的设计；机制来源标在最后一列）

| # | 假阳性 | 长什么样 | 检测方法 | 来源 |
|---|---|---|---|---|
| 1 | **触发器效应**：消息没被读，只是"有消息"就切换了推理模式 | OME 大、CAG ≈ 0 | 跑 $M_{oth}$，报 OPE=OME+CAG 分解 | 【2607.26773】 |
| 2 | **门其实关着** | 硬二值化后 gate 全 0，性能不变 | 报门控开启率；消息置零后 accuracy 不掉 | `【上游】`C2C 源码有被注释掉的门稀疏正则 |
| 3 | **增益来自 prompt / 形状变化** | $M_{rand}$ 也能涨 | $M_{rand}$ 组；H5 形状 KS 检验 | 设计 |
| 4 | **增益来自额外算力** | SSG ≈ 0 或跨零 | $M_{self}$ + FLOPs 配平（记录两条路径实际 prefill+decode token 与 FLOPs） | 【2607.26773】GSM8K/Qwen3-4B 实测 CAG=+5.17pp 但 SSG=−2.00pp 跨零 |
| 5 | **只用了聚合统计，没用结构** | $M_{shuf}$ 不掉分 | 打乱消息内部 token / 层顺序 | 设计 |
| 6 | **视觉信息根本没过去**（VLM 一号坑） | $M_{imgswap}$ 下 CAG 不变 | 换图不换文 | 设计（$W_a$ 断点的评估落点） |
| 7 | **latent 不如文本** | VSG ≤ 0 | $M_{capt}$ 对照 | `【上游】`VW GSM8K −4.6pp 实例 |
| 8 | **模块自己在做题**（上界蒸馏专属） | $M_{randsender}$ 下仍提升 | sender 输入整体换无关样本 | 设计 |
| 9 | **答案搬运而非证据传递** | 线性探针能从消息高 selectivity 解出标准答案 | L1 探针目标设为 gold answer | 设计；上界蒸馏时 teacher 见过答案，风险高 |
| 10 | **采样噪声冒充效应** | CIC 与 H4 噪声地板同量级 | H4 地板；主实验 temperature=0 | 设计（补 2607.26773 未定义的 receiver-instability） |
| 11 | **train/test 门不一致** | 软 gate 推理与硬 gate 推理数字差很多 | 两种都报 | `【上游】`C2C 训练软 gate、推理硬二值 |
| 12 | **只在特定模型对上成立** | 换 sender/receiver 就崩 | 3×3 迁移矩阵 | `【上游】`vec2vec 异族仅 3/15 seed 收敛 |

**另外两个 PS 相关的警告** 【论文，1903.05168, Lowe et al., AAMAS'19，`【上游】`】：
- **SC/PS 高 ≠ 有通信**。原文展示：**即使消息在被观察前就被打乱（scrambled），agent 依然表现出高 SC** —— 因为动作头与通信头共享网络特征，**即使通信参数完全没训练**，消息也会和动作相关。
- `【设计】` **对你是直接警告**：你在**训**一个通信模块，sender 侧表征天然与 sender 的答案相关，**你的 PS 一定会很好看，而它什么也不证明**。**不要把 PS 当卖点。** 报 PS 时必须给 label permutation 参照。
- **IC 会假阴性**：不 condition on 环境上下文的观测型指标不可信。

---

## 6. 实验矩阵表（按性价比排序）

`【设计】` 成本以「1 卡·小时」为单位，假设 7B 级 VLM、vLLM serving。

| # | 实验 | 目的 | 输入 / 条件 | 期望观测 | 判决标准 | 成本 |
|---|---|---|---|---|---|---|
| **E0** | **健全性套件** | 证明注入代码没写错 | 20 题 × {H1…H6} | H2 精确为 0；H3 NLD=1；H4 给出噪声地板 | **任一不过 → 停工修 bug** | **< 1 卡·小时** ✅ |
| E1 | 错配 loss gap 监控 | 一号早期信号 | 训练内嵌，每 50 步 held-out batch | $\Delta L$ 单调上升 | 500 步内非单调 → 杀配方 | ≈ 0（内嵌） |
| E2 | 几何 / 门控监控 | 防坍缩 | 训练内嵌，每步 | PR 不塌、cos<0.9、gate 开启率>0.2 | 任一越界 → 杀 | ≈ 0（内嵌） |
| E3 | 探针 + 控制任务 | L1 可解码性 | 2k 条消息 + 属性标签 + 随机标签 | selectivity > 0 且 MDL 压缩比 > 1 | selectivity≈0 → 消息里没东西 | 0.5 |
| E4 | **200 题三条件快评** | CAG 符号 | $\{M_0, M_{cur}, M_{oth}\}$ × 200 题 | CAG 点估计为正 | CAG < 0 → 杀；discordant<20 → 加题 | 3 |
| E5 | **$M_{imgswap}$ 换图对照** 🔴 | 视觉证据是否过去 | 同 E4 题目，sender 换图 | CAG 显著下降 | **CAG 不变 → VLM 故事不成立** | 1 |
| E6 | $M_{randsender}$ 对照 | 模块是否自己做题 | sender 输入换无关样本 | 增益消失 | 仍提升 → 不是通信 | 1 |
| E7 | $M_{self}$ 算力配平 | 排除额外算力 | receiver 自生成，FLOPs 对齐 | SSG > 0 | SSG 跨零 → 只能写成 test-time compute 分配器 | 3 |
| E8 | $M_{capt}$ 文本通道 | VSG | sender caption → 文本 | VSG > 0 | **VSG ≤ 0 → 不该发** | 4 |
| E9 | $M_{tf}$ 免训练基线 | TFG | LatentMAS / Procrustes 闭式 | TFG > 0 | TFG ≤ 0 → 训练没有价值 | 4 |
| E10 | $M_{rand}$ + $M_{shuf}$ | 排除扰动/聚合统计 | 噪声 & 打乱消息 | 两者都接近 $M_0$ | 接近 $M_{cur}$ → 通道是幌子 | 2 |
| E11 | **全量五条件 × 5 seed** | 主结果 | $\{M_0,M_{cur},M_{oth},M_{self},M_{xbench}\}$ × 640 题 × 5 seed | CAG 区间不跨零；OPE=OME+CAG 对得上 | McNemar p<0.05 + bootstrap CI 不跨零 | 60 |
| E12 | $M_{ub}$ 上界 + recovery | 定天花板 | 单模型看全信息 | recovery ratio | 报告用，无杀 | 3 |
| E13 | 带宽 / 秩曲线 | 带宽有没有被用起来 | K ∈ {1,4,8,16,32,64} 或 rank ∈ {8,16,32,64} | CAG 随 K 上升后饱和 | **K=1 就饱和 → 没在传细粒度信息** | 20 |
| E14 | 逐层 NLD 定位 | 该训哪几层 | 逐层/逐组恢复 | NLD 曲线，深层集中 | 指导下一轮设计 | 15 |
| E15 | AIG：对称 vs 不对称 | 是否真的是"通信" | HiddenBench 式两版本 | AIG > 0 显著 | AIG≈0 → 不是通信 | 20 |
| E16 | 3×3 迁移矩阵 | 鲁棒性 | 3 sender × 3 receiver | 至少 6/9 组合 CAG>0 | < 4/9 → 过拟合到模型对 | 40 |
| E17 | 样本效率曲线 | 配方比较主轴 | CAG@{100,500,2k,10k} 步 × 2 配方 | 曲线分离 | 用于 A/B 判决 | 复用 ckpt |
| E18 | L3 Pareto | 值不值 | 准确率 vs FLOPs / token / p95 时延 | Pareto 前沿 | 不在前沿 → 只能当 analysis | 10 |
| E19 | ASO 配方 A/B | 最终判决 | 5 seed × 2 配方的 CAG 分布 | $\varepsilon_{\min}$ | ≤0.2 = 稳定优；>0.5 = 不可区分 | 复用 |

### 裁剪建议 `【设计】`

- **只有 1 张卡 / 3 天** → E0 → E1/E2（内嵌）→ E4 → E5 → E8。这五个决定了"要不要继续做"。特别是 **E5 和 E8 都是杀手锏且都很便宜**。
- **投稿最小集** → E0 + E11（主表）+ E5/E6/E7/E8/E9/E10（对照消融表）+ E13（带宽曲线）+ E18（Pareto）。E14/E15/E16 是加分项。
- **绝对不能省** → E0（否则数字全是噪声）、E7（$M_{self}$，审稿人一定会问）、E8（$M_{capt}$，VLM 论文的存在性证明）。

---

## 7. 现成实现清单

| 用途 | 仓库 / 包 | 状态 |
|---|---|---|
| ASO 显著性检验 | `Kaleidophon/deep-significance`，pip `deepsig` | 【本次核实】 |
| MDL 探针 | `lena-voita/description-length-probing` | 【本次核实】 |
| 控制任务探针 | 【论文 1909.03368】方法简单，自己实现即可 | 【本次核实】 |
| PS/PL/CIC 参考实现 | `facebookresearch/measuring-emergent-comm`、`facebookresearch/egg`、`olipinski/emlangkit`、`Near32/ReferentialGym` | `【上游】` |
| 免训练对齐基线 | `lucmos/relreps`（相对表示）；Procrustes 用 `scipy.linalg.orthogonal_procrustes` | `【上游】` |
| 协同感知血脉基线 | `MediaBrain-SJTU/Where2comm`、`ai4ce/DiscoNet` | `【上游】` |
| 信息不对称任务 | HiddenBench【2505.11556，本次核实】、Sotopia-ToM【2605.02307】、CalBench【2605.09823】 | 部分仅摘要 |
| activation patching 实操 | 【论文 2404.15255, Heimersheim & Nanda】"How to use and interpret activation patching" | 【本次核实】 |
| **五指标本体** | **2607.26773 全文未提供任何 GitHub 链接** | `【上游】`逐节核查 |
| 本地已有 | `/home/yilin/tmp/mm-latent-repos/heterogeneous-latent-mas`（Vision Wormhole 官方码）、`/home/yilin/tmp/mm-latent-repos/ViF`、`/home/yilin/tmp/mm-latent-repos/VisMem` | 已核实存在 |

---

## 8. 三句话总结

1. **换主判据**：ΔAccuracy 会被"触发器效应"污染到符号都翻转（实测 −1.00pp = −6.17 + 5.17，以及 +1.67pp = +3.96 − 2.29）。预注册 **CAG** 为唯一主 endpoint，用 OPE=OME+CAG、CAG=DSC+SSG 两条**免费的恒等式**把单一数字拆成三块可解释的量。

2. **早杀**：训练循环内每 50 步测一次**错配消息 loss gap**，500 步不单调上升就杀掉配方 —— 这是全套协议里唯一零成本却能省掉 90% 算力的东西。

3. **VLM 的存在性证明只有两条**：$M_{imgswap}$（换图不换文，CAG 必须掉）和 $M_{capt}$（VSG = 你的通道减去 caption 通道，必须 > 0）。两个都便宜，两个目前都没人做，两个都能一击判死。样本量按我算的来：**640 题检 5pp，60 题只能检 16pp**，别用 60。

---

**Sources**（本次会话新核实的）
- [Designing and Interpreting Probes with Control Tasks (ACL Anthology)](https://aclanthology.org/D19-1275/)
- [Information-Theoretic Probing with Minimum Description Length (ACL Anthology)](https://aclanthology.org/2020.emnlp-main.14/)
- [lena-voita/description-length-probing (GitHub)](https://github.com/lena-voita/description-length-probing)
- [Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals (TACL)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00359/98091/Amnesic-Probing-Behavioral-Explanation-with)
- [Deep Dominance — How to Properly Compare Deep Neural Models (ACL Anthology)](https://aclanthology.org/P19-1266/)
- [Kaleidophon/deep-significance (GitHub)](https://github.com/Kaleidophon/deep-significance)
- [Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs / HiddenBench (arXiv:2505.11556)](https://arxiv.org/abs/2505.11556)
- [Sotopia-ToM (arXiv:2605.02307)](https://arxiv.org/html/2605.02307)
- [CalBench (arXiv:2605.09823)](https://arxiv.org/pdf/2605.09823)
- [How to use and interpret activation patching (arXiv:2404.15255)](https://arxiv.org/pdf/2404.15255)
- [VS-Bench (arXiv:2506.02387)](https://arxiv.org/html/2506.02387)
- [Exact McNemar's Test and Matching Confidence Intervals (CRAN exact2x2)](https://cran.r-project.org/web/packages/exact2x2/vignettes/exactMcNemar.pdf)
- [Resolution Diagnostics for Paired LLM Evaluation (arXiv:2605.30315)](https://arxiv.org/pdf/2605.30315)

---

## 附：关键发现速览

1. **主判据必须换掉**：不能用 ΔAccuracy（OPE）判断训练配方好坏，因为它可以由方向相反的两块合成——2607.26773 实测 GSM8K/Qwen3-4B 上 OPE=−1.00pp 却 = OME(−6.17) + CAG(+5.17)，而 Qwen3-8B 上 OPE=+1.67pp 却 = OME(+3.96) + CAG(−2.29)（内容有害但总分为正）。协议的**主 endpoint 应预注册为 CAG（内容归因增益）= Ū_cur − Ū_oth**，OPE 只作报告项。训练路线尤其危险：端到端 CE 的最易下降方向就是把模块训成"触发器"而非"信道"，签名正是 OME 大而 CAG≈0。

2. **训练路线独有的一号早期信号是"错配消息的 loss gap" ΔL = L(M_oth) − L(M_cur)**，训练循环内每 N 步多跑一次 forward 即可，零额外成本，是 CAG 在 loss 层面的免费代理。判决规则：500 步内 ΔL 不单调上升 → 直接杀掉配方，不用等下游。配套三个防坍缩监控：消息有效秩 PR=(Σλ)²/Σλ²、batch 内平均 cosine（→1 即坍缩）、门控开启率（→0 即 Interlat 所说的 shortcut）。

3. **样本量是这类实验的生死线，我算过了**：配对二值结果、discordant rate 20% 时，检出 5pp 效应需 **≈620 题**，检出 10pp 需 **≈150 题**，而 n=60 只能检出 **16pp** —— 这正好解释了 2607.26773 为何多个区间跨零。正确检验是 **McNemar 精确检验**（discordant pairs >25 时用卡方版），power 只取决于 discordant pair 数量，**报告里必须给出 discordant 对数，<20 则任何结论不可信**。训练随机性用 ≥5 个 seed + ASO 检验（deep-significance / deepsig，Dror et al. ACL'19），ε_min ≤ 0.2 才算"稳定优于"。

4. **VLM 侧有两个别人都没做的致命对照，成本极低但一击致命**：M_imgswap（把 sender 的图换成无关图、文本上下文不变）—— 若 CAG 不变，说明通道传的全是文本可得信息，W_a 断点根本没被跨过；M_capt（sender 把所见写成 caption 走文本通道）—— 定义 **VSG = Ū_cur − Ū_capt**，VSG ≤ 0 则整个 latent 通道在 VLM 上没有存在理由（文本还可读可审计）。Vision Wormhole (2602.15382) 是最接近的工作但完全没有任何通道级指标，这是空档。

5. **L1 探针数字不做控制任务就是废数**：必须同时报 control task（随机标签）准确率并给出 selectivity（Hewitt & Liang, 1909.03368, EMNLP'19），更稳的做法是改报 MDL 描述长度（Voita & Titov, 2020.emnlp-main.14，代码 github.com/lena-voita/description-length-probing）。同时**绝不要报"消息里有几 bit"**——distribution-free MI 下界受 O(ln N) 限制、InfoNCE ≤ log K，只能报相对置换参照的可分性。
