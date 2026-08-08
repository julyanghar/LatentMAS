# 通道评估学：怎么判断一个 agent 间通信通道本身好不好

> 标注约定：**【原文】**=论文里写的（附 arXiv 号）；**【推断】**=我的判断，未经论文验证。两者严格不混。

---

## 0. 先立骨架：为什么"只看下游准确率"必然骗人

整个领域的逻辑可以压成**一条五级阶梯**，每一级都是上一级的严格加强，每一级都能被独立证伪。你设计实验的时候，本质上就是在爬这个阶梯：

| 级别 | 命题 | 反例（这一级能挡住什么） |
|---|---|---|
| L1 | 消息里**编码了**发送方的信息 | 编码了但接收方根本不看 |
| L2 | 接收方对消息的**存在**有反应 | 有反应但只是"被打断了"，换任何消息都一样 |
| L3 | 接收方对消息的**身份/内容**有反应 | 有反应但没变好，甚至变坏 |
| L4 | 内容带来了**任务价值** | 有价值，但接收方自己想一遍也能有 |
| L5 | 价值**必须来自另一个 agent** | —— |

**【原文，2607.26773】** 这篇把问题精确表述为 "capacity-versus-usage problem"（容量 vs 使用），原话是 "greater representational capacity does not establish that the receiver uses task-relevant information"。这句话就是整个评估学存在的理由。

**【原文，2607.26773】** 最硬的实证证据：Qwen3-4B 在 GSM8K 上，latent 通信相对无通信的**总效应是 −1.00pp**（看起来通道有害、或至少无用）。但分解后是：

```
OPE(总效应) −1.00pp = OME(消息存在效应) −6.17pp + CAG(内容归因增益) +5.17pp
```

也就是说：**这个通道的内容其实实实在在帮了 5.17pp，只是"塞任何消息进去"这个动作本身伤了 6.17pp。** 只报总效应的话，你会得出"这个通道没用"的结论，然后放弃一个其实内容有效、只是注入方式有问题的设计。反过来 Qwen3-8B 在 GSM8K 上总效应 +1.67pp，分解成 OME=+3.96pp + **CAG=−2.29pp**——内容其实是**有害**的，涨分全靠"有消息"这件事，这是更糟的情况，但总效应是正的，论文一律会写"我们的方法有效"。

**【原文，2607.26773】** 作者的结论句："similar overall performance can therefore arise from different communication mechanisms"，并主张 "controlled message comparisons" 应成为 latent 通信的**标准评估**。

**【推断】** 这两个方向相反的例子对你尤其关键：**训练一个通信模块最可能的失败模式，就是把它训成一个"触发器"而不是"信道"**——receiver 学会"只要有消息就切换到某个推理模式"，内容多少无所谓。这个失败在准确率上看起来像成功（OME 大），只有 CAG 分解能抓住它。training-free 的 LatentMAS 反而不太会有这个问题（它没有可被优化的自由度去学触发器），所以**你换成训练路线，就必须自带这套审计，否则审稿人会问而你答不出来**。

---

## 一、因果审计族（最直接可用，先抄这套）

### 1.1 论文本体

**【原文】** *Do Latent Channels Actually Communicate? A Causal Audit of Latent Multi-Agent LLM Communication*，Huixiang Zhang, Mahzabeen Emu，arXiv **2607.26773**。

**【原文，2607.26773】** 干预点定义得非常干净：所有替换都发生在 "after sender-side construction and before receiver-side injection"（发送方构造完成之后、接收方注入之前），因此对 **embedding / hidden state / KV cache 三种通道一视同仁**。这是它能当通用协议的原因——**你的 VLM 版本换了通道形态，这套审计一行不用改。**

### 1.2 四种受控消息替换（+ 一种诊断用）

**【原文，2607.26773】**

| 记号 | 名称 | 构造方式 | 控制掉了什么 |
|---|---|---|---|
| $M_0$ | 无消息 | $\varnothing$ | 基线 |
| $M_{\text{cur}}$ | 当前样本消息 | $M_{\text{cur}} = \phi_S(e)$ | —— |
| $M_{\text{oth}}$ | 他样本消息 | $M_{\text{oth}} = \phi_S(e')$，$e' \ne e$，同 benchmark、**长度匹配** | 保留消息的结构/统计，只换内容 |
| $M_{\text{self}}$ | 自生成消息 | $M_{\text{self}} = \phi_R^{(c^\star)}(e)$，receiver 自己走同一接口、**算力配平** | 控制掉"多算了一遍"的收益 |
| $M_{\text{xbench}}$ | 跨 benchmark 消息 | 另一数据集的长度匹配消息 | 诊断"同域"本身值多少 |

$M_{\text{oth}}$ 是整套方法的灵魂：**它让"有消息"和"有对的消息"这两件事第一次分开了。** 长度匹配是必要的，否则你测到的是序列长度差异而不是内容差异。

### 1.3 五指标 + 派生量

**【原文，2607.26773】** 记号：$\bar U_a = \mathbb{E}_e[U(Y_a)]$（$U$ 为 exact-match 准确率或声明的任务分），$\bar D_{a,b} = \mathbb{E}_e[D(P_a, P_b)]$（$D$ 为 **Jensen–Shannon 散度**，在**首个 completion token** 上算）。

| 指标 | 全称 | 估计量 | 对应阶梯 |
|---|---|---|---|
| **PS** | Positive Signaling | $I(M_{\text{cur}}; X)$，$X$ 主要取"发送方自己答对没" | L1 |
| **PL** | Positive Listening | $\bar D_{\text{cur},0}$ | L2 |
| **CIC** | Causal Influence of Communication | $\bar D_{\text{cur},\text{oth}}$ | L3（预测层面）|
| **CAG** | Content-Attributable Gain | $\bar U_{\text{cur}} - \bar U_{\text{oth}}$ | L4（任务层面）|
| **SSG** | Self-Substitution Gap | $\bar U_{\text{cur}} - \bar U_{\text{self}}$ | L5 |

派生量：

| 指标 | 估计量 |
|---|---|
| **OPE** (Overall Performance Effect) | $\bar U_{\text{cur}} - \bar U_0$ |
| **OME** (Other-Example Message Effect) | $\bar U_{\text{oth}} - \bar U_0$ |
| **BME** (Benchmark-Match Effect) | $\bar U_{\text{oth}} - \bar U_{\text{xbench}}$ |
| **DSC** (Derived Self-Generated Contrast) | $\bar U_{\text{self}} - \bar U_{\text{oth}}$ |

**【原文，2607.26773】** 两条**精确分解恒等式**（这是全文最值钱的东西）：

$$\text{OPE} = \text{OME} + \text{CAG}, \qquad \text{CAG} = \text{DSC} + \text{SSG}$$

**【推断】** 这两个恒等式是恒等的（代入定义即得），不是经验发现，所以**免费**——你只要多跑三组条件（$M_0, M_{\text{oth}}, M_{\text{self}}$），就能把你论文里那个单一的 Δ 准确率拆成三块有独立解释的量。审稿成本极低，说服力提升极大。

**【原文，2607.26773】** 还有一个恢复度指标 **NLD**（Normalized Layer Difference），用于层级/组件级定位：

$$\text{NLD}(C) = \frac{\text{LD}_{\text{rest}}(C) - \text{LD}_{\text{oth}}}{\text{LD}_{\text{cur}} - \text{LD}_{\text{oth}}}$$

其中 LD 是 **teacher-forced 答案对数概率之差**。含义：只恢复组件集合 $C$（比如某几层的 KV）时，恢复了多少比例的"当前样本消息"效果。$\text{NLD}=1$ 表示完全恢复。

**【推断】** NLD 是你做"逐层/逐 head 通道重要性分析"的现成工具。对你的场景特别有用：**如果你训一个模块只负责部分层，NLD 直接告诉你这些层承载了多少通道价值**，可以拿来做"该训哪几层"的经验依据，而不是拍脑袋。

### 1.4 三条硬健全性检查（必须先过，不过就是有 bug）

**【原文，2607.26773】** 原话："Three hard checks validate the implementation: masking the message must reduce identity effects to the numerical floor, replacing an other-example message with the current-example message must yield zero CIC and CAG, and full restoration must give NLD = 1."

翻译成可执行的三条：

1. **掩码检查**：把消息 mask 掉，身份类效应（CIC）必须掉到数值地板。
2. **自替换检查**：用 $M_{\text{cur}}$ 去"替换" $M_{\text{oth}}$ 的位置（即两边都是当前样本），CIC 和 CAG 必须**恰好为 0**。
3. **完全恢复检查**：恢复全部组件，NLD 必须 $=1$。

**【推断】** 这三条是我在整份材料里见到的**最好的工程实践**，强烈建议你直接抄进 CI。理由：latent 通道的注入代码极易写错（层错位、位置索引偏一、mask 没对齐），而这类 bug **不会崩溃**，只会让你测到一堆无意义的数字。第 2 条尤其毒——它是一个"本该平凡相等"的边界 case，任何注入路径的不对称都会在这里露馅。

### 1.5 统计与实验设置

**【原文，2607.26773】**
- 模型：Qwen3-4B、Qwen3-8B
- 数据：GSM8K（4B: 100 题 × 3 seed；8B: 60 题 × 2 seed）、ARC-Challenge（80×2 / 40×2）、MATH-500（60×1 / 40×1）
- 通道：LatentMAS 的 KV cache relay，latent 步数 $m=40$（从 $\{0,2,5,10,20,40\}$ 选出），nucleus sampling $T=0.6$、top-$p=0.95$，生成上限 2048 token，thinking mode 开
- $M_{\text{oth}}$ 每题独立分配 $K=4$ 条长度匹配消息
- 统计：任务类指标用 **paired bootstrap 置信区间** + **sign-flip / 精确配对检验**；PS 用 **label permutation** 作参照

**【原文，2607.26773】** 其他关键数字：
- MATH-500 / Qwen3-4B：OPE **+15.00pp** = OME +8.33pp + CAG **+6.67pp**（区间 [0.42, 12.50]pp，不跨零）
- MATH-500 / Qwen3-8B：OPE +10.00pp，几乎全部来自 OME +8.13pp；CAG 仅 +1.88pp 且**区间跨零**
- GSM8K / Qwen3-4B：**CAG = +5.17pp 但 SSG = −2.00pp（区间跨零）** —— 论文据此说 "example-specific content and other-agent value are distinct"

**【推断】** 最后这条对你至关重要：**内容有价值 ≠ 需要另一个 agent**。CAG 正、SSG 跨零，意思是"消息内容确实有用，但 receiver 自己花同样算力想一遍也能得到"。如果你的 VLM MAS 落在这个格子里，那你训的不是通信模块，是个**变相的 test-time compute 分配器**——这仍然可以是个贡献，但必须诚实地这么写，否则一个懂行的审稿人用 $M_{\text{self}}$ 就能把你打穿。

### 1.6 这套方法的软肋（你要补的）

**【原文，2607.26773】** 关于 PS 的估计，论文只说用 "a cross-fitted lower bound" 并与 permutation 参照对比，**没有指明估计器类型**（MINE / InfoNCE / 分箱 / 分类器均未说明）。作者自认："Distribution-free mutual-information lower bounds are limited by sample size"，并把结果解释为 "finite-sample lower bounds rather than direct measurements"。

**【原文，2607.26773】** 提到但**未给操作定义**的两项 validity check：**message-distribution similarity** 和 **receiver-instability**。$M_{\text{oth}}$ 的长度匹配只写了 "approximately length-matched"，无容差阈值；$M_{\text{self}}$ 的算力预算 $c^\star$ 未给具体值。

**【原文】** 我逐节核查全文，**未发现任何 GitHub / 代码仓库链接**。

**【推断】** 三个必须自己补的洞：

1. **receiver 噪声地板必须自己量**。同一条 $M_{\text{cur}}$ 重复跑 $K$ 次，算 $\bar D_{\text{cur},\text{cur}'}$ 作为自身基线。任何 $\bar D_{\text{cur},\text{oth}}$（即 CIC）**必须显著大于这个地板**才有意义，否则你在报告采样温度。论文提了 "receiver-instability" 但没给做法，这就是那个缺口。$T=0.6$ 的采样下这个地板不会小。
2. **样本量**。40–100 题去测几个 pp 的效应，多个区间跨零是必然的。配对设计（同题、同 seed、只换 message）是把方差压下来的唯一办法，**务必固定随机种子并逐题配对**，不要报两组独立均值之差。
3. **PS 的估计器要自己选并诚实标注**（见第三节，理论上限很低）。

---

## 二、涌现通信族（十年积累，指标最成熟，且是上面那篇的源头）

### 2.1 血脉关系

**【推断，但证据很强】** 2607.26773 的 PS / PL / CIC 三个名字**逐字**来自 Lowe et al. 2019。**这篇 2026 年的审计论文，本质上是把 2019 年 MARL 涌现通信的成熟指标搬进 LLM latent 通道，并补上了 CAG/SSG 两个任务价值层的新指标。** 这对你有两个直接含义：(a) 这套指标经过了十年的对抗性检验，可放心用；(b) 如果你要做**训练**版本的通道，涌现通信文献里训练场景的坑（下面 2.3）你会原样踩到，那边已经有答案了。

### 2.2 指标与公式

**【原文】** *On the Pitfalls of Measuring Emergent Communication*，Ryan Lowe, Jakob Foerster, Y-Lan Boureau, Joelle Pineau, Yann Dauphin，**AAMAS 2019**，arXiv **1903.05168**。代码：`github.com/facebookresearch/measuring-emergent-comm`。

**【原文，1903.05168】**

**Speaker Consistency (SC)** —— positive signaling：

$$SC = \sum_a \sum_m p(a,m)\log\frac{p(a,m)}{p(a)p(m)}$$

即 agent 的消息与其**自身后续动作**之间的互信息。

**Context Independence (CI)** —— positive signaling：

$$CI(p_{cm}, p_{mc}) = \frac{1}{|C|}\sum_c p_{mc}(m^c \mid c)\cdot p_{cm}(c \mid m^c)$$

衡量消息与任务概念的对齐（要求一符号一概念）。

**Instantaneous Coordination (IC)**：与 SC 同式，但 $p(a,m)$ 换成"一个 agent 的消息与**另一个** agent 的动作"跨 episode 共现平均。

**Causal Influence of Communication (CIC)** —— positive listening：概率取**反事实干预**语义，$p(a \mid m_j)$ 是强制发送 $m_j$ 时的动作概率，$p(a)$ 对所有可能消息边缘化；定义为 "measures the causal effect that one agent's message has on another agent's behaviour"。

> **注意**：我抓到的 CIC 具体表达式在 HTML 转换中疑似有损（形式上不像标准互信息），**建议你以原 PDF 为准**，不要直接照抄我这里的符号。语义（干预式互信息）是确定的。

### 2.3 它揭示的三个陷阱（每个都能直接毙掉一个错误实验设计）

**【原文，1903.05168】**

1. **SC 高 ≠ 有通信**。论文展示了：**即使消息在被观察前就被打乱（scrambled），agent 依然表现出高 SC**。原因是动作输出头与通信输出头**共享神经网络特征**，网络会按"打算做的动作"来分离表征——**即使通信参数完全没训练**，消息也会和动作相关。
2. **IC 会假阴性**。反例：收益矩阵随机变化时，agent 2 对 agent 1 的诚实消息做了最优响应，但因为跨 context 平均把依赖关系抹掉了，**IC = 0**。核心教训：**不 condition on 环境上下文的观测型指标不可信**。
3. **正确结论**："agents can exhibit positive signaling without positive listening"，且只有**干预式**测量能区分相关与因果。

**【推断】** 陷阱 1 对你是**致命的直接警告**：你打算**训**一个通信模块，那么 sender 侧的表征天然会和 sender 的答案相关——**你的 PS 一定会很好看，而它什么也不证明**。这就是为什么 2607.26773 坚持 PS 必须对 permutation 参照报告，而且为什么整套阶梯的重心在 CIC/CAG 而不是 PS。**不要把 PS 当卖点。**

### 2.4 组合性 / 结构指标：Topographic Similarity

**【原文，检索结果一致】** TopSim 是涌现通信里最广用的组合性指标：**计算输入空间距离与消息空间距离的 Spearman 相关**。高 TopSim 表示相似输入产生相似消息、不相似输入产生不相似消息。

**【推断】** 对连续 latent 通道，TopSim 是**唯一一个不需要 receiver 参与就能算的结构指标**，而且极便宜：取一批样本，算输入侧成对距离（VLM 场景可以用图像 embedding 或标注相似度）、算消息侧成对距离（KV/hidden 的 L2 或余弦），做 Spearman。它能证伪的假设是"**通道学到了一个结构保持的映射**"。低 TopSim + 高 CAG 是个有意思的组合（说明有用但不结构化）；高 TopSim + 低 CAG 说明结构保持了但 receiver 不吃这一套——后者提示问题在 receiver 侧对齐而非 sender 侧编码。

### 2.5 现成实现

**【原文，检索确认】**

| 仓库 | 内容 |
|---|---|
| `github.com/facebookresearch/measuring-emergent-comm` | Lowe et al. 2019 配套代码（SC / IC / CIC） |
| `github.com/facebookresearch/egg` | EGG，Emergence of lanGuage in Games，离散通道多 agent 游戏工具箱（EMNLP'19 demo, ACL Anthology D19-3010） |
| `github.com/olipinski/emlangkit` | 涌现语言分析工具箱；`emlangkit.metrics.topsim`；统一 `Language` 类接口，输入两个 numpy 数组（messages, observations） |
| `github.com/Near32/ReferentialGym` | Referential game 框架，含 TopSim 等组合性指标 |

**【推断】** `emlangkit` 的接口最适合你直接复用：它只要 (messages, observations) 两个数组，你把 latent 消息 flatten 成向量、把视觉输入的标注当 observation 就能跑 TopSim。EGG 是离散通道的，改造成本高，主要当参考读。

---

## 三、信息论族（**读完这节你会决定不做这一族的绝对量**）

### 3.1 那个诱人的"几万 bit"论证

**【原文】** *Beyond Tokens: A Unified Framework for Latent Communication in LLM-based Multi-Agent Systems*，Yingzhuo Liu 等（BUPT，`liuyingzhuo86@bupt.edu.cn`），arXiv **2606.05711**。配套 repo：`github.com/enochliu98/Awesome-Latent-Communication`。综述截止日 2026-07-15，覆盖 18 篇代表工作。

**【原文，2606.05711 §3.1.2】** 逐字引用：

> "$I(\mathbf{h}_{\text{context}}; y)$ is upper-bounded by $H(y) \le \log_2|\mathcal{V}| \approx 15\text{–}17$ bits"

论证：$\mathbb{R}^d$（$d \ge 4096$，32-bit 浮点）的 hidden state 承载 ~40,000 bits，离散化成 token 后压缩了 **$10^3$–$10^4$ 倍**。

**【原文，2606.05711】** 论文**自己承认**这是 *loose* upper bound，且**未提供严格的信道容量分析或 rate-distortion 理论**。§8.4 明确把严格信道容量、rate-distortion、带宽分析列为 **open problem**。它还给了一个三轴分类（WHAT: embeddings/hidden states/KV-caches/other；WHICH: latent 对齐 + 层对齐；HOW: concatenation/prepending/数学运算/cross-attention/cache restoration），但这是**组织性的，不是信息论的**，且**未提出任何新指标**。

**【推断】** 这个 "40000 bits vs 15 bits" 论证在 intro 里非常好用（它就是整个 latent communication 领域的 motivation），**但它是通道容量的上界，不是通道实际传输量的下界**。把它当 motivation 写，不要当结果报。两者混淆是这个领域最常见的修辞滑坡。

### 3.2 为什么"量出实际有几 bit"在理论上做不到

**【原文】** *Formal Limitations on the Measurement of Mutual Information*，David McAllester, Karl Stratos，**AISTATS 2020**，arXiv **1811.04251**。核心负面结果：**任何从 $N$ 个样本估出的 distribution-free 高置信互信息下界，都不可能大于 $O(\ln N)$**。具体地，Donsker–Varadhan 下界在计入基本统计考量后，**永远无法产生大于 $\ln N$ 的高置信值**。

**【原文】** *On Variational Bounds of Mutual Information*，Ben Poole 等，**ICML 2019**，arXiv **1905.06922**。**InfoNCE 上界为 $\log K$**（$K$ = 负样本数）；若真实 $I(X;Y) > \log K$ 则该界必然松。论文还证明现有变分下界在 MI 大时会退化为高偏差或高方差，并给出可插值的连续族。

**【推断】** 把数字代进去看看有多绝望：你跑 1000 个样本，$\ln 1000 \approx 6.9$ nats $\approx$ **10 bits**。也就是说，**即使你的 latent 消息真的携带 40000 bits，你用 1000 个样本能诚实报告的下界最多 10 bits。** 要诚实地报到 100 bits，你需要 $e^{100 \ln 2} \approx 10^{30}$ 个样本。这条路是封死的，不是工程问题。

**【推断】** 所以正确的做法有三条，按推荐度排序：

1. **相对量而非绝对量**（推荐，也是 2607.26773 的做法）：报 $\hat I$ 与 **permutation 参照**（把 $X$ 标签打乱后重估）的差，并给 permutation 分布的分位数。你声称的不是"有 $n$ bits"，而是"显著高于零信息基线"。这个声明**可以被证伪，也可以被诚实支持**。
2. **降维到任务相关的低基数变量**：取 $X$ = 二元的"sender 答对没"（$H(X) \le 1$ bit），或 $K$ 类标签（$H(X) \le \log K$）。此时真实 MI 本来就小，$O(\ln N)$ 天花板不再是瓶颈，估计可以是紧的。**这正是 2607.26773 取 $X$ = sender 答案正确性的原因**（我的推断，论文未解释此选择）。
3. **换成可解码性**（见第四节）：用探针准确率代替 bit 数。它是 MI 的一个有偏但**可复现、可比较**的代理。

### 3.3 Rate-distortion / 信息瓶颈：现实可行的那部分

**【原文，检索确认】** IB 是 rate-distortion 的推广（把失真约束换成"相关信息"约束）。语义通信领域已经有成熟的 rate–distortion–perception 框架（如 arXiv 2405.09995 *Semantic Communication via Rate Distortion Perception Bottleneck*、2509.10061 *Semantic Rate-Distortion Theory with Applications*）。

**【推断】** 你**不需要**做形式化的 RD 理论。你需要的是它的**经验版本**：**rate–utility 曲线**（见 §5.2）。这在 V2X 那边已经是标准做法六年了，而在 LLM latent communication 这边基本没人画（依据：2606.05711 把它列为 open problem）。**这是一个几乎白送的贡献点。**

---

## 四、表征分析族（探针 / 几何）

### 4.1 线性探针（可解码性）

**【原文】** 方法论范例：*Tool-Call Dependency Structure is Linearly Decodable in LLM Agent Residual Streams*，arXiv **2605.25310**。做法：用**低容量 logistic 探针**从**冻结的** residual stream 读取工具调用依赖图，发现依赖边**在第 14 层开始变得线性可解码**。

**怎么算**：冻结通道输出 → 取消息向量 → 训一个 logistic regression / 线性头预测目标属性 → 报测试集准确率，对 permutation 标签基线做对比。

**需要什么**：一批带属性标注的样本（属性 = 你希望通道传的东西，如"图里有几个人""目标物体类别""约束是否满足"）。**探针必须低容量**，否则你测的是探针的能力不是消息的信息量。

**能证伪什么**：假设"消息里编码了属性 $A$"。探针接近随机 → 强证伪。

### 4.2 但可解码性有一个致命的反证

**【原文】** *Observable Patterns Are Not Explanations: A Causal-Geometric Analysis of Latent Reasoning Models*，arXiv **2606.12689**。

**【原文，2606.12689】** 方法三件套：
- **因果干预**：对 latent thought 表征做**梯度子空间干预**——取梯度矩阵的 top 奇异向量（累计能量 99%），在强度 $\alpha \in \{0, 0.5, 1, 1.5, 2, 5, 10, 25, 50, 100\}$ 下 ablate 或放大这些 "loss-sensitive directions"；另用 **causal tracing**，测量把干净激活 patch 进被污染前向传播时的 **KL 散度恢复量**。
- **几何指标**：相邻梯度子空间之间的**主角相似度**（稳定性）；轨迹的 **Markov 性**（用 $R^2$ 拟合从 $h_{t-1:t-n}$ 预测 $h_t$）；梯度子空间的维数。
- **核心论断**：可观察模式（BFS 式搜索、可解码的中间步骤、注意力分布）是 **epiphenomenal（附带现象）**——**它们在缺少所谓机制的对照模型里同样出现**。原话主张 "latent thoughts should be treated as hidden computation, not hidden explanation"，**"Decodability alone cannot establish mechanism; only causal influence on model behavior matters."**

**【推断】** 这篇给探针族划了一条硬边界：**探针只能给必要条件，不能给结论。** 正确的用法是——

- 探针**失败** → 强证据："消息里没有这个信息"（可以直接下结论）
- 探针**成功** → 弱证据："消息里可能有这个信息，但 receiver 未必用"（必须再跑 CIC/CAG）

**这个非对称性要写进你的方法论。** 顺带，2606.12689 的"对照模型也出现同样模式"这个论证结构，你可以直接搬来做**训练版通道的对照实验**：训一个**随机初始化 / 只训了几步**的通信模块作为对照，如果它的探针指标同样好看，说明你的探针在测架构而不是训练。

### 4.3 CKA / 表征相似度：谨慎使用

**【原文】** *Reliability of CKA as a Similarity Measure in Deep Learning*，Davari, Horoi 等，**ICLR 2023**，arXiv **2210.16156**。另有 *Deceiving the CKA Similarity Measure in Deep Learning*（Horoi 等，OpenReview `hITONWhDIIJ`；**我未找到对应的 arXiv 号，请自行核实**）。

**【原文，2210.16156 及相关】** 三条批评：
1. **早期层的高 CKA 值不代表特征更有用或更相似**；高 CKA 不保证功能相似或可迁移，低 CKA 也不能确证行为分歧。
2. CKA 对一大类**在 ML 中自然出现的简单变换敏感**，对离群点和保线性可分性的变换尤其脆弱。
3. **反例**：CKA 值可以骤降到 0，而**同时存在一个线性分类器能以 >90% 准确率分类两组表征**。

**【原文，检索结果】** 推荐的补充诊断：线性探针、滤波器可视化、类可分性。全局 CKA 会掩盖细粒度的、与目标属性相关的泄漏；**subspace-level CKA**（把评估限制在任务判别方向上，如 logistic regression 的单个投影）能揭示全局相似度测不到的可迁移性。

**【推断】** 对你的场景：CKA 适合回答"sender 空间和 receiver 空间**长得像不像**"，这在跨模型/跨模态通道里是个自然的问题（尤其是 VLM，你要把东西塞进视觉端口）。但**不要把 CKA 当通道质量指标**——上面第 3 条反例说明它可以在功能完全保留时归零。把它当**训练诊断**（对齐损失有没有在起作用）而不是**结论指标**。

### 4.4 一个便宜好用的几何指标：类间可分性（inter-label separation）

**【原文】** *Semantic Register Compression in Multi-Agent LLM Cascades: Measurement, Predictors, and Cross-Domain Generalization*，Manuele Tele Junior Fernandez（独立研究者），arXiv **2607.14119**。

**【原文，2607.14119】** 定义"语义寄存器压缩"= "the loss of category distinguishability that occurs when an intermediate agent rewrites inputs in an evaluative or analytical style"——**中间 agent 的改写让不同类别的样本在语义上互相靠近**，而中间推理表面上依然流畅，从而**掩盖了下游决策能力的下降**。

主指标 **inter-label separation**：用 all-MiniLM-L6-v2 编码 pipeline 各阶段文本，按 ground-truth 标签分组，算标签质心

$$c_{t,y} = \frac{1}{|E_{t,y}|}\sum_{e \in E_{t,y}} e$$

再算所有质心两两欧氏距离的均值（$K$ = 标签数）：

$$S_t = \frac{2}{K(K-1)}\sum_{i<j}\lVert c_{t,i} - c_{t,j}\rVert_2$$

派生的**压缩百分比**：

$$C_t = \frac{S_{\text{input}} - S_t}{S_{\text{input}}} \times 100$$

正值 = 压缩，接近零 = 保持，负值 = 扩张。

**【原文，2607.14119】** 主要发现：
- **压缩定位在 Evaluator 阶段，Collector 阶段约 0% 变化**（阶段级定位）
- **非单调**：20 级 prompt 强度梯度上，压缩在 level 9（平衡意见）达峰 29.1%，而非最强批判强度
- 五个 Evaluator 变体的因果对照：原始 16.6% 压缩、变体 A（正反平衡）28.3%、变体 B（倾向支持可信度）**扩张 45.0%**、变体 C（1–10 数值打分）21.5%、变体 D（恒等透传）无压缩
- 跨域：LIAR 10.3%、SST-5 28.2%、医疗分诊 Triagegeist 9.1%——**同一 Evaluator prompt 下压缩强度依域而变**
- 输出分布坍缩：最终 Decider 向中间类别集中（如 "half-true" 占 48%，SST-5 "neutral" 占 56%）

**【推断】** 这个指标对你极有价值，理由有三：(a) **不需要 receiver**，纯 sender 侧就能算，迭代极快；(b) **可以逐阶段/逐层算**，直接告诉你信息在哪一步被压掉了；(c) 它把"通道退化"变成了一个**几何上可视化的**东西。变体 D（恒等透传）作为对照的设计也值得抄——**它是"通道健全性"的天然上界**。

局限：它需要有类别标签，且质心距离对嵌入器的尺度敏感，所以只能做**相对比较**（同一嵌入器、同一数据、不同阶段/不同通道设计），不要跨设置比绝对值。

### 4.5 反演/重构攻击：把"隐私风险"反过来当"信息量探针"

**【原文】** *LCGuard: Latent Communication Guard for Safe KV Sharing in Multi-Agent Systems*，arXiv **2605.22786**。它是**防御机制**不是检测器：学一个变换让共享 KV 里的敏感信息更难被重构。报告的指标是 **Attack Success Rate (ASR)**（对手重构敏感输入的成功率）、**Reconstruction Loss**（对抗解码器恢复输入的损失）、**Privacy Score / Leak Rate**。其信号来源是"**重构损失与先验损失之间的差距**"——即通信表征是否比基线预期含有更多关于敏感输入的信息。论文**未讨论**把它当通道质量诊断用。

**【原文】** VLM 侧的类似手法：*Detached Skip-Links and $R$-Probe*，arXiv **2603.20020**——在冻结的 ViT encoder 和 adapter 上挂一个**轻量重构头**，通过"仅从冻结 MLLM 的**末层 image token** 能多忠实地重构输入图像"来评估视觉-语言桥接机制。

**【推断】** 这是一个很漂亮的**语义反转**：隐私文献里 "重构成功率高 = 坏"，在通道评估里就是 "**重构成功率高 = 通道确实携带了信息**"。而且它比 MI 估计**实际得多**——重构损失是一个具体、可复现、无理论上限问题的数字。

对你的 VLM 场景，我建议的具体形式（**【推断】**）：在 latent 消息上挂一个解码头，测**能否重构出源图像的关键语义**（不必是像素级，可以是 caption / 检测框 / 分割掩码）。这个"**语义重构保真度**"配合 §5.2 的 rate 轴，就是一条真正的 rate–distortion 曲线。而且它对"$W_a$ 只对文本词表成立"这个理论断点是**直接的实证检验**：如果视觉信息在通道里被重构不出来，断点就是实的。

---

## 五、消融与曲线族（最便宜、最该先做）

### 5.1 标准对照组清单

**【原文，多来源汇总】** 综合 2607.26773 的四条件、Lowe et al. 的打乱消息实验、以及 MARL 通信消融的常规做法（检索确认："random，消息替换为随机值；no，消息向量置零；两种条件下性能都显著下降"）：

| 对照组 | 构造 | 控制掉什么 | 期望结果（通道健康时）|
|---|---|---|---|
| 零消息 / $M_0$ | 不注入 | 基线 | 最低 |
| **零向量消息** | 注入全零 | "注入这个动作"本身的扰动 | 与 $M_0$ 接近，若差很多说明注入本身有副作用 |
| **随机消息** | 匹配统计量的随机向量 | 消息的分布/范数 | 应接近 $M_0$ |
| **置换消息 $M_{\text{oth}}$** | 同 benchmark 他样本、长度匹配 | 消息的结构与域 | 应显著低于 $M_{\text{cur}}$ |
| **跨域消息 $M_{\text{xbench}}$** | 他数据集、长度匹配 | 域匹配的价值 | BME = $\bar U_{\text{oth}} - \bar U_{\text{xbench}}$ |
| **自生成 $M_{\text{self}}$** | receiver 自己、算力配平 | "多算一遍"的价值 | SSG > 0 才说明需要另一个 agent |
| **纯文本通道** | 同角色结构、文本传递 | latent 相对文本的增量 | 你论文的主对手 |
| **恒等透传** | sender 输出直接当消息 | 通道模块本身的损耗 | 上界参照（借自 2607.14119 变体 D）|

**【推断】** 注意**零向量消息**和**随机消息**是两个不同的对照，很多论文只做一个。零向量测"注入位置/长度的结构性扰动"，随机消息测"分布匹配但无信息"。$M_{\text{oth}}$ 比随机消息强得多，因为它连"这看起来像一条合法消息"都控制住了。**递进地看：$M_0$ → 零向量 → 随机 → $M_{\text{xbench}}$ → $M_{\text{oth}}$ → $M_{\text{cur}}$ 是一条信息量单调递增的阶梯**，画成一张柱状图就是你论文里最有说服力的一张图。

### 5.2 rate–utility 曲线（V2X 传统，LLM 侧的空档）

**【原文】** *Where2comm: Communication-Efficient Collaborative Perception via Spatial Confidence Maps*，**NeurIPS 2022**，arXiv **2209.12836**，代码 `github.com/MediaBrain-SJTU/Where2comm`。评估协议：**检测性能（AP@IoU=0.50）与通信带宽之间的 trade-off 曲线**。论文报告在 CoPerception-UAVs 上以 **5000 倍更少的通信量**仍优于 When2com。它靠空间置信图只发送"稀疏但感知关键"的部分，并可**动态调整参与通信的空间区域以适应变化的带宽**。

**【原文】** *Learning Distilled Collaboration Graph for Multi-Agent Perception*（DiscoNet），**NeurIPS 2021**，arXiv **2111.00643**，代码 `github.com/ai4ce/DiscoNet`。关键设计：**以"早期协作（共享原始数据、拥有全局视野）"的模型作为 teacher / Upper-bound**，蒸馏指导中间协作（共享特征）的 student。原文强调：**没有蒸馏正则，矩阵值边权就不反映有信息量的空间注意力**。

**【推断】** 这两篇合起来给了你两个 LLM latent communication 领域**基本没人用**的评估工具：

1. **rate–utility Pareto 前沿**：横轴 = 每次 handoff 的通信量（KV 的 float 数 × 位宽，或折算成等价 token 数），纵轴 = 任务效用。**不要报单点，报曲线。** 这样你和文本通道的比较才是公平的——文本通道也在这条曲线上有个点。判断两个训练算法孰优的**正确方式就是比 Pareto 前沿，而不是比某一个配置下的准确率**。
2. **oracle gap**：定义一个"上界通道"（比如：receiver 直接拿到 sender 的**全部**原始输入 + 全部 KV，无压缩无对齐损失），你的通道离它多远。这个 gap 是**归一化的、跨设置可比的**，比裸准确率有信息量得多。DiscoNet 的 early-collaboration teacher 就是这个思想。

**【推断，这条最重要】** 你问"如何设计实验判断训练算法的好坏"——**答案就是这两条**。同一个 rate 预算下比 utility，或者同一个 utility 下比 rate；再加上 oracle gap 做归一化。单点准确率比较是没有信息量的，因为不同方法的通信预算根本不一样。

### 5.3 剂量–响应曲线（VLM 特别适用）

**【原文】** *Diagnosing Visual Ignorance in Vision-Language Models*，Zhou, Zhang, Wang, Wang（北京大学），arXiv **2606.06890**。

**【原文，2606.06890】** 两类方法：
- **内部机制分析**：在 baseline 与 fine-tuned 模型之间**互换 decoder 层**的替换实验；**逐层监督 MLP 探针**，追踪 ground-truth 语义 vs 语言先验语义随 transformer 深度的变化。
- **外部行为评估**：**progressive visual decay metric** —— 用**多步高斯模糊**（核大小 $1\times1$ 到 $61\times61$）逐级退化视觉输入，追踪两个指标：
  - **identical answer rate $E$**：预测是否与原图一致
  - **consecutively identical rate $C$**：答案是否在**所有**模糊等级下都不变

四个宏观指标：各模糊等级下的平均一致答案概率、连续一致率（**用于滤掉随机猜测噪声**）、退化下的准确率、限制在预测不变样本上的准确率。

**【推断】** 这套直接可以搬到 latent 通道上，而且是**你的场景最需要的那个实验**。做法：不模糊图像，而是**逐级退化 latent 消息**（加噪声 / 逐步截断秩 / 逐层置零 / 逐步量化），画出效用随退化程度的曲线。

它能回答一个 CAG 回答不了的问题：**通道的信息是集中还是冗余？** 曲线陡降 = 信息脆弱且集中；曲线平缓 = 冗余（可以压缩，见 §5.2）；曲线**平坦到底** = 灾难信号，说明消息内容根本不重要（等价于 CAG ≈ 0，但这个测法更便宜，不需要构造 $M_{\text{oth}}$）。

$C$（连续一致率）这个设计尤其值得抄——它专门用来在**答案空间受限**（多选题）时滤掉随机猜测造成的假一致，你做 VQA 类任务一定会遇到这个问题。

### 5.4 约束保持率（closed-world，ground truth 可穷举）

**【原文】** *State Compression in Two-Agent LLM Relays: A Closed-World Study of Constraint Preservation*，Anantha Sharma, Sheeba Elizabeth John, Kaarthik Senthil Kumar, Saratsuhas Vijayababu，arXiv **2607.18265**（v1, 2026-05-20）。**未提及公开代码**。

**【原文，2607.18265】** 设计：受控旅行规划基准，**固定库存（10 家酒店、10 个航班）、50 个目标实例**。每个目标指定硬约束（预算上限、必需设施、酒店最大距离、出发时间窗、舱位等级）。**ground truth 通过穷举枚举计算**，从而可客观打分可行性。

上游 Researcher agent 对照约束审计全部库存选项，产出长 trace → 经过压缩层 → 下游 Booker agent（**只收到 goal 和压缩后的 payload**）做预订决策。

**四种 hand-off 条件**：无压缩（对照）、叙事式摘要（250 词上限）、schema 约束的 JSON 抽取、基于嵌入的剪枝（余弦阈值 0.75）。

**指标**：主指标 **feasibility accuracy**（Booker 的选择是否满足全部硬约束）；次要指标 optimal match accuracy、optimality gap（美元成本差）；效率指标 compression ratio、tokens per run；表征诊断用嵌入空间余弦相似度。

**结果**：JSON 抽取 feasibility **0.96**，叙事式摘要跌到 **0.48**。结论是**结构化、可审计的格式比单纯的简洁更能保住约束**。

**【推断】** 这个设计的方法论价值远超它的具体任务：**造一个 ground truth 可穷举的封闭世界，然后逐条检查"哪条信息在通道里死了"。** 它把"通道质量"从一个模糊的准确率变成了**逐槽位、可归因的清单**。

对你的 VLM 场景，我建议造一个**合成视觉封闭世界**：可控场景（$N$ 个物体、属性、空间关系全部已知），sender 看图，receiver 只通过 latent 通道拿信息并回答关系类问题。这样你能得到**逐属性的保持率**（颜色保住了吗？数量？空间关系？遮挡？），而不是一个笼统的 VQA 准确率。**这个实验的诊断力比任何真实 benchmark 都强**，而且合成数据无污染、可任意扩样本量（正好解决 §1.6 说的样本量问题）。

### 5.5 鲁棒性 / 完整性（对抗视角）

**【原文】** *When Latent Agents Lie: KV-Cache Integrity in Multi-Agent LLM Collaboration*，Luís Brito（ESTG/IPVC）、Carlos Baquero（FEUP/UP），arXiv **2606.28958**。

**【原文，2606.28958】** 设置：planner / specialist / verifier / coordinator 的角色序列架构，split-evidence 任务；65 条改造自 HiddenBench 的记录 + HotPotQA 完整验证集（7405 条）复现；Qwen3-4B 为主、Qwen3-8B 辅；指标 EM / F1。干净条件下 latent 协作 EM/F1 = **0.338 / 0.486**，纯文本协作 = **0.231 / 0.369**。

**三层威胁**：
- **Tier A（语义）**：仅恶意文本承诺
- **Tier B（非语义）**：hidden state 操纵——"random latent-thought corruption"、"scale-8 latent-thought multiplication"、符号翻转
- **Tier C（自适应）**：白盒梯度优化的**范数匹配**扰动

**非语义攻击最具灾难性，性能坍缩到接近零。**

三种防御的对比：可见 verifier 过滤（**检测不到 hidden state 污染**）；幅度隔离（对朴素攻击有效，**对范数匹配的自适应攻击失效**）；传输层 MAC（HMAC-SHA256 清单，绑定 specialist 身份、会话、模型、承诺哈希、payload 摘要）——**接受 774/774 诚实 payload，拒绝 295/295 篡改 payload**。

**【推断】** 两点可用：
1. **"scale-8 乘法"和"符号翻转"是极其便宜的通道敏感性探针**，不需要任何对抗训练就能跑。如果你的通道对这些扰动**毫不敏感**，那不是鲁棒，那是 receiver 根本没在用消息内容（又一个 CAG≈0 的廉价代理）。**敏感性是"通道在工作"的必要条件。** 这是一个反直觉但正确的判据。
2. **latent 通道的鲁棒性剖面本身就是一个可报告的通道属性**。训练出来的通道有没有比 training-free 的更鲁棒（或更脆弱）？这是个开放问题，也是个现成的实验。**【推断】** 我的猜测是训练版更脆弱——因为训练会让 receiver 依赖通道的特定统计结构——但这需要实测。

---

## 六、给你的场景（VLM MAS + 训练一个通信模块）的具体建议

### 6.1 最关键的发现：你的直接前作没做通道评估

**【原文，2602.15382，我逐节核对其评估部分】** Vision Wormhole 的评估：

- **完全没有直接的通道质量指标** —— 无重构误差、无互信息、无探针分析
- 指标只有：多选题准确率（ARC / GPQA / MedQA）、数学准确率（GSM8K / AIME 2024/2025）、代码 pass@1（MBPP-Plus / HumanEval-Plus）、端到端 wall-clock
- 对照组：TextMAS（同角色结构、不同通道）、单模型基线、GSM8K 上尝试的 LatentMAS-Hybrid（**不稳定，被放到附录 F**）、OCR 图像中继基线（附录 E）、弱监督 codec 变体（<100 条锚文本）
- 文本通道有双重身份：**训练时当蒸馏 teacher**（label-free 自监督，损失 = hidden-state L2 + logits softmax 的 KL，原文措辞 "representational fidelity and output-distribution fidelity"）；**运行时当 baseline**
- **通道保真度只能从"下游准确率是否退化"间接推断**

**【原文，2602.00471】** L²-VMAS 我只拿到摘要页，无法确认其消融与通道级评估的细节（摘要只给了平均准确率 +2.7–5.4%、token 用量 −21.3–44.8%）。**我没能核实它是否做了 latent memory 的随机化/移除消融，请自行读全文确认。**

**【推断，这是你的机会】** 综合起来：**"多模态 + latent 通信 + 训练"这块地确实被占了（L²-VMAS、Vision Wormhole、MACF），但"这些通道到底传了什么"这块地是空的。** 2607.26773 只审计了文本 LatentMAS（Qwen3 + GSM8K/ARC/MATH），**视觉侧一片空白**。而视觉侧恰恰是理论最可疑的地方（$W_a$ 断点）。

### 6.2 VLM 特有的必做对照：视觉专属对照组

**【推断，方法直接组合自 2607.26773 的 $M_{\text{oth}}$ 构造 + 2606.06890 的视觉退化】**

$W_a$ 断点的本质是：**视觉信息不过 $W_{\text{in}}$，所以对齐没有理论保证。** 那么最该被证伪的假设就是：

> H0：这个 latent 通道其实只传了文本/语言侧的内容，视觉信息根本没过去（下游涨分来自 sender 的语言推理，不是视觉证据）。

**证伪 H0 需要三个视觉专属对照**（在标准 $M_{\text{oth}}$ 之外额外做）：

| 对照 | 构造 | 若通道真的传视觉，期望 |
|---|---|---|
| **换图不换文** | 同一文本 query，sender 看**另一张图**生成消息 | 效用应显著下降（"视觉 CAG" > 0）|
| **换文不换图** | 同一张图，sender 收到**另一个** query | 分离出 query 条件化的贡献 |
| **图文错配** | 图和文来自不同样本 | 效用应接近或低于 $M_0$ |

**核心判据（我的推断）：定义 "visual-CAG" = $\bar U_{\text{cur}} - \bar U_{\text{图换文不换}}$。如果 visual-CAG ≈ 0 而标准 CAG > 0，那么你的通道传的是文本内容，视觉端口是摆设——这直接坐实了 $W_a$ 断点是实的。** 这一个实验就能决定你整个项目的方向，而且**在你训任何东西之前就可以先在 training-free 基线上跑**，成本极低。我认为**这应该是你的第一个实验**。

### 6.3 训练路线特有的三个陷阱（training-free 没有的）

**【推断】**

**陷阱一：通道退化成"触发器"。** receiver 学会"有消息 = 切换到某个推理模式"，内容无关。
- **签名**：OME 大、CAG ≈ 0、CIC 接近 receiver 噪声地板
- **抓法**：$M_{\text{oth}}$ 对照（§1.2）；或更便宜的：随机消息对照，若随机消息也涨分就是它

**陷阱二：通道退化成"传答案"而不是"传工作记忆"。** 如果你的训练信号是下游任务损失，最短路径就是让 sender 把答案编码进消息，receiver 解码出来——这时候多智能体结构就没意义了，等价于一个 pipeline。
- **签名**：PS 在 $X$ = sender 最终答案 时很高，但在 $X$ = sender 中间证据/视觉属性 时很低
- **抓法**：**多个 $X$ 的 PS 谱**（这是我对 2607.26773 的扩展；它只用了一个 $X$ = 答案正确性）。做法：定义一组语义变量 $X_1, \dots, X_k$（最终答案、中间步骤、视觉属性、约束满足），逐个测 PS。**通道的"信息剖面"比单个 PS 值有信息量得多。**

**陷阱三：训练集泄漏 / 域过拟合。** 训出来的通信模块可能只在训练域上是通道，换域就退化。
- **抓法**：**BME**（$\bar U_{\text{oth}} - \bar U_{\text{xbench}}$，来自 2607.26773）+ 跨域泛化测试（2607.14119 的三域设计：同一模块、不同域，看压缩率是否稳定）

### 6.4 分层验收标准（可直接当实验计划）

**【推断，综合上述所有来源】**

**L0 —— 健全性（不过就是 bug，不是结果）**
- [ ] 掩码 → CIC 掉到数值地板【2607.26773】
- [ ] 自替换 → CIC = 0 且 CAG = 0（精确零）【2607.26773】
- [ ] 完全恢复 → NLD = 1【2607.26773】
- [ ] 恒等透传对照跑通，作为上界参照【借自 2607.14119 变体 D】
- [ ] **receiver 噪声地板已测量**：同一 $M_{\text{cur}}$ 重复 $K$ 次的 $\bar D_{\text{cur},\text{cur}'}$【我的补充，论文未给做法】

**L1 —— 通道在传（必要非充分）**
- [ ] PS 显著高于 permutation 参照（**报相对量，不报 bit 数**）
- [ ] PL = $\bar D_{\text{cur},0}$ 显著 > 0
- [ ] 探针可从消息读出目标属性（**记住：探针成功只是弱证据**【2606.12689】）

**L2 —— 传的是内容不是信号（关键关卡）**
- [ ] **CIC = $\bar D_{\text{cur},\text{oth}}$ 显著大于 receiver 噪声地板**
- [ ] **CAG = $\bar U_{\text{cur}} - \bar U_{\text{oth}}$ 的 paired bootstrap 区间不跨零**
- [ ] 报告完整分解 OPE = OME + CAG（**不要只报 OPE**）
- [ ] VLM 专属：**visual-CAG 不跨零**（§6.2）

**L3 —— 值得（贡献声明的门槛）**
- [ ] SSG > 0 且区间不跨零（否则你做的是 test-time compute 分配，如实写）
- [ ] rate–utility Pareto 前沿优于文本通道（**不是单点比较**）【2209.12836 范式】
- [ ] oracle gap 相对基线缩小【2111.00643 范式】
- [ ] 跨域：BME 与跨域压缩率显示通道不是训练域过拟合

### 6.5 优先级建议

**【推断】** 如果时间有限，按这个顺序做，每一步都可能让你提前止损：

1. **visual-CAG 实验**（§6.2），在 training-free 基线上跑 —— 决定项目是否成立，成本最低
2. **$M_{\text{oth}}$ 对照 + OPE 分解**（§1.2–1.3）—— 决定你的方法声明是否诚实
3. **合成封闭世界的逐属性保持率**（§5.4 改造版）—— 诊断力最强，样本量无限
4. **rate–utility 曲线**（§5.2）—— 领域空档，几乎白送的贡献
5. **剂量–响应曲线**（§5.3）—— 便宜的冗余度分析
6. PS 谱 / 探针 / TopSim —— 锦上添花，别当主证据

---

## 七、总表

| 指标 | 家族 | 怎么算 | 需要什么 | 能证伪什么 | 现成实现 |
|---|---|---|---|---|---|
| **PS** $I(M;X)$ | 因果审计 / 涌现通信 | 交叉拟合 MI 下界 vs permutation 参照 | 消息 + sender 侧标签 $X$ | "消息里编码了 $X$" | `facebookresearch/measuring-emergent-comm`；MI 估计器见 `gtegner/mine-pytorch` 等 |
| **PL** $\bar D_{\text{cur},0}$ | 因果审计 | 首 token 分布的 JS 散度 | 两次 receiver 前向 | "receiver 对消息存在有反应" | 无（自实现，~30 行）|
| **CIC** $\bar D_{\text{cur},\text{oth}}$ | 因果审计 / 涌现通信 | 首 token JSD（vs 他样本消息）| $M_{\text{oth}}$ 构造 + 噪声地板 | "receiver 对消息**身份**有反应" | 无 |
| **CAG** $\bar U_{\text{cur}}-\bar U_{\text{oth}}$ | 因果审计 | 配对准确率差 + paired bootstrap | 同上 + 任务标注 | **"内容有任务价值"（最重要）** | 无 |
| **SSG** $\bar U_{\text{cur}}-\bar U_{\text{self}}$ | 因果审计 | 配对准确率差 | 算力配平的 $M_{\text{self}}$ | "价值来自另一个 agent" | 无 |
| **NLD** | 因果审计 | teacher-forced logprob 差的归一化 | 组件级恢复能力 | "某层/某组件承载了通道价值" | 无 |
| **SC / CI** | 涌现通信 | 消息–自身动作互信息 / 概念对齐 | 离散化消息 | signaling（**注意：可被架构伪造**）| `facebookresearch/egg`、`olipinski/emlangkit` |
| **TopSim** | 涌现通信 | 输入距离 vs 消息距离的 Spearman | 一批样本，无需 receiver | "通道是结构保持映射" | `emlangkit.metrics.topsim`、`Near32/ReferentialGym` |
| **线性探针** | 表征分析 | 冻结消息 + 低容量 logistic 头 | 属性标注 | 探针失败→强证伪；成功→弱证据 | scikit-learn 即可；范例 arXiv 2605.25310 |
| **CKA / subspace-CKA** | 表征分析 | 中心化核对齐 | 两侧表征 | 空间几何相似性（**不可当质量指标**）| 广泛可得；**先读 2210.16156** |
| **inter-label separation** $S_t$, $C_t$ | 表征分析 | 标签质心两两欧氏距离均值 | 类标签 + 句嵌入器 | "某阶段压掉了类可分性" | 公式见 2607.14119，自实现 |
| **语义重构保真度** | 表征分析 | 挂解码头，测 caption/框/mask 恢复 | 解码头 + 标注 | "通道携带视觉信息" | 范式见 2603.20020 (R-Probe)、2605.22786 (ASR) |
| **消融阶梯** | 消融 | $M_0$/零/随机/$M_{\text{xbench}}$/$M_{\text{oth}}$/$M_{\text{cur}}$ | 只要能跑 receiver | "通道到底有没有用" | 无（但最便宜）|
| **rate–utility 曲线** | 曲线 | 通信量 vs 效用的 Pareto 前沿 | 可调压缩率 | **"方法 A 优于方法 B"（唯一公平比法）** | 范式见 `MediaBrain-SJTU/Where2comm` |
| **oracle gap** | 曲线 | 与"全信息上界通道"的距离 | 上界通道实现 | 归一化的通道效率 | 范式见 `ai4ce/DiscoNet` |
| **剂量–响应曲线** | 曲线 | 逐级退化消息，测 $E$ / $C$ | 退化算子 | "信息集中还是冗余"；平坦→通道无用 | 范式见 2606.06890 |
| **约束保持率** | 消融 | 封闭世界穷举 ground truth，逐槽检查 | 合成受控数据 | **逐条信息哪条死了** | 范式见 2607.18265（无代码）|
| **扰动敏感性** | 鲁棒性 | scale-8 乘法 / 符号翻转 / 随机腐蚀 | 无 | 不敏感 → receiver 没在用内容 | 范式见 2606.28958 |

---

## 八、诚实声明：我没找到 / 没核实的部分

1. **2607.26773 没有公开代码**。我逐节核查全文，未发现任何 GitHub 链接。五指标需要自己实现（工作量不大，但 $M_{\text{oth}}$ 的长度匹配、$M_{\text{self}}$ 的算力配平这两处的具体做法论文没给，你的实现会与原文有出入）。
2. **2607.26773 的 PS 估计器类型未知**。原文只说 "cross-fitted lower bound"，未指明 MINE / InfoNCE / 分箱 / 分类器。
3. **2607.26773 的 "receiver instability" 和 "message-distribution similarity" 无操作定义**。§6.4 里那条噪声地板做法是**我的补充**，不是论文的。
4. **Lowe et al. 的 CIC 公式我抓到的版本疑似 HTML 转换有损**（形式上不像标准互信息）。语义（干预式互信息）确定，**符号请以原 PDF 为准**：`arxiv.org/pdf/1903.05168`。
5. **Relative Representations (2209.15430, ICLR'23) 的具体评估指标我没拿到**。我尝试了 abs 页和 ar5iv（后者报 "Conversion to HTML had a Fatal error"），只抽到一个 "pairwise distance average"。我**没有**核实它是否用了 Jaccard k-NN / MRR —— 请直接读 PDF。代码在 `github.com/lucmos/relreps`。另有后续工作 *Improving Relative Representations with Learned Anchors and Whitened Inner Products*（arXiv **2605.30596**），我未阅读。
6. **L²-VMAS (2602.00471) 的消融细节我没拿到**，只有摘要级信息。它是否做了 latent memory 的随机化/移除对照，我不知道。
7. **"Deceiving the CKA Similarity Measure in Deep Learning" 我只有 OpenReview id `hITONWhDIIJ`，没有 arXiv 号。**
8. **2604.13349（OBF 压缩）我只拿到搜索摘要**（PDF 超过 fetch 大小限制，45 页）。已知：作者 Yiping Li, Zhiyu An, Wan Du；保留 9.9%–20.2% 的 prompt KV 状态可达 full KV relay 的 97%–120% 准确率（9 个 benchmark）。**搜索结果称其为 "PLDI 2026 论文的扩展版"，这个会议归属我认为可疑（PLDI 是编程语言会议），请自行核实。** 这篇很可能有你要的 rate–utility 数据点，值得单独读。
9. **我没有找到任何一篇论文实测过 "latent 消息里有多少 bit"**。这与 §3.2 的理论限制一致——**我倾向于认为这在方法论上就是做不到的，而不是没人做**。
10. **9 个多小时的检索里，我没有找到任何针对 VLM latent 通道的因果审计工作。** 这是我认为最确切的空档。

---

## 附：关键发现速览

1. **已经有人把这套做完了，而且就是你要的那篇**：arXiv 2607.26773（Zhang & Emu）把 Lowe et al. AAMAS'19 的涌现通信指标整套搬进 LLM latent 通道，给出五指标 + 两条精确分解恒等式 `OPE = OME + CAG` 和 `CAG = DSC + SSG`。核心实证结果是"总效应可以由两个方向相反的分量合成"——Qwen3-4B 在 GSM8K 上总效应 −1.00pp，却分解成 OME=−6.17pp + CAG=+5.17pp。**只看下游准确率必然骗人，这是被实测证明的，不是推测。**

2. **评估的骨架是一条五级阶梯，每一级都能独立证伪**：有容量≠有使用（PS）→ 有反应≠因果（PL）→ 有效果≠内容有效果（CIC/CAG，关键）→ 内容有效果≠需要另一个 agent（SSG）→ 可解码≠被使用（2606.12689 的"可观察模式是附带现象"论证）。训练版通道最可能死在第三级：receiver 学成"看到消息就切模式"，内容无所谓，签名就是 OME 高而 CAG≈0。

3. **不要试图报"消息里有几个 bit"——理论上做不到**。2606.05711 那个"hidden state ~40000 bits vs token ~15 bits"只是松上界；McAllester & Stratos (1811.04251, AISTATS'20) 证明任何 distribution-free 高置信 MI 下界不超过 O(ln N)，1000 个样本天花板约 10 bit；Poole et al. (1905.06922) 证明 InfoNCE ≤ log K。**正确用法是相对量：对 permutation 参照的相对可分性**，这也正是 2607.26773 采取的做法。

4. **最大的空档就在你的赛道上**：Vision Wormhole (2602.15382) 是最接近的 VLM latent 通道工作，但我逐节核对其评估部分——**它完全没有任何通道级指标**，只有下游准确率和 wall-clock，没有探针、没有重构误差、没有随机/置换消息对照。VLM 侧的 $W_a$ 理论断点恰恰要求一个"视觉专属对照组"（换图不换文 / 图文错配），而目前没人做。

5. **可直接抄的现成实现存在，但审计那篇没有代码**：facebookresearch/measuring-emergent-comm、facebookresearch/egg、olipinski/emlangkit、Near32/ReferentialGym、lucmos/relreps、MediaBrain-SJTU/Where2comm、ai4ce/DiscoNet 都可用；2607.26773 全文未提供任何 GitHub 链接（我逐节核查过），五指标需自己实现。另外它三个数据集样本量仅 40–100 题，多个置信区间跨零——**配对设计 + 样本量是这类实验能否测出东西的生死线**。
