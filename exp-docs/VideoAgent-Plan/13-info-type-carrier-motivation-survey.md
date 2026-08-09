# 13 · "信息类型决定载体" motivation 的文献审计——有没有人说过、证据链两侧各有什么

> 缘起：用户问"感知信息和推理信息的最佳传递载体不同（感知→latent/topk 概率，推理→文本），这个 motivation 有没有相关研究"。
> 前提对齐：这个表述 = [doc08](08-motivation-evolution-info-type.md) 的 **v3 版本**，其中"感知×topk 概率"已被 P2-1 自家数据否决（+1.6 n.s. 双基座封口）；本文审计的是 **v4 存活版**："文本饱和推理信道（三连续载体全阴）+ 文本削损感知信道（像素 +6.0 显著）→ 载体分化：推理留文本、感知走视觉表征（须训练桥）"。
> 方法：9 agent、171 次检索、6 角度（载体路由查重/感知瓶颈/思考介质×任务/转写损失分型/认知科学锚点/反方证据）+ 批评员补 2 轮，110 条筛得约 65 篇。**证据等级：全部来自 abstract 页（非全文）；标 ⚠ 的数字来自搜索摘要须重核。**

---

## §0 三句话判决

1. **motivation 本体没被人说过**：LLM 多智能体文献里**不存在**"按信息类型（感知 vs 推理）选通信载体"的表述或实验——最近的 HyLaT 按"冗长度/关键性"切且方向反着（推理→latent 图效率、结论→文本），媒介选择论文按算力带宽切，C2C 按层切；Beyond Tokens 综述的分类轴里干脆没有信息类型这一维。**逐信息类型 × 载体的受控析因实验（= 我们的 P2 矩阵）全网没有第二份**——这是框架贡献，不是撞车区。
2. **但它有 40 年的思想血统**：管理学 Media Richness Theory（1986，"媒介丰富度须匹配消息类型"）、认知科学双编码理论（Paivio，言语码/意象码分立）、Larkin & Simon 1987（同信息不同表征计算不等价、孰优取决于内容类型）——全是人类版的同一句话，没有任何一篇落到 LLM agent 通信。引它们=给框架接血统，不构成 scoop。
3. **v4 不对称的两侧证据链都很厚，但感知侧有一个必须正面处理的活口**：caption 政策混淆（Q-ViD 实测"先看题再写 caption"就能 +3.5~+4.2，量级逼近我们的 +6.0）——我们有部分防御（两臂都带按需工具通道且该通道已证饱和；8B 时代先知地图臂实测 +5.6），但"VL 读者 × 先知书 vs 像素"这一格没跑过，是唯一能反杀 +6.0 的攻击路径。

---

## §1 查重结果：谁离"按类型选载体"最近（都不构成 scoop）

| 工作 | 切分/选择判据 | 与我们的差 |
|---|---|---|
| [HyLaT](https://arxiv.org/abs/2605.25421)（2026-05）| 冗长认知信号→latent、简短关键信号→文本；判据=效率+可解释性，由训练数据结构诱导 | **方向反着**（推理内容进 latent）；无感知信道（纯文本 QA）；从不主张逐类型的**精度**增益。我们的判据是**保真度**，他们的是压缩率 |
| [JMSRA](https://arxiv.org/abs/2605.25422)（2026-05）| token vs KV-cache 自适应选择；判据=算力/带宽运行域 | 系统/网络判据，与信息类型无关；无精度分型消融 |
| [C2C](https://arxiv.org/abs/2510.03215)（ICLR 2026）| 可学门控选**哪些层**接受 cache 通信 | 按层效用路由，不按消息类型 |
| [GEASS](https://arxiv.org/abs/2605.01733)（2026-05）| 逐题门控信不信 caption：**全局题 caption 有帮助、细节题反而有害** | 单 VLM 内部、非 agent 通信——但它是"文本够不够用取决于所查信息类型"的最直接单模型实测，且门控本身是现成路由器 |
| [Capabilities & Limits of Latent CoT](https://arxiv.org/abs/2602.01148)（2026-02）| 理论：探索型内容→连续 latent、精确符号执行→离散 token，主张按任务自适应切换 | 单模型内思考介质，非 agent 间载体；但给了我们推理行全阴的机制（已定稿线索=单路径内容，连续载体无从发挥叠加优势） |
| [Visual Planning](https://arxiv.org/abs/2505.11409)（ICLR 2026 oral）| 空间任务纯图像规划 80.6% vs 文本 53.6%（+27pp）| 单模型内、需 RL 训练；是"感知内容配视觉介质"的最强单模型宣言 |

检索 agent 的原话结论："**No paper with a per-information-category carrier ablation (same workflow, carrier varied factorially across perceptual vs reasoning messages) comparable to our 500-question McNemar matrix**"。

**思想血统锚点**（引来接谱系，全在人类/认知科学侧）：[Media Richness Theory](https://dl.acm.org/doi/10.1287/mnsc.32.5.554)（Daft & Lengel 1986：媒介丰富度匹配消息含混度）· [Larkin & Simon 1987](https://onlinelibrary.wiley.com/doi/10.1111/j.1551-6708.1987.tb00863.x)（同信息、异表征、计算不等价且内容依赖——"(sometimes) worth ten thousand words"的 sometimes 就是重点）· [双编码理论](https://link.springer.com/article/10.1007/BF01320076)（Clark & Paivio 1991：言语码/意象码两套系统）——我们的矩阵 = 这三者的 LLM-agent 版首次受控检验。

---

## §2 v4 不对称的两侧证据链（外部独立，全部可引）

### 2.1 "文本削损感知信道"侧（厚）

- **机制定位**：[VLMs are blind](https://arxiv.org/abs/2407.06581)（ACCV 2024）——线性探针证明**视觉编码器里信息在、坏在译成语言那一步**，副标题就是我们的论点；[BLINK](https://arxiv.org/abs/2404.12390)（ECCV 2024）——14 类感知任务"resist mediation through natural language"，GPT-4V 51.3 vs 人类 95.7；[Eyes Wide Shut](https://arxiv.org/abs/2401.06209)（CVPR 2024）。
- **错误分解**：[MMMU](https://arxiv.org/abs/2311.16502) 感知错误 ~35% > 推理 ~26%；[PAPO](https://arxiv.org/abs/2507.06448) ⚠——RL 训过推理的多模态模型 **67% 错误仍是感知**；[VisOnlyQA](https://arxiv.org/abs/2412.00947)（COLM 2025）剥离推理后纯感知照样崩。
- **转写损失定量**：[CaptionQA](https://arxiv.org/abs/2511.21025)（换 caption 掉最多 32%，逐类别分解）；[Describe-Then-Generate](https://arxiv.org/abs/2509.18179)（图→文→图往返：99.3% 感知退化、系统性丢色度/几何/风格——转写按类型丢信息的最直接量化）；[TemporalBench](https://arxiv.org/abs/2410.10818)（时序细节是 caption 系统性漏掉的类型）；[IsoBench](https://arxiv.org/abs/2404.01266) 反向格见 §2.2。
- **跨域同构**：[UGround](https://arxiv.org/abs/2410.05243)（ICLR 2025）——GUI agent 里 HTML/a11y 文本序列化（=界面的 caption）是有损瓶颈，纯像素 grounding 胜出：**别的 agent 域已复现同款不对称**。
- **单模型介质切换**：[Whiteboard-of-Thought](https://arxiv.org/abs/2406.14562)（同一读者，文本 CoT 0% → 视觉介质 92%）；[MVoT](https://arxiv.org/abs/2501.07542)（ICML 2025）；[Visual Planning](https://arxiv.org/abs/2505.11409)。
- **语言学根据地**（"损失是语言的属性，不是我们 captioner 的锅"）：[色彩词 IB 理论](https://www.pnas.org/doi/10.1073/pnas.1800521115)（PNAS：词是连续感知空间的率受限有损码，且近最优——顺带解释为什么红利是 +6 不是 +30）；[20 语言五感可编码性](https://www.pnas.org/doi/10.1073/pnas.1720419115)（PNAS：转写保真度本身就依类型而变）；[Differential Ineffability](https://onlinelibrary.wiley.com/doi/10.1111/mila.12057)。

### 2.2 "文本饱和推理信道"侧（同样厚）

- **文本蒸馏推理跨模型无损**：[Distilling Step-by-Step](https://arxiv.org/abs/2305.02301)（770M 学 540B 的文本 rationale 反超教师）；[DeepSeek-R1](https://arxiv.org/abs/2501.12948)（80 万条**纯文本** CoT 跨架构蒸馏到 1.5B-70B，不需要任何隐状态/logit 访问）——推理经文本传递在最大尺度上被证充分。
- **符号内容文本碾压视觉载体**：[IsoBench](https://arxiv.org/abs/2404.01266)（COLM 2024）——同构表示对照，Claude-3 Opus 图像版比文本版**低 28.7 分**。与 CaptionQA 拼成双面对照：**同内容换载体，符号型文本赢、感知型图像赢**——这就是"信息类型决定载体"的已发表两翼，只是没人放进同一个受控框架。
- **连续载体在推理上的清醒剂**：[To CoT or not to CoT](https://arxiv.org/abs/2409.12183)（CoT 红利 95% 集中在带"="的题）；[Soft Tokens, Hard Truths](https://arxiv.org/abs/2509.19170)（免训练软输入塌缩成贪心；RL 训完 pass@1 也只打平，最佳推理配方还是离散 token）；[Reasoning by Superposition](https://arxiv.org/abs/2505.12514)（NeurIPS 2025 理论：连续思考赢在"并行假设叠加"型内容——**已定稿的线索/主张是单路径内容，正是文本饱和的那类**）。
- **认知锚**：[Fedorenko et al., Nature 2024](https://www.nature.com/articles/s41586-024-07522-w)（语言=为通信优化的码，非思维本身——注意此文被 Coconut 一系引去支持"latent 思考"，我们矩阵恰好裁决了双向读法：**思考侧读法在 agent 间转移上失败、通信码读法存活**，这是我们独有的引用角度）；[LoT 假说](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/abs/best-game-in-town-the-reemergence-of-the-languageofthought-hypothesis-across-the-cognitive-sciences/76F46784C6C07FF52FF45B934D6D3542)（推理内容天生离散组合式→文本序列化近无损）。
- **涌现通信侧**：[Mahaut et al.](https://arxiv.org/abs/2302.08913)（TMLR：异构预训练视觉网络社区——离散码折损指称精度但**换来可教性/互操作性**，连续码反之）——文本=可互操作但有损、连续=高保真但难对齐，正是我们不对称的权衡结构。

---

## §3 反方证据与调和（论文必须正面接的四拨）

### 3.1 推理侧"latent 赢文本"的反例（威胁 v4 前半句）

| 反例 | 声称 | 调和轴（我们的阴性为何不矛盾） |
|---|---|---|
| [CIPHER](https://arxiv.org/abs/2310.06272)（ICLR 2024）| 概率嵌入辩论 +0.5~5.0% | **多跳立场聚合** vs 我们单跳转述；弱模型；P2 已实测其机制单跳退化（587/587 不自停） |
| [Communicating Activations](https://arxiv.org/abs/2501.14082)（ICML 2025）| 免训练激活嫁接 +27% | 小模型+**中间层嫁接**（非末层注入进 prompt 位）；协调博弈类任务 |
| [LatentMAS](https://arxiv.org/abs/2511.20639)（ICML 2026 Spotlight）| 免训练 latent 协作 +14.6% | 同 backbone agent、KV 级工作记忆（比单点隐状态注入富得多）、多跳累积场景 |
| [Coconut](https://arxiv.org/abs/2412.06769)（COLM 2025）| 连续思考胜文本 CoT | **要训练**+单模型内部+只赢搜索型任务（GSM8K 输）——恰好落在 2505.12514 的理论边界内 |
| [Soft Thinking](https://arxiv.org/abs/2505.15778)（NeurIPS 2025）| 免训练概念 token +2.48 | 单模型**不间断前向流**内，无跨 agent prompt 边界；且被 Soft Tokens Hard Truths 的塌缩分析反咬 |
| [C2C](https://arxiv.org/abs/2510.03215)/[Interlat](https://arxiv.org/abs/2511.09149)/[Token Assorted](https://arxiv.org/abs/2502.03275) | 训练后连续载体胜文本 | 全要训练——**所以 v4 判决句必须限定"免训练"**：一旦允许训练，推理信道的文本饱和主张就不能下（这也是为什么 B 臂只押感知侧） |

四条调和轴一句话：**训练与否 × 单模型内流 vs 跨 agent 边界 × 多跳聚合 vs 单跳转述 × 弱模型 vs 强读者**。我们的全阴都落在"免训练×跨边界×单跳×强读者"格——没有任何反例在这个格里测过。

### 3.2 感知侧"文本够用"的反例（威胁 v4 后半句）

- [Vamos](https://arxiv.org/abs/2311.13627)/[ObjectMLLM](https://arxiv.org/abs/2504.07454)：caption 够用、嵌入无增益；对象信息**转成文本**反而最好。→ 调和：可符号化的感知内容（框、动作标签）文本确实赢——**v4 须收窄为"抗符号化的感知残差"**；他们的"视觉臂"是嵌入不是像素进强 VLM。
- [LLoVi](https://arxiv.org/abs/2312.17235)/[SiLVR](https://arxiv.org/abs/2505.24869)/[Socratic Models](https://arxiv.org/abs/2204.00598)：纯文本管线屠榜。→ 调和：读者更强+多感官文本（ASR），载体与读者强度混淆；我们的 McNemar 对同读者隔离载体。
- [MVU](https://arxiv.org/abs/2403.16998)：盲答也高分 → benchmark 语言先验。→ 我们的差分设计部分免疫，但报告 +6.0 时应在感知关键子集上分层展示。
- [Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987)（ICML 2024 position）：模态表征随规模收敛→载体差异渐近消失。→ 概念性论证非受控实验；**有限规模的未收敛残差正是 +6.0 住的地方**；且这是全场唯一接近"反对模态匹配载体"的立场文。

### 3.3 ⚠ 最锋利的活口：caption 政策混淆（批评员定级最高）

- [PromptCap](https://arxiv.org/abs/2211.09699)（ICCV 2023）开创、[Q-ViD](https://arxiv.org/abs/2402.10698) 在视频 QA 实测：**同一读者，只把 caption 从通用改成"先看题再写"，+3.5（NExT-QA）/+4.2（STAR）**——量级直逼我们的 +6.0。[VidCtx](https://arxiv.org/abs/2412.17415)（ICME 2025）补了另一半：问题条件化 caption 之上**加原始帧仍 +3.0**。
- 攻击句式："你的 +6.0 是通用 caption 政策的锅，不是文本载体的极限。"
- **我们的既有防御**（写论文时要摆出来）：① C/D 两臂**都**带按需工具通道（验证员点名帧→摄影师现场写=天然问题条件化文本），且工具通道已被整场战役证饱和（四路增强全无效）——即"问题条件化文本"在两臂中可得的前提下像素仍 +6.0；② 8B 时代先知地图臂（caption 生成时带题目）实测 +5.6 p=.013——我们**承认并量化过**政策效应本身。
- **仍缺的一格**：同读者（VL）× 先知书 vs 像素的正面对决没跑过（8B 时代先知臂是 8B 读者+拼接税基座）。这是唯一能实质缩水 +6.0 的实验缺口——若日后有人裁定补格，它是感知侧最有价值的一条对照（**本文只记录，不立卡**）。
- 同族还有**读者侧再归因**线：[T3](https://arxiv.org/abs/2410.06166)（时序瓶颈在读者 LLM，纯文本训练就能修）、[Arrow of Time](https://arxiv.org/abs/2605.07568)（时序信息死在 Q-Former 接口）、[MERIT](https://arxiv.org/abs/2604.11399)——主张"看似载体有损，实为接口/读者坏了"。我们的同读者配对设计天然控住读者变量，这条对我们威胁小，引来展示已控混淆即可。

### 3.4 单模型 latent 视觉 token 的通胀警告

[Beyond Visual Memory](https://arxiv.org/html/2606.01287v1)：Mirage 式 latent 视觉 token 的收益 78-100% 来自边界标记/格式而非视觉内容。→ B 臂验收必须带 marker-only / dummy-latent 对照（与 doc11 §4.3 同一清单）。

---

## §4 写作配方：motivation 段怎么落笔（v4 口径）

1. **开题引两翼已发表事实**：IsoBench（符号内容文本≫图像）+ CaptionQA/BLINK（感知内容图像≫文本）——"载体最优性依信息类型翻转"在输入层已是公案，但**无人在 agent 通信层受控检验**（Beyond Tokens 综述无此轴，HyLaT 判据是效率非保真）。
2. **接思想血统**：Media Richness / 双编码 / Larkin&Simon——40 年前的人类组织与认知版本，LLM-agent 版空缺。
3. **亮矩阵**：我们的 P2 信息类型×载体析因（同工作流同读者逐题配对）= 该假设的首次受控检验；判决=对称版不成立，**不对称版成立**：推理行文本饱和（免训练边界内）、感知行文本有损（像素 +6.0，30B 复核进行中）。
4. **调和反例**：用 §3.1 的四轴表一次性接掉 CIPHER/LatentMAS/Coconut/C2C；用 §3.2 收窄感知主张到"抗符号化残差"；正面披露 caption 政策效应（自家先知臂 +5.6）并给出双臂共享工具通道的防御。
5. **落到 B 臂**：不对称⇒latent 只该进感知信道+免训练已证不行⇒训练桥（引 Fedorenko 双向裁决作认知注脚）。

## §5 检索盲区

搜索引擎级检索，未爬引文图；[Visual Thoughts](https://arxiv.org/abs/2505.15510) 四种视觉思想形态、MMMU-Pro 46% ⚠、CapPO 51% ⚠、Q-ViD 表 3 数字等均须全文重核；涌现通信文献里"按内容类型分离散/连续信道"的论文未找到（若存在将是 EC 侧真正的 MOTIV-STATED，值得写作前再扫一次）；Fedorenko 被引全部在单模型 latent 推理侧，无人用于 agent 间载体选择——该引用角度目前是我们独占。

相关：[doc08 motivation 演化](08-motivation-evolution-info-type.md) · [doc11 baseline 血统调研](11-baseline-lineage-survey-2026.md) · [doc10 全臂图鉴](10-all-arms-atlas-explained.md)
