# 11 · 三条 baseline 血统 2026 年 8 月全量调研——过时了吗、撞车了吗、该换谁

> 缘起：用户两问——(1) 我们是否还在"给定视觉 agentic workflow 里测 baseline→找问题→改进"这条路上；(2) LatentMAS / Coconut / CIPHER 三个 baseline 是否过时、后续工作长什么样。
> 方法：9 个检索 agent、206 次网络检索、六个角度并行扫 2025-01 至 2026-08 文献 + 查漏批评员补两轮，共筛得 112 条、去重后约 70 篇。**证据等级注记：全部数字来自 abstract 页抓取（非全文核验），标 ⚠ 的条目连 abstract 都未直接抓到、仅来自搜索摘要——写 related work 前必须全文重核。**
> 读法：赶时间只读 §0 判决 + §6 必读清单。§1–§5 按"血统现状 → 领域大势 → 撞车地图 → 头名主张改口 → 可借配方"展开。

---

## §0 五句话判决

1. **方向没偏，且已多走一步**：三 baseline 里 2.5 个已在本 workflow 内测完并判阴（P2 战役，见 [10-all-arms-atlas-explained](10-all-arms-atlas-explained.md)），"找出的问题"=免训练 latent 通信全线无红利、唯一显著红利是感知行像素；当前 pixel-dividend-30b 是改进（B 臂 adapter）动工前的动机钉死，不是偏航。
2. **LatentMAS 没过时，反而升舱了**：ICML 2026 Spotlight（v4 2026-08-03 刚更新），仍是免训练派头牌——但它自己声明"同构模型限定、异构需可训 adapter"，**这句话就是我们 B 臂的立项引文**。
3. **Coconut 在自家血统内被取代**（RL 训练系 Latent-GRPO/SofT-GRPO/SWITCH 上位），**CIPHER 绝后**（2026 综述确认它仍是嵌入消息辩论的独苗、无直接后继）——我们的 CIPHER 阴性判决没有任何已发表工作能推翻，反而有了机制解释（Greedy Pitfall）。
4. **领域一年内整体从"免训练注入"转向"训练小桥"**（C2C→LCF→MoT→Interlat→Vision Wormhole），正撞我们 B 臂方向；**"VLM 视觉证据→纯文本 LLM 验证员 + 接收方决定看什么"这个格子截至 2026-08 仍空**，但四面墙都在逼近，其中 MACF（视频×latent×多 agent）必须全文核验。
5. **"+6.0 像素红利"的头名叙事必须改口**：caption 有损已被 CaptionQA/ViSIL/VideoSEAL 各自量化过，且存在 Vamos/ObjectMLLM 反向结果——我们独占的只剩"**同内容同读者、工作流内受控归因**"这一种测法。

---

## §1 三条血统各自走到哪了

### 1.1 LatentMAS：升舱 + 自曝软肋

- **本尊**：[Latent Collaboration in Multi-Agent Systems](https://arxiv.org/abs/2511.20639)（Zou et al., 2025-11 → **ICML 2026 Spotlight**，v4 2026-08-03）。免训练、末层隐状态自回归"latent thoughts"+共享 latent 工作记忆，9 个文本 benchmark 上 +14.6%/省 70-83% token。官方代码 Gen-Verse/LatentMAS。
  - 对我们的意义：P2 的 CL 臂（闭式 M+范数匹配注入，+0.4 n.s.）测的就是它的机制。**引用时必须用 ICML 终版并解释我们的阴性 vs 它的阳性**：它的 agent 共享同一 backbone 且任务是数学/代码推理；我们是异构管线+感知型任务。⚠ v3/v4 改了什么未核验——**若终版加了异构支持，对 B 臂是直接威胁，最优先查**。
- **亲儿子**：[RecursiveMAS](https://arxiv.org/abs/2604.25917)（同作者群，2026-04）——**已经做了"训练异构 agent 间连接件"**（RecursiveLink，+8.3%），把"训练桥连异构 agent"这半句新颖性吃掉了。剩给我们的：它纯文本推理域、无视觉、无感知证据角色。
- **旁支**：[LatentMem](https://arxiv.org/abs/2602.03036)（latent 记忆而非信道）、[When Less Latent Leads to Better Relay](https://arxiv.org/abs/2604.13349)（免训练：latent 中继只留 10-20% KV 就够，现成压缩配方）、[CanonicalMerge](https://arxiv.org/abs/2607.01308)（多路 latent 合并的顺序敏感性与免训练解法）、[Out of Sight, Not Out of Mind](https://arxiv.org/abs/2605.28214)（latent 信道的对抗攻击面——limitations 段的现成引文）。

### 1.2 Coconut：血统未死，本尊退位

- 现任最强代表已不是 Coconut/Soft Thinking，而是 **RL 训练系**：[Latent-GRPO](https://arxiv.org/abs/2604.27998)（2026-04，诊断三大失败模式后修 GRPO）、[SofT-GRPO](https://arxiv.org/abs/2511.06411)、[Soft Tokens, Hard Truths](https://arxiv.org/abs/2509.19170)（RL 训出的连续 CoT 也只在 pass@32 赢、pass@1 打平——**给 B 臂收益预期定调的清醒剂**）、[SWITCH](https://arxiv.org/abs/2606.13106)（latent 模式变成可学的开关策略）。
- **批评线已成建制**（全是我们阴性结果的机制解释，见 §4）：[Greedy Pitfall](https://arxiv.org/abs/2508.03440)、[Illusion of Superposition](https://arxiv.org/abs/2604.06374)（superposition 只在 from-scratch 模型出现，预训练模型塌缩到单 token）、[RL for Latent-Space Thinking](https://arxiv.org/abs/2512.11816)（RL 救不回来，仍输文本 CoT）、[Capabilities and Fundamental Limits](https://arxiv.org/abs/2602.01148)（理论：latent CoT 强在探索型任务、弱在精确计算）。
- **多模态分支爆发**（对我们既是竞品也是配方库）：训练系 [Mirage](https://arxiv.org/abs/2506.17218)（CVPR 2026，心理意象 latent token 三段课程）、[Monet](https://arxiv.org/abs/2511.21395)、[LVR](https://arxiv.org/abs/2509.24251)（"重建 query 相关视觉 token"这个训练目标可直接抄）、[ILVR](https://arxiv.org/abs/2512.05665)（稀疏选择性蒸馏，解视频 token 量大的问题）；**视频域**：[DyLaR](https://arxiv.org/abs/2608.04124)（2026-08，Qwen3-VL 上视频 QA latent 推理 54.0→58.2——**若 B 臂论文比视频 QA，这是评委必点的单模型 latent baseline**）、[Future-L1](https://arxiv.org/abs/2606.05769)。免训练反例：[DMLR](https://arxiv.org/abs/2512.12623)（test-time latent 优化 + 逐步视觉注入有效——**"免训练全阴"论述的最强反例，须解释：它靠检索注入新视觉信息，而我们的免训练臂只重排已有信息**）。
- Coconut 本体照旧没法当我们的臂（要训练、单模型），这半个缺口由 B 臂补——立论没变。

### 1.3 CIPHER：官方绝后，机制定罪

- [2026 多智能体辩论综述](https://arxiv.org/abs/2607.26212)（141 篇系统回顾）：辩论文献里 CIPHER **仍是嵌入消息的独苗**，无直接后继——概率加权嵌入消息这条路等于被社区默杀。
- 血统实际流向了**激活/隐状态/KV**：[Communicating Activations](https://arxiv.org/abs/2501.14082)（ICML 2025，免训练中间层激活嫁接，+27% 且 1/4 算力——**若要刷新"免训练最强对照臂"，换它**）、[State Delta Trajectory](https://arxiv.org/abs/2506.19209)（文字+状态增量并行发，"增强而非替换"设计）、[Thought Communication](https://arxiv.org/abs/2510.20733)（NeurIPS 2025 Spotlight 理论：可识别地拆共享/私有思想再传——顺带解释裸状态注入为何无效）。
- 我们观察到的 CIPHER 退化模式（软自回归 587/587 次 48 步顶满不自停）**未见发表**——这本身是论文里可写的一手观察。

---

## §2 领域大势：一年内从"免训练注入"翻到"训练小桥"

时间线一眼看穿：

| 时间 | 工作 | 桥 | 异构? | 载荷 |
|---|---|---|---|---|
| 2025-10 | [C2C](https://arxiv.org/abs/2510.03215)（ICLR 2026） | 训练 KV 投影+逐层门控 | ✅ 文本 LLM 对 | 文本语义（比文字信道 +3.1-5.4%） |
| 2025-11 | [Interlat](https://arxiv.org/abs/2511.09149)（ACL 2026） | 冻结发送方+训练 adapter+微调接收方 | 声称 ✅（⚠须细读） | 末层隐状态 |
| 2026-02 | [Vision Wormhole](https://arxiv.org/abs/2602.15382) | 训练通用视觉编解码器（蒸馏文字信道） | ✅ 4 个 VLM 家族 | **推理轨迹伪装成视觉输入** |
| 2026-05 | [LCF](https://arxiv.org/abs/2605.22863) | 13MB 小 adapter（C2C 的 4%） | ✅ | 不同上下文场景（比 C2C +23% EM） |
| 2026-05 | [HyLaT](https://arxiv.org/abs/2605.25421) | 逐 agent MLP adapter | ✅ | **混合协议：关键信号留文字、大块认知走 latent** |
| 2026-05 | [Bicameral](https://arxiv.org/abs/2605.11167) | ~1% 参数翻译网+抑制门，纯任务损失训出 | 同尺寸对 | 逐步隐状态耦合 |
| 2026-06 | [See What I See](https://arxiv.org/abs/2606.13594) | KV 变换（先重建后生成两阶段） | 仅跨尺寸 | KV cache（标题带视觉，实际纯文本） |
| 2026-07 | [MoT](https://arxiv.org/abs/2607.28979) | 翻译器混合+轨迹对齐修正损失 | ✅ 真跨架构 | KV cache |

**大白话**：整个领域用一年时间集体验证了我们 P2 的结论——免训练不行，得训桥；而且桥可以很小（13MB～1% 参数）。这对 B 臂是**双面消息**：立论被全领域背书（好），但"训练桥"本身已无新颖性可言（所以新颖性必须落在§3 说的那个格子上）。

---

## §3 撞车地图：格子还空，四面墙在逼近

我们的格子 = **训练桥 × 异构（VLM→纯文本 LLM）× 载荷是真视觉证据 × agentic 工作流 × 接收方决定看什么 × 视频 QA**。六路检索独立收敛出同一批最近邻，每个都缺至少两个要素：

| 最近邻 | 有什么 | 缺什么（=我们守住的） |
|---|---|---|
| [Vision Wormhole](https://arxiv.org/abs/2602.15382) ⚠须全文 | 训练桥+异构+走视觉通路 | **方向反了**（把推理轨迹注入 VLM 视觉口；我们是把视觉证据注入文本 LLM）、载荷非真视觉证据、无视频、接收方不点菜 |
| [MACF](https://arxiv.org/abs/2605.00444) ⚠**最高风险，须全文** | 视频+latent 协议+多 agent+"文字中介有损"同款动机 | agent 全是 MLLM（同质感知）、固定分段而非接收方主动索要。**若全文发现协调者是纯文本 LLM，撞车等级陡升** |
| [Interlat](https://arxiv.org/abs/2511.09149) ⚠异构细节须核 | 训练桥+声称异构+真 agent 通信 | 纯文本任务、无视觉载荷、接收方不点菜 |
| [RecursiveMAS](https://arxiv.org/abs/2604.25917) | 训练异构连接件+agent | 纯文本推理域、无感知 |
| [Zero-Shot Vision Encoder Grafting](https://arxiv.org/abs/2505.22664)（ICCV 2025，代码 facebookresearch/zero） | **机制上和 B 臂同物**：训练视觉 token 喂给未动过的纯文本 Llama-70B | 非 agent、非通信框架——但评委会拿它+[VLM connector 综述](https://arxiv.org/abs/2506.04788)把我们按进 BLIP-2/LLaVA connector 血统里，**必须抢先自我定位** |
| [L2V-CoT](https://arxiv.org/abs/2511.17910)（AAAI 2026） | 免训练跨架构 LLM↔VLM latent 注入 | 方向反+传风格不传内容+非 agent |
| [VideoSEAL](https://arxiv.org/abs/2605.12571)（ICML 2026） | agentic 视频 QA+"caption 轨迹不可信、须像素级验证"——**离我们头名结论最近** | 验证员是 MLLM 而非桥接的文本 LLM、无 latent 信道 |
| [Temporal CoT](https://arxiv.org/abs/2507.02001)（NeurIPS 2025）/ [WorldMM](https://arxiv.org/abs/2512.02425)（CVPR 2026） | "接收方决定看什么"已单独存在 | 都在单模型/多模态模型内部，无跨模型边界 |

**先例审计更新**（对照 [01-prior-art-audit](01-prior-art-audit.md)）：当年判决"真空只剩'发送方是 agent'+'接收方决定看什么'"在 2026-08 仍成立，但两条都从"空地"变成了"交集地"——单独哪条都有人做过，**只有六要素交集没人占**。novelty 段写法必须从"没人做过 X"改成"没人在 agentic 视频证据链上同时做到 X∧Y∧Z"。

---

## §4 头名主张改口 + P2 阴性的增值

### 4.1 "+6.0 像素红利"面临的四类先例

1. **同款量化已发表**：[CaptionQA](https://arxiv.org/abs/2511.21025)（图像域：换 caption 掉最多 32%）、[ViSIL](https://arxiv.org/abs/2601.09851)（视频域信息论框架：关键帧摘要比文字摘要 +7% VQA）、[RAVEN](https://arxiv.org/abs/2606.25206)（机器人域直接说"避开有损的图转文"）。→ "caption 有损"不能再当发现讲。
2. **反向结果活着**：[Vamos](https://arxiv.org/abs/2311.13627)（ECCV 2024：caption 够用、视觉嵌入几乎无增益）+ 同组 [ObjectMLLM](https://arxiv.org/abs/2504.07454)（ICCV 2025：结构化对象信息转纯文本反而最强）。→ 评委必拿这两篇打我们，**须解释我们何以反转**：他们是"整段视频→固定表征→单读者"，我们是"验证员按假设定向索证"的工作流内测法——信息需求是 query 条件化的，caption 恰恰在长尾细节上丢分（与 [GEASS](https://arxiv.org/abs/2605.01733) 的"caption 帮全局题、害细节题"发现互证）。
3. **caption 臂强度要求被抬高**：[SiLVR](https://arxiv.org/abs/2505.24869)（TMLR：纯 caption+字幕在长视频 benchmark 屠榜）、[LVNet](https://arxiv.org/abs/2406.09396)（问题条件化选帧再 caption）、[Nar-KFC](https://arxiv.org/abs/2505.24158)（ICLR 2026：关键帧+叙事混合）。→ 若 D 臂 caption 不到 SiLVR 强度，评委一句"你 caption 太弱"就能泄掉 +6.0。**反驳弹药已在手**：探针战役证明书的来源（LaViLa vs 我们的书）对 8B 严格 0.0 分差、[CapQuiz](https://aclanthology.org/2026.acl-long.777/)（ACL 2026）提供 caption 质量的 QA 效用认证法。
4. **语言先验混淆**：[MVU](https://arxiv.org/abs/2403.16998)（ICLR 2025：长视频 benchmark 不看视频也能拿高分）。→ 需要一条"无证据盲臂"对照佐证 +6.0 确实来自视觉证据（我们的 A/D/C 差分设计部分免疫此问题，但明写更稳）。

**存活的头名主张**（收窄后反而更硬）：*首个在真实 agentic 视频 QA 工作流内、同内容同读者、逐题配对（McNemar）的 caption-vs-像素受控归因；并给出免训练 latent 全谱系（概率/软 token/隐状态注入）的同工作流阴性对照。* ——这个测法确实没人做过（scoop-check 角度明确确认"无人在 agent 环内跑过同内容 frames-vs-captions 受控对比"）。

### 4.2 P2 阴性结果的 2026 增值：从"我们没测出来"变成"领域已解释为什么"

| 我们的阴性臂 | 已发表的机制解释 | 引文价值 |
|---|---|---|
| Σp·E 软 token（T+/P2-1/CP 全阴） | [Greedy Pitfall](https://arxiv.org/abs/2508.03440)：接收方实际只读 argmax，软信道免训练不携带增量信息 | 阴性≠实现瑕疵，是机制必然 |
| CL（LatentMAS 式注入净零） | [Illusion of Superposition](https://arxiv.org/abs/2604.06374)：预训练模型塌缩单 token；[Thought Communication](https://arxiv.org/abs/2510.20733)：裸状态混杂共享/私有内容 | 免训练裸注入先天无效 |
| CIPHER 臂（不自停退化） | 未见发表——一手观察 | 可作我们的贡献点写 |
| 整体"免训练全阴→训练桥" | [MCOUT](https://arxiv.org/abs/2508.12587)（同机制加训练即转阳）、[What's Holding Back LVR](https://arxiv.org/abs/2605.18445)（latent token 常因果惰性；**其 dummy-token 替换协议应搬进 B 臂验收**）、[因果审计](https://arxiv.org/abs/2607.26773)（latent 通信收益须过替换对照才算数） | B 臂立论三重背书 + 现成验收协议 |

### 4.3 B 臂验收协议要提前抬标（评委已有武器）

2026 的批评线已经把"latent 信道真的在传信息吗"的举证标准立起来了：**dummy/乱序 latent 替换对照**（2605.18445、2607.26773）、**boundary-marker 对照**（[Beyond Visual Memory](https://arxiv.org/abs/2606.01287)：好几个方法的收益 78-100% 来自格式而非内容）、**shortcut 抑制**（[Unsilencing](https://arxiv.org/abs/2605.02735)：接收方若还能读文字就会绕开 latent——B 臂训练时 D 臂式 caption 不能同时在场，或需显式路由）。这三条写进 B 臂实验卡，比评委先动手。

---

## §5 该换谁 + 可借配方

### 5.1 对比表格刷新建议（讨论稿，未立卡）

- **不必重跑的**：P2 四主对比——阴性结论有机制文献背书，且无任何"免训练但更强"的新方法出现（唯一候补 DMLR 机制不同，可文字解释而非补臂）。
- **B 臂论文的对比表新面孔**（按必要性排序）：
  1. 免训练 latent 对照：**官方 LatentMAS**（Gen-Verse 代码替代我们的复现）＋可选 [Communicating Activations](https://arxiv.org/abs/2501.14082)；
  2. 训练信道对照：**C2C 或 LCF** 适配进 workflow（LCF 更合身——我们的发送方/接收方本来就不同上下文）；
  3. 强 caption 臂：现有 D 臂 + SiLVR 式增强（至少论述层面对齐）；
  4. 单模型视频 latent：**DyLaR**（引用+概念对比即可，跑不跑看篇幅）；
  5. 混合臂：Nar-KFC 式"关键帧+文字"免训练混合——**若 B 臂赢不了这个免训练混合，训练就白费**，这是最诚实的下界。
- **Host 系统彩蛋**：我们的宿主 VideoHV-Agent（[04-host-frameworks-explained](04-host-frameworks-explained.md) §2）已发表为 **CVPR 2026**（[Think, Then Verify](https://arxiv.org/abs/2603.04977)）。好消息：宿主从"GitHub 项目"升级成同行评审系统，实验平台可信度+1；待办：核对 camera-ready 与我们移植的 GitHub 版有无协议差异，并正式引用。

### 5.2 B 臂训练配方货架（全部可抄）

| 配方 | 来源 | 用在哪 |
|---|---|---|
| surrogate 训练→零样本嫁接进大模型 | [Grafting](https://arxiv.org/abs/2505.22664)（代码公开） | 桥在小代理 LM 上训、嫁接进 30B 验证员，省算力 |
| 隐状态对齐蒸馏（teacher=读 caption 的运行） | [CODI](https://arxiv.org/abs/2502.21074) | 桥输出对齐"验证员读同帧 caption 时的状态"，D 臂即天然 teacher |
| 冻结发送方+训 adapter+微调接收方 | [Interlat](https://arxiv.org/abs/2511.09149) | 整体训练框架模板 |
| 先重建后生成两阶段课程 | [See What I See](https://arxiv.org/abs/2606.13594) | 稳定桥训练 |
| 13MB 级小 adapter+只传增量 | [LCF](https://arxiv.org/abs/2605.22863) | 控制训练预算与上下文占用 |
| 注入层位选择：中层优于输入层 | [How Visual Reps Map](https://arxiv.org/abs/2506.11976) | 与我们 KVCOMM 分析的中后层漂移发现互证 |
| latent 载荷 90% 可丢 | [When Less Latent](https://arxiv.org/abs/2604.13349) | 180 帧证据的压缩预算 |
| 可解释性：latent 渲染回图像/最近邻读出 | [Latent Sketchpad](https://arxiv.org/abs/2510.24514)、[LatentLens](https://arxiv.org/abs/2602.00462) | 审计桥到底传了什么，论文可解释性章节 |
| 文本 LLM 天生有视觉先验（桥可以小的理由） | [Learning to See Before Seeing](https://arxiv.org/abs/2509.26625)、[Penguin-VL](https://arxiv.org/abs/2603.06569) | motivation 段 feasibility 引文 |

---

## §6 必读清单与未决项

**全文必读五篇（按风险排序，读完才能写 related work）**：
1. [MACF](https://arxiv.org/abs/2605.00444) ——确认协调者是否纯文本 LLM（是→撞车等级陡升）；
2. [LatentMAS v4](https://arxiv.org/abs/2511.20639) 改版说明——是否已加异构支持；
3. [Vision Wormhole](https://arxiv.org/abs/2602.15382) ——载荷与方向的精确边界；
4. [Interlat](https://arxiv.org/abs/2511.09149) ——"异构也行"的实际覆盖范围；
5. [VideoSEAL](https://arxiv.org/abs/2605.12571) ——与我们头名结论的精确差分 + camera-ready 数字。

**检索盲区（诚实记录）**：全程搜索引擎级检索，未爬 Semantic Scholar 引文图——LatentMAS/CIPHER 2026-06 之后的低调直接后继可能漏网；非 arXiv 的 venue-only 论文覆盖不全；多数 venue 标注未经 camera-ready 核实。约每季度值得重扫一次撞车角度（§3 那张表的六要素查询串可复用）。

**与 memory 的对账**：[01-prior-art-audit](01-prior-art-audit.md) 的 NavGPT-2/Sterner 判决不变；"接收方决定看什么"的负结果先例（SR 67.52→21.46）仍在，但 Temporal CoT/WorldMM 证明该机制在 2025-2026 已有正结果实现——负结果先例的杀伤力下降，须改为"实现路线敏感"表述。
