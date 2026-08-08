# 多模态 × LatentMAS：现有工作盘点

> **这份文档回答什么**：想做「多模态 + LatentMAS 式隐空间多智能体协作」，**已经有谁做过、做到哪一步、代码在哪**，以及**真正剩下的空位在哪**。
>
> **检索时点**：2026-08-01
> **版本**：**v2**（v1 的核心结论被推翻，见下）
>
> **配套**：代码解析见 `code-to-paper-mapping.md`；论文精读见 `LatentMAS_论文总结.md`。

---

## 0. v1 → v2 的三条更正（先看这个）

v1 基于 8 次定向检索得出「交叉区基本空白」。随后一轮 9 智能体并行扫描（7 个检索角度 + 完备性批评 + **专门试图反证的对抗智能体**，共 22+22 次检索）**推翻了其中两条**：

| v1 的说法 | 实际情况 | 更正依据 |
|---|---|---|
| ❌「多模态 × latent MAS 的交叉基本是空的」 | **至少 4 篇已经在做**，其中 L²-VMAS 与该设想高度重合 | §2 |
| ❌「共享视觉前缀 KV 是显然但没人做的切口」 | **一整片文献**：Kamera / VLCache / MPIC / Omni-Flow / OxyGen / EPD 拆分…… | §6 |
| ✅「$W_a$ 在 VLM 上没有定义、理论断掉」 | 成立，且 L²-VMAS 用几乎相同的措辞独立指出了这一点 | §8.1 |

### 为什么 v1 会错——这条比结论本身更有用

对抗智能体给出的诊断值得原文抄下来：

> *"the field's own index is stale. The Awesome-Latent-Communication list and its companion survey 'Beyond Tokens' between them name exactly ONE multimodal method — Vision Wormhole — and miss all four counterexamples. The multimodal work is not being filed under 'latent communication'; it is filed under vision/video/medical-imaging venues and uses vocabulary like 'latent memory', 'communication tokens', and 'visual flow' instead of 'KV-cache transfer'. **The gap is bibliographic, not scientific.**"*

**教训**：沿着 latent-MAS 那条文献线检索，会稳定地复现「多模态是空白」这个错误结论——因为多模态那边的人**不用这套词**。有效的检索姿势是**去掉 "KV cache" 和 "LatentMAS"，换成任务域词（video / VQA / radiology / navigation）+ "instead of text"**。

---

## 1. 修正后的结论

**交叉区不是空的，但也没被占满。** 准确的状态是：

| 象限 | 有人做了吗 | 代表 |
|---|---|---|
| 多模态 + latent 通信 + **要训练** | ✅ **已被占据** | L²-VMAS、MACF、Vision Wormhole、CogRad |
| 多模态 + 非文本通信 + **免训练** | ❌ **空**（v2 曾误标 ViF；v3 更正：ViF 有可训练 ViFBlock + 两阶段训练，且公开代码是 stub） | —— |
| 多模态 + **KV 工作记忆** + 免训练 | ❌ **仍然空着** | —— |
| 「LatentMAS 直接搬到 VLM 上能跑到什么程度」这个数字 | ❌ **无人报告** | L²-VMAS 引了 LatentMAS 但**没做实验对比** |

最后两行就是现在真正可下手的地方。

---

## 2. ★ 已经在做「多模态 latent MAS」的四篇

### 2.1 L²-VMAS — 与该设想重合度最高，必须先读

**Dual Latent Memory for Visual Multi-agent System**
[arXiv 2602.00471](https://arxiv.org/abs/2602.00471) · Xinlei Yu, Chengming Xu, … Shuicheng Yan · 2026-01-31 提交，2026-06-05 修订

| | |
|---|---|
| **传什么** | 两组 latent memory，**每个单元是 key-value pair，value 存 hidden state 序列**：<br/>· **Perception memory**：多粒度视觉特征（层次下采样的 local/regional/global）<br/>· **Thinking memory**：按**高熵语义边界**切分的生成 hidden state chunk |
| **怎么取** | 下游 agent **主动查询**（熵触发 + 可学习门控），不是被动接收 |
| **要训练** | **要**。三阶段 RL：①随机激活优化 memory 构建 → ②冻结 memory、训编排 → ③端到端联合。GQA 数据集，8×H200 |
| **Backbone** | GLM-4.1V-9B-Thinking、InternVL-3.5-8B、LLaVA-OV-1.5-8B、Qwen3-VL 2B/4B/8B/32B |
| **Benchmark** | MMbench、MMStar、RealWorldQA、SimpleVQA、MuirBench、BLINK、MVBench、LVBench |
| **战果** | +2.7~5.4% acc，token −21.3~44.8%（Qwen3-VL-8B-Thinking：+5.4% / −43.6%） |
| **Repo** | **无** |
| **与 LatentMAS** | **引用了**（"Zou et al., Latent collaboration in multi-agent systems"），并论证这类方法 *"are not directly transferable to VMAS"*，理由是没处理 **high-dimensional visual inputs** 和 **perceptual-cognitive information conflation**。**但没有做任何实验对比。** |

> **这一条最关键**：核心主张（agent 轮次越多、准确率反降而 token 爆炸 → 把通道搬进 latent 空间）和「多模态 LatentMAS」几乎是同一句话。**如果目标是发一篇提出新方法的论文，这个位子已经被占了。**
>
> 但它留了两个明确的口子：**(a) 它要三阶段 RL，不是 training-free；(b) 它把 LatentMAS 当 related work 挂着，却从没量过 LatentMAS 在 VLM 上的实际表现。**

### 2.2 ViF — 视觉 relay token 通道（⚠️ v3 更正：**不是免训练**，代码是 stub）

**Visual Multi-Agent System: Mitigating Hallucination Snowballing via Visual Flow**
[arXiv 2509.21789](https://arxiv.org/abs/2509.21789) · Xinlei Yu 等（**与 L²-VMAS 同一作者组**）· **[github.com/YU-deep/ViF](https://github.com/YU-deep/ViF)** ✓

- **问题定义很好用**：提出 *multi-agent visual hallucination snowballing*——幻觉在一个 agent 处萌芽、被后续 agent 放大，根因是**过度依赖文本流中转视觉信息**。用 turn-/layer-/token-wise 注意力分析证明：视觉注意力分配随 agent 轮次递减。
- **机制**：挑出**中间层注意力单峰**的那批视觉 token（最能保住视觉证据、但在深层 agent 轮次里逐渐消失），重新语境化后插进**下一个 agent** 的输入序列，再做 attention reallocation 放大这个模式。
- ⚠️ **不是免训练**【一手核实】：`vif/models/vif_block.py` 的 `ViFBlock` 是可训练 `nn.TransformerEncoder`，repo 有 `train_stage1.py`/`train_stage2.py` 两阶段，Stage 2 连 base LLM 都解冻。摘要的 "plug-and-play" 指对 MAS 拓扑即插即用。
- ⚠️ **公开代码跑不了**【一手核实】：`vif/` 全包 257 行；`vif/models/base_stub.py` 的 `BaseVLMStub.forward` 直接 `hidden = torch.randn(B,T,D)`，从未接过真实 VLM。
- 8 个 benchmark × 4 种 MAS 结构 × 10 个 base model。
- 未引用 LatentMAS。

> ⚠️ **v2 曾建议「从这个 repo 起步最省事」——已作废**。唯一可跑的骨架是 Vision Wormhole 的 `heterogeneous-latent-mas`（LatentMAS 的 fork）。ViF 的价值在**问题定义**（hallucination snowballing + 注意力分析）和**公式**，不在代码。

### 2.3 MACF — 长视频场景

**Scaling Video Understanding via Compact Latent Multi-Agent Collaboration**
[arXiv 2605.00444](https://arxiv.org/abs/2605.00444)

把长视频切段交给各有预算的局部 agent，agent 把局部观测编码成**共享嵌入空间里的 compact token** 交给中央协调者——摘要明确写了 *"reliance on textual intermediates"* 是要解决的问题。**需要 curriculum training**。Video-MME / LongVideoBench / LVBench / MLVU。

### 2.4 Vision Wormhole — 借视觉通路，但做的是文本任务

**The Vision Wormhole: Latent-Space Communication in Heterogeneous Multi-Agent Systems**
[arXiv 2602.15382](https://arxiv.org/abs/2602.15382) · **[github.com/xz-liu/heterogeneous-latent-mas](https://github.com/xz-liu/heterogeneous-latent-mas)** ✓（v1 说「没找到 repo」，**更正**）

- **机制**：发送方跑 LatentMAS 式 latent rollout → Perceiver 式编码器压成 $K_u$ 个 universal token → 各模型族的解码器变成带门控的扰动，**残差加到一张 dummy image 的视觉嵌入上** → 接收方 VLM 从自己的视觉端口「吃」进对方的思维。
- **要训练**（label-free 师生蒸馏，以文本通道为教师）。
- **9 个 benchmark 全是文本**：GSM8K、ARC-E/C、GPQA、MedQA、MBPP+、HumanEval+、AIME24/25。**证实它用多模态当手段、不解决多模态任务。**
- 对 LatentMAS 的评价（原文）：*"Training-free variants are effective when agents share a backbone or have compatible internal state formats, but this assumption is restrictive for heterogeneous teams built from independently trained model families."*
- **关于「naive 移植会崩」——原文比传言弱**：实际表述是 *"A heterogeneous LatentMAS-Hybrid adaptation in a GSM8K stress setting; the adaptation is unstable on the evaluated cross-provider pairs (Appendix F)"*。即：**跨厂商配对、GSM8K 文本压力测试下不稳定**，不是「VLM 上 degenerate」。检索智能体给出的强表述已被本人核对推翻，**别在 related work 里引用那个强版本**。

### 2.5 CogRad — 放射科报告

[arXiv 2607.03853](https://arxiv.org/abs/2607.03853)，agent 间传注意力权重和区域嵌入。更像训练好的流水线而非自主 agent，作为反例强度最弱。

---

## 3. 纯文本多模态 MAS（可直接拿来当骨架 / baseline）

这一类通信仍是文本，正是「换成 latent 通道」的**单变量改造对象**。

| 工作 | 场景 | Repo |
|---|---|---|
| **ViF** (§2.2) | 通用，4 种 MAS 拓扑 | **[YU-deep/ViF](https://github.com/YU-deep/ViF)** ✓ |
| **MDocAgent** [2503.13964](https://arxiv.org/abs/2503.13964) | 长文档理解，5 个 agent（general/critical/text/image/summarizing）+ 双路 RAG。**每个 agent 都重编码同一批页面图像**——正是共享 KV 要消掉的冗余。报告 +12.1% | **[aiming-lab/MDocAgent](https://github.com/aiming-lab/MDocAgent)** ✓ |
| **BeMyEyes** [2511.19417](https://arxiv.org/abs/2511.19417) | 小 VLM 当 perceiver 描述图像 + 大文本 LLM 当 reasoner，多轮文本对话。**「视觉过文本瓶颈」的最纯形态，最干净的 ablation 起点**。MMMU / MMMU-Pro / MathVista / MathVision | 无 |
| **Visual Para-Thinker++** [2606.09290](https://arxiv.org/abs/2606.09290) | 单策略多角色；**推理引擎对图像只做一次共享 prefill 建公共视觉 KV**，再并行解码各角色分支 KV。2.5× 吞吐 | 无 |
| **VipAct** (AAAI 2026) [2410.16400](https://arxiv.org/abs/2410.16400) | **免训练** orchestrator + 专家 VLM + 视觉工具，细粒度感知任务 | 无 |
| **MACT** [2508.03404](https://arxiv.org/abs/2508.03404) | 视觉文档，4 agent + 按复杂度分配 test-time 算力。+9.9~11.5% | 无（称将发布） |
| **MMedAgent-RL** (ICLR 2026) [2506.00555](https://arxiv.org/abs/2506.00555) | 医疗 VQA，RL 训协作策略。**+23.6%，是我见到的最高多模态 MAS 增益** | 无 |
| **M3Prune** [2511.19969](https://arxiv.org/abs/2511.19969) | 多模态多智能体 RAG 的通信图剪枝（拓扑侧省钱） | 无 |
| **Training-Free MLLM Orchestration** (ICML 2026) [2508.10016](https://arxiv.org/abs/2508.10016) | 控制 token 路由 + **刻意选择 text-centric memory 而非嵌入**。同为免训练、同档会议——**这是必须比的 honest control，它的设计理由就是你要反驳的论点** | 无 |

### 3.1 Benchmark 选哪个

| Benchmark | 特点 | Repo |
|---|---|---|
| **COMMA** (TMLR 2025) [2410.07553](https://arxiv.org/abs/2410.07553) | **唯一专门隔离「智能体间通信质量」的多模态 benchmark**：两个 agent 持有**不对称**视觉信息，必须交流才能解题。杀手级动机数据：GPT-4o / o4-mini / R1-Onevision / LLaVA-CoT 在 agent-agent 协作上**几乎打不过随机基线**——即文本通道被实测证明是瓶颈。<br/>⚠️ 它**刻意设定为「只能用语言通信」**，换成 latent 通道会改变 benchmark 前提，需要在写作上处理 | **[tossowski/COMMA](https://github.com/tossowski/COMMA)** ✓ |
| **MECoBench** [2606.31966](https://arxiv.org/abs/2606.31966) | 2026 年最新，具身环境多模态 agent 协作，评测协议**专门区分「真协作收益」与「单体能力」** | **[q-i-n-g/MECoBench](https://github.com/q-i-n-g/MECoBench)** ✓ |
| **L²-VMAS 那 8 个** | MMbench / MMStar / RealWorldQA / SimpleVQA / MuirBench / BLINK / MVBench / LVBench | 通用 |

---

## 4. 单模型多模态隐式推理（「推理端」的技术储备）

| 工作 | 机制 | Repo |
|---|---|---|
| **Heima** [2501.19201](https://arxiv.org/abs/2501.19201) | **第一个多模态 latent CoT 框架**（2025-01），早于本表所有多模态条目。related work 漏了它会显得不熟悉领域 | — |
| **Mirage** (CVPR'26) [2506.17218](https://arxiv.org/abs/2506.17218) | 解码时插隐式视觉 token，hidden state 当下一个 token 续接，不生成像素。蒸馏→文本监督→RL 三段 | **[UMass-Embodied-AGI/Mirage](https://github.com/UMass-Embodied-AGI/Mirage)** ✓ |
| **IVT-LR** (ACL'26 Findings) [2510.12603](https://arxiv.org/abs/2510.12603) | 每步 = 隐文本(上步 hidden) + 隐视觉(选中图像 embedding)。+5.45% / >5× | **[FYYDCC/IVT-LR](https://github.com/FYYDCC/IVT-LR)** ✓ |
| **Render-of-Thought** (ACL'26) [2601.14750](https://arxiv.org/abs/2601.14750) | 文本 CoT 渲染成图，冻结视觉编码器编码成压缩推理 token。**这一族里最接近免训练的**，3-4× 压缩 | **[TencentBAC/RoT](https://github.com/TencentBAC/RoT)** ✓ |
| **ReGuLaR** [2601.23184](https://arxiv.org/abs/2601.23184) | 变分隐推理，隐思维是**分布**而非点。想让各 agent 采样到不同隐思维、拿到真多样性，这是唯一现成的形式化 | **[FanmengWang/ReGuLaR](https://github.com/FanmengWang/ReGuLaR)** ✓ |
| **DeepSeek-OCR** [2510.18234](https://arxiv.org/abs/2510.18234) | 文本渲染成图后 7-20×（最高 60×）token 压缩，<10× 时保真 ~97%、20× 时 ~60% | **[deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)** ✓ |
| OneLatent / ImgCoT / CoLVR / LVR / Monet / CoVT | 单 token 压缩、渲染成图、对比优化等 | 见 [Awesome-Latent-CoT](https://github.com/EIT-NLP/Awesome-Latent-CoT) |

> **DeepSeek-OCR 那组数字是「视觉通道容量」的唯一硬指标**：如果打算让 agent 消息走视觉通路，它就是你的 bits-per-vision-token 预算和退化曲线。同时也是最便宜的 baseline——把 agent A 的 CoT 渲染成图直接喂给 agent B。

---

## 5. 多智能体隐通信（纯文本，「通信端」的技术储备）

| 工作 | 传什么 | 训练 | Repo |
|---|---|---|---|
| **LatentMAS** (ICML'26 spotlight) [2511.20639](https://arxiv.org/abs/2511.20639) | 逐层 KV（prefill + 潜在思维） | 否 | [Gen-Verse/LatentMAS](https://github.com/Gen-Verse/LatentMAS) ✓ |
| **KVCOMM** (NeurIPS 2025) [2510.12872](https://arxiv.org/abs/2510.12872) | **免训练**复用上游 KV：维护一个「cache 偏移」锚点池，按最近邻加偏移 + 位置平移 | 否 | [HankYe/KVCOMM](https://github.com/HankYe/KVCOMM) |
| **C2C** [2510.03215](https://arxiv.org/abs/2510.03215) | 全 KV + 学到的投影融合器 + 逐层门控 | 是 | [thu-nics/C2C](https://github.com/thu-nics/C2C) ✓ |
| **AC** (ICML 2025) [2501.14082](https://arxiv.org/abs/2501.14082) | 在中间层暂停 B 的前向，把 A 的激活直接拼/加进去再续算。零参数 | 否 | — |
| **Interlat** (ACL 2026) [2511.09149](https://arxiv.org/abs/2511.09149) | 只传**一个**末层末 token hidden state | 部分 | [XiaoDu-flying/Interlat](https://github.com/XiaoDu-flying/Interlat) ✓ |
| **SDE** (EMNLP 2025) [2506.19209](https://arxiv.org/abs/2506.19209) | hidden state 的**增量**轨迹 | 否 | [LittleDinoC/StateDelta](https://github.com/LittleDinoC/StateDelta) ✓ |
| **CIPHER** (ICLR 2024) [2310.06272](https://arxiv.org/abs/2310.06272) | 词表分布加权的 embedding 期望（"soft token"）。**这一族的源头** | 否 | [chaudatascience/…](https://github.com/chaudatascience/cipher_multiagent_debate) ✓ |
| **PACT** [2606.05304](https://arxiv.org/abs/2606.05304) | 把自由文本压成结构化 action-state record | — | — |

### 5.1 三个「不能不知道」的发现

1. **KVCOMM 的偏移锚点法**（免训练，解决前缀不一致下的 KV 复用）——多模态 MAS 里重叠上下文就是**图像 token 块**，又长、各 agent 又完全相同，这正是让它在不同角色 prompt 下合法复用的钥匙。
2. **SDE 的负面结论**：直接注入原始 hidden state **会伤害**接收方，传**增量**才有用。视觉 hidden state 的范数/各向异性与文本差异更大 → **多模态版可能必须用 delta 或按模态归一，而不是裸注入**。
3. **PACT 是 text 侧的强 baseline**：latent 通道号称的 token 节省，可能只要「别发散文」就能拿到大半。不控这一项，审稿人一定问。

### 5.2 已经长出来的质疑与安全分支

[Do Latent Channels Actually Communicate? A Causal Audit](https://arxiv.org/html/2607.26773v1)（**动手前先读**）· [LCGuard](https://arxiv.org/pdf/2605.22786) · [When Latent Agents Lie](https://arxiv.org/html/2606.28958v1) · [Out of Sight, Not Out of Mind](https://arxiv.org/html/2605.28214v1) · [Cache Merging as CRDT](https://arxiv.org/pdf/2607.01308)

---

## 6. VLM 的 KV 复用与压缩（**v1 说「没人做」，实际一大片**）

v1 §5.1 提的「多个 agent 看同一张图、视觉 KV 纯冗余、共享视觉前缀」这个切口，**系统侧早已被反复做过**：

| 方向 | 代表 | Repo |
|---|---|---|
| **位置无关的多模态 KV 复用** | **Kamera** [2606.23581](https://arxiv.org/abs/2606.23581)：KV 存成 position-free，复用时对 K 做精确 RoPE 重旋转 + 低秩 conditioning patch | — |
| | **MPIC / InfoBlend** [2502.01960](https://arxiv.org/abs/2502.01960)：按多模态输入为键把 KV 落盘，并行加载 + 重算首段 | — |
| **图像级 cache 命中** | **VLCache** [2512.12977](https://arxiv.org/abs/2512.12977)：哈希原图，命中则同时载入视觉编码器输出和解码器 KV，只重算 2-5% | [Odysseusq/VLCache](https://github.com/Odysseusq/VLCache) |
| **分布式 KV 共享** | **Omni-Flow** [2606.31093](https://arxiv.org/abs/2606.31093)：GPU/CPU/SSD 分页 KV + 多模态前缀匹配 | [meituan-longcat/omni-flow](https://github.com/meituan-longcat/omni-flow) |
| **共享观测跨任务 KV** | **OxyGen** [2603.14371](https://arxiv.org/abs/2603.14371)：VLA 里同一帧相机图只 prefill 一次，视觉 KV 跨任务共享 | — |
| **回看图像** | **PRCR** [2606.26631](https://arxiv.org/abs/2606.26631)：视觉 K/V 连同空间坐标存边表，重看时重绑位置，免 replay | — |
| **视觉 KV 压缩/淘汰** | LOOK-M (EMNLP'24 Findings)、VL-Cache、AirCache、HybridKV (ACL'26)、FlashCache (CVPR'26)、MEDA、KVCapsule、FlowMM、LightVLM、GUI-KV、STaR-KV | LOOK-M 有 [repo](https://github.com/SUSTechBruce/LOOK-M) |
| **服务侧拆分** | EPD disaggregation [2501.05460](https://arxiv.org/abs/2501.05460) → HydraInfer / RServe / TriInfer(MLSys'26) | — |
| **具身 KV 记忆** | KEEP (DAC 2026) [2602.23592](https://arxiv.org/abs/2602.23592) | [PKU-SEC-Lab/KEEP…](https://github.com/PKU-SEC-Lab/KEEP_Embodied_Memory) |
| **跨 LLM KV 复用（早于全部 2026 工作）** | **DroidSpeak** (NSDI 2026) [2411.02820](https://arxiv.org/abs/2411.02820)：明确框定为多智能体流水线的 inter-LLM 通信 | — |

**结论**：不能再把「共享视觉前缀」当新意讲，只能当**工程前提**引用。真正没被做的是「共享视觉 KV **作为 agent 间语义通道**」，而不是「共享视觉 KV 省算力」。

---

## 7. 更早的血脉：不引会被 reviewer 抓

完备性批评智能体点出的最大盲区，**这条对投稿影响很大**：

### 7.1 协同感知（V2X / V2V）——一整个做了 6 年的领域

多模态 agent **传中间神经特征而非原始数据或文本**，是自动驾驶协同感知的标配：

- **When2com** (CVPR 2020)：学一个 handshake 决定**何时发、发给谁**
- **Where2comm** (NeurIPS 2022)：决定**发隐特征的哪一块空间切片**
- **DiscoNet** (NeurIPS 2021)：从「看得见全部」的上界做师生蒸馏到带宽受限的隐通道——**这正是 Vision Wormhole 的训练配方，早了五年**
- 目录：[Little-Podi/Collaborative_Perception](https://github.com/Little-Podi/Collaborative_Perception)

> ⚠️ **「多模态 + latent + 多智能体是空白」这个说法，只在「LLM 形状的系统」里成立。CVPR 背景的审稿人不会接受这个框定。**

### 7.2 术语的源头

- **Relative Representations Enable Zero-Shot Latent Space Communication** (ICLR 2023)——**"latent communication" 这个词就是这篇造的**，其基于锚点的免训练对齐，正是 Vision Wormhole 用 ridge regression 重新推导的东西
- **DIAL** (NeurIPS 2016)——连续 agent 间消息、梯度穿过通道的起点
- **Coconut** [2412.06769](https://arxiv.org/abs/2412.06769)——「hidden state 当下一个输入」这个循环的源头，LatentMAS / Mirage 都在用它。v1 和 Beyond Tokens 综述都漏了本体

---

## 8. 真正剩下的空位（修正版）

### 8.1 $W_a$ 在 VLM 上没有定义 —— **唯一一条经受住了反证的理论空位**

LatentMAS 的闭式对齐算子

$$W_a=(W_{out}^\top W_{out}+\lambda I)^{-1}W_{out}^\top W_{in}$$

**只对文本词表成立**（$W_{in}$ = token embedding 表、$W_{out}$ = LM head）。VLM 的视觉输入**根本不过 $W_{in}$**（走 vision encoder + projector）。于是末层 $h$ 的对齐目标是文本嵌入分布、视觉 token 分布，还是混合？

- 对齐到文本嵌入 → 潜在思维被拽向语言侧，视觉信息可能被压掉
- 对齐到视觉 token 分布 → **没有闭式解**（projector 不可逆、无「词表」可做最小二乘）
- 混合 → 比例成为新超参，training-free 卖点岌岌可危

**Theorem A.1 的 Wasserstein 上界在这里整个失效**（其推导前提就是「存在一张 token 嵌入表作为对齐目标」）。

L²-VMAS 用几乎同样的措辞独立指出了这个问题（*"perceptual-cognitive information conflation"*），**但它是用三阶段 RL 绕过去的，没有给出免训练的答案**。这条空位仍然开着。

> 相关隐患：纯文本上，若 `tie_word_embeddings=True`，$W_a$ 已经退化成近似单位阵（见 `code-to-paper-mapping.md` §4.2）。多模态版别继承一个本来就虚的组件。

### 8.2 免训练 + KV 级 + 多模态，三者同时成立

- L²-VMAS：多模态 ✅ KV 形状 ✅ 免训练 ❌
- ViF：多模态 ✅ 免训练 ✅ KV 级 ❌（传的是 visual token）
- LatentMAS：KV ✅ 免训练 ✅ 多模态 ❌

**三者交集是空的。** 但要注意：这可能是**因为它做不到**（SDE 的负面结论、Vision Wormhole 的跨厂商不稳定都是警告），而不是没人想到。所以先做可行性验证，别先许诺方法。

### 8.3 缺失的对照点

**「LatentMAS 原样搬到 VLM 上，到底能跑到什么程度」这个数字没有人报告过。** L²-VMAS 引了它但没测。这不是新方法，是**文献里缺的一个 baseline 格子**——门槛低、价值确定。

### 8.4 选择性传输

LatentMAS 的 handoff 是**全有或全无**。LACO（协同驾驶，[2605.22504](https://arxiv.org/abs/2605.22504)）有显式的显著性选择器决定**传哪部分隐内容**；Where2comm 六年前就做了空间切片选择。**消息里带视觉状态时，「选什么传」比文本场景重要得多**，而 LatentMAS 完全没有这个维度。

### 8.5 可审计性

多模态里「视觉专家看到了什么」本就难解释，再套一层隐空间 = 黑箱套黑箱。而 LatentMAS 的 debug 探针模式（论文附录 F）**在官方代码里根本没实现**（`code-to-paper-mapping.md` §4.9）。ViF 的注意力分析是现成的可解释性抓手，可以借。

---

## 9. 证据分级

| 结论 | 依据 | 级别 |
|---|---|---|
| 25+ 个 arXiv ID 与 GitHub 链接可达 | 逐个 `curl` 取 HTTP 码，全部 200 | **实测** |
| L²-VMAS 的记忆内容 / 三阶段 RL / backbone / benchmark / 引用 LatentMAS 但无对比 | WebFetch 全文 HTML | **实测** |
| ViF 的机制 / 免训练 / 有代码 | WebFetch arXiv abs 页 | **实测（abs 页）** |
| Vision Wormhole 的 9 个 benchmark 全为文本；对 LatentMAS 的评价原文 | WebFetch 全文 HTML | **实测** |
| 「naive 移植 LatentMAS 到 VLM 会 degenerate」 | ❌ **已推翻**：原文只说跨厂商配对在 GSM8K 上不稳定（Appendix F），非 VLM 场景 | **纠正记录** |
| L²-VMAS 是 ICML 2026 | 检索智能体声称，arXiv abs 页**未显示** venue | **待核** |
| §6 那批 VLM KV 工作的机制描述 | 检索智能体二手转述，未逐篇 fetch | **二手，未核** |
| §7 的 V2X 谱系 | 完备性批评智能体，未逐篇 fetch | **二手，未核** |
| §8 全部空位判断 | 上述事实的推论 | **推断** |
| 「交叉区非空」这个总判断 | 对抗智能体明确判定 **REFUTED**，4 个反例（L²-VMAS 最致命），逐个核实 | **强证据** |

---

## 10. 修订记录

- **v1（2026-08-01）**：8 次定向检索。结论「交叉区基本空白」。**该结论已作废。**
- **v3（2026-08-01）**：4 篇论文精读 + 源码逐行核对（见 `papers/`）。更正 ViF「免训练」错标；确认四篇**全部要训练**；新增 M-RoPE 位置坍缩一手证据。详见 `papers/research-2026-08-01/30_verdict-and-roadmap.md`。
- **v2（2026-08-01）**：9 智能体并行扫描（7 检索角度 + 完备性批评 + 对抗反证，共 44+ 次检索，181 条原始条目）+ 5 篇关键论文全文核实 + 25 个链接 curl 实测。
  - 推翻 v1 的「交叉空白」结论（§0、§2）
  - 推翻 v1 的「共享视觉前缀没人做」结论（§6）
  - 更正 v1 的「Vision Wormhole 无 repo」（§2.4）
  - 推翻检索智能体的「naive 移植会 degenerate」强表述（§2.4、§9）
  - 保留并强化 $W_a$ 理论空位（§8.1）
