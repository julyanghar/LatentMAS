# Q&A 04：多模态 MAS / 多模态 LatentMAS 全景 —— workflow 特点、动机（性能 vs 加速）、有没有强 baseline

> **日期**：2026-08-03 · **模式**：问答 + 文献核实（不跑实验）
> **问题来源**（用户原述）：
> 1. 目前的多模态 MAS / 多模态 LatentMAS 框架，是否有一个强的 baseline？
> 2. 详细讲解现有框架：①workflow 的特点；②为什么要用——提升 performance 还是加速推理？多模态 agent 也许不能像文本 4-agent 那样靠 critic/planner 简单涨分，是否有更有效的多模态 MAS 框架解决 VQA？另一方面文本 LatentMAS 是为了减少推理步数、加速推理。
> 3. 追加：文献核实要尽可能找**有代表性的、开源的** Visual MAS。
> 4. 追加：L²-VMAS Table 1 里 VMAS 似乎一直比 Single 好，它的 workflow 是什么？和 LatentMAS 4-agent 一样吗？
> 5. 追加：它们用 MMBench/MMStar/RealWorldQA/SimpleVQA，我们用什么？规模与题型对比。
>
> **证据基础**：本项目 Phase 1 + Phase 2 实测（[phase1-results.md](../VMAS-latent-collab/phase1-results.md)、[phase2-results.md](../VMAS-latent-collab/phase2-results.md)）+ 本轮两个并行 workflow（landscape 19 agent / 开源核验 44 agent）+ 我本人 curl / 读 PDF / 读源码核实。
> **证据分级**：`【实测】`本项目跑出的数字 · `【核实】`我本人 curl/读原文验证 · `【论文】`论文报告值 · `【调研】`subagent 报告未二次核对 · `【推断】`判断。

---

## 0. 三句话结论

1. **没有"一个"强 baseline**，但有三层不同意义的答案：**方法层**有清晰的阶梯（单模型 + CoT + self-consistency + thinking + 工具），且数字明确；**框架层**有一批真开源可跑的 Visual MAS（下表 38 个已逐仓库核实），但没有任何一个被公认为"标准对照"；**多模态 latent MAS 层**——真空，且头号工作 L²-VMAS **代码承诺在 v2 里被删掉、GitHub 账号已注销**（我本人核实 404）。

2. **你的怀疑是对的，而且有硬数字**：在**等算力**下，靠 planner/critic/debate 这类"角色分工"做多模态 MAS **打不过同一个模型的 self-consistency**。DART 的表：3×QwenVL 辩论 61.9~62.1%，而同 backbone 的 SC@5 是 **63.7%**，且辩论烧 11× FLOPs / 136× 延迟。真正带来增益的是**异质 backbone**和**外部感知工具**，不是角色。

3. **"为什么用"在文本和多模态里答案不同**：文本 LatentMAS 是**省 decode 步数**（4–4.3× 实测加速）；多模态里省步数的收益被视觉 prefill 摊薄，**真正的价值是消掉"视觉证据过文字瓶颈"的损失**——这条我们自己测出来了（X−T = +9.8pp → Qwen3-VL 上 **+17.8pp**），而且是全场最大的效应量。

---

## 1. 有没有强 baseline —— 分三层回答

### 1.1 方法层：有明确阶梯，而且每一级值多少分是已知的

审稿人 2026 年期待一个多模态 MAS 至少打过这五级（数字均为【论文】/【调研】）：

| 级 | 内容 | 典型增益 | 关键坑 |
|---|---|---|---|
| ① | 单 VLM 单次 | — | Qwen3-VL-8B-Instruct/Thinking：MMMU 69.6/74.1、MathVista 77.2/81.4、MMStar 70.9/75.3、MMBench-EN 84.5/85.3、RealWorldQA 71.5/73.5 |
| ② | + CoT | **平均是负的** | 多模态 6 库平均 IO→CoT：Qwen2.5-VL-7B 45.2→41.7、InternVL3-8B 49.7→47.9、GPT-4o-mini 44.3→38.5。CoT 只在数学/符号上有效（+12~14），感知题上有害 |
| ③ | + self-consistency@5（**等算力对照**） | **+2.0~2.4** | DART 阶梯：MMMU 58.8→60.8、A-OKVQA 61.3→63.7。但在感知重的库上可以是**负**的（M3MAD 多模态平均 InternVL3-8B 49.7→46.4） |
| ④ | + thinking mode / test-time scaling | **+4~7**（推理库），**感知/OCR 上为负** | Qwen3-VL-8B Instruct→Thinking：MMMU +4.5、MuirBench +12.4，但 BLINK −4.4、OCRBench **−77** |
| ⑤ | + 工具增强单 agent | **~+5，高分辨率上 +12.6** | Qwen3-VL-8B V* 77.5(无工具) → **90.1**(+工具)。作者原话：工具增益"consistently outweigh those from simply increasing model size"。但**天真的工具 agent 更差**：ViperGPT 54.0 / Chameleon 51.6 vs 同 backbone CoT 58.8 |

**判读**：一个 +1.8~4.0 点的多模态 MAS 增益，**落在 self-consistency 的区间内、低于 thinking mode、远低于工具**。所以"MAS 涨 3 点"本身在 2026 年不构成卖点，除非同时给出等算力对照。

**还有两条硬门槛**：
- **饱和阈值**【论文】：等算力下，单 agent 准确率 **>~45%** 之后，加 agent 的边际收益转负（交互项 β=−0.236, p=0.004，"baseline paradox"，在 94% 的验证配置上正确预测符号）。MMBench/MMStar 都远在 45% 以上。
- **backbone 越强增益越小**：RECONCILE 在 Gemma3+Qwen2-VL 上 +7%，在 GPT-4.1 上只 **+1.5%**；agentic scaffolding 的贡献跨三代 Claude 从 19.4pp → 3.8pp → **0.9pp**。

### 1.2 框架层：有真开源的，但没有"标准对照"

本轮扫了 **119 条**候选、**116 个唯一仓库**，对 40 个做了逐仓库核验（GitHub API + 读文件判 stub），其中 **38 个确认真实非 stub**。按"是否真 MAS"和"能否本地跑开放权重"两轴排（★=推荐当对照）：

#### A. 真多 agent + 全开放权重（可以直接当 baseline）

| 系统 | 会议 | 仓库 | ★ | workflow | 动机 | 报告增益 |
|---|---|---|---|---|---|---|
| **Insight-V** | CVPR'25 Highlight | [dongyh20/Insight-V](https://github.com/dongyh20/Insight-V) 240★ | ★★★ | **2 agent，都看图**：reasoning agent 出长 JSON 推理链 → summary agent 看图+链，被训练成**可以否决**这条链、直接从图作答 | 纯准确率。理由：长 CoT 数据直接监督会**损伤感知能力**，所以拆开 | vs LLaVA-NeXT-8B：MMMU +5.1、MMBench +9.4、**MMStar +14.3**、ChartQA +8.0，7 库均 +7.0% |
| **Cola** | NeurIPS'23 | [cliangyu/Cola](https://github.com/cliangyu/Cola) 106★（2023 停更） | ★★ | **协调者 + N 个 VLM**：每个 VLM 独立看图出 caption + 候选答案 + 置信度；**coordinator LLM 完全不看图**，只读文本 | 准确率；明确对标 naive ensemble | A-OKVQA MC 77.7 vs ensemble baseline **56.6** |
| **MDocAgent** | 2025 | [aiming-lab/MDocAgent](https://github.com/aiming-lab/MDocAgent) 353★ | ★★ | **5 agent 模态分工**：general / critical / text / image / summarizing；**只有 image+general agent 看像素** | 准确率；防单一模态压制另一模态 | 5 库平均 **+12.1%** vs 前 SOTA |
| **IdealGPT** | EMNLP'23 | [Hxyou/IdealGPT](https://github.com/Hxyou/IdealGPT) 39★ | ★ | **迭代 3 模块**：Questioner LLM（盲）拆子问题 → Answerer VLM 看图答 → Reasoner LLM（盲）聚合，**允许说"信息不足"** 触发下一轮 | 准确率 | VCR **+10**、SNLI-VE **+15**（绝对） |
| **VideoMultiAgents** | 2025 | [PanasonicConnect/VideoMultiAgents](https://github.com/PanasonicConnect/VideoMultiAgents) 39★ | ★ | LangGraph 星型：agent1/2/3 + organizer，organizer 条件回边 | 长视频准确率 | — |
| **MACT** | NAACL'25 Findings | [boschresearch/MACT](https://github.com/boschresearch/MACT) 6★ | — | planner + coder 双 agent + 工具 | ⭐**效率/可及性**：不微调、不用闭源模型也能打平 GPT-4 | 4 库中 3 库超 SOTA | ⚠️ **是文本表格 QA，不是视觉**——"视觉 MACT"不存在，这是本地清单的一处错认 |

#### B. 真多 agent 但 API-first（当参照可以，当本地 baseline 不行）

| 系统 | 仓库 | workflow 亮点 |
|---|---|---|
| **PixelCraft** (ICLR'26) | [microsoft/PixelCraft](https://github.com/microsoft/PixelCraft) 30★ | **6 角色**（dispatcher/planner/reasoner/planning critic/visual critic/tool agents）+ ⭐**planner 管理的 image memory**：可回看、分支、改轨迹，不是线性追加。**通道里传真实图像 crop，不是纯文本**。grounding 模型是开放权重 PixelCraft-3B |
| **MMCTAgent** | [microsoft/MMCTAgent](https://github.com/microsoft/MMCTAgent) 78★ | planner 迭代 + **vision-based critic** 验证；论文**同时报有/无 critic** —— 少见地隔离了 critic 的价值 |
| **Mobile-Agent-v2** (NeurIPS'24) | [X-PLUG/MobileAgent](https://github.com/X-PLUG/MobileAgent) 9k★(家族) | ⭐**唯一把"压缩上下文"当架构动机的**：planning agent 把长图文历史压成**纯文本进度串**，decision agent 只吃短文本+当前截图。**同作者同 backbone 的单 agent vs 多 agent 消融，+30% 任务完成率**——本清单里最干净的 MAS-vs-single 对照 |
| **MAPS** | [exoskeletonzj/MAPS](https://github.com/exoskeletonzj/MAPS) 4★ | 7 agent，4 段求解链 + Socratic critic，"Big Seven 人格"prompt 分化。报 +15.84% |
| **DART** | [nsivaku/dart](https://github.com/nsivaku/dart) 3★ | ⭐**用分歧触发工具**：agent 意见冲突处才调工具。**其 Table 1 是全场最干净的等算力对照**（见 §3.1） |
| **BDoG** | [thecharm/BDoG](https://github.com/thecharm/BDoG) 19★ | 辩论的共享状态是**实体关系图**而非 transcript，专治多模态辩论的"视觉细节被琐碎化 + 意见漂移" |

#### C. 常被误当成 MAS 的（写 related work 时别混）

- **Visual Sketchpad** [Yushi-Hu/VisualSketchpad](https://github.com/Yushi-Hu/VisualSketchpad) 287★ —— AutoGen 上的 assistant + code executor，**executor 不是第二个推理者**；本质是单 agent 视觉 CoT。数字很强（V* 80.3、BLINK 空间 83.9）
- **ViperGPT** 1717★ / **VisProg** 774★ / **Chameleon** 1139★ / **MM-ReAct** 967★ / **HuggingGPT** 25k★ —— 单 agent + 工具编排
- **Prophet** 278★ / **M3DocRAG** 71★ / **VideoTree** 166★ —— 两段 pipeline / RAG，不是 MAS

#### D. 无代码 / stub（引用时必须标注）

| | 状态 |
|---|---|
| **L²-VMAS** (ICML'26) | ⭐【核实】**无代码**。v1 论文承诺 `github.com/YU-deep/L2-VMAS`，我 curl：**仓库 404，且 `YU-deep` 这个 GitHub 账号本身 404（已注销）**；v2 的 HTML 里**一个 github 链接都没有**（只剩 arXiv/LaTeXML 样板）——即作者在改版时**删掉了代码承诺**而非履行它 |
| **ViF** | 论文广告的 `YU-deep/ViF` 死链；实际在 [xlyu0106/ViF](https://github.com/xlyu0106/ViF)（我 curl 200）。核心 forward 曾是 `torch.randn`（本项目此前实测） |
| **MACF** | 无代码；论文两次写"see Appendix"，而 12 页 PDF **没有附录** |
| **MMedAgent-RL** (ICLR'26) | 无代码（作者 24 个公开仓库里没有）。⚠️ 陷阱：`Wangyixinxin/MMedAgent`(267★) 是**另一篇**更早的单 agent 工作 |
| **VipAct** (AAAI'26) / **InsightSee** / **Visual Para-Thinker++** / **EAGLE** | 均无公开代码 |
| **FaST** (Sys2-LLaVA) | 仓库存在（179 个 .py）但关键件缺失，判为不可跑 |

#### E. 评测与benchmark（有代码，直接可用）

[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) 4346★ · [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) 4321★（两者 2026-08-03 仍在更新）· [COMMA](https://github.com/tossowski/COMMA) 3★（唯一隔离 agent 间通信质量的多模态 benchmark）· [MECoBench](https://github.com/q-i-n-g/MECoBench) · [Agent-X](https://github.com/mbzuai-oryx/Agent-X) · [VisualAgentBench](https://github.com/THUDM/VisualAgentBench) · [OSWorld](https://github.com/xlang-ai/OSWorld) · [VisualWebArena](https://github.com/web-arena-x/visualwebarena) · [MASLab](https://github.com/MASWorks/MASLab)

### 1.3 多模态 latent MAS 层：**真空仅限"多模态 latent VQA"**，不是整条线真空

> **口径先定死**（三条跨切主张已按 LACO 更正，勿再用旧说法）：
> - ❌ ~~"四篇里没有一篇是 KV 级的"~~ → LACO **传真·逐层 KV**（只传前 $L_{comm}$ 层），KVComm（arXiv:2510.12872）也是 KV 级
> - ❌ ~~"没有一篇是免训练的"~~ → LACO **明确 training-free**，且保留原架构与预训练权重
> - ❌ ~~"多步多模态 latent 思考没人做过"~~ → LACO **m=10 步 ILD，还做了 m 消融（Figure 6）**
> - ✅ 站得住的收窄版：**"真空存在于『多模态 latent MAS + VQA/通用视觉理解』这一格"**。LACO 占住了「免训练 + KV 级 + m 步 latent」三格，但**任务域是 CARLA 闭环驾驶**（多车、各车自有视角、动作输出），且**它的 m 消融把"m 的深度"与"有没有 latent 通道"混在一起**（见下表 ①）。我们剩的是**任务域 + 通道级对照 + 干净的 m 隔离**，不是"首次做 KV 级/免训练/多步"

【核实】**LACO**（[arXiv:2605.22504](https://arxiv.org/abs/2605.22504)，2026-05-21）—— 我逐句读了 HTML 原文：

- **training-free**（原文两处明说；"preserving their original architectures and pretrained weights"）
- 传 **KV cache**，且**只传前 $L_{comm}$ 层，默认 10%**（原文 Eq.5 $\mathcal{P}=\mathcal{KV}_{CHSA}^{(1:L_{comm})}$；Eq.6 只在 $l\le L_{comm}$ 拼接、Eq.7 深层"terminates external injection"）。SSKD 是**截断式结构过滤，不是梯度训练**——所谓 distillation 只是"truncating the Deep-Stream, retaining the Shallow-Stream"
  - ⚠️ **原文自己对 SSKD 的展开不一致**：Abstract 写 "**Structured Semantic** Knowledge Distillation"，而 §4.4 章节标题写 "**Shallow-Stream** Knowledge Distillation"。引用时建议只写缩写 SSKD 并注明此不一致，别选边站
- ⭐ **LACO 的 ILD 明确建立在本项目的基础论文之上**：原文 §4.2 "Following [45], rather than projecting $h^{(0)}$ to language space, reasoning unfolds entirely in latent space via $m$ iterative forward passes"，其中 **[45] = Zou et al., *Latent Collaboration in Multi-Agent Systems*, arXiv:2511.20639 = LatentMAS 本身**；连 $W_a\approx W_{out}^{\dagger}W_{in}$ 伪逆投影都同源。→ **LACO 是 LatentMAS 在闭环驾驶上的下游改造**，related work 里必须这样定位，不能写成平行工作
- ⚠️ **它把 LatentMAS 归入被批评的一方**：§5.3 "naive latent sharing [45, 41]"（[41] = KVComm, arXiv:2510.12872）即"会造成 identity confusion"的做法；补充材料 Table 3 "Comparison between LACO and naive latent communication strategy" 是**已发表的、LatentMAS 式全层 KV 直传 vs 选择性浅层传的正面对打**，且**全层直传输**：ORION V0 DS 30.70→35.48、RC 61.87→68.98；SimLingo V0 28.36→35.73。结论句 "simply sharing the full latent KV cache is insufficient for effective multi-agent collaboration"
- **m = 10 步 iterative latent deliberation**（原文："set to $m=10$ to ensure sufficient depth for intent crystallization"）
- CHSA 保留率 ρ=0.3
- 关键发现：**agent identity confusion** —— "naive full-KV fusion can induce agent identity confusion, where the receiver over-attends to another agent's internal states, leading to cross-agent representation entanglement"
- backbone：ORION / SimLingo / LMDrive，含 **LLaMA / LLaVA / Vicuna** —— 真 VLM/VLA
- 指标：CARLA 闭环 DS/RC + **通信延迟(ms) + 通信量(KB)**。ORION 行：Noncollab 26.68 → Language 29.34（**7802 ms**）→ Visual 31.48（82ms/8208KB）→ **LACO 35.48（430ms/4881KB）** —— 比语言通信**快 18×且分更高**
- 消融 Table 3（SSKD 深度 5/10/30/50/Full）ORION V0：34.04/35.48/35.70/34.70/**32.46** —— **全层 KV 最差**（且与 Table 2「ILD✓ CHSA✓ SSKD×」行的 32.46 完全一致，两处互证）
  - ⚠️ **但"实证了浅层-only"是过度解读，已更正为倒 U**：默认 10% **不是最优**——30% 在 6 列里赢 4 列（ORION V0 35.70>35.48、SimLingo V0 35.97>35.73、SimLingo V1 30.03>29.22、LMDrive V1 30.85>30.13）。原文自己的措辞是 "Performance improves substantially when the shared depth falls within the **10%–30% range**, where different VLA backbones achieve their respective optima"，且 5% 太浅也不行（"Extremely shallow sharing yields limited improvement"）。正确表述：**浅到中层（10–30%）最优、全层最差**，不是"越浅越好"
- ⚠️ **CHSA（保留率剪枝）几乎不换准确率，是纯带宽手段**：Table 2「ILD✓ CHSA×(Full) SSKD✓」= 36.03/31.60/35.55/30.06/28.26/30.09 vs 完整 LACO 35.48/31.65/35.73/29.22/28.40/30.13 —— ORION V0 **关掉 CHSA 反而更高（36.03>35.48）**、SimLingo V1 也更高（30.06>29.22），其余四列仅 +0.04~+0.18。Table 4 同向：ORION 100% 保留 = 36.03 为该行最高。→ 引用 LACO 时**不能说剪枝提升了精度**
- ⚠️ **18× 是 ORION 单行的最好情形，不是普遍值**：逐行 Language→LACO = ORION 7802/430=**18.1×**、SimLingo 1300/382=**3.4×**、LMDrive(LLaMA) 8509/215=39.6×、(LLaVA) 8021/215=37.3×、(Vicuna) 8340/203=41.1×。原文结论段自报 "**~20×**" 与 "payload 比 Visual 降 **40%~90%**"
- ⚠️ **相对 Visual 共享，LACO 是更慢的**：430 vs 82ms（ORION）、382 vs 52ms、215 vs 95ms —— **慢 2.3~7.3×**；它赢的是**通信量**（4881 vs 8208KB=−41%；SimLingo 103 vs 896=−89%）**和 DS**。→ 我们引它做"加速"论证时必须说清"快是相对语言通信，相对视觉共享是更慢但更省带宽且更准"

**对本项目的含义（要害）**：

| 本项目原以为空的格子 | LACO 的占位情况 | 我们还剩什么 |
|---|---|---|
| 免训练 latent MAS 上 VLM | **已占**（training-free + LLaVA backbone） | 任务域不同（VQA vs 闭环驾驶） |
| 多步多模态 latent 思考（m 步） | **已占**（m=10） | ⚠️ **原记"LACO 从不消融 m"是错的，已更正**：LACO **Figure 6 就是 m 消融**（§5.4 "Moderate Latent Deliberation Optimizes Intent Crystallization"：从 zero latent steps 到 moderate 有 "substantial performance leap across all VLA backbones"，再往上 "diminishing returns and sometimes leads to performance degradation"，归因 semantic drift / over-reasoning）。所以我们与它的关系**不是"补它缺的对照"，而是方向相反的结论冲突**，必须正面处理。<br>⭐ 我们真正剩下的是**两处它的 m 消融不能排除的混淆**：① LACO 的 m=0 基线按原文"inherently restricts the system to naive perception sharing"——**m=0 时通道内容本身变了**（没有 latent trace 可传→退回感知共享），所以它测到的"leap"混淆了"latent 思考深度"与"有没有 latent 通道"两件事；我们的 M1 是**通道固定、只扫 m**，才隔离出 m 的净贡献。② Figure 6 是位图，**原文正文与表格均无 m 的具体数值**（无法核对幅度）。我们的 M1 零载荷（m∈{0,5,10,20,40} 全平，三 backbone×三任务四重复现）应表述为"**在 VQA、通道固定的条件下 m 的净贡献为零**"，而非"没人做过 m 消融" |
| 层深选择 / 深层融合失效 | **已占**（identity confusion + 浅层 only） | 他们的机制是"跨 agent 表征纠缠"（不同车不同视角）；我们的设定是同一张图，需要区分 |
| 通道级对照（shuffle / 换图 / CAG / 盲探针） | **未占** | ⭐ 我们独有，四篇+LACO 全部没有 |

其余四篇（L²-VMAS / ViF / MACF / Vision Wormhole / CogRad）作为**命名方法全部要训练**（细节见 [research-digest.md](../VMAS-latent-collab/research-digest.md) §2.1）。本轮对抗验证补两条精修：

- **ViF 的"免训练"更站不住**：freeze 表逐字为 "Large Language Model — Frozen / Trainable"，源码 `configs/stage1.yaml:22-25 freeze_llm: true` vs `configs/stage2.yaml:23 freeze_llm: false` —— **Stage 2 解冻 base LLM**。"plug-and-play" 只对 MAS 拓扑成立。
- ⭐ **但"没人跑过免训练多模态 latent MAS"这句是错的**：**MACF 的 `LatentMAS*` 基线就是**——MACF 原文"we adapt and modify the official LatentMAS code to support multimodal understanding"，4-role 串行 + 逐层 KV，在 4 个长视频库上跑出 **55.7 / 48.5 / 33.2 / 42.3**（Table 1），相对单 Qwen3-VL-8B **1 胜 2 负 1 平 ≈ 净零增益**。
  → 对我们的含义：**这是一个已发表的、方向为负的先例**。好消息是它不可审计（MACF 的 12 页 PDF 没有附录，`LatentMAS*` 配置无从复查），而我们的 C1 是**可复现、带通道级对照、并且定位到了负增益的原因**（角色不添新信息源 + M1 零载荷）。写作时应主动引它并说明我们为什么不同。

**另一个被忽略的强 baseline**：L²-VMAS §2.1 自己测出**最强的文本传输方式是 conclusion-only**（稳定 +0.5~1.1%），而它主表实际对比的 full-content 在第 10 轮是 **净 −3.8%**。**四篇没有一篇拿 conclusion-only 当 baseline** —— 这是"文本 baseline 被削弱"的最具体形式，也是我们应该补的一条臂。

---

## 2. workflow 的特点：六种范式 + 三条共性

### 2.1 六种范式

| # | 范式 | 代表 | 通道里传什么 | 谁看图 | 拿到增益的机制 |
|---|---|---|---|---|---|
| 1 | **协调者-专家** | Cola | caption + 候选答案 + 置信度 | 只有专家 VLM，**协调者全盲** | 异质模型互补 |
| 2 | **推理-总结双 agent** | Insight-V | 长 JSON 推理链 | **两个都看图** | 把"长链推理"和"感知保真"拆到两组权重上 |
| 3 | **拆解-回答-聚合迭代** | IdealGPT | 子问题 ↓ / 子答案 ↑ | 只有 answerer | 显式置信度门控 + 允许说"不知道" |
| 4 | **规划-工具-批判（+图像记忆）** | PixelCraft / VipAct | 文本 **+ 真实图像 crop** | 多个 + 工具 | ⭐ 外部工具给高保真像素级事实 |
| 5 | **模态分工** | MDocAgent | 各 agent 的局部答案 | 只有 image agent | 防止文本证据压制视觉证据 |
| 6 | **同质迭代精修（图拓扑）** | L²-VMAS 的 VMAS baseline | 上家整段输出塞进下家 instruction | **每个 agent 都看图** | 集成/精修（但见 §3.1，甜区很窄） |

### 2.2 和 LatentMAS 4-agent 的对照（回答追问 4）

L²-VMAS 的 VMAS baseline（附录 D.1 原文）：
> "each agent processes and generates outputs **independently**. These outputs are then systematically integrated as a core component of the **input instruction for the subsequent agent** in the topology chain."

| | LatentMAS（[methods/\_\_init\_\_.py:12-17](../../methods/__init__.py)） | L²-VMAS 的 VMAS |
|---|---|---|
| 角色 | 固定 4 角色 Planner→Critic→Refiner→Judger | **无角色**；$N_A$ 原文叫 "number of agent **turns**" |
| 拓扑 | 线性链（另有 hierarchical） | 6 种（linear/layered/centralized/random/complete/**dynamic**），主表用 dynamic（G-Designer 学出来的图） |
| 数量 | 4 | **5 轮**（论文未标；§2.1 报 Qwen3-VL-8B-Thinking 第 5 轮 5,241 token，与 Table 1 该格 5241 完全一致→推断） |
| 中间 agent 解码文本？ | **不解码**（只有 Judger 出字） | 全部解码整段 |
| 谁看图 | 我们 C1：**只有 Planner** | **全部**（Eq.2 "$T_n$ ... including the textual task query **and visual inputs**"；token 口径含每 agent 的 "visual feature tokens"） |

**Table 1 里 VMAS 并非一直优于 Single**，四类系统性反例：

| # | 反例 | 数字 |
|---|---|---|
| 1 | Table 1 GLM-4.1V-9B-Thinking | avg **66.7→66.6**；RealWorldQA **69.0→67.2** |
| 2 | Table 2 Qwen3-VL-**32B**-Thinking | **75.6→75.1**；32B-Instruct 仅 75.0→75.4 |
| 3 | Table 5 MuirBench-Thinking | **75.5→73.0** |
| 4 | ⭐ 论文自己 §2.1（Qwen3-VL-8B-Thinking/MMBench） | turn1(=Single) 84.8 → **turn3 峰值 86.6** → **turn6 起跌破 Single** → turn10 **低 2.6%**；token 557→5,241(turn5)→**16,840(turn10)** |

→ 文本 VMAS 只有一个**窄甜区**（2–5 轮、4B–8B、非 Thinking、未饱和库）。Table 1 固定 $N_A$=**5**，恰在峰值(3)之后、跌破点(6)之前。**引用 Table 1 必须连 §2.1 曲线一起引**。
补充：VMAS 每题 2,190–7,446 token vs Single 318–730 = **5–10× 算力**换 +1.6~3.8 点；论文 Figure 3 自测「传得越多越糟」（Conclusion-only 稳定 +0.5~1.1%，Full-content 第 10 轮 **净 −3.8%**）；Table 1 InternVL 行两处印刷错（461 写成 3461、2245 写成 3011）。

### 2.3 三条共性（跨全部范式）

1. **几乎所有系统的 agent 间通道都是文本**，唯一例外是 PixelCraft（传 crop）、EAGLE（传 grounding 区域）、LACO / 四篇 latent（传张量）。
2. **图像被重复编码 N 次** —— 这是 latent/KV 共享要消掉的冗余，也是 L²-VMAS token 口径里被明确计入的部分。
3. **"角色"基本是 prompt 换皮**，权重不分化。唯一权重真分化的是 Insight-V（两套 checkpoint）和 MMedAgent-RL（RL 分角色训）——**也正是增益最大的两个**（+7.0% / +23.6%）。这不是巧合，见 §3.1。

---

## 3. 为什么要用：性能还是加速 —— 两半答案相反

### 3.1 性能：角色本身几乎不带增益（有等算力硬证据）

> **先声明证据边界**（对抗验证的裁定）：**四篇多模态 latent MAS 论文没有任何一篇报告等算力单智能体上界，也没有一篇跑 CoT / self-consistency / best-of-N 对照**（[papers/external/00_cross-comparison.md](../papers/external/00_cross-comparison.md) 已把"等算力单智能体上界"列为待建对照）。所以"等算力下多模态 MAS 是否赢"在**那条文献线里根本没被测过**。下面的等算力数字来自**文本通信的 visual MAS**（DART 报了 FLOPs）和 L²-VMAS 自己的 token 曲线——这是目前能拿到的最好代理，方向一致，但不能当"已定论"。

**最干净的一组数（DART Table 1 + Tables 10/11，A-OKVQA/MMMU/NaturalBench，backbone 固定 QwenVL）**【调研，来自论文表】：

| 方法 | A-OKVQA | MMMU | NaturalBench | 生成 token | TFLOPs | 延迟 |
|---|---|---|---|---|---|---|
| 单 agent CoT | 61.3 | 58.8 | 79.2 | 80.5 | 19.04 | 0.16 s |
| **单 agent self-consistency@5** | **63.7** | **60.8** | **80.1** | — | ~95 | — |
| 3×QwenVL 辩论（consensus） | 61.9 | 60.5 | 80.0 | 1031.8 | 215.66 | 21.84 s |
| 3×QwenVL 辩论（judge） | 62.1 | 60.3 | 79.9 | 1196.8 | 236.83 | 27.92 s |

**同质角色辩论烧 11.3× FLOPs、136× 延迟，只换 +0.6~0.8pp，且在三个库上全部输给 self-consistency（A-OKVQA 上 −1.6~1.8pp，SC 只花约一半 FLOPs）。** 换 backbone 同向：3×MiniCPM-o 57.4 vs SC 58.2；3×Ovis2 62.8/63.0 vs SC 64.6。DART 自己承认在 M3D 上"multi-agent debate underperforms the strongest single-agent baseline by 2.72%"。

**扩模型比加角色更划算**：Qwen2.5-VL-**32B** 单 agent A-OKVQA 62.7% @ 88 TFLOPs，vs 3×7B 辩论 61.9~62.1% @ 216~237 TFLOPs。

**DART 自己的增益来自哪？** 消融：塌成单模型（同角色）**−4.6pp**；去 grounder 工具 −2.4pp；去 captioning −1.4pp；把专用工具全换成一个通用 VLM −2.9pp。→ **异质性 + 工具，不是角色。**

其余同向证据（【调研】）：
- **MAD 系统扫**（5 方法 × 9 库 × 4 模型，统一 6 次 LLM 调用）：36 个配置里"没有一个 MAD 方法对 CoT 的胜率超过 20%"，**self-consistency 一致最强**；修正方式是异质（Heter-SoM 最多 +5.8%）
- **辩论会因谄媚主动掉分**：3×Mistral CommonSenseQA −5.0 / MMLU −9.2；2×LLaMA+1×Mistral MMLU −12.0
- **多模态专属机制**：视觉注意力**跨轮衰减 62%**（turn1→20：0.165→0.099→0.063，中层最惨 ~60%）；unimodal token 子集从 1.22% 崩到 0.10%。ViF 的修法把幻觉滚雪球降 33.6~39.8%，但**准确率只回收 2.7~3.8%**
- **critic 贡献可以是精确的零**：cartoon VQA 上 Full(Visual+Language+Critic)=0.8819 与 Visual+Language=0.8819 **完全相同**；Language-Only=0.8403 与 Language+Critic=0.8403 也完全相同
- **同 backbone 分角色会掉分**：单 agent Pixtral-Large MathVision 32.10 / RealWorldQA 70.25 → 全角色 Pixtral 的 MAD 31.05（−1.05）/ **62.24（−8.01）**
- **COMMA**：多模态 agent 在 agent-agent 协作上打不过随机基线（random 18.70%，LLaVA-CoT 14.97%、R1-OneVision 16.81% 均**低于随机**；GPT-4o 41.74%、o4-mini 53.98%，Human+GPT-4o 69.01%）
- **MAST**：7 个开源 MAS 框架 1642 条轨迹，14 种失败模式，失败率 41%~86.7%；论文开篇即"performance gains on popular benchmarks are often minimal"

### 3.2 我们自己的实测：把上面的文献结论复现并放大

【实测】[phase2-results.md](../VMAS-latent-collab/phase2-results.md) C1（Planner→Critic→Refiner→Judger，**仅 Planner 看图**，通信介质为唯一变量）：

| | Single | Text-MAS | **Latent-MAS(m=0)** | Latent-MAS(m=10) |
|---|---|---|---|---|
| LLaVA-OV / GQA | 59.8 | 43.9 | **59.8** | 59.0 |
| LLaVA-OV / MMStar | 53.9 | 38.7 | **50.1** | 49.4 |
| Qwen3-VL / GQA | 63.9 | 47.1 | **59.1** | 59.8 |
| Qwen3-VL / MMStar | 61.8 | 49.9 | **62.2** | 62.3 |

- **Text-MAS 比 Single 掉 14–16pp**（双 backbone，p<1e-20）
- **Latent(m=0) − Text = +12.1~13.6pp**；Latent − Single = **−1.9~−2.2pp**

> ⚠️ **以上是 2026-08-03 的宽松判分版，仅作历史记录。** 下面的补臂实验推翻了"latent 把通信税降到 ≈0"这个表述。

#### ⭐ 2026-08-04 补臂 + 换判分口径后的修订（C1-rev）

上面标的"审稿风险"已按预注册规则做完：**新增 Text-MAS-sighted 臂（4 个 agent 全看图 + 累积上游全文）**，4 配置 × n=800/798，判分改 **GQA 官方 exact match**（原宽松匹配偏向话多的臂 +8.9pp，偏向单模型只 +0.6pp）。物证 [review-report.md](/home/yilin/modify-code-runs/c1-text-sighted-arm/review-report.md)、主表 [phase2-results.md](../VMAS-latent-collab/phase2-results.md) 的 `C1-rev` 节。

| 配置 | textsighted − Single | p (McNemar) | Latent(m=0) − Text（旧） | **Latent(m=0) − Textsighted（新）** |
|---|---|---|---|---|
| LLaVA-OV / GQA | −1.8 | 0.19 n.s. | +15.8 | **+1.2** |
| LLaVA-OV / MMStar | +0.9 | 0.55 n.s. | +11.4 | **−4.7** |
| Qwen3-VL / GQA | **−17.6** | 8.3e-24 | +17.3 | **+9.9** |
| **Qwen3-VL / MMStar**（预注册基准格） | **+0.9** | 0.66 n.s. | +12.3 | **−0.5** |

三条结论：

1. **"下游盲"确实人为削弱了对照**（基准格落区间①：sighted ≈ Single）。**原来的 +12~14pp 主卖点，大部分是我们自己蒙住了对照臂的眼睛。**
2. **latent 通道的价值必须改写成效率论证**：省 N−1 次视觉编码 + 零中间 token，**不是准确率优势**（准确率差落在 −4.7 ~ +9.9，两个 MMStar 格为负）。
3. ⚠️ **但"我们的 text 臂弱于文献标准"这个自责要收回**：拓扑调研核实 **8 个角色分工接力系统里 7 个都是"头部 caption 一次、之后盲推"**（MAPS 源码：`Aligner/Scholar/Solver.call_LLM` 签名里没有 image 参数；只有 Insight-V 全程多模态）。→ 我们的 text 臂**复现的正是领域主流设计**；sighted 臂量的是"主流设计在桌上留了多少分"。

**副产物**：Qwen3-VL/GQA 的 **−17.6 是真实掉分**（最宽松口径仍 −14.7pp，错例含 `gold=yes→ans=no` 真实翻转）→ 四轮讨论在简单感知题上会把对的答案聊反，与上面 §表里"CoT 在多模态上平均为负"同向。

### 3.3 加速：文本上的"省步数"论证在多模态里要重算

**文本 LatentMAS**【论文】：output token −70.8~83.7%，**实测墙钟 4×~4.3×**（8×A100，HF + vLLM 双栈；对 vLLM 优化的 TextMAS 仍 2.6×~7×）。latent thought 以逐层 KV 传递，$m\in\{0,10,20,40,80\}$，40–80 最优。**但它仍比单模型调用慢 3.9×**（ARC-E/Qwen3-14B 2124s vs 551s）——latent 去掉了大部分多智能体税，没让 MAS 变便宜。

**多模态里时间去哪了**（【推断】roofline + 【论文】实测交叉验证）：

| 场景 | decode | 视觉 prefill | ViT | 上游文本 prefill |
|---|---|---|---|---|
| 4-agent，1×1080p 每 agent 重编码，500 tok/agent，**bs=1** | **93.0%** | 4.0% | 1.1% | 1.9% |
| 同上，**bs=32** | 29.2% | **40.3%** | — | — |
| bs=32，50 tok/agent | 4.9% | **67.3%** | — | — |
| bs=32，32 帧视频，50 tok | — | **74.5%** | — | — |

- 关键比值：8B VLM 在 A100 上 prefill 0.114 ms/token（compute-bound）vs decode 10.46 ms/token（带宽 bound）= **~92×**。所以**bs=1 时 decode 绝对主导**，"视觉 token 主导 prefill"那个直觉只在**大 batch + 短输出**时才成立。
- 视觉 token 量【核实自 config】：Qwen3-VL（16 patch × merge2 → 32×32 px/token）1080p = **1,980 token**，4K = 8,040；LLaVA-OV anyres_max_9 单图 = **7,290**。1080p 的视觉 prefill 只等于 decode **~22 个 token** 的时间（bs=1）。
- ⭐ **实测的 Amdahl 天花板**：Visual Para-Thinker++（Main + 4 Worker + Summary，V*，~2520 输出 token）把视觉前缀 KV 共享后 **364s → 312s = 1.17×** —— 即**全部重复视觉 prefill 只占墙钟 ≤14.3%**。而并行 decode 本身值 **3.29×**（1197s→364s）。
- 【实测】我们的 C2：Latent(m=0) 墙钟 ≈ Single（4.60 vs 4.73s，**4-agent 协作几乎白送**），Text 慢 2.2–2.6×，m=10 纯亏 +0.6~0.9s。

**结论**：在多模态里，"latent 省 decode 步数"依然是主要杠杆（bs=1 下 decode 占 93%），**但省的不只是 decode——还省下游 agent 读上游长文本的 prefill**。而"省重复视觉编码"这条在小 batch 下天花板只有 ~1.17×，**不能当主卖点**，只能当大 batch 服务场景的论证。

**其他多模态 latent 工作的效率口径**（区分实测 vs 代理指标）：

| 工作 | 加速证据类型 | 数字 |
|---|---|---|
| **LatentMAS** | ✅ 实测墙钟 | 4~4.3×（vs TextMAS） |
| **IVT-LR** | ✅ 实测墙钟（4×A6000） | Qwen2-VL CoT 106.3 步/3.10s/42.5% → 10.0 步/**0.65s**/71.8%。只减文本 AR 步，**视觉 token 仍在上下文里** |
| **Render-of-Thought** | ✅ 实测（H20）+ token 代理 | GSM-Hard 8.55s→1.84s（~4.6×）；推理时直接发 latent embedding，**不跑视觉编码器** |
| **Heima** | ❌ 只有 token 数，**全文无墙钟** | MMStar 181.0→12.8 token，但准确率 61.1%→58.0%（**−3.1pp**） |
| **L²-VMAS** | ❌ 只有 token 代理，**无墙钟/显存/吞吐** | token −21.3~44.8% |
| **Mirage** | ❌ **完全不提效率** | 纯准确率论文 |
| **Vision Wormhole** | ✅ 实测但方差巨大 | 【调研，与本地文档冲突】54 格中位 1.65×，**7 格是减速**（最低 0.57×）；长 CoT(AIME) 中位 2.96×、短输出 QA 1.53×、代码 1.09×。本地 digest 记的"1.02× / −4.6pp"是 **Table 2 的一格**（GSM8K）。**此条待二次核对原文**|
| **LACO** | ✅ 实测通信延迟 + 通信量 | vs 语言：Language 7802ms → **LACO 430ms**（ORION 行 18.1×；逐行 3.4~41.1×，原文自报 ~20×），且 DS 更高（29.34→35.48）。⚠️ vs 视觉共享：**LACO 更慢**（430 vs 82ms）但通信量 −41%（4881 vs 8208KB）且 DS 更高。**无端到端墙钟总表**，Figure 5 为位图无数值 |

---

## 4. 那什么样的多模态 MAS 才对 VQA 真有效？四条从证据反推的设计原则

0. ⭐ **最强的一条：扩大感知视野——这是 self-consistency 在数学上无法替代的**。MACF 在长视频上 **+4.5~+7.7 绝对点**（LVBench +7.0、MLVU-Test +7.7），机制是 **6 个 agent × 16 帧 = 96 帧**，而单模型在同样的 per-agent 16×224×224 预算下只看 16 帧。**采样 k 次、或对已看到的 16 帧想得更久，都无法恢复从未看到的 80 帧里的证据——任何算力都不行。** 这是本轮找到的唯一"不可替代"的多模态 MAS 增益机制。同族做法：多视图/多 crop/高分辨率分块、多帧分工、不同传感器。
   → 对 VQA 的直接推论：**要让 MAS 赢，就得让不同 agent 看到不同的像素**（分块高分辨率、V* 式引导搜索、多轮 crop），而不是让它们对同一张图轮流发表意见。
1. **别指望角色，指望异质**。凡是能等算力打赢 self-consistency 的，都是异质配置（DART 3 个不同 VLM、Heter-SoM、RECONCILE、WISE、ColMAD）。同质换 prompt 一致失败。
2. **让 agent 带来新的视觉信息，而不是新的意见**。工具（+5~12.6）、图像 crop/放大（V* 77.5→90.1）、多粒度视图、检索——这些是"新信息"；critic 复述一遍不是。我们 C1 的 −2pp 就是"角色不添新信息"的直接证明。
3. **通道里必须能过视觉证据**。文本通道每过一次掉一截（我们实测 −14~16pp；X−T 随任务难度和模型强度单调增大：+2.8 → +9.8 → **+17.8pp**）。要么每个 agent 都看图（文献做法，代价是 N 倍视觉编码），要么用 latent/KV 通道（我们的做法，代价是工程适配）。**PixelCraft 和 EAGLE 是第三条路：通道里直接传像素/区域。**
4. **选任务要选"文字必然有歧义"的地方**。感知题上任何 test-time 脚手架都是平的（"CoT/SC/BoN 在 MMBench 三部分上都是 minimal or no improvement"）；饱和库上 >45% 之后加 agent 边际为负。GQA 那类**开放式组合关系推理**才是文字瓶颈最显形的地方。

---

## 5. Benchmark 对表（回答追问 5）

### 我们的（从 [a1_blind_probe.py:55-75](/home/yilin/tmp/phase2/a1_blind_probe.py) 实读）

| 库 | 全量 | 我们用 | 题型 | 随机地板 | 实测盲猜 | 判分 |
|---|---|---|---|---|---|---|
| **GQA** testdev_balanced | **12,578** | n=**800**（random_state=42） | **开放式短答**；结构 verify·query·logical·compare·choose × 语义 global·rel·attr·obj·cat = 组合关系推理 | ≈0 | **33.0%**（约 1/3 可靠语言先验蒙） | exact + 冠词归一 + 前缀/包含宽松 |
| **MMStar** val | **1,500**（6 类各 250） | n=**798**（category 分层 133×6） | **四选一 MC** | 25% | **27.4%** → **vision-indispensable 被我们实证** | letter 匹配 |
| POPE（已退出） | 9,000 | 1,000 | yes/no | 50% | — | — |

MMStar 六类【核实自本地 parquet】：coarse perception / fine-grained perception / instance reasoning / logical reasoning / math / science & technology，各 250。

### 他们的

| 库 | 规模 | 题型 | 特点 |
|---|---|---|---|
| **MMBench** | dev **4,377** / test 6,718【核实 HF】 | MC，20 能力维度，**CircularEval** | 报 80–90 → **已近饱和**；正是他们 D.2 解释负增益的理由 |
| **MMStar** | **1,500** | 四选一 MC | ⭐ **唯一重叠库** |
| **RealWorldQA** | **765**【核实 xai-org/RealworldQA】 | MC + 短答 | 真实照片（大量车载/街景） |
| **SimpleVQA** | **2,025**（ICCV'25） | 短答 | **事实性/世界知识**，非视觉推理 |
| + MuirBench / BLINK / MVBench / LVBench | | 多图 / 视频 | |
| **训练集** | **GQA** | | ⚠️ 我们的主库是他们的训练集 |

### 四条判读

1. **题型分布相反**：他们以 MC 为主（25% 地板、易饱和、CircularEval 抗蒙）；我们主力 GQA 开放短答（无地板、绝对分低、判分规则敏感）。**同一个 Δ 含义不同**——MC 上 +2 点可能一半是选项噪声。
2. ⚠️ **并表前有硬关**：同模型同库我们的 Single 明显低于他们——**Qwen3-VL-8B MMStar 我们 61.8 vs 他们 70.4**（同一模型差 8.6pp）；LLaVA-OV 我们 53.9 vs 他们 LLaVA-OV-**1.5**-8B 67.0（部分是模型不同）。Qwen3-VL 那 8.6pp 只能来自评测协议（n=798 子集 / 严格 letter-only prompt / 无 thinking / greedy）。**必须先把 Single 校准到公开值附近**（Qwen3-VL 官方 MMStar Instruct 70.9 / Thinking 75.3），否则整张表不可信。→ 建议直接用 VLMEvalKit 或 lmms-eval 跑一遍 Single 做校准。
3. **不能声称"我们在 GQA 上超过 L²-VMAS"**——GQA 是他们训练集，他们没报 GQA 评测。
4. **规模上不吃亏**：我们每格 n=800/798（McNemar 检出力 ≈5pp）；**RealWorldQA 只有 765 题**，他们那上面的 ±2 点不比我们更稳。

**可利用的空档**：他们 4 个单图库**没有一个是开放式组合关系推理**（MMBench/MMStar 是 MC、SimpleVQA 考知识、RealWorldQA 是场景理解）。GQA 恰是"文字描述空间关系必然有歧义"最显形处，也是我们 X−T 差距的来源——**选 GQA 有方法论理由，要在写作时说出来**。

---

## 6. 待办（本轮新增）

| # | 事项 | 理由 | 优先级 |
|---|---|---|---|
| 1 | 加 **Text-MAS-sighted 臂**（每 agent 都看图，文本传结论） | 现有 Text-MAS 是下游盲，弱于文献标准 VMAS，会被直接攻击（§3.2） | 🔴 高 |
| 2 | **Single 校准**：用 VLMEvalKit/lmms-eval 复跑 Qwen3-VL-8B MMStar，对齐官方 70.9 | 现差 8.6pp，并表前必须解释或修掉（§5.2） | 🔴 高 |
| 3 | **LACO 进 related work，定位为"LatentMAS 的驾驶下游改造"，并把我们的 m 消融重定位为"去混淆版"（不是"补空白"）** | 它已占"免训练 + KV 级 + m 步 latent + 层深选择"四格，**且 Figure 6 已消融 m**（原记"从不消融 m"已更正）。我们的差异化只剩三条：任务域(VQA vs 闭环驾驶)、通道固定下隔离 m、通道级对照。另：LACO §4.2 直接 "Following [45]" 引 LatentMAS(2511.20639)，并把 LatentMAS 列为"naive latent sharing"被自己打赢（补充材料 Table 3）——**必须主动正面处理，否则等于让评审替我们发现**（§1.3） | 🔴 高 |
| 4 | 效率论证改口径：主打 decode + 上游文本 prefill，**不主打省视觉编码** | 小 batch 下省视觉 prefill 天花板仅 ~1.17× 实测（§3.3） | 🟠 中 |
| 5 | 加 **self-consistency@5 对照臂** | 它是 2026 的等算力标准 baseline，+2.0~2.4pp，正好在我们效应量附近；且四篇全都没做 | 🟠 中 |
| 5b | 加 **conclusion-only 文本臂** | L²-VMAS 自测的最强文本传输方式（+0.5~1.1% 稳定），四篇没一篇用它当 baseline —— 不补就是"打弱化版文本 MAS" | 🟠 中 |
| 5c | 引用 **MACF 的 `LatentMAS*` 基线**并说明差异 | 它是已发表的免训练多模态 latent MAS 运行结果且**净零增益**（1 胜 2 负 1 平）；不主动处理会被当成"我们复现了一个已知失败" | 🟠 中 |
| 6 | 二次核对 Vision Wormhole 的加速分布（1.02× 是一格还是总体） | 本地 digest 与本轮调研冲突（§3.3 表末） | 🟡 低 |
| 7 | 纠正本地清单：**"视觉 MACT"不存在**（boschresearch/MACT 是文本表格 QA） | 避免 related work 写错（§1.2 A 表脚注） | 🟡 低 |

---

## 附：本轮核实方法与可复现命令

```bash
# L²-VMAS 代码承诺被删（v1 有、v2 无）
curl -s "https://arxiv.org/html/2602.00471v1" | grep -o "github\.com/[A-Za-z0-9._/-]*" | sort -u
# 期望: github.com/YU-deep/L2-VMAS
curl -s "https://arxiv.org/html/2602.00471v2" | grep -o "github\.com/[A-Za-z0-9._/-]*" | sort -u
# 期望: 只有 arXiv/LaTeXML 样板，无代码链接
curl -s -o /dev/null -w "%{http_code}\n" -L https://github.com/YU-deep/L2-VMAS   # 期望 404
curl -s -o /dev/null -w "%{http_code}\n" -L https://github.com/YU-deep          # 期望 404（账号已注销）
curl -s -o /dev/null -w "%{http_code}\n" -L https://github.com/xlyu0106/ViF      # 期望 200

# L²-VMAS 原文（PDF 在 papers/external/L2-VMAS_method-analysis/）
python3 -c "import pypdf; r=pypdf.PdfReader('Dual Latent Memory for Visual Multi-agent System.pdf'); print(r.pages[17].extract_text())" | grep -A3 "multi-agent baseline"

# LACO 原文核实（2026-08-03 全部实跑过；注意 arxiv.org 无直连 DNS，curl 走代理可通）
curl -s "https://arxiv.org/html/2605.22504v1" -o /tmp/laco.html
python3 -c "import re,html,sys; s=open('/tmp/laco.html',errors='replace').read(); t=re.sub(r'<[^>]+>',' ',re.sub(r'<(script|style).*?</\1>','',s,flags=re.S)); t=html.unescape(t); open('/tmp/laco.txt','w').write(t)"
grep -c "training-free" /tmp/laco.txt          # 期望 2（Abstract + Intro）
grep -cF 'L_{comm}=10' /tmp/laco.txt           # 期望 1（默认层深 10%；注意必须用 -F，反斜杠转义会打空）
grep -o "m=10" /tmp/laco.txt                   # 期望命中（ILD 步数）
grep -o "7802\|430\|32.46\|35.48" /tmp/laco.txt # 期望全部命中（Table 1 / Table 3）
# ⭐ m 消融确实存在（推翻"从不消融 m"）：
grep -o "varying the number of iterative latent steps" /tmp/laco.txt   # 期望命中（Figure 6）
grep -o "Moderate Latent Deliberation Optimizes Intent Crystallization" /tmp/laco.txt
# ⭐ ILD 源自 LatentMAS：
grep -o "Following \[ 45 \]" /tmp/laco.txt      # 期望命中
grep -o "arXiv:2511.20639\|2511.20639" /tmp/laco.txt  # 期望命中 = LatentMAS
grep -o "naive latent sharing \[ 45 , 41 \]" /tmp/laco.txt  # LatentMAS 被列为被批评方
# SSKD 展开不一致（Abstract vs §4.4）：
grep -o "Structured Semantic Knowledge Distillation" /tmp/laco.txt   # Abstract
grep -o "Shallow-Stream Knowledge Distillation" /tmp/laco.txt        # §4.4 标题
# 元数据：arXiv:2605.22504v1 [cs.AI] 21 May 2026, KAIST, 主题 cs.AI + cs.CV

# benchmark 规模
curl -s "https://datasets-server.huggingface.co/size?dataset=Lin-Chen/MMStar"      # 1500
curl -s "https://datasets-server.huggingface.co/size?dataset=xai-org/RealworldQA"  # 765
curl -s "https://datasets-server.huggingface.co/size?dataset=lmms-lab/MMBench_EN"  # dev 4377 / test 6718
python3 -c "import pandas as pd; print(len(pd.read_parquet('/data/yilin/datasets/GQA/testdev_balanced_instructions/testdev-00000-of-00001.parquet')))"  # 12578
```

> ⚠️ GitHub **未认证 API 每小时只有 60 次**配额，批量核验会被 403。403 **不等于**仓库不存在——改用 `curl -L https://github.com/OWNER/REPO` 看网页状态码（本文档所有 404/200 均用此法取得）。
