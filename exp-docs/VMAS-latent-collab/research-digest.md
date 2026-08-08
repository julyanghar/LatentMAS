# 调研总汇总：多模态 × LatentMAS

> **这份文档是什么**：本项目至今全部调研结果的**单一入口**。76 万字的原始材料（17 份文档）压到这一份里，按「查了什么 → 发现了什么 → 对你的路线意味着什么」组织。
>
> **日期**：2026-08-01
> **你的路线**（已确认）：**先把 LatentMAS + 多模态 MAS 跑成 baseline，再根据实验结果做改进。** 本文按这个目标组织——重点是**baseline 会在哪里断、每个断点对应什么改进抓手**（§7）。
>
> **证据分级**：`【一手】`= 我本人读源码 / 跑代码核实；`【论文】`= 论文原文，已 fetch；`【调研】`= subagent 报告，未二次核对；`【推断】`= 判断。

---

## 1. 调研规模与方法

| 轮次 | 规模 | 产出 |
|---|---|---|
| 第一轮 检索 | 8 次定向检索 | v1 综述（结论后被推翻） |
| 第二轮 并行扫描 | **9 智能体**：7 个检索角度 + 完备性批评 + **对抗性反证**，44+ 次检索，181 条原始条目 | v2 综述，推翻 v1 |
| 第三轮 论文精读 | **7 智能体**：4 篇论文 ×（全文 + 源码）+ 交叉对比 + 魔鬼代言人 + 实验设计 | `papers/` 下 4 份精读 + 3 份审查 |
| 第四轮 训练路线 | **7 智能体**：训练目标学 / 通道评估学 / 跨模型对齐 / 监督信号 + 算法提案 + 评估协议 + 对抗审查 | `papers/1x` + `papers/2x` |
| 一手核实 | 我本人：clone 2 个 repo、逐行读码、跑 M-RoPE 探针、curl 验证 37 个链接 | §4 全部结论 |

**方法论教训（值得单独记）**：第一轮之所以错，是因为**沿着 latent-MAS 那条文献线检索，会稳定复现「多模态是空白」这个错误结论**——多模态那边的人不用这套词。有效姿势是**去掉 "KV cache" / "LatentMAS"，换成任务域词（video / VQA / radiology / navigation）+ "instead of text"**。

---

## 2. 地形图：这个领域现在长什么样

### 2.1 ★ 已经在做「多模态 latent 通信 MAS」的 5 篇

| | 传什么张量 | 训练 | Backbone | Benchmark | 代码 | 关键数字 |
|---|---|---|---|---|---|---|
| **L²-VMAS** (ICML'26) [2602.00471](https://arxiv.org/abs/2602.00471) | **hidden state 序列 + 单个检索向量**（**不是 KV cache**，见 §4.2） | **三阶段 PPO**（100k+80k+50k 步），**8×H200** | GLM-4.1V-9B、InternVL-3.5-8B、LLaVA-OV-1.5-8B、Qwen3-VL 2B/4B/8B/32B | MMbench、MMStar、RealWorldQA、SimpleVQA、MuirBench、BLINK、MVBench、LVBench | 无 | **绝对 +1.8~4.0 点**（不是 2.7-5.4 点） |
| **ViF** [2509.21789](https://arxiv.org/abs/2509.21789) | 视觉 relay token（~2%） | **两阶段**，Stage 2 连 base LLM 都解冻 | 10 个 base model | 8 benchmark × 4 MAS 结构 | **stub** | 丢 relay token 掉 40.7 分 |
| **MACF** [2605.00444](https://arxiv.org/abs/2605.00444) | K=32 定长通信 token | 双侧 LoRA + 2 层 MLP adapter，三阶段课程，4×A100 | — | Video-MME / LongVideoBench / LVBench / MLVU | 无 | 缺任一阶段会**跌破单模型基线** |
| **Vision Wormhole** [2602.15382](https://arxiv.org/abs/2602.15382) | universal token → 残差加到 dummy image 视觉嵌入 | 每模型一个 **~41M codec**，**400 步 / bs 2 / A6000，3–6 GPU·时** | Qwen3-VL-2B、LFM2.5-VL-1.6B、Gemma、SmolVLM2 | **9 个全是文本任务** | **[✓ 完整可跑](https://github.com/xz-liu/heterogeneous-latent-mas)** | GSM8K **−4.6pp**，只 1.02× 加速 |
| **CogRad** [2607.03853](https://arxiv.org/abs/2607.03853) | 注意力权重 + 区域嵌入 | 是 | — | 放射科报告 | 无 | 强度最弱 |

**⚠️ 五篇全部要训练。「免训练」那一整列是空的。**

**谱系**：L²-VMAS 与 ViF 是**同一作者组**（Xinlei Yu / Chengming Xu / Jiangning Zhang / Xiaobin Hu / Shuicheng Yan）。ViF(2025-09) → L²-VMAS(2026-01) 是同一条演进线，选题时要预期他们的下一步。

### 2.2 纯文本多模态 MAS（可当骨架 / 对照组）

| 工作 | 为什么对你有用 | Repo |
|---|---|---|
| **MDocAgent** [2503.13964](https://arxiv.org/abs/2503.13964) | 5 agent 文档理解，**每个 agent 重编码同一批页面图像**——正是共享 KV 要消掉的冗余。+12.1% | [✓](https://github.com/aiming-lab/MDocAgent) |
| **BeMyEyes** [2511.19417](https://arxiv.org/abs/2511.19417) | perceiver VLM + reasoner LLM，**「视觉过文本瓶颈」的最纯形态**，最干净的单变量 ablation 起点。MMMU/MathVista | 无 |
| **Visual Para-Thinker++** [2606.09290](https://arxiv.org/abs/2606.09290) | **对图像只做一次共享 prefill 建公共视觉 KV**，再并行解码分支 KV。2.5× 吞吐 | 无 |
| **VipAct** (AAAI'26) [2410.16400](https://arxiv.org/abs/2410.16400) | **免训练** orchestrator + 专家 + 工具 | 无 |
| **MMedAgent-RL** (ICLR'26) [2506.00555](https://arxiv.org/abs/2506.00555) | **+23.6%，见到的最高多模态 MAS 增益** | 无 |
| **Training-Free MLLM Orchestration** (ICML'26) [2508.10016](https://arxiv.org/abs/2508.10016) | 同为免训练、同档会议，**刻意选 text-centric memory 而非嵌入**——必须比的 honest control | 无 |
| **MACT** [2508.03404](https://arxiv.org/abs/2508.03404) | 按复杂度分配 test-time 算力。+9.9~11.5% | 称将发布 |
| **M3Prune** [2511.19969](https://arxiv.org/abs/2511.19969) | 通信图剪枝（拓扑侧省钱），审稿人会问的正交轴 | 无 |

### 2.3 Benchmark 该选哪个

| Benchmark | 特点 | Repo |
|---|---|---|
| **COMMA** (TMLR'25) [2410.07553](https://arxiv.org/abs/2410.07553) | **唯一专门隔离「智能体间通信质量」的多模态 benchmark**：两 agent 持**不对称**视觉信息。杀手级动机数据：**GPT-4o / o4-mini / R1-Onevision 在 agent-agent 协作上几乎打不过随机基线**。⚠️ 它刻意设定「只能语言通信」，换 latent 通道要在写作上处理 | [✓](https://github.com/tossowski/COMMA) |
| **MECoBench** [2606.31966](https://arxiv.org/abs/2606.31966) | 具身多模态协作，协议**专门区分「真协作收益」与「单体能力」** | [✓](https://github.com/q-i-n-g/MECoBench) |
| **L²-VMAS 那 8 个** | 数字能直接进它的表做对照 | 通用 |

### 2.4 单模型多模态隐推理（推理端技术储备）

| 工作 | 机制 | Repo |
|---|---|---|
| **Heima** [2501.19201](https://arxiv.org/abs/2501.19201) | **第一个多模态 latent CoT**（2025-01），related work 漏了会显得不熟悉领域 | — |
| **Mirage** (CVPR'26) [2506.17218](https://arxiv.org/abs/2506.17218) | 隐式视觉 token，蒸馏→文本监督→RL | [✓](https://github.com/UMass-Embodied-AGI/Mirage) |
| **IVT-LR** (ACL'26) [2510.12603](https://arxiv.org/abs/2510.12603) | 每步 = 隐文本 + 隐视觉。+5.45% / >5× | [✓](https://github.com/FYYDCC/IVT-LR) |
| **Render-of-Thought** (ACL'26) [2601.14750](https://arxiv.org/abs/2601.14750) | CoT 渲染成图，**这族里最接近免训练的**，3-4× 压缩 | [✓](https://github.com/TencentBAC/RoT) |
| **ReGuLaR** [2601.23184](https://arxiv.org/abs/2601.23184) | 隐思维是**分布**不是点——想让各 agent 采样到不同隐思维，唯一现成的形式化 | [✓](https://github.com/FanmengWang/ReGuLaR) |
| **DeepSeek-OCR** [2510.18234](https://arxiv.org/abs/2510.18234) | **视觉通道容量的唯一硬指标**：7-20×（最高60×）压缩，<10× 保真~97%、20× 时~60% | [✓](https://github.com/deepseek-ai/DeepSeek-OCR) |

### 2.5 MAS 隐通信（纯文本，通信端技术储备）

| 工作 | 传什么 | 训练 | Repo |
|---|---|---|---|
| **LatentMAS** (ICML'26) | 逐层 KV | 否 | [✓](https://github.com/Gen-Verse/LatentMAS) |
| **KVCOMM** (NeurIPS'25) [2510.12872](https://arxiv.org/abs/2510.12872) | **免训练**复用上游 KV：锚点池存 cache 偏移，最近邻加偏移 + 位置平移 | **否** | [HankYe/KVCOMM](https://github.com/HankYe/KVCOMM) |
| **C2C** (ICLR'26) [2510.03215](https://arxiv.org/abs/2510.03215) | 全 KV + 学习融合器 + 逐层门控 | 是 | [✓](https://github.com/thu-nics/C2C) |
| **AC** (ICML'25) [2501.14082](https://arxiv.org/abs/2501.14082) | 中间层暂停 B 的前向、拼 A 的激活。**零参数** | 否 | — |
| **Interlat** (ACL'26) [2511.09149](https://arxiv.org/abs/2511.09149) | 单个末层 hidden；**唯一显式设计防坍缩损失的** | 部分 | [✓](https://github.com/XiaoDu-flying/Interlat) |
| **SDE** (EMNLP'25) [2506.19209](https://arxiv.org/abs/2506.19209) | hidden 的**增量**；关键负面结论见 §5.4 | 否 | [✓](https://github.com/LittleDinoC/StateDelta) |
| **CIPHER** (ICLR'24) [2310.06272](https://arxiv.org/abs/2310.06272) | 词表加权 embedding 期望。**这族的源头** | 否 | [✓](https://github.com/chaudatascience/cipher_multiagent_debate) |
| **PACT** [2606.05304](https://arxiv.org/abs/2606.05304) | 结构化 action-state record。**text 侧强 baseline** | — | — |

### 2.6 质疑与安全分支（选题避让 + 方法借用）

- ⭐ [**Do Latent Channels Actually Communicate? A Causal Audit**](https://arxiv.org/html/2607.26773v1) (2607.26773) —— **动手前必读**，§6 的评估方法学全部来自这篇
- [LCGuard](https://arxiv.org/pdf/2605.22786)（安全 KV 共享）· [When Latent Agents Lie](https://arxiv.org/html/2606.28958v1)（KV 完整性）· [Out of Sight, Not Out of Mind](https://arxiv.org/html/2605.28214v1)（隐蔽攻击）· [Cache Merging as CRDT](https://arxiv.org/pdf/2607.01308)

### 2.7 VLM 的 KV 复用/压缩（工程前提，别当新意）

**「多 agent 看同一张图、视觉 KV 冗余」这个切口，系统侧早已被反复做过**：Kamera（位置无关 KV 复用 + RoPE 重旋转）[2606.23581](https://arxiv.org/abs/2606.23581)、VLCache（哈希原图、只重算 2-5%）[2512.12977](https://arxiv.org/abs/2512.12977)、MPIC/InfoBlend [2502.01960](https://arxiv.org/abs/2502.01960)、Omni-Flow（分布式分页 KV）[2606.31093](https://arxiv.org/abs/2606.31093)、OxyGen（VLA 同帧跨任务共享）[2603.14371](https://arxiv.org/abs/2603.14371)、PRCR（回看图像）[2606.26631](https://arxiv.org/abs/2606.26631)、LOOK-M / VL-Cache / AirCache / HybridKV / FlashCache / MEDA / FlowMM / GUI-KV，服务侧 EPD 拆分 [2501.05460](https://arxiv.org/abs/2501.05460)。

**跨 LLM KV 复用早于全部 2026 工作**：**DroidSpeak** (NSDI'26) [2411.02820](https://arxiv.org/abs/2411.02820)，明确框定为多智能体流水线的 inter-LLM 通信。

> **结论**：「共享视觉前缀」只能当**工程前提**引用。真正没被做的是「共享视觉 KV **作为 agent 间语义通道**」，不是「共享视觉 KV 省算力」。

### 2.8 更早的血脉（不引会被审稿人抓）

- ⚠️ **协同感知 V2X 做了六年「传中间神经特征而非文本」**：**When2com** (CVPR'20，学 handshake 决定何时发/发给谁)、**Where2comm** (NeurIPS'22，决定发哪块空间切片)、**DiscoNet** (NeurIPS'21，从「看得见全部」的上界蒸馏到带宽受限隐通道——**正是 Vision Wormhole 的训练配方，早五年**)。目录：[Little-Podi/Collaborative_Perception](https://github.com/Little-Podi/Collaborative_Perception)
- **Relative Representations** (ICLR'23)：**"latent communication" 这个词是这篇造的**，基于锚点的**免训练**对齐
- **DIAL** (NeurIPS'16)：连续 agent 间消息的起点
- **Coconut** [2412.06769](https://arxiv.org/abs/2412.06769)：「hidden state 当下一个输入」这个循环的源头

---

## 3. 空格子：修正后的地盘划分

按 **通信介质 × 是否训练 × 任务是否真多模态** 切开：

| 格子 | 状态 | 判据 |
|---|---|---|
| A-② 免训练：**LatentMAS 原样上 VLM**（伪 token 走输入嵌入层） | 🟡 **没人做，成本极低** | 三篇分别引用/复现/压测，**无一给出干净数字**。**这就是你要跑的 baseline** |
| A-③ 免训练：视觉 token 段中继 | 🟢 **没人做，且技术上不需要 $W_a$** ⭐ | projector 输出**天然在输入嵌入空间**。ViF 证明了这批 token 承载视觉证据，却又训了个 $f$ 去处理——免训练版从没试过 |
| A-④ 免训练：**逐层 KV + 多模态**（原始目标格） | 🔴 **没人做，可行性未验证** | 三个结构性障碍见 §5.5 |
| A-④ 需训练：多模态版 C2C | 🔴 **没人做** | 路径清晰，居然无人做 |
| 跨切面：**异构 backbone 的真多模态 latent 通信** | 🔴 空 | VW 异构但纯文本；L²-VMAS 要求共享 $d_{model}$ |
| 跨切面：**消息内容可复原性的度量** | 🔴 空 | 四篇零通道级指标 |
| 跨切面：**多步多模态 latent 思考** | 🔴 空 | L²-VMAS 只注入 8 个 token 就回文本解码；MACF 局部 agent **不解码任何 token**。**无一有 $m$ 步隐空间自回归** |

---

## 4. 一手核实的五条发现（我本人验证，非二手）

### 4.1 ⭐ M-RoPE 位置坍缩：LatentMAS 搬到 Qwen-VL 会静默失效

LatentMAS 内循环（`models.py:339-346`）调 `model(inputs_embeds=..., past_key_values=..., use_cache=True)`，**不传 `position_ids` 也不传 `cache_position`**。纯文本没事，Qwen3-VL 的 M-RoPE 三轴会退化。我跑出来的：

```
prefill pos: [[0,1,2,3,4,4,4,4,6,7,8,9],   ← t 轴
              [0,1,2,3,4,4,5,5,6,7,8,9],   ← h 轴
              [0,1,2,3,4,5,4,5,6,7,8,9]]   ← w 轴  三轴在图像 span 上正常分化
latent step 0: pos=[[0], [0], [0]]
latent step 1: pos=[[0], [0], [0]]
latent step 2: pos=[[0], [0], [0]]
latent step 3: pos=[[0], [0], [0]]
```

**prefill 正常，每个 latent step 全落在位置 0，三轴全塌，不报任何错。** 后果：$m$ 步潜在思维互相重叠在同一位置、跨 agent KV 拼接时第二个 agent 位置从 0 重启、**静默**。
（探针用随机初始化小 config，验证的是**调用路径**，与权重无关。transformers 4.57.1。真实权重上需复现。）

### 4.2 L²-VMAS 传的**不是** KV cache

论文里 "key-value pairs" 是**字典意义**。形状证据：$\mathbf{K}\in\mathbb{R}^{d_{model}}$（**单个**检索向量）、$\mathbf{V}\in\mathbb{R}^{l\times d_{model}}$（**一层** hidden 序列，**无层维度、无 head 维度**）。消费方式是当 $L=8$ 个伪 token embedding 插进解码序列，**走完整前向自己产生 KV**。
→ **「逐层 KV 级通信」那格 L²-VMAS 并没有占。**

其余两条祛魅：**绝对增益只有 +1.8~4.0 点**（2.7-5.4% 是相对提升）；**相对单 agent，token 仍贵 5 倍**（645→3473）；Eq.2 里 $\mathcal{S}_n$ 仍含 "accumulated inter-agent transmissive contexts"，**文本通道可能根本没切断**（全文最大未说明点）。

### 4.3 Vision Wormhole 的「LatentMAS 异构失败」证据有 bug

`methods/latent_mas_hybird.py`：

```python
gram = torch.matmul(W_out_A_f32.T, W_out_A_f32)   # ← 用【全量】W_out_A
if vocab_A != vocab_B:
    W_out_A_f32 = W_out_A_f32[:min_vocab, :]      # ← 之后才截断
    W_in_B_f32  = W_in_B_f32[:min_vocab, :]
rhs = torch.matmul(W_out_A_f32.T, W_in_B_f32)     # ← 用【截断后】的
W_cross = torch.linalg.solve(gram_reg, rhs)
```

Gram 与 RHS 来自不同矩阵 → **不是任何一致最小二乘问题的解**；且按下标配对两个不同词表。**bug 只在 `vocab_A≠vocab_B` 时触发，正好是他们报告失败的 cross-provider 场景。**

### 4.4 ViF 的代码是 stub，且不是免训练

`vif/models/base_stub.py::BaseVLMStub.forward` 直接 `hidden = torch.randn(B,T,D)`——**"base VLM" 是随机张量**；全包 257 行；`ViFBlock` 是可训练 `nn.TransformerEncoder`，有 `train_stage1/2.py`。摘要的 "plug-and-play" 指对 MAS 拓扑即插即用。

### 4.5 Vision Wormhole 的 repo 是 LatentMAS 的直接 fork

同名文件全在（`models.py` 416→853 行、`run.py` 251→1127 行），训练脚本 + 超参 + 锚点数据齐全。**唯一可跑的工程骨架。**

---

## 5. 已知的坑与负面结果（有实证的）

| # | 坑 | 证据 | 级别 |
|---|---|---|---|
| **5.1** | **参数配平后通信收益消失** | [2510.00494] 254M 单模型比 124M+124M 双系统**困惑度更低**；同 budget 一半参数的 soft-embedding baseline "nearly matched"；GPT-2 上双系统只高 +0.4pp；latent 子空间坍缩 $\bar H_{\text{off}}=0.9873$ | 🔴 已实证 |
| **5.2** | **蒸馏的「上界」可能不存在** | L²-VMAS 实测 VLM 多 agent **第 3 轮见顶、第 6 轮起低于单 agent**，token 涨 30× → 「拼全部输入的单 VLM」可能不如 receiver 单独干 | 🔴 已实证 |
| **5.3** | **massive activation 让 MSE loss 学出常数** | [2402.17762] 少数 input-agnostic 维度值大数量级；[2503.03321] LMM 中无关视觉 token 的高激活维度**与 BOS sink 完全相同**。旁证：VW 源码补丁密度（clip ±50/±80/±20、非有限损失跳步、InternVL 关掉 KL） | 🟠 已实证+推断 |
| **5.4** | **裸注入 hidden 有害，必须传 delta** | SDE 实测："w/o delta" 在 4 个设置里**掉到纯文本 baseline 以下**（Q-7B Quasar-T 0.2950 vs NL 0.3050）。且改所有层显著掉点，只能改 top-1~3 层 | 🟠 已实证 |
| **5.5** | **KV 级三个结构性障碍** | (a) RoPE 位置拼接（§4.1 已实锤）；(b) KV 长度随 agent 线性爆炸；(c) **ViF 实测视觉证据集中在 ~2% token → 传全部视觉 KV 是 97% 浪费**。旁证：**MACF 明确用输入嵌入层通信绕开 KV 位置冲突** | 🟡 |
| **5.6** | **冻结 backbone 不省显存** | 梯度要反传到注入层，中间所有层激活都得留。冻结只省优化器状态 | 🟡 |
| **5.7** | **端到端 CE 有平凡解 = 忽略通道** | Interlat 是唯一显式防坍缩的：去掉 $\mathcal{L}_{sep}$ → "shortcut behavior，模型学会直接忽略 latent 通信"；去掉 $\mathcal{L}_{align}$ → 概率质量堆到怪异 token。旁证【一手】C2C 源码 `SFT_train.py:443-449` 的门控正则**被注释掉了** | 🟡 |
| **5.8** | **最便宜的方案天花板最低** | VW：400 步 A6000，但 GSM8K **−4.6pp**、只 1.02× 加速 | 🟡 |

---

## 6. 评估方法学（本次最有价值的沉淀）

### 6.1 主指标不能是准确率

[2607.26773] 实测，同一批实验：

| 模型 | ΔAcc (OPE) | = 消息**存在**效应 (OME) | + 消息**内容**效应 (CAG) |
|---|---|---|---|
| Qwen3-4B / GSM8K | **−1.00pp** | −6.17 | **+5.17** |
| Qwen3-8B | **+1.67pp** | +3.96 | **−2.29**（内容有害但总分为正） |

准确率可由方向相反的两块合成。主 endpoint 预注册为

$$\mathrm{CAG}=\bar U_{\text{cur}}-\bar U_{\text{oth}}$$

（收到**正确**消息 vs 收到**别的样本的**消息）。ΔAcc 只作报告项。

### 6.2 分层指标

| 层 | 测什么 | 手段 |
|---|---|---|
| **L1 通道层** | 消息里有多少信息 | 探针 + **控制任务 selectivity**（Hewitt & Liang）或 **MDL 描述长度**（Voita & Titov，[代码](https://github.com/lena-voita/description-length-probing)）。**绝不要报「消息里有几 bit」**（MI 下界受 $O(\ln N)$ 限制、InfoNCE ≤ log K） |
| **L2 协作层** | 通信本身的因果影响 | CAG、positive signaling/listening、信息不对称任务上的收益 |
| **L3 任务层** | 下游 | 准确率、token、时延 |

### 6.3 必备对照组

零消息 / 随机消息 / **置换消息（shuffle）** / 文本消息 / 免训练 latent 消息 / 上界（信息完整单模型）/ **同参数量 receiver-only LoRA**（对 §5.1 免疫）

### 6.4 ⭐ VLM 独有的两个致命对照（别人都没做）

| 对照臂 | 构造 | 判读 |
|---|---|---|
| $M_{\text{imgswap}}$ | sender 的**图换成无关图**，文本不变 | CAG 不变 → 通道传的全是文本可得信息 |
| $M_{\text{capt}}$ | sender 把所见写成 caption 走文本 | $\mathrm{VSG}=\bar U_{\text{cur}}-\bar U_{\text{capt}}$。**VSG ≤ 0 → latent 通道在 VLM 上没有存在理由** |

### 6.5 样本量（生死线）

配对二值、discordant rate 20%：检出 **5pp 需 ≈620 题**，10pp 需 ≈150，n=60 只能检出 **16pp**。用 **McNemar 精确检验**；**报告必须给 discordant 对数，<20 则结论不可信**。训练随机性 ≥5 seed + ASO 检验（deep-significance），$\varepsilon_{\min}\le0.2$。

### 6.6 训练路线的一号早期信号

$$\Delta L=L(M_{\text{oth}})-L(M_{\text{cur}})$$

训练循环内每 N 步多跑一次 forward，**零成本**。**500 步内不单调上升 → 直接杀掉配方。** 配套防坍缩监控：消息有效秩 $\mathrm{PR}=(\sum\lambda)^2/\sum\lambda^2$、batch 内平均 cosine（→1 即坍缩）、门控开启率（→0 即 shortcut）。

---

## 7. ★ 对你的路线：baseline 会在哪断，每个断点对应什么改进

你的路线是「先跑 baseline，再据结果改进」。**好消息是我们已经知道它会在哪断了**——这四个断点就是你的改进抓手清单，而且是有序的。

| 断点 | 会看到什么现象 | 诊断手段 | → 对应的改进 |
|---|---|---|---|
| **① M-RoPE 位置坍缩**（§4.1，已实锤） | 效果平淡或差，**无任何报错** | dump latent step 的 `position_ids` | 显式传 `position_ids` / `cache_position`；决定 latent thought 在 (t,h,w) 三轴上占什么坐标——**这本身就是一个设计问题，没人回答过** |
| **② $W_a$ 无定义** | 潜在思维漂移 | 算 $\|W_a-I\|_F/\|I\|_F$ | 【调研预判】这其实是**次要**风险——LatentMAS repo 默认 $W_a$ 就是单位阵。**改进方向反而是 §3 的 A-③：走视觉 token 段，projector 输出天然在输入嵌入空间，$W_a$ 问题直接消失** |
| **③ 视觉 KV 97% 冗余** | 显存爆 / 长上下文退化 | 统计 KV 链长度增长 | ViF 已证证据集中在 ~2% token → **选择性传输**。参考 Where2comm 的空间切片选择、LACO 的显著性选择器、KVCOMM 的偏移锚点法（免训练） |
| **④ 裸注入 hidden 有害** | 比纯文本 baseline 还差 | 对比 delta vs 绝对状态注入 | SDE 已给答案：**传增量、且只改 top-1~3 层**。多模态上大概率更严重（视觉 hidden 的范数/各向异性差异更大） |

### 7.1 baseline 阶段的四张卡（全部零训练）

| # | 实验 | 判决标准 | 成本 |
|---|---|---|---|
| 1 | $W_a$ 是否恒等 | $\|W_a-I\|_F/\|I\|_F$ | **1 小时** |
| 2 | KV 前置数值等价性：一次性 prefill vs 分两段续算，逐层比 hidden | 是否逐位相等；不等则定位到 M-RoPE | 3 小时 |
| 3 | ⭐ **盲 agent 视觉可达性探针**（生死线）：A 看图、B 看不见图，只经 latent 通道 | **$X-L>5$pp** | 1 天 |
| 4 | **shuffle-KV 安慰剂**：传给 B 的 KV 换成别的样本的 | 即 CAG。增益不消失 → 通道是触发器不是信道 | 半天 |

**主实验台建议先锁 LLaVA-1.5-7B**（标准 1D RoPE），把 M-RoPE 混淆隔离掉，确认机制本身能不能工作；再切 Qwen-VL 测「位置编码是不是唯一的坑」。骨架用 `/home/yilin/LatentMAS` 原版，只从 `heterogeneous-latent-mas` 抄跨模型对齐算子和 partition runner。

**本机条件**：8× RTX 6000 Ada 48GB，够做 2B–8B 级 VLM 的全部诊断实验。

### 7.2 如果之后要走训练路线

**最值钱的原创点**：VLM 给了文本 MAS 结构上不存在的东西——**M-RoPE 让每个 image token 带 $(h,w)$ 坐标，同一张图喂两个模型即得逐 token 对应**。一张 256-token 图 = 256 个平行锚点，Procrustes 只需 1000–2000 个 → **8 张图就够**。Vision Wormhole 没走这条是因为它允许 sender 是纯文本 LLM，**是做不到，不是没想到**。

**第二个**：把教师从「文本通道」换成「**看完整未裁剪图像的同一个接收方模型**」。VW 的文本教师是**结构性天花板**（GSM8K text 80.8% vs VW 76.2%，学生打不过老师）；split-view 让上界随手可得。

**第三个**：把 CAG 从事后指标**搬进 RL 奖励** $r_{\text{content}}=\mathbb{1}[\text{correct}|M_{cur}]-\mathbb{1}[\text{correct}|M_{oth}]$，从结构上禁止模型学成触发器。L²-VMAS 做不到不是算力问题——**它的 pipeline 里从不构造 $M_{oth}$**。

**别先做 RL**：三个用 RL 的已发表工作没有一个是纯 RL 冷启动；MACF 消融显示跳过中间监督直接上端任务 loss 增益接近零。

### 7.2b 训练路线的显存与时间账（4 卡 × 48GB，实际余量 ~35GB/卡）

#### 显存

**唯一的实测锚点**【一手，VW README】：Qwen3-VL-2B + LFM2.5-VL-1.6B、bs=2、latent_steps=1024、41M codec，**跑在单张 A6000 48GB 上**。A6000 与 RTX 6000 Ada 同级同显存 → **2B 级配置单卡确定可行**。

估算表（**【推断】**，bs=1、seq≈1.5k、含视觉 token）：

| 配置 | 权重 bf16 | 模块 grad+optim | 激活（无 ckpt） | 合计 | 35GB 够吗 |
|---|---|---|---|---|---|
| 2B(sender,no_grad) + 2B(receiver) | 8GB | ~0.5GB | 6–8GB | **~17GB** | ✅ 舒服 |
| 2B + 7B(receiver) | 18GB | 0.5GB | ~15GB | ~34GB | ⚠️ 卡边 |
| 7B + 7B | 28GB | 0.5GB | ~15GB | ~44GB | ❌ 超 |
| 7B + 7B + gradient checkpointing | 28GB | 0.5GB | ~3GB | ~32GB | ⚠️ 勉强 |

**省显存手段，按性价比排序：**

1. ⭐ **sender 离线预计算消息**——sender 是冻结的（VW 就是），它的输出可以**跑一遍存盘**，训练时**根本不加载 sender**。砍掉一半权重 + 全部 sender 激活。磁盘代价：VW 的 codec 输出 1024×512 fp16 ≈ 1MB/样本，20k 样本 = 20GB。**同时也是最大的省时间手段**（sender forward 只跑一次而不是每 epoch 一次）。⚠️ 前提是 sender 不参与训练。
2. ⭐ **注入点往深层挪**——注入在第 $k$ 层而非输入层，只有 $k$ 层以上要保留激活。**SDE 的负面结论正好支持这个**（只改 top-1~3 层最好）——**省显存和方法正确性同向，这种情况很罕见，应该优先利用**。
3. gradient checkpointing：激活 $O(L)\to O(\sqrt L)$，代价 ~30% 时间。
4. 降到 2B/4B（VW 先例证明能出结论）。
5. bs=1 + 梯度累积。

> ⚠️ **注意 LoRA / 冻结 backbone 不省激活**（坑 §5.6），只省优化器状态。别把它当显存方案。

#### 时间

| 环节 | 成本 |
|---|---|
| 一次 codec 训练（VW 量级 400 步 / bs 2） | **3–6 GPU·时** |
| 一轮评测：620 题 × 6 对照臂 = 3720 次推理，7B VLM ~3s/题 | 单卡 ~3h；**4 卡并行 ~45min** |
| **合计一轮迭代** | **半天到一天** |

**时间风险的真正来源不是单次训练，是迭代次数。** 两个杠杆：

1. ⭐ **ΔL 早停**（§6.6）：$\Delta L=L(M_{oth})-L(M_{cur})$，训练循环内每 N 步多跑一次 forward，零成本。**500 步内不单调上升就杀掉配方**。而 VW 总共才 400 步 —— 意味着**你在一次完整训练的时间内就能判死刑**，把"一个配方 = 一天"压到"一个配方 = 1–2 小时"。
2. **分层评测**：早期筛配方用 **150 题**（只能检出 10pp，够粗筛），活下来的才上 **620 题**（检出 5pp）。省 4 倍。≥5 seed 只在最终结论时要，中间迭代 1 seed。

> ⚠️ VW 的 400 步 = **0.27× 数据覆盖率**（800 次抽样 / 3000 条锚点），**连一遍都没过完**。所以它是"最小可行"，不是"充分训练"——别把它当性能上限的参照。

#### 关键判断：把「必须训练」本身当成待验证假设

三种结果对应三条路：

| 诊断结果（第一周四张卡，零训练） | 含义 | 要不要训 |
|---|---|---|
| 修好 M-RoPE 后 CAG **显著为正** | 通道能传信息，免训练就行 | **不用训** |
| CAG **归零**（尤其「换图」臂归零） | 视觉信息压根没过通道 | **训也救不了**，发负面结论 |
| CAG 为正但小 / 不稳 | 通道有信息但用不好 | ✅ **只有这个中间态，训练才有价值** |

### 7.3 止损线

**如果盲 agent 探针在「换图」臂上 CAG 归零**——视觉信息压根没过 latent 通道——那训不训都没意义，把这个负面结论本身发出去。

---

## 8. 文档索引（76 万字原始材料）

### 顶层
| 文件 | 内容 |
|---|---|
| **`research-digest.md`** | 👈 本文，单一入口 |
| `multimodal-latent-mas-survey.md` | 领域盘点 v3（含 v1→v2→v3 的更正记录与教训） |
| `code-to-paper-mapping.md` | **LatentMAS 源码逐行解析**：论文公式落到哪一行、10 处代码与论文对不上的地方 |
| `LatentMAS_论文总结.md` | LatentMAS 原论文精读 |
| `WA_ALI~1.MD` | $W_a$ 从零推导 |
| `LATENT~2.MD` | 生成式假设可证伪性检验手册 |

### `papers/external/` — 论文精读（读别人的）
| 文件 | 内容 |
|---|---|
| `L2-VMAS_method-analysis.md` | 最强竞品，双 latent memory + 三阶段 PPO |
| `ViF_method-analysis.md` | 视觉 relay token；代码是 stub、且要训练 |
| `MACF_method-analysis.md` | 长视频，K=32 通信 token + 三阶段课程 |
| `VisionWormhole_method-analysis.md` | 唯一可跑骨架，LatentMAS 的 fork |
| `00_cross-comparison.md` | 四篇总对比 + 谱系图 + 地盘划分 |

### `papers/research-2026-08-01/` — 训练路线调研
| 文件 | 内容 |
|---|---|
| `10_training-objectives-dissection.md` | **九个方法的损失函数逐个解剖**（含公式与源码行号） |
| `11_channel-evaluation-methods.md` | **通道评估学**（因果审计 / 信息论 / 涌现通信指标） |
| `12_cross-model-alignment-routes.md` | 免训练与少训练的跨模型对齐路线及精度上限 |
| `13_supervision-and-model-size.md` | 监督信号从哪来 + 通信模块参数量对比 |
| `20_trained-comm-proposals.md` | 三个训练式方案（零训练 / Perceiver codec / RL） |
| `21_evaluation-protocol.md` | 完整评估协议（分层指标 / 对照组 / 样本量 / 反作弊） |

### `papers/research-2026-08-01/` — 对抗审查与实验设计
| 文件 | 内容 |
|---|---|
| `22_devils-advocate-training-route.md` | 训练路线最强反对意见 + 总判决 |
| `31_experiment-cards-trainingfree.md` | 免训练路线可执行实验卡（本机环境实测） |
| `32_devils-advocate-trainingfree.md` | 免训练路线最强反对意见 + M-RoPE 一手证据 |
| `30_verdict-and-roadmap.md` | 方向判决与路线（比本文更细的论证） |
