# 方向判决与路线：多模态 × LatentMAS 值不值得做

> **这份文档回答什么**：基于 4 篇论文精读 + 源码核对 + 训练式方案调研（本目录其余 11 份文档），给出**这个方向值不值得做、会踩什么坑、实验怎么设计、训练算法怎么改进和评判**。
>
> **日期**：2026-08-01
> **证据口径**：`【一手】`= 我本人读源码/跑命令核实；`【论文】`= 论文原文；`【调研】`= subagent 报告，我未二次核对；`【推断】`= 判断。

---

## 0. 三十秒版

| 问题 | 答案 |
|---|---|
| 值不值得做 | **值得，但主张要从「我做了个新方法」换成「已发表的负结果是实现 artifact，不是机制失效」** |
| 为什么 | 四篇全部**要训练**，「免训练」整列是空的；而三处宣称「LatentMAS 迁不过来」的负面证据，**逐个查下去都是实现缺陷**（§0.1） |
| 最该先做的一件事 | **§0.1 那个位置编码实验已经做完了**，结论是阳性。下一步是把它扩到真实模型 + 真实 benchmark |
| 主指标 | **不是准确率**，是 **CAG（内容归因增益）**。有实证：ΔAcc = −1.00pp 的格子里 CAG 其实是 **+5.17** |
| 训练路线 | 可行但要换主张。**VLM 独有一条文本 MAS 没有的捷径：M-RoPE 提供免费的稠密平行锚点，8 张图就够做 Procrustes 对齐** |
| 本机条件 | **8× RTX 6000 Ada 48GB**，够做 2B–8B 级 VLM 的全部诊断实验 |

---

## 0.1 ⭐ 本次最重要的发现：把 LatentMAS 搬到 Qwen-VL，位置编码会静默坍缩

**【一手，我本人跑的】** LatentMAS 的隐空间自回归内循环调用签名是（`models.py:339-346`）：

```python
outputs = self.model(inputs_embeds=latent_embed,   # [B,1,D]
                     attention_mask=latent_mask,
                     past_key_values=past, use_cache=True)
```

**没有 `position_ids`，也没有 `cache_position`。** 纯文本模型下这没问题——HF 会从 past 长度推出位置。但 Qwen3-VL 用的是 **M-RoPE（三轴 t/h/w）**，`get_rope_index` 只在给了 `input_ids + image_grid_thw` 时才触发；走 `inputs_embeds` 且不给 `position_ids` 时会退化。

实测输出（transformers 4.57.1，随机初始化的小号 Qwen3VL config，位置计算与权重无关）：

```
prefill pos: [[0,1,2,3,4,4,4,4,6,7,8,9],    ← t 轴
              [0,1,2,3,4,4,5,5,6,7,8,9],    ← h 轴
              [0,1,2,3,4,5,4,5,6,7,8,9]]    ← w 轴   三轴在图像 span 上正常分化
latent step 0: pos=[[0], [0], [0]]
latent step 1: pos=[[0], [0], [0]]
latent step 2: pos=[[0], [0], [0]]
latent step 3: pos=[[0], [0], [0]]
```

**prefill 的 M-RoPE 完全正常，但每一个 latent step 都落在位置 0，三个轴全塌，而且不报任何错。**

后果：
1. $m$ 步潜在思维**互相重叠在同一个位置**，也和序列开头撞车；
2. 跨 agent 的 KV 前置拼接，第二个 agent 的位置从 0 重启；
3. **静默** —— 你只会看到「效果不好」，不会看到任何异常。

> **这条把整个方向的主张换掉了。** 三处宣称「LatentMAS 类方法迁不到多模态/异构」的证据，现在逐个可疑：
> - **Vision Wormhole 附录 F**（跨 tokenizer 直传崩溃）→ 实现有最小二乘 bug（§2.3，我核过源码）
> - **MACF**「测了但配置疑似错配」→ 【调研】
> - **L²-VMAS**「not directly transferable to VMAS」→ **从未做过实验**，只是论证
>
> 加上这条位置编码坍缩，合理的新主张是：**「已发表的多模态 latent MAS 负结果，是位置编码与对齐算子的实现 artifact，而非机制失效」** —— 这是一篇分析/勘误型论文，不和 L²-VMAS 抢方法赛道，单卡可做，而且**结论无论正负都能发**。

**注意**：探针用的是随机初始化的小 config，验证的是**调用路径**而非模型质量。下一步必须在真实 Qwen3-VL / Qwen2.5-VL 权重上复现，并确认修好 `position_ids` 之后行为是否恢复。

---

## 0.2 关于 L²-VMAS 的三条祛魅（都改变了竞争格局）

【论文，逐条核实】

1. **它传的不是 KV cache。** 论文里的 "key-value pairs" 是**字典意义**的键值对，与 KV cache 无关。形状证据：$\mathbf{K}\in\mathbb{R}^{d_{model}}$（**单个**向量做检索键），$\mathbf{V}\in\mathbb{R}^{l\times d_{model}}$（**一层** hidden state 序列，**没有层维度、没有 head 维度**）。消费方式是当作 $L=8$ 个伪 token 的 embedding 插进解码序列，走一次完整前向自己产生 KV。
   → **「逐层 KV 级通信」那一格，L²-VMAS 并没有占。它占的是「输入嵌入层伪 token」那格。**
2. **摘要的 2.7–5.4% 是相对提升百分比，不是绝对点数。** 核对主表：GLM 66.6→69.5（**+2.9 点**，报 +4.3%）、InternVL 68.0→69.8（**+1.8 点**）、Qwen3-VL-8B-Thinking 72.5→76.5（**+4.0 点**）。**绝对增益只有 1.8–4.0 点。** 同理 token 的 −21.3~−44.8% 只相对文本 VMAS；**相对单智能体仍贵约 5 倍**（645 → 3473 tokens）。
3. **它的文本通道可能根本没被切断。** Eq.2 里 $\mathcal{S}_n$ 的定义仍包含 "accumulated inter-agent transmissive contexts from predecessor agents"，全文没有一处说明 agent 之间还传不传文本、传多少。【推断】这与「token 只降 21–45% 而非 90%+」自洽。**这是全文最大的未说明点。**

venue 已确认：**ICML 2026 已接收**（正文页脚 "Machine Learning, ICML" + 作者主页标注 Accepted 04/2026）。之前标的「待核」可以销掉。

---

## 1. 地形：四篇全部要训练，免训练那一整列是空的

【调研+一手】四篇逐个核实的结论：

| | 通信介质 | 要训练？ | 训什么 | 成本 | 代码 |
|---|---|---|---|---|---|
| **L²-VMAS** | 输入嵌入层伪 token（L=8）+ 熵触发 | **要** | 三个压缩器 + Gumbel 门控，**三阶段 PPO**（100k+80k+50k 步） | **8×H200** | 无 |
| **MACF** | 定长通信 token（K=32） | **要** | 双侧 LoRA + 每 backbone 一个 2 层 MLP adapter，三阶段课程 | 4×A100-80G | 无 |
| **ViF** | 视觉 token 段（~2% relay token） | **要** | Projector + 自研 Transformer 块 $f$；Stage 2 连 base LLM 都解冻 | 未报 | **stub-only** |
| **Vision Wormhole** | 视觉端口（dummy image 残差注入） | **要** | 每模型一个 ~41M 的 Universal Visual Codec | **400 步 / bs 2 / A6000，约 3–6 GPU·时** | **完整可跑** |

**两条一手更正**（推翻我之前给出的说法）：

- ⚠️ **ViF 不是 training-free**。【一手】`vif/models/vif_block.py` 的 `ViFBlock` 是可训练 `nn.TransformerEncoder`；`vif/utils/training.py::build_optimizers` 把 `vif.` / `projector` / `vision` 参数单列一组学习率。论文摘要的 "plug-and-play" 指的是**对 MAS 拓扑即插即用**，不是不用训练。
- ⚠️ **ViF 的公开代码跑不了**。【一手】整个 `vif/` 包 257 行；`vif/models/base_stub.py::BaseVLMStub.forward` 直接 `hidden = torch.randn(B,T,D)` —— **"base VLM" 是随机张量**；`scripts/run_multi_agent.py` 调的是 `VLMInterface(mode='stub')`。公式（relay 选择、注意力再分配）是真的，但从未接过真实 VLM。

> **推论**：所以**唯一能当工程骨架的是 Vision Wormhole 的 `heterogeneous-latent-mas`**，它是 LatentMAS 代码库的直接 fork（同名文件全在，`models.py` 416→853 行），训练脚本、超参、锚点数据齐全。

---

## 2. 空格子在哪，以及每个空格的性质

【调研】把设计空间按 **通信介质 × 是否训练 × 任务是否真多模态** 切开，真多模态那一象限的免训练列：

| 空格 | 性质 | 判据 |
|---|---|---|
| **A-② 免训练**：LatentMAS 原样搬到 VLM（伪 token 走输入嵌入层） | **没人做，成本极低** | 三篇分别引用/复现/压测，**无一给出干净数字**。这是文献里缺的一个 baseline 格子，**不是新方法** |
| **A-③ 免训练**：视觉 token 段的免训练中继 | **没人做，且技术上不需要 $W_a$** ⭐ | projector 输出**天然就在输入嵌入空间**。ViF 证明了这个张量承载视觉证据（丢掉掉 40.7 分），却又训了个 $f$ 去处理它——**免训练版本从未被试过** |
| **A-④ 免训练**：逐层 KV + 免训练 + 多模态（**你的原始目标格**） | **没人做，可行性未验证** | 三个结构性障碍见 §3.5 |
| **A-④ 需训练**：多模态版的 Cache-to-Cache | **没人做** | 技术路径清晰，居然无人做 |

### 2.1 A-③ 那格是本次最漂亮的发现

$W_a$ 之所以在 VLM 上没定义，是因为它要把末层 hidden **掰回文本 token 嵌入表**。但**视觉 token 本来就是 projector 直接产出的、已经在输入嵌入空间里的向量** —— 走视觉 token 段这条路，**根本不需要 $W_a$，理论断点自动消失**。

【推断】这可能是整个方向里性价比最高的一个切口：ViF 已经证明了「哪 2% 的视觉 token 承载证据」，而它把这批 token 交给一个**训出来的** $f$ 去重新语境化。免训练版本（直接搬运、或只做模长/白化归一）**从没人试过**。

### 2.2 但别急着宣称 A-④ 可行

【调研】三个具体障碍，每一个都是结构性的：

1. **RoPE 位置拼接**：多 agent 各自从 0 起的位置编码，拼接后语义会乱。【一手】LatentMAS 的 `past_key_values` 在 agent 循环里从不清空、是**堆叠式**的（见 `code-to-paper-mapping.md` §3.4），到 VLM 的 2D/M-RoPE 上不能假设同样成立。
2. **KV 长度线性爆炸**：多个 agent × 每张图几百到几千视觉 token。
3. **97% 带宽浪费**：ViF 实测视觉证据集中在 ~2% 的 token 上，传全部视觉 KV 是纯浪费；20 轮 circular 拓扑下必爆。

【调研】旁证：**MACF 明确用输入嵌入层通信来绕开 KV 位置冲突** —— 这是一个「别人试过、选择绕开」的信号。

### 2.3 ⭐「LatentMAS 上异构失败」这个负面证据，站不住

Vision Wormhole 附录 F 报告：跨 tokenizer 直传时 Qwen+Gemma 在 GSM8K 上准确率全程 ≤0.5%、PPL 从 8.1e5 涨到 8.6e7。这被广泛读作「LatentMAS 思想不能上异构」。

**【一手】我读了那份实现 `methods/latent_mas_hybird.py`，它有两个问题：**

```python
gram = torch.matmul(W_out_A_f32.T, W_out_A_f32)     # ← 用【全量】W_out_A 算 Gram
...
if vocab_A != vocab_B:
    W_out_A_f32 = W_out_A_f32[:min_vocab, :]        # ← 之后才截断
    W_in_B_f32  = W_in_B_f32[:min_vocab, :]
rhs = torch.matmul(W_out_A_f32.T, W_in_B_f32)       # ← 用【截断后】的算 RHS
W_cross = torch.linalg.solve(gram_reg, rhs)
```

1. **Gram 矩阵与 RHS 来自不同的矩阵**。正规方程解的是 $(\tilde X^\top \tilde X+\lambda I)^{-1}X^\top Y$，其中 $\tilde X$（全量）$\ne X$（截断）。**这不是任何一个一致的最小二乘问题的解。**
2. **按下标配对两个不同的词表**。`min_vocab` 截断隐含假设「模型 A 的 token id $i$ 和模型 B 的 token id $i$ 是同一个词」。跨 tokenizer 时这完全不成立，配对本身是无意义的。

注意 bug 只在 `vocab_A != vocab_B` 时触发 —— **正好就是他们报告不稳定的那个 cross-provider 场景**。

> **结论**：被证伪的是**「用词表配对的闭式算子跨 tokenizer 直传」这一个具体做法的一个有 bug 的实现**，不是「LatentMAS 思想不能上异构」。修 bug 重做一遍，本身就是可发表的更正。

---

## 3. 会踩什么坑（按严重度）

### 3.1 🔴 参数配平假阳性 —— 已有实证先例，不是推测

【调研，2510.00494 全文核实】在结构同构的 latent 通信双系统上做参数配平对照：254M 单模型（=124M base + 124M coprocessor）验证困惑度**更低**；同 latent budget、一半参数的 soft-embedding baseline "nearly matched" 最好的双模型变体，GPT-2 上双系统只高 **+0.4pp**；latent 子空间坍缩量化为 $\bar H_{\text{off}}=0.9873$、silhouette $=-0.1694$。

**即：「训出来的增益来自参数量而非通信」这个假阳性，已经被别人撞到过。** 审稿人一定会问「是不是参数量的功劳」，你必须有同参数量 receiver-only LoRA 的配平臂。

### 3.2 🔴 上界可能根本不存在

所有「蒸馏」路线都假设存在一个信息完整的教师（DiscoNet 式）。但【调研，L²-VMAS 原文】实测：**VLM 多 agent 准确率第 3 轮见顶，第 6 轮起低于单 agent baseline，token 涨 30×**。

也就是说「把所有 agent 的输入拼给一个 VLM」很可能**不如 receiver 单独干** —— privileged teacher 在 VLM 长上下文场景不成立。**这条零训练、一天可证伪，应该在做任何其他事之前先做。**

### 3.3 🟠 massive activation 会让 MSE 类 loss 学出常数

【调研】2402.17762：少数 input-agnostic 维度值大数量级，LLaMA2-7B 上置零 4 个即崩溃。2503.03321：LMM 中无关视觉 token 的高激活维度**与 BOS sink 维度完全相同**，且存在明确 modality gap。

合起来：$\|h_{\text{vis}}-h_{\text{text}}\|^2$ 的梯度被 sink 维度独占，学出来的 $\Delta$ 主要是**常数偏置** —— 这恰好制造 §4 里说的 OME 型假阳性。

【一手旁证】Vision Wormhole 源码的补丁密度很能说明问题：logit clip ±50/±80/±20、非有限损失直接跳步、InternVL 要专门关掉 KL 项。

### 3.4 🟠 「冻结 backbone 省显存」是误解

梯度必须从 receiver 输出反传到注入层，**中间所有层的前向激活都要保留**；冻结只省优化器状态。两个 8B VLM 权重 bf16 就 32GB，24/48GB 卡基本做不了。

而成本坐标里有个残酷的相关性：**最便宜的方案（Vision Wormhole，400 步 A6000）恰好是天花板最低的那个**（GSM8K −4.6pp、只有 1.02× 加速）。

### 3.5 🟡 SDE 的负面结论在多模态上只会更严重

【调研】SDE 实测：注入**原始** hidden state（"w/o delta"）在 4 个设置里掉到**纯文本 baseline 以下**（Q-7B Quasar-T 0.2950 vs NL 0.3050）。且改所有层会显著掉点，只能改 top-1~3 层。

视觉 hidden 的范数/各向异性与文本差异更大 → **多模态版大概率必须用 delta 或按模态归一，不能裸注入**。

### 3.6 🟡 novelty 的复述测试

如果你的方法能被三个外行博士生复述成「**C2C 换成 VLM**」，就没有 novelty。这一条是可以自查的。

---

## 4. 实验怎么设计（能不能验证假设 —— 能，而且主指标要换）

### 4.1 ⭐ 主指标不能是准确率

【调研，2607.26773 实测】同一批实验里：

| 模型 | ΔAcc (OPE) | = 消息**存在**的效应 (OME) | + 消息**内容**的效应 (CAG) |
|---|---|---|---|
| Qwen3-4B / GSM8K | **−1.00pp** | −6.17 | **+5.17** |
| Qwen3-8B | **+1.67pp** | +3.96 | **−2.29**（内容有害，但总分为正） |

**准确率可以由方向相反的两块合成。** 主 endpoint 必须预注册为

$$\mathrm{CAG} = \bar U_{\text{cur}} - \bar U_{\text{oth}}$$

即「收到**正确**消息」与「收到**别的样本的**消息」之差。ΔAcc 只作报告项。

训练路线尤其危险：端到端 CE 最容易下降的方向就是把模块训成**触发器**而非**信道**，签名正是 **OME 大而 CAG≈0**。

### 4.2 ⭐ VLM 独有的两个致命对照（别人都没做，成本极低）

| 对照臂 | 构造 | 判读 |
|---|---|---|
| **$M_{\text{imgswap}}$** | 把 sender 的**图换成无关图**，文本上下文不变 | 若 CAG 不变 → 通道传的全是文本可得信息，**$W_a$ 断点根本没被跨过** |
| **$M_{\text{capt}}$** | sender 把所见写成 caption 走文本通道 | 定义 $\mathrm{VSG}=\bar U_{\text{cur}}-\bar U_{\text{capt}}$。**VSG ≤ 0 则整个 latent 通道在 VLM 上没有存在理由**（文本还可读可审计） |

【调研】Vision Wormhole 是最接近的工作，**完全没有任何通道级指标**，只有下游准确率和 wall-clock。这是确认存在的空档。

### 4.3 ⭐ 样本量是生死线（已算过）

配对二值结果、discordant rate 20% 时：

| 要检出的效应 | 需要题数 |
|---|---|
| 5pp | **≈620** |
| 10pp | ≈150 |
| 16pp | 60 |

正确检验是 **McNemar 精确检验**（discordant pairs > 25 时用卡方版），power **只取决于 discordant pair 数量**。**报告里必须给出 discordant 对数，< 20 则任何结论不可信。** 训练随机性用 ≥5 seed + ASO 检验（deep-significance），$\varepsilon_{\min}\le 0.2$ 才算「稳定优于」。

### 4.4 ⭐ 训练配方好坏的一号早期信号（省算力关键）

$$\Delta L = L(M_{\text{oth}}) - L(M_{\text{cur}})$$

训练循环内每 N 步多跑一次 forward 即可，**零额外成本**，是 CAG 在 loss 层面的免费代理。

**判决规则：500 步内 $\Delta L$ 不单调上升 → 直接杀掉这个配方，不用等下游跑完。**

配套三个防坍缩监控：
- 消息**有效秩** $\mathrm{PR}=(\sum\lambda)^2/\sum\lambda^2$
- batch 内平均 cosine（→1 即坍缩）
- 门控开启率（→0 即 Interlat 说的 shortcut behavior）

### 4.5 探针类指标的纪律

- L1 探针**不做控制任务就是废数**：必须同时报 control task（随机标签）准确率并给 selectivity（Hewitt & Liang, EMNLP'19），更稳的是改报 **MDL 描述长度**（Voita & Titov）。
- **绝不要报「消息里有几 bit」**：distribution-free MI 下界受 $O(\ln N)$ 限制、InfoNCE $\le \log K$。只能报**相对置换参照的可分性**。

---

## 5. 训练算法怎么改进（第四问）

### 5.1 已有方法的训练目标，其实只有四种绕法

【调研】没有任何一个方法使用「平行隐状态数据」—— 因为这东西物理上不存在。四种绕法：

| 绕法 | 代表 | 本质 | 弱点 |
|---|---|---|---|
| **A 自造配对** | Vision Wormhole | 同一冻结模型，teacher 走文本、student 走视觉端口 | 教师是**结构性天花板**（学生打不过老师） |
| **B 干脆不对齐** | C2C / MoT / MACF | 任务 CE 一路 backprop 穿过通信模块 | 有平凡解 = 忽略通道 |
| **C 特权教师** | DiscoNet | teacher 看全局视角 | **只在仿真器里能拿到**；VLM 上界可能不存在（§3.2） |
| **D 结果奖励** | L²-VMAS | 只有 accuracy，PPO 硬啃 | 样本效率极低（230k 步） |

**只有 Interlat 显式设计了防坍缩损失并写出了两个失效模式的名字**：

$$\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{task}}+\lambda_S\mathcal{L}_{\text{sep}}+\lambda_A\mathcal{L}_{\text{align}}$$

去掉 $\mathcal{L}_{\text{sep}}$（匹配/错配 latent 的加权 JS 散度）→ *"shortcut behavior，模型学会直接忽略 latent 通信"*；去掉 $\mathcal{L}_{\text{align}}$ → 模型把概率质量堆到怪异 token 上刷目标。

【一手】旁证：C2C 源码 `script/train/SFT_train.py:443-449` 有一段**被注释掉**的门控稀疏正则 `loss += 0.0025 * gate` —— 说明「要不要正则化门」在他们那儿也是悬而未决的。

### 5.2 ⭐ VLM 给了文本 MAS 结构上不存在的东西：免费的稠密平行锚点

【调研，2409.12191 核实】M-RoPE 给每个 image token 分配 $(h,w)$ 坐标，所以**同一张图喂给两个模型，即得逐 token 对应**。

- 一张 256-token 的图 = **256 个平行锚点**
- 500 张图 = 128,000 行
- 而 Procrustes 类对齐只需 ~1000–2000 个锚点 → **8 张图就够**

**Vision Wormhole 没走这条路，是因为它允许 sender 是纯文本 LLM，图像锚点在它的设定里不存在 —— 是做不到，不是没想到。** 你的设定（VLM↔VLM）里它天然存在。

### 5.3 三个在训练量轴上拉开的方案

| | 训练量 | 结构 | 消息大小 | 成本 |
|---|---|---|---|---|
| **A** | **零 SGD 步** | <100 个标量 + 8.4M 闭式解系数（网格锚点 + M-RoPE 去旋转/再旋转搬运） | 0.66 MB | **1 卡半天** |
| **B** | 中 | ~37M 参数 Perceiver codec | 32 KB | 4 卡 8–12h，20k 无标注样本 |
| **C** | 重 | B + RLOO | 自适应 | 8 卡 1–2 天 |

**B 的核心差异化**：把教师从「文本通道」换成「**看完整未裁剪图像的同一个接收方模型**」。split-view 构造让「上界 = 没裁剪的原图」随手可得 —— 这是 DiscoNet 式 holistic-view teacher **第一次不需要仿真器**。

**C 唯一真正新的东西**：把 CAG 从事后审计指标**搬进 RL 奖励函数**

$$r_{\text{content}}=\mathbb{1}[\text{correct}\mid M_{\text{cur}}]-\mathbb{1}[\text{correct}\mid M_{\text{oth}}]$$

从结构上禁止模型学成触发器。【调研】L²-VMAS 做不到这件事**不是算力问题** —— 它的 pipeline 里从不构造 $M_{\text{oth}}$，没有反事实对照就算不出 CAG。

**不要先做 C**：三个用 RL 的已发表工作没有一个是纯 RL 冷启动；MACF 消融显示跳过中间监督直接上端任务 loss 增益接近零。**你不能优化一个你还测不准的量。**

---

## 6. 推荐路线

### 第一周：四张实验卡（全部零训练，按性价比排序）

| # | 实验 | 看什么 | 判决 | 成本 |
|---|---|---|---|---|
| **0** | ✅ **已完成**：M-RoPE 位置坍缩探针（§0.1） | latent step 的 position_ids | **已是阳性** → 下一步在真实 Qwen3-VL 权重上复现 | 已花 10 分钟 |
| **1** | **$W_a$ 是否恒等**：加载目标 VLM，算 `realign_matrix`，看 $\|W_a-I\|_F/\|I\|_F$ | 是否 ≈0 | 【调研】预判 $W_a$ 断点其实是**次要**风险——LatentMAS repo 默认就是单位阵（`code-to-paper-mapping.md` §4.1） | **1 小时** |
| **2** | **KV 前置的数值等价性**：同一段输入，「一次性 prefill」vs「分两段用 past_key_values 续算」，逐层比对 hidden | 是否逐位相等 | 不等 → 定位到 M-RoPE；这是卡 0 在真实模型上的确认 | **3 小时** |
| **3** | ⭐ **盲 agent 视觉可达性探针**（生死线）：agent A 看图、agent B 看不见图，只经 latent 通道。测 B 的准确率 $X$ 与「B 什么都没收到」的 $L$ | **$X-L>5\text{pp}$** | 这是整个方向的生死线：**视觉信息到底能不能过 latent 通道** | 1 天 |
| **4** | **shuffle-KV 安慰剂对照**：把传给 B 的 KV 换成**别的样本**的 | 即 CAG。增益若不消失 → 通道是触发器不是信道 | 卡 3 的必备配对，缺了结论不成立 | 半天 |
| **5** | **上界存在性**：拼全部 agent 输入的单 VLM vs receiver 单独 vs 文本 MAS | 上界是否显著高于两者 | 若上界不存在 → 蒸馏路线（B/C）全部作废 | 1 天 |
| **6** | **视觉 hidden 分布体检**：top-4 维度能量占比 | <20%？ | 若 sink 维度独占 → MSE 类 loss 不可用，必须换 delta / 白化 | 1 小时 |

> 【调研】主实验台建议先锁 **LLaVA-1.5-7B**（标准 1D RoPE），把 M-RoPE 这个混淆因子隔离掉，确认机制本身能不能工作；再切到 Qwen-VL 系测「位置编码是不是唯一的坑」。
> 骨架用 `/home/yilin/LatentMAS` 原版，只从 `heterogeneous-latent-mas` 抄两个零件（跨模型对齐算子 + partition runner）。

### 第二步：选一个能活下来的主张（都不和 L²-VMAS 抢格子、单卡可做、结论正负都能发）

- **B1「审计优先」**（魔鬼代言人首推）：把 causal audit 协议扩到 VLM，加**视觉专属对照臂**（换图不换文 / 图文错配）。主张不是「我涨点」，是「**视觉 latent 通道到底传没传视觉信息**」。Vision Wormhole 没有任何通道级指标 —— 确认存在的空档。
- **B2「低秩补丁」**：per-layer rank-32 $\Delta$KV + 只在 receiver **末层**监督（Kamera 证明丢失部分低秩且集中深层；Model Stitching 证明必须在末层做特征匹配）。主张变成「**用 C2C 的 1/50 参数拿到可比效果**」—— 参数量小本身就是「是不是参数量功劳」这个质疑的免疫。
- **B3「负面结论」**：设计**只有看图才能答**的题（细粒度定位、计数、纹理），测 latent 通道 vs 文本通道。传不过去也是一篇有价值的论文，成本远低于正面工作。

### 止损线

**如果实验 1 显示上界不存在，且 A-② 的 baseline 在「换图」臂上 CAG 归零** —— 即视觉信息压根没通过 latent 通道传输 —— 那么整个「VLM latent 通信」的前提就不成立，训不训都没意义。**此时应该把这个负面结论本身发出去（B3），而不是继续做方法。**

---

## 7. 三个我没能消除的风险

1. **跨模型 KV 的线性相容性没有任何直接证据。**【调研】唯一先例 2412.20677 是**同模型跨 head**。所有 relative-representation / Procrustes / vec2vec 的实证都建立在「一条样本一个向量」的 embedding 上，**没有一篇验证过 per-token × per-layer × per-head 的 KV 张量满足同样的几何**。这既是最大的未验证假设，也是最大的机会 —— 但顺序是「先验证，再宣称」。
2. **split-view 构造可能太简单**。如果裁出的两个视角各自都能独立答对，CAG 会趋近 0，不是因为通道没用，而是因为任务不需要通道。**必须先跑 sanity：sender 单独 / receiver 单独 / oracle 三条准确率必须显著分离**，否则数据集废了。
3. **图像 token 网格对应依赖固定分辨率**。Qwen2.5-VL 的 dynamic resolution 会让 token 网格随输入变化，方案 A 必须锁死输入分辨率；做 baseline 对比时**两边都要锁**，否则测到的是分辨率差异。

---

## 8. 本目录文档索引

| 文件 | 内容 |
|---|---|
| `../external/L2-VMAS_method-analysis/` | 最强竞品，双 latent memory + 三阶段 PPO |
| `../external/ViF_method-analysis.md` | 视觉 relay token；**代码是 stub、且要训练** |
| `../external/MACF_method-analysis.md` | 长视频，K=32 通信 token + 三阶段课程 |
| `../external/VisionWormhole_method-analysis.md` | **唯一可跑骨架**，LatentMAS 的 fork |
| `../external/00_cross-comparison.md` | 四篇总对比表 + 谱系图 + 地盘划分 |
| `10_training-objectives-dissection.md` | 九个方法的损失函数逐个解剖 |
| `11_channel-evaluation-methods.md` | 通道评估学（因果审计/信息论/涌现通信指标） |
| `12_cross-model-alignment-routes.md` | 免训练与少训练的跨模型对齐路线 |
| `13_supervision-and-model-size.md` | 监督信号从哪来 + 通信模块参数量对比 |
| `20_trained-comm-proposals.md` | 三个训练式方案（A/B/C） |
| `21_evaluation-protocol.md` | 完整评估协议（分层指标/对照组/样本量/反作弊） |
| `22_devils-advocate-training-route.md` | 训练路线的最强反对意见 + 总判决 |
| `31_experiment-cards-trainingfree.md` | 免训练路线的可执行实验卡（本机环境实测） |
| `32_devils-advocate-trainingfree.md` | 免训练路线的最强反对意见 + M-RoPE 一手证据 |
