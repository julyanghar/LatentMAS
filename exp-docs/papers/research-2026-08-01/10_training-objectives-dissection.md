# 训练目标学：可训练 latent 通信模块的损失函数逐篇解剖

> 标注规则：【论文】= 论文原文明确写的；【源码】= 我读了实际代码；【推断】= 我的判断，论文没这么说。
> 本地仓库两份已核实：`/home/yilin/tmp/mm-latent-repos/heterogeneous-latent-mas`（Vision Wormhole 官方码，README 自带 arXiv 2602.15382）、`/home/yilin/tmp/mm-latent-repos/ViF`（另一条线，2509.21789）。C2C 我另外 clone 到 `/tmp/claude-1008/-home-yilin/e4c133de-4904-4f07-bf05-a54b12262277/scratchpad/C2C`。

---

## 0. 先说结论：这个领域的"目标函数"其实只有四种

拆完九篇之后，把"损失函数长什么样"这层剥开，真正的分野不在损失项的形式，而在**监督信号从哪里变出来**。所有方法都卡在同一个死结上——**没有 (发送方隐状态, 接收方应该收到的隐状态) 这种平行语料**，因为这东西在物理上就不存在。四种绕法：

| 绕法 | 代表 | 本质 |
|---|---|---|
| **A. 自造配对** | Vision Wormhole | 同一个冻结模型，teacher 走文本通道、student 走视觉端口。配对是**制造**出来的，不是采集的，成本 = 一堆无标注文本 |
| **B. 干脆不对齐** | C2C、MoT、MACF、Where2comm、When2com | 不给通信模块任何直接监督，让接收方的任务 CE 一路 backprop 穿过通信模块 |
| **C. 特权信息教师** | DiscoNet | teacher 看得到全局视角（上界），student 只有单视角。**只在仿真器里能拿到** |
| **D. 结果奖励** | L²-VMAS | 只有 accuracy，PPO 硬啃 |

这个分类比"用不用蒸馏"有用得多，因为它直接决定了你的数据成本和能不能在 VLM 上复现。

---

## 1. 逐篇解剖

### 1.1 C2C（2510.03215, ICLR'26）—— Cache Fuser

**(a) 损失函数的数学形式**

【论文】主损失就是**接收方输出上的标准 next-token 预测损失**，和 SFT 一模一样。论文没给方程号，原话是 "standard next-token prediction loss on the Receiver's response predictions"。

$$\mathcal{L}_{\text{C2C}} = -\sum_t \log p_{\text{receiver}}(y_t \mid x, y_{<t};\ \text{FusedKV})$$

**注意：论文里根本没有 Fuser 的数学公式**（我 fetch 了 abs / html / v1 三个版本，只有 Eq.(3)(4) 的高层记法，Fuser 内部靠 Figure 5 示意）。所以下面的公式是我**【源码】**从 `rosetta/model/projector.py::C2CProjector.forward` 抄出来的，这是目前唯一能拿到 Fuser 真实数学形式的地方：

$$
\begin{aligned}
z_K &= \text{MLP}_1(W_{\text{in}}^K [\,K_{\text{src}} \,\Vert\, K_{\text{tgt}}\,]) \\
\Delta K &= W_{\text{out}}^K \text{MLP}_2^{\text{proj}}(z_K), \qquad s_K = \sigma(W^K_{\text{scalar}} \text{MLP}_2^{\text{scalar}}(z_K)) \\
g_K &= \begin{cases}
\sigma\!\left(\dfrac{\ell_K + g}{\tau}\right), & g \sim \text{Gumbel}, \ \text{训练时} \\[6pt]
\mathbb{1}[\ell_K > 0], & \text{推理时（硬二值）}
\end{cases} \\
K_{\text{out}} &= \boxed{K_{\text{tgt}} + g_K \cdot s_K \cdot \Delta K}
\end{aligned}
$$

Value 路径同构。三件事值得记：
- **残差**：输出 = 接收方原 KV **加**一个增量，不是替换。
- **两级门**：`g_K` 是**逐层标量**（per-layer scalar parameter，Gumbel-sigmoid 学的，推理时硬二值化 = 这层到底注不注）；`s_K` 是**逐 head 逐 token 的连续缩放**（input-aware，就是论文说的 dynamic weighting）。这两个是不同粒度的东西，论文文字容易读混。
- 温度退火【源码】：`gate_temp = init * (final/init)^(step/anneal_steps)`，指数退火，config 里 `1.0 → 0.001`，`anneal_steps=1929`。

**⚠️ 一个只有读码才能发现的东西**：`script/train/SFT_train.py:443-449` 有一段**被注释掉的门控稀疏正则**：

```python
# gate = torch.sigmoid(gate_logit / proj.gate_temperature)
# loss += 0.0025 * gate
```

也就是说他们试过给门加 L1 式惩罚（逼它少注入），最后**没启用**。【推断】这暗示要么效果不好，要么门本来就不容易全开——但无论哪种，说明"要不要正则化门"在他们那儿是个悬而未决的问题。

**另外**：repo 里还有一条**没被采用的路线** `script/train/oracle_train_kvcache_mse.py`，直接对 hidden state 做逐层匹配，支持三种损失【源码】：

$$\mathcal{L}_{\text{oracle}} = \frac{\sum_{t} m_t \cdot d(h^{\text{rosetta}}_t,\, h^{\text{base}}_t)}{\sum_t m_t},\quad d \in \{\text{MSE},\ \text{L1},\ 1-\cos\}$$

（`compute_single_layer_hidden_states_loss`，attention-mask 加权平均）。但它调用的 `oracle_forward` 在发布的 `rosetta/` 模块里**找不到定义**，属于失效/实验残留代码。**这条信息对你有用**：C2C 作者同时实现了"表示匹配"和"端到端 CE"两条目标函数，最后 ship 的是 CE。

**(b) 监督信号来源**：OpenHermes-2.5 的**指令-回答对**，即普通 SFT 数据。**不需要平行隐状态**，但需要有标注的问答语料。

**(c) 训练成本**【论文+源码 config `recipe/train_recipe/C2C_0.6+0.5.json`】：
- 主实验 OpenHermes-2.5，**500,000 样本，1 epoch**；lr `1e-4` linear，warmup 10%，grad clip 1.0
- 有效 batch = `per_device 4 × grad_accum 8 × num_processes 8` = **256**（和论文报的 256 对上）
- 变体：LongBench-E 1,896 样本（batch 16）；MMLU auxiliary_train 15,000（batch 128）
- GPU：README 给的是 `torchrun --nproc_per_node=8`；**论文没明说训练用什么卡**，只说评测在单张 A100

**(d) 冻不冻 backbone**：全冻。config 里明写 `"freeze": ["teacher","base"]`【源码】，论文也说 "both Sharer and Receiver LLMs remain frozen"【论文】。

**(e) 不稳定/坍缩**：论文**没有**讨论。Table 8 的消融是逐步加组件、单调变好。【推断】这是个缺口——纯 CE + 残差门控存在一个平凡解（门关到底、退化成接收方单模型），论文没做"门是不是真的开了"的报告，那个被注释掉的正则更像是撞过这个问题的痕迹。

---

### 1.2 Vision Wormhole（2602.15382）—— Universal Visual Codec ⭐ 和你的设想最近

这篇我**同时有论文全文和官方源码**，能交叉验证，所以给最细。

**(a) 损失函数**【论文 Eq.(2)，逐字】：

$$
\mathcal{L}_{\text{codec}} = \lambda_h \big\Vert h_{\text{vis}} - \text{stopgrad}(h_{\text{text}}) \big\Vert_2^2
+ \lambda_{kl}\, \tau^2\, \mathrm{KL}\!\left(\text{softmax}\frac{\ell_{\text{text}}}{\tau} \,\Big\Vert\, \text{softmax}\frac{\ell_{\text{vis}}}{\tau}\right)
+ \lambda_{rms}\big(\text{RMS}(\Delta_{\text{inj}}) - \text{RMS}(\bar{X}^{(i)}_{\text{img}})\big)^2
$$

权重【论文附录 B.2 + 源码 `train_vision_latent_mas_codec_new.py:1920-1922` 完全一致】：$\lambda_h=1.0$，$\lambda_{kl}=0.25$（$\tau=1.0$），$\lambda_{rms}=0.1$。

注入方式【论文 Eq.(1)】：
$$X^{(i)}_{\text{img}} = \bar{X}^{(i)}_{\text{img}} + g_i \cdot \text{Resample}(\Delta_i;\, L^{(i)}_{\text{img}}),\qquad g_i \in (0,1)$$

源码对得上（`train_vision_latent_mas_codec_new.py:1607/1634/1659`）：`inputs_embeds[b, pos, :] = base_img + add`——**dummy 图像 embedding 打底，codec 输出叠加上去**。又是残差。

**三项分别在干什么（这是最值钱的部分）**：
- **MSE 项**：表示保真。student 的边界 hidden state 要匹配 teacher 的。
- **KL 项**：输出分布保真。论文附录 C.1 特别说明了为什么要它——"The KL term is especially informative because it provides a rich gradient over the entire vocabulary distribution (not only a single target token)"【论文】。源码支持 top-k 截断（`--vision_codec_kl_topk`）和 logit clip ±80。
- **RMS 项**：**幅度对齐到真实图像 token 的 RMS**。这是"别飞出视觉流形"的最低成本代理。源码 `loss_stats = F.mse_loss(inj_rms, dummy_rms)`（第1712行）。**这一项是这篇最实用的原创**，别的方法都没有——因为别人不往视觉端口写东西。

**(b) 监督信号从哪来（关键）**：

**teacher 和 student 是同一个冻结模型，差别只在输入通道**【论文+源码 `:1482-1506`】：
- teacher prompt：`{"role":"user","content": f"Message:\n{t}\n\nAcknowledge."}` —— 消息 `t` 以**文本**形式明写
- student prompt：**不含** `t`，但带一张 dummy 图，图像 token span 被 codec 输出覆写

所以"平行数据"是**制造**出来的：给定任意一条无标注文本，(文本通道输出, 视觉通道输出) 这一对自动成立。**不需要任何人工标注，也不需要任何跨模型平行隐状态**。论文摘要的原话是 "trained by label-free teacher–student distillation against the text channel, requiring no parallel hidden-state supervision"【论文】。

anchor 语料 = cos_e + OpenCodeReasoning + PRM800K 混合，本地文件 `/home/yilin/tmp/mm-latent-repos/heterogeneous-latent-mas/data/vision_codec_anchor_text/mixed_cose_ocr_prm800k.jsonl`，我数过 **3000 行**。

**(c) 训练成本（全场最低，低到离谱）**【论文附录 B.2 + README + 源码】：
- **400 步，batch size 2 → 总共 800 次 anchor 抽样**。论文自己算了：对 3000 条 anchor 而言只有 **0.27× 的数据覆盖率**（800/3000），连一遍都没过完
- AdamW，lr `2e-4`，grad clip 1.0
- codec 架构：$D=512$，$K_u=1024$ 个 universal token，$K_{img}=256$ 个注入 token，6 层 transformer / 8 头 / dropout 0.10；latent rollout 长度 $T=1024$
- 硬件：**NVIDIA A6000**【论文】
- **弱监督版只用 90 条文本**（30 条 × 3 来源，对应本地 `mixed_cose_ocr_prm800k_small.jsonl`，我数过正好 90 行），论文结论："a weakly supervised codec trained from fewer than 100 anchor texts preserves the runtime profile while producing configuration-dependent accuracy gains"【论文】——注意措辞，runtime 收益保住了，**准确率收益是"看配置"**，没敢说稳赢

**(d) 冻不冻**：VLM 全冻【源码 `:1380-1382`】：`wrapper.model.eval()` + 所有参数 `requires_grad_(False)`；只训 `enc` + `dec`（`:1471` `codec_params = list(enc.parameters()) + list(dec.parameters())`）。

**(e) 不稳定/坍缩**：论文没有专门章节，但**源码里全是防炸的补丁**，这本身就是证据【源码】：
- `:1715-1733` 非有限损失检测 + **跳过该步**（DDP all-reduce 保证所有 rank 一起跳），打印 `Non-finite loss at step {N}`
- latent clip ±50、logit clip ±80、injection clip ±20、`nan_to_num(posinf=1e4)`
- `all_bad_grad_streak` 计数器（连续坏梯度追踪）
- InternVL 有专门开关 `--vision_codec_internvl_disable_kl` 直接**关掉 KL 项**（`:1448-1455`）——【推断】说明某些模型上 KL 项会炸

【推断】这套补丁密度说明这个目标函数在实践中相当脆，尤其 KL 项。

**另外，跨模型对齐用的是闭式 ridge，不是训练**【论文附录 C.2 + 源码 `:2249-2260`】：给定 anchor 文本集，模型 $i$ 与参考模型 $r$ 的 universal token 矩阵之间解 ridge regression 得仿射映射。这和 LatentMAS 的 $W_a=(W_{out}^\top W_{out}+\lambda I)^{-1}W_{out}^\top W_{in}$ 是同一族的闭式最小二乘思路。**合并多个 codec 时是重新 refit ridge，不是端到端重训**。

---

### 1.3 Interlat（2511.09149, ACL'26）—— 目标函数设计得最认真的一篇 ⭐⭐

如果你只读一篇，读这篇。它是唯一**把失效模式写进论文、并为每个失效模式配一个损失项**的。

**(a) 损失函数**【论文】：

第一阶段（actor 学会"读懂" latent）：
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_S \mathcal{L}_{\text{sep}} + \lambda_A \mathcal{L}_{\text{align}}$$

- $\mathcal{L}_{\text{task}}$：监督位置上的标准交叉熵
- $\mathcal{L}_{\text{sep}}$：**加权 Jensen–Shannon 散度**，比较"喂对的 latent"与"喂错配 latent"两个条件分布——**逼模型对 latent 内容敏感**
- $\mathcal{L}_{\text{align}}$：KL + cosine 的双重正则，约束"以 latent 为条件的预测"贴近"以语言 plan 为条件的预测"

第二阶段（压缩器，actor 冻结）：
$$\mathcal{L}_{\text{compress}} = \lambda_{\text{task}}\mathcal{L}_{\text{task}} + \lambda_{\text{pref}}\mathcal{L}_{\text{pref}} + \lambda_{\text{geom}}\mathcal{L}_{\text{geom}}$$
- $\mathcal{L}_{\text{pref}}$：不确定性加权的一致性损失，在"latent 确实降低了熵"的地方匹配压缩前后的行为
- $\mathcal{L}_{\text{geom}}$：逐步平均特征方向的 cosine，保住全局语义朝向

课程【论文】：
$$H^{(r)} = [\underbrace{\text{token embeddings}}_{\text{前 } \lfloor r\cdot L\rfloor \text{ 个位置}}] \oplus [\underbrace{\text{latent states}}_{\text{其余位置}}],\qquad r \sim \mathcal{U}\{0,\ 0.1,\ \dots,\ 1.0\}$$
每步随机抽一个替换率 $r$，从"全 token"平滑滑到"全 latent"。

**(b) 监督来源**：任务标注（Alfworld 用 Song et al. 2024 的数据；MATH 用标准 benchmark）。**不需要平行隐状态**——$\mathcal{L}_{\text{sep}}$ 的"负样本"是**把别的样本的 latent 错配过来**，免费造出来的对比对。

**(c) 成本**【论文】：actor 全量 SFT，lr `1e-5`；reasoning model 全量微调 lr `5e-5`；bfloat16 + FlashAttention-2 + DeepSpeed，batch 16。**GPU 数量和训练时长论文没报**，数据集规模也没报——这是这篇的透明度短板。压缩后推理最高快 24×。

**(d) 冻不冻**：**不冻**。actor 是全量 SFT（无 LoRA）。第二阶段训压缩器时才冻 actor。这是本文与 C2C / Vision Wormhole / MoT / L²-VMAS 最大的分歧点——它是唯一动 backbone 的。

**(e) 不稳定/坍缩（本文最大贡献，直接引用）**【论文】：
- "Learning to interpret latents **from scratch is unstable**"；没有课程学习会出现 "**highly unstable training dynamics**"（Figure 7）
- **去掉 $\mathcal{L}_{\text{sep}}$ → "shortcut behavior; the model learns to ignore the latent communication"**。这就是纯 CE 端到端的那个平凡解，被点名了
- **去掉 $\mathcal{L}_{\text{align}}$ → 模型 "exploit the objective by shifting probability mass toward idiosyncratic tokens"**。即目标函数被 hack：为了压低 $\mathcal{L}_{\text{sep}}$ 而输出怪 token，散度是拉开了，任务效用却掉了

**这两条是整份调研里最重要的工程情报。** 它说明：只要你的目标函数里有"逼模型对消息敏感"这一项，就必须同时有一项"但别用歪门邪道敏感"来配平。单独任何一项都会坏。

---

### 1.4 L²-VMAS（2602.00471, ICML）—— 三阶段 RL

**(a) 损失/奖励**【论文附录 C.5】：**PPO，"the reward signal is primarily tied to the accuracy"**。

**论文没有给出奖励函数的数学形式**，也没有任何 shaping 项、没有 KL-to-ref 之外的正则（`target_kl=0.02` 是 PPO 自带的）。就是**结果奖励**。我把附录 Table 4 的配置抄全：

| | Stage I | Stage II | Stage III |
|---|---|---|---|
| 解锁模块 | $\mathcal{C}$, $\mathcal{C}_{merge}$ | $\mathcal{C}$, $\mathcal{C}_{refine}$, $\sigma$ | 全部 |
| clip_range $\epsilon$ | 0.2 | 0.2 | 0.1 |
| steps | **100k** | **80k** | **50k** |
| learning_rate | `1e-4` | `5e-5` | `2e-5` |
| gae_lambda | 0.95 | 0.95 | 0.98 |

共通：Adam、linear decay、`max_grad_norm=0.5`、`target_kl=0.02`、`gamma=0.995`、`n_steps=2048`、`num_envs=8`、`mini_batch=128/256`。**总计 230k PPO 步**。

三阶段语义【论文】：
- **Stage I**：latent memory **随机触发**（不用 entropy 触发），逼系统频繁读写，"enhancing the robustness of memory encoding"，只解锁压缩器
- **Stage II**：**冻结 Stage I 训好的全部 memory synthesis 组件**，只训 orchestration（门 $\sigma$ + refine 压缩器）
- **Stage III**：全解锁联合训

门控【论文 Eq.(20)】：Gumbel-Sigmoid + 温度退火 $\tau_e = \max(\tau_{\min}, \tau_0\cdot\lambda^e)$，理由是 "mitigating vanishing gradients typically associated with hard binary routing"。

压缩器架构【论文附录 C.1】：Perceiver 式——可学习 query token 序列 $S_{target}$ 拼在内容后面，masked self-attention（Eq.10-12），mask 保证 query 能看内容、内容看不到 query。**没有任何重构损失**，压缩器纯靠 PPO 的 accuracy 奖励学出来。

**(b) 监督来源**：GQA 数据集，"no exposure to the test benchmarks"【论文】。**论文没报数据量**。

**(c) 成本**：**8 × NVIDIA H200 141G**【论文】，230k PPO 步。**时长没报**。这是全场最贵的。

**(d) 冻不冻**：VLM backbone 全冻【论文】"we keep the base VLM backbone of each agent frozen while solely training the custom-designed external memory components"，理由是保住泛化性。

**(e) 不稳定**：论文没有专门讨论，**但 Stage II 的设计动机就是防不稳定**——原文："We freeze all memory synthesis components trained in Stage I **to ensure the distribution of the memory repository remains stable and prevents training instability**"【论文】。这是一条被藏在阶段设计里的稳定性情报：**读写要分开训，否则读的目标在动、写的分布也在动，会互相追**。

---

### 1.5 MACF（2605.00444）—— 最朴素的 curriculum

**(a) 损失**【论文】：三阶段**全是交叉熵**，没有任何对齐/蒸馏项：

$$
\mathcal{L}_{\text{cap}} = \text{CE}(C,\ A_0(c)) \quad\to\quad
\mathcal{L}_{\text{evi}} = \text{CE}(y,\ A_0([c;\text{Emb}(q)])) \quad\to\quad
\mathcal{L}_{\text{col}} = \text{CE}(y,\ A_0([c^{(1)};\dots;c^{(M)};\text{Emb}(q)]))
$$

课程的三级是**输入复杂度递增**（单 agent 描述 → 单 agent 带 query → 多 agent 拼接），损失形式一模一样。所谓 "progressively enforces semantic alignment, evidence summarization, and cross-agent coordination" 其实就是换数据集，不是换目标函数。

**(b) 监督来源**：**要真标注**。Stage 1 用 LLaVA-Video-178K（0-30s 子集）的 caption；Stage 2 用 Video-R1 的 image-QA；Stage 3 用 Video-R1 + Molmo2 子集的 video-QA。这是本调研里**唯一硬依赖大规模人工/合成标注的**。

**(c) 成本**：**4 × A100 80GB**【论文】。数据量和时长**都没报**。

**(d) 冻不冻**：adaptor（**2 层 MLP**）投到共享空间，参数在各 local agent 间 tied。【推断】论文措辞含糊，看起来是 adapter 微调而非全冻，但没有明确的 freeze 声明。compact token 数 $K \in \{16,32,48\}$，主实验 $K=32$。

**(e) 不稳定**：论文没提。

---

### 1.6 MoT / Mixture of Thoughts（2509.21164）—— 唯一把正则消融做全的

虽然你的清单里只写了"MoT 的投影层"，但这篇的目标函数其实是全场**最完整、最可抄**的，而且它是**异构、backbone 全冻、无需 pairwise translator**——和你的设想同构。

**(a) 损失函数**【论文附录 B，逐字】：

$$\mathcal{L}_{\text{LM}} = -\sum_{t=0}^{T-1}\log p_{m^*}(y_t\mid x, y_{<t})$$

$$\mathcal{L}_{\text{ent}} = -\mathbb{E}_x\big[\mathcal{H}(\pi_\tau)\big],\qquad \pi_\tau = \text{softmax}(s/\tau)$$

$$\mathcal{L}_{\text{bal}} = \left(\frac{\text{std}(f)}{\text{mean}(f)}\right)^2,\qquad f_m = \frac{1}{B}\sum_{i=0}^{B-1}\mathbb{1}[m \in \mathcal{I}^{(i)}_{\text{active}}]$$

$$\mathcal{L}_{\text{con}} = \frac{1}{2T}\sum_{t=0}^{T-1}\Big[\mathrm{KL}\big(p^{(0)}_t\Vert p^{(1)}_t\big) + \mathrm{KL}\big(p^{(1)}_t\Vert p^{(0)}_t\big)\Big]$$

$$\boxed{\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \lambda_{\text{ent}}\mathcal{L}_{\text{ent}} + \lambda_{\text{bal}}\mathcal{L}_{\text{bal}} + \lambda_{\text{con}}\mathcal{L}_{\text{con}}}$$

$\lambda_{\text{ent}}=0.01$, $\lambda_{\text{bal}}=0.01$, $\lambda_{\text{con}}=0.05$【论文 Table 13】。

$\mathcal{L}_{\text{con}}$ 的构造很聪明：**同一个输入采两次独立 Gumbel 噪声 $g_0,g_1$，得到两套 top-K 专家选择，逼两套输出分布互相靠拢**——直接惩罚"路由抖一下结果就变"。

**消融数字（全场唯一一个逐项报增益的）**【论文 Table 4】：

| 配置 | 准确率 | 增量 |
|---|---|---|
| 仅 LM | 52.42 | – |
| + Entropy | 54.36 | +1.94 |
| + Balance | 57.29 | +2.93 |
| + Consistency | 59.44 | +2.14 |
| **累计** | | **+7.02** |

**纯 CE 只有 52.42，三个正则加起来贡献了 +7.02。** 这是"端到端 CE 不够用"的最直接量化证据。

**(b) 监督来源**：MMLU/GSM8K/CMMLU/ARC-C/HumanEval 的 train split 并集（有标注 QA）。

**(c) 成本**【论文附录】：**A100 80GB** 训练（推理 RTX A6000 48GB）；AdamW lr `5e-5`，batch 64，**50 epochs**，warmup 1000 步，weight decay 0.01，FP16，max length 512。7 个 7B–8B 专家。**卡数和时长没报。**

**(d) 冻不冻**：**专家 backbone 全冻**，只训 router $\Theta_{\text{router}}$、正/反投影器 $\{(F^\ell_m, R^\ell_m)\}$、逐层共享 attention $\{(W_Q^\ell,W_K^\ell,W_V^\ell,W_O^\ell)\}$。router 的 prompt encoder 用**冻结的 DeBERTaV3-base**。

**(e) 不稳定**：**明确讨论**【论文】——"instability in expert selection can degrade convergence and efficiency"，为此才引入 routing consistency 损失。离散 top-K 用 straight-through + Gumbel 松弛。

---

### 1.7 SDE（2506.19209, EMNLP'25）—— 免训练，但提供了最有价值的**负面对照**

**(a) 损失**：**没有**。完全免训练，无任何可学模块（我把全文 22 页抽出来 grep 过，"train" 只在参考文献里出现）。

**(b)(c)(d)** 不适用。要求**同一模型的多个实例**【论文】"operates with multiple instances of the same model"——所以它连异构都不支持。

**(e) 它的价值在于两条负面结论**【论文】：

**定义**（注意：delta 是**沿 token 轴**的差分，不是沿层轴——这点很多二手总结写错了）：
$$s^l_i = h^l_{A,i} - h^l_{A,i-1}$$

**注入**（当 steering vector 用）：
$$h^{l\,\prime}_{B,j} = \begin{cases} h^l_{B,j} + s^l_i, & \text{位置 } j \text{ 对应 token } t_i \\ h^l_{B,j}, & \text{otherwise}\end{cases}$$

**负面结论 1：注入原始 hidden state 会掉到纯文本以下**【论文 Table 4，EM/准确率】：

| 模型 | | Quasar-T | CWQ | College Math | Formal Logic |
|---|---|---|---|---|---|
| Qwen-7B | NL（纯文本） | 0.3050 | 0.3117 | 0.3617 | 0.4762 |
| | **w/o delta（原始 hidden）** | **0.2950** ↓ | 0.3133 | 0.4033 | **0.4616** ↓ |
| | SDE（delta） | **0.3150** | **0.3167** | **0.4433** | **0.5198** |
| Llama-8B | NL | 0.2850 | 0.3250 | 0.2450 | 0.3889 |
| | **w/o delta** | **0.2750** ↓ | **0.2967** ↓ | 0.2467 | 0.3942 |
| | SDE | **0.3050** | **0.3517** | **0.2967** | **0.4220** |

论文原话："in some cases, the performance of the variant even falls below that of using natural language alone. This indicates that **directly augmenting with unprocessed hidden states may introduce noise, thereby impairing the agent's reasoning**."

**负面结论 2：注入层数不能贪**【论文】。改 top-k 层（$k\le4$）和只改 top-1 差不多，但"**modifying all layers leads to a significant performance drop**"，建议只改 **1–3 层**。

---

### 1.8 DiscoNet（NeurIPS'21, 2111.00643）—— 带宽受限蒸馏

**(a) 损失**【论文，经 ar5iv】：

$$\mathcal{L}_{kd}(\mathbf{H}^s_i, \mathbf{H}^t_i) = \sum_{n=1}^{\bar{K}\times\bar{K}} D_{KL}\Big(\sigma\big((\mathbf{H}^s_i)_n\big) \,\Big\Vert\, \sigma\big((\mathbf{H}^t_i)_n\big)\Big)$$

**逐空间格点**做 KL，softmax 沿**通道维**归一化（把特征图当分布看）。总损失：

$$\mathcal{L}_s = \sum_{i=1}^{M}\Big(\mathcal{L}_{det}(\mathbf{Y}^s_i, \hat{\mathbf{Y}}^s_i) + \lambda_{kd}\mathcal{L}_{kd}(\mathbf{H}^s_i,\mathbf{H}^t_i) + \lambda_{kd}\mathcal{L}_{kd}(\mathbf{M}^s_i,\mathbf{M}^t_i)\Big)$$

在**两处**加约束：post-collaboration 特征图 $\mathbf{H}$ 和 post-decoder 特征图 $\mathbf{M}$。检测损失 = BCE（前背景分类）+ Smooth-$L_1$（框回归）。

矩阵值边权：$\mathbf{W}_{j\to i} = \Pi(\mathbf{F}^s_{j\to i}, \mathbf{F}^s_i) \in \mathbb{R}^{\bar{K}\times\bar{K}}$，$\Pi$ 是 4 层 $1\times1$ 卷积把通道从 $2\bar{C}$ 降到 1，再跨 agent 逐格 softmax。

**(b) 监督来源**：**特权信息 teacher**——teacher 做 early collaboration 用**全局视角**输入，student 做 intermediate collaboration 只有单视角。**这要求你能同时拿到"全知"和"受限"两份输入**，只在 V2X-Sim 这类仿真器里成立。这是**唯一真正意义上的"平行数据"**，而且它是仿真造出来的。

**(c) 成本**【论文】：V2X-Sim 1.0，**10,000 帧**（8,000 train / 900 val / 1,100 test），每场景 2–5 个 agent；**单张 RTX 3090**；$\lambda_{kd} = 10^5$（注意这个量级，说明 KL 项数值极小需要巨大权重拉起来）；每 epoch 从 ~1200s 涨到 ~1500s（KD 开销 +25%）。

**(d)** 端到端训练，无冻结。**(e)** 论文未讨论不稳定。

---

### 1.9 Where2comm（NeurIPS'22）& When2com（CVPR'20）—— "零额外目标函数"路线

这两篇放一起，因为它们给出同一个惊人结论：**"发什么/发给谁/何时发"这套通信策略，可以完全不用任何专门的损失函数学出来。**

**Where2comm (a)**【论文 §4.6，逐字】：
$$\mathcal{L} = \sum_{k=0}^{K}\sum_{i}^{N} \mathcal{L}_{det}\big(\hat{O}^{(k)}_i,\ O_i\big)$$

**就这一项，纯检测损失，跨所有通信轮次求和。没有任何通信损失。**

关键设计【论文】："the functionality of the spatial confidence generator is the same as the classification in the detection decoder. To promote parameter efficiency, our spatial confidence generator **reuses the parameters of the detection decoder**." —— **空间置信图直接复用检测头的分类输出，零新增参数、零新增损失。**

离散选择怎么处理【论文附录 7.3】：**不做可微松弛**，而是拆成双层优化——先固定 $\theta$ 按 top-$b_1$ 排序得到二值选择矩阵 $\mathbf{M}$（不可微，但也不需要梯度），再固定 $\mathbf{M}$ 用标准监督学习优化 $\theta$（Adam）。

课程【论文】：先逐步**升高**带宽和轮数，再**随机采样**带宽/轮数以提鲁棒。一个模型覆盖整条 performance-bandwidth 曲线。

**When2com (a)**【论文，经 ar5iv 2006.00176】：
$$\mathcal{L} = \mathcal{H}(y_j, \tilde{y}_j)$$
**只有任务损失（分割交叉熵）**，原文："only supervision from downstream tasks... **without the need for explicit ground-truth communication labels**"。

离散化：**softmax 注意力天然可微**，$M = \sigma(\cdot)$ 行 softmax；推理时才做阈值剪连接。端到端训练，$\Theta = (\theta_k, \theta_q, \theta_e, \theta_a)$ 全部联合优化，**无冻结**。

**(e)** 两篇都没讨论坍缩。【推断】它们能靠纯任务损失活下来，是因为**任务本身在结构上强制了通信必要性**——单视角看不见的物体，不通信就检测不到，梯度会硬逼着通道打开。LLM/VLM agent 场景**没有这个保证**（接收方自己也能把题做个七七八八），所以纯任务损失更容易退化到"忽略通道"。这正是 Interlat 的 $\mathcal{L}_{\text{sep}}$ 想补的洞。

---

### 1.10 Relative Representations（ICLR'23）—— 免训练锚点对齐（对照组）

**(a)** 无损失函数。用样本对一组固定 anchor 的**相对相似度**（角度）作为表示，"**without any additional training**"【论文】。

**(b)** 代价转移到了别处：**需要一组能被两个空间共同编码的平行 anchor 样本**。这是"平行数据"的一种弱化形式——不需要平行隐状态，但需要平行**输入**。

Vision Wormhole 的 universal-space ridge 对齐是同一族思路的闭式版本。

---

### 1.11 反方证据：Causal Audit（2607.26773）—— 这些目标函数真的训出通信了吗

这篇是对上面所有工作的横向拷问，直接冲着"训练目标是否有效"来的。

**方法**【论文】：在"发送方表示进入接收方"的边界处做**受控消息替换**，四种消息设置：$M_0=\emptyset$（无消息）/ 当前样本的消息 / **别的样本的消息（长度匹配）** / 接收方自产消息（算力匹配）。五个度量：PS（发送方变量可解码性）、PL（有无消息的分布变化）、**CIC**（当前 vs 他样本消息的预测偏移）、**CAG**（样本特定内容带来的任务增益）、SSG（发送方 vs 接收方自产消息的差）。

**结论（数字）**【论文】：
- MATH-500 / Qwen3-4B：总 **+15.00pp** = **+8.33pp 来自"消息存在"** + 6.67pp 来自内容
- MATH-500 / Qwen3-8B：**+8.13pp 的他样本消息效应压过了仅 +1.88pp 的内容贡献**
- GSM8K / Qwen3-4B：总效应近零（−1.00pp），却掩盖了两个相反分量：他样本消息 −6.17pp、内容特定增益 +5.17pp
- 原话："**Aggregate accuracy does not identify how a latent message affects the receiver.**"

审计对象是 **LatentMAS**（KV-cache relay），但框架声称适用于 embedding / hidden state / KV cache 各种载体。

**这对训练目标学的含义**【推断】：如果一大块增益来自"消息存在"而非"消息内容"，那么用**纯任务 CE** 训出来的模块，很可能只是学会了"利用一个额外的、内容无关的计算/注意力偏置"，而不是学会了通信。**Interlat 的 $\mathcal{L}_{\text{sep}}$（匹配 vs 错配 latent 的 JS 散度）在数学形式上恰好就是 CIC 的可微版本**——这是这份调研里我认为最重要的一个连接。

---

## 2. 对比总表

| 方法 | 损失类型（形式） | 监督来源 | 训练成本 | 需要平行数据? | Backbone |
|---|---|---|---|---|---|
| **C2C** (2510.03215) | 纯 next-token CE（+ 一个**被注释掉**的门控正则 `0.0025·gate`） | OpenHermes-2.5 指令-回答对 | 500k 样本 ×1 epoch，有效 batch 256，lr 1e-4，8 进程；**训练卡型未报** | ❌ 否 | 🧊 全冻（sharer+receiver） |
| **Vision Wormhole** (2602.15382) | **MSE(1.0) + KL(0.25, τ=1) + RMS 幅度正则(0.1)** | **自造配对**：同模型文本通道当 teacher，视觉端口当 student | **400 步 × bs2 = 800 抽样**（0.27× 覆盖），lr 2e-4，**A6000**；**弱监督版仅 90 条文本** | ❌ 否（label-free） | 🧊 全冻，只训 enc+dec |
| **Interlat** (2511.09149) | $\mathcal{L}_{task}+\lambda_S\mathcal{L}_{sep}^{\text{(JS)}}+\lambda_A\mathcal{L}_{align}^{\text{(KL+cos)}}$；压缩阶段 $\mathcal{L}_{task}+\lambda_{pref}+\lambda_{geom}$ | 任务标注 + **错配 latent 自造负样本** | bf16+FA2+DeepSpeed，bs16，actor lr 1e-5；**卡数/时长/数据量均未报** | ❌ 否 | 🔥 **不冻**（actor 全量 SFT） |
| **L²-VMAS** (2602.00471) | **PPO，仅 accuracy 结果奖励**（无 shaping，无重构损失） | GQA（量未报） | **230k PPO 步**（100k+80k+50k），**8×H200 141G**；时长未报 | ❌ 否 | 🧊 全冻 VLM |
| **MACF** (2605.00444) | **三阶段全是 CE**：$\mathcal{L}_{cap}\to\mathcal{L}_{evi}\to\mathcal{L}_{col}$ | **真标注**：LLaVA-Video-178K caption + Video-R1 QA | **4×A100 80GB**；数据量/时长未报 | ❌ 否（但要大量标注） | ⚠️ adapter(2层MLP)，冻结情况论文含糊 |
| **MoT** (2509.21164) | $\mathcal{L}_{LM}+0.01\mathcal{L}_{ent}+0.01\mathcal{L}_{bal}+0.05\mathcal{L}_{con}$（**+7.02% 逐项消融**） | 5 个 benchmark train split 并集 | **A100 80GB**，50 epochs，bs64，lr 5e-5；卡数/时长未报 | ❌ 否 | 🧊 全冻 7 专家 + 冻结 DeBERTa router encoder |
| **SDE** (2506.19209) | **无（免训练）** | — | 0 | — | 🧊 不动（但**要求同模型实例**） |
| **DiscoNet** (2111.00643) | $\mathcal{L}_{det}+\lambda_{kd}\mathcal{L}_{kd}^{\text{(逐格 KL)}}$（两处），$\lambda_{kd}=10^5$ | ✅ **特权信息 teacher**（全局视角 vs 单视角） | V2X-Sim 10,000 帧，**单张 3090**，+25% epoch 开销 | ✅ **是**（仿真器造） | 🔥 端到端 |
| **Where2comm** (2209.12836) | $\mathcal{L}=\sum_k\sum_i\mathcal{L}_{det}$ —— **只有任务损失，零通信损失**；置信图**复用检测头参数** | 检测标注 | 4 数据集；离散选择走双层优化不做松弛 | ❌ 否 | 🔥 端到端 |
| **When2com** (CVPR'20) | $\mathcal{L}=\mathcal{H}(y_j,\tilde y_j)$ —— **只有分割 CE**，无通信标签 | 分割标注 | — | ❌ 否 | 🔥 端到端全部联合 |
| **Relative Repr.** (ICLR'23) | **无（免训练）** | — | 0 | ⚠️ 需**平行 anchor 输入**（非平行隐状态） | 🧊 不动 |

---

## 3. 横切规律：五条

**① 平行隐状态数据在所有方法里都被绕过了，只有 DiscoNet 是例外，而它靠仿真器作弊。**
这意味着你的新设想**不会卡在数据上**——但也意味着"我们首创无需平行数据"不是卖点，是入场券。

**② 残差/增量注入已经是共识结构，而且有负面对照撑着。**
C2C `target + gate·scale·Δ`、Vision Wormhole `X̄_img + g·Resample(Δ)`、SDE `h + s_i`。SDE Table 4 是唯一做了严格消融的：**直接注入原始 hidden state 会掉到纯文本 baseline 以下**。别再考虑"替换式"注入了。

**③ 门控设计已经收敛：Gumbel-sigmoid + 温度退火。**
C2C（$1.0\to0.001$ 指数退火）、L²-VMAS（Eq.20 同款）、MoT（Gumbel top-K + straight-through）。三篇独立收敛到同一个方案。这块直接抄，不要花时间。

**④ 纯任务 CE 不够，但只有两篇量化了"不够多少"。**
MoT：纯 LM 52.42 → 加三个正则 59.44（**+7.02**）。Interlat：去掉 $\mathcal{L}_{sep}$ 直接 shortcut 到忽略通道。而 C2C（纯 CE、正则被注释掉）和 MACF（纯 CE）**都没做这个消融**，它们的模块是否真在通信，按 causal audit 的标准是**未经检验**的。

**⑤ 目标函数越"直接"，成本越低。**
按成本排序：Vision Wormhole（800 次抽样，A6000）< DiscoNet（单卡 3090）< C2C（500k×1，8 进程）< MoT（50 epochs A100）< L²-VMAS（230k PPO，8×H200）。**规律很清楚：监督信号越稠密（逐 token hidden state / logits 分布），越便宜；越稀疏（只有最终 accuracy），越贵。** L²-VMAS 用 230k 步 PPO 的稀疏奖励去训一个压缩器，而 Vision Wormhole 用 800 步稠密蒸馏训一个 codec，差了两三个数量级——**这个 gap 就是"目标函数设计"的价值本身**。

---

## 4. 对你新设想（训一个小模型做 VLM agent 间 latent 通信）的直接含义

**地形已被占的部分**（前面已核实，这里只说和目标函数相关的）：
- L²-VMAS 占了「多模态 + latent 通信 + 要训练」，但它的**目标函数是空的**——只有 accuracy 结果奖励，没有任何表示层面的损失。
- Vision Wormhole 占了「视觉端口 + label-free 蒸馏」，目标函数扎实（三项），但**完全没有防坍缩项**，而且训练量小到 0.27× 覆盖。

**没被占的缝隙（我认为是真缝，不是硬凑）**：

1. **把 Interlat 的防坍缩三件套搬到多模态，没人做过。**
   $\mathcal{L}_{sep}$（匹配 vs 错配 latent 的 JS）+ $\mathcal{L}_{align}$（防怪 token）这一对，目前只在纯文本 agent 上验证过。**多模态场景下这个洞只会更大**——因为接收方 VLM 自己也能看图，比纯文本 agent 更容易忽略传来的消息，shortcut 的诱惑更强。这是个可证伪的假设，且实验很好设计（错配 latent 就是 batch 内 shuffle）。

2. **把 causal audit 的 CIC/CAG 从"评测指标"变成"训练目标"，完全无人做。**
   CIC 的定义（当前样本消息 vs 他样本消息的预测偏移）和 $\mathcal{L}_{sep}$ 在数学上几乎是一回事，只差一个可微化。**把审计指标直接变成损失**——这是一个干净的、有出处的、别人还没占的动作。而且它自带一个天然的实验叙事："我们不只是报 accuracy，我们直接优化并报告内容归因增益"。

3. **Vision Wormhole 的 RMS 幅度正则可以推广。**
   $\lambda_{rms}(\text{RMS}(\Delta_{inj}) - \text{RMS}(\bar X_{img}))^2$ 是"留在视觉流形上"的最粗糙代理——只匹配了一阶幅度。【推断】可以升级成分布匹配（如逐通道均值/方差、或对真实图像 token 的 MMD）。这是个小改动但有明确的物理动机，而且**恰好绕开了你原来担心的 $W_a$ 理论断点**：$W_a$ 只对文本词表成立，但"把注入向量约束到视觉 token 的统计分布上"根本不需要词表，也就不需要 $W_a$。

4. **读写分离训练的必要性在多模态下没被验证过。**
   L²-VMAS 用 Stage II 冻结 synthesis 来"prevent training instability"，但它没做消融证明这一步必要。这是个便宜的实验。

**要避开的坑（有实证支撑的）**：
- 别一次改所有层（SDE：改全部层显著掉点，只改 top-1~3）
- 别用替换式注入（SDE：原始 hidden state 掉到文本 baseline 以下）
- 别只报 accuracy（causal audit：+15pp 里可能 8pp 和内容无关）
- 别指望纯 CE（MoT：正则贡献 +7.02；Interlat：无 $\mathcal{L}_{sep}$ 直接 shortcut）
- KL 项会炸（Vision Wormhole 源码给 InternVL 专门加了关 KL 的开关）

---

## 5. 我没查到 / 论文没报的（诚实清单）

- **C2C**：论文**完全没有 Fuser 的数学公式**（只有 Figure 5 示意），我给的公式全部来自源码 `rosetta/model/projector.py:947-1024`。Fuser 参数量论文和 README 都没报。训练用什么卡也没报（只说评测单卡 A100）。
- **Interlat**：GPU 数量、训练时长、数据集规模**全部未报**。$\lambda_S$、$\lambda_A$ 的具体数值我没找到。
- **L²-VMAS**：奖励函数**没有数学形式**，只有一句 "primarily tied to the accuracy"；GQA 用了多少条没报；训练时长没报。
- **MACF**：数据量、训练时长没报；backbone 到底冻不冻论文措辞含糊（只说 adaptor 是 2 层 MLP、参数 tied）。
- **MoT**：卡数（只说 "A100 80GB GPUs"，复数但没数字）、训练时长没报。
- **DiscoNet**：epoch 数没报（只有每 epoch 秒数）。
- **Where2comm**：GPU 型号/数量、epoch 数在我抽取到的正文里没找到。
- **SDE 的 PDF** 我是用 pymupdf 从 ACL Anthology 抽的文本（`/tmp/claude-1008/-home-yilin/e4c133de-4904-4f07-bf05-a54b12262277/scratchpad/sde.txt`），Table 4 数字是从文本流里读的，**建议你复核一眼原表**，因为 PDF 表格抽取偶尔会串行。
- **When2com** 我没能直接拿到 CVPR 官方 PDF（403），用的是 ar5iv 的 arXiv 版（2006.00176）。
- 我**没有**找到任何一篇把 causal audit 的指标当训练目标用的工作——但"没找到"不等于"不存在"，这条缝隙的新颖性建议你再定向检索一次确认。

---

## 附：关键发现速览

1. **没有任何一个方法使用"平行隐状态数据"**——全部靠四种绕法之一：自造配对（Vision Wormhole：同一个冻结模型，teacher走文本通道、student走视觉端口，配对从无标注文本免费制造）、端到端任务CE（C2C/MoT/MACF/Where2comm/When2com，根本不对齐表示）、特权信息教师（DiscoNet：teacher看全局视角，只在仿真器里可得）、RL结果奖励（L²-VMAS：只有accuracy）。这是整个地形最反直觉的一点。

2. **只有 Interlat 显式设计了防坍缩损失，并写出了两个失效模式的名字**：$\mathcal{L}_{total}=\mathcal{L}_{task}+\lambda_S\mathcal{L}_{sep}+\lambda_A\mathcal{L}_{align}$。去掉 $\mathcal{L}_{sep}$（匹配/错配latent的加权JS散度）→ "shortcut behavior, 模型学会直接忽略latent通信"；去掉 $\mathcal{L}_{align}$（KL+cosine）→ 模型"把概率质量堆到怪异token上"来刷目标。纯CE端到端有一个平凡解＝忽略通道，C2C/MoT 都没显式防（C2C源码里 `loss += 0.0025 * gate` 的门控正则被注释掉了）。causal audit (2607.26773) 从实证上确认这风险是真的：MATH-500/Qwen3-4B 的 +15.00pp 里 +8.33pp 来自"消息存在"、只有 +6.67pp 来自内容。

3. **残差/增量注入是跨领域收敛的结构，不是可选项**。C2C 源码 `output_key = target_key + gate*scale*projected_key`；Vision Wormhole Eq.(1) `X_img = X̄_img + g·Resample(Δ)`；SDE $h^l_{B,j}+s^l_i$。SDE 提供了负面对照：注入原始hidden state（"w/o delta"）在4个设置里掉到纯文本baseline以下（Q-7B Quasar-T 0.2950 vs NL 0.3050；L-8B CWQ 0.2967 vs 0.3250）。另外 SDE 还发现改所有层会显著掉点，只能改 top-1~3 层。

4. **门控设计已经统一**：Gumbel-sigmoid + 温度退火。C2C 1.0→0.001（源码指数退火，anneal_steps=1929）、L²-VMAS Eq.(20) 同款、MoT Gumbel top-K + straight-through。这块没有创新空间，直接抄。

5. **成本跨4个数量级，且最便宜的那个恰好是唯一完全不要标注的**。Vision Wormhole：400步、batch=2、共800次anchor抽样、A6000；弱监督版只用 **90条文本**（30条×3来源）仍保住runtime收益。对比 L²-VMAS：PPO 100k+80k+50k=230k步、8×H200、只有accuracy稀疏奖励训一个压缩器。C2C：OpenHermes 500k样本×1epoch、有效batch 256、8卡。
