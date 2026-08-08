# 跨模型 latent 对齐邻域调研：哪些路线可以少训甚至不训，精度上限在哪

> 调研日期 2026-08-01。每条事实标注 **[论文]**（附 arXiv 号，正文已核）或 **[推断]**。
> 我实际下载并全文读过的 PDF 存在 `/tmp/claude-1008/-home-yilin/e4c133de-4904-4f07-bf05-a54b12262277/scratchpad/`（`relrep / semalign / vec2vec / linalign / kamera / blindmatch / qkvcomm / semeq / bootanchor / stitch` 各一份 `.pdf` + `.txt`）。

---

## 0. 一句话结论

**"把 A 的表示翻译成 B 能读的表示"这件事，最省事的不是相对表示（锚点法），而是直接闭式估一个正交/仿射变换。** 相对表示有个被这条线自己后续工作推翻的隐藏成本：接收方必须先在相对空间里重训一次。真正"零训练 + 高精度"的组合是：**少量平行锚点 → 闭式 Procrustes（或 Latent Functional Maps）→ 直接喂给原生解码器**。完全不要平行数据的方案（vec2vec / QAP 盲匹配）存在，但精度上限只够粗粒度语义，且训练不稳定。

而在你真正关心的 **KV / 逐层张量** 这一层，上述所有方法**一篇都没验证过**——这是空白，也是你的机会窗口。

---

## 1. 核心交付物：训练成本 vs 精度上限阶梯

| 层级 | 路线 | 需要训练吗 | 需要平行数据吗 | 用量 | 精度上限（相对 no-stitch 上界） | 来源 |
|---|---|---|---|---|---|---|
| **T0 完全零训练** | Latent Functional Maps（谱/函数映射） | 否（只解一个凸问题 + kNN 图） | 少量对应点 | **5 个起步，≤50 饱和** | 检索 MRR 0.99（300 锚点）；5 锚点 MRR>0.8 | 2406.14183, NeurIPS'24 |
| **T0 完全零训练** | Procrustes / 仿射闭式解（Semantic Alignment） | 否（SVD 闭式） | 平行锚点 | **≈ d 个（1000–2000）** | 视觉 **97–98% 保真**；文本 **85–91% 保真** | 2311.00664, NeurIPS'23 |
| **T0.5 半零训练** | 相对表示（锚点 + cosine） | **接收端解码器要重训一次** | 平行锚点 | 300–500 | 视觉/跨语言 **85–90% 保真**；跨架构文本 **仅 83%，方差极大** | 2209.15430, ICLR'23 |
| **T1 极轻训练** | 岭回归 / 仿射映射末层 hidden | 是（闭式 + 岭正则） | 平行文本 | **4,000 条** | 强→弱模型 **97% 保真**；弱→强 **崩** | 2603.18908 |
| **T1 极轻训练** | 低秩条件化补丁（rank-32） | **否**（单次前向 SVD） | 无 | 一次前向 | multi-hop 完全恢复到 re-prefill 天花板 | 2606.23581 |
| **T2 中等训练** | Model stitching（stitch layer） | 是（1 epoch 特征匹配 + 任务微调） | 无标签（FFM 阶段） | 任务无关数据即可 | 超过两个源模型各自的 linear probe | 2603.12433, CVPR'26 |
| **T3 重训练** | vec2vec（对抗 + 循环一致） | 是（GAN 式） | **完全不要** | 10K 起步，50K≈1M | 同族 cos 0.92 / top-1 100%；**异族只有 3/15 seed 收敛** | 2505.12540, NeurIPS'25 |
| **T0 但天花板极低** | QAP 盲匹配（Gromov-Wasserstein） | 否 | **完全不要** | 只要两边的类簇 | N=10 类 80–100%；**N=100 急剧衰减**；无监督分类 51.1% | 2503.24129, CVPR'25 |

**读法（我的推断）**：横着看你会发现一个反直觉的事实——**训练量和精度不是单调关系**。T0 的 Procrustes 精度高于 T0.5 的相对表示，也高于 T3 的 vec2vec（在有平行数据的前提下）。真正决定精度的是**你能不能拿到平行数据（Rosetta stone）**，而不是你训了多少。

---

## 2. Relative Representations 这条线（锚点法，免训练分支）

### 2.1 原论文 [2209.15430, ICLR 2023] —— 全文已读

**公式**（论文 §3.1 原文）：
```
r_x(i) = ( sim(e_x(i), e_a(1)), ..., sim(e_x(i), e_a(|A|)) )
```
其中 `sim` 选的是 **cosine similarity**。论文原话："we choose the cosine similarity as the similarity function due to the properties it induces"。

**核心假设**（论文 §3 原文）：两个 latent space 之间的变换 T **保角**（angle-preserving），即 `∠(e_x(i), e_x(j)) = ∠(T e_x(i), T e_x(j))`。cosine 只对保角变换不变——**这就是它的理论边界**。

**锚点怎么选**（论文 §3.1 + Appendix A.2）：
- **平行锚点（parallel anchors）**：两个域之间存在部分对应 `Γ : P_X → P_Y`，取 `A_X ⊆ P_X`，另一边直接取 `Γ(A)`。**这是必须的**——两边锚点必须语义对齐。
- **OOD 锚点**：锚点不必来自训练分布。论文用 WikiMatrix 的平行句当锚点、去做 Amazon Reviews 的分类，仍然 work。
- 选点策略消融（Appendix A.2 Table 7/8）：`uniform`（均匀随机）、`fps`（最远点采样）、`kmeans`（K-means 质心最近词）、`top-k`（跳过前 400 个后的 k 个最高频词）。论文原话："We expect strategies that better cover the absolute space with anchors to be the most effective ones."

**需要多少个锚点**：
- 主实验 **300**（word embedding、Cora 节点分类）；跨语言实验 **500**；Wikipedia 平行锚点 **768**。
- 消融（Figure 6）结论有个关键分叉（论文原文）：**当 embedder 是可训练的**，锚点越多性能单调变好；**当 embedder 冻结**，"increasing the number of anchors does not always improve the performance"，会出现坍缩和不稳定。
- 论文亲口指认的一个失败点：`rexnet-100` 是唯一一个 latent 维度 **高于**锚点数的编码器，"the biggest drop in stitching performance happens when the decoder is trained on it"。→ **经验法则（推断）：锚点数必须 ≥ latent 维度**。

**零样本拼接实测数字**（论文 Table 4 / Table 5）：

跨语言（Amazon Reviews coarse，weighted F1）：
| decoder | encoder | 绝对表示 | 相对(Translated) | 相对(Wikipedia) |
|---|---|---|---|---|
| en | en（匹配） | 91.54 | 90.06 | 90.45 |
| en | es | **43.67** | **82.78** | 78.53 |
| en | fr | 54.41 | 78.49 | 70.41 |
| en | ja | 48.72 | 65.72 | 66.31 |

跨架构（BERT-cased/uncased/ELECTRA/RoBERTa 互换）：
| 数据集 | 绝对 Non-Stitch | 绝对 Stitch | 相对 Non-Stitch | 相对 Stitch |
|---|---|---|---|---|
| TREC | 91.70 | **21.49** | 88.08 | **75.89 ± 5.38** |
| DBpedia | 98.62 | 6.96 | 97.42 | **80.47 ± 21.14** |
| Amazon-Coarse | 87.81 | 49.58 | 85.08 | 72.37 ± 7.32 |
| Amazon-Fine | 55.35 | 19.01 | 48.92 | 33.24 ± 7.21 |

**读这张表的三个要点（推断）**：
1. 绝对表示直接拼 ≈ 随机猜（21.49 / 6.96），证明"直接注入原始 hidden state"确实不行——**这和你已知的 SDE (2506.19209) 负面结论是同一件事的两个证据**。
2. 相对表示把 6.96 拉到 80.47，但**标准差 21.14** —— 这不是一个可以放心上生产的数字。
3. 天花板损失：TREC 从 88.08 掉到 75.89，**约 12 个点**。这就是相对表示的精度上限：**大概保住 83–90%**。

**论文自认的局限**（§6 原文）：cosine 只是一种选择、锚点集组成与表达力的关系"demands additional research"、训练成本受锚点数量和更新频率直接影响、拼接可以扩展到多层（未做）。

### 2.2 隐藏成本：解码器必须重训一次 ⚠️

这是**整条线最容易被漏掉的一点**，由后续论文点破 [2311.00664 §3.1 原文]：

> "in order to perform the stitching procedure in Moschella et al. (2023), **decoders must be trained from scratch at least once** to process samples in this shared relative space."

**这对你的项目意味着什么（推断）**：如果接收方 agent 是一个现成的冻结 VLM，你**没法**直接用相对表示——因为你无法重训它的解码栈。相对表示只在"你自己训接收头"时才是 training-free。

### 2.3 后续工作（把这条线推到哪了）

| 工作 | arXiv / 会议 | 做了什么 | 关键数字 |
|---|---|---|---|
| **Bootstrapping Parallel Anchors** | 2303.00721, ICLR'23 Tiny Paper | 用 Sinkhorn OT 从少量种子锚点推出更多平行锚点 | **15 个种子 → 300 个平行锚点**，"reducing the number of required parallel anchors by one order of magnitude"；15 个 OOD 种子即可跨域零样本拼接。**局限（原文）**："Future research is needed to remove the need for an initial parallel seed." |
| **From Bricks to Bridges** | 2310.01211, ICLR'24 | 不变性的**乘积**（product of invariances），不需要事先知道是哪类变换 | 在图像/文本/图三种模态的 stitching 上取得最佳 |
| **Latent Functional Maps** | 2406.14183, **NeurIPS 2024 主会** | 谱几何 / 函数映射。**完全零训练**（kNN 图 + 凸优化） | **5 个锚点 MRR>0.8**；**≤50 锚点**性能饱和；词向量检索 LFM+Ortho **MRR 0.99**；跨 CNN 层对应识别 **99.8%**（CKA 99.6%）。局限（原文）：对特征向量个数敏感、假设完全对应、纯无监督设定仍待改进 |
| **Latent Space Translation via Inverse Relative Projection** | 2406.15057 | 相对投影 + 其逆，把相对表示打回目标模型的绝对空间 | *（PDF 文本层抽不出，未核到具体数字，标为未核实）* |
| **Multi-Way Representation Alignment** | 2602.06205（2026-07） | **M ≥ 3 个模型**的对齐。广义 Procrustes（GPA）建一个共享 "universe"，复杂度 **O(M²) → O(M)** | 加一个新模型只需 1 个映射而非 (M−1)² 个；GCPA 在 TED-Multi 10 语种 rank-1 **0.503**（GCCA 0.487）、Market-1501 mAP **19.9%**（18.0%）、Flickr8k rank-1 **55.0%**（52.1%）。**需要匹配样本，无免训练变体** |
| **Learned Anchors + Whitened Inner Product (PARAM/WIP)** | 2605.30596（2026-05，preprint） | 指出**原版"随机锚点 + cosine"三个硬伤**：随机锚点线性相关导致维度坍缩、原点附近 cosine 噪声放大、丢掉模长信息 | CIFAR-100 CNN→Transformer F1 **56.67 → 78.64**；GPT2↔TinyLlama R@1 **1.72% → 95.67%**。但**要训练**（4 项损失、95 epoch）；300 锚点接近峰值；平行点超过 5,000–10,000 后收益边际 |
| **Semantic Channel Equalization** | 2411.19719, IEEE | 把相对表示当成**通信协议**：锚点数 = 相对空间维度 = **信道带宽**；提出 K-means "prototypical anchors" | 无需重训；锚点越多精度越高但压缩比越低；**用伪逆把相对表示打回接收端绝对空间**（cosine 无精确逆，只能梯度下降求伪逆；欧氏距离且锚点数 > 隐空间维度时有精确逆） |

**2411.19719 这篇对你特别有用（推断）**：它是唯一一篇把"latent 通信"当作**带宽受限信道**来建模的，直接对应你要的"agent 间通信模块"。它的 `R⁻¹` 伪逆构造（`(A_γᵀ A_γ)⁻¹ A_γᵀ`）在功能上就是 LatentMAS 的 `W_a` 的对应物——**都是"把共享空间的东西打回接收方原生空间"的闭式最小二乘**。

---

## 3. 直接估变换：Procrustes / CCA / 最优传输

### 3.1 Latent Space Translation via Semantic Alignment [2311.00664, NeurIPS 2023] —— 全文已读

**这是"少训不训"路线里我认为最该抄的一篇。**

**做法**（论文 §3.2）：
1. **预处理**：维度不同就 zero-pad 补齐；对每个特征做 standard scaling（零均值单位方差），**统计量只从锚点集算**，以便反标准化。
2. **估变换** `Y = T(X)`，四个逐级收紧的版本：
   - `affine`: `T(x) = RX + b`，SGD 优化
   - `linear`: `b = 0`，**最小二乘闭式**
   - `l-ortho`: 对 linear 得到的 R 做 SVD 强制正交
   - `ortho`: **Procrustes 分析求最优正交 R，闭式**

**要多少锚点**：论文原文 "in a quantity comparable with the dimensionality of the absolute representation"，实测用 **1000**（500 维空间，自编码器实验）、**2000**（N24News 跨模态）。

**关键结论**（论文 §4.1 原文）：`ortho` 和 `affine` 是最好的两个，而 `ortho` 是**简单高效的闭式算法**、`affine` 要 SGD。→ **预训练编码器之间的变换"确实基本是正交的"（mostly orthogonal）**。

**实测数字**（Table 1，跨架构零样本拼接，SVM 线性核解码器）：

| 数据集 | No-Stitch（上界） | absolute（下界） | relative | affine | linear | l-ortho | **ortho** |
|---|---|---|---|---|---|---|---|
| CIFAR10 | 0.95±0.03 | 0.16±0.22 | 0.80±0.22 | 0.92±0.05 | 0.88±0.11 | 0.90±0.09 | **0.93±0.04** |
| CIFAR100-C | 0.85±0.07 | 0.11 | 0.54±0.25 | 0.78 | 0.73 | 0.77 | **0.81±0.07** |
| CIFAR100-F | 0.76±0.09 | 0.07 | 0.30±0.24 | 0.68 | 0.62 | 0.64 | **0.71±0.09** |
| F-MNIST | 0.88±0.01 | 0.15 | 0.63±0.23 | **0.86±0.01** | 0.83 | 0.82 | 0.85±0.02 |
| MNIST | 0.96±0.01 | 0.15 | 0.50±0.22 | **0.94±0.01** | 0.89 | 0.81 | 0.91±0.02 |
| TREC（文本） | 0.87±0.12 | 0.20 | 0.36±0.13 | **0.82±0.12** | 0.74 | 0.57 | 0.79±0.11 |
| AG News | 0.73±0.09 | 0.25 | 0.39 | 0.65 | 0.62 | 0.61 | **0.66±0.10** |
| DBpedia | 0.78±0.23 | 0.07 | 0.16±0.10 | **0.66±0.24** | 0.62 | 0.57 | 0.66±0.22 |
| IMDB | 0.61±0.04 | 0.50 | 0.51 | 0.59 | 0.57 | 0.56 | 0.59 |

**这张表要看三件事（推断）**：
1. **relative 列被 ortho 列全面吊打**（CIFAR100-F: 0.30 vs 0.71）。因为这里解码器**没有**在相对空间重训——公平比较下，直接估变换赢。
2. **视觉侧精度上限极高**：CIFAR-10 0.95→0.93（97.9% 保真），CIFAR100-C 0.85→0.81（95.3%）。
3. **文本侧掉得更多**：TREC 0.87→0.79（91%），DBpedia 0.78→0.66（85%，且 ±0.22）。论文给的解释（§Role of Scaling）：**文本更依赖模长信息**（引 Oyama et al. 2022：encoding 的 scale 与词频相关），所以做 L2 归一化（相对表示的标准做法）在文本上损失更大。

**跨模态（N24News，文本↔图像）**：用 2000 锚点 + ortho + SVM 头，论文原话"This shows the importance of translating from good encoders, that can even **improve** unimodal decoder performances"——**从强编码器翻译过来的表示，喂给弱解码器，反而比它自己的原生输入更好**。

**局限**（§Future works 原文）：最优锚点数量随任务/数据集变化未知；哪些因素决定 latent space 兼容性（如内在维度）未知；锚点集粒度 vs 条件数的权衡未研究。

### 3.2 CCA / GCCA / 广义 Procrustes

[2602.06205] 给了一个干净的判据（论文原文）：
- **严格等距对齐（Procrustes）对 stitching 更好**，因为它精确保住每个模型内部的距离和角度；
- **但对 retrieval 是次优的**，"agreement-maximizing methods like CCA typically prevail"；
- 所以他们提 GCPA = GPA 建骨架 + 事后方向性修正。

**对你的项目（推断）**：你的场景是 stitching（把表示塞进接收方继续算），不是 retrieval，所以**正交/Procrustes 是对的选择，别用 CCA**。

### 3.3 最优传输
- Sinkhorn 用在 [2303.00721] 里发现平行锚点。
- [2505.12540] 把 Earth Mover's / Sinkhorn / Gromov-Wasserstein 当作 oracle-aided 基线（"Naïve baseline"），vec2vec 在多数指标上超过它们。
- [2503.24129] 用 Gromov-Wasserstein 距离作为纯无监督匹配的目标。

---

## 4. LLM 隐空间专项：Characterizing Linear Alignment Across Language Models [2603.18908] —— 全文已读

**这是离你的 LLM 场景最近的一篇实证。**

**做法**（论文 §4）：学一个**仿射映射**（岭正则 λ=10⁻⁴），把源模型 B 的**倒数第二层 hidden state** 映到目标模型 A 的空间，然后直接过 A 的 final norm + LM head 解码。训练数据 **4,000 条**（MMLU 或 Alpaca）。

**MMLU 实测**（Table 3，100 题，贪心解码）：

| 模型 1 | 模型 2 | M1 原生 | M2 原生 | M1→M2 | M2→M1 |
|---|---|---|---|---|---|
| Llama3-8B | Qwen2.5-7B | 58 | 70 | 48 | **68** |
| Gemma3-270M | Llama3-8B | 22 | 58 | 20 | **49** |
| Gemma3-270M | Qwen2.5-7B | 22 | 70 | 21 | **68** |
| Llama3.2-1B | Llama3-8B | 42 | 58 | 36 | **58** |
| Llama3.2-1B | Qwen2.5-7B | 42 | 70 | 28 | **69** |

**三个可直接用的结论**：
1. **强→弱几乎无损**（Qwen 70 → Llama 头解码 68，97% 保真）；**弱→强严重退化**（Gemma-270M 22 → Qwen 头 21，没救回来）。论文原文："source representational capacity, not the target head, is a limiting factor."
2. **跨模型嵌入相似度分层非常明显**：Qwen-7B↔Llama-8B **0.740**、Qwen-14B↔Llama-8B 0.736、Qwen-4B↔Qwen-14B 0.707 —— 而 Gemma-270M↔Qwen-30B 只有 **0.181**。**尺寸相当 → 线性可对齐；尺寸差一个量级 → 对不上**。
3. **tokenizer 兼容性指标与生成质量相关系数 r = 0.822**（论文 §1）。→ 换 tokenizer 是硬伤。

**对你的项目的直接含义（推断）**：如果你的 VLM MAS 里 agent 用**同一个 backbone**（或尺寸相当的同族），4,000 条平行数据 + 一个闭式岭回归就够做末层 hidden 的跨模型翻译。但这只是**末层、每 token 一个向量**——不是逐层 KV。

---

## 5. 完全无平行数据的路线（精度上限最低的一档）

### 5.1 vec2vec [2505.12540, NeurIPS 2025] —— 全文已读

**做法**（论文 §4）：CycleGAN 思路搬到 embedding。用 MLP（残差 + LayerNorm + SiLU）而非 CNN（"embeddings do not have any spatial bias"）。损失 = 对抗 + 重建 + **循环一致（cycle-consistency）** + **向量空间保持（VSP，保 pairwise 距离）**。架构 = 每模型一个 input adapter `A_i` + 一个**共享 translator T**。

**数字**：
- in-distribution：cos **最高 0.92**、top-1 **最高 100%**、rank 低至 1（摘要说 "as high as 0.96"）
- OOD 稳健：在 NQ（Wikipedia）上训，在 TweetTopic（800 条推文，含 emoji）和 MIMIC（医疗）上仍然高相似度
- **数据量**（Table 7）：**10K** embedding 就"好于随机"；**50K 已经接近 1M**
- 消融（Table 6）：VSP 和 CC 各自去掉都显著变差

**致命的稳定性问题**（论文 Appendix，原文）：
> "For the translations between **related** models, vec2vec training was relatively stable across random seeds: **14 out of 15** seeds achieved at least 80% top-1 accuracy... In contrast, translation between **unrelated** models proved significantly less stable, with **only 3 out of 15** runs achieving convergence."

**跨模态**（Table 4 / Table 9）：能翻到 CLIP 空间但"not as strong"；用纯文本训的 granite→clip 翻译器在 MS COCO 上能做**非平凡**的跨模态检索——但论文自己说这只是"promising direction"。

**边界（推断）**：vec2vec 全部是**句级单向量** embedding。它的 VSP 损失保的是"样本之间的 pairwise 距离"，前提是你有一批样本。**KV cache 是一条序列内部逐 token 逐层的张量，没有天然的"一批可比样本"**——VSP 这个关键损失项在 KV 上怎么定义，是开放问题。

### 5.2 It's a (Blind) Match! [2503.24129, CVPR 2025] —— 全文已读

把纯无监督视觉-语言匹配形式化为 **二次分配问题（QAP）**，只用两侧的 pairwise 相似度（Gromov-Wasserstein）。

- **小规模（N=10）**：多数模型好于 10% 随机基线；DINOv2 在 CIFAR-10 上 **80%**、CINIC-10 上 **100%**。论文观察："**pre-training strategy** seems to have a larger impact than the model size"（DINOv2 平均比第二好的预训练策略高 5.3% / 7.6%）。
- **大规模（N→100）**：ImageNet-100 / CIFAR-100 上"performance drops for larger problem sizes"，且有些类别会**拖累**整体匹配。
- **无监督分类器**（K-means 聚类中心 + QAP 匹配语言 embedding）：最好 DINOv2 + All-Roberta-large-v1 **51.1%**（随机 10%）。论文原话："clearly inferior to the supervised models"。
- 论文自问自答："**Can we match arbitrary embeddings? Not with existing models yet.**"

**结论（推断）**：**纯无监督跨模态对齐的天花板 = 粗粒度类别级**。做 agent 间传工作记忆需要的是细粒度、token 级、可组合的信息，这条路目前撑不住。

### 5.3 补充：Do Vision and Language Encoders Represent the World Similarly? [2401.05224, CVPR 2024]
用 CKA + seeded graph matching，发现**训练充分的视觉编码器与语言编码器有出奇高的语义相似度，与是否做过对齐训练无关**；提出 local CKA 度量，在跨域/跨语言 caption 检索上**优于相对表示**。

---

## 6. Model Stitching：Revisiting Model Stitching in the Foundation Model Era [2603.12433, CVPR 2026] —— 全文已读

这篇给了**训练一个 stitch layer 的最佳实践**，而且结论非常反直觉，我认为对你的项目最有指导意义：

**核心发现（论文 §3.2 原文）**：
- **在拼接点做特征匹配（Layer Feature Matching, LFM）会失败**——"low feature matching error at the stitch position" 不等于下游好。
- **必须在目标模型的最终输出层做特征匹配（Final Feature Matching, FFM）**。而且 FFM **不需要任何标签**（原文："this improvement is achieved without using any labels; the stitch layer is trained only to [match final features]"）。
- 直接端到端任务损失（TLT）单独用也很差：fMoW 上只有 **25.1%**。
- **两阶段**：(i) FFM 预训练 stitch layer → (ii) 任务损失微调。

**其他可抄的细节**：
- stitch layer 家族对比（Table 3）：**MLP > Linear > LoRA**（"MLP consistently outperforms Linear and the LoRA option"）。
- **stitch layer 可以任务无关地预训**：用 LLaVA-1.5 数据训的 stitch layer，在 fMoW 上仍然好用（甚至因为 LLaVA-1.5 更多样而更好）。
- 训练成本很低：FFM 阶段"train the stitch layer for one epoch"。
- 结论：DINOv2 ↔ SigLIP2 这类**异构** VFM（不同数据、不同目标、不同模态监督）**确实可拼**，拼接后超过两个源模型各自的 linear probe，且好处不是来自 stitch layer 增加的容量（有 self-stitch 对照）。
- VST（VFM Stitch Tree）：某配置下只用 **4.3% 额外资源**就恢复多 VFM 收益。

**更早的血脉**：Bansal et al., *Revisiting Model Stitching to Compare Neural Representations*, NeurIPS 2021 (2106.07682)——把 stitching 当作表示相似性的度量工具。Lenc & Vedaldi 更早提出浅层线性层可以拼接不同设定下训练的模型。

---

## 7. KV / 逐层这一层：离你最近，也是最空白的一层

### 7.1 KVCOMM [2510.12872, NeurIPS 2025] —— 你点名要挖的，已全文核

**先澄清一个容易混淆的点（重要）**：KVCOMM 的 "anchor" 和相对表示的 "anchor" **是完全不同的两个概念**。KVCOMM 的锚点是**同一个模型、同一段文本、在不同前缀下的 KV 偏差样本**，不是跨模型的语义对应点。

**anchor pool 存什么**（论文原文，每个 placeholder 三样）：
1. base KV-cache：**不带任何外部上下文**独立算出来的；
2. base 与该 agent 上下文中实际 KV 的 **offset**；
3. 相邻后继 prefix segment 的 KV offset。

**offset 怎么算和怎么用**（论文 Eq.6/7）：
```
(k̂/v̂)_φ = (k/v)_φ + Σ_ψ w_{φ→ψ} · Δ(k/v)^φ_{(m,ψ)}
(k̂/v̂)_p = (k/v)_p + Σ_ψ w_{φ→ψ} · Δ(k/v)^p_{(m,ψ)}      # 相邻前缀段，复用同一组权重
w_{φ→ψ} = softmax( −‖h_φ − h_ψ‖ )                        # 基于 embedding 距离的 softmax 权重
```

**位置怎么平移（RoPE 处理，这段最关键）**：
- 问题（论文原文）：同一 token 在不同位置上，raw Key 相差一个正交旋转 `R_Δ`，"whose difference can be **orders of magnitude larger** than the contextual deviation we care about"。
- 解法：比相似度前**先把存的 key 反旋转 `R_{−Δ}`**（de-rotate）。
- 应用时 **K 和 V 处理方式不同**：K 是"先旋到正确位置，再加估计的 offset"；**V 没有位置信息，offset 直接加**。
- 消融铁证：去掉 key rotation 的位置对齐，**MMLU 掉到 43.1%**。

**匹配判据**（Eq.5）：
```
P_anchor(φ) = ( L_φ > max_{ψ∈A} L_ψ )  ∪  ( H_{φ|A} > γ · log|A_φ| )
```
第一项：新内容比现有锚点都长（保证位置对齐正确）；第二项：对候选锚点的距离权重**熵足够低**（说明匹配集中）。默认 **γ = 0.3**。

**池管理**：容量 **V = 20**；满了就在"最早加入的一批"里丢**最少访问**的那个。

**效果**：MMLU 复用率 67.6%（准确率 69.9%）、GSM8K 71.0%（79.6% vs 基线 81.7%）、HumanEval 77.8%（83.2% vs 85.1%）。5 agent 下 TTFT 加速 1.11× / 5.06× / 6.14× / 6.85× / **7.82×**（428.6ms → 55ms）。对比 CacheBlend：固定 80% 复用但 GSM8K 82%→57%、HumanEval 86%→31% 崩掉。

**它自己承认的局限（对你至关重要）**：
- "assumes **every agent runs the same model architecture**... groups of **homogeneous agents**，different weights requiring further exploration"
- "Currently evaluated on **text only**. Extension to image, video, or audio input remains **future work**." ← **这就是你要占的地**
- 只加速 prefill，decode 不受益
- 需要结构化 prompt 里有可识别的 placeholder，"fully dynamic or unstructured multi-agent scenarios... not covered"

### 7.2 DroidSpeak [2411.02820, NSDI 2026]
跨 LLM 复用 KV，但限定"**same architecture**"（微调变体）。传 **E-cache（输入 embedding）+ KV-cache**，接收方**选择性重算少数几层**。吞吐最高 **4×**、prefill 约 **3.1×**，F1/Rouge-L/代码相似度损失可忽略。

### 7.3 Kamera [2606.23581, 2026-06 preprint] —— **我认为对你启发最大的一篇** —— 全文已核

它做的是**多模态、位置无关的 KV 复用**，training-free。虽然是同模型内的复用，但它揭示的**机理**可以直接迁移到你的跨 agent 场景：

**机理发现（论文 §2/§4）**：
1. 盲目复用 KV 丢的是"**跨 chunk 条件化（cross-chunk conditioning）**"。这个损失是**不对称的**：单跳读出（direct readout）被标准 state-merge **精确且免费**地恢复；剩下的残差是**弥散的、低秩的、集中在深层**。
2. 后果："**leaves single-hop recall intact while halving multi-hop accuracy**"。MLA 上 multi-hop 0.41→0.28、GQA 上 0.28→0.15。
3. **这个残差不是稀疏 token**：oracle token 选择器要选约 **50%** 的 token 才够；但在**特征维上是低秩的**——"≈90% of its output-relevant energy in **≈32 directions**"。
4. **深度很重要**：同样的注入，"explains ≈27% of the final deficit applied **shallow** but **≈97% applied deep**"。

**修法**：存一个 position-free 的 canonical `KV(B|∅)` + 一个 **rank-m 条件化补丁**（一次前向算出 Δ，取 top-m SVD 因子 `{U_m, V_m}`，约占 2% 的页面大小）。上线时 = 精确 RoPE 重旋 + 一次 rank-m GEMM。**训练量 = 零**（"supervised by one forward, paid once"）。

**数字**：
- MM-NIAH retrieval：盲重用 0.74→0.38，**rank-16 补丁恢复到 0.72**；multi-hop reasoning-image split：0.59→0.41，**rank-16 恢复到 0.64**
- 成本：**rank-64 达天花板，只占 segment KV 字节的 ≈25%；rank-16 ≈6%**
- 对比 token 轴基线（CacheBlend / VLCache / EPIC / ShadowKV）：特征轴 patch 关掉 **98–100%** 的 reuse→re-prefill KL，token 重算只关掉 **10–71%**；在答案翻转的样本上 patch 恢复正确决策 **96%**，token 基线只有 **21–44%**
- 跨 6 个 backbone（GQA / MLA / MHA / Dense / MoE）都成立；**"The conditioning signal is strongest in redundant vision and video streams"**，音频弱一些，纯文本两 chunk 设定下几乎没有
- 一次前向的成本在 **约 9 次复用**后摊平

### 7.4 Q-KVComm [2512.17914] —— 谨慎对待
声称做"跨架构 KV 翻译"，但实际机制只是**一阶+二阶矩对齐**（论文 Eq.11）：
```
KV_calibrated = (KV_s − μ_s)/σ_s × σ_r + μ_r
```
维度不同时"compute scalar statistics by averaging across all dimensions"。**这只是全局标准化，不是几何对齐**。实验只到 1.1B–1.5B 模型、7 页论文。**我的判断：这不能算跨模型对齐方案，可以当反面参照。**

---

## 8. 视觉侧：不同 VLM 的视觉表征能不能互通

| 证据 | 结论 |
|---|---|
| Blind Match [2503.24129] | 纯无监督下**粗粒度可以**（N=10 类 80–100%），细粒度不行（N=100 崩） |
| Do V&L Encoders Represent World Similarly [2401.05224, CVPR'24] | 训练充分的视觉编码器与语言编码器语义相似度很高，**与是否对齐训练过无关**；local CKA 在跨域检索上**优于相对表示** |
| Revisiting Model Stitching [2603.12433, CVPR'26] | DINOv2 ↔ SigLIP2 **异构 VFM 可靠可拼**，但要训一个 MLP stitch layer（1 epoch FFM 无标签 + 任务微调）；fMoW 上 77.8–92.4% |
| Semantic Alignment 跨模态 [2311.00664] | N24News 上用 2000 锚点 + Procrustes 零样本拼文本编码器 ↔ 图像分类头，**可行**，甚至能提升单模态解码器性能 |
| vec2vec [2505.12540] | 纯文本训的 granite→CLIP 翻译器能做非平凡的跨模态检索，但"not as strong" |
| 多编码器 VLM（DINOv2+SigLIP 拼接） | 工程界普遍做法，但都是**训 projector**，没有免训练版本 |

**我明确找不到的**：**一篇零训练地把 VLM-A 的视觉 token 翻译进 VLM-B 视觉端口的工作**。你提到的 Vision Wormhole (2602.15382) 是训练路线（每个模型族一个解码器 + 师生蒸馏）。→ **这块地是空的**（我的判断，基于 16 次不同措辞检索未命中）。

---

## 9. 更早的血脉：V2X 协同感知（你已知，我补两条对齐视角）

你列的 When2com / Where2comm / DiscoNet 解决的是"**何时发 / 发哪块 / 怎么压**"，但它们**都假设各 agent 用同一个 backbone**——它们从来不需要解决"跨模型对齐"。而 Relative Representations (ICLR'23) 造 "latent communication" 这个词、给的是**免训练对齐**。

**我的判断**：这两条血脉在文献上**几乎没有交叉**。"用 V2X 的通信策略（何时发/发哪块）× 用表示对齐社区的跨模型翻译（怎么让对面读懂）"这个组合，是一个真实的空白。而 [2411.19719] 的语义信道均衡是目前唯一把两边接起来的尝试（且是在无线通信社区，不在 ML 社区）。

---

## 10. 对你项目的直接判断

### 10.1 哪些能不训

**可以完全不训的（推荐优先试）**：
1. **闭式 Procrustes / 岭回归求跨端口映射**。LatentMAS 的 `W_a = (W_out^T W_out + λI)^{-1} W_out^T W_in` **本质上就是一个带岭正则的最小二乘闭式解**（我的推断）——它属于 [2311.00664] 的 `linear` 那一档。**把它推广到视觉端口，需要的不是新理论，而是一组"视觉侧的平行锚点"**。
2. **视觉侧平行锚点从哪来（免费的 Rosetta stone）**：任何 caption 数据集里，`(图像, 描述该图的文本)` 天然就是一对平行锚点——图像走视觉端口、文本走 `W_in`，两边在同一个 latent 位置上表达同一语义。这就绕过了"视觉输入不过 `W_in`"的理论断点（**我的推断，未见论文这么做**）。
3. **锚点数量预算**：按 [2311.00664] 的经验取 ≈ hidden dim（4096 级别）；按 [2406.14183] 的 LFM 可以压到 **≤50**。建议先跑 LFM 那条，因为它零训练且锚点需求低一个数量级。

**可以近乎不训的**：
4. **Kamera 式 rank-32 低秩补丁**，由一次前向 SVD 得到，不训练。[2606.23581] 已经证明这在**六个 VLM backbone** 上都成立，且"vision and video show the gap and recover"——**视觉流的条件化信号最强**，正好是你的场景。

### 10.2 精度上限（照抄上面的表）

| 你的用法 | 现有证据给出的上限 |
|---|---|
| 同族/同尺寸 LLM，末层 hidden 翻译 | **≈97% 原生保真**（Qwen 70 → Llama 头 68）[2603.18908] |
| 跨架构视觉编码器，单向量层面 | **≈95–98% 保真**（CIFAR 0.95→0.93）[2311.00664] |
| 跨架构文本编码器 | **≈85–91% 保真**，且方差大（±0.22）[2311.00664] |
| 尺寸差一个量级 | **崩**（Gemma-270M → Qwen 头，22→21）[2603.18908] |
| 完全无平行数据 | **粗粒度类别级**，51.1% 无监督分类 [2503.24129] |
| KV 盲复用不修 | **单跳无损、多跳腰斩**（0.59→0.41）[2606.23581] |
| KV 复用 + rank-16 补丁 | **恢复到 re-prefill 天花板**（0.64）[2606.23581] |

### 10.3 三个必须写进方案的风险

1. **⚠️ 所有对齐证据都是"每样本一个向量"，没有一篇验证过 KV 张量**。relative rep / Procrustes / vec2vec 全部作用在句级或图级 embedding 上。KV 是 `[layer × head × token × dim]`，而且带 RoPE 位置耦合。**"保角变换假设"在 KV 上是否成立，是完全未验证的**。这既是最大风险也是最大的可发论文的点（我的推断）。**建议先做一个便宜的验证实验**：拿两个同族 VLM，取同一段文本在两边的某层 K/V，用 1000 个平行 token 位置估一个 Procrustes，量残差与 cosine——一天能跑完，直接决定整个方案生死。

2. **⚠️ 监督信号该打在哪里，已有明确答案且反直觉**。[2603.12433] 实测：**在注入点做特征匹配会失败，必须在接收方最终输出层匹配**。Kamera 独立佐证：同样的修正"applied shallow 只解释 27%，applied deep 解释 97%"。→ 如果你要训一个通信模块，**损失函数应该打在接收方 agent 的末层特征 / 输出分布上，而不是"让我的 KV 长得像你的 KV"**。这一条能省你至少一轮失败实验。

3. **⚠️ 相对表示不能直接用在冻结的接收方 VLM 上**（要重训解码器）。如果你想保留"training-free"卖点，走 Procrustes/LFM，别走相对表示。

### 10.4 我看到的真实空白（按可行性排序）

| 空白 | 谁最近 | 为什么还空着 |
|---|---|---|
| **跨模型 KV 的锚点/Procrustes 对齐** | KVCOMM（同模型跨前缀）、DroidSpeak（同架构） | KVCOMM 明写 "homogeneous agents / text only"；相对表示线从没碰过 KV |
| **VLM 视觉端口的零训练互译** | Vision Wormhole（训练）、Blind Match（无监督但粗粒度） | 没人试过用 caption 数据当视觉侧平行锚点 |
| **M≥3 agent 的共享 latent universe** | Multi-Way Representation Alignment [2602.06205]（O(M) GPA） | 这篇是 2026 年的，还没人把它接到 MAS 上 |
| **通信策略 × 跨模型对齐的合流** | V2X 那条线（同 backbone）、语义信道均衡 [2411.19719]（无线社区） | 两个社区不互引 |

---

## 11. 我明确找不到 / 不确定的

- **Latent Space Translation via Inverse Relative Projection (2406.15057)** 的具体数字——PDF 文本层抽不出，只确认了存在和标题。
- **Revisiting Model Stitching** 训练 stitch layer 到底用多少数据——论文只说 FFM 阶段 "one epoch"、linear probing 阶段最多 100 epoch + early stopping，**没给样本量**。
- **Q-KVComm (2512.17914)** 的可信度我持保留——7 页、1.1B–1.5B 模型、"跨架构翻译"实为矩对齐。
- **没有找到**任何一篇把 relative representations / Procrustes 用在 transformer **逐层 KV** 上的工作（16 次不同措辞检索）。如果它存在，我没找到。
- **Kamera (2606.23581)** 是 2026-06 的 preprint，under review，未经同行评议。

---

## 参考文献（arXiv 号已逐条核对）

**锚点 / 相对表示线**
- Relative representations enable zero-shot latent space communication — [2209.15430](https://arxiv.org/abs/2209.15430), ICLR 2023
- Bootstrapping Parallel Anchors for Relative Representations — [2303.00721](https://arxiv.org/abs/2303.00721), ICLR 2023 Tiny Paper
- From Bricks to Bridges: Product of Invariances — [2310.01211](https://arxiv.org/abs/2310.01211), ICLR 2024
- Latent Space Translation via Semantic Alignment — [2311.00664](https://arxiv.org/abs/2311.00664), NeurIPS 2023
- Latent Functional Maps — [2406.14183](https://arxiv.org/abs/2406.14183), NeurIPS 2024
- Latent Space Translation via Inverse Relative Projection — [2406.15057](https://arxiv.org/abs/2406.15057)
- Latent Communication in Artificial Neural Networks（综述/学位论文）— [2406.11014](https://arxiv.org/abs/2406.11014)
- Relative Representations enable Efficient Semantic Channel Equalization — [2411.19719](https://arxiv.org/abs/2411.19719), IEEE
- Multi-Way Representation Alignment — [2602.06205](https://arxiv.org/abs/2602.06205)
- Improving Relative Representations with Learned Anchors and Whitened Inner Products — [2605.30596](https://arxiv.org/abs/2605.30596)
- Learning Relative Representations for Fine-Grained Multimodal Alignment with Limited Data — [2605.16834](https://arxiv.org/abs/2605.16834)

**Stitching / 线性对齐**
- Revisiting Model Stitching to Compare Neural Representations — [2106.07682](https://arxiv.org/abs/2106.07682), NeurIPS 2021
- Revisiting Model Stitching In the Foundation Model Era — [2603.12433](https://arxiv.org/abs/2603.12433), CVPR 2026
- Characterizing Linear Alignment Across Language Models — [2603.18908](https://arxiv.org/abs/2603.18908)

**无平行数据**
- Harnessing the Universal Geometry of Embeddings (vec2vec) — [2505.12540](https://arxiv.org/abs/2505.12540), NeurIPS 2025
- It's a (Blind) Match! Vision-Language Correspondence without Parallel Data — [2503.24129](https://arxiv.org/abs/2503.24129), CVPR 2025
- Do Vision and Language Encoders Represent the World Similarly? — [2401.05224](https://arxiv.org/abs/2401.05224), CVPR 2024

**KV 层面**
- KVCOMM — [2510.12872](https://arxiv.org/abs/2510.12872), NeurIPS 2025
- DroidSpeak — [2411.02820](https://arxiv.org/abs/2411.02820), NSDI 2026
- Kamera: Unified Position-Invariant Multimodal KV Cache — [2606.23581](https://arxiv.org/abs/2606.23581)
- Q-KVComm — [2512.17914](https://arxiv.org/abs/2512.17914)（可信度存疑）

---

## 附：关键发现速览

1. **免训练路线里，"直接估变换"（Procrustes/仿射闭式解）比"相对表示（锚点+cosine）"更强，且是唯一真正零训练的**。相对表示有个被广泛忽略的坑：接收方解码器**必须先在相对空间里训练一次**（论文原文 2209.15430 §3.1 + 2311.00664 §3.1 明确指出）。而 Latent Space Translation via Semantic Alignment（2311.00664, NeurIPS'23）用 ~1000–2000 个平行锚点 + 闭式 Procrustes 就能跳过重训解码器，且效果反超相对表示：CIFAR-10 上 no-stitch 0.95 / 绝对基线 0.16 / 相对表示 0.80 / ortho **0.93**。文本侧掉得更多（DBpedia 0.78→0.66±0.22）。

2. **锚点需求量的实测阶梯**：Latent Functional Maps（2406.14183, NeurIPS'24，全零训练）**5 个锚点** MRR>0.8、**≤50 个**基本饱和，是目前锚点效率的最优点；Procrustes 需要 ~d 个（1000–2000）；相对表示原文用 300–500；Bootstrapping Parallel Anchors（2303.00721）用 Sinkhorn OT 从 **15 个种子锚点**推出 300 个，把平行数据需求降一个数量级——但**仍需要一个初始种子，无法完全去掉**。

3. **完全无平行数据的方案存在但精度上限很低且不稳定**。vec2vec（2505.12540, NeurIPS'25，CycleGAN 式对抗+循环一致+VSP）cos 可达 0.92、top-1 可达 100%，10K 条 embedding 就能起步、50K≈1M；但**同族模型 15 个种子里 14 个收敛，异族模型只有 3/15 收敛**。纯无监督跨模态（Blind Match, 2503.24129, CVPR'25，QAP+Gromov-Wasserstein）在 N=10 类上 DINOv2 能到 80–100%，**类数升到 100 就急剧衰减**，无监督分类器只有 51.1%（随机 10%）。结论：**完全免平行数据只能对齐粗粒度语义，不足以撑 agent 间的细粒度工作记忆传递**。

4. **KV/逐层这一层，跨模型对齐是空白**。KVCOMM（2510.12872）的"anchor"跟相对表示的锚点是**完全不同的东西**（同模型、跨前缀的 offset 池，V=20，softmax(−‖h_φ−h_ψ‖) 加权，K 要先 de-rotate 再加偏移；去掉位置对齐 MMLU 掉到 43.1%），它明确写死"同架构同权重"。DroidSpeak（2411.02820, NSDI'26）只跨同架构微调变体。所有 relative-rep/Procrustes/vec2vec 的证据**全部建立在"每条样本一个向量"的 embedding 上，没有任何一篇验证过 per-token×per-layer×per-head 的 KV 张量也满足同样的几何**——这是最大的未验证假设，也是最大的机会。

5. **Kamera（2606.23581）给了"少训"路线一个极强的先验**：KV 重用丢掉的那部分（跨 chunk 条件化）在特征维上是**低秩的（rank≈32 就够，90% 输出相关能量在 ~11% 方向里）且集中在深层**；一个**免训练**、由单次前向 SVD 得到的 rank-m 补丁就能修复，rank-16 把 MM-NIAH multi-hop 从 0.41 拉回 0.64（盲重用 0.59→0.41），rank-64 达天花板且只占 ~25% KV 字节。配合 Revisiting Model Stitching（2603.12433, CVPR'26）的发现——**在拼接点做特征匹配会失败，必须在接收方"最终输出层"做匹配（Final Feature Matching，无需标签）**——可以推出：若真要训，训一个 per-layer rank-32 的低秩补丁 + 在接收方末层监督，规模比 C2C 的 Cache Fuser 小两个数量级。
