# Q&A 01：训练 vs 免训练的本质、损失函数设计、多模态 MAS 现状、以及"h=embedding 叠加"新想法

> **日期**：2026-08-03 · **模式**：问答（不跑实验）
> **问题来源**（用户原述）：
> 1. 多模态+LatentMAS 为什么采用训练方案而不是免训练方案？训练方案具体怎么训、损失函数怎么设计？
> 2. 问题的本质：文本模态的 thinking/通信低效；多模态场景文字描述空间关系不精确有歧义 → 两个核心：①latent thinking 中如何把 hidden state 转为有效 embedding；②agent 通信中同构 LLM 是否能直接用 KV
> 3. 现有多模态 MAS / 多模态 LatentMAS 框架什么样？有强开源 baseline 吗？是 4-agent latent 链吗？
> 4. 新想法：h 混合了连续思考（多 token 叠加），LMHead+softmax 做了离散化；对应 embedding 或是多 token embedding 的加权叠加——能否用 LM head 概率找叠加规律？能否设计实验？
>
> **证据基础**：本项目 Phase 1（POPE 诊断）+ Phase 2（A1/A2/A3/B1/B2 正式验证），详见 [phase2-results.md](../VMAS-latent-collab/phase2-results.md)。

---

## 一、为什么用训练 vs 免训练——用我们自己的数据把问题切开

这个问题不能整体回答：**我们的实验证明它的两半答案相反**。

### 本质问题②（同构 LLM 能否直接用 KV）：免训练已解决，不需要训练

A1/B1 三 backbone 交叉验证的结论：

```
盲 agent 只凭 KV 作答:  X−L = +26pp (LLaVA-OV) / +35pp (Qwen3-VL)
打平自己看图的上界:     X ≈ U (三模型复现)
安慰剂排除:            错图 KV 在 MMStar 上把分数拖到随机线以下 (12.5% < 25%)
                       ——A2 诚实读错图被带偏, 通道传真实内容的最强证据
```

**同构模型间 KV 传递：免费、无损、载荷巨大。** 唯一工程前提是 M-RoPE 位置接管（tracker 模块，top-1 100% 验收）——几百行代码，不是训练。

用户本质问题①②的动机（文本低效、空间关系歧义）在我们数据里有**定量版本**：X−T 差距随任务难度与模型强度增长（POPE +2.8pp → GQA/MMStar +10pp → Qwen3-VL +17.8pp）。**"文字瓶颈越来越贵"是测出来的曲线，不是直觉。**

### 本质问题①（h 如何转为有效 embedding）：免训练全部失败——训练的真正战场

| 免训练做法 | 几何行为 | 行为载荷 |
|---|---|---|
| 裸 h + 模长归一（LatentMAS 默认） | 与全词表正交的"外星笔迹"（cos 0.06），rollout 中样本间越写越雷同（区分度近腰斩） | **零**（X≈Xprm，三 backbone × 三任务 × m∈{0..40} 全平） |
| 真 W_a 闭式解 | 三模型三种病理签名（塌质心 / rollout 累积拉力 / 塌向非质心公共方向后混沌发散） | **零到负**（LLaVA-1.5 上 EOS 毒性=mean-init 填充行工程巧合） |

**准确答案：通信（M2）不用训；思考（M1）想有效可能必须训。** 根因：模型从没被训练过"读伪 embedding"——没有梯度信号教过它这些连续向量是有意义的思维。

**训练变得必要的三种情形**：
1. 让 latent 思维携带载荷（M1 复活）
2. 异构模型（KV 形状/语义不兼容 → 学习翻译器，C2C/Vision Wormhole 的地盘）
3. 选择性/压缩传输（选择器与压缩器需要训练信号）

四篇已发表工作（L²-VMAS/MACF/ViF/VW）全部训练，正因为它们的主张都落在这三格里。

---

## 二、训练方案怎么训、损失怎么设计

核心困境：**「(发送方 hidden, 接收方应收到什么)」的平行语料物理上不存在**。所有方案的本质区别不在损失形式，在**监督信号从哪变出来**——四种绕法：

### 路线 A：自造配对（Vision Wormhole）——最便宜

同一个冻结模型，teacher 走文本通道、student 走 latent 通道，配对免费制造：

```
teacher 输入:  "Message:\n{任意文本t}\n\nAcknowledge."     ← t 明文
student 输入:  不含 t, 带 dummy 图, 图像 token 位被 codec 输出覆写

L = λh·‖h_vis − stopgrad(h_text)‖²                        (λh=1.0, 边界hidden对齐)
  + λkl·τ²·KL(softmax(l_text/τ) ∥ softmax(l_vis/τ))       (λkl=0.25, 全词表分布对齐)
  + λrms·(RMS(Δ注入) − RMS(真图token))²                    (λrms=0.1, 幅度锚定视觉流形)
```

实测成本：41M codec、400 步、batch 2、单张 48G 卡。弱点：**教师是天花板**（VW 实测 GSM8K 学生比文本通道低 4.6pp）。

### 路线 B：不对齐、端到端任务 CE（C2C / MoT / MACF）

接收方 next-token CE 一路反传穿过通信模块。C2C 融合器（源码实测形式）：

```
K_out = K_target + g_K · s_K · ΔK        ← 残差注入
        g_K: 逐层 Gumbel-sigmoid 门 (温度 1.0→0.001 指数退火, 推理硬二值)
        s_K: 逐头逐 token 连续缩放
```

弱点：**平凡解 = 门关死退化成单模型**。C2C 源码中被注释掉的门控正则（`loss += 0.0025*gate`）是撞过此问题的痕迹。

### 路线 C：特权教师蒸馏（DiscoNet 血统）

teacher 看完整信息、student 收带宽受限 latent，从上界往下蒸。VLM 场景的天然构造：**split-view**（图裁两半给两 agent，"看完整图的同一模型"即随手可得的教师）——文本 MAS 没有这个便利。风险：先验证上界存在（L²-VMAS 实测多 agent 第 6 轮低于单体——上界不保证存在）。

### 路线 D：结果奖励 RL（L²-VMAS）

只有 accuracy 当奖励，三阶段 PPO 230k 步、8×H200。样本效率最低，且**结构上无法区分"传了内容"和"多了可学习组件"**（其 pipeline 从不构造反事实消息，算不出 CAG）。

### 防坍缩与配方评判（比损失形式更决定成败）

**Interlat 的防坍缩损失**（唯一显式设计的）：

```
L = L_task + λS·L_sep + λA·L_align
    L_sep:   匹配/错配 latent 的加权 JS 散度
             去掉 → "shortcut behavior, 模型学会直接忽略 latent 通道"(原文)
    L_align: KL+cosine; 去掉 → 概率质量堆到怪异 token 刷分
```

**两条硬规矩**（本项目评估协议）：
1. **ΔL 早停**：`ΔL = L(错配消息) − L(正确消息)`，每 N 步一次 forward 零成本；**500 步内不单调上升 → 杀配方**（CAG 的 loss 层免费代理）
2. 终审用 **CAG（X−Xshuf 型对照）不用准确率**——端到端 CE 最易学出"触发器而非信道"，准确率看不出该假阳性

**具体建议**：VLM↔VLM 有文本 MAS 没有的捷径——同图喂两模型，M-RoPE 的 (h,w) 坐标给出**逐 token 天然对应**（一张 256-token 图=256 个平行锚点，8 张图够做 Procrustes 对齐）。起步方案 = split-view 教师（路线 C）+ 残差注入 + Interlat 的 L_sep + ΔL 早停；约 37M Perceiver 模块，单卡可训。

---

## 三、现有多模态 MAS / 多模态 LatentMAS 框架现状

**结论先说：没有一篇是 4-agent latent 链，也没有强的可跑开源 baseline。**（均为本项目读过论文+源码验证）

| 工作 | 拓扑 | 通信介质 | 是 4-agent latent 链吗 | 代码 |
|---|---|---|---|---|
| **L²-VMAS** (ICML'26) | **外挂记忆系统**，套在任意 VMAS 拓扑上（测 6 种结构） | 双 latent 记忆库（hidden 序列+检索键），熵触发主动检索、L=8 伪 token 注入 | ❌ 检索增强，非链式 KV | **无** |
| **ViF** | 4 种 MAS 结构的插件 | ~2% 视觉 relay token 插进下家输入 | ❌ 文字 MAS+视觉补丁 | **stub**（核心 forward 是 `torch.randn`，实测） |
| **MACF** | 局部 agent 并行 → 中央协调者 | K=32 定长通信 token | ❌ 星型 | 无 |
| **Vision Wormhole** | hub-and-spoke | Perceiver codec → 视觉端口注入 | ❌（9 个 benchmark 全纯文本任务） | ✓ 可跑（LatentMAS fork），但不做多模态任务 |

文字系多模态 MAS 有代码的：MDocAgent（5 agent 文档理解）、COMMA（通信质量 benchmark）——均文本通信。

→ **「多模态任务 + LatentMAS 式 4-agent 链式 KV」这个组合，本项目 C1 是（就已核实文献而言）第一个完整实现**：没 baseline 可抄，同时意味着我们的数字就是 baseline。

---

## 四、新想法评估："h = 多 token embedding 叠加"——方向好，有直系亲属，需一处修正，可设计干净实验

### 4.1 直系亲属（该想法已有两代实现）

1. **CIPHER**（ICLR'24，[2310.06272](https://arxiv.org/abs/2310.06272)）：多 agent 辩论中传的消息 = **按 LM head 概率加权的 embedding 期望** `ê = Σ_x p(x)·W_in[x]`，免训练
2. **Soft Thinking**（NeurIPS 2025，[2505.15778](https://arxiv.org/abs/2505.15778)，[项目页](https://soft-thinking.github.io/)）：单模型 latent 推理版——每步不采样、把概率加权 embedding 混合喂回当下一输入；免训练，数学/代码 pass@1 +2.48、token −22.4%。后续：[Soft Concept Mixing (2511.16885)](https://arxiv.org/html/2511.16885)

用户独立推导出了这条线的核心构造——idea 有血统。且据本项目文献盘点，**无人把它用于多模态 MAS 的 latent 思考步**——恰是我们 M1 失败之处。

### 4.2 需修正的一处："h 本身=叠加"在几何上不成立（本项目实测反例）

```
若 h 是 token embeddings 的凸组合:
  模长应 ≲ 1 (embedding 均值 1.08)      实测 ‖h‖ = 122~319
  方向应贴近成分 embedding              实测与全词表最大 cos 仅 0.06
```

正确图景：**h ≠ 叠加本身；h = "LM head 能读出叠加的载体"**——h 里既有对应词表混合的成分（LM head 经内积读出、被 122 倍模长放大），也有词表读不到的大量成分（massive activation、内部簿记）。

- ✗ "h 是哪些 embedding 加权平均构成的"——问法不成立（分解不唯一、大头不在词表典型区）
- ✓ "**把 h 翻译成 `ê = Σ p_x·W_in[x]` 这个流形内代理，是否保留有效载荷**"——可测，且直指 M1 失败机制

小修正：LMHead+softmax 本身**不是**离散化（softmax 输出连续分布）；离散化发生在其后的采样/argmax。该构造恰是"停在 softmax、不做离散化"。

### 4.3 为什么该构造可能治好 M1——与三个失败签名逐条对上

| M1 已知病 | soft-token 反馈 ê 的性质 |
|---|---|
| 裸 h 外星笔迹（cos 0.06，模型没见过） | ê 是真 embedding 凸组合，**天然在输入流形内**，模型读得懂 |
| W_a 回归塌缩（全词表最小二乘 → 公共方向/质心） | ê 权重是**逐样本 softmax 分布**，无回归、无均值收缩 |
| 免训练卖点 | **保持免训练** |

### 4.4 实验设计（三张卡，挂进现有 pipeline；纯问答模式下先记设计，待实验窗口开启再跑）

**E1（行为端，决定性）：X_soft 臂**
latent rollout 反馈从 `v = norm(h)` 换成 `v = Σ_top-k p_x·W_in[x]`（带温度 τ、top-k 截断防长尾噪声），其余不动，挂进盲探针当第八臂。
- 判决：X_soft − Xprm > 2pp 且显著 → **M1 首次携带载荷，免训练修复成立**（推翻"砍 M1"、换更强正面结论）；仍≈0 → "叠加内容无超出图像 KV 的增量"，M1 之死与表示形式无关，结论更硬。**两头都能写。**
- 成本：改十行 + n=800×2 库，数小时。

**E2（几何端，验证"叠加规律"）：稀疏分解 vs LM head 分布**
对每个 h 解稀疏重构 `min ‖h/‖h‖ − Σ_{x∈S} c_x·Ŵ_in[x]‖, |S|=k`，比较最优系数 c 与 `softmax(W_out·h)` 的 top-k：同批 token？权重 Spearman 相关？
- 技术坑：全词表秩满、无约束分解平凡——**必须加稀疏约束**才有意义
- 判决：高相关 → "LM head 概率≈叠加系数"规律成立（E1 构造获得机制解释）；低相关 → ê 仍可能有用（E1 独立判），但"规律"说法收回

**E3（参数面）：τ/k 扫描 + 区分度轨迹**
用 A3 现成管线量 ê 的样本区分度随 rollout 演化——预测应**不再**出现 I 配置的"口水页衰减"。τ∈{0.5,1,2}、k∈{5,20,100}。

**预注册式预判**（记录于看结果之前）：E1 在 GQA/MMStar 上大概率仍≈0——这两个任务的答案信息已被图像 KV 全额携带（Xprm≈U），默想页没有增量空间；**soft-token 反馈真正该显灵的是"多步推理链+答案不直接躺在图里"的任务**（MathVista 型）。E1 若在 GQA 平了别急判死，加测推理重的库再下结论。

---

## 引用

[Soft Thinking (2505.15778)](https://arxiv.org/abs/2505.15778) · [Soft Thinking 项目页](https://soft-thinking.github.io/) · [Soft Concept Mixing (2511.16885)](https://arxiv.org/html/2511.16885) · [CIPHER (2310.06272)](https://arxiv.org/abs/2310.06272) · 本项目数据：[phase2-results.md](../VMAS-latent-collab/phase2-results.md)、[phase1-results.md](../VMAS-latent-collab/phase1-results.md)
