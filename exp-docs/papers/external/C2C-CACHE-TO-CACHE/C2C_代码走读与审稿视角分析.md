# C2C 代码走读与审稿视角分析

> 会话日期：2026-08-04
> 对象：*Cache-to-Cache: Direct Semantic Communication Between Large Language Models*（ICLR 2026）
> 材料：论文 PDF + 本地 clone 的官方实现 [C2C/](C2C/)（commit `113c3a9`）
> 关联文档：[Cache-to-Cache_论文总结.md](Cache-to-Cache_论文总结.md)

本文分两部分：**上半部分**是 Fuser「沿特征维拼接」的实现细节（论文正文只有一句话，细节全在代码里）；**下半部分**是从审稿人视角对其 workflow 设定的分析，以及「两角色优于单角色是否足以证明不是 toy」这一判据的讨论。

---

# 第一部分：Fuser 的「沿特征维拼接」到底怎么做

## 1.1 结论：拼的是通道，不是序列

| | 拼哪一维 | 结果 | 后果 |
|---|---|---|---|
| ❌ 序列维拼接（式 (1)(4) 里的 $\oplus$） | 第 2 维 N | 缓存变长 $N_R+N_S$ | attention 要多看一堆 token，缓存变大 |
| ✅ **特征维拼接**（Fuser 内部） | 最后一维 | **N 不变**，每 token 向量变宽 | 拼完立刻被 Linear 压回原宽度，缓存尺寸不变 |

「沿特征维拼接」= 同一个位置 $i$ 的两个 token 向量首尾接起来，接完是中间量，不进 attention。这也是论文强调 "without increasing cache size" 的原因。

## 1.2 拼接前的形状处理

HuggingFace 每层的 K/V cache 是 `(B, H_kv, N, D_head)`，注意 `H_kv` 是 **GQA 后的 KV 头数**，不是 query 头数。C2C 把头和头维压平：

```
(B, H, N, D) --transpose(1,2)--> (B, N, H, D) --view--> (B, N, H*D)
```

`H*D` 就是所谓的**特征维**。压平这一步关键：Sharer 与 Receiver 的头数、头维通常不同，压平后不需要头与头一一对应，交给 MLP 自己学怎么混。

## 1.3 真实数字（论文主实验配对）

Sharer = Qwen2.5-0.5B-Instruct，Receiver = Qwen3-0.6B：

| | 层数 | KV 头数 H | head_dim D | **特征维 H×D** |
|---|---|---|---|---|
| Sharer Qwen2.5-0.5B | 24 | 2 | 64 | **128** |
| Receiver Qwen3-0.6B | 28 | 8 | 128 | **1024** |

层映射 terminal alignment：Receiver 第 27 层 ↔ Sharer 第 23 层。设 B=1、对齐后 N=170（论文 Table 3 实测平均输入长度）。该层 Key 的完整链路：

| 步骤 | 张量 | 形状 |
|---|---|---|
| ① Sharer K cache | `source_key` | (1, **2**, 170, **64**) |
| ① Receiver K cache | `target_key` | (1, **8**, 170, **128**) |
| ② 压平头 | `source_key_flat` | (1, 170, **128**) |
| ② 压平头 | `target_key_flat` | (1, 170, **1024**) |
| ③ **沿最后一维拼接** | `key_cat` | (1, 170, **1152**) ← 128+1024 |
| ④ 投影层 `key_in` = Linear(1152→1024) | `key_hidden` | (1, 170, **1024**) |
| ⑤ 特征融合层 `key_mlp1`（RMSNorm+FFN+残差） | `key_hidden` | (1, 170, 1024) |
| ⑥a 投影支路 → Linear(1024→1024) → 还原头 | `projected_key` | (1, **8**, 170, **128**) |
| ⑥b 动态加权支路 → Linear(1024→**8**) → sigmoid | `norm_key_scalar` | (1, 8, 170, **1**) 每 token 每头一个 (0,1) 权重 |
| ⑦ 门控（每层每 K/V 一个标量） | `key_gate` | 训练期 Gumbel-sigmoid，推理期 0/1 硬门 |
| ⑧ 残差写回 | `target_key + gate * w * projected_key` | (1, 8, 170, 128) —— **与输入同形** |

对应源码 [projector.py:960-972](C2C/rosetta/model/projector.py#L960-L972)：

```python
source_key_flat = source_key.transpose(1, 2).contiguous().view(B, N, Hs * Ds)   # (1,170,128)
target_key_flat = target_key.transpose(1, 2).contiguous().view(B, N, Ht * Dt)   # (1,170,1024)
key_cat = torch.cat([source_key_flat, target_key_flat], dim=-1)                 # (1,170,1152)
key_hidden = self.key_in(key_cat)                                               # Linear(1152→1024)
```

Value 走**完全独立**的一套参数（`value_in` / `value_mlp1` / …），K 与 V 不共享。

## 1.4 手算级小例子（一个 token）

Sharer 2 个 KV 头、head_dim=2；Receiver 2 个 KV 头、head_dim=3。看第 i=5 个 token 的 Key：

```
Sharer   head0 = [0.7, -0.2]      head1 = [1.1, 0.3]
Receiver head0 = [0.4, 0.9, -0.5] head1 = [-1.2, 0.1, 0.6]
```

压平（head-major，先排完 head0 再排 head1）：

```
s = [0.7, -0.2, 1.1, 0.3]                      # 2×2 = 4
t = [0.4, 0.9, -0.5, -1.2, 0.1, 0.6]           # 2×3 = 6
```

拼接（代码里 source 在前）：

```
key_cat[5] = [0.7, -0.2, 1.1, 0.3 | 0.4, 0.9, -0.5, -1.2, 0.1, 0.6]   # 长度 10
             └──── Sharer 4 维 ───┘└────── Receiver 6 维 ──────┘
```

然后 `key_in` = `Linear(10 → hidden)`。170 个 token 各自独立做同样的事，token 之间不串。

**为什么「先拼再过一个 Linear」是个聪明写法**：把权重按列切成两块 $W=[W_s\,|\,W_t]$，则

$$W\cdot[\,s\,;\,t\,]+b = W_s s + W_t t + b$$

即 **concat + Linear ≡ 两路缓存各自线性变换后相加**，混合系数全部学出来。投影层因此同时完成「把 Sharer 128 维映到 1024 维空间」和「与 Receiver 混合」两件事。

## 1.5 C2C vs C2C-C：差别在拼接时机

| | 拼接输入 | 拼接后宽度（上例） |
|---|---|---|
| **C2C（正文默认）** | Sharer **原始** cache ⊕ Receiver cache | 128 + 1024 = **1152** |
| **C2C-C（附录 A.1.3）** | Sharer 先过 3 层 MLP 升到 Receiver 维度，再 ⊕ | 1024 + 1024 = **2048** |

C2C-C 更强（PGR 34%→79%），但正文为求简单选了前者。

## 1.6 三个隐含前提

1. **N 必须严格相等**，同一下标必须是同一段文字——这正是 token 对齐（解码再重编码 + 最大覆盖选择）存在的唯一理由。拼接本身没有任何对齐能力。
2. **K 是过了 RoPE 之后的**（见 [wrapper.py:229-233](C2C/rosetta/model/wrapper.py#L229-L233) monkeypatch 的 Qwen3 attention），拼进去的是带位置信息的 key。
3. **每个 Receiver 层一个独立 Fuser**（[SFT_train.py:345-356](C2C/script/train/SFT_train.py#L345-L356)，`num_projectors = 28`），不共享。按 recipe 配置（`hidden_dim=1024, intermediate_dim=1024, num_layers=3`）推算：K 侧 8.53M、K+V 17.07M/层、28 层合计 **≈ 4.78 亿可训练参数**——比 Receiver 本身还大。此数字与论文 Table 6 报告的 `C2C 478M` **完全吻合**，可作为推算正确的交叉验证。

---

# 第二部分：审稿视角的分析

## 2.1 代码里的「workflow」到底是什么

所谓 LLM collaboration workflow，本质是 [unified_evaluator.py:1143-1297](C2C/script/evaluation/unified_evaluator.py#L1143-L1297) 里一个 `for` 循环的三个分支：

| 模式 | 代码位置 | 实际做的事 |
|---|---|---|
| **单模型** | `model_type == "hf"` | 普通 `model.generate()` |
| **T2T** | [multi_stage.py:25-363](C2C/rosetta/baseline/multi_stage.py#L25-L363) `TwoStageInference` | Sharer 生成 background → 拼成 3 条 message（user 问背景 / assistant 答 / user 问原题）→ Receiver 回答 |
| **C2C** | [wrapper.py:405-585](C2C/rosetta/model/wrapper.py#L405-L585) `RosettaModel.forward` | 同一段 input_ids 先跑 Receiver 再跑 Sharer，逐层 projector 融合后写回 Receiver 的 cache |
| **Query-level routing** | **仓库里没有实现** | 全仓 grep `rout` 只命中 README 里链接 R2R 的一行；仅附录 A.3.3 文字描述 |

## 2.2 成立的批评（均有代码/论文证据）

### ① T2T baseline 被系统性削弱 —— 最硬的一条

三处叠加：

**(a) T2T 的 Sharer 看不到选项。** [unified_evaluator.py:1146-1167](C2C/script/evaluation/unified_evaluator.py#L1146-L1167)：

```python
# Extract question without options
question_text = example.get('question', '')      # 只有题干
prompt_with_options = prompt                      # 完整题目只给第二阶段
```

而 C2C 的 Sharer 拿到完整 prompt（题干 + 4 选项 + 指令），因为两模型 prefill 同一段 input_ids（[wrapper.py:471-479](C2C/rosetta/model/wrapper.py#L471-L479)：`# Backward compatibility: use same input for all models`）。4 选 1 任务中，看不看得到选项是决定性的。

**(b) T2T 的 Sharer 被 prompt 明令禁止解题。** 论文附录 Text 3 原文：

> *In one clear sentence, describe the most essential background knowledge needed to answer the question: {QUESTION}. **Do NOT directly solve or give answer to the question.***

一句话 + 禁止给答案。而 C2C 传过去的是 Sharer 全部 28 层、全部 token 的内部状态，其中就包含它对四个选项的判断。这不是「文本通道带宽低」，是通道被人为掐住。

**(c) 训练量不对等。** C2C fuser 在 OpenHermes 500k 上训 1 epoch（4.78 亿参数），T2T 是纯 zero-shot prompting，一步没训。

> **可写进 review**：把 T2T 的 Sharer 换成「看到完整题目、允许直接给出候选答案和理由」，重跑 Table 4。若 +3.1~5.4% 优势仍在，结论才立得住。

### ② 两个 agent 输入相同、无分工、单向、单轮

- 输入相同（同上）
- **单向**：Receiver 的状态永不回传，无 feedback
- **只作用于 prompt 段**：response 段 `kv_cache_index = [-1, 0]`（[dataset_adapters.py:493-494](C2C/rosetta/train/dataset_adapters.py#L493-L494)、[unified_evaluator.py:795](C2C/script/evaluation/unified_evaluator.py#L795)），Receiver 一旦开始解码，Sharer 彻底出局

严格讲这里**没有「回合」，谈不上 workflow**，是一次性前向融合。

### ③ 不可组合：每对模型要单独训一个 fuser

HF 上 7 个 checkpoint 全是特定配对。文本通信的核心工程价值是运行时任意插拔，C2C 丢掉了这个性质。**这才是「不是真实工作流」最实质的含义**——不是任务简单，是部署形态不成立。论文附录 A.5.2 承认 $O(N^2)\to O(N)$ 仍是 open problem。

### ④ 主表任务窄

`unified_eval.yaml`：`max_new_tokens: 64`、`do_sample: false`、答案靠字符串匹配。Receiver-only 在 MMLU-Redux 35.53%，随机基线 25%。在刚超随机的 0.6B 模型上加一个 4.78 亿参数、见过 50 万样本的模块然后涨 10 点，有多少是「通信」值得怀疑。

## 2.3 会被作者挡回来的批评

| 批评 | 为什么挡得住 |
|---|---|
| 「只是多了参数/多训了数据」 | Table 6 的 `Single`（[baseline_config.json](C2C/recipe/train_recipe/baseline_config.json) 里 `freeze: []`，同数据全量微调 Receiver）和 `Identical` 两个对照都做了，比多数 ensemble 论文规范 |
| 「只做了选择题 toy」 | 代码里确有 `TwoStageRosetta`（T-C2C）、`multi_source_fusion_mode: parallel/sequential`（多 Sharer）、`include_response.json`（Sharer=Qwen3-32B，融合覆盖 response 段）、LongBench/GSM8K/MATH-500 分支。只是都放在附录 |
| 「延迟测量不公平」 | 代码里 Sharer prefill 是**串行**跑的（[wrapper.py:443](C2C/rosetta/model/wrapper.py#L443) → [wrapper.py:492](C2C/rosetta/model/wrapper.py#L492) 同一 CUDA stream），没有实现论文说的并行 prefill。Table 3 的 2.5× 是串行下测出来的，属于**低报**，批这条等于送分 |

## 2.4 真正致命的点：framing 与实现不符

> **两个模型输入完全相同 ⇒ 不存在信息不对称 ⇒「通信」的必要条件不成立。**

通信要求 sender 拥有 receiver 没有的信息（工具结果、私有上下文、更长窗口、另一模态）。本文中 Sharer 的唯一「优势」是「它是另一个网络」，所以传的不是 *information*，是 *inductive bias*。这个方法的正确名字是 **learned late fusion / cross-model feature stitching**，不是 communication protocol。

一旦如此定位，**缺失的 baseline 全部暴露**（论文一个都没比）：

| 该比而没比的 | 为什么致命 |
|---|---|
| **Logit ensembling** | 最便宜的 late fusion，0 训练成本，MC 任务通常很强 |
| **Hidden-state as soft prefix**（Sharer 表示投影成若干 soft token 拼在 Receiver 前面） | 信息量同级但实现简单得多，且保留序列维、可插拔 |
| **Weight merging / model soup** | 同族同尺寸时的标准做法 |
| **同等训练预算的 T2T**（哪怕只把 Receiver 在带 background 的对话格式上 SFT 一遍） | 直接检验优势是否来自训练而非缓存通道 |

**可执行的判据**（作者本可做而未做）：给 Sharer 一段 Receiver 完全看不到的私有上下文（检索片段 / 长文档 / 图像），看 C2C 能否把它传过去。能传，「通信」成立；不能传，就只是 fusion。LongBench 最接近，但两边看的仍是同一段长文。

---

## 2.5 关键讨论：「两角色优于单角色」能否证明不是 toy？

**结论：不能。** 这篇恰好是最好的反例——「2 > 1」作者已经证明了，但批评依然成立。

### (a) 作者早已做到「2 > 1」

Table 4（Sharer=Qwen2.5-0.5B，Receiver=Qwen3-0.6B，MMLU-Redux）：

| Receiver 单干 | Sharer 单干 | Routing | T2T | **C2C** |
|---|---|---|---|---|
| 35.53 | 38.42 | 35.58 | 41.03 | **42.92** |

C2C 超过了两个单角色的上限，且 Table 6 控住了参数量。按此判据争议本应结束，但审稿人仍会批 toy——**说明判据不对**。

### (b) toy 批的是外部有效性，不是内部有效性

| | 问的问题 | 这篇的表现 |
|---|---|---|
| **内部有效性** | 在这个设定里效果是真的吗？ | 做得不错（有 Single/Identical 对照） |
| **外部有效性** | 这个设定能代表真实场景吗？ | ← **toy 攻击的是这里** |

「2 > 1」只回答第一栏。一个方法完全可以在自己的设定里真实有效，同时那个设定没人会部署——这正是 toy 的定义。

### (c) ⭐ 作者自己的 Table 6 表明「第二个角色」只贡献约 15%

把 Table 6 按增量拆开（基线 = Qwen3-0.6B 单干）：

| 步骤 | OpenBook | ARC-C | MMLU | C-Eval |
|---|---|---|---|---|
| ① Receiver 单干 | 39.20 | 41.04 | 35.53 | 32.04 |
| ② **+ 训练好的融合模块，但 Sharer = 它自己**（Identical, 529M） | 50.60 | 52.52 | 42.17 | 40.34 |
| ③ **+ 把 Sharer 换成另一个模型**（C2C, 478M） | 52.60 | 54.52 | 42.92 | 41.77 |
| **② 的增量（与「第二个模型」无关）** | +11.40 | +11.48 | +6.64 | +8.30 |
| **③ 的增量（「跨模型通信」的净贡献）** | **+2.00** | **+2.00** | **+0.75** | **+1.43** |
| 净贡献占比 | 15% | 15% | 10% | 15% |

**85~90% 的收益来自「多挂了一个 5 亿参数、训过 50 万样本的缓存变换模块」，与对面是谁无关。** 论文自己承认这呼应 latent reasoning / looped transformer——那部分的机制是「多算一遍」，不是「通信」。真正归属于「两个角色」的只有 0.75~2.0 个点。

> 判据在此翻车：「两角色比单角色好」是真的，但「好」的绝大部分不来自「两角色」。

### (d) 「2 > 1」在强 Sharer 下根本不成立

Table 9（Sharer=Qwen3-4B，Receiver=Qwen3-0.6B）：

| | C-Eval | ARC-C | MMLU-Redux | OpenBook |
|---|---|---|---|---|
| Qwen3-4B **单干** | **68.09** | **87.48** | **71.38** | **79.40** |
| C2C（两角色协作） | 44.40 | 60.17 | 45.92 | 55.20 |

协作系统被「直接用强的那个」拉开 20~27 点。论文用的 PGR（29%~41%）**其定义本身就意味着不如强模型单干**（PGR=100% 才追平）。

延迟同理（Table 3）：Receiver 单干 308ms，Sharer 单干 346ms，**C2C 445ms**——协作是三者中最慢的。

所以严格成立的命题只是「**两角色 > 弱的那个角色**」。

### (e) 正确的判据：四关

| | 判据 | 这篇过了吗 |
|---|---|---|
| **① 净贡献可分离** | 换掉 Sharer 带来实质差异（Identical 对照要拉开明显差距） | ❌ 只有 10-15% |
| **② 等预算对比** | 同参数、同延迟下打得过廉价融合（logit ensemble / soft prefix / merging） | ❌ 一个都没比 |
| **③ 超越 max(单角色)** | 而不只是超越 Receiver | ⚠️ 弱 Sharer 成立，强 Sharer 严重不成立 |
| **④ 信息不对称** | Sharer 掌握 Receiver 拿不到的信息（工具/私有上下文/另一模态） | ❌ 两边输入完全相同 |

①②③ 决定「是否真的有效」，**④ 决定「设定是否 toy」**。这篇栽在 ④，而 ① 又暴露了效应量的真实来源。

---

## 2.6 对本项目（多模态 LatentMAS）的启示

四关就是实验清单，两条天然占优、两条必须自己啃：

| 关 | 我们的处境 | 行动 |
|---|---|---|
| **④ 信息不对称** | ✅ 天然过关——VLM 看得到图、LLM 看不到，「通信」这个词用得起。可直接写进 motivation 作为与 C2C 的 differentiator | 在 motivation 里明确对比 C2C 的对称输入 |
| **③ 超越 max(单角色)** | ✅ 相对好过——跨模态时「直接用强的那个」往往不可行（纯文本模型看不了图） | 仍需报告两个单模型的完整数字 |
| **① 净贡献可分离** | ❗ **必答题** | **做完 pipeline 后立刻跑 Identical 对照**（Sharer=Receiver 自己），把「多算一遍的收益」与「跨模型/跨模态的收益」分开报。若净贡献也只有 1 点，方法不成立——别等写论文才发现 |
| **② 等预算对比** | ❗ 逃不掉 | 至少补 logit ensemble、hidden-state soft prefix 两条 baseline。C2C 靠「首创」躲过了，我们不能，因为审稿人手里已有 C2C 这个先例 |

**额外机会**：不可组合性是 C2C 留下的最大空白。谁先做出「训一次、任意模型插拔」的潜空间通信（即附录 A.5.2 那个没做完的 $O(N)$ 方案：$M$ 个投影器把各 Sharer 投到统一潜空间 + $N$ 个融合器分别注入各 Receiver），谁就接管这条线。

---

## 附：本次核查用到的关键源码位置

| 内容 | 位置 |
|---|---|
| Fuser 主体（拼接/投影/动态加权/门控） | [projector.py:862-1024](C2C/rosetta/model/projector.py#L862-L1024) `C2CProjector` |
| 消融变体（对应 Table 8 阶梯） | [ablation_projector.py](C2C/rosetta/model/ablation_projector.py) `ablation_level` 0-4 |
| C2C 运行时（prefill + 逐层融合写回） | [wrapper.py:405-585](C2C/rosetta/model/wrapper.py#L405-L585) |
| 层映射（terminal / depth-normalized） | [model_utils.py](C2C/rosetta/train/model_utils.py) `last_aligned_sources` / `k_nearest_sources` |
| 维度装配（head_dim、num_key_value_heads） | [SFT_train.py:311-356](C2C/script/train/SFT_train.py#L311-L356) |
| T2T baseline | [multi_stage.py:25-363](C2C/rosetta/baseline/multi_stage.py#L25-L363) |
| T-C2C 混合流 | [multi_stage.py:364-800](C2C/rosetta/baseline/multi_stage.py#L364-L800) `TwoStageRosetta` |
| 评测总入口（三分支） | [unified_evaluator.py:1143-1297](C2C/script/evaluation/unified_evaluator.py#L1143-L1297) |
| 训练配方（主实验配对） | [C2C_0.6+0.5.json](C2C/recipe/train_recipe/C2C_0.6+0.5.json) |
| 融合作用范围（instruction 段 only） | [dataset_adapters.py:480-498](C2C/rosetta/train/dataset_adapters.py#L480-L498) |
