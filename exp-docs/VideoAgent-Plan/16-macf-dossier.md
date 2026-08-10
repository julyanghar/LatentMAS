# 16 · MACF 精读档案（arXiv 2605.00444，对照面与开口清单）

> 2026-08-10 精读工作流产物存档。用途：HourVideo 战役消融设计的对照面依据 + 论文 related work 弹药。
> 一句话：MACF=「训练出来的 latent 信道 + 每 agent 感知 token 等预算」；无关键帧臂、无比特级预算对齐、
> 无因果审计、无 HourVideo、信道完全依赖三阶段训练、v1 无附录无代码不可复现——我们清单上的每个开口都开着。

ary": "Full-text read of MACF (arXiv 2605.00444) for ablation-design decisions",
  "agentCount": 1,
  "logs": [],
  "result": {
    "report": "# MACF 精读回报（arXiv 2605.00444v1，2026-05-01 提交，ICML 2026 投稿格式，12 页）

标题：Scaling Video Understanding via Compact Latent Multi-Agent Collaboration。作者：Kerui Chen, Jinglu Wang, Jianrong Zhang, Ming Li, Yan Lu, Hehe Fan（微软亚研+浙大背景组合，Jinglu Wang/Yan Lu 是 MSRA）。

**先报三个读全文才能发现的硬信息**：
- **arXiv v1 没有附录**。正文两处写 "More details are provided in the appendix"（§4.1）和 "The implementation details of these three baselines are provided in the Appendix"（§4.1 Baselines），但 PDF 全 12 页 = 正文 9 页 + 参考文献 3 页，附录不存在。训练超参、baseline 实现细节全部无从查证。
- **图文不一致**：§4.3 正文说 agent 数消融 "We study this effect on MLVU-Test"，但 Figure 4 图题是 LongVideoBench，且 M=6 点 ≈56.8 恰好等于 Table 1 里 LongVideoBench 的 56.8——图实际是 LongVideoBench，正文写错。
- **增益数字对不上**：§4.2 声称对 text 通信赢 "20.3%, 18.6%, 9.7%, 15.9%"，后三个 = 与 Table 1 MapReduce 的**绝对分差**（56.8−38.2=18.6 等），但第一个 Video-MME 60.4−46.7=13.7 ≠ 20.3；对 LatentMAS 声称 "9.3%, 14.1%, 10.5%, 10.3%"，与 Table 1 的绝对差（4.7/8.3/7.0/6.9）和相对差（8.4%/17.1%/21.1%/16.3%）**都对不上**，无法复算。对 Qwen3-VL-8B 的 "+4.5/+6.1/+7.0/+7.7%" 则精确 = 绝对分差（该文把绝对点数写成 %）。

---

## 1. 有没有关键帧/图像转发臂？

**没有。** 通信协议对照只有两条臂（§4.1 Baselines、Table 1 底块、§4.2 "Effective communication representation"）：
1. **MapReduce**（Pang & Wang 2025 "Mr. Video"）= 文字通信，把原实现的闭源模型换成 Qwen3-VL-8B；
2. **LatentMAS**（Zou et al. 2025, arXiv:2511.20639）= KV-cache 直传，"adapt and modify the official LatentMAS code to support multimodal understanding"。

即：对照 = latent tokens vs 文字 caption vs KV-cache，**不存在任何"local agent 把关键帧/图像/visual tokens 转发给 coordinator"的臂**。关键帧检索只出现在 Figure 1(b) 作为 motivation 里被批评的范式（"高成本、依赖中间文字质量"），从未进实验表。coordinator 在所有臂里都看不到任何像素。也没有 retrieval 系（VideoAgent/Video-RAG 等）进对照表。

**预算匹配**：Table 1 三个 multi-agent 系统的 Perc. budget 列都标 16\*224\*224（每 agent），引擎统一 Qwen3-VL-8B——感知侧名义上匹配（但 baseline 的 agent 数、分段方式在缺失的附录里，未核实）。**通信侧不匹配**（见问题 2）。

## 2. 等预算问题（对我们最要紧）

- **文字臂 caption 是多少 token**：正文没写每 agent caption 长度，唯一数据是 Table 3：MapReduce 通信吞吐 **387 tokens**（应为总量，≈64 token/agent×6），MACF **192 tokens**（=6 agent×K32），LatentMAS **784 KV-Cache**（单位含糊，未说明层数×维度）。
- **总信息量可比吗**：不可比，且方向微妙——按 token **数**算文字臂反而多（387 vs 192），论文借此把 latent 说成"吞吐最低最便宜"（Table 3 + §4.3 Communication cost）；但按**比特**算，192 个 d 维连续向量（d 未给，Qwen3-VL-8B 隐层 4096，bf16 下 ≈1.5MB）碾压 387 个离散 token（≈0.8KB），差 3 个数量级。**论文全篇用 token 计数当通信预算（Eq 2/6），从未讨论、从未承认连续 token vs 离散 token 的信息容量不对等。** 这是它等预算叙事的最大软肋，也是我们等比特消融的直接开口。
- **另一个被"identical budget"话术掩盖的点**：B^per 定义是**每 agent 每次前向**的像素预算（§3.1 Eq 1）。Table 1 上块的单模型 baseline 总共只吃 16×224² ≈0.8M 像素，MACF 推理时 6 agent 各吃满 = **总像素 6 倍**。对 MapReduce/LatentMAS 是公平的（同样多 agent），对单 MLLM 行的"identical budget constraints"（摘要、§5 原话）实为 per-forward-pass 口径，不是系统总预算。
- **还有一个训练混淆**：MapReduce/LatentMAS 是 training-free pipeline，MACF 是在 Video-R1+Molmo2 上 LoRA 微调过的——通信介质对照与"有无域内训练"完全耦合，论文未承认。

## 3. 训练细节（§3.3 + §4.1）

三阶段 curriculum（Figure 2 b/c/a）：
- **Stage 1 语义对齐**：单 agent 出 comm tokens → coordinator 解码 caption，CE loss（Eq 8）。数据 = LLaVA-Video-178K 的 **0–30s 子集** caption。
- **Stage 2 证据摘要**：Image-QA SFT，c=A_m(X,q)（query 条件化），coordinator 拿 [c; Emb(q)] 出答案（Eq 9）。数据 = **Video-R1 的图像部分**。
- **Stage 3 跨 agent 协作**：Video-QA，M=4 个 agent，loss Eq 10。数据 = **Video-R1 视频部分 + Molmo2 子集**。
- 硬件：4×A100 80GB。**GPU 时、样本量、epoch、lr、LoRA rank 全部没给**（推给不存在的附录）。可训参数（Figure 2 图例）：局部 agent LoRA + coordinator LoRA + 可学习 communication token（query 向量）+ adapter（2 层 MLP）。局部 agents 参数共享（§3.3）。
- **没有 training-free 消融**。latent 信道不训练能不能用没直接测，但 Table 4 给了强旁证：去掉 Stage 3（只训对齐+摘要）Video-MME 48.2 / MLVU 36.5，**低于 Qwen3-VL-8B 单模型基线 55.9/41.5**；去 Stage 1 更惨（45.5/36.3）。三阶段平均掉 13.9/3.0/12.5 点。结论：该信道**完全依赖训练**，zero-shot 不可用——这正好反衬 training-free latent 转发（LatentMAS 路线）与它的差异定位。

## 4. 信道实现（§3.2、§4.3 Table 5 讨论、Figure 2）

- **消息形状**：c^(m) ∈ R^{K×d}（Eq 4），K = 每 agent token 数，主设置 **K=32**（消融 16/32/48，Table 2），**d 数值全文未给**（只说 "embedding dimension"，即 coordinator 输入嵌入维）。
- **提取方式**：局部 agent 输入 = query q + 视觉输入 X^(m) + K 个**可学习 communication token**（拼在序列尾，Figure 2a）；取这 K 个位置的**最后一层 hidden states**（Figure 2 图题原话 "extracted from the last-layer hidden states"），过 **2 层 MLP adapter** 投影进共享 latent 空间（§4.3："the adapter module (2-layer MLP), which projects any backbone's hidden states into a shared latent space with fixed dimensionality"，这也是异构 backbone 能混用的机制）。
- **coordinator 消费方式**：输入序列 = [c^(1); delimiter; c^(2); …; c^(M); Emb(q)]（Eq 10 + Figure 2 图例明确有 "Delimiter token"），latent 块以 inputs_embeds 形式直接进 coordinator，后接问题嵌入。**prompt 模板未给**。星形拓扑、单轮、单向（local→coordinator），无回传无多轮。
- **M-RoPE/位置编码：零讨论**。全文没有 RoPE/M-RoPE/position 任何字样（我 grep 过 PDF 全文）。Qwen3-VL 视觉 token 走 M-RoPE，而 latent token 进 coordinator 时按什么 position id 编、6 个 agent 的时序如何注入，只有隐式的块顺序 + delimiter——完全留白。

## 5. 全部数字表

**Table 1 主表**（Video-MME / LongVideoBench / LVBench / MLVU-Test，%）：
- 无预算约束块：GPT-4o 71.9/66.7/30.8/54.9；LLaVA-Next-Video-34B 52.0/50.5/32.2/–；ShareGPT4Video-8B 39.9/41.8/–/33.8；Kangaroo-8B 56.0/54.8/39.4/46.5；VideoLLaMA2.1-7B 54.9/–/36.2/45.6；VideoLLaMA3-7B 66.2/59.8/45.3/47.7。
- 16\*224\*224 预算块：Qwen2.5-VL-72B 59.5/51.0/35.7/44.2；Qwen3-VL-30B 58.3/55.9/38.0/44.0；Qwen2.5-VL-7B 53.7/48.1/32.2/38.7；LLaVA-OneVision1.5-8B 56.1/54.4/36.4/41.8；Keye1.5-VL-8B 55.6/56.4/37.6/41.6；InternVL2.5-8B 47.7/43.4/32.3/39.1；Qwen3-VL-8B 55.9/50.7/33.2/41.5。
- Multi-agent 块（引擎均 Qwen3-VL-8B）：**MapReduce\* 46.7/38.2/30.5/33.3**（注意：比单模型 Qwen3-VL-8B 还低一大截，文字中转是净伤害）；**LatentMAS\* 55.7/48.5/33.2/42.3**（≈单模型持平略赚）；**Ours(Qwen2.5-VL-7B) 58.2/52.5/37.7/47.6；Ours(Qwen3-VL-8B) 60.4/56.8/40.2/49.2；Ours(LLaVA-OV1.5-8B) 59.9/57.6/41.1/49.4**。
- **Table 2**（K 消融，MLVU/LVBench/LongVideoBench）：K=16→46.0/38.7/54.1；K=32→49.2/40.2/56.8；K=48→49.0/40.1/57.2（饱和）。
- **Table 3**（通信成本）：MapReduce 5.156s/387 tokens；LatentMAS 0.649s/784 KV-Cache；Ours 0.537s/192 tokens。
- **Table 4**（阶段消融，Video-MME/MLVU）：无S1→45.5/36.3；无S2→56.9/46.7；无S3→48.2/36.5；全→60.4/49.2。
- **Table 5**（固定 coordinator=Qwen3-VL-8B 换局部 agent）：Qwen3-VL-8B 60.4/56.8/40.2/49.2；Qwen2.5-VL-7B 56.4/51.1/36.5/47.8；Qwen2.5-VL-7B+LLaVA-OV1.5-8B 混合 56.6/52.0/37.0/48.1。
- **Figure 4**（agent 数 M，实为 LongVideoBench）：M=1≈48.5, 2≈52.8, 3≈54.6, 4≈55.7, 5≈56.4, 6≈56.8（训练 M=4，推理外推到 6 仍涨）。
- **Figure 5**（分辨率 128–384，LongVideoBench）：K=32 曲线 53.4→55.7→56.8→57.1→58.1→58.2（320 后饱和）；K=48 曲线 53.2→55.4→57.2→57.4→58.8→59.4（上限被通信带宽解锁）。
- **失败分析/limitations：不存在。** 全文没有 Limitations 节、没有 MACF 自身的失败案例；唯一定性分析（Figure 3 + §4.2）是**文字臂**的失败：多狗视频里 caption 把 "brown-and-white dog" 压成 "white dog" 引发指代绑定错误，agent6 报绳子红色带偏 coordinator。Impact Statement 是模板文。

## 6. 它明确没做的事（我们的开口，逐条核实为"全文无"）

1. **等预算三介质消融**：无。没有图像/关键帧转发臂；文字 vs latent 没有按比特或按等 token 对齐过；通信预算只有 token 计数口径。
2. **消息替换因果审计**：无。没有任何 swap/屏蔽单个 agent 消息、shuffle 顺序、错配 query 之类的因果归因实验。
3. **HourVideo**：无。四个 benchmark 是 Video-MME/LongVideoBench/LVBench/MLVU-Test，没有 HourVideo，也没有按时长分层的分析。
4. **异构角色分工**：无。局部 agent 权重绑定、均匀时序切段、干同样的活；Table 5 的"异构"只是**backbone 异构**，不是角色/任务异构。
5. **接收方主动索取**：无。严格单轮单向星形，coordinator 不能回问、不能二次取证；K 每 agent 恒定，无自适应带宽分配。发送方倒是 query-aware（Eq 4 c=A_m(X,q)）。
6. 附送几个额外空位：无 training-free 信道消融；无 retrieval 系对照；latent token 位置编码/M-RoPE 空白；无 delimiter/adapter/取哪层 hidden states 的设计消融；无 M>6 外推；无音频；训练/推理 agent 数不匹配只靠 Fig 4 一条曲线带过。

## 7. 复现口径

- **代码：未开源。** 正文无 GitHub 链接，arXiv abs 页 Comments 仅 "12 pages"，无 project page。
- **Backbone**：主结果局部 agent 和 coordinator 都是 **Qwen3-VL-8B**（引 Qwen3 technical report 2505.09388；Instruct 还是 Thinking 未说明，精确 HF checkpoint 未给）；泛化用 Qwen2.5-VL-7B、LLaVA-OneVision-1.5-8B。§4.1 baseline 列表里 "Qwen-VL-30B" 是笔误（表里为 Qwen3-VL-30B）。
- **评测口径**：MLVU-Test 只用多选题子集；"follow the decoding configuration used in the official demo code"；**Video-MME 带不带字幕未说明**（GPT-4o 71.9 是带字幕口径的数，混引嫌疑但无法坐实）。
- **复现缺口清单**：d 维度、LoRA rank、adapter 宽度、prompt 模板、三阶段样本量/超参/GPU 时、baseline 的 agent 数与分段方式、字幕政策——全在缺失的附录里。**当前 v1 实际不可复现。**

## 与我们的关系（一句话定位）
这篇 = "训练出来的 latent 信道 + 等 per-agent 感知预算" 叙事，与我们 VLM-LatentMAS 线（training-free KV/latent 转发）正面相邻：它把 LatentMAS 改成多模态当 baseline 并赢它 4.7–8.3 点（绝对），但赢的代价是三阶段域内训练；它的等预算只在"感知 token 数"层面成立，通信侧比特预算、图像转发臂、因果审计、接收方索取全部空白——我们问题清单里的每一条开口都还开着。

## 本地物证
- PDF：/tmp/claude-1008/-home-yilin/9778ba7c-17da-4a86-82a0-2c60bfed94d4/scratchpad/macf.pdf
- HTML 转文本全文：/tmp/claude-1008/-home-yilin/9778ba7c-17da-4a86-82a0-2c60bfed94d4/scratchpad/macf.txt
- 附录核查（第 9–12 页抽取）：/tmp/claude-1008/-home-yilin/9778ba7c-17da-4a86-82a0-2c60bfed94d4/scratchpad/macf_appendix.txt
- 图 1–5：同目录 fig_x1.png … fig_x5.png（x4/x5 即 Figure 4/5 消融曲线）"
  },
  "workflowProgress": [
    {
      "type": "workflow_phase",
      "index": 1,
      "title": "Read"
    },
    {
      "type": "workflow_agent",
      "index": 1,
      "label": "macf-reader",
      "phaseIndex": 1,
      "phaseTitle": "Read",
      "agentId": "a5a6ed95a668236f0",
      "model": "claude-fable-5",
      "state": "done",
      "startedAt": 1786373396772,
      "queuedAt": 1786373394263,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "# MACF 精读回报（arXiv 2605.00444v1，2026-05-01 提交，ICML …",
      "promptPreview": "精读 MACF 论文全文：arXiv 2605.00444《Scaling Video Understanding via Compact Latent Multi-Agent Collaboration》。优先 WebFetch https://arxiv.org/html/2605.00444 （HTML 全文），不够再取 PDF。逐节读完后中文回报，重点回答以下决定我们消融设计的问题：
1. **有没有关键帧/图像转发臂**？它的对照臂清单：latent 协议 vs 什么（MapReduce 文字?还有别的?）——每个对照的输入预算(帧数/分辨率/token)是否与 latent 臂匹配？
2. **等预算问题**：local agent 16帧×224 的"perception budget"下，文字对照臂的 caption 是多少 token？coordinator 收到的总信息…",
      "lastProgressAt": 1786373724455,
      "tokens": 69033,
      "toolCalls": 20,
      "durationMs": 327682,
      "resultPreview": "{"report":"# MACF 精读回报（arXiv 2605.00444v1，2026-05-01 提交，ICML 2026 投稿格式，12 页）\
\
标题：Scaling Video Understanding via Compact Latent Multi-Agent Collaboration。作者：Kerui Chen, Jinglu Wang, Jianrong Zhang, Ming Li, Yan Lu, Hehe Fan（微软亚研+浙大背景组合，Jinglu Wang/Yan Lu 是 MSRA）。\
\
**先报三个读全文才能发现的硬信息**：\
- **arXiv v1 没有附录**。正文两处写 \"More details are provided in the appendix\"（§4.1）和 \"The implementation details o…"
    }
  ],
  "totalTokens": 69033,
  "totalToolCalls": 20
