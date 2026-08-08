# `papers/` 索引与状态标注

> **这个目录是什么**：本项目的**论证与证据层**。压缩后的单一入口是上一层的 [research-digest.md](../VMAS-latent-collab/research-digest.md)（§8 是它的原始索引）；这里放的是没被压缩掉的原文。
>
> **切分维度是「谁写的」，不是「主/辅」**：
> - **`external/`** —— 读别人的：外部论文的精读、原始 PDF、代码走读
> - **`research-2026-08-01/`** —— 写给自己的：那一轮的调研 / 提案 / 评估协议 / 魔鬼代言人 / 实验卡
>
> **为什么目录名带日期**：`research-2026-08-01/` 全部写于 **Phase 1 开跑之前**。Phase 1/2 的实测已经覆盖或推翻了其中一部分。目录名带日期是为了让新会话第一眼就知道"这是那一轮的东西"，必须对照下表的状态列再决定信不信。
>
> **状态记号**：`✅ 仍权威`（可直接引用）· `⚠️ 部分被覆盖`（须对照右列的新文档）· `📌 重新变关键`（因新方向再次成为主线依据）· `❌ 已作废`（见上一层 `TRASH/`，本目录暂无）

---

## `external/` —— 外部论文精读（读别人的）

| 文件 | 内容 | 状态 | 被什么覆盖 / 补充 |
|---|---|---|---|
| [`00_cross-comparison.md`](external/00_cross-comparison.md) | 四篇多模态 latent MAS 的总对比 + 谱系图 + 地盘划分 | ⚠️ 部分被覆盖 | 地盘划分已按 **LACO** 收窄（真空只剩「多模态 latent + VQA」一格）→ [../Q&A/04 §1.3](../Q&A/04-multimodal-mas-landscape-and-baselines.md)；"四篇无一报等算力单智能体上界"这条**仍权威且被反复引用** |
| [`L2-VMAS_method-analysis/`](external/L2-VMAS_method-analysis/) | 最强竞品精读 + **原始 PDF** + HTML。双 latent memory + 三阶段 PPO | ✅ 仍权威 | Table 1/2/3/5 数字已被逐格复核；新增：**v2 删掉了代码承诺，`YU-deep` 账号已注销（404）** → [../Q&A/04 §1.2 D](../Q&A/04-multimodal-mas-landscape-and-baselines.md) |
| [`ViF_method-analysis.md`](external/ViF_method-analysis.md) | 视觉 relay token；代码是 stub 且要训练 | ⚠️ 地址更正 | 论文广告的 `YU-deep/ViF` 已死；**实际仓库在 [xlyu0106/ViF](https://github.com/xlyu0106/ViF)**（HTTP 200）。另：视觉证据集中比例真值是 **1.22%**（非 2%） |
| [`MACF_method-analysis.md`](external/MACF_method-analysis.md) | 长视频，K=32 通信 token + 三阶段课程 | ✅ 仍权威 | 2026-08-03 双路取证（HTML+PDF）确认**所有可核查点均正确**；新增：Table 3 的 `784 KV-Cache` 可反解为 **16 帧 × 49 token/帧**，暗示其 `LatentMAS*` 基线可能只拿到 16 帧 → [../Q&A/05 §3.1](../Q&A/05-credible-mas-screening-and-graft-plan.md) |
| [`VisionWormhole_method-analysis.md`](external/VisionWormhole_method-analysis.md) | 唯一可跑骨架，LatentMAS 的直接 fork | ⚠️ 一处待复核 | 加速分布有争议：本地记「GSM8K 1.02× / −4.6pp」，另一轮调研称那只是 Table 2 一格、54 格中位 1.65× —— **未定，引用前必须复核原文** |
| [`C2C-CACHE-TO-CACHE/`](external/C2C-CACHE-TO-CACHE/) | Cache-to-Cache（ICLR'26）：**PDF + 论文总结 + 代码走读 + repo clone** | ✅ 新增 2026-08-04 | 是 latent 通信里**加性融合**那一路的代表（对照 LatentMAS 的前缀拼接），设计空间对比见 [../Q&A/05](../Q&A/05-credible-mas-screening-and-graft-plan.md) |

---

## `research-2026-08-01/` —— 本项目调研与论证（写给自己的）

### 训练路线调研

| 文件 | 内容 | 状态 | 说明 |
|---|---|---|---|
| [`10_training-objectives-dissection.md`](research-2026-08-01/10_training-objectives-dissection.md) | **九个方法的损失函数逐个解剖**（含公式与源码行号） | ✅ 仍权威 | 损失函数解剖不随实验过期。含 C2C 源码 `SFT_train.py:443-449` 门控正则被注释掉这类一手发现 |
| [`11_channel-evaluation-methods.md`](research-2026-08-01/11_channel-evaluation-methods.md) | **通道评估学**：因果审计 / 信息论 / 涌现通信指标 | ✅ 仍权威 | CAG、控制任务 selectivity、MDL 描述长度这套方法学是 Phase 1/2 全部对照臂的来源 |
| [`12_cross-model-alignment-routes.md`](research-2026-08-01/12_cross-model-alignment-routes.md) | 免训练与少训练的跨模型对齐路线及精度上限 | 📌 **重新变关键** | brain-LLM ↔ tool-VLM 的 latent 通道方向直接依赖这份（Relative Representations 的免训练锚点法、Procrustes、vec2vec 等）。作者当时下载的 10 篇 PDF 路径已失效，需要时重取 |
| [`13_supervision-and-model-size.md`](research-2026-08-01/13_supervision-and-model-size.md) | 监督信号从哪来 + 通信模块参数量对比 | ✅ 仍权威 | — |

### 方案 / 协议 / 审查

| 文件 | 内容 | 状态 | 说明 |
|---|---|---|---|
| [`20_trained-comm-proposals.md`](research-2026-08-01/20_trained-comm-proposals.md) | 三个训练式方案（零训练 / Perceiver codec / RL） | ⚠️ 前提已变 | 写在 Phase 1 之前。**M1 零载荷**（四重复现）改变了"latent 思考步需要被训好"这个前提；同时 2026-08-03 的筛选确认**"训练过的 agent 间通信模块"是唯一被证明能把多智能体转成增益的东西** → 方案本身反而更值钱，但要按新前提重写 |
| [`21_evaluation-protocol.md`](research-2026-08-01/21_evaluation-protocol.md) | **完整评估协议**：分层指标 / 对照组 / 样本量 / 反作弊 | ✅ **权威，且被主线直接引用** | [phase1-cards-explained.md](../VMAS-latent-collab/phase1-cards-explained.md) 的方法学出处。**不是"辅助"文档** |
| [`22_devils-advocate-training-route.md`](research-2026-08-01/22_devils-advocate-training-route.md) | 训练路线最强反对意见 + 总判决 | ⚠️ 部分被回答 | 其中若干反对已被 Phase 2 数据直接回答；但"参数配平后通信收益消失"那条仍然成立且重要 |

### 判决 / 执行

| 文件 | 内容 | 状态 | 说明 |
|---|---|---|---|
| [`30_verdict-and-roadmap.md`](research-2026-08-01/30_verdict-and-roadmap.md) | 方向判决与路线（比 digest 更细的论证） | ⚠️ 写在 Phase 1 前 | 方向判决已被 Phase 1/2 结果 + 2026-08-03/04 的三轮筛选大幅更新 → 以 [../Q&A/04](../Q&A/04-multimodal-mas-landscape-and-baselines.md) 与 [../Q&A/05](../Q&A/05-credible-mas-screening-and-graft-plan.md) 为准 |
| [`31_experiment-cards-trainingfree.md`](research-2026-08-01/31_experiment-cards-trainingfree.md) | 免训练路线可执行实验卡 | ⚠️ **环境事实已过期 + 卡片已执行完** | 头部写"**没有** Qwen3-VL 权重"——**现在有了**（`/data/yilin/models/Qwen3-VL-8B-Instruct`）。四张卡已全部跑完并有判决 → [../phase1-results.md](../VMAS-latent-collab/phase1-results.md)。**新会话不要照着这份的环境事实去跑** |
| [`32_devils-advocate-trainingfree.md`](research-2026-08-01/32_devils-advocate-trainingfree.md) | 免训练路线最强反对意见 + M-RoPE 一手证据 | ⚠️ 证据已被超越 | 其 M-RoPE 位置坍缩探针已被 **B1 接管模块**超越（两级验收，top-1 100%）；且 2026-08-04 在 transformers 源码层发现坍缩比它记录的更严重（第 2..M 个 agent 的**整段 prompt** 都不调 `get_rope_index`） |

---

## 引用纪律

1. **先看状态列再引**。`⚠️` 的必须同时读右列指向的新文档。
2. **`research-2026-08-01/` 里的环境事实一律不要直接采信**（权重、数据集、显存余量都变了）。以 [../Q&A/05](../Q&A/05-credible-mas-screening-and-graft-plan.md) 的可复现命令为准。
3. **`external/` 的论文数字仍可直接引**，但 L²-VMAS 的 Table 1 有两处印刷错误（InternVL 行 Single avg token 应为 **461** 不是 3461；L² 行应为 **2245** 不是 3011）—— 引用前自己算列均值。
4. 作废文档进上一层的 `TRASH/`，不放这里。

## 变更记录

- **2026-08-04**：按「谁写的」拆成 `external/` 与 `research-2026-08-01/`，新增本 README 与状态标注。同步更新了 5 处入向引用（[phase1-results.md](../VMAS-latent-collab/phase1-results.md)、[phase1-cards-explained.md](../VMAS-latent-collab/phase1-cards-explained.md) ×2、[../Q&A/04](../Q&A/04-multimodal-mas-landscape-and-baselines.md)、`TRASH/`）、[research-digest.md](../VMAS-latent-collab/research-digest.md) §8 的三个小标题、以及 `30_verdict-and-roadmap.md` 的内部索引。
