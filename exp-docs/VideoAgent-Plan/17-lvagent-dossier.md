# 17 · LVAgent 精读档案（arXiv 2503.10200 / ICCV 2025，baseline 复现规格书）

> 2026-08-10 实现级精读存档。一句话：增益来自检索+淘汰而非集成；同质 3×VL-30B 复现不破坏机制
> （2-4 天薄壳）；HourVideo 上三个结构性预设败因（单 chunk 喂食/小时级召回崩/无字幕腿）；
> 官方代码有 GT 条件缓存,复现其数字须绕开并标注。

ary": "LVAgent (2503.10200) implementation-grade read for baseline reproduction",
  "agentCount": 1,
  "logs": [],
  "result": {
    "report": "# LVAgent（arXiv 2503.10200, ICCV 2025）实现级精读报告

**信源**：[arXiv HTML 全文](https://arxiv.org/html/2503.10200) + [GitHub 仓库](https://github.com/64327069/LVAgent)（master 分支，41 stars）。核心源码已下载到本地精读：
- [discuss_final_lvbench.py](/tmp/claude-1008/-home-yilin/9778ba7c-17da-4a86-82a0-2c60bfed94d4/scratchpad/discuss_final_lvbench.py)（主循环，逐行读完）
- [all_model_agent.py](/tmp/claude-1008/-home-yilin/9778ba7c-17da-4a86-82a0-2c60bfed94d4/scratchpad/all_model_agent.py)（agent 封装）
- [all_model_util.py](/tmp/claude-1008/-home-yilin/9778ba7c-17da-4a86-82a0-2c60bfed94d4/scratchpad/all_model_util.py)（全部 prompt 原文）

以下标注【实测=代码/论文原文确认】【推断】。

---

## 1. 完整管线

**四阶段：Selection → Perception → Action → Reflection，最多 3 轮，动态淘汰。**

**模型阵容（全开源，无闭源 API）**【实测】：模型库 = Qwen2-VL-7B/72B、LLaVA-Video-72B、LongVU-7B、InternVL-2.5-8B/78B、Oryx-34B、Aria-28B。代码里实际跑的队伍（LongVideoBench）= **InternVL2.5-8B + InternVL2.5-78B + LLaVA-Video-72B** 三个 agent，硬编码在 discuss_final_lvbench.py:462-471。

**Selection（离线，每个数据集做一次）**：无标签抽 150 个视频，各候选 agent 跑一遍 Perception+Action，多数票答案当伪标签，按伪标签准确率选 top-3 组队。代码中该阶段不存在——直接硬编码选好的三个模型。

**每轮流程（代码实测，与论文有出入处标注）**：
- **Perception**：每 agent 三小步：(a) 看 4 帧（`watch_samp`）+ 字幕，回答 Yes/No"需不需要看全片"；(b) 生成 key information K_i（≤50 词文本，第 2 轮起改为基于讨论历史重生成）；(c) 若 No→检索：视频切 **6 等份 chunk，每 chunk 随机采 16 帧**，用 ASP-CLIP 算 `Sim = ASP(帧, K_i) + ASP(帧, Q)`，**代码里是纯 argmax 选 1 个最优 chunk**（论文写的"阈值 0.8 以上多选"在放出的代码里不存在，`get_max_frame_block` 只取最高分块）；若 Yes→全局随机 16 帧。
- **Action**：agent 拿选中的 16 帧回答选择题（"Respond with only the letter"），解析规则 `result.split('Answer: ')[-1][0]`。**早停**：3 agent 中某答案 ≥2 票即定案返回（第 1 轮条件 `len(answer_set)<3`；第 2 轮 2 agent 须一致）。
- **Reflection**（仅在无共识时触发）：(a) 每 agent 用另一次调用生成选该答案的 reasoning；(b) 每 agent 给包括自己在内的所有 agent 的 reasoning 打 1-10 分（JSON 格式输出）；(c) 汇总分数，**总分最低者淘汰**。平分时按硬编码优先级淘汰 `['intern_8b','llava_72b','intern_78b']`（小模型先死）；JSON 解析失败时回退到硬编码默认分 `{'intern_78b':8, 'intern_8b':6, 'llava_72b':7}`——**淘汰机制里有偏向大模型的先验**，论文没写。(d) 幸存者基于讨论历史（各家答案+理由+分数+被淘汰者及原因）重生成 K_i，进下一轮。
- **终局**：3→2→1，若三轮都无共识，最终答案 = 唯一幸存 agent 第 3 轮的回答（不是投票）。

**Prompt 要点**（全文在 all_model_util.py，可直接照抄）：decide_watch（Yes/No 判断）、info_prompt（生成 50 词 key info）、answer_prompt（字幕全文+选择题+"Respond with only the letter"）、reason_prompt（"deeply analyze the logical steps and evidence"）、discuss_prompt（互评 1-10 输出 dict）、history_prompt（含被淘汰者的答案+"this reason was deemed unconvincing"）。注意 LongVideoBench/VideoMME 的**每个 prompt 都内嵌整段字幕**。

## 2. 每 agent 的视频输入形态

【实测】决策用 4 帧 → 回答用 **16 帧**（全局随机或最优 chunk 内随机）；检索打分时 6×16=96 帧过 ASP-CLIP（224×224，便宜）。分辨率按模型自身管线：InternVL 448×448 每帧 1 tile+thumbnail；LLaVA-Video 官方 preprocess。生成参数值得注意：InternVL 两个 agent `do_sample=True`（有随机性），LLaVA-72B 贪心。工具只有一个：**ASP-CLIP 检索器** = CLIP4Clip 架构（seqTransf 头，4 层时序 transformer/8 头）在自建 LongVR 数据集（82K 条，ActivityNet/OpenVid/ViTT/MovieChat/Youcook2 拼的，5-720s 片段）上微调。**checkpoint 未放出**（代码里是他们集群的私有路径）。无搜索引擎/OCR/memory bank。

## 3. 关键数字

主表（accuracy）：
| 模型 | EgoSchema | LongVideoBench | MLVU | VideoMME (无/有字幕) |
|---|---|---|---|---|
| GPT-4o | 72.2 | 66.7 | 65.3/72.1 | 71.9/77.2 |
| Gemini-1.5-Pro | 71.1 | 64.0 | 67.4/77.4 | 75.0/81.3 |
| Qwen2-VL-7B | 66.7 | 55.6 | — | 63.3/69.0 |
| **LVAgent** | **82.9** | **80.0** | **83.9** | **81.7/86.6** |
| vs 此前 SOTA | +5.0 | +13.3 | +8.2 | +6.1/+6.2 |

**更诚实的净增益口径**：消融表里"无 Perception 无 Reflection"的裸 3-agent 投票 = 77.9/63.9/75.7/72.1——这一行和 InternVL2.5-78B 单模型公开成绩几乎重合（MLVU 75.7、VideoMME 72.1 完全一致）【推断：裸集成 ≈ 最强单成员水平】。即**管线相对最强单成员的真实增益 ≈ LongVideoBench +16、MLVU +8、VideoMME +9.6，几乎全部来自 Perception(检索)+Reflection(淘汰)，纯异构集成本身贡献≈0**。轮数消融：1→3 轮 LongVideoBench 69.0→80.0。检索器消融：ASP-CLIP > LongCLIP > 原版 CLIP（全 benchmark）。效率表：LVAgent 平均 71.2 帧/33.6s vs Qwen2-VL-72B 568 帧/90.5s。

**无 HourVideo、无小时级结果**。四个 benchmark 最长 60 分钟（VideoMME long 子集 30-60min）；LVBench(Zhipu, 均长 68min) 也没测。

## 4. 代码开源状况 + 换同质 3×Qwen3-VL 是否破坏机制

**开源但不完整、不可直接跑**【实测】：
- 只有 LongVideoBench 的驱动脚本；纯 `transformers` + `decord` + threading（无 vLLM/lmdeploy），三模型按 device_map 切层塞 8×A800。
- **两个关键缺件**：(1) ASP-CLIP 微调 checkpoint 未放出，LongVR 数据集也未放出；(2) 主脚本的输入是一个预处理好的 `lvbench_all_result_change.json`（含每 agent 的 `watch_samp`/`sample_idx`/`info` 初值），**生成它的预处理脚本不在仓库里**。
- **一处需要警惕的实现**（discuss_final_lvbench.py:206-207, 264-265）：`if 'Yes' not in watch and answer_dict[agent_name] == chr(ord('A') + anno['correct_choice']): sample_dict[agent_name] = {...}`——**只有答对时才缓存该次采到的帧**，且脚本把结果回写进自己的输入 json（断点续跑设计）、后续轮会读 `anno[agent_name]['sample_dict']` 复用缓存帧。是否污染了论文数字取决于未放出的预处理链，但复现时必须绕开这套缓存，重跑他们的号码有虚高风险【推断，代码证据确凿、影响面未验】。

**同质 3 实例是否破坏机制**：论文**没有同质消融**，全部实验都是异构组合。机制拆解：三个收益来源里，(a) **Perception 检索**完全不依赖模型多样性，照搬即得；(b) **投票去相关**会缩水——但代码里每 agent 独立随机采帧 + do_sample=True + 各自生成不同 K_i，同权重下输入侧仍有真实多样性；(c) **互评淘汰**语义会变（同模型互评趋同、平分回退更频繁），且硬编码的模型优先级/默认分需换成对称规则。结合第 3 点"裸集成贡献≈0、增益主要来自机制"，**同质复现不破坏其主增益来源，反而是论文没做过的干净消融点**——这对你们是加分项而非风险。Selection 阶段在同质下自然退化为空操作，可删。

## 5. 计算成本（按代码精确清点）

每题 MLLM 调用数：
- 第 1 轮：每 agent 3 次（watch 判断 + key info + 回答）× 3 = **9 次**；早停命中即结束（效率表 33.6s/题说明多数题在此结束）。
- 无共识加 Reflection：reason×3 + 打分×3 + 历史 info×3 = +9 → 18 次。
- 第 2 轮：回答×2（+早停判定）→ 20 次；再无共识 +6 → 26 次。
- 第 3 轮：回答×1 → **最坏 27 次**。
另加每轮每个走检索的 agent 6 次 CLIP4Clip 前向（224px、16 帧，可忽略）。即 **9（最好）/ 27（最坏）次 MLLM 调用/题**，输出都很短（max_new_tokens 64-128）。

## 6. 判定：本地 3×Qwen3-VL-30B 复现

**结论：全新实现（薄壳重写），但工作量小；仓库当规格书用，不当代码库用。** 理由：现成代码缺预处理脚本和 ASP-CLIP 权重、路径全硬编码、模型封装绑死 InternVL/LLaVA、还有上述 GT 条件缓存——修它比重写贵。重写清单：
1. 讨论主循环 ~400 行（逻辑在 discuss_final_lvbench.py 里已完全看清，prompt 从 all_model_util.py 照抄）；
2. 模型侧一个 vLLM server 即可：同质 3 agent = 同一 server 三路采样（不同 seed/温度/帧子集），**不需要载 3 份权重**，2 卡跑 30B-A3B 绰绰有余，远低于 6 卡上限；
3. 检索器替代：ASP-CLIP 拿不到 → 用原版 CLIP/SigLIP 打 chunk 分（论文 Table 5 证明原版 CLIP 版本可跑、只是分低），或直接用 Qwen3-VL 自己给 chunk 打相关分；如实标注偏离。
估计 **2-4 天**出可跑 baseline。

**HourVideo 上的机制预期**【推断】：
- **可能赢**：needle/局部片段类问题——6-chunk 检索把 1 小时压到"最相关的 10 分钟里的 16 帧"，比全局均匀 16 帧强得多；多路独立采帧+投票压方差。
- **大概率输**：(a) 跨段聚合类任务（summarization、全片时序推理）——代码只把**单个** chunk 的 16 帧喂给回答 agent，跨 chunk 信息结构性丢失；(b) 1 小时视频每 chunk 10 分钟只采 16 帧给 CLIP 打分（1 帧/37.5s），检索本身召回率会崩；(c) 其 prompt 体系严重依赖字幕（LongVideoBench/VideoMME 每个 prompt 嵌整段字幕），HourVideo 是无字幕第一人称视频，这条腿直接没了——这也解释了它为什么只在有字幕/≤60min 的 benchmark 上报数。当 baseline 时建议把"chunk 数固定 6"和"只喂最优 chunk"作为它在小时级上的两个预设败因来打。"
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
      "label": "lvagent-reader",
      "phaseIndex": 1,
      "phaseTitle": "Read",
      "agentId": "aa9c0ccc36d88d37c",
      "model": "claude-fable-5",
      "state": "done",
      "startedAt": 1786373827846,
      "queuedAt": 1786373825534,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "# LVAgent（arXiv 2503.10200, ICCV 2025）实现级精读报告 **信…",
      "promptPreview": "实现级精读 LVAgent：arXiv 2503.10200（多轮动态协作长视频理解）。WebFetch https://arxiv.org/html/2503.10200 全文 + 搜其 GitHub 代码仓。中文回报，目标=判断能否在本地 3×Qwen3-VL-30B 上复现当 baseline：
1. 完整管线：几个 agent、什么模型阵容（闭源 API 还是开源?）、几轮、每轮做什么（选拔/讨论/淘汰的精确规则与 prompt 要点）
2. 每 agent 的视频输入形态：帧数/采样/分辨率；有没有检索/工具
3. 关键数字：各 benchmark 分数表（尤其 vs 单模型基线的净增益、用的什么单模型）；有没有 HourVideo/小时级结果
4. 代码开源吗？依赖什么框架？把模型阵容换成同一个本地 VL（同质3实例）是否破坏其机制（它的收益依赖模型多样性吗——文中有没有同质消…",
      "lastProgressAt": 1786374188797,
      "tokens": 71403,
      "toolCalls": 16,
      "durationMs": 360950,
      "resultPreview": "{"report":"# LVAgent（arXiv 2503.10200, ICCV 2025）实现级精读报告\
\
**信源**：[arXiv HTML 全文](https://arxiv.org/html/2503.10200) + [GitHub 仓库](https://github.com/64327069/LVAgent)（master 分支，41 stars）。核心源码已下载到本地精读：\
- [discuss_final_lvbench.py](/tmp/claude-1008/-home-yilin/9778ba7c-17da-4a86-82a0-2c60bfed94d4/scratchpad/discuss_final_lvbench.py)（主循环，逐行读完）\
- [all_model_agent.py](/tmp/claude-1008/-home-yilin/9…"
    }
  ],
  "totalTokens": 71403,
  "totalToolCalls": 16
