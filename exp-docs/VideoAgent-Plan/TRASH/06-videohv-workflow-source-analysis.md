# VideoHV-Agent 工作流源码解剖

> **日期**：2026-08-05 · **repo 已 clone**：`/home/yilin/VideoHV-Agent/`（⚠️ 外层是项目页，**代码在内层 `VideoHV-Agent/VideoHV-Agent/`**，本文行号链接指向它）
>
> **读者起点**：知道我们的五臂实验与 P1–P5（[00-plan §3](00-plan.md) / [03 §0.5](03-eight-risk-papers-explained.md)）。**其余从零讲。**
>
> **读法**：§0 ⭐ 三个修正 doc 04 的发现 · §1 先认人 · §2 循环逻辑手把手 · §3 关键细节与判分坑 · §4 我们的切口 · §5 数据资产 · §6 五臂落点 · §7 对照表。

---

## TL;DR

1. 思路与 VideoAgent 相反：**不是"看了再答"，是"先把每个选项改写成可检验的假设，再派验证员拿视频证据去证实/证伪"**——像法庭，不像问答。
2. 四个角色（出题人/判官/验证员/选答人）**全是文本 LLM**；感知有三份**离线资产**（LaViLa 逐帧 caption、GPT-3.5 分段摘要、CogAgent 物体检测）+ 一个**按需 caption 工具**（验证员现场调，≤5 帧/次）。
3. ⭐ **doc 04 的"100% OpenAI API 要整体重写"需要三分**：**验证员默认就打本地 Qwen**（`localhost:8000/v1`，作者注释原话 "Local LLM (Qwen-compatible OpenAI API)"）；只有 caption 工具与结构化三阶段默认 OpenAI，且**全部走环境变量可换**。本地化比之前判的更近。

---

## §0 ⭐ 三个修正 doc 04 的源码发现

| # | 发现 | 源码 | 改变什么 |
|---|---|---|---|
| 1 | **验证员默认本地 Qwen**：`LLM_BASE_URL` 默认 `http://localhost:8000/v1`，注释 *"Local LLM (Qwen-compatible OpenAI API)"* | [config.py:21-23](../../../VideoHV-Agent/VideoHV-Agent/video_hv/config.py#L21-L23) | **作者自己就是本地 Qwen 跑验证员的**。"100% OpenAI"只对 caption 工具（[config.py:26-28](../../../VideoHV-Agent/VideoHV-Agent/video_hv/config.py#L26-L28)）和结构化三阶段（[config.py:31-34](../../../VideoHV-Agent/VideoHV-Agent/video_hv/config.py#L31-L34)）成立，**且全有 env 开关**。H3 spike 连代码都不用改，只设 env |
| 2 | **三个数据集的感知资产全带**：`load_data/` 里 egoschema 58M + nextqa 12M + intentqa 9.9M（标注/caption/摘要/检测齐） | [load_data/](../../../VideoHV-Agent/VideoHV-Agent/load_data/) | 之前说"NextQA/IntentQA 什么都没有"不对——**缺的只是 pipeline 代码（77 行 cli + prompts 适配）和预抽帧图片**，H5 工作量下调 |
| 3 | **假设是从"摘要"生成的，不是原始 caption** | [runner.py:78-85](../../../VideoHV-Agent/VideoHV-Agent/video_hv/pipelines/egoschema_openai/runner.py#L78-L85) 传入的是 `action_caption_summaries` + `object_detection_summaries` | 出题人读的是 GPT-3.5 写的 4 段摘要 + 物体清单——**信息瓶颈比想象中更窄**，这对我们的 motivation 有利 |

## §1 先认人（四个角色 + 两类感知）

| 角色 | 干什么 | 引擎（默认） | 源码钉子 |
|---|---|---|---|
| ✍️ **出题人** | 把每个选项改写成**可检验假设**（必须写明实体/动作/时序；明显离谱的选项直接不立） | OpenAI 结构化（env 可换） | [openai_stages.py:34](../../../VideoHV-Agent/VideoHV-Agent/video_hv/pipelines/egoschema_openai/openai_stages.py#L34) |
| ⚖️ **判官** | 这几条假设**互相区分得开吗**？打 0–1 分 + 给出**一条鉴别线索**（"检查递东西发生在说话之前还是之后"） | 同上 | [openai_stages.py:126](../../../VideoHV-Agent/VideoHV-Agent/video_hv/pipelines/egoschema_openai/openai_stages.py#L126) |
| 🔍 **验证员** | 带着鉴别线索查证据：读 180 帧 caption 上下文，**可现场调 caption 工具看指定帧（≤5帧/次，≤2 轮）**，输出 verified/partially/not_verified | ⭐ **本地 Qwen** | [verifier.py:38](../../../VideoHV-Agent/VideoHV-Agent/video_hv/verifier.py#L38)，工具轮上限 [verifier.py:107](../../../VideoHV-Agent/VideoHV-Agent/video_hv/verifier.py#L107) |
| 🧑‍⚖️ **选答人** | 读全部验证记录，选幸存假设对应的选项（被明确要求"只参考、不照单全收"） | OpenAI 结构化 | [openai_stages.py:153](../../../VideoHV-Agent/VideoHV-Agent/video_hv/pipelines/egoschema_openai/openai_stages.py#L153) |

**感知资产（全离线，问题无关）**：LaViLa 逐帧 caption（[cli.py:22](../../../VideoHV-Agent/VideoHV-Agent/video_hv/pipelines/egoschema_openai/cli.py#L22)）· GPT-3.5 的 4 段 dpcknn 摘要 + CogAgent 物体检测摘要（[cli.py:21](../../../VideoHV-Agent/VideoHV-Agent/video_hv/pipelines/egoschema_openai/cli.py#L21)）。
**按需感知**：caption 工具（[vision_tools.py:37](../../../VideoHV-Agent/VideoHV-Agent/video_hv/vision_tools.py#L37)）——给帧号区间，VLM 现场写描述；⚠️ **prompt 是通用的"详细描述人物/物体/动作/场景"**（[vision_tools.py:53](../../../VideoHV-Agent/VideoHV-Agent/video_hv/vision_tools.py#L53)），**不带问题也不带假设**（A′ 臂 = 就改这一处）。

> 📌 **桥句**：**四个文字角色轮流说话，只有 caption 工具那一步真的看图——而它看图时不知道自己在为哪个假设找证据。**

## §2 循环逻辑手把手

入口 [cli.py:19](../../../VideoHV-Agent/VideoHV-Agent/video_hv/pipelines/egoschema_openai/cli.py#L19) → 每题进 [runner.py:30](../../../VideoHV-Agent/VideoHV-Agent/video_hv/pipelines/egoschema_openai/runner.py#L30)。先采样：**180 个帧索引**（linspace，[constants.py:6](../../../VideoHV-Agent/VideoHV-Agent/video_hv/pipelines/egoschema_openai/constants.py#L6)）——caption 上下文和候选图片都取这 180 个。

**主循环 = 最多 3 轮 refinement**（[runner.py:72](../../../VideoHV-Agent/VideoHV-Agent/video_hv/pipelines/egoschema_openai/runner.py#L72)，`MAX_REFINEMENT_ROUNDS=3`）：

```
第 r 轮：
  ① 立假设   r=0：从【摘要+物体清单+选项】生成          [runner.py:78]
             r>0：读上一轮验证失败的反馈重新立           [runner.py:91]
  ② 只剩 1 条假设？→ 直接答它对应的选项，结束            [runner.py:191-193]
  ③ 判官打区分度分 → score < 0.5 就打回重写一次          [runner.py:114,127]
     并产出一条【鉴别线索】（全体假设共用一条！）
  ④ 验证员带线索开工（本地 Qwen）：
       上下文 = 180 帧的 LaViLa caption
       可调 caption 工具 ≤2 轮、每次 ≤5 帧               [verifier.py:96,107]
       输出 verified / partially_verified / not_verified
  ⑤ 选答人整合出答案                                     [runner.py:174]
  ⑥ 验证记录里含 "not_verified" → 进下一轮重立假设；否则结束  [runner.py:188-190]
异常兜底：random 答案                                     [runner.py:196]
```

### 手把手例题

*"C 在修完自行车后干了什么？0.骑走 1.收工具 2.洗手 3.打电话 4.拍照"*

| 步 | 发生什么 |
|---|---|
| 立假设 | 5 个选项 → 4 条假设（"拍照"与摘要严重不符，不立）："修车后 C 把工具放进包里"… |
| 判官 | "假设 0 和 1 都涉及收尾动作，区分度 0.6"；鉴别线索："**检查 C 放下扳手之后，手接触的下一个物体是车把还是工具包**" |
| 验证员 | 读 180 帧 caption，定位修车结束在 ~150 帧附近 → 调 caption 工具看 148–152 帧 → 返回"C 将扳手放入蓝色工具包" → 假设 1 verified，假设 0 refuted |
| 选答人 | 整合 → **答 1** |

### 追问回填（2026-08-05）：验证员怎么读 caption——两层机制

> **追问原话**："验证员怎么读帧的 caption 的？是把所有的 caption 全部吃进去，然后再定位具体多少帧？"

**是的，而且"定位"和"验证"用的是两层不同的 caption：**

```
runner.py:44   sampled_frame_captions = read_cap(per_frame_captions, frame_indices)
               # NUM_FRAME_SAMPLES=180（constants.py:6）；EgoSchema 3min@1fps 恰好 180 帧
               # → linspace(0,179,180) = 全量一条不落
runner.py:161  external_model_answer(..., sampled_frame_captions, ...)   ← 整个 dict 给验证员
verifier.py:84 "Context summary (video caption / description): {caption}"  ← prompt 从第一个 token 起就装着全图
```

| 层 | 是什么 | 干什么 | 谁产的 |
|---|---|---|---|
| **第 1 层：全量粗地图** | 180 条 LaViLa 一句话（≈1.7k token，**每次验证员调用都付**） | **定位**：LLM 读地图，语言里直接报帧号 "frame_range: 148-152" | LaViLa，离线 |
| **第 2 层：按需放大镜** | 指定 ≤5 帧的**现场细 caption**（真实图片 base64 发给 caption 模型） | **验证**：更强模型重新看图 → 新证据 | GPT-4o 默认 / 本地 Qwen3-VL，在线，≤2 工具轮 |

⭐ **与 VideoAgent 的定位机制完全相反**：

| | VideoAgent | VideoHV |
|---|---|---|
| 大脑起步看到 | 只有 5 条 caption | **全部 180 条** |
| 定位靠什么 | 嵌入检索机器（"希望看到的画面"→文本嵌入 vs 帧特征点积） | **LLM 自己读地图报帧号——循环里没有任何嵌入检索** |
| "再看"拿到什么 | 该帧的**同一份**离线粗 caption（无新信息源） | 现场细 caption（**真的新信息**） |

**推论**：VideoAgent 的"再看"只是把粗地图多翻几页；VideoHV 的"再看"才是真放大镜。这也是 §手术里"两条文字通道天然分离、可分开消融"的机制根源——全图通道（定位用）与工具通道（验证用）各自可换 latent。**效率账注意**：全图那 ≈1.7k token 是每次验证员调用的固定开销 × 线索数 × refinement 轮数，B 臂换掉全图槽的收益空间比换工具槽大。

### 追问回填（2026-08-05）之二：有了 180 条 caption，为什么还要调 caption 工具？

> **追问原话**："已经有 180 帧 caption 了，为什么还要调用 caption 工具？"

**因为两者根本不是同一个东西：①是"巡逻日志"，②是"调监控录像"。**

| | ① 180 条粗 caption | ② 工具返回的细 caption |
|---|---|---|
| 谁写的 | LaViLa（小 caption 模型） | GPT-4o / Qwen3-VL（强 VLM） |
| 何时写 | 离线一次性 | 现场，验证员点名后 |
| 看什么 | 每帧扫一眼 | ≤5 帧的**原始像素**（base64 真发过去） |
| 形态 | **平均 7.2 词**："#C C picks up a wrench" | 数句细描述："扳手放进蓝色工具包，拉上拉链，无手机" |

**例**：线索="修好车后收拾工具（B）还是拿手机（C）？"——粗地图 150 帧只写 "#C C puts the tool down"（放哪？之后拿了啥？没写→**判不了**）。验证员用①定位到 150 帧附近，调②看 148–152 帧原始像素 → "扳手入蓝色工具包，无手机" → B 证实。**①告诉你去哪找，②告诉你那里到底发生了什么**——卡片目录 vs 翻开那一页。

**为什么不一开始把 180 条全写细**：经济账——180 帧全细 = 每视频离线付 180 次强 VLM 调用 + ~9k 词上下文，而绝大多数帧与任何问题无关。两层设计 = 便宜的全覆盖 + 只在要害处付精度，与 VideoChat-A1 coarse-to-fine、SPARC 低分辨率搜+选区高清是**同一个经济学**。更根本地：粗 caption 问题无关，永远不可能提前写对每道题要的细节；工具的问题条件化发生在**选帧**上。

**备注**：工具非必调（`tool_choice="auto"`）——线索简单时验证员直接答，零工具调用。

**衔接"之三"的主刀判定**：②的细 caption 虽细**但仍是文字**（"蓝色工具包"写了，拉链拉几厘米、旁边还有什么没写）——**B 臂换的正是这份"细但仍有损"的文字**；①只干定位，粗无所谓，留文字。

### 追问回填（2026-08-05）之三：latent 通信发生在哪——四条边逐条判

> **追问原话**："要确认一下 latent comm 发生在哪里。比如 VideoHV，文本作为交流工具时验证员读 180 帧 caption——换成 latent 来交流呢？"

**换 latent 后"交流"不是一条边。VideoHV 里 viewer→reasoner 方向有三条边 + 一条反向控制边**：

| 边 | 文字版 | latent 版 | 判定 |
|---|---|---|---|
| **② 工具通道** | caption 工具返回细 caption（`role:"tool"` 消息） | ⭐ Qwen3-VL **只 prefill ≤5 帧不解码** → adapter → m 伪 token 插在 tool 消息位置 | ⭐ **主切口** |
| **① 全图通道** | 180 条粗 caption 当定位地图（≈1.7k tok/次） | 在线编码 180 帧＝36k 视觉 token prefill，**倒亏**；⭐ 可行版=**离线 latent caption 库**（每帧预算 4~8 伪 token，180×4≈720 位置，32KB/帧；⚠️ 继承"问题无关"限制，对标 A 臂非 A′） | **第二切口**，单独消融臂 |
| **③ 摘要通道** | caption → Qwen3-8B 写 4 段摘要 → 出题人/判官 | 保持文字 | 不动——文本→文本边，viewer 不在场，换它=重跑 LatentMAS 已知结论 |
| **反向控制边** | 验证员报 "frame_range: 148-152" | 保持文字 | 不动——审计约束 2 的语义锚 + 我们的新颖性 (d) 本身 |

**主切口选②不选①的三个理由**：(1) **含金量**——①只管定位（粗即可），②是证据；Sterner 的细粒度结论（latent 在颜色/计数/指代绑定赢）和 MACF"棕白狗→白狗"病例都落在证据通道；(2) **效率**——②省 caption 模型的 decode（贵的那半），viewer prefill ≤5 帧很小；①在线换 latent 为省 1.7k 文字付 36k 视觉 prefill；(3) **风险**——"从伪 token 做定位决策"正是 NavGPT-2 Table 5 负结果压着的事，归 D0 探针管辖，先证证据通道再碰定位通道。

**注入力学（比 VideoAgent 单槽复杂一档的具体形态）**：

```
序列 = [system][W_prompt 文字（①地图保持文字）]
       [assistant: tool_call 文字]
       [tool 消息: ◄◄ m 个伪 token ►►]        ← 注入位点 1
       [assistant: 继续推理 文字]
       [第二次 tool_call…]                     ← ≤2 工具轮 → 最多两个位点
```

整条序列走 `inputs_embeds`（文字过 embed_tokens、伪 token 位置放 adapter 输出）；注入点后面还有文字要生成，位置记账连续；vLLM OpenAI API 不吃 `inputs_embeds` → B 臂自有 serving（前文已记）。

> ⭐ **一句话设计原则（本次问答净产出）**：**凡承担「控制/定位」功能的信息走文字；凡承担「证据/感知载荷」的信息走 latent。** 四条边对号：反向控制（文字）· ①定位（先文字，latent 库进消融）· ②证据（latent 主刀）· ③意见（文字不碰）。这是 [00-plan §8.3 约束 2](00-plan.md)"混合通道"在本 host 上的具体展开。


### 追问回填（2026-08-05）之四：摘要通道在哪——主循环里为什么"看不到"它

> **追问原话**："摘要通道我也没懂，原来的主循环没看到。"

**因为摘要跟 180 条 caption 一样是离线资产，主循环里它只在"填 prompt 槽"那一下出现**——就是出题人 prompt 里那行不起眼的 `Context summary (optional): {action_clip_context}`，槽里装的正是它。

**通俗版：把一道题从头到尾走一遍，看每个人桌上放的是什么纸**

**开跑之前（做数据集时就干完了，跟答题无关）**——两个工人提前干活，产物存进文件柜：

**工人 A（LaViLa，看图的小模型）**：把 180 帧每帧扫一眼，写一本 **180 行流水账**：

```
frame 0:   有人蹲在自行车旁
frame 1:   有人拿起扳手
…
frame 150: 有人放下工具
…
frame 179: 有人站在门口
```

**工人 B（GPT-3.5，只会读文字）**：他**不看视频**，拿工人 A 的流水账压成一页 **4 幕剧情梗概**：

```
第 1 幕（0-45）：  一个人检查倒放的自行车，摆弄链条
第 2 幕（45-90）： 用扳手拧后轮螺丝
第 3 幕（90-135）：转动踏板测试，收起工具
第 4 幕（135-180）：起身离开车库
```

**"摘要通道"就是这一页梗概。** 它不是主循环里谁生成的——是答题时从文件柜里抽出来的旧纸，所以主循环里"看不到"它的出生，只看得到它被塞进 prompt 的那一下。

**答题开始：每个人桌上放的纸**

| 谁 | 桌上放的纸 |
|---|---|
| **出题人** | 问题 + 选项 + **那页梗概** |
| **判官** | 假设列表 + **那页梗概** |
| **验证员** | **那本 180 行流水账**（+ 一部能调监控的电话） |
| **选答人** | 假设 + 验证记录 + **又是那页梗概** |

**出题人为什么需要梗概**——看它干活就懂：

> 任务："把每个选项改写成可检验的假设，明显不可能的直接扔。"
> 出题人看梗概："第 4 幕说他起身**离开**车库，没提骑车——A『骑走了』可能性低，扔；全程没提手机和喝水，不敢确定——B、C 留着立假设。"

**没有这页纸，这活干不了**：光看问题和选项，它根本不知道视频讲什么，"明显不可能"从何判起？但它也**不需要** 180 行流水账——判断"哪个选项离谱"看梗概就够，流水账反而是噪音。

**为什么这张纸换 latent 没意义**——看清谁在跟谁交接：

```
视频 ──工人A(看图)──► 流水账文字 ──工人B(读字)──► 梗概文字 ──► 出题人(读字)
        ▲
        └── "看图变文字"的翻译损失发生在这一步，只发生在这一步
```

工人 B 和出题人**都是只读文字的文员**——视频早在工人 A 那一步就已经变成文字了。把"工人 B → 出题人"这段换成 latent，等于**在两个都没看过视频的文员之间搞脑电波传输**——省几个字的传话损耗而已，跟"看图丢不丢信息"这个我们真正关心的问题无关。（"文员对文员"的 latent 通信 = LatentMAS 的已知领地。）

**验证员调监控那条线不一样**：监控画面（像素）→ 变成文字回给验证员——**这是全流程里唯一一次"看图变文字"发生在答题现场的地方**，翻译损失就在这、还是问题相关的。所以主刀切那里。

**一句话**：摘要 = 开跑前用流水账压出来的一页剧情梗概，复印三份给出题人/判官/选答人；它全程是文字倒手文字，viewer 的翻译损失不在这条线上，所以不动它。

**离线制作流水线（开跑前一次性）**：

```
视频帧 → LaViLa → 180 条逐帧 caption ─┬──────────────→ 验证员的定位地图
                                      └→ dpcknn 聚类切 4 段
                                         → GPT-3.5 每段写一份摘要
                                         → summaries_egoschema_gpt-3.5-turbo-1106_dpcknnsplit_4clips.json
视频帧 → CogAgent → 物体检测摘要 → JSON   ⚠️ 见下：死参数
```

**主循环里的分发（五个阶段签名逐一核过）**：

| 吃什么 | 谁吃 | 怎么进 prompt |
|---|---|---|
| **4 段摘要** | **全部五个结构化阶段**：出题人、判官、两个重生成、选答人 | `read_clips()` 组成 `{"clip 0": 摘要0, …}` 填进 `{action_clip_context}` 槽 |
| **180 条逐帧 caption** | **只有验证员** | `{caption}` 槽（追问之一） |
| **工具细 caption** | 验证员按需 | `role:"tool"` 消息（追问之二） |
| ⚠️ **物体检测摘要** | **没人吃**：形参名带下划线 `_object_detection_summaries`，全文件 grep 无一处引用 | **死参数**——runner 传进去、五个阶段全部原样丢弃 |

**三层粒度的完整图景**（每层对应一种角色需要）：

```
最粗  4 段摘要（≈分 4 幕的剧情梗概）→ 出题人写贴合视频的假设、排除明显不可能的选项
      （prompt 原话 "if the option is clearly inappropriate for the context,
        do not generate a corresponding hypothesis"——没有梗概这步做不了）
中    180 条逐帧一句话 → 验证员定位
最细  按需工具细 caption → 验证员验证
```

**为什么摘要通道在 latent 手术图里判"不动"**：它的内容虽源自 viewer（LaViLa caption），但**进入推理侧的那道边界是文本→文本**（Qwen3-8B 读 caption 文本写摘要）——viewer 边界在 caption 那步已经跨过了。把这条边换 latent = 纯文本 LLM 之间的 latent 通信 = LatentMAS 已知领地（C1 实测对称设定增益 ≈0）。**名册落点**：摘要由 Qwen3-8B 从 Qwen3-VL 重生成的 caption 重新预算（替换 GPT-3.5 版资产）。

⭐ **顺带升级名册第 4 条**：CogAgent 检测摘要原判"正式臂建议砍"——现源码证实它**本来就没被用过**，砍 = 零损失（这也顺手解释了为什么论文里没有它的消融）。

## §3 关键细节与判分坑

| # | 细节 | 位置 | 对我们的意义 |
|---|---|---|---|
| 1 | ⚠️ **单假设直接取 `option_labels[0][0]`**——选项字符串的**首字符**当答案 | [runner.py:192](../../../VideoHV-Agent/VideoHV-Agent/video_hv/pipelines/egoschema_openai/runner.py#L192) | 选项号 ≥10 或格式漂移就错；复现要修或至少计数 |
| 2 | ⚠️ **异常 → random 答案** | [runner.py:196](../../../VideoHV-Agent/VideoHV-Agent/video_hv/pipelines/egoschema_openai/runner.py#L196) | 同 C1 纪律：**解析/异常率必须单独报** |
| 3 | 一轮只有**一条**鉴别线索、**一次**验证调用（不是每假设一次） | [runner.py:114-164](../../../VideoHV-Agent/VideoHV-Agent/video_hv/pipelines/egoschema_openai/runner.py#L114-L164) | 验证预算极省——效率账要如实记 |
| 4 | 验证员的工具轮上限 2、每次 ≤5 帧，超限后**强制"不用工具直接答"** | [verifier.py:107](../../../VideoHV-Agent/VideoHV-Agent/video_hv/verifier.py#L107) / [verifier.py:173-186](../../../VideoHV-Agent/VideoHV-Agent/video_hv/verifier.py#L173-L186) | receiver-steered 感知是**硬预算内**的——正好和我们"按需重看"的效率论证同框 |
| 5 | detection / tracking 工具是**空壳**（`pass`） | [verifier.py:30-35](../../../VideoHV-Agent/VideoHV-Agent/video_hv/verifier.py#L30-L35) | 实际可用工具只有 caption 一个 |
| 6 | 结构化三阶段全走 `client.beta.chat.completions.parse` + 3 个 Pydantic 结果模型 | [openai_stages.py:52](../../../VideoHV-Agent/VideoHV-Agent/video_hv/pipelines/egoschema_openai/openai_stages.py#L52) 等 5 处 | H3 spike 的靶子：8B + vLLM `json_schema` 撑不撑得住 |
| 7 | 并发 3 线程 | [cli.py:67](../../../VideoHV-Agent/VideoHV-Agent/video_hv/pipelines/egoschema_openai/cli.py#L67) | — |

## §4 我们的切口

**B 臂手术位置 = 验证员的工具返回**：caption 工具的返回内容在 [verifier.py:160-167](../../../VideoHV-Agent/VideoHV-Agent/video_hv/verifier.py#L160-L167) 以 `role:"tool"` 消息进对话。手术：**工具改为返回"Qwen3-VL 编码这 ≤5 帧的伪 token"**，在把 messages 送进本地 Qwen3-8B 时于该位置插入伪 token（`inputs_embeds` 拼接，多轮中段注入——比 VideoAgent 的单槽替换复杂一档）。次要切口：`W_prompt` 里的 180 帧 caption 上下文（[verifier.py:84](../../../VideoHV-Agent/VideoHV-Agent/video_hv/verifier.py#L84)）也可整块换伪 token（这是"初始上下文"通道，与"按需工具"通道可分开消融——**这个 host 天然把两条通道分开了，VideoAgent 分不开**）。

**A′ 臂 = 一行**：[vision_tools.py:53](../../../VideoHV-Agent/VideoHV-Agent/video_hv/vision_tools.py#L53) 的通用描述 prompt 里加上问题与当前假设。

## §5 数据资产清单（H4 卡对象）

| 资产 | 状态 |
|---|---|
| EgoSchema：标注 ×2 + LaViLa caption + GPT-3.5 摘要 + CogAgent 检测（58M） | ✅ repo 自带 |
| NextQA：标注 ×2 + CogAgent caption + 摘要 + atphard 划分（12M） | ✅ repo 自带 |
| IntentQA：标注（9.9M） | ✅ repo 自带 |
| **预抽帧图片目录** `/hy-tmp/subset_image/{video_id}/`（[constants.py:10](../../../VideoHV-Agent/VideoHV-Agent/video_hv/pipelines/egoschema_openai/constants.py#L10) 写死，[runner.py:38](../../../VideoHV-Agent/VideoHV-Agent/video_hv/pipelines/egoschema_openai/runner.py#L38) 可传参覆盖） | ❌ 要自己从视频抽 |
| NextQA / IntentQA 的 **pipeline 代码** | ❌ 要仿 egoschema_openai 写（cli 77 行 + prompts 适配，量不大） |

## §6 五臂落点

> 📌 各角色用什么模型以 [00-plan §6.0 模型名册](00-plan.md) 为准（文本侧全 Qwen3-8B / 看图侧全 Qwen3-VL-8B；正式臂 caption 由 Qwen3-VL-8B 统一重生成）。

| 臂 | 实现 | 改动量 |
|---|---|---|
| **A** | 原生：三处 env 指向本地（结构化/验证员/caption 工具）| **零代码**（= H3 spike 本体） |
| **D** | 验证员与结构化引擎换成 Qwen3-VL（只读文字） | env 级 |
| **C** | 验证员改为直接收帧图（OpenAI 多模态消息本来就支持，[verifier.py:47](../../../VideoHV-Agent/VideoHV-Agent/video_hv/verifier.py#L47) 已在 base64 编帧） | 小 |
| **A′** | caption 工具 prompt 带上问题+假设 | **一行** |
| **B** | §4 手术：工具返回 → 伪 token → `inputs_embeds` | 大（自有推理栈） |

## §7 人话 ↔ 源码名对照

| 人话 | 源码 |
|---|---|
| 出题人立假设 | `generate_initial_hypotheses` / `regenerate_hypotheses_after_*` |
| 判官打分 + 鉴别线索 | `judge_hypothesis_distinctness` → `(score, clue, reasons)` |
| 验证员开工 | `external_model_answer`（verifier.py） |
| caption 工具 | `vision_tools.caption`，schema 由 `create_tool_schema` 生成 |
| 选答人 | `select_answer_from_verification` |
| 摘要资产 | `video_summary_bundle["action_caption_summaries" / "object_detections_summaries"]` |
| 区分度门槛 / 轮数上限 / 采样帧数 | `DISTINCTION_SCORE_THRESHOLD=0.5` / `MAX_REFINEMENT_ROUNDS=3` / `NUM_FRAME_SAMPLES=180` |

---

**相关**：[05-videoagent-workflow-source-analysis.md](05-videoagent-workflow-source-analysis.md) · [04](04-host-frameworks-explained.md) · [00-plan §6](00-plan.md)
