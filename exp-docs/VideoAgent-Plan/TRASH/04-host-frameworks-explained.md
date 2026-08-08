# 四个候选 Host 框架，分别是怎么设计的

> **缘起**：[00-plan.md §6](00-plan.md) 的 Host 选型表只给了判定没讲设计。这份从零讲清每个框架**怎么干活**，以及判定为什么是那样。
>
> **读者起点**：知道我们的 idea（viewer 看帧 → latent → 纯文本 reasoner → 循环取帧）和五臂实验（[00-plan §3.1](00-plan.md)）。**其余术语从零讲。**
>
> **证据来源**：⭐ **五个框架的源码都有本地拷贝，本文的流程描述是我逐文件读代码核出来的**，不是转述论文。注意坑：本地有**两个同名 VideoAgent**——`va_inference.py` 是记忆增强版（ViCLIP+SQL+ReAct 那篇），**真正的 wxh1996 版是 `wx.py`**，别混。源码拷贝在会话 scratchpad（临时目录，会清），持久引用请走 GitHub 链接。
>
> **读法**：§0 先说清"我们拿 host 干什么"→ §1–§5 逐框架（**§1 最详细，因为是首选**）→ §6 判定表回收 · §7 改造点 · §8 术语表。

> ⚠️ **2026-08-05 升级**：两个入选 host 已完整 clone（`/home/yilin/VideoAgent`、`/home/yilin/VideoHV-Agent`）并逐行解剖成独立文档——**[05（VideoAgent）](05-videoagent-workflow-source-analysis.md)** · **[06（VideoHV）](06-videohv-workflow-source-analysis.md)**。本文基于 scratchpad 局部副本写成，**与 05/06 冲突处以 05/06 为准**（已知冲突：VideoAgent 检索是预计算特征非在线 ViCLIP；VideoHV 验证员默认本地 Qwen 而非 OpenAI；VideoHV 三库数据资产随 repo 带齐）。

---

## TL;DR

| 框架 | 一句话设计 | reasoner 纯文本？ | 可本地化？ | 判定 |
|---|---|---|---|---|
| ⭐ **VideoAgent**（wxh1996，ECCV'24） | 读 caption → 自评信心 → 不够就"说出还想看什么"→ 检索加帧 → 再答，最多 3 轮 | ✅ GPT-4-1106，全程不碰像素 | ⚠️ 大脑闭源但**换 Qwen3-8B 是一行事** | **首选** |
| **VideoHV-Agent**（Haorane） | 把每个选项改写成**可检验假设**，派验证员按需调 caption 工具去找证据 | ✅ 全部角色都是文本 LLM，只有 caption 工具碰图 | ❌ **100% OpenAI API，连 torch 都不 import** | 结构完美但要整体重写 |
| **VideoChat-A1**（AAAI'26） | **一个 VLM** 自己干全部：聚类分镜 → 选镜头 → 看 → 不够就反思、细分镜头再看 | ❌ viewer=reasoner 同一个本地 VLM | ✅ **零改动能跑** | 只能当冒烟台 |
| **LVAgent**（ICCV'25） | 多个 VLM 评审团：各自看帧答题 → 互相打分 → **踢掉最低分** → 再看再讨论 | ❌ 全是 VLM | ✅ 全开源本地 | 不满足理想态 |
| **A4VL** | 多 VLM 并行线程：各写"感知线索"选自己的帧 → 答题 → 交叉评审 → 剪枝 | ❌ 全是 VLM | ✅ 全开源本地 | 不满足理想态 |

---

## §0 Host 框架是什么，我们拿它干什么

我们的论文形态是"**在已有的视频 agentic workflow 上，把 viewer→reasoner 那条文字边换成 latent 通道**"。**Host = 被我们换边的那个 workflow。** 所以选型只看三条：

1. **reasoner 是不是纯文本的**——理想态要求 reasoner 没有视觉塔（这样"文字边丢信息"才是真实存在的边，latent 才有的换）；
2. **能不能本地开源化**——我们要摸到两个模型的内部（KV、嵌入层），API 模型摸不到；
3. **循环是不是真的**——"接收方决定 viewer 下一步看什么"（我们全部的新颖性 (d)）得在这个框架里**本来就存在**，不能是我们硬造的。

**贯穿全文的例题**（EgoSchema 风格的长视频题）：

> 一段 3 分钟第一视角视频。**问：这个人修好自行车之后，接下来做了什么？A) 骑走了 B) 收拾工具 C) 打电话 D) 喝水**

---

## §1 ⭐ VideoAgent（wxh1996）——"看不清就说出来还想看什么"

**出处**：[VideoAgent: Long-form Video Understanding with LLM as Agent](https://arxiv.org/abs/2403.10517)（ECCV 2024）· [GitHub: wxh1996/VideoAgent](https://github.com/wxh1996/VideoAgent) · 本地源码 `wx.py`（约 340 行，我全文读过）

### 1.1 先认人（三个角色）

| 角色 | 是谁 | 干什么 |
|---|---|---|
| 🔤 **captioner** | LaViLa（视频 caption 模型） | ⚠️ **离线**把全视频**每一秒一帧、每帧写一句话**（平均 7.2 词），存成 JSON。**循环开始前就全部写完了** |
| 🧠 **reasoner** | **GPT-4-1106-preview，纯文本** | 读 caption 答题、自评信心、说"我还想看什么" |
| 🔍 **检索器** | EgoVLP 类的文本-视频嵌入模型 | 把 reasoner 的"想看什么"变成具体帧号 |

> ⚠️ **一个读论文看不出来、读代码才发现的关键点**：main 函数第一件事就是 `all_caps = json.load(open("lavila_subset.json"))`——**全部 caption 是离线预算好的**。循环里所谓"再看一眼"，**实际是"从这本预算好的字典里多取几行"**，captioner 从不在线运行。这直接影响我们的改造（见 §7）。

### 1.2 手把手：例题怎么走

```
### Step 1 ###
均匀取 5 帧：np.linspace(1, num_frames, num=5)   → 比如帧 [1, 45, 90, 135, 180]
读这 5 帧的 caption：
    "frame 1: 一个人蹲在自行车旁"
    "frame 45: 手在转动脚踏板"
    "frame 90: 把扳手放进包里"        ← 其实答案线索在这
    "frame 135: 人站着"
    "frame 180: 人在门口"
GPT-4 答题 → "B) 收拾工具"
GPT-4 自评（self_eval）："对刚才的决策过程打个信心分 1/2/3"
    → 假设它答 2（"部分够"）

### Step 2 ###（信心 < 3 才进）
GPT-4 写下「希望看到的画面」（generate_description_step）：
    对认为相关的帧间隔各写一句陈述句式的画面描述（格式强制 "frame of xxx"）
    → 比如 {"segment_id":"3", "duration":"90-135", "description":"frame of the person picking up a phone"}
    ⚠️ 必须是陈述句不是疑问句——这句话下一步要当检索 query 去和帧嵌入做点积
      （详见 [05 的措辞辨析](05-videoagent-workflow-source-analysis.md)：格式是假想画面，功能是检索请求）
检索器把这些描述逐句和帧嵌入做匹配（frame_retrieval_seg_ego）→ 返回帧号 [102, 117]
帧集合并入排序 → [1, 45, 90, 102, 117, 135, 180]
读这 7 帧的 caption → GPT-4 再答 → 再自评
    → 信心 3，输出 B，结束

### Step 3 ###（还不够就再来一次，然后强制作答）
```

**硬编码细节**（都从 `wx.py` 核的）：最多 **3 轮**（Step 1/2/3 是三段写死的代码，不是 while 循环）；信心分从 JSON `{"confidence": "xxx"}` 里解析；答案解析失败就 **random.randint(0,4) 瞎猜**；异常时直接强制作答。

### 1.3 为什么是首选

| 判据 | 情况 |
|---|---|
| **reasoner 纯文本** | ✅ GPT-4-1106 从头到尾只见过 caption 字符串，一个像素没碰 |
| **循环是真的** | ✅ 由信心分驱动，【实测】**72.2% 的题至少触发一次 Step 2**、73.3% 的自评说"报告不够用"——不是摆设 |
| **本地化工作量** | `get_llm_response()` 一个函数换成 Qwen3-8B 调用；captioner 换 Qwen3-VL 重算 caption 文件。**结构一行不用动** |
| **效果背书** | 论文：EgoSchema 54.1 / NExT-QA 71.3，平均只用 **8.4 帧** |
| **就是我们的 A 臂** | 五臂实验的 A 臂（现状）**就是这个循环原样**，B 臂只换 ②③ 两步 |

---

## §2 VideoHV-Agent——"把选项变成假设，派人去证伪"

**出处**：[GitHub: Haorane/VideoHV-Agent](https://github.com/Haorane/VideoHV-Agent) · 本地源码 `vhv/`（README + pipelines 全读过）

### 2.1 设计思想：不是"看了再答"，是"先立假设再找证据"

README 自述一句话：**hypothesis generation → distinction judging → tool-based verification → answer selection**。四个角色全是**文本 LLM 的 API 调用**：

| 角色 | 干什么（源码函数名） |
|---|---|
| **出题人** | `generate_initial_hypotheses`：把每个选项改写成**可检验假设**——必须写明实体、动作、时序关系；明显不靠谱的选项直接不立假设 |
| **判官** | `judge_hypothesis_distinctness`：这几条假设**互相区分得开吗**？分不开就打回重写（`regenerate_after_low_distinction`） |
| **验证员** | `external_model_answer`：一个带工具的循环——LLM 可以调 **caption 工具**（"给我 37–41 帧的描述"，**每次限 ≤5 帧**），拿到描述后判断每条假设"证实/证伪/证据不足"；验证失败也打回重立假设（`regenerate_after_failed_verification`） |
| **选答人** | `select_answer_from_verification`：读全部验证记录，选幸存的假设 |

**例题怎么走**：四个选项被改写成四条假设（"该人在修车后骑车离开了现场"…）→ 判官说 A 和 D 太像，打回重写 → 验证员调 caption 工具看 85–90 帧："手把工具放进包"→ 假设 B 证实、A 证伪 → 选 B。

### 2.2 为什么"结构完美但要整体重写"

**结构完美**：它是五个框架里 **P4 味道最浓**的——验证员**主动、按需、带着明确目的**去调感知工具（"我要验证假设 X，给我看 Y 段"），这正是我们 (d) 要的"接收方决定看什么"，而且**假设驱动**比 VideoAgent 的"缺什么补什么"更有方向性。

**⚠️ "要整体重写"这个判断 2026-08-05 修正为"中等适配 + 一个未验证的脆弱性"**：源码从头到尾是 OpenAI SDK 调用、`requirements` 连 torch 都没有——**但这反而意味着换 `base_url` 指向本地 vLLM 的 OpenAI 兼容服务就能通**（Qwen3-8B 管四个角色、Qwen3-VL-8B 管 caption 工具；vLLM 新版支持 `json_schema` 结构化输出）。**真正的风险不是工程量**，是它全程 `client.beta.chat.completions.parse` + 4 个 Pydantic schema 的**多级结构化流程在 8B 模型上的存活率未验证**（C1 实测过 8B 对格式指令 83.3% 不遵循）——**必须先跑半天 spike**（20 题冒烟、量格式存活率）再承诺。

⭐ **两个源码核出的实验红利**（2026-08-05）：
1. **caption 工具的 prompt 是通用描述**（"describe each image in detail... people, objects, actions, and scenes"），**不带问题、不带假设**——它的问题条件化只发生在**选帧**。→ **A′ 臂（问题条件化 caption）在它上面是一行 prompt 改动**，比 VideoAgent（要重生成整套离线 caption）便宜得多。
2. 因此双 host 恰好把文字损失拆成"选帧"与"caption 内容"两个成分，两边跑 A′ 可定位损失来源——单 host 做不到。

⚠️ **再补两个源码事实（2026-08-05 二轮核查）**：
1. **仓库只有 `pipelines/egoschema_openai/` 一条 pipeline**——论文声称的 NextQA/IntentQA **代码不在仓库**，多 benchmark 要自己写。
2. **它是混合制不是纯按需**——⚠️ **2026-08-05 修正谁吃什么**（runner.py 核）：出题人/判官吃的是 **GPT-3.5 预算的 4 段摘要 + CogAgent 物体检测摘要**（不是逐帧 caption）；**LaViLa 逐帧 caption 全量 180 条给了验证员当"定位地图"**（每次调用 ≈1.7k token 固定开销），验证员再按需调 caption 工具拿现场细 caption（两层机制详见 [06 追问回填](06-videohv-workflow-source-analysis.md)）；帧路径写死 `/hy-tmp/subset_image/{video_id}`（需预抽帧）。

**新分工（[00-plan §6.1](00-plan.md)，核实状态见 [§6.3](00-plan.md)、预检卡见 [§6.4](00-plan.md)）**：VideoAgent = 判死台（先跑）；**VideoHV = 主打 host**（最强文字基线 + P4 最浓，待 spike 通过后承诺）。

---

## §3 VideoChat-A1——"一个人自己分镜、自己挑、自己看"

**出处**：[VideoChat-A1: Thinking with Long Videos by Chain-of-Shot](https://github.com/SpXace/VideoChat-A1)（**AAAI 2026 已录用**）· 本地源码 `va1/`

### 3.1 设计思想：长视频是由镜头（shot）组成的

它的批评对象正是 VideoAgent 这类："现有 agent 忽略了长视频由多个 shot 组成这个事实，经常检索回冗余甚至噪声的时间上下文。"

**流程**（`run_longvideobench.py` 核出）：

```
① 分镜：全视频帧特征 → k-means 聚类（first_cluster）→ 按聚类边界切成 shots
② 选镜头：CLIP4Clip 式选择器给每个 shot 对问题打分 → 取最相关的 shot
③ 看：本地 VLM（Qwen2.5-VL 系）拿这个 shot 的 32 帧直接回答
④ 够不够：check_if_sufficient —— 够就出答案
⑤ 不够：反思（"解释你刚才为什么选这个答案"）+ 明确说缺什么
   （update_info 原文提示词："determine what NEW visual information
     or missing evidence is required"）
⑥ 把选中的 shot 再细分（coarse-to-fine），回 ② 在更细的粒度上重选重看
```

**卖点**：VideoMME(w/subs) 77.0、EgoSchema 70.1，**只用 7% 的输入帧、12% 的推理时间**就逼近 GPT-4o。

### 3.2 为什么"只能当冒烟台"

**看第 ③ 步就明白**：看视频的和做推理的是**同一个本地 VLM**。**它根本没有"viewer→reasoner"那条跨模型的文字边**——我们想换的边在这个框架里不存在。

但它有三样别人没有的：**零改动能跑**（纯 HF、单模型、轮次串行）、**摸得到 `past_key_values`**、循环结构真实（⑤⑥ 是货真价实的"说缺什么→再看"）。→ **当管道冒烟台**：跨轮 KV 复用、M-RoPE tracker、latent 注入的代码先在它身上调通，再搬去 VideoAgent。

---

## §4 LVAgent——"评审团吵架，谁没道理谁出局"

**出处**：[LVAgent](https://arxiv.org/abs/2503.10200)（ICCV 2025）· 本地源码 `lv_agent.py` + `lv_disc.py`

### 4.1 设计（从源码核的）

**评审团 = 几个异构 VLM**：源码里实例化的是 InternVL-8B、InternVL-78B、LLaVA-Video-72B（类里还备着 Qwen-72B）。

```
第 1 轮：每个 VLM 各自均匀看 16 帧 → 独立作答 + 写理由
交换：把所有人的〈答案+理由〉发给每个人
互评：每人给自己和别人的理由打 1–10 分（discuss_text_process）
淘汰：总分最低者出局——源码原话拼进历史里：
      "However, this reason was deemed unconvincing,
       so this answer was removed from the discussion."
提炼：幸存者各自总结"回答这题还需要什么关键信息"（history_info）
第 2+ 轮：带着历史信息，换不同的帧块再看（get_frame_idx_path 按轮次
      平移采样窗口）→ 再答 → 再评 → 直到多数一致或轮数用完
```

**例题怎么走**：InternVL-8B 说 A（理由含糊）、InternVL-78B 说 B（理由引用了"帧 90 手放工具进包"）、LLaVA-72B 说 B → 互评后 8B 得分最低出局 → 幸存两人多数一致 → B。

### 4.2 为什么不满足理想态

**全是 VLM、人人都看得见帧、传的只是意见**——这是 [Q&A/08](../Q&A/08-mas-topologies-and-research-route.md) 分类里的**对称设定**（辩论/投票那一族）。我们的 C1 已经量过：**对称设定下 latent 相对文字的增益 ≈ 0**（+1.2/−4.7/+9.9/−0.5）。在它身上换通道，等于重跑一遍已知答案的实验。

---

## §5 A4VL——"并行流水线 + 感知线索选帧 + 交叉评审"

**出处**：arXiv 2603.14052【调研，未逐字核对论文】· 本地源码 `a4vl_pr.py`（819 行，结构核过）

**和 LVAgent 是近亲**（同样多 VLM + 互评 + 剪枝），三处不同（源码核的）：

1. **并行**：每个 agent 一个 Python 线程（`threading.Thread`），同时跑，不是轮流发言；
2. ⭐ **感知线索选帧**：每个 agent 先写一句 **perception clue**（"需要看修车结束后手部动作的片段"），一个**微调过的 CLIP4Clip 选择器**（`AspClipSelector`，checkpoint `pytorch_model_0.0011.bin`）拿这句话给帧打分，**每个 agent 看到的帧是自己的线索选出来的**——不是均匀采样；
3. **剪枝逻辑更成型**：`_run_cross_review` 交叉打分 → `_prune_agent_pool` 按总分踢人（带优先级平票规则）→ `_majority_if_any` 查多数。

**判定同 LVAgent**：全 VLM 对称设定，不满足"reasoner 纯文本"。但它的 **perception-clue→选帧**机制值得记一笔——那是"用一句话操纵感知"的现成实现，跟我们 B 臂里"reasoner 的取帧请求"是同一个形状。

---

## §6 判定表回收：为什么这么排

| | VideoAgent | VideoHV | VideoChat-A1 | LVAgent / A4VL |
|---|---|---|---|---|
| 有没有我们要换的**跨模型文字边** | ✅ caption 边，且是唯一信息通道 | ✅ caption 工具返回边 | ❌ **不存在**（单模型） | ⚠️ 有边但两端都看得见帧（对称） |
| reasoner 纯文本 | ✅ | ✅ | ❌ | ❌ |
| 循环真实（(d) 存在） | ✅ 72.2% 触发 | ✅ 假设驱动，最浓 | ✅ 但单模型 | ⚠️ 有轮次但驱动力是"分歧"不是"缺信息" |
| 本地化成本 | **低**（换两个模型） | **高**（整体重写） | **零** | 零（但没意义） |
| **角色** | ⭐ **主 host（A/B 臂宿主）** | 第二 host 候选 | **管道冒烟台** | 不用（其对称设定已被 C1 判死） |

---

## §7 对我们改造的具体含义（以 VideoAgent 为例）

### §7.0 先回答一个根本问题：host 都是 caption 媒介，我们不是要做 latent 吗？（2026-08-05 用户追问回填）

**对，所有 host 都用 caption 交流——而且必须如此，这是入选条件不是缺陷。** 三层关系：

1. **世界上不存在自带 latent collaboration 的现成框架**（[01 先例审计](01-prior-art-audit.md) 的结论：那个格子是空的）。"找一个自带 latent 的 host"这个选项从来不存在。
2. **host 提供三样东西，latent 不在其中**：骨架（循环/选帧/角色/benchmark，所有臂共用不动）· 文字基线臂（A/D/A′ 原生跑）· **一条清晰的 caption 边（给我们切的）**。没有 caption 边就没东西可换、没有对照——**选 host 筛的就是这条边切口干不干净**。
3. **latent 是我们做的手术，只切那一条边**：A 臂 = viewer 写 caption（文字）拼进 reasoner prompt；B 臂 = viewer 只编码产 KV → adapter → 伪 token → **直接插进 reasoner 的输入嵌入序列**。其余（循环、自评、取帧请求、检索）保持文字、保持不动。

⚠️ **由此一个必须写明的工程事实：B 臂走不了 chat API。** 伪 token 是嵌入层对象，"文字进文字出"的接口塞不进去。五臂分两档：

| 臂 | 怎么跑 |
|---|---|
| A / D / A′ / C | ✅ API 式：`base_url` → 本地 vLLM |
| **B** | ❌ **reasoner 调用必须换成自有推理栈**：问题文字嵌入 + 伪 token 拼成嵌入序列，`inputs_embeds` 喂入（HF 最直接；vLLM prompt-embeds 支持【待查】；viewer 侧抽 KV 是 LMCache 本行） |

**两个 host 的切口位置**（工作量相同，不影响 host 排序）：VideoAgent = `wx.py` 里 `{caption}` 拼进 prompt 的槽（函数边界干净）；VideoHV = verifier 的 caption 工具**返回值进对话**的位置（多轮中段注入，稍复杂）。

---

六臂实验（[00-plan §3.1](00-plan.md)）全部臂共用本文 §1.2 那个循环，只动两处：

| 循环环节 | A 臂（现状） | B 臂（我们的） |
|---|---|---|
| ② captioner | ⚠️ **离线 LaViLa 全帧 caption 字典** | ⭐ **在线 Qwen3-VL 只编码被选中的帧，不解码文字** → adapter → 伪 token |
| ③ reasoner 读什么 | caption 字符串 | 伪 token + 问题（文字） |
| ④ 自评+取帧请求 | 文字（保留） | **文字（保留！）**——这就是审计 §8 约束 2 说的"细文字控制通道当语义锚" |
| 检索器 | 文本→帧嵌入匹配 | 保留不动 |

> ⚠️ **§1.1 那个"caption 全是离线预算好的"发现在这里兑现成两件事**：
> 1. **A 臂的 caption 是问题无关的**（写 caption 时不知道会被问什么）——这正是 A′ 先知臂要量的那个缺口的来源；
> 2. **B 臂把"离线全帧"改成"在线按需"，本身就改变了成本结构**——效率账（§7.2）里必须把"LaViLa 离线跑全视频每秒一帧"的成本算进 A 臂，不能只算 GPT-4 的调用。

---

## §8 术语表

| 词 | 意思 |
|---|---|
| **Host 框架** | 被我们换通道的现成 agentic workflow；我们的方法以"插件"形式活在它身上 |
| **EgoSchema / NExT-QA / LongVideoBench / VideoMME** | 长视频问答基准；EgoSchema 是 3 分钟第一视角视频 |
| **LaViLa** | 视频 caption 模型，VideoAgent 的原配 captioner，平均每句 7.2 词 |
| **self_eval / confidence** | reasoner 答完后给自己的决策过程打 1/2/3 分；<3 触发下一轮 |
| **EgoVLP / CLIP4Clip** | 文本-视频对齐嵌入模型，用来"拿一句话找到最匹配的帧/镜头" |
| **shot（镜头）** | VideoChat-A1 的核心单位：视频里视觉上连续的一段；用帧特征聚类切出来 |
| **coarse-to-fine** | 先在粗粒度选镜头，不够就把选中的镜头再细分、更细地重选 |
| **假设检验式（VideoHV）** | 把每个选项改写成可证实/证伪的命题，去视频里找证据，而不是开放式地"理解视频" |
| **交叉评审 / 剪枝** | 多 agent 互相给理由打分，总分最低者的答案被移出讨论 |
| **对称 / 不对称设定** | 对称=所有 agent 看得见同样的输入，传的是意见；不对称=一端看过像素另一端没有（详见 [Q&A/08 §2](../Q&A/08-mas-topologies-and-research-route.md)） |

---

## 相关文档

- [00-plan.md](00-plan.md) —— §3 五臂实验 · §6 Host 选型表 · §7 分阶段计划
- [Q&A/08](../Q&A/08-mas-topologies-and-research-route.md) —— 五种拓扑与对称/不对称之辨
- [Q&A/05](../Q&A/05-credible-mas-screening-and-graft-plan.md) —— §7.3 graft 排名（VideoChat-A1 为什么零改动能跑）
- [03-eight-risk-papers-explained.md](03-eight-risk-papers-explained.md) —— SportMV-Agent（另一个占了 workflow 形状的框架）
