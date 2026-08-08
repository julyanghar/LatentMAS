# VideoAgent（wxh1996）工作流源码解剖

> **日期**：2026-08-05 · **repo 已 clone**：`/home/yilin/VideoAgent/`（本文所有行号链接指向它）· 论文 [2403.10517](https://arxiv.org/abs/2403.10517)（ECCV 2024，EgoSchema **54.1** / NExT-QA **71.3**，平均只用 **8.4 / 8.2 帧**，引擎 GPT-4-1106）
>
> **读者起点**：知道我们的五臂实验（[00-plan §3](00-plan.md)）。**其余从零讲。**
>
> **读法**：§1 先认人 · §2 手把手例题 + 逐步表格 · §3 ⭐ 四个改变判断的源码发现 · §4 我们的切口 · §5 数据资产清单 · §6 五臂落点 · §7 人话↔源码名对照。
>
> **与 [04](04-host-frameworks-explained.md) 的分工**：04 是四框架概览与判定；本文是**完整 clone 后的逐行解剖**，行号可点击。04 里与本文冲突处以本文为准（04 已挂修正指针）。

---

## TL;DR

1. 整个系统 = **一个纯文本 GPT-4 大脑 + 一本离线写好的 caption 字典 + 一个预计算特征检索员**。大脑全程一个像素都看不到。
2. 循环最多 3 轮：**看 5 帧的 caption 答题 → 自评信心 → 不够就"写下希望看到的画面"→ 检索 → 补 caption 再答**。
3. ⭐ **仓库是个"录像回放件"，不是"能开火的灶"**：LLM 调用缓存未命中会 `input()` 阻塞、写缓存是空操作、检索读的是预计算特征文件。**本地化要动三处，都很小但必须动**（§3）。

---

## §1 先认人（三个角色）

| 角色 | 是谁 | 输入 → 输出 | 源码钉子 |
|---|---|---|---|
| 🧠 **大脑** | GPT-4-1106（纯文本，**从不碰像素**） | caption 文字 + 问题 → 答案 / 信心 / "我还想看什么"的描述 | [main.py:73](../../../VideoAgent/main.py#L73) `get_llm_response`，默认模型写死在 [main.py:74](../../../VideoAgent/main.py#L74) |
| 👁 **眼睛（离线的）** | LaViLa 模型**事先**给全视频每秒 1 帧写好的 caption 字典 | 帧号 → 一句话（`#C` 开头=戴相机的人的动作，`#O`=别人的动作） | 数据文件 `lavila_subset.json`（Drive 下载）；取用在 [main.py:225](../../../VideoAgent/main.py#L225) `read_caption` |
| 🔎 **检索员** | 预计算的帧特征 + 文本嵌入缓存做点积 | 大脑**希望看到的画面描述** → 段内最相似的真实帧号 | [utils_clip.py:14](../../../VideoAgent/utils_clip.py#L14) `frame_retrieval_seg_ego` |

> 📌 **桥句**：**大脑负责"要什么"，检索员负责"找哪帧"，眼睛的活早就干完了。** 三者只通过文字和帧号交流。

## §2 手把手：一道题怎么走

题目（EgoSchema 风格，视频 180 秒 = 180 帧 caption）：*"贯穿视频，C（戴相机者）主要在完成什么任务？0.修自行车 1.做饭 2.整理花园 …"*

### Step 1：均匀看 5 帧，先答一次（[main.py:242-250](../../../VideoAgent/main.py#L242-L250)）

```
sample_idx = linspace(1,180,5) = [1, 45, 90, 135, 180]
读这 5 帧的 caption →
  frame 1:  "#C C picks up a wrench"
  frame 45: "#C C flips the bicycle upside down" …
→ ask_gpt_caption（CoT 自由文本 + JSON 答案）→ answer = 0
→ self_eval：把【刚才的完整 prompt + 大脑的作答过程】给大脑自己看，
  按三档标准打信心：1=信息不足 / 2=部分 / 3=充分     [main.py:169-186]
```

### Step 2：信心 < 3 才进（[main.py:252-285](../../../VideoAgent/main.py#L252-L285)）

```
① 把视频按已看帧切段：{1: "1-45", 2: "45-90", 3: "90-135", 4: "135-180"}
② generate_description_step：大脑写下**它希望在该段看到的画面**，指定段号——
   {"segment_id":"2", "duration":"45-90", "description":"frame of C adjusting the bicycle chain"}
   ⭐ 这就是"receiver 决定看什么"的具体形态：一句「希望看到的画面」描述
③ frame_retrieval_seg_ego：拿这句话的文本嵌入，只在第 2 段（帧 45..90）的
   预计算帧特征里做点积、argmax → 比如帧 67        [utils_clip.py:22-27]
④ 帧 67 的 caption 加进来 → 重新作答 → 重新自评
```

> 📌 **措辞辨析（2026-08-05 追问回填）——"想象"是它的格式，"希望看到"才是它的功能。** 初版写"大脑想象缺的帧长什么样"，有歧义：像在说大脑**断言**那段视频里有这个画面（幻觉味）。实际机制是——prompt 要求输出**陈述句式的画面描述**（`"frame of xxx"`，原文："For each frame identified as potentially relevant, provide a concise description focusing on essential visual elements"），因为下一步要拿这句话的**文本嵌入去和真实帧特征做点积**，query 必须长得像 caption 才能匹配（写成疑问句"他是否拿了手机？"嵌入会对不上）。**所以这句描述的身份是检索请求，不是对帧内容的断言**：假想的画面若不存在，检索照样返回该段最相近的真实帧——猜错不是错误，是探针（那帧可能恰好反驳这个选项，带一点假设检验味，同 [06 VideoHV](06-videohv-workflow-source-analysis.md) 的显式假设机制）。对我们的意义：这一步正是 (d)「接收方决定 viewer 看什么」的现成形态，B 臂里保留为文字控制通道。

### Step 3：还不行就再来一轮，然后**强制作答**（[main.py:287-317](../../../VideoAgent/main.py#L287-L317)）

与 Step 2 相同，但答完**不再自评**——直接 `generate_final_answer` 收尾。任何一步抛异常也走强制作答兜底。

### 逐步状态表

| 步 | 已看帧集合 | 大脑动作 | 输出 | 信心 |
|---|---|---|---|---|
| 1 | {1,45,90,135,180} | CoT 作答 + 自评 | answer=0 | 2 |
| 2 | +{67}（检索来的） | 写希望看到的画面→检索→重答+自评 | answer=0 | 2 |
| 3 | +{112} | 写希望看到的画面→检索→**强制答** | **answer=0（终）** | — |

**终止条件**：信心=3 提前结束，或走完 3 轮。每题平均落在 **8.4 帧**——这就是论文标题里那个数。

## §3 ⭐ 四个改变判断的源码发现

### 3.1 仓库是"回放件"：不能直接开火

两处证据，都一行：

- **缓存未命中会阻塞**：[main.py:91-92](../../../VideoAgent/main.py#L91-L92) —— `print("Not hit cache", key); input()`。没缓存就停下等人敲回车。
- **写缓存是空操作**：[utils_general.py:8-9](../../../VideoAgent/utils_general.py#L8-L9) —— `def save_to_cache(key, value): return None`。**跑出来的新结果根本不会存。**

→ 作者发布的是"能逐字重放我们那次 GPT-4 实验"的复现件（这对我们**离线核实 72.2% 重调用率**是恩赐），但 **H2 卡（本地跑）必须先动这两刀**：删 `input()` 阻塞、把 save_to_cache 写成真的。

### 3.2 检索员不在线跑 CLIP，读的是两份预计算文件

[utils_clip.py:6](../../../VideoAgent/utils_clip.py#L6) 文本侧读 `cache_clip.pkl`（**描述字符串→嵌入的字典**），[utils_clip.py:15](../../../VideoAgent/utils_clip.py#L15) 帧侧读 `ego_features_448/{video_id}.npy`。

⚠️ **后果**：我们本地跑时大脑（换成 Qwen3-8B）会写出**缓存里没有的新画面描述** → `cache_clip[input]` 直接 KeyError。**H2 必须自配一对"文本编码器+帧特征"**（同一个模型出的才能做点积；EgoVLP/SigLIP/ViCLIP 任选，但两侧必须同源）。帧特征可一次性预计算，结构不变。

### 3.3 判分坑（C1 纪律直接适用）

| 坑 | 位置 | 行为 |
|---|---|---|
| 答案解析失败 | [main.py:54](../../../VideoAgent/main.py#L54) / [main.py:318-320](../../../VideoAgent/main.py#L318-L320) | **random 0–4 混进准确率** |
| 信心解析失败 | [main.py:70](../../../VideoAgent/main.py#L70) | 记 1 → **偏向多跑循环** |

→ 我们的复现必须**单独报解析失败率**（同 C1 的抽取失败率双报），不能让 random 污染臂间对比。

### 3.4 单线程写死

[main.py:350](../../../VideoAgent/main.py#L350) `max_workers=1`。本地化后想提速要自己扩，注意 cache 写入线程安全。

## §4 我们的切口（外科手术位置）

**caption 唯一流进大脑的通道**是 4 个 prompt 函数里的 `{caption}` 槽：

| 函数 | 用途 | 行 |
|---|---|---|
| `ask_gpt_caption` | 第一轮 CoT 作答 | [main.py:189](../../../VideoAgent/main.py#L189) |
| `ask_gpt_caption_step` | 后续轮 CoT 作答 | [main.py:207](../../../VideoAgent/main.py#L207) |
| `generate_final_answer` | 强制收尾 | [main.py:116](../../../VideoAgent/main.py#L116) |
| `generate_description_step` | 写"希望看到的画面"（取帧请求） | [main.py:134](../../../VideoAgent/main.py#L134) |

**B 臂手术**：`read_caption`（[main.py:225](../../../VideoAgent/main.py#L225)）返回的 `{"frame 45": "一句话", ...}` 换成 `{"frame 45": 〈伪 token〉, ...}`；上述四个 prompt 的"问题+指令"部分照常嵌入，caption 位置插伪 token，整条序列用 `inputs_embeds` 喂本地 Qwen3-8B（[04 §7.0](04-host-frameworks-explained.md) 的两档运行方式）。**取帧请求（`generate_description_step` 的输出）保持文字**——审计约束 2 的"细文字控制通道"。

## §4.5 追问回填（2026-08-05）：要不要外部 API · 离线要生成什么

> **追问原话**："如果用 VideoAgent 框架，是否需要外部 API？离线阶段要生成哪些内容？"

**外部 API：原版恰好一个（OpenAI GPT-4 大脑，`main.py:23 client = OpenAI()`，含 `json_object` response_format）；本地化后零个。**

⭐ **检索器"零 API 但也零模型"的实锤**（`utils_clip.py` 全文 30 行）：

```python
cache_clip = pickle.load(open("cache_clip.pkl", "rb"))
def get_embeddings(inputs):
    return [cache_clip[input] for input in inputs]   # ← 纯查表，仓库里没有文本编码器
```

作者把实验中出现过的所有取帧描述的嵌入**预算成字典**——"回放件"的实锤。换脑后新描述必 KeyError → **H2 第三刀（同源编码器对）是必需品不是优化**。

**大脑 JSON 风险比 VideoHV 低一档**：只要 `{"final_answer":"xxx"}` / `{"confidence":"xxx"}` 单层小 JSON，非多级 Pydantic。

**本地化落点（同栈原则下）**：`get_llm_response` 改**进程内 HF 调用**（非 vLLM base_url——B 臂必然 HF `inputs_embeds`，A 臂同栈）。

**离线要生成三件 + 一个在线装配件**：

| # | 生成什么 | 用什么 | 备注 |
|---|---|---|---|
| ① | 抽帧（1fps JPG） | ffmpeg | EgoSchema 原视频要下载；caption 重生成/B 臂 viewer/C 臂都靠它 |
| ② | 全帧 caption | Qwen3-VL-8B（vLLM 批量，资产不受同栈约束） | 替换 LaViLa / nextqa_allcaps；统一 viewer 原则 |
| ③ | 帧特征库（.npy） | 选定编码器**视觉塔** | 同源分叉：(a) 原版 ego_features+EgoVLP 文本塔 / (b) **全自算 SigLIP 或 ViCLIP 对（推荐——不依赖 Drive、两库统一）** |
| — | ~~摘要~~ / ~~聚类分段~~ | — | **VideoAgent 无摘要通道**；段边界运行时现定义 |
| ⊕ | **在线装配件**：③那对的**文本塔** | 运行时现算大脑新写的取帧描述嵌入 | H2 第三刀 |

**⊕ 两座塔到底用在哪一行（追问回填）**：只有一处——`frame_retrieval_seg_ego`（utils_clip.py）把大脑的句子变帧号：

```python
frame_embeddings = np.load(f"ego_features_448/{video_id}.npy")
#  ▲ 视觉塔在这——活【离线】干完了，运行时只 load 产物（180 帧×每帧一向量）
text_embedding = get_embeddings([description])   # = cache_clip[句子]
#  ▲ 文本塔在这——被查表替身挡住（作者预算了实验里出现过的所有句子）
seg_similarity = text_embedding[idx] @ seg_frame_embeddings.T   # 相似度 = 这一行
```

**两座塔全程不答题——它们是"把大脑一句话翻译成一个帧号"的搜索引擎零件。** 在仓库里视觉塔只剩产物（.npy）、文本塔只剩替身（cache_clip.pkl），**塔本体都不在**——"回放件"最直观的一面：能重放旧实验，无法处理任何一句新话。"同源"落到实处 = 那行点积的左右手必须活在同一坐标系。

**⊕ 的三点澄清（追问回填）**：(1) **必须是 CLIP 家族双塔跨模态模型**（文本↔图像检索），普通文本 embedding 模型（BGE/Qwen3-Embedding）干不了；**Qwen3-VL 也干不了**——生成式 VLM 没有现成的对齐嵌入空间。所以它是名册外的**第三个模型**（几百 M 级）。(2) **同源铁律**：文本塔与视觉塔必须**同一次训练里一起训出来的一对**——CLIP 式对比学习靠"配对图文拉近"把两塔逼进同一坐标系，点积当相似度用完全靠这场共同训练；两个独立训练的模型坐标系毫无对应，**混搭的点积不报错但是噪声 → 检索"看起来在工作"实际返回随机帧（静默失败，比崩溃难查一个量级）**。所以换文本塔必须连帧特征一起重算，换一侧不换另一侧=静默灾难。📌 彩蛋：这就是本项目主病（viewer↔reasoner 空间不对齐）的微缩版——检索器的药是"选天生同源的一对"（CLIP 家满地都是），latent 通道没有天生同源可选 → 只能训 adapter（P5 存在的理由）。(3) ⭐ **实验控制点：检索器在循环骨架里、不在被换的通道上——五臂共用同一个**，选型不影响臂间对比，H1/H2 定一次全臂统一。

**双 host 镜像差异**：VideoHV 定位=LLM 读 180 条地图（零编码器、每次付 ≈1.7k token）；VideoAgent 定位=嵌入检索（装编码器对、大脑只见 5~8 条 caption）。**离线件本框架少一件（无摘要），在线装配件多一个（编码器对）。**

---

## §5 数据资产清单（H1 卡的核查对象）

| 资产 | 在哪 | 状态 |
|---|---|---|
| NExT-QA 全量 1fps caption | `nextqa_allcaps_1fps.zip` | ✅ **repo 自带** |
| LLM 回放缓存 / 文本嵌入缓存 | `cache_llm.pkl` / `cache_clip.pkl` | ✅ repo 自带（**仅回放用**） |
| EgoSchema 标注 `subset_anno.json` | [Google Drive](https://drive.google.com/drive/folders/1ZNty_n_8Jp8lObudbckkObHnYCvakgvY)（README 给的） | ❌ 要下 |
| EgoSchema caption `lavila_subset.json` | 同上 | ❌ 要下 |
| 帧特征 `ego_features_448/*.npy` | 同上 | ❌ 要下（或自算） |
| 原始视频/帧（C 臂、A′ 臂要用） | EgoSchema / NExT-QA 官方渠道 | ❌ 要下 |

## §6 五臂落点

> 📌 各角色用什么模型以 [00-plan §6.0 模型名册](00-plan.md) 为准（文本侧全 Qwen3-8B / 看图侧全 Qwen3-VL-8B；正式臂 caption 由 Qwen3-VL-8B 统一重生成，LaViLa 仅复现层用）。

| 臂 | 在这个 host 上怎么实现 | 改动量 |
|---|---|---|
| **A** | 原生流程，大脑换本地 Qwen3-8B（`base_url`）+ §3.1 两刀 + §3.2 编码器对 | 小 |
| **D** | 同 A，大脑换 Qwen3-VL-8B（只读文字） | 极小 |
| **C** | 4 个 prompt 函数的 `{caption}` 换成**同帧号的原图**，走 vLLM 多模态消息 | 中 |
| **A′** | 用 Qwen3-VL **重生成** caption，prompt 里带上问题 | 中（要原始帧） |
| **B** | §4 手术：caption 槽 → 伪 token → `inputs_embeds` | 大（自有推理栈） |

## §7 人话 ↔ 源码名对照

| 人话 | 源码 |
|---|---|
| 大脑作答（第一轮/后续轮） | `ask_gpt_caption` / `ask_gpt_caption_step` |
| 自评信心（1/2/3） | `self_eval` + `parse_text_find_confidence` |
| "我希望看到的画面"（取帧请求） | `generate_description_step` 输出的 `frame_descriptions` |
| 分段检索 | `frame_retrieval_seg_ego`（段内 argmax） |
| caption 字典取用 | `read_caption` |
| 强制收尾 | `generate_final_answer` |
| 回放缓存 | `cache_llm.pkl` + `get_from_cache` |

---

**相关**：[06-videohv-workflow-source-analysis.md](06-videohv-workflow-source-analysis.md)（另一个 host）· [04](04-host-frameworks-explained.md) · [00-plan §6](00-plan.md)
