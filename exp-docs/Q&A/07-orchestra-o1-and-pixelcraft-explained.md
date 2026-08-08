# 两个"最接近通关"的多模态 MAS 是怎么设计的 —— Orchestra-o1 与 PixelCraft 拆解

> **这份文档是什么**：[Q&A/05 §4.3](05-credible-mas-screening-and-graft-plan.md) 里挑出的两个"最接近通关"的系统，把它们的框架设计讲透。
>
> **读者起点**：知道"多个 AI 一起干活"是怎么回事。不需要读过这两篇论文。
>
> **读法**：§1 讲两者的根本差别（先读这个，后面才好懂）· §2 Orchestra-o1 · §3 PixelCraft · §4 ⚠️ **代码与论文对不上的地方**（本文最有价值的一节）· §5 我们能拿来用的零件 · §6 术语表。
>
> **方法**：6 个 agent 并行深读——每篇一路读论文、一路读仓库源码，外加 2 条对抗核验。**凡是标【核实】的都读了原文或源码**，仓库全部 `curl` 验活。

---

## TL;DR

1. **两者的协作轴是垂直的**：Orchestra-o1 **横着切任务**（并行派活给多个子智能体），PixelCraft **竖着磨一张图**（六个角色轮流加工同一张图）。
2. ⚠️ **Orchestra-o1 的招牌算法 DA-GRPO 在代码里不存在** —— `grep` 整个仓库只在一张图的标题里出现过一次，实际跑的是**标准 verl GRPO**。所谓创新退化成"奖励函数里给评委看了专家的答案"。
3. ⚠️ **PixelCraft 的招牌"图像记忆"在代码里是一个裸 dict**：`image_pool = {"original_image": path, "processed_image": []}` —— 没有类、没有索引、没有键，只是一个按时间顺序 append 的路径列表。

---

## §1 先分清两种协作轴

| | **Orchestra-o1** | **PixelCraft** |
|---|---|---|
| 协作方向 | **横向**：把任务切成几块，同时派给几个人 | **纵向**：一个任务上反复加工图像，多角色轮流把关 |
| 角色怎么来的 | **运行时现写**（指令 + 模型 + 工具，三个字段填出来） | **静态六角色**，每个一个文件 |
| 用几个模型 | ⭐ 每个子任务可以指定不同模型 | ⭐ **所有推理角色共用一个** |
| 通道里传什么 | 子智能体的**轨迹文字** | **文字 + 真实图像 crop** |
| 任务类型 | 全模态（音频+视频+图像），大海捞针式检索 | 图表 / 几何图，细节密集 |
| **能不能隔离"多请人"** | ❌ 做不到——模型和角色一起变 | ✅ **全场最好** |

> **桥句**：**Orchestra-o1 解决"人手不够"，PixelCraft 解决"看不清"。**

---

## §2 Orchestra-o1：把"招人"做成一个工具调用

**论文**：[arXiv:2606.13707](https://arxiv.org/abs/2606.13707)v1，2026-06-10，CUHK/LIGHTSPEED/PKU/THU/Tongji
**代码**：[zfkarl/Orchestra-o1](https://github.com/zfkarl/Orchestra-o1) ✅200，⚠️ **默认分支是 `master` 不是 `main`**（查 `main` 会 404，我第一次就踩了）· 70★ · 2026-06-15 最后 push · **无 LICENSE 文件**（README 挂了 MIT 徽章但链接是死的）
**权重**：[Karl28/Orchestra-o1-8B](https://huggingface.co/Karl28/Orchestra-o1-8B) ✅200，就是普通 Qwen3-8B（36 层/4096/32 头/8 KV 头）
**数据**：[RUC-NLPIR/OmniGAIA](https://huggingface.co/datasets/RUC-NLPIR/OmniGAIA) ✅200，**360 题**

### 2.1 一句话

> 一个**纯文本**的"经理"读题 + 读之前助手交回来的文字汇报，然后吐出**一个 JSON**，里面列着几件可以同时干的活——每件活自带自己的指令、自己的模型、自己的可用工具——系统把它们全部并发发出去，再把结果的文字贴回记录本，最多重复 10 轮。

### 2.2 谁在场（【核实】读源码）

| 角色 | 干什么 | **看得到什么** | 用什么模型 |
|---|---|---|---|
| **MainAgent** | 每轮读题 + 历史 → 吐一个 JSON：要么 `delegate_task`，要么 `complete` | ⚠️ **纯文本！** 媒体只以**路径字符串**出现（`prompts/omnigaia.py:62`）。**一个像素、一段波形都收不到** | GPT-5（主表）/ Orchestra-o1-8B（开源臂） |
| **SubAgent** ×K | 标准 ReAct 循环，最多 30 步 | 自己的任务指令 + 经理手写的 context + **被过滤过的工具子集** | 经理在 `model` 字段里点名。⚠️ 但配置里只暴露一个选项：`sub_models: ["gpt-5"]` |
| **感知工具** | **把媒体变成文字** ← 所有的"看"和"听"都发生在这里 | 文件路径 + 一句自然语言问题 | 硬编码前沿 API：图像 `gpt-5`、音频 `gpt-4o-audio-preview`、视频 `gpt-5` |
| **动作工具** | 搜索 / 抓网页 / 跑代码 | — | 无 LLM（Serper / Jina / 沙箱 Python） |
| ⚠️ **轨迹压缩器** | **每个子任务跑完，先把完整轨迹压成 5-10 条要点，经理才能看** | 子智能体的完整轨迹 | ⚠️ **硬编码 `gpt-5`**（`delegate.py:306-308`）——**即使用 Qwen3-8B 当经理，每个子任务也要付一次 GPT-5** |
| 评委（评测用） | 判最终答案对不对 | 问题/标答/预测 | `gpt-4o` |

> **注意最后两行**：论文的角色列表里**没有**轨迹压缩器，但它是 Eq.14 的 Ω(·)，且**是文字瓶颈的实际位置**。

### 2.3 控制流（【核实】逐行读的）

```
入口: python bench_orchestra_o1_omnigaia.py --config config/benchmarks/orchestra_o1_omnigaia.yaml

外层循环 (omnigaia_runner.py:222-239)
for attempt_idx in range(10):                       # max_attempts = 10
    action, resp = await main_agent.step(None, [])  # ← 注意传的是 None/[]：经理对环境无状态
    if action_name == "complete": break

经理的一轮 (main_agent.py:176-245)
  ① attempt += 1
  ② 把【整个】历史从头重新渲染一遍          ← 每轮 O(历史长度) 重排
  ③ 拼 prompt
  ④ 一次 LLM 调用
  ⑤ 从自由文本里正则抠 JSON               ← 不是原生 tool-calling
  ⑥ 只有两个工具可选: delegate_task | complete
  ⑦ 执行 → 把结果 append 进 task_entries

派活分支 (delegate.py:162-163)
  coroutines = [self._execute_single_task(t, i) for i, t in enumerate(tasks)]
  subtask_results = await asyncio.gather(*coroutines)      # ← 并发就这一行
```

### 2.4 ⭐ 唯一的真想法

> **把"动作"从"一个子任务"改成"一串子任务"，并且每个元素从一句指令扩成四元组（指令，上下文，模型，工具子集）。**

为什么这管用——**因为预算按"轮"算，不按"任务"算**。提示词里明说：

> "Each delegation (regardless of how many parallel subtasks) counts as **ONE attempt**"
> "with N remaining attempts, you can run **N rounds of parallel subtasks**"

于是经理被激励去**把宽度做大、把深度做浅**：K 个独立子目标从花 K 轮变成花 1 轮。**这就把一个延迟问题变成了一个规划问题。**

论文说的"在线子智能体特化"，机制上就是经理往三个字段里填值：`model`（从一张渲染出来的**价格表**里选，所以成本是显式的推理输入）、`tools`（对动作空间做正则过滤）、`context`。

### 2.5 训练：DA-GRPO

**只训经理**（Qwen3-8B 全参数）。子智能体、工具、评委全部冻结的商业 API——论文自己在 Limitations 里承认："the sub-agent backends remain fixed during training."

**"DA" = Decision-Aligned（决策对齐）**。动机原文：

> "for agent orchestration, final-answer reward is **sparse and expensive** because it requires executing the whole multi-agent system. DA-GRPO instead evaluates **each sampled main-agent decision directly at the current orchestration state**"

奖励（4 维加权）：
```
r = 0.10·格式 + 0.10·动作合法 + 0.20·工具选得合理 + 0.60·决策质量
```
由 **claude-haiku-4.5** 一次调用打出，评委的提示词里**给它看了 GPT-5 专家在这一步的决策**（但明说"不必照抄"）。

⚠️ **详见 §4.1：这个算法在代码里不存在。**

### 2.6 数字

OmniGAIA（360 题，LLM 评委）：

```
同 backbone + 同工具的 ReAct 对照 ：GPT-5  53.9  →  Orchestra-o1  72.8   (+18.9)
开源臂                            ：Qwen3-8B ReAct 12.5 → 26.3（纯框架，零训练）→ 30.0（训练后）
```

**读法**：
- **+18.9 是真·同 backbone 同工具**（原文 "under the same perception and action tools"）—— 这是它 E1 通过的原因
- **12.5 → 26.3 是"框架本身"值多少（+13.8）**；**26.3 → 30.0 才是训练值多少（+3.7）**
- ⭐ **增益在最强 backbone 上更大**（GPT-5 +18.9 > Qwen3-8B +13.8）——和"多智能体只在弱模型上有效"这个常见伪影**相反**

---

## §3 PixelCraft：六个角色围着一张"图像白板"转

**论文**：[arXiv:2509.25185](https://arxiv.org/abs/2509.25185)，**ICLR 2026**（仓库描述确认，arXiv 页无会议标注）
**代码**：[microsoft/PixelCraft](https://github.com/microsoft/PixelCraft) ✅200 · MIT · 31★ · 2026-07-17 最后 push · **只有 1 个 commit**（squash 过）
**权重**：[zss01/PixelCraft-3B](https://huggingface.co/zss01/PixelCraft-3B) ✅200

### 3.1 一句话

> 不是让模型"盯着图想"，而是让它**不断加工出新的图**（裁子图、放大区域、遮图例、画辅助线），并且**所有加工过的图都留在盘上**，规划者下一步可以挑**任意一张历史图**继续加工。

### 3.2 谁在场（目录结构和角色 1:1）

```
src/dispatcher/select_tool.py     调度员：这题该开哪些工具   ⚠️ 离线跑，不在循环里
src/planners/action_planner.py    规划者：主循环在这里（max_steps=40）
src/agents/reasoner.py            推理者（只有 918 字节）
src/critics/plan_critic.py        规划批判者                ⚠️ 离线跑
src/critics/visual_critic.py      视觉批判者                ✅ 在循环里
src/agents/tool_agents.py (42KB)  工具智能体（最大的文件）
src/tools/grounding.py            ← 微调过的 Qwen2.5-VL-3B 定位模型
src/tools/crop_subfigure.py       裁子图
src/tools/add_axvline.py          画辅助线
src/tools/code_compiler.py        跑代码
```

> ⚠️ **注意"离线"那两个**：调度员和规划批判者**不在 agent 循环里跑**，是预先跑好、结果存成 JSON 再喂进来的。所以真正在线的角色只有：规划者、工具智能体、视觉批判者、推理者。**论文的"六角色动态工作流"在代码里没有那么动态。**

### 3.3 图像记忆 —— 论文的核心主张

论文原文（§3.1）：

> "A critical feature here is the introduction of an **image memory**. Conventional approaches that feed all historical images into the context suffer from severe long-context overhead and are restricted to a **linear, chain-like** reasoning pattern. The planner's image memory stores all intermediate visual outputs, allowing it to **adaptively recall any historical image**. This enables the exploration of **alternative reasoning branches**."

以及："a **cognitive whiteboard**"（认知白板）。

**代码里它长这样**：

```python
image_pool = {"original_image": image_path, "processed_image": []}   # action_planner.py:175
...
image_pool["processed_image"].append(current_image_path)             # 每生成一张新图 append
output_path = f"{output_dir}/{step + 1}.jpg"                         # 每步的图独立落盘
self.image_path_description = {}                                     # 路径 → 描述 的索引
```

而 `image_path_description` 是**传进工具、再被工具返回更新**的：

```python
(observation, current_image_path, model_response,
 self.tool_call_history, self.image_path_description,   # ← 进去又出来
 executed_successfully) = self.tool_agent.execute_action(...)
```

⚠️ **诚实评价见 §4.2**：这个"记忆"比论文描述的朴素得多。

### 3.4 两个自适应退出（这个设计值得抄）

```python
# ① 调度员判定这题工具 ≤1 个 → 直接单次作答，根本不进多智能体循环
if tool_selection_json and len(tool_json["tools"]) <= 1:
    direct_answer = True

# ② 规划者中途决定"直接从原图抽信息就够了" → 跳出循环
if "extract_information" in action and self.original_image_path in action:
    direct_answer = True
```

→ **不是所有题都付多智能体的代价。**

### 3.5 训练

**只训一个模型：PixelCraft-3B**（Qwen2.5-VL-3B-Instruct 全参数 SFT）。所有推理侧角色（规划者/推理者/两个批判者/调度员）**都是冻结 API 的纯提示词**。

- 数据 53,000 条：GPT-4o 写图表规格 → 模板渲染（其中 ~10k 是多子图）+ 2,000 张来自 Inter-GPS/Geometry3K 的几何图
- 目标：自回归 SFT，输出 `<|box_start|>[x1,y1,x2,y2]<|box_end|>` 绝对坐标
- 代价：1 epoch，4×A100，lr 1e-5
- 效果：定位 IoU **0.26 → 0.93**（⚠️ 但测试集是同一套合成管线产出的 500 条，**同分布**）

### 3.6 数字

三个 backbone × 三个图表 benchmark（LLM 评委）：

| 方法 | GPT-4o | GPT-4.1-mini | Claude-3.7-Sonnet |
|---|---|---|---|
| Direct answer | 49.6 | 58.6 | 67.1 |
| **CoT** | **51.1** | **63.8** | **68.3** |
| Debate | 50.7 | 62.4 | 67.7 |
| Reconcile | 52.4 | 63.5 | 68.5 |
| Refocus（唯一的工具类前作） | **47.2** | **60.7** | **62.4** |
| **PixelCraft** | **55.2** | **68.1** | **73.9** |

（CharXiv 列；论文自报的 Δ 是对 Direct answer 算的）

**三个必须自己算的读法**：
1. **论文的 Δ 是对"直接答题"算的**。均值 +7.22。**对 CoT 的诚实差值只有 +3.37**，对每列最好的基线只有 +3.08
2. ⚠️ **Refocus（唯一用工具的前作）在 9 个格子里普遍低于纯 CoT** —— 说明"天真地用工具会更差"，PixelCraft 真正在比的是"工具用得好不好"
3. **它打赢的基线在 9 格里有 7 格是 CoT**，另外 2 格是 Reconcile

### 3.7 ⭐ 全场隔离得最干净的那个数：+3.1

Table 3 的 `Visual CoT` 对照，论文原话：

> "utilized the **exact same prompts and tools** but employed a **single planner** to handle both task planning and visual reasoning monolithically"

**同提示词、同工具、同像素，只是把多个角色塌成一个 planner** → **65.0 vs 68.1 = +3.1**

这就是"多智能体本身值多少"，因为其它变量全被控住了。**全部 97 个筛查系统里，只有这一个把这个对照做出来了。**

配套的 Table 4 逐角色消融：
```
什么都没有            63.8
+ 只加工具            65.0     ← 工具值 +1.2
+ 调度员              65.9
+ 调度员 + 视觉批判者  67.5
+ 规划批判者          68.1     ← 四个协调角色合计 +3.1（在工具之上）
```

---

## §4 ⚠️ 代码与论文对不上的地方（本文最有价值的一节）

### 4.1 Orchestra-o1：**DA-GRPO 在代码里不存在**

【核实】`grep -rn "DA-GRPO\|DAGRPO\|da_grpo\|Decision-Aligned"` 扫遍整个仓库 —— **只在 `README.md:134` 一张图的标题里出现过一次**，别处一个字都没有。

实际发布的训练脚本是**标准 verl GRPO**（`train_grpo_qwen3_8b.sh:107`：`algorithm.adv_estimator=grpo`）。

> **所以"新算法"退化成了"一个奖励函数的选择"** —— 给 LLM 评委看专家的决策，仅此而已。不是算法创新。

其他三条：

| # | 问题 |
|---|---|
| 2 | ⚠️ **头条是主张不是结果**："72.8% / 超第二名 10.3 个点"——**第二名是谁，README 和摘要里从头到尾没说过**。仓库里**零结果文件、零日志、零消融**。而**发布的 8B 模型根本没有公布任何数字** |
| 3 | ⚠️ **训练不可复现**：启动脚本要的两个数据准备脚本**不在仓库里** |
| 4 | ⚠️ **"开源"是有水分的**：即使用 Qwen3-8B 当经理，**轨迹压缩器仍硬编码 gpt-5**，评委是 gpt-4o，奖励模型是 claude-haiku-4.5，`sub_models` 只暴露 `["gpt-5"]`。**72.8% 那个数需要 GPT-5 API，本地只能跑出 30.0 那一臂** |

### 4.2 PixelCraft：**"图像记忆"是一个裸 dict**

论文用"认知白板""可以探索替代推理分支"来描述它。代码里【核实】：

- **没有类，没有 store，没有 manager。** 就是函数里一个局部变量：
  `image_pool = {"original_image": image_path, "processed_image": []}`
- `grep -rn image_pool src/` **总共 12 处**：1 处创建、1 处 append、4 处透传、6 处在提示词类里
- **图片没有键**：`processed_image` 就是一个**按时间顺序 append 的绝对路径列表**

**那"分支"是真的吗？** 是的，但机制比论文描述的朴素：因为**每张中间图都独立落盘**（`{output_dir}/{step+1}.jpg`）、且有 `image_path_description` 这张 path→描述 的索引表，**规划者在提示词里可以点名任意一张历史图**。所以"回看"成立，但它是靠"文件都还在 + 提示词里列出来"实现的，不是靠什么数据结构。

其他两条：

| # | 问题 |
|---|---|
| 2 | ⚠️ **核心主张只被测了一次**："image memory" 出现在标题框架、摘要、第一条贡献里，但**唯一的实验证据就是 Table 3 那一格** |
| 3 | ⚠️ **调度员和规划批判者是离线跑的**，不在循环里——论文的"动态三阶段工作流"在代码里没那么动态 |

### 4.3 两篇共有的诚实边界

| | Orchestra-o1 | PixelCraft |
|---|---|---|
| 论文自己承认 | 只训经理；子智能体全冻结 | ① 让 MLLM 自动生成工具**不可靠**，六个工具是**手工筛出来的**（468 个候选聚类到 4 个，再让 GPT-o3 重写，再手调）② **"需要强 backbone，弱的可能不行"——而全文没评测任何开源权重的推理 backbone** |
| 数字上的软肋 | 无等算力对照、无 agent 数消融、无同质化消融 | Δ 对 Direct answer 算；3–4× 延迟；工具目录是手工产物，所以"泛化性比 Refocus 好"是"手工筛得更好"的说法 |

---

## §5 我们能拿来用的零件

| 零件 | 出处 | 对我们的用处 |
|---|---|---|
| ⭐ **塌成单 agent 的对照** | PixelCraft Table 3 | **同提示词、同工具、同像素，只把多角色换成一个**——这正是我们 `text` vs `textsighted` 实验的同类做法。**可以直接引用为方法学先例** |
| ⭐ **按题自适应跳过多智能体** | PixelCraft 两处 `direct_answer = True` | 对应 [Q&A/06 §4.2](06-latent-collab-design-space-and-brain-tool-edge.md) 空格 #11「**没人按请求自适应选机制**」——PixelCraft 在 agent 层做了，**latent 通道层面还没人做** |
| **"预算按轮算不按任务算"** | Orchestra-o1 的提示词设计 | 一个把延迟问题转成规划问题的巧劲；若我们做多轮 latent 链，可借 |
| **中间图落盘 + path→描述索引** | PixelCraft | 比"把所有历史图塞进上下文"省得多，且允许回看。**若我们做视频多轮选帧，这是现成的状态管理方案** |
| **框架 vs 训练的分离测量** | Orchestra-o1 的 12.5 → 26.3 → 30.0 | 少见地把"脚手架值多少"和"训练值多少"分开报了（+13.8 vs +3.7）。**我们的论文也该这么拆** |

**不能拿来用的**：Orchestra-o1 的开源臂（轨迹压缩器硬编码 GPT-5，本地跑不出头条数字）· PixelCraft 的推理侧（全闭源，论文自己承认没测过开源 backbone）。

---

## §6 术语表

| 词 | 一句话 |
|---|---|
| **ReAct** | 一种 agent 循环：想一步 → 调一个工具 → 看结果 → 再想，直到给出答案 |
| **GRPO** | 一种强化学习算法；同一个问题采样一组答案，用组内相对好坏当优势信号，省掉价值网络 |
| **verl** | 一个开源 RL 训练框架，GRPO 的常用实现 |
| **LLM-as-judge** | 用一个大模型当评委给答案打分（这两篇的准确率都是这么来的） |
| **grounding（定位）** | 给定一句描述，在图上框出对应位置 |
| **IoU** | 两个框重叠程度的度量，1 是完全重合 |
| **同分布测试集** | 测试数据和训练数据来自同一个生成管线 —— 分数会偏高 |
| **消融（ablation）** | 拆掉某个部件看分数掉多少 |

---

## §7 相关链接

- [05-credible-mas-screening-and-graft-plan.md](05-credible-mas-screening-and-graft-plan.md) —— 这两个系统为什么被选出来（六门槛筛查）
- [06-latent-collab-design-space-and-brain-tool-edge.md](06-latent-collab-design-space-and-brain-tool-edge.md) —— 通道设计空间与空格清单
- [08-mas-topologies-and-research-route.md](08-mas-topologies-and-research-route.md) —— 五种拓扑与路线选择
