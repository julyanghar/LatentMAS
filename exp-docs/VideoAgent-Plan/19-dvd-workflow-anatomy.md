# 19 · DeepVideoDiscovery (DVD) Workflow 解剖：角色、工具、本地化判定

> 2026-08-10。基于官方源码逐文件解析（/home/yilin/DeepVideoDiscovery，微软官方，NeurIPS 2025，MIT）。
> 论文：arXiv 2505.18079；LVBench 74.2 SOTA（用 OpenAI o3）。与 doc18(LVAgent) 对照阅读。

**一句话总纲**：DVD 是"**一个带工具箱的单研究员**"——一个 orchestrator LLM 在函数调用循环里自主决定
"全局浏览 → 语义检索 → 帧级细看 → 作答"，没有多 agent、没有投票；视频被预处理成可检索的
caption 数据库当"环境"用。

---

## 一、角色与对应模型

| 角色 | 官方模型 | 干什么 | 本地化替身 |
|---|---|---|---|
| **Orchestrator（研究员）** | OpenAI **o3** | 函数调用循环的大脑：读题→选工具→读返回→迭代→finish(答案)；MAX_ITERATIONS=3 | Qwen3-30B-A3B（我们 server 带 hermes 工具解析 ✓） |
| **Caption VLM（建库员）** | gpt-4.1-mini | 离线：视频 2fps/360p 切 10s clip，逐 clip 写 caption 入库 | VL-30B（**官方已放出 LVBench/Video-MME/LongVideoBench/EgoSchema 四个 benchmark 的现成 caption 库**，可跳过） |
| **Tool VLM（细看员）** | gpt-4.1-mini | frame_inspect 工具的执行者：对指定 clip 取 ≤50 帧回答细节问题 | VL-30B |
| **Embedding（图书管理员）** | text-embedding-3-large (3072d) | clip caption 的向量索引，供语义检索 | 本地嵌入模型（bge/gte/Qwen3-Embedding 任一） |

## 二、工具箱（dvd_core.py:29）

```
tools = [frame_inspect_tool, clip_search_tool, global_browse_tool, finish]
```

| 工具 | 功能 | 备注 |
|---|---|---|
| global_browse_tool | 全局浏览 caption 库 top-K（默认 300 条） | 长视频的"目录页" |
| clip_search_tool | 按语义 query 检索相关 clip caption | 嵌入近邻；top-k 可让 agent 自决 |
| frame_inspect_tool | 对指定 clip 拉真帧给 Tool VLM 细看 | **像素通道**；LITE_MODE=True 时被移除（纯字幕模式） |
| finish | 提交最终答案并终止 | 循环出口 |

## 三、一题的生命周期

```
离线一次: 视频 → 2fps/360p → 10s clips → caption 库 + 向量索引   （官方包可跳过）
在线每题:
  orchestrator 读题(单选模式 SINGLE_CHOICE_QA=True)
    ↺ ≤3 轮迭代:
       想一步 → 调 global_browse / clip_search / frame_inspect
       ← 工具返回文字(或帧细看结论)进上下文
    → finish(答案字母)
```

无多 agent、无投票、无淘汰——**"接收方决定看什么"被做成了字面意义**：orchestrator 逐步缩小
时空范围（全局→片段→帧），恰是我们先例审计里点名的方向的工业级实现。

## 四、本地化判定：一行补丁级

1. **API 客户端**（dvd/utils.py:75 `call_openai_model_with_tools`）：设了 OPENAI_API_KEY 就走
   `https://api.openai.com/v1`（硬编码）——**改一行指向 http://localhost:18050/v1 即通**；
   Azure 分支（CLI credential）无视即可。payload 是标准 chat/completions + tools + tool_choice，
   与 vLLM 的 OpenAI 兼容层 + hermes 工具解析完全对口。
2. **配置**（dvd/config.py）：三个模型名 + LITE_MODE=False（HourVideo 无字幕，必须开像素通道）+
   embedding 资源指向本地。
3. **嵌入服务**：任选 bge-large/gte/Qwen3-Embedding 本地起一个（几行代码，或 vLLM embedding 服）。
4. **决定性注意**：官方 temperature=0.0 ✓；o3 换 30B 后 reasoning 强度降档，复现分数会低于论文
   （74.2 是 o3 的分）——但我们要的是"agentic vs 单 VLM 同模型对比"，不是复刻 74.2。
5. **复现资产**：reproduce/ 目录有完整脚本（decode_frames/transcribe/prepare_database/run_benchmark）
   + REPRODUCE.md；官方 EgoSchema caption 包 → **视频没到手就能先在 EgoSchema 500 上本地跑通全流程
   并与我们已有的 73.4 单 VLM 门槛同场对比**。

**工作量预估**：端点补丁+配置+嵌入服 ≈ 1 天；EgoSchema 本地试跑再 1 天；HourVideo 建库
（视频到手后 caption 500×~46min，VL-30B 约 1-2 天 GPU）。

## 五、三宿主对照（门 2.0 赛马的设计轴）

| | DVD | LVAgent | 我们 G2 |
|---|---|---|---|
| 结构 | 单研究员+工具 | 三全能选手+代码裁判 | 专职侦察+聚合者 |
| 视觉访问 | 按需逐级放大（检索→帧） | 检索选 1 块×16 帧 | 全覆盖分段+关键帧接力 |
| "看→想"接口 | caption 库+帧细看返回 | 16 帧直喂+文字讨论 | 段报告+接力帧 |
| 信道介质消融位 | caption 库的介质（文字→关键帧→latent）+工具返回介质 | 讨论流介质 | 报告/接力介质 |

三者的"看→想"接口各不相同——正好构成介质定律的多宿主复现矩阵（对 doc16 开口 #1/#2 的部署）。
