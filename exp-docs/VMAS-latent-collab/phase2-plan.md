# Phase 2 实验设计

> **日期**：2026-08-02 · **前置**：Phase 1 全部通过（见 [phase1-results.md](phase1-results.md)）
> **⚠️ 范围修订（2026-08-02，用户裁定）**：POPE 判为低价值任务、退出 Phase 2；Qwen2.5-VL-7B 权重已删（卡 2 测量数据保留于 `~/tmp/phase1/card2_qwen.json`）。
> **由此 Phase 1 结论重新分级**：
> - **仍然成立（与 benchmark 无关）**：卡 1 全部几何测量、卡 2 KV 等价性与 M-RoPE 坍缩、质心塌缩/填充行/区分度三组探针
> - **降级为"诊断证据"、需在新 benchmark 重立**：H4（X−L +41.7pp）、H5（安慰剂）、X>T、X≈U、M1 零载荷（X≈Xprm）、XWa 全 EOS——这些全部测自 POPE，正式主张改由 GQA + MMStar 承担
> **Phase 2 的验证负担因此加重**：A1 从"复刻"升格为"主验证"。

---

## 三条轴，五个实验，按优先级排序

### 轴 A：任务类型泛化 —— "M1 零载荷"在推理型任务上还成立吗（最优先）

Phase 1 最锋利也最脆的结论：**m 步默想页零载荷**（X≈Xprm）。POPE 上答案直接躺在图像 KV 里，不需要"思考"——**推理型任务才是这个结论的真正考场**。

**实验 A1：盲 agent 探针主验证（GQA + MMStar 双 benchmark，全臂）** ← 升格：承担 H4/H5/X>T/M1 的全部正式主张
- 数据（**样本量统一 800/benchmark**，2026-08-02 裁定，不跑全集）：**GQA** testdev 平衡子集（组合推理，开放短答，exact-match+同义归一判分），n=800，过滤常识可猜题；**MMStar** n=800（从 1500 题按 6 个能力维度分层抽样 ~133/维，保持平衡；**设计上强制看图**——vision-indispensable 保证 L 臂趴在 25% 随机线；L²-VMAS 同款可并表）
- 功效账（修正：检出阈值 ∝ 1/√n，以 620→5pp 为锚）：单 benchmark n=800 → 可检出 **~4.4pp**；粗对比（X−L、X−Xshuf，预期 >15pp）绰绰有余；细对比（X−T、X−Xprm，可能 2-3pp）→ **两库合并 n=1600 检出 ~3.1pp**；若效应落在边缘区间，只对涉事臂追加样本（自适应扩样：只加数据、判决标准不动）
- 臂：**全臂** U / L / T / X(m=10) / Xprm / Xshuf / XWa（顺带复核 EOS 毒性是否跨任务）
- **判决**：
  - X−L > 5pp 双 benchmark 复现 → H4 正式成立（不再依赖 POPE）
  - X−Xshuf > 4pp → H5 正式成立
  - X−Xprm > 2pp 且 p<0.05 → **默想页在推理任务有载荷，"砍掉 M1"收窄为感知型结论**；仍 ≈0 → 升级为跨任务结论
- 成本：1.5 天（复用卡 3 pipeline：GQA 改判分、MMStar 改 MCQ 解析）

**实验 A2：m 扫描 × 任务类型**
- 在 GQA 与 MMStar 上各扫 m ∈ {0, 5, 10, 20, 40}（感知型锚点引用 Phase 1 POPE 的两点数据 X≈Xprm，不重跑）
- **判决**：若推理型 benchmark 的 acc-m 曲线有正斜率 → "默想的价值随任务推理深度增长"——比"M1 无用"有趣得多的结论
- 成本：半天（X 臂 only × 5 个 m 值 × 2 benchmark）

（可选加测：MathVista-mini —— 视觉数学推理，"需要思考"的极端情形。若 GQA 出现正信号则必做。）

**实验 A3：Wa/latent 区分度测量正式版**（源自追问探针 `probe_wa_distinct.py` 的升格）
- 问题：翻译/默想后的伪 embedding 还保留多少样本特有信息？（区分度预算 = 1 − 两两 cos）
- 扩展三个维度：样本 8→**200+ prompt**；模型：文本 Qwen3-8B + **VLM 双主力（Qwen3-VL-8B / LLaVA-OV-1.5）**；步数：step-0 → **整条 m 步 rollout 轨迹**（画区分度-步数曲线，检验多步累积降级）
- 配置对比：v_I vs v_Wa vs 原始 h；并与该模型上的行为读出（A1/A2 的 X−Xprm）做关联——**"几何残存信息"与"行为载荷"是否同涨落**是关键读数
- 先行数据（n=8, Qwen3-8B, step-0）：h 本身两两 cos 0.848（各向异性窄锥）；Wa 保留 73% 区分度预算（0.152→0.111）；Wa-latent 共享的公共方向不只是质心
- 成本：半天（纯前向+统计，无生成）

### 轴 B：模型泛化 —— M-RoPE 接管模块 + 两个假说验证

**实验 B1：M-RoPE 接管模块实现 + Qwen3-VL 盲探针**（二次修订：开发靶从 Qwen2.5-VL 换为 Qwen3-VL-8B，注意其 mrope_interleaved=True 与 2.5 的实现差异；卡 2 等价性测试需在 Qwen3-VL 上重跑）
- 实现卡 2 变体(c) 的生产版：跨 agent 续算时维护"当前最大三轴坐标"，文字按 max+1 三轴同值续、图像按网格铺（复刻 `get_rope_index` 的跨 past 版本）
- 验收：先过卡 2 等价性测试（top-1 100%），再跑 POPE 盲探针 n=500
- **判决**：X−L 在 Qwen3-VL 上复现 >5pp → 机制跨 backbone 成立，"KV 通道传视觉信息"从单模型结论升级为双模型结论
- 成本：模块 1-2 天 + 评测半天。**这是解锁 Qwen 系全部后续实验的钥匙**

**实验 B2：EOS 毒性=工程巧合假说验证（附带便车，几乎零成本）**
- 背景：追问实验发现质心塌缩在纯文本 Qwen3-8B 上也存在（cos 0.46-0.61）但无行为毒性；候选解释 C = 毒性需要"塌缩目的地住着强匹配+空白语义居民"（LLaVA mean-init 填充行两条全中，Qwen 随机 init 填充行不中）
- 操作：B1 跑通后加一个 XWa 臂；先查 Qwen3-VL 填充行的初始化性质（随机 or mean，5 分钟脚本）
- **预测（预注册）**：Qwen3-VL 的 XWa **不闭嘴但也不显著加分**（前提=其填充行非 mean-init，先查再预测）。中 → 假说 C 闭环，"EOS 惨案一半是 LLaVA 词表工程巧合"可写进论文；不中 → 假说 C 推翻，重新归因
- 成本：搭 B1 便车，+2 小时

### 轴 C：规模化 —— baseline 论文的主表（A、B 通过后再开）

**实验 C1：完整 4-agent MAS 主表**（原卡 5）
- **⚠️ Backbone 铁律（2026-08-03 用户裁定，覆盖以下历史修订）：现役只用 Qwen3-VL-8B-Instruct + llava-onevision-qwen2-7b-ov-hf 两个。** LLaVA-1.5 仅 Phase 1 仪器/附录；LLaVA-OV-1.5 与 Qwen3-VL-2B 弃用（前者在盘 16GB 可删）。以下为历史记录：
- Backbone（2026-08-02 修订，对齐文献当代标准）：
  - **Qwen3-VL-8B-Instruct**（主力①：L²-VMAS 与 VW 都用 Qwen3-VL 系，数字可直接并表；M-RoPE interleaved，靠 B1 解锁；下载中）
  - **LLaVA-OneVision-1.5-8B**（主力②：L²-VMAS 同款。**⚠️ 2026-08-02 实地更正：静态读其自定义 modeling 代码发现它是三轴 M-RoPE 架构**（get_rope_index/image_grid_thw/rope_deltas 俱全，穿 LLaVA 马甲的 Qwen-VL），**不是 1D**——与 Qwen3-VL 同归 B1 解锁组；trust_remote_code 加载，接口三关静态检查通过）
  - **llava-onevision-qwen2-7b-ov-hf**（**顶替"现代 1D 生态位"**：Qwen2-LM 标准 1D RoPE、transformers 原生类、2024-08——A1 的立即可跑平台；正下载）
  - **由此 B1 模块升格**：M-RoPE 是 2026 现代 VLM 的常态（三个现代 backbone 里两个是），位置接管不是边角修复而是**现代一代的必备适配器**——论文框架顺势上调
  - Qwen3-VL-2B（快速迭代用，VW 同款）
  - ~~Qwen2.5-VL-7B~~ **已删除**（2026-08-02 用户裁定）：B1/B2 直接在 Qwen3-VL 上做；卡 2 的 33.0/19.3/1.6 为其历史测量（数据留存 `card2_qwen.json`），论文引用时注明模型并在 Qwen3-VL 重测
  - LLaVA-1.5-7B **定位为仪器而非参考模型**：①Phase 1 全部数据的载体（删=不可复现）；②唯一跑过全套诊断的标准 1D-RoPE 对照台——现代模型出怪结果时回它身上区分"机制 vs 位置编码"；③论文中入附录/诊断节，不进主表
- Benchmark：GQA / MMStar / ScienceQA-IMG（+可选 MathVista-mini）。POPE 已退出（Phase 1 数字仅作附录诊断锚点）；避开 MMBench——L²-VMAS 在其上有 3 处负增益的前科
- 方法臂：Single / Text-MAS / Latent-MAS(m=0 即纯 KV 版) / Latent-MAS(m=10) / +选择性传输
- 统计：greedy 解码（确定性，免 seed 方差）；每格 n=800（与 A1 对齐）；McNemar + discordant 报告
- **这是"LatentMAS 搬到 VLM 到底什么水平"那个文献缺格的正式答案**
- 成本：4 卡 × 2 天

**实验 C2：效率三口径**（原卡 6，与 C1 同跑）
- 生成 token 数 / 端到端墙钟 / 峰值显存，agent 数 N∈{1,2,4,6,8} 扫 KV 链增长
- 必须同时报 vs Text-MAS 和 vs Single 两列（L²-VMAS 只报前者的教训）

---

## 优先级与依赖

```
A1 (GQA 复刻, 1天) ──────────┐
A2 (m 扫描, 半天) ────────────┤→ 决定"M1 零载荷"结论的写法(收窄 or 升级)
                              │
B1 (M-RoPE 模块, 2天) ──→ B2 (毒性假说, +2h) → 决定 W_a 章节的写法
                              │
            A1+B1 都通过 ──→ C1+C2 (主表, 2天) → baseline 论文的实验节完成
```

**总预算：约 1 周（4 卡）。** 若 A1 出现"推理任务默想页有载荷"的正信号，优先追加 MathVista + m 扫描细化——那将是比 baseline 更有趣的一篇的种子。

## 与四篇已有工作的对表

| 我们的格 | 他们 |
|---|---|
| training-free + 盲探针 + 安慰剂对照 + 载荷定位 | 四篇全部要训练，且**无一做过任何通道级对照** |
| M-RoPE 接管（B1） | 四篇无一处理（MACF 绕开、VW 放弃 KV 路线） |
| GQA/MMStar 数字 | 可与 L²-VMAS 的表直接并排（它 8 个 benchmark 里有这两个的近亲） |
