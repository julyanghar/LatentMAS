# Q&A 工作空间

问答会话的沉淀区：每次深入问答整理成一篇编号 md。规矩：

- **编号命名**：`NN-english-slug.md`，中文内容英文文件名
- **每篇头部**记录日期、问题原述、证据基础（链接到 exp-docs 的实验文档）
- **主张标注证据级别**：本项目实测 / 文献（带 arXiv 号）/ 推断——三者不混
- 问答中产生的**实验设计**先记录在篇内（含判决标准与预注册预判），待实验窗口开启再执行
- 追问导致的修正**就地改原篇**，不另开新篇（同 [答疑回填纪律](../VMAS-latent-collab/phase1-cards-explained.md)）

## 目录

| # | 主题 | 日期 |
|---|---|---|
| [01](01-training-vs-trainingfree-and-soft-token-idea.md) | 训练 vs 免训练的本质切分（M2 免费/M1 需训）· 四种监督来源与损失设计 · 多模态 MAS 现状（无 4-agent latent 链、无强开源 baseline）· **"h=embedding 叠加"新想法评估**（亲属=CIPHER/Soft Thinking；修正=h 是载体非叠加本身；E1/E2/E3 实验设计 + 预注册预判） | 2026-08-03 |
| [02](02-why-append-works-in-text-but-breaks-in-multimodal.md) | 为什么纯文本 append 就行、多模态却有位置问题——**默认猜号规则（=past 长度）在 1D 下恒中（token序号≡位置号），M-RoPE 两处断裂**：图像占"面"打破恒等式（177 token→位置 44）+ 续算时算位置的原料（grid_thw）缺失→回退到 0；tracker=自己记账 | 2026-08-03 |
| [04](04-multimodal-mas-landscape-and-baselines.md) | **多模态 MAS / 多模态 LatentMAS 全景**：三层"有没有强 baseline"（方法阶梯 / 38 个已核实开源 Visual MAS / latent 侧真空且 L²-VMAS 代码承诺被删）· 六种 workflow 范式 · **性能 vs 加速两半答案相反**（角色等算力打不过 SC；扩大感知视野是唯一不可替代机制）· L²-VMAS 的 VMAS ≠ LatentMAS 4-agent（每 agent 都看图）· benchmark 对表 · **新发现 LACO**（arXiv:2605.22504，免训练+真 KV 级+m=10 步 latent，**且 §4.2 "Following [45]" 直接建立在 LatentMAS/2511.20639 之上、把 LatentMAS 列为被自己打赢的 "naive latent sharing"**）→ 三条跨切主张已更正（"没人做 KV 级/免训练/多步 latent"全部作废，真空收窄为「多模态 latent + VQA」一格；"LACO 从不消融 m"亦作废，其 Figure 6 就是 m 消融）+ MACF 的 `LatentMAS*` 负增益先例 · 7 条待办 | 2026-08-03 |
| [05](05-credible-mas-screening-and-graft-plan.md) | **判据 + 抢先风险 + graft 方案**（04 的执行篇）：MAS vs 单agent+工具三判据（**VideoAgent×2 / OmniAgent 全是单agent+工具**）· 六门槛 E1-E6（**E4(d) 帧数混淆**）· **MACF 抢先 + 两条稻草人证据**（源码层确认第2..M个agent整段prompt不调 get_rope_index / 784 反解=16帧）· **Qwen3-VL 视频无时间缩放因子→tracker近乎免费泛化** · 显存账（KV仅7.34GiB，真墙是LLaVA-OV 32k窗/瞬时激活11.34GiB）· **graft 排名 A4VL→LVAgent→VideoChat-A1** · **M1为何文本必需多模态零载荷的机制解释**（prompt段里本来有没有东西）· 10 条更正清单 | 2026-08-03 |
| [06](06-latent-collab-design-space-and-brain-tool-edge.md) | **latent collaboration 设计空间 + brain-tool 想法体检**：六条正交轴，主轴是**接收方怎么消费**（前缀拼接=LatentMAS / 加性原位融合=C2C / 交叉注意=空 / 重新前向=VW / 恢复+稀疏重算=DroidSpeak 系）· **12 个空格子**（⭐**多模态 KV 传递整片为空**、异质+免训练=空、decode 加速=空、k>2 组合自承破损、无 head-to-head、**因果验证仅一篇且打脸**）· **本机实测跨模型漂移**（M-RoPE↔1D 是伪问题；rope_theta 5e6vs1e6 + k/v_proj 漂移 14–27% + 层0 V-cache 误差 31.9% → 朴素免训练交接不可行）· ⚠️ **brain-tool 三条支撑全被推翻**（TAMA 唯一直测 +2.68/SE1.37/p≈0.145；增益靠直接传图即可得；L²-VMAS 已占 framing 且在我们 fleet 上）· 两个极便宜证伪实验 α/β | 2026-08-04 |
| [03](03-missing-positive-control-in-card1.md) | **卡 1 缺失的阳性对照** → **E4 已补测通过**：文本自谱占比 3.0-5.6×基线、文本互邻是视觉到文本的 3-4 倍 → "两个世界"保级+量化修正（薄饼中等厚度 19-40%；Qwen3-VL 视觉有弱非零对齐 1.7×基线） | 2026-08-03 |
| [07](07-orchestra-o1-and-pixelcraft-explained.md) | **两个最接近通关的闭源系统的 workflow 深读**：Orchestra-o1（V\* 77.5→90.1）与 PixelCraft-3B（工具化 3B 的角色）· 感知工具具体在哪一步被调用 · 训练的 3B 实际是"为工具服务" | 2026-08-04 |
| [08](08-mas-topologies-and-research-route.md) | **五种拓扑 + 路线取舍**：文本 MAS 四种模式（顺序接力/并行投票/辩论/**MoA 从零讲解**）在多模态里的对应实例 · ⭐ **第五种「agent+感知工具」是性质不同的不对称边**（文字边承载一次模态转换）· §2.1 **C1 的 text/textsighted 臂正好切在这条缝上，2026-08-04 结果已回填** · 三条路线取舍 → 建议路线丙 | 2026-08-04 |

> ⚠️ **2026-08-04 全库同步**：C1 补臂（`textsighted`）+ 官方 exact match 判分口径落地后，**"latent 把通信税从 −15pp 降到 ≈0 / +12~14pp 主卖点"这个表述在 04 / 05 / 08 三篇里全部已改写**。唯一权威主表 = [phase2-results.md](../VMAS-latent-collab/phase2-results.md) 的 `C1-rev` 节。新结论一句话：**latent 通道的价值是效率（省 N−1 次视觉编码 + 零中间 token），不是准确率**。
