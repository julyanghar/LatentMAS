# 分支：Phase 1/2 实验结果分析与问答工作区

> 未判决草稿：本文件结论不得被其它文档引用；只有 /branch close 能把它升为权威。

> **本分支的特殊约定**（用户 2026-08-03 裁定）：这不是假设驱动的探索分支，是**开放式工作区**——用途：①对 Phase 1/2 实验结果做机制级分析；②承接用户的追问与解答。**没有预设判决标准，只有用户手动决定才能关闭。** 分析中若孵化出可判真假的具体想法，另开子分支，不占用本区。

---

## 当前状态（覆盖式）

**主线背景**（权威结论，见 [../phase2-results.md](../VMAS-latent-collab/phase2-results.md)，可引用）：Phase 1（四张诊断卡）+ Phase 2（A1/A2/A3/B1/B2/C1/C2）全部完成。核心：KV 通道传视觉信息（X−L +26~35pp）、M1 默想页零载荷（四重复现）、W_a 噪声级（三模型三签名）、M-RoPE 接管模块两级验收、4-agent 主表（latent 把通信税 −15pp→≈0，但未超 Single）。

**开放问题清单**（分析候选，无强制判决线）：
1. **默想页零载荷的成因**——三嫌疑人（口水页/纸太小/不识字）+ 第四嫌疑人（冗余）。区分实验已设计好备用：金页测试（判可读性）× 线性探针带控制任务（判内容存在性），2×2 判决表见轨迹 2026-08-03 段
2. **Latent-MAS 为何没超 Single**——对称角色不添新信息源；非对称设计（多图/分工/工具）是否能正向涨分
3. **Xshuf 在 MMStar 低于随机线**（12.5% vs 25%）的机制——错图 KV 主动误导的定量刻画
4. **Qwen3-VL 的 XWa 轻微负漂**（−1.9pp, p=0.011）——与 LLaVA-OV 的无差有何不同
5. （随问答持续追加）

**关闭条件**：仅用户手动裁定。

**下一步**：无固定下一步——由用户的问题驱动。

## 环境自检

持久态（2026-08-03 实跑）：
```bash
cat /home/yilin/tmp/phase2/a1_gqa_m10_shard*.jsonl /home/yilin/tmp/phase2/a1_mmstar_m10_shard*.jsonl | wc -l
# 期望: 1598
ls /data/yilin/models/
# 期望: llava-1.5-7b-hf LLaVA-OneVision-1.5-8B-Instruct llava-onevision-qwen2-7b-ov-hf Qwen3-VL-8B-Instruct
python -c "import json; r=json.load(open('/home/yilin/tmp/phase2/a3_vlm.json')); print(r['Wa']['step0']['budget'])"
# 期望: 0.0242
ls /home/yilin/tmp/phase2/c1_*_shard*.jsonl | wc -l
# 期望: 24 个文件 (C1 全量)
```
需重建态：无。跑新探针前 `nvidia-smi` 核实 GPU 空闲（用户裁定最多占 6 卡）。

## 起点锚点

- 2026-08-03；主线：Phase 1+2 收官、待论文写作；全部原始数据在 `~/tmp/phase1/` 与 `~/tmp/phase2/`（seed=42 + greedy，可精确复现）
- 文档地图：结果 [../phase2-results.md](../VMAS-latent-collab/phase2-results.md) / [../phase1-results.md](../VMAS-latent-collab/phase1-results.md)；讲解 [../phase1-cards-explained.md](../VMAS-latent-collab/phase1-cards-explained.md) / [../phase2-plan-explained.md](../VMAS-latent-collab/phase2-plan-explained.md)；调研总汇 [../research-digest.md](../VMAS-latent-collab/research-digest.md)

---

## 轨迹（追加式）

### 2026-08-03 开区
- 初版误建为假设驱动分支（latent-page-zero-payload），经用户裁定重构为开放式分析问答工作区
- 保留三嫌疑人区分设计作为候选方向 #1：**甲·金页测试**（页槽换成图像池化嵌入，金页有效→"不识字"排除）× **乙·默想页线性探针**（带控制任务 selectivity，读不出图像属性→"口水页"）；组合判决表：甲有效+乙无信息=口水页 / 甲有效+乙有信息=冗余 / 甲无效=不识字

## 问答记录（追加式）

（用户的追问与解答落此处；涉及权威结论修正的，经确认后同步改对应正式文档）
