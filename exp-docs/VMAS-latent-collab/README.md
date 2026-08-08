# VMAS-latent-collab —— 图像 VQA 时代的 VMAS latent collaboration 研究线（已归档）

> **归档日期**：2026-08-05 · **归档原因**：主战场转向**视频理解**的 latent collaboration（[../VideoAgent-Plan/](../VideoAgent-Plan/00-plan.md)）。本目录集中 Phase 1 / Phase 2 时期针对**图像 VQA + 4-agent 角色链**设定的全部实验文档。
>
> ⚠️ **定位是"归档"，不是"作废"**。三点必须记住：
>
> 1. **C1-rev 的结论仍然有效，而且正是它逼出了转向**：补上 textsighted 臂 + 官方判分后，latent 通道的准确率优势从 +12~17 塌到 +1.2/−4.7/+9.9/−0.5 → **latent 的价值是效率不是准确率**。这条结论直接迁移进视频线的立论（[00-plan](../VideoAgent-Plan/00-plan.md) 的"主张写效率/帕累托前沿"）。
> 2. **一批机制资产直接复用进视频线**（见 [00-plan §9](../VideoAgent-Plan/00-plan.md)）：M-RoPE 接管模块（两级验收 top-1 100%）、跨模型漂移探针（rope_theta / k/v_proj / 第0层31.9%）、盲探针/Xshuf/U 臂对照机器、M1 零载荷结论（四重复现）、c1_rescore.py 判分脚本。
> 3. **实验纪律的教训全部继承**：官方判分口径、抽取失败率双报、宽松匹配偏臂（+8.9 vs +0.6）、random 兜底污染准确率、smoke 写独立文件防覆盖。

## 目录

| 文件 | 内容 |
|---|---|
| [phase1-results.md](phase1-results.md) | Phase 1：盲探针 / X≈U≫T / "两个世界"E4 阳性对照 |
| [phase1-cards-explained.md](phase1-cards-explained.md) | Phase 1 实验卡通俗解释 + 答疑回填纪律的出处 |
| [phase2-plan.md](phase2-plan.md) / [phase2-plan-explained.md](phase2-plan-explained.md) | Phase 2 计划及通俗版 |
| ⭐ [phase2-results.md](phase2-results.md) | **主结果**：C1 四臂（Single/Text/Textsighted/Latent）+ `C1-rev` 修订节（官方判分 + McNemar + 决定性对比）——**引用 C1 数字唯一权威出处** |
| [research-digest.md](research-digest.md) | 早期调研 digest（多模态 MAS 全景第一轮） |
| [beta-scale-factor.md](beta-scale-factor.md) | "45% 饱和阈值 β=−0.236" 的核查（结论：不稳健，主效应 p=0.487） |
| dl_llava_ms.log | 当期下载日志 |

## 相关但**不在**本目录的

- **实验脚本与数据**：`/home/yilin/tmp/phase2/`（c1_mas.py、c1_rescore.py、各 jsonl）
- **C1 补臂的 modify-code 物证**：`/home/yilin/modify-code-runs/c1-text-sighted-arm/`
- **调研问答**（Q&A/01–08）与**文献分析**（papers/）：仍是两条线共用的资产，留在原位
- **branches/phase12-analysis-qa.md**：branch skill 的检查点，按 skill 约定留在 branches/
