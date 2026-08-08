# 本地论文 PDF

先例审计中**逐一读过**的论文原件，随手可查、可 grep。每篇同时存 `.pdf` 和 `pypdf` 抽出的 `.txt`。

> ⚠️ **grep `.txt` 前先规整**：PDF 抽出的文本有连字符断行和随机换行，检索长句会漏。建议
> ```python
> import re; t=open(f).read(); n=re.sub(r'-\n','',t); n=re.sub(r'\s+',' ',n)
> ```

## 推翻我们两条主张的（讲解见 [02](../../VideoAgent-Plan/02-navgpt2-and-sterner-explained.md)）

| 文件 | 论文 | 为什么在这 |
|---|---|---|
| `2407.12366_NavGPT-2.pdf` | NavGPT-2（ECCV 2024） | Table 5（策略网络消融 SR 67.52→21.46）= 判死点 D0 来源；Table 6（四个冻结纯文本 LLM 互换）杀掉"独立挑选"这个区分点 |
| `2403.11317_Sterner_TaleOfTwoApproaches.pdf` | A Tale of Two Approaches（Cambridge 2024） | Table 1（caption vs embedding，0-shot 45.9 vs 41.4）= "latent 比文字保真"这条前提被打掉的直接证据 |

## 八篇危险论文（讲解见 [03](../../VideoAgent-Plan/03-eight-risk-papers-explained.md)）

| 文件 | 论文 | 一句话 |
|---|---|---|
| `2606.24470_LatentBridge.pdf` | The Latent Bridge | **唯一跑了 F/T/L 三臂消融**的论文；作者自己否掉"文字带宽不够"这个故事；v1→v2 架构教训背书输入层注入 |
| `2605.09317_Mem-W.pdf` | Mem-W | **五占四**（P1+P2+P4+P5）；差的只有模型边界。⚠️ 审计初版引的那句话是伪造的 |
| `2607.11844_SportMV-Agent.pdf` | Beyond the Single Camera | 迭代主动选视角；⚠️ 审计初版说它是纯文本 orchestrator，**错**——GPT-4.1 在此当多模态用 |
| `2607.19547_ChronoStitch.pdf` | ChronoStitch | ⭐ **"positions are necessary, but not sufficient"**——推翻我们"位置修正免费"的假设 |
| `2604.00829_LinguDistill.pdf` | LinguDistill | 唯一真把逐层 KV 交给冻结纯文本 LM 的；但老师是学生自己的前身，且只在训练时存在 |
| `2607.26773_CausalAudit.pdf` | Do Latent Channels Actually Communicate? | +15 分里有 8.33 分"给别的题的消息也能拿到"→ CAG 必须进主表 |
| `2607.14103_LimitsOfText.pdf` | Channels, Alignment, and the Limits of Text | 文字丢 88% SAE 特征，**但那 88% 不重要**；作者自认任务太浅（我们的开口） |
| `2606.07512_MemDreamer.pdf` | MemDreamer | 纯文字通道在 LVBench 上比整段像素高 +8.7~21.2，上下文小 41~124× |
| `2602.06566_SPARC.pdf` | SPARC（ICML 2026） | 命名冲撞 + 效率对手（TTFT 200×、E2E 50×） |

**判决汇总**：[01-prior-art-audit.md](../../VideoAgent-Plan/01-prior-art-audit.md)

> 审计里提到但**未归档**的（MACF / L²-VMAS / ViF / LACO / Vision Wormhole / BeMyEyes / SeeingEye / HyLaT / C2C / Beyond-Tokens survey 等）散在 `/home/yilin/tmp/priorart/`。需要哪篇转正说一声。
