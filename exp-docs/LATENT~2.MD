# LatentMAS 生成式假设 —— 可证伪性检验手册

> **用途**：这不是综述，是一份**可执行的观察实验协议**。目标是把"生成式假设到底成不成立"这个看似哲学的问题，压成**几个可以跑出来的数字**。
>
> **配套**：原理见 `Wa_alignment_from_zero.md`（$W_a$ 从零讲解）、`LatentMAS_论文总结.md`（全文精读）。本文只管**验证**。
>
> **核心结论预告**：假设**不可证明**（它是关于训练所得权重的经验命题），但**完全可测**——而论文没测。最关键的那个数字只要 **20 行代码**。

---

## 0. 先厘清：哪些结论依赖哪个假设

动手之前必须分清打击面。论文有**两个不同假设**，撑着**不同结论**，严谨度差很多：

| 结论 | 依赖的假设 | 严谨性 | 本手册是否检验 |
|---|---|---|---|
| **Thm 3.3** KV 传递无损 | **无** | ✅ **真严谨**（纯确定性归纳） | 否，无需 |
| **$W_a$ 的公式本身** | **无** | ✅ 就是最小二乘拟合，**定义上不需要假设** | 否，无需 |
| **Thm A.1** Wasserstein 上界 | **生成式假设 (GA)** | ⚠️ 假设成立才成立 | ✅ **E1, E4** |
| **Thm 3.1** 表达力 235×–471× | **Assumption B.1**（线性表示假设 + 三值系数 $c_i\in\{0,\pm1\}$） | ⚠️⚠️ 更强 | ✅ **E7**（较难） |
| **Thm 3.4** 的 TextMAS 对比侧 | 继承 Thm 3.1 | ⚠️⚠️ | 间接 |
| 实验增益 +2.3%~5.3% | **无**（实测） | ✅ 实证 | ✅ **E3** |

> **⚠️ 关键区分**：**"$W_a$ 这个做法"不需要任何假设**（它就是个最小二乘拟合，你可以什么都不假设直接算）。
> **需要假设的是"为什么这么做是对的"这个解释**。所以即使假设被证伪，$W_a$ 照样可能有效——**别把"理论错"等同于"方法废"**。

**两个假设的原文：**

### GA —— Generative Assumption（生成式假设，附录 A）

存在语义权重 $\sigma\sim P_\sigma$（$1\times\lvert V\rvert$，$\|\sigma\|_2\le1$），使得

$$\underbrace{\sigma\sim P_\sigma}_{\text{隐变量}}\ \longrightarrow\ \begin{cases}e=\sum_{x\in V}\sigma_x W_{in,x}=\sigma W_{in} & \text{观测 1}\\[4pt] h=\beta\sum_{x\in V}\sigma_x W_{out,x}=\beta\,\sigma W_{out} & \text{观测 2}\end{cases}$$

**"同一个 $\sigma$"才是全部内容**（见 §1.1）。

> **📛 命名说明（重要）**：**「GA」是本手册起的简称——论文从未给这个假设起过名字**，附录 A 只是裸陈述了它。这本身值得留意：**全篇最关键的那个假设，连个名字都没有，很容易被读者滑过去。**
> 叫"生成式"有原文依据：附录 A.1 原话是 "We assume that $P_e$ and $P_h$ can be **generated** by..."——`generated` 是论文自己的用词。统计里 generative model 的含义正是"数据如何被产生"：**先有语义 $\sigma$，它再通过两张表分别生成 $e$ 和 $h$；$\sigma$ 是那个看不见的共同源头。**

### B.1 —— Linear Representation Hypothesis（线性表示假设，附录 B）

$h=\sum_{i=1}^{d_h}c_i s_i$，$\{s_i\}$ 线性无关语义基，**三值系数** $c_i\in\{0,\pm1\}$。**论文对这个假设是有命名的**（引 Park et al. 2023）。

### ⚠️ 两者相互独立，别搞混

| 简称 | 全称 | 论文命名了吗 | 核心断言 | 撑着 |
|---|---|---|---|---|
| **GA** | Generative Assumption | ❌ **没有**（裸陈述） | **同一个 $\sigma$** 同时生成 $e$ 和 $h$ | Thm A.1 → $W_a$ 的**解释** |
| **B.1** | Linear Representation Hypothesis | ✅ 有 | $h$ 是语义基的**三值**线性组合 | Thm 3.1 → 表达力 235×–471× |

**GA 塌了不影响 Thm 3.1，B.1 塌了不影响 $W_a$。** 本手册 **E1–E4 打 GA**，**E7 打 B.1**。

---

## 1. 把哲学问题压成一个数

### 1.1 GA 里哪部分是**空的**，哪部分有**内容**

逐条拆：

- **"$\exists\sigma:\ e=\sigma W_{in}$"** —— **近乎空话**。$W_{in}$ 是 $\lvert V\rvert\times d_h$ 且秩为 $d_h$，$\lvert V\rvert\gg d_h$，所以**任意** $e\in\mathbb{R}^{d_h}$ 都能这样写，而且写法有无穷多种。
- **"$\exists\sigma:\ h=\beta\sigma W_{out}$"** —— **同样近乎空话**，理由相同。
- **"两者是同一个 $\sigma$"** —— ⚠️ **这才是全部内容。**

**"同一个 $\sigma$"的可操作等价物是什么？** 取 $\sigma=\mathbf{1}_x$（纯词 $x$），GA 立刻给出

$$\beta\,W_{out,x}\,W_a\approx W_{in,x}\quad\text{对每个 }x$$

再由线性推广到一般 $\sigma$。所以——

> **🎯 GA 的全部可测内容 = "存在一个线性映射 $M$，使 $W_{out}M\approx W_{in}$ 且残差小"。**
> **而"残差多小"就是一个可以直接算出来的数。哲学问题到此终结。**

### 1.1b 更锋利的刻画：GA ⟺ 两个 147840 维的核必须重合

上面还漏了一层。$\sigma$ 不但**存在**，还**不唯一**（$\lvert V\rvert\gg d_h$，解空间维度 $\lvert V\rvert-d_h$）。于是 GA 的真内容浮出水面：

$$\boxed{\ \text{GA}\iff\textbf{所有能生成同一个 }h\textbf{ 的 }\sigma\textbf{，必须给出同一个 }e\ }$$

否则就能造出"同一个 $h$、两个矛盾的 $e$"（§1.4 的冲突构造即是）。这**恰好等价于**存在线性映射 $W_{out}M=W_{in}$，也等价于纯线性代数条件：

$$\ker(W_{out})\subseteq\ker(W_{in}),\qquad\ker(W):=\{\sigma:\sigma W=0\}$$

**由于两个核维数相同**（都是 $\lvert V\rvert-d_h$，前提是两表都列满秩），**包含 ⟺ 相等**：

```
Qwen3-8B: dim ker(W_out) = |V| - d_h = 147840
GA 严格成立 <=> 两个独立训练出来的 147840 维子空间 完全重合  -> 概率为 0
```

> **⚠️ 由此得到本手册最重要的定调：严格意义下 GA 必假。**
> **所以别去问"GA 真不真"（问了也是必假），只问"它假到什么程度"——那就是 $\rho$。**
>
> **$\rho$ 的几何含义也随之清晰**：它量的是**这两个 147840 维核有多"对齐"**。
> $\rho\to0$ ⟹ 两核近乎重合，GA 是极好的近似；$\rho\to0.986$ ⟹ 两核彼此随机，GA 无内容。
> **这就是 E1 在测的东西。**

### 1.2 关键简化：$\beta$ 与这个检验**无关**

要测的是 $\min_{W_a}\|\beta W_{out}W_a-W_{in}\|_F$。**令 $M:=\beta W_a$**，目标变成

$$\min_{M}\ \|W_{out}M-W_{in}\|_F$$

$\beta$ **被完全吸收**了——它只是给 $W_a$ 换个缩放，不影响"线性关系存不存在"。

> **推论：E1 不需要跑模型前向、不需要估 $\beta$、不需要任何数据。只要两张权重表。**
> $\beta$ 只在你真把 $h$ 喂进去时才重要（E3/E4/E6）。

### 1.3 判据：拿什么当尺子

定义**相对残差**

$$\boxed{\ \rho:=\frac{\min_M\|W_{out}M-W_{in}\|_F}{\|W_{in}\|_F}\ \in[0,1]\ }$$

- $\rho=0$：GA 严格成立（存在精确线性映射）。
- $\rho\approx$ null：GA 毫无内容（两表的线性关系不比随机强）。

**null baseline 有闭式**（已数值验证）：两张**无关**的随机表做最小二乘，残差为

$$\rho_{\text{null}}\approx\sqrt{1-\frac{d_h}{\lvert V\rvert}}$$

> **为什么**：最小二乘把 $W_{in}$ 的每一列投影到 $W_{out}$ 的 $d_h$ 维列空间上；随机向量落在随机 $d_h$ 维子空间里的能量占比是 $d_h/\lvert V\rvert$，剩下的全是残差。

**实测验证**（numpy，见 §6.0）：

| $\lvert V\rvert$ | $d_h$ | 比例 | 实测 $\rho_{\text{null}}$ | 理论 $\sqrt{1-d_h/\lvert V\rvert}$ |
|---|---|---|---|---|
| 3700 | 100 | 37.0 | 0.9863 | 0.9864 |
| 1520 | 41 | 37.1 | 0.9862 | 0.9864 |
| 7600 | 205 | 37.1 | 0.9865 | 0.9864 |
| 2000 | 500 | 4.0 | 0.8657 | 0.8660 |
| 400 | 200 | 2.0 | 0.7101 | 0.7071 |

**正控**（GA 严格成立时）：

| 构造 | $\rho$ |
|---|---|
| 权重绑定 $W_{in}=W_{out}$ | $1.4\times10^{-16}$ ✅ |
| 真存在线性关系 $W_{in}=W_{out}R$ | $5.0\times10^{-16}$ ✅ |

### 1.4 校准曲线：$\rho$ 到底怎么读

构造 $W_{in}=W_{out}R+\text{噪声}$，看 $\rho$ 随"噪声/信号"比如何变化（实测）：

| 噪声/信号 | $\rho$ | 含义 |
|---|---|---|
| 0.0 | 0.000 | GA 严格成立 |
| 0.1 | 0.098 | GA 是极好的近似 |
| 0.3 | 0.283 | GA 是不错的近似 |
| 0.5 | 0.441 | GA 勉强 |
| 1.0 | 0.697 | 噪声和信号一样大 |
| 2.0 | 0.882 | GA 基本没内容 |
| $\infty$ | 0.986 | = null，完全无关 |

> **📏 这张表就是判读 $\rho$ 的尺子。** 拿到真实模型的 $\rho$，往这里一对，立刻知道 GA 处在什么水位。

### 1.5 ⚠️ Qwen3-8B 的 null 参照

$$\lvert V\rvert=151936,\quad d_h=4096\ \Longrightarrow\ \frac{\lvert V\rvert}{d_h}=37.1,\quad \rho_{\text{null}}\approx\mathbf{0.9864}$$

**这意味着：$\rho$ 的"及格线"极低。** 只要 $\rho<0.98$ 就已经比随机强，但那**远不足以支撑 Thm A.1 的上界有意义**。真正需要的是 $\rho$ 小到让 $\|\beta W_{out}W_a-W_{in}\|_F$ 成为一个**紧**的界。

> **论文全文没有这个数字。这是本手册要补的第一个洞。**

---

## 2. 实验清单（按性价比排序）

| # | 实验 | 成本 | 价值 | 结论强度 |
|---|---|---|---|---|
| **E0** | 权重是否绑定 | 1 分钟 | ★★★ | **前置门槛，必跑** |
| **E1** | 相对残差 $\rho$ | 10 分钟 | ★★★ | **核心，直接量化 GA** |
| **E2** | 线性 vs 非线性 | 1 小时 | ★★★ | **判决性：能证伪 GA** |
| **E3** | $W_a$ vs $\frac1\beta I$ vs $I$ | 需跑 pipeline | ★★★ | **判决性：增益是不是纯缩放** |
| **E4** | Thm A.1 的界紧不紧 | 半天 | ★★ | 定理是否 vacuous |
| **E5** | 逐词残差分布 | 20 分钟 | ★ | 哪些词翻得准 |
| **E6** | $\beta$ 的稳定性 | 半天 | ★ | 单标量是否站得住 |
| **E7** | 三值系数假设 (B.1) | 难 | ★★ | Thm 3.1 的地基 |

---

### E0 —— 权重是否绑定（**必须先跑，否则后面全白做**）

**为什么是门槛**：若 $W_{in}=W_{out}$，则 $W_a=\frac1\beta I$，$\rho\equiv0$，**GA 平凡成立、$W_a$ 退化成纯缩放**。这时候讨论"对齐"毫无意义，$W_a$ 的全部增益都来自 $\beta$ 这个标量。

```python
import torch
from transformers import AutoModelForCausalLM

MODEL = "Qwen/Qwen3-8B"          # 也跑 4B / 14B，小模型更可能绑定
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32,
                                         device_map="cpu", low_cpu_mem_usage=True)
W_in  = m.get_input_embeddings().weight.detach()   # [|V|, d_h]
W_out = m.lm_head.weight.detach()                  # [|V|, d_h]

print("W_in :", tuple(W_in.shape), " W_out:", tuple(W_out.shape))
print("同一块内存(tied)? ", W_in.data_ptr() == W_out.data_ptr())
print("数值相等?         ", torch.allclose(W_in, W_out))
print("config.tie_word_embeddings =", m.config.tie_word_embeddings)
```

**判读**

| 结果 | 含义 | 下一步 |
|---|---|---|
| `tied = True` | $W_a=\frac1\beta I$，**GA 平凡**，"对齐"是假象，增益纯来自缩放 | **直接跳 E3**，并在结论里点明 |
| `tied = False` | GA 有实质内容 | 继续 E1 |

> **这一条本身就可能是个发现**：如果论文测的某些 Qwen3 尺寸恰好绑定权重，那它报告的 +2.3%~5.3% 在那些尺寸上**就是纯 $\beta$ 缩放的功劳**，与"输入-输出对齐"这个叙事无关。论文未说明各尺寸的绑定情况。

---

### E1 —— 相对残差 $\rho$（**核心实验**）

**问题**：真实的 $W_{out}\to W_{in}$ 线性关系有多好？
**成本**：只需两张权重表，**不跑前向、不用数据、不用估 $\beta$**（§1.2）。

```python
import torch

# 承接 E0 的 W_in, W_out
# ⚠️ 必须 float64：正规方程求逆对精度极敏感，fp16/fp32 会毁掉结果
A = W_out.double()          # [|V|, d_h]
B = W_in.double()           # [|V|, d_h]
V, dh = A.shape

def rel_resid(A, B, lam_scale=1e-8):
    G   = A.T @ A                                   # [d_h, d_h]
    lam = lam_scale * torch.diag(G).mean()          # 轻微 ridge，仅保数值稳定
    M   = torch.linalg.solve(G + lam*torch.eye(dh, dtype=G.dtype), A.T @ B)
    return ((A @ M - B).norm() / B.norm()).item(), M

rho, M = rel_resid(A, B)
rho_null = (1 - dh/V) ** 0.5

print(f"|V| = {V},  d_h = {dh},  超定倍数 = {V/dh:.1f}")
print(f"真实相对残差 rho      = {rho:.4f}")
print(f"随机 null 基线        = {rho_null:.4f}")
print(f"相对 null 的改进      = {(rho_null - rho)/rho_null*100:.1f}%")
print(f"条件数 cond(W_out^T W_out) = {torch.linalg.cond(A.T @ A).item():.2e}")
```

**同时跑真正的 null（同形状随机表），确认基线**：

```python
Ar = torch.randn(V, dh, dtype=torch.float64)
Br = torch.randn(V, dh, dtype=torch.float64)
print("实测 null =", rel_resid(Ar, Br)[0], " 理论 =", rho_null)
```

#### 判读表（对照 §1.4 校准曲线）

| $\rho$ 落点 | 判读 | 对 Thm A.1 的意义 |
|---|---|---|
| $<0.15$ | **GA 是很好的近似**，论文站得住 | 上界紧，定理有意义 |
| $0.15\sim0.45$ | GA 部分成立，线性抓到了主要结构 | 上界偏松但非空洞 |
| $0.45\sim0.85$ | **GA 很勉强**，线性只抓到零头 | 上界很松，定理接近 vacuous |
| $>0.85$（逼近 0.986） | **GA 基本被证伪**，两表线性关系近乎随机 | **上界 vacuous，Thm A.1 形同虚设** |

> **我的预测（可被打脸，请务必实测）**：$\rho$ 会落在 **0.6~0.9** 区间——即"比随机强，但远算不上好近似"。
> **依据**：(a) 37 倍超定；(b) 大模型普遍**解绑**权重且有收益——若一个 $d_h^2$ 的线性映射就能从 $W_{out}$ 还原 $W_{in}$，那多花 $\lvert V\rvert d_h$ 个参数解绑就是浪费；(c) $W_a$ 只买来 +2.3%~5.3%，而非"消除 OOD"。
> **如果实测 $\rho<0.3$，我的判断就是错的，论文的理论比我以为的扎实得多。**

---

### E2 —— 线性 vs 非线性（**判决性实验**）

**问题**：GA 断言 $W_{out}\to W_{in}$ 的关系是**线性**的。如果一个**非线性**映射能显著击败线性，**"线性"这个假设本身就是瓶颈 → GA 被证伪。**

```python
import torch, torch.nn as nn

dev = "cuda"
A32, B32 = A.float().to(dev), B.float().to(dev)     # [|V|, d_h]
Bn = B32.norm()

# 划分 train/test 词表，防止"记住词表"而非"学到关系"
g   = torch.Generator().manual_seed(0)
perm = torch.randperm(V, generator=g)
tr, te = perm[:int(V*0.9)].to(dev), perm[int(V*0.9):].to(dev)

def fit(model, steps=4000, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(steps):
        idx = tr[torch.randint(len(tr), (4096,), device=dev)]
        loss = ((model(A32[idx]) - B32[idx])**2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return ((model(A32[te]) - B32[te]).norm() / B32[te].norm()).item()

# 基准1：闭式线性（E1 的 M），在 test 词上评估
with torch.no_grad():
    rho_closed = ((A32[te] @ M.float().to(dev) - B32[te]).norm() / B32[te].norm()).item()

# 基准2：训练出来的线性 —— 检查闭式解是否已最优
rho_lin = fit(nn.Linear(dh, dh, bias=False).to(dev))

# 基准3：非线性 MLP —— 判决线性是否为瓶颈
mlp = nn.Sequential(nn.Linear(dh, 4*dh), nn.GELU(), nn.Linear(4*dh, dh)).to(dev)
rho_mlp = fit(mlp)

print(f"闭式线性 (test)  rho = {rho_closed:.4f}")
print(f"训练线性 (test)  rho = {rho_lin:.4f}")
print(f"非线性MLP(test)  rho = {rho_mlp:.4f}")
```

#### 判读表

| 观察 | 结论 |
|---|---|
| $\rho_{\text{闭式}}\approx\rho_{\text{训练线性}}$ | ✅ 闭式解已达线性最优，论文的 $W_a$ 没浪费；$\beta,\lambda$ 估计合理 |
| $\rho_{\text{训练线性}}\ll\rho_{\text{闭式}}$ | ⚠️ 闭式解**没解好**（多半是 $\beta$ 或 $\lambda$ 的锅），论文的 $W_a$ 有改进空间 |
| $\rho_{\text{MLP}}\approx\rho_{\text{线性}}$ | ✅ **线性不是瓶颈** → GA 的"线性"部分站得住 |
| $\rho_{\text{MLP}}\ll\rho_{\text{线性}}$ | 🔴 **GA 被证伪**：关系存在但**非线性**。论文的整个线性叙事不成立，且提示"训一个小 adapter"能显著超过 $W_a$ |

> **这是本手册里最能出结果的一条**：无论结果如何都是发现。若 MLP 大幅获胜，你直接拿到一个"改进 LatentMAS"的方向（代价是牺牲 training-free）。

---

### E3 —— $W_a$ vs $\frac1\beta I$ vs $I$（**判决增益的真正来源**）

**问题**：论文报告 $W_a$ 带来 +2.3%~5.3%。**这到底是"对齐"的功劳，还是仅仅"缩放"的功劳？**

$W_a$ 干了两件事：**(a) 换坐标系（对齐）**、**(b) 修量级（$\frac1\beta$）**。这个消融把两者拆开：

| 变体 | 含义 | 下游准确率 |
|---|---|---|
| $e=h$ | 什么都不做（论文的"无 $W_a$"基线） | ? |
| $e=\frac{1}{\beta}h$ | **只缩放，不对齐**（$W_a\to\frac1\beta I$） | **? ← 关键格子** |
| $e=hW_a$ | 完整方法 | ? |
| $e=h\cdot\frac{\|W_a\|_F}{\sqrt{d_h}}$ | 另一种纯缩放对照 | ? |

```python
# 在 LatentMAS 官方实现里，把生成 W_a 的那一步替换即可
def make_Wa(mode, W_in, W_out, beta, lam=1e-6):
    dh = W_out.shape[1]
    if mode == "identity":  return torch.eye(dh) / beta      # 只缩放
    if mode == "none":      return torch.eye(dh)             # 什么都不做
    G = W_out.T @ W_out
    return torch.linalg.solve(G + lam*torch.eye(dh), W_out.T @ W_in) / beta
```

#### 判读表

| 观察 | 结论 |
|---|---|
| $\frac1\beta I$ 的增益 $\approx$ $W_a$ 的增益 | 🔴 **"对齐"是叙事，真正起作用的是标量缩放。** 论文最响的那个卖点被架空——$W_a$ 的 $d_h^2$ 个元素基本没贡献 |
| $W_a\gg\frac1\beta I\gg$ 无 | ✅ 对齐**确有**独立贡献，论文站得住 |
| $\frac1\beta I\approx$ 无 | 说明 $\beta$ 不重要，增益全来自对齐（与 E1 的 $\rho$ 应当呼应） |

> **⚠️ 若 E0 显示权重绑定，这一格是数学必然**（$W_a\equiv\frac1\beta I$），无需实验即知增益纯来自缩放。
> **这条实验成本最高（要跑完整 pipeline），但结论最致命。**

---

### E4 —— Thm A.1 的上界紧不紧（**定理是否 vacuous**）

**问题**：Thm A.1 说 $d_W(P_{\hat e,W_a},P_e)\le\|\beta W_{out}W_a-W_{in}\|_F$。**上界只有紧才有意义。** 论文证了界，从没量过它。

```python
# 1) 收集真实的 h（跑一批 MedQA/GSM8K，取最后层隐状态）
#    需 output_hidden_states=True，取 hidden_states[-1]
H = collect_real_hidden_states(model, prompts, n=5000)   # [n, d_h]

# 2) 左边：对齐后向量 与 真实入场向量 的实际分布距离
E_hat = H @ Wa                        # [n, d_h]   对齐后
E_real = W_in                         # [|V|, d_h] 真实入场向量（离散！）
lhs = sliced_wasserstein(E_hat, E_real, n_proj=512)   # 高维用 sliced-W 近似

# 3) 右边：定理给的界
rhs = (beta * W_out @ Wa - W_in).norm().item()

print(f"实际分布距离 (LHS) = {lhs:.4f}")
print(f"定理上界   (RHS) = {rhs:.4f}")
print(f"松弛倍数 RHS/LHS = {rhs/lhs:.1f}x")
```

#### 判读表

| 松弛倍数 | 结论 |
|---|---|
| $<5\times$ | 界紧，Thm A.1 有实际指导意义 |
| $5\sim50\times$ | 界松，只能算定性指引 |
| $>50\times$ | 🔴 **界 vacuous**——定理是**真的但没用**（"距离不超过一千公里"） |

> **注意一个真实的张力**（值得单独观察）：$P_e$ **是离散的**（只有 $\lvert V\rvert$ 个点，因为真实输入就是 token），而 $P_{\hat e}$ 是**连续**的。**两者的 Wasserstein 距离有一个不可能为零的下界。** GA 假装 $P_e$ 连续，这是它与现实脱节的第一处——**建议顺手量一下这个"不可约距离"**，它是 LHS 的天花板。

---

### E5 —— 逐词残差分布

**问题**：$\rho$ 是个平均数，掩盖结构。哪些词翻得准？

```python
with torch.no_grad():
    per_tok = (A @ M - B).norm(dim=1) / B.norm(dim=1)   # [|V|] 每个词的相对误差
print("中位数", per_tok.median().item(), " p10", per_tok.quantile(.1).item(),
      " p90", per_tok.quantile(.9).item())

# 与词频挂钩：高频词是否翻得更准？（用 tokenizer 在语料上统计词频）
# 与范数挂钩：
print("corr(残差, ||W_in,x||) =",
      torch.corrcoef(torch.stack([per_tok, B.norm(dim=1)]))[0,1].item())
```

**看点**：若**高频词残差显著更低**，说明 $W_a$ 实际上是"对常用词对齐得好、长尾一塌糊涂"——那么它在推理任务上的表现会依赖 latent 思维是否落在高频区。这是个论文完全没碰的角度。

---

### E6 —— $\beta$ 的稳定性

**问题**：论文用**单个全局标量** $\beta=\lVert h\rVert\big/\overline{\lVert W_{in,x}\rVert}$，"每次运行算一次"。但真实 $\lVert h\rVert$ 随层、位置、输入变化很大。

```python
H = collect_real_hidden_states(...)         # [n, d_h]
norms = H.norm(dim=1)
print("||h|| 分布: mean %.2f  std %.2f  p5 %.2f  p95 %.2f  max/min %.1f" % (
    norms.mean(), norms.std(), norms.quantile(.05), norms.quantile(.95),
    (norms.max()/norms.min())))

# 关键：latent 步数推进时，||h|| 会漂移吗？（这直接关系 Fig 8 的">80 步退化"）
for step in range(0, 80, 10):
    print(step, hidden_norm_at_latent_step(step))
```

**看点**：若 $\lVert h\rVert$ 的 p95/p5 差好几倍，单个 $\beta$ 就是**系统性错配**，且错配随 latent 步数累积——**这可能才是 Fig 8 里"超过 80 步就退化"的真正原因**（论文归因于"冗余信息"，但"$\beta$ 失配导致漂移"是更简洁的解释）。**验证方法**：改用**逐步自适应** $\beta_t=\lVert h_t\rVert/\overline{\lVert W_{in}\rVert}$，看退化点是否右移。若右移 → 论文对 Fig 8 的归因是错的。

---

### E7 —— 三值系数假设（Thm 3.1 的地基，较难）

Assumption B.1 断言 $h=\sum_i c_i s_i$ 且 $c_i\in\{0,\pm1\}$。**"三值"是 $3^{d_h}$ 那个计数的唯一来源**，也就是 235×–471× 的全部依据。

**但真实隐状态的系数是连续实数**，不是三值。可做的检查：

```python
# 用 SAE / 字典学习 在真实 h 上求一组稀疏基，看系数分布
# 若系数明显连续/长尾 → "三值"是纯粹为了让计数好看的技术性假设
```

> **⚠️ 注意**：这里有个微妙之处——**若系数是连续的，隐状态集合的基数是"不可数无穷"，Thm 3.1 的计数论证反而"更成立"（下界更大）**。所以三值假设其实是个**保守化**手段。真正的问题不在基数，而在于**基数 ≠ 可达性**：模型实际用到的隐状态只是这个集合里极小的一撮。**Thm 3.1 量的是"容量上界"，不是"实际利用率"**——这个 gap 才是 235× 与实测最优 $m=40\sim80$ 之间巨大差距的根源。

---

## 3. 总判读表：几个数字拼出的图景

跑完 E0–E3，把结果填进这张表，就能给"这套理论有多可靠"下结论：

| E0 绑定? | E1 $\rho$ | E2 MLP vs 线性 | E3 $\frac1\beta I$ vs $W_a$ | **综合结论** |
|---|---|---|---|---|
| 是 | — | — | 相当 | 🔴 **$W_a$ 是纯缩放**，"对齐"叙事在该尺寸上完全架空 |
| 否 | <0.3 | 相当 | $W_a$ 明显更好 | ✅ **论文理论扎实**，GA 是好近似 |
| 否 | 0.6~0.9 | 相当 | $W_a$ 明显更好 | ⚠️ **GA 很粗糙但线性没错**；Thm A.1 松但方法有效——**理论是弱解释，不是支撑** |
| 否 | 0.6~0.9 | **MLP 大胜** | — | 🔴 **GA 被证伪**：关系非线性。同时得到改进方向（训 adapter） |
| 否 | >0.85 | — | 相当 | 🔴 **最坏情况**：GA 近乎无内容 **且** 增益纯来自缩放。整个 $W_a$ 叙事崩塌 |

---

## 4. 陷阱清单（会让你白跑一天）

1. **精度**：正规方程**必须** float64。fp16/bf16 会让 $(W_{out}^\top W_{out})^{-1}$ 完全失真。先看 `cond(W_out^T W_out)`——若 $>10^{8}$，即使 float64 也要加大 $\lambda$。
2. **先查绑定（E0）**：不然 $\rho=0$ 会让你以为"GA 完美成立"，其实只是同一张表。
3. **$\beta$ 在 E1 里是无关变量**（§1.2）——别浪费时间估它，它被吸收进 $M$ 了。
4. **形状**：`lm_head.weight` 和 `embed_tokens.weight` 在 PyTorch 里**都是 $[\lvert V\rvert,d_h]$**。别转置错（详见 `Wa_alignment_from_zero.md` §7 Q2）。
5. **E2 必须划 train/test 词表**：$d_h^2\approx1.7\times10^7$ 个参数配 $\lvert V\rvert=1.5\times10^5$ 个样本，**过拟合风险真实存在**。在训练过的词上比残差没有意义。
6. **null 要同形状**：$\rho_{\text{null}}$ 只依赖 $d_h/\lvert V\rvert$，务必用**真实的** $\lvert V\rvert, d_h$ 算。
7. **特殊 token**：词表里有大量从未训练过的保留位（Qwen3 有一批 `<|extra_x|>`），它们的嵌入可能是随机初始化的**噪声**，会**虚高** $\rho$。**建议同时报告"剔除未使用 token 后"的 $\rho$**——这可能是个不小的修正。
8. **多尺寸都跑**：4B/8B/14B 的绑定状态和 $\rho$ 可能不同。**如果 $\rho$ 随规模系统性变化，本身就是个发现。**

---

## 5. 一句话总结

> **GA 严格意义下必假**（§1.1b：它要求两个独立训练的 147840 维子空间完全重合，概率为 0）。所以**"证明 GA"是个伪问题**——真问题是**"它假到什么程度"**，而这**完全可以测**，**最关键的那个数只要 20 行代码**（E1）。
>
> **论文证了 Thm A.1 这个上界，却从没量过界有多大。** 而随机 null 基线高达 **0.986**，意味着**不测就无法判断这个上界是不是 vacuous**。
>
> **这不是"论文造假"，是"能廉价验证却回避了验证"。** 而且要区分清楚：**即使 GA 被证伪，$W_a$ 照样可能有效**——它是个不需要任何假设的最小二乘拟合，+2.3%~5.3% 是实测。**被证伪的会是"解释"，不是"方法"。**

---

## 6. 附录

### 6.0 本文所有基线数字的生成脚本（已跑过，可复现）

```python
import numpy as np
rng = np.random.default_rng(0)

def resid(A, B):
    G = A.T @ A
    M = np.linalg.solve(G, A.T @ B)
    return np.linalg.norm(A @ M - B) / np.linalg.norm(B)

# null baseline：两表无关
for V, dh in [(3700,100), (1520,41), (7600,205), (2000,500), (400,200)]:
    A = rng.normal(size=(V,dh)); B = rng.normal(size=(V,dh))
    print(V, dh, round(resid(A,B),4), round(np.sqrt(1-dh/V),4))

# 正控：GA 严格成立
V, dh = 3700, 100
A = rng.normal(size=(V,dh)); R = rng.normal(size=(dh,dh))
print("tied      :", resid(A, A))       # ~1e-16
print("linear-rel:", resid(A, A@R))     # ~1e-16

# 校准曲线：GA + 噪声
base = A @ R; s = np.linalg.norm(base)/np.sqrt(base.size)
for nl in [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]:
    print(nl, round(resid(A, base + nl*s*rng.normal(size=base.shape)), 4))
```

### 6.1 记号速查

| 符号 | 含义 | 形状 |
|---|---|---|
| **GA** | **G**enerative **A**ssumption 生成式假设（**本手册起的简称，论文未命名**） | — |
| **B.1** | Linear Representation Hypothesis 线性表示假设（论文有命名） | — |
| $W_{in}$ | 输入嵌入表，第 $x$ 行 = 词 $x$ 的入场向量；代码 `embed_tokens.weight` | $\lvert V\rvert\times d_h$ |
| $W_{out}$ | 输出嵌入表（LM head），第 $x$ 行 = 词 $x$ 的退场向量；代码 `lm_head.weight` | $\lvert V\rvert\times d_h$ |
| $M=\beta W_a$ | 吸收了 $\beta$ 的线性映射（E1 的直接求解对象） | $d_h\times d_h$ |
| $\rho$ | 相对残差 $\|W_{out}M-W_{in}\|_F/\|W_{in}\|_F$；**几何含义 = 两个核的对齐度** | 标量 $\in[0,1]$ |
| $\rho_{\text{null}}$ | 随机基线 $\approx\sqrt{1-d_h/\lvert V\rvert}$（Qwen3-8B: 0.9864） | 标量 |
| $\sigma$ | GA 里的语义权重向量（隐变量） | $1\times\lvert V\rvert$ |
| $\ker(W)$ | 核 $\{\sigma:\sigma W=0\}$，维度 $\lvert V\rvert-d_h$；**GA ⟺ $\ker(W_{out})=\ker(W_{in})$** | — |

### 6.2 待填结果表（跑完填这里）

| 模型 | tied? | $\lvert V\rvert$ | $d_h$ | $\rho_{\text{null}}$ | $\rho_{\text{实测}}$ | $\rho$ (剔除未用 token) | $\rho_{\text{MLP}}$ | 结论 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | | | | | | | | |
| Qwen3-8B | | | 4096 | 0.9864 | | | | |
| Qwen3-14B | | | | | | | | |
| Llama-3.1-8B | | | | | | | | |

---

*本文的 null 基线、正控、校准曲线均由 numpy 实测（见 §6.0），非估计值。对真实模型权重的 $\rho$ 本文未测——那正是留给你的实验。*
