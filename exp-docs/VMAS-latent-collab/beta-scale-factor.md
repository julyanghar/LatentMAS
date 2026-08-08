# 尺度因子 β：论文怎么写的，代码怎么实现的，差在哪

> **这份文档回答什么**：LatentMAS 的对齐算子里那个 $\beta$ 到底是什么、为什么需要它、代码的实现和论文的公式差在哪一步，以及**这个差异什么时候真的会影响你的实验结果**。
>
> **起因**：`code-to-paper-mapping.md` §3.3 里那张对比表写得太简，"归一化基准"一列没说清**除的是谁**，容易误读。这份是展开版。
>
> **前置**：读过 `LatentMAS_论文总结.md` 的 §附录 A 或 `WA_ALI~1.MD` 更好，但不是必须——本文从范数讲起。

---

## 1. 一分钟版

| | |
|---|---|
| **$\beta$ 是什么** | 隐状态模长 ÷ 平均 token 嵌入模长。实测在 Qwen3-0.6B 上 **= 132** |
| **为什么需要它** | 末层隐状态比一个典型 token 嵌入**大两个数量级**，不缩放直接喂回去等于往序列里塞了个异常"响"的东西 |
| **最容易误解的点** | $\beta$ 含 $\lVert h\rVert$，所以 **$W_a$ 不是常数矩阵**，逐样本都在变。实现上必须拆成「固定矩阵 × 逐样本标量」 |
| **论文 vs 代码差在哪** | 只差那个标量的**分母**：论文除 $\lVert h\rVert$（过矩阵之前），代码除 $\lVert hM\rVert$（过矩阵之后） |
| **什么时候咬人** | **只在「开了 `--latent_space_realign` + 非 tied embedding 模型」时。** 默认配置下两版逐位相同 |

---

## 2. 概念梯子

### 2.1 范数就是模长

$h$ 是一个向量（末层隐状态，Qwen3-0.6B 上是 1024 维）。它的 **L2 范数**：

$$\lVert h\rVert=\sqrt{h_1^2+h_2^2+\cdots+h_d^2}$$

几何上就是**把向量看成从原点出发的箭头，箭头有多长**。

> 严格说"范数"是更大的概念——L1 范数是各分量绝对值之和、L∞ 是最大分量。**只有 L2 范数才等于模长。** LatentMAS 全程用 L2，代码里就是 `tensor.norm(dim=-1)`（PyTorch 默认 L2）。

### 2.2 为什么隐状态不能直接当输入嵌入

LatentMAS 的核心动作是：**取末层隐状态 $h$，当成下一步的输入嵌入喂回去**（不过 LM head、不采样）。

问题是这两种东西**尺度完全不在一个量级**：

- **输入嵌入** $W_{in,x}$：嵌入表里的一行，模型见惯了的东西
- **末层隐状态** $h$：经过全部 $L$ 层残差累加之后的产物

Transformer 的残差流是层层相加的，越深模长越大。实测（Qwen3-0.6B）：

```
平均 token 嵌入模长   avg = (1/|V|)·Σ_x ‖W_in,x‖  =   0.9251
末层隐状态模长                        ‖h‖         = 122.1708
```

**大 132 倍。** 直接喂回去，等于往序列里塞了一个比正常 token「响」两个数量级的东西——注意力分布会被它带偏。

### 2.3 所以 β 就是那个"倍数"

$$\beta=\frac{\lVert h\rVert}{\frac{1}{|V|}\sum_{x\in V}\lVert W_{in,x}\rVert}=\frac{\lVert h\rVert}{\text{avg}}$$

**白话**：$\beta$ 就是"这个隐状态比一个典型 token 嵌入大几倍"。把它除掉，就缩回了正常尺度。

论文把 $1/\beta$ 乘进对齐矩阵：

$$W_a=\frac{1}{\beta}\underbrace{\big(W_{out}^\top W_{out}+\lambda I\big)^{-1}W_{out}^\top W_{in}}_{\text{记作 }M}$$

---

## 3. ⭐ 关键误解：$W_a$ 不是常数矩阵

很自然会读成"$\beta$ 是一个作用在 $W_a$ 上的常数"。**但它不是。**

看 $\beta$ 的定义——它含 $\lVert h\rVert$，而 $h$ 是**每个样本、每一步隐空间自回归都不同**的隐状态。所以：

```
                    固定不变                       逐样本变
              ┌──────────────────┐          ┌──────────────┐
    W_a   =   │        M         │    ×     │     1 / β    │
              └──────────────────┘          └──────────────┘
                  算一次即可                   每步重算
```

### 3.1 但「拆成两步」不是被迫的妥协，是恒等变形

⚠️ **本节初版把因果讲错了，此处更正。**

先把代价算清楚——`M` 和 `W_a` 差**三个数量级**，别混为一谈：

| 量 | 表达式 | 代价（Qwen3-8B，$\lvert V\rvert=151936$，$d=4096$） | 频率 |
|---|---|---|---|
| $M$ | $(W_{out}^\top W_{out}+\lambda I)^{-1}W_{out}^\top W_{in}$ | $d^2\lvert V\rvert\approx 2.5\times10^{12}$ 次乘法 + 一次 $d\times d$ solve | **必须只算一次** |
| $W_a$ | $\frac{1}{\beta}M$ | $d^2\approx1.7\times10^{7}$（**只是标量乘矩阵**） | 逐样本重算完全做得到 |

**所以"逐样本重算 $W_a$ 不现实"是错的**——不现实的是重算 $M$，而 $W_a=\frac{1}{\beta}M$ 只需要标量缩放。

更进一步：因为**标量乘法与矩阵乘法可交换**，下面三种写法**逐位相同**：

$$\text{(a)}\ \ W_a=\tfrac{1}{\beta}M,\ \ e=hW_a
\qquad
\text{(b)}\ \ e=\tfrac{1}{\beta}(hM)
\qquad
\text{(c)}\ \ e=(hM)\cdot\frac{\text{avg}}{\lVert h\rVert}$$

代价：

```
(a)  d²（造 W_a） + d²（h @ W_a） = 2d²，且每步要分配一个 d×d 新矩阵
(b)  d²（h @ M）  + d（缩放）      ≈ d²
```

(b) 只是**省一半算力、免去反复分配显存**的写法。**不是妥协，不是近似，就是把标量挪了个位置。**

### 3.2 分母取错和拆分无关，而且**动机不明**

⚠️ **本文写过两版对"为什么会这样"的解释，都被推翻了，此处记录并作废：**

| 作废的说法 | 为什么站不住 |
|---|---|
| ~~"逐样本重算矩阵太贵，所以只能拆成两步"~~ | 贵的是造 $M$，不是造 $W_a=\frac1\beta M$（差三个数量级）。而且拆分是恒等变形，不需要用代价解释。见 §3.1 |
| ~~"拆完之后手边只剩 $hM$，所以顺手用了 $\lVert hM\rVert$"~~ | **$h$ 就在作用域里**——[models.py:206](../../models.py#L206) 的 `hidden_fp32` 是局部变量，第 209 行写 `hidden_fp32.norm(...)` 和写 `aligned.norm(...)` **代价完全一样**。技术上不成立 |

```python
206  hidden_fp32 = hidden.to(torch.float32)              # h 在这
207  aligned = torch.matmul(hidden_fp32, matrix)          # h·M
209  aligned_norm = aligned.norm(dim=-1, keepdim=True)    # 取了 ‖h·M‖
     #             ^^^^^^^ 换成 hidden_fp32 就是论文版，一样一行
```

**诚实的结论：不知道作者为什么这么写。** 可能有意（模长恒定更稳）、可能是习惯性写法、也可能论文的 $\beta$ 定义和代码从未对齐过。**没有证据能区分。**

因此本文**只陈述可证实的三条**，不再给动机：

1. 代码用 $\lVert hM\rVert$ 当分母 —— 读码可证（[models.py:209](../../models.py#L209)）
2. 论文闭式解等价于用 $\lVert h\rVert$ 当分母 —— 由正规方程推导可证
3. 二者只在 $M\ne I$ 时不等价 —— 代数可证（见 §6）

代码的**拆分本身完全正确**（[models.py:158-213](../../models.py#L158-L213)）；分歧只在取哪个模长这一件事上，且与拆分无因果关系。

---

## 4. 分歧点：分母是 $\lVert h\rVert$ 还是 $\lVert hM\rVert$

两版算出的**方向完全一样**（都是 $hM$），只差**乘多长**：

$$e_{\text{论文}}=\underbrace{(hM)}_{\text{方向}}\times\frac{\text{avg}}{\lVert h\rVert}
\qquad\qquad
e_{\text{代码}}=\underbrace{(hM)}_{\text{方向}}\times\frac{\text{avg}}{\lVert hM\rVert}$$

| 版本 | 分母取自 | 得到的模长 |
|---|---|---|
| **论文** | $\lVert h\rVert$ —— 过矩阵**之前** | $\lVert e\rVert=\text{avg}\cdot r$，**随样本浮动** |
| **代码** | $\lVert hM\rVert$ —— 过矩阵**之后** | $\lVert e\rVert=\text{avg}$，**恒定** |

其中 $r=\dfrac{\lVert hM\rVert}{\lVert h\rVert}$ 是矩阵 $M$ 对这个特定 $h$ 的**拉伸率**。

### 代码原文

[models.py:204-213](../../models.py#L204-L213)：

```python
def _apply_latent_realignment(self, hidden, model):
    matrix, target_norm = self._ensure_latent_realign_matrix(model, hidden.device, self.args)
    hidden_fp32 = hidden.to(torch.float32)
    aligned = torch.matmul(hidden_fp32, matrix)                    # h·M

    aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)   # ‖h·M‖  ← 分母用的是这个
    pre_aligned = aligned.detach().clone()
    self.pre_aligned = pre_aligned
    aligned = aligned * (target_norm / aligned_norm)               # × avg / ‖h·M‖
    return aligned.to(hidden.dtype)
```

`target_norm` 的来源在 [models.py:177](../../models.py#L177)：

```python
target_norm = input_weight.norm(dim=1).mean().detach()   # = avg
```

---

## 5. 差多少

代几个 $r$ 进去（$\text{avg}=0.9251$）：

| $r$（$M$ 的拉伸率） | 论文 $\lVert e\rVert=\text{avg}\cdot r$ | 代码 $\lVert e\rVert=\text{avg}$ |
|---|---|---|
| 0.5（$M$ 压缩一半） | 0.46 | **0.93** |
| 1.0（$M$ 保模长） | 0.93 | 0.93 |
| 2.0（$M$ 拉长一倍） | 1.85 | **0.93** |

**一句话**：

> 论文保留了「$M$ 把这个向量拉伸了多少」这个信息；**代码把它抹平了**，强制每一步潜在思维的模长恒等于 $\text{avg}$。

### 谁"对"

代码那版更**稳**（模长恒定，不会随样本飘），但它**不是论文那个优化目标的解**。论文求的是

$$\min_{W_a}\ \big\lVert \beta\,W_{out}W_a-W_{in}\big\rVert_F^2$$

令导数为零得正规方程，解出来才是 $W_a=\frac{1}{\beta}M$。**要严格复现 Theorem A.1 的 Wasserstein 上界数值，必须用论文那版**（分母取 $\lVert h\rVert$）。

---

## 6. ⭐ 但默认配置下两版完全等价

这是最实用的一条。看 [models.py:179-183](../../models.py#L179-L183)：

```python
if self.args.latent_space_realign:
    pass                              # 用真的 M
else:
    realign_matrix = torch.eye(...)   # M = I  ← 换成单位阵
```

不传 `--latent_space_realign` 时 $M=I$，于是 $\lVert hM\rVert=\lVert h\rVert$，$r\equiv1$，**两版逐位相同**。

| 配置 | $M$ | 两版是否等价 |
|---|---|---|
| **默认**（不传 flag） | $I$ | ✅ **完全相同**，分歧不存在 |
| 传 `--latent_space_realign` | 真的 $M$ | ❌ 分岔，差 $r$ 倍 |
| **tied embedding 模型**（Qwen3-0.6B / 4B） | $\approx I$ | ⚠️ 近似相同 |

最后一行的原因：tied 时 $W_{out}=W_{in}=W$，代进去

$$M=(W^\top W+\lambda I)^{-1}W^\top W\approx I$$

即**整个对齐算子塌成恒等映射**，$W_a$ 的全部作用只剩模长归一。（详见 `code-to-paper-mapping.md` §4.2。）

---

## 7. 对复现的影响：什么时候该改那三行

| 你的配置 | 要不要改代码 |
|---|---|
| 不传 `--latent_space_realign` | **不用**。两版等价 |
| Qwen3-0.6B / 4B（tied）+ 传 flag | 基本不用。$M\approx I$，$r\approx1$ |
| **Qwen3-8B / 14B（untied）+ 传 flag** | ⚠️ **要**。这是唯一真会分岔的组合 |

改法（若要严格对齐论文）：把 [models.py:209-212](../../models.py#L209-L212) 的分母从 `aligned.norm()` 换成 `hidden_fp32.norm()`：

```python
# 论文版
src_norm = hidden_fp32.norm(dim=-1, keepdim=True).clamp_min(1e-6)   # ‖h‖，不是 ‖h·M‖
aligned  = aligned * (target_norm / src_norm)
```

> ⚠️ **改之前先做等价性自检**：在 $M=I$ 的配置下，改前改后必须逐位相同（因为此时 $\lVert h\rVert=\lVert hM\rVert$）。这是个平凡边界 case，但正是它能抓出符号/维度写错。参见 memory 里那条「参照系必须独立」的教训。

---

## 8. 证据

| 结论 | 依据 | 级别 |
|---|---|---|
| $\beta=\lVert h\rVert/\text{avg}$，$W_a=\frac{1}{\beta}M$ | `LatentMAS_论文总结.md` L144、L147 转录自论文 | **论文原文** |
| $\text{avg}=0.9251$、$\lVert h\rVert=122.17$、$\beta=132.07$ | 本机加载 Qwen3-0.6B fp32 实测（`~/tmp/beta_check.py`） | **实测** |
| 代码分母取 $\lVert hM\rVert$ | 读 [models.py:209-212](../../models.py#L209-L212) | **静态读码** |
| 关 flag 时 $M=I$ | 读 [models.py:179-183](../../models.py#L179-L183) | **静态读码** |
| Qwen3-0.6B tied / 8B untied | 本机 `config.json` | **实测** |
| Qwen3-4B tied | 官方 config 记忆，本机无缓存 | **待核** |
| 「代码版不是论文优化目标的解」 | 由正规方程推导 | **数学可推导** |
