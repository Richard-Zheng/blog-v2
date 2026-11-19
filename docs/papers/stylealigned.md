# Style Aligned

Style Aligned Image Generation via Shared Attention [arxiv:2312.02133](https://arxiv.org/abs/2312.02133) [alphaxiv](https://alphaxiv.org/abs/2312.02133)

Page 5 left panel:

> sharing the attention to only one image in the set results in diverse sets that share a similar style. However, in that case, we have noticed that the style of different images is not well aligned. **We suspect that this is due to low attention flow from the reference to the target image.**
>
> As illustrated in Fig. 4, **to enable balanced attention reference**, we normalize the queries $Q_t$ and keys $K_t$ of the target image using the queries $Q_r$ and keys $K_r$ of the reference image **using the adaptive normalization operation (AdaIN)** [\[26\]](https://arxiv.org/abs/1703.06868):
>
> $$\hat{Q}_t = \text{AdaIN}(Q_t, Q_r) \quad \hat{K}_t = \text{AdaIN}(K_t, K_r),$$
>
> where the AdaIn operation is given by:
>
> $$\text{AdaIN}(x, y) = \sigma(y)\left(\frac{x - \mu(x)}{\sigma(x)}\right) + \mu_y,$$
>
> and $\mu(x), \sigma(x) \in \mathbb{R}^{d_k}$ are the mean and the standard deviation of queries and keys across different pixels. Finally, our shared attention is given by
>
> $$\text{Attention}(\hat{Q}_t, K_{rt}^T, V_{rt}),$$
>
> where $K_{rt} = \begin{bmatrix} K_r \\ \hat{K}_t \end{bmatrix}$ and $V_{rt} = \begin{bmatrix} V_r \\ V_t \end{bmatrix}$.

后文似乎也没有对于为什么要用 `AdaIN` 的进一步分析。唯一的线索就是“We suspect that this is due to low attention flow from the reference to the target image.”，也就是在共享注意力的情况下，参考图像对目标图像的影响不够强。

现在就来简单分析一下。用下标 $r$ 表示 Reference（长度 $N_r$），下标 $t$ 表示 Target（长度 $N_t$），单头特征维度为 $d$。沿序列维度对每个通道做一阶均值与中心化扰动分解：

$$q_{i} = \mu_{q} + \tilde{q}_{i}, \qquad k_{j} = \mu_{k} + \tilde{k}_{j}$$

其中序列期望 $\mathbb{E}[\tilde{q}] = \mathbb{E}[\tilde{k}] = 0$，通道协方差分别为 $\Sigma_{q}, \Sigma_{k} \in \mathbb{R}^{d \times d}$。将 注意力分数 (logits) 展开为四项分量：

$$s(i,j) = \frac{1}{\sqrt{d}} q_{i}^\top k_{j} = \frac{1}{\sqrt{d}} \Big( \underbrace{\mu_{q}^\top \mu_{k}}_{\text{1. 均值内积}} + \underbrace{\tilde{q}_{i}^\top \mu_{k}}_{\text{2. Query 平移}} + \underbrace{\mu_{q}^\top \tilde{k}_{j}}_{\text{3. Key 静态偏置}} + \underbrace{\tilde{q}_{i}^\top \tilde{k}_{j}}_{\text{4. 内容匹配}} \Big)$$

经过 softmax 得到 attention 权重

$$
A(i,j) = \frac{\exp(s(i,j))}{\sum_{j'} \exp(s(i,j'))}
$$

第 1 项 $\mu_{q}^\top \mu_{k}$ 和索引无关，是常数，可以提取出来。在 softmax 时，分子分母同时存在，会被约去。

第 2 项 $\tilde{q}_{i}^\top \mu_{k}$ 只与 Query 索引 $i$ 有关。由于 softmax 是沿着 Key 维度计算的，$i$ 也视为常数，分子分母同时存在，也会被约去。

(待续)

以下是 LLM-generated 内容

---

在 DiT（Diffusion Transformer）中将 Reference 与 Target 沿序列维度拼接做联合自注意力（Joint / Shared Attention）时，$Q$ 和 $K$ 的统计分布失配会同时破坏 **Softmax 的跨分支平移不变性**、引发**指数级方差劫持**，并扭曲**分支内部的有效温度与显著性分布**。

**1. 拼接注意力的分块与配分函数解耦**

为避免与扩散时间步 $\tau \in [0, 1]$ 混淆，用下标 $r$ 表示 Reference（长度 $N_r$），下标 $t$ 表示 Target（长度 $N_t$），单头特征维度为 $d$。沿序列维度拼接后的投影矩阵及注意力分数（Logits）分块如下：

$$Q = \begin{bmatrix} Q_r \\ Q_t \end{bmatrix}, \quad K = \begin{bmatrix} K_r \\ K_t \end{bmatrix}, \quad S = \frac{1}{\sqrt{d}} Q K^\top = \frac{1}{\sqrt{d}} \begin{bmatrix} Q_r K_r^\top & Q_r K_t^\top \\ Q_t K_r^\top & Q_t K_t^\top \end{bmatrix} = \begin{bmatrix} S_{rr} & S_{rt} \\ S_{tr} & S_{tt} \end{bmatrix}$$

考察第 $i$ 个 Target token 的输出 $o_{t,i}$。分别定义其对 Reference 与对自身的局部配分函数（指数和）：

$$Z_{tr}(i) = \sum_{j=1}^{N_r} \exp\big(s_{tr}(i,j)\big), \qquad Z_{tt}(i) = \sum_{m=1}^{N_t} \exp\big(s_{tt}(i,m)\big)$$

引入跨分支总注意力占比 $\lambda_i = \frac{Z_{tr}(i)}{Z_{tr}(i) + Z_{tt}(i)} = \sigma\big(\log Z_{tr}(i) - \log Z_{tt}(i)\big)$（$\sigma$ 为 Sigmoid 函数），输出 $o_{t,i}$ 可严格拆解为**分支间门控竞争**与**分支内归一化聚合**两层结构：

$$o_{t,i} = \lambda_i \sum_{j=1}^{N_r} \underbrace{\frac{\exp\big(s_{tr}(i,j)\big)}{Z_{tr}(i)}}_{p_{tr}(j \mid i)} v_{r,j} + (1 - \lambda_i) \sum_{m=1}^{N_t} \underbrace{\frac{\exp\big(s_{tt}(i,m)\big)}{Z_{tt}(i)}}_{p_{tt}(m \mid i)} v_{t,m}$$

由此可见，$(Q_r, K_r)$ 与 $(Q_t, K_t)$ 的统计分布差异会从两个正交层面干扰生成：一是改变对数配分函数差 $\log Z_{tr}(i) - \log Z_{tt}(i)$，导致全局融合权重 $\lambda_i$ 失衡；二是改变条件概率 $p_{tr}(j \mid i)$ 的尖锐度与偏置，破坏局部语义对齐。

**2. DiT 中统计分布失配的物理来源**

* **信噪比与流形不对称**：在条件生成（如 FLUX Kontext、OminiControl 等）中，Reference 通常为干净潜变量（$\tau_r = 0$），而 Target 为含噪状态 $z_t = (1-\tau_t)x_t + \tau_t \epsilon$，两者的空间相关性与高频能量谱截然不同。
* **adaLN 调制参数差异**：DiT 在进入注意力线性层前执行 $h = \gamma(\tau) \odot \operatorname{LN}(z) + \beta(\tau)$；若两路采用不同的时间步条件（$\tau_r \neq \tau_t$），投影前的通道均值 $\beta$ 与缩放 $\gamma$ 直接不同，经 $W_Q, W_K$ 线性映射后产生不同的一阶与二阶矩。
* **图像内容与风格偏置**：即使处于相同噪声水平，不同图像的全局色调、对比度及前景占比也会导致序列维度上的通道均值与方差出现偏移。

**3. Logits 的四项展开与平移不变性破缺**

沿序列维度对每个通道做一阶均值与中心化扰动分解：

$$q_{t,i} = \mu_{q_t} + \tilde{q}_{t,i}, \qquad k_{t,m} = \mu_{k_t} + \tilde{k}_{t,m}, \qquad k_{r,j} = \mu_{k_r} + \tilde{k}_{r,j}$$

其中序列期望 $\mathbb{E}[\tilde{q}_t] = \mathbb{E}[\tilde{k}_t] = \mathbb{E}[\tilde{k}_r] = 0$，通道协方差分别为 $\Sigma_{q_t}, \Sigma_{k_t}, \Sigma_{k_r} \in \mathbb{R}^{d \times d}$。将 $s_{tt}(i,m)$ 与 $s_{tr}(i,j)$ 展开为四项物理分量：

$$s_{tt}(i,m) = \frac{1}{\sqrt{d}} \Big( \underbrace{\mu_{q_t}^\top \mu_{k_t}}_{\text{I. 全局均值内积}} + \underbrace{\tilde{q}_{t,i}^\top \mu_{k_t}}_{\text{II. Query 相关平移}} + \underbrace{\mu_{q_t}^\top \tilde{k}_{t,m}}_{\text{III. Key 静态偏置}} + \underbrace{\tilde{q}_{t,i}^\top \tilde{k}_{t,m}}_{\text{IV. 局部内容匹配}} \Big)$$

$$s_{tr}(i,j) = \frac{1}{\sqrt{d}} \Big( \underbrace{\mu_{q_t}^\top \mu_{k_r}}_{\text{I'. 跨流形均值内积}} + \underbrace{\tilde{q}_{t,i}^\top \mu_{k_r}}_{\text{II'. 跨流形 Query 平移}} + \underbrace{\mu_{q_t}^\top \tilde{k}_{r,j}}_{\text{III'. Ref Key 静态偏置}} + \underbrace{\tilde{q}_{t,i}^\top \tilde{k}_{r,j}}_{\text{IV'. 跨流形内容匹配}} \Big)$$

这里存在一个关键机制差异：在标准单序列自注意力中，第 I 项和第 II 项均与 Key 索引无关，由于 $\operatorname{Softmax}(s + c\mathbf{1}) = \operatorname{Softmax}(s)$，Key 的均值 $\mu_k$ 会被分子分母完全约去。但在拼接序列中，$K_r$ 子块的平移量为 $c_r(i) = \frac{1}{\sqrt{d}} q_{t,i}^\top \mu_{k_r}$，而 $K_t$ 子块的平移量为 $c_t(i) = \frac{1}{\sqrt{d}} q_{t,i}^\top \mu_{k_t}$；只要 $\mu_{k_r} \neq \mu_{k_t}$，**分段常数平移就会打破全局 Softmax 的平移不变性**。

**4. 跨分支门控 $\lambda_i$ 的二阶累积量推导（均值偏移与方差劫持）**

固定查询向量 $q_{t,i}$，将 Key 视为来自各自经验分布的样本，则 $s_{tr}(i,j)$ 的条件均值和条件方差分别为：

$$m_{tr}(i) = \frac{1}{\sqrt{d}} q_{t,i}^\top \mu_{k_r}, \qquad v_{tr}(i) = \frac{1}{d} q_{t,i}^\top \Sigma_{k_r} q_{t,i}$$

当头维度 $d$ 较大时，由中心极限定理可将内积近似为高斯分布 $s \sim \mathcal{N}(m, v)$。利用高斯分布的矩母函数 $\mathbb{E}[e^s] = \exp(m + \frac{1}{2}v)$，对配分函数取对数做二阶累积量（Cumulant）展开：

$$\log Z_{tr}(i) \approx \log N_r + \frac{1}{\sqrt{d}} q_{t,i}^\top \mu_{k_r} + \frac{1}{2d} q_{t,i}^\top \Sigma_{k_r} q_{t,i}$$

将其与 $\log Z_{tt}(i)$ 相减，得到决定跨分支注意力分配的对数几率（Log-odds）：

$$\operatorname{logit}(\lambda_i) = \log \frac{\lambda_i}{1 - \lambda_i} \approx \underbrace{\log \frac{N_r}{N_t}}_{\text{序列长度偏置}} + \underbrace{\frac{1}{\sqrt{d}} q_{t,i}^\top (\mu_{k_r} - \mu_{k_t})}_{\text{一阶 Key 均值偏移}} + \underbrace{\frac{1}{2d} q_{t,i}^\top (\Sigma_{k_r} - \Sigma_{k_t}) q_{t,i}}_{\text{二阶 Key 协方差偏移}}$$

进一步对 $q_{t,i}$ 求空间期望，得到全局平均对数几率：

$$\mathbb{E}_{q_t}\big[\operatorname{logit}(\lambda_i)\big] \approx \log \frac{N_r}{N_t} + \frac{1}{\sqrt{d}} \mu_{q_t}^\top (\mu_{k_r} - \mu_{k_t}) + \frac{1}{2d}\Big[\operatorname{Tr}\big(\Sigma_{q_t}(\Sigma_{k_r} - \Sigma_{k_t})\big) + \mu_{q_t}^\top (\Sigma_{k_r} - \Sigma_{k_t}) \mu_{q_t}\Big]$$

* **同源均值亲和度抑制**：由于 $Q_t, K_t$ 同源（共享同一图像潜变量与时间步调制），预训练模型往往使 $\mu_{q_t}^\top \mu_{k_t}$ 显著高于跨分布的 $\mu_{q_t}^\top \mu_{k_r}$，系统性压低 $\lambda_i$，造成参考特征注入不足；同时残差项 $\frac{1}{\sqrt{d}}\tilde{q}_{t,i}^\top(\mu_{k_r}-\mu_{k_t})$ 会产生与局部语义无关的伪空间掩码。
* **指数凸性引发的方差劫持**：由 Jensen 不等式可知 $\exp(\cdot)$ 严格凸，方差更大的一侧会产生更极端的右尾高分并主导指数和（体现在 $+\frac{1}{2}v$ 项）。哪一路 Key 的方向方差 $q_{t,i}^\top \Sigma_k q_{t,i}$ 偏大，哪一路就会在分母竞争中“赢家通吃”；在双向全注意力下，若含噪 $K_t$ 方差偏大，还会反向通过 $S_{rt}$ 污染干净的 Reference 特征流。

**5. 分支内分布 $p_{tr}(j \mid i)$ 的扭曲（$Q_t$ 与 $Q_r$ 失配效应）**

在 Reference 分支内部计算 $p_{tr}(j \mid i) = \frac{\exp(s_{tr}(i,j))}{Z_{tr}(i)}$ 时，与 $j$ 无关的第 I'、II' 项被约去，仅剩第 III' 项和第 IV' 项。令 $\sigma_x \in \mathbb{R}^d$ 为各通道标准差，写出标准化变量 $\hat{x}_c = \tilde{x}_c / \sigma_{x,c}$：

$$p_{tr}(j \mid i) = \operatorname{Softmax}_j \left( \underbrace{\frac{1}{\sqrt{d}} \sum_{c=1}^{d} \mu_{q_t, c} \sigma_{k_r, c} \hat{k}_{r,j,c}}_{\text{静态 Attention Sink 项}} + \underbrace{\frac{1}{\sqrt{d}} \sum_{c=1}^{d} (\sigma_{q_t, c} \sigma_{k_r, c}) \, \hat{q}_{t,i,c} \hat{k}_{r,j,c}}_{\text{各向异性逆温度加权匹配项}} \right)$$

对比 Reference 原生自注意力 $p_{rr}(j \mid i)$（由 $\mu_{q_r}, \sigma_{q_r}$ 驱动），用 $Q_t$ 去查询 $K_r$ 会引入两类失真：

* **静态 Sink 坍缩（$\mu_{q_t} \neq \mu_{q_r}$）**：第一项与查询位置 $i$ 无关，当含噪调制引入较大的 $\mu_{q_t}$ 且方向偏离 $\mu_{q_r}$ 时，少数偶然与 $\mu_{q_t}$ 对齐的参考 token 会获得极高静态偏置，导致所有 Target 位置坍缩式地关注同一批背景或边缘噪点。
* **有效温度失真（$\sigma_{q_t} \neq \sigma_{q_r}$）**：乘积 $\beta_{\text{eff}, c} = \sigma_{q_t, c}\sigma_{k_r, c}$ 决定了各通道的有效逆温度；若 $\sigma_{q_t}$ 整体偏小，注意力熵增大并退化为均匀平均（生成纹理模糊）；若 $\sigma_{q_t}$ 偏大，分布过度尖锐化（引发生硬的块状粘贴伪影）。

**6. 为什么现代 DiT 的 QK-RMSNorm 与 RoPE 无法自动消除该问题？**

* **QK-RMSNorm 与序列统计正交**：SD3、FLUX 等模型对每个 token 沿通道维 $d$ 施加 RMSNorm（设缩放权重为单位向量），仅将单向量约束在半径为 $\sqrt{d}$ 的超球面上（$\Vert{}\bar{k}\Vert{}_2^2 = d$）。沿序列维求期望可得球面能量守恒关系：

$$\mathbb{E}\big[\Vert{}\bar{k}\Vert{}_2^2\big] = \Vert{}\mu_{\bar{k}}\Vert{}_2^2 + \operatorname{Tr}(\Sigma_{\bar{k}}) = d \implies \operatorname{Tr}(\Sigma_{\bar{k}}) = d - \Vert{}\mu_{\bar{k}}\Vert{}_2^2$$

RMSNorm 无法约束球面点云的**均值轴向 $\mu_{\bar{k}}$** 与**锥体张角（窄化程度 $\Vert{}\mu_{\bar{k}}\Vert{}_2$）**。例如当 adaLN 的共享偏置 $\beta(\tau)$ 较强时，token 聚集在窄锥内（$\Vert{}\mu_{\bar{k}}\Vert{}_2$ 增大、$\operatorname{Tr}(\Sigma_{\bar{k}})$ 减小），球面上的均值差 $\mu_{\bar{k}_r} - \mu_{\bar{k}_t}$ 和协方差差 $\Sigma_{\bar{k}_r} - \Sigma_{\bar{k}_t}$ 依然显著存在。

* **RoPE 与非零均值的耦合波纹**：施加旋转位置编码 $R(p)$ 后，跨分支内积的一阶项变为 $\frac{1}{\sqrt{d}} \mu_{q_t}^\top R(p_j - p_i) \mu_{k_r}$。在 $S_{tt}$ 对角线附近相对位移接近零（$R(0) = I$），均值完全相干叠加；而在 $S_{tr}$ 中若引入时间/帧轴偏移 $\Delta t \neq 0$ 或空间拼接偏移，旋转矩阵会改变 $\mu_{q_t}$ 与 $\mu_{k_r}$ 的夹角，不仅改变跨分支均值偏置，还会在空间上叠加一层随相对坐标周期性震荡的位置伪偏置。

**7. 由数学式导出的针对性修正方案**

* **序列维 QK-AdaIN（如 StyleAligned）**：沿序列维对齐一阶与二阶统计量，令 $\hat{Q}_t = \sigma(Q_r)\odot \frac{Q_t - \mu(Q_t)}{\sigma(Q_t)} + \mu(Q_r)$ 及 $\hat{K}_t = \sigma(K_r)\odot \frac{K_t - \mu(K_t)}{\sigma(K_t)} + \mu(K_r)$，直接令 $\mu_{\hat{k}_t} = \mu_{k_r}$、对角方差一致，并将 $p_{tr}$ 的有效温度还原至原生流形。
* **RoPE 前序列中心化（Key/Query Centering）**：在施加 RoPE 之前沿序列维减去均值（$K \leftarrow K - \mu_k$），可同时归零跨分支一阶平移差、静态 Sink 项 $\mu_q^\top \tilde{k}$ 以及 RoPE 旋转均值产生的空间伪波纹。
* **Logit 门控补偿与非对称掩码**：在 $S_{tr}$ 上显式加上随时间步衰减的偏置 $\log \gamma(\tau)$ 以抵消 $\frac{1}{2d} q_{t,i}^\top(\Sigma_{k_r} - \Sigma_{k_t})q_{t,i} + \log\frac{N_r}{N_t}$，或将 $S_{rt}$ 掩码置为 $-\infty$ 以阻断含噪 Target 对 Reference 的反向污染。