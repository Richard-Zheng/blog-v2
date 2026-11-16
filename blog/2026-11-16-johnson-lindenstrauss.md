---
title: Johnson-Lindenstrauss 引理
tags: ['AI']
---

确实是很神奇的一个引理。不过学完以后暂时还没想到要怎么用。

主要参考苏剑林的博客 [让人惊叹的Johnson-Lindenstrauss引理：理论篇](https://www.spaces.ac.cn/archives/8679) 和 [让人惊叹的Johnson-Lindenstrauss引理：应用篇](https://www.spaces.ac.cn/archives/8706)

<!-- truncate -->

## 证明

**单位模引理**：随机取向量 $\boldsymbol{u}\in\mathbb{R}^{n}$ 每个分量独立重复采样自 $\mathcal{N}(0,1/n)$，那么它的模长 $\|\boldsymbol{u}\|$ **大概率**接近 1。有多大概率呢？设 $\varepsilon \in (0,1)$ 为 $\|\boldsymbol{u}\|$ 偏移 1 的距离，则有

$$
P(|\Vert \boldsymbol{u}\Vert^2 - 1| \geq \varepsilon) \leq 2\exp\left(-\frac{\varepsilon^2 n}{8}\right)
$$

随着 $n$ 的增大，保距 $\varepsilon$ 失败的概率指数级下降，趋近于 0.

**正交性引理**：设 $\boldsymbol{u},\boldsymbol{v}\in\mathbb{R}^n$ 是独立重复采样自 $\mathcal{N}(0,1/n)$ 的两个向量， $\varepsilon\in(0,1)$ 是给定常数，那么我们有

$$
P(|\langle \boldsymbol{u}, \boldsymbol{v}\rangle| \geq \varepsilon) \leq 4\exp\left(-\frac{\varepsilon^2 n}{8}\right)
$$

也就是两个向量的点积**大概率**很接近 0。同样，随着 $n$ 的增大，点积超过 $\varepsilon$ 的概率指数级下降。

现在证明 Johnson-Lindenstrauss 引理：

**数学版 JL 引理**：给定 $N$ 个向量 $\boldsymbol{v}_1,\boldsymbol{v}_2,⋯,\boldsymbol{v}_N \in \mathbb{R}^m$ 和 $n > \frac{24\log N}{\varepsilon^2}$ ，而随机矩阵 $A\in\mathbb{R}^{n\times m}$ 独立重复采样自 $\mathcal{N}(0,1/n)$ ，$\varepsilon\in(0,1)$ 是给定常数，那么至少有 $N-1/N$ 的概率，使得对于所有的 $i \neq j$ ，都成立

$$
(1-\varepsilon)\Vert \boldsymbol{v}_i - \boldsymbol{v}_j\Vert^2 \leq \Vert A\boldsymbol{v}_i - A \boldsymbol{v}_j\Vert^2 \leq (1+\varepsilon)\Vert \boldsymbol{v}_i - \boldsymbol{v}_j\Vert^2
$$

证明：

令单位向量 $\boldsymbol{u}=\frac{\boldsymbol{v}_i - \boldsymbol{v}_j}{\Vert \boldsymbol{v}_i - \boldsymbol{v}_j\Vert}$. 对于单位向量 $\boldsymbol{u}$，$A\boldsymbol{u}$ 的每个分量也是独立重复采样自 $\mathcal{N}(0,1/n)$ 的。（比较显然好理解就不证明了）利用“单位模引理”得到

$$
P\left(\left|\left\Vert \frac{A(\boldsymbol{v}_i - \boldsymbol{v}_j)}{\Vert \boldsymbol{v}_i - \boldsymbol{v}_j\Vert}\right\Vert^2 - 1\right| \geq \varepsilon\right) \leq 2\exp\left(-\frac{\varepsilon^2 n}{8}\right)
$$

此结果对于任意 $i \neq j$ 都成立，遍历所有的 $i \neq j$ 的组合（共 $\binom{N}{2} = \frac{N(N-1)}{2}$ 个），至少有一项保距失败的概率不超过

$$
P\left(\exists (i,j):\,\left|\left\Vert \frac{A(\boldsymbol{v}_i - \boldsymbol{v}_j)}{\Vert \boldsymbol{v}_i - \boldsymbol{v}_j\Vert}\right\Vert^2 - 1\right| \geq \varepsilon\right) \leq {N\choose 2} \cdot 2 \exp\left(-\frac{\varepsilon^2 n}{8}\right)
$$

对其取反，全部保距成功的概率至少为

$$
1 - {N\choose 2} \cdot 2\exp\left(-\frac{\varepsilon^2 n}{8}\right) = 1 - N(N-1)\exp\left(-\frac{\varepsilon^2 n}{8}\right)
$$

下一步很关键：代入 $n > \frac{24\log N}{\varepsilon^2}$，得到

$$
1 - N(N-1)\exp\left(-\frac{\varepsilon^2 n}{8}\right)\geq 1 - N(N-1)N^{-3}\geq 1-N^{-1}
$$

也就是至少有 $\frac{N-1}{N}$ 的概率使得所有的 $i \neq j$ 都满足

$$
P\left(\forall (i,j):\,\left|\left\Vert \frac{A(\boldsymbol{v}_i - \boldsymbol{v}_j)}{\Vert \boldsymbol{v}_i - \boldsymbol{v}_j\Vert}\right\Vert^2 - 1\right| \leq \varepsilon\right) \geq \frac{N-1}{N}
$$

$P$ 里面的式子就是

$$
(1-\varepsilon)\Vert \boldsymbol{v}_i - \boldsymbol{v}_j\Vert^2 \leq \Vert A\boldsymbol{v}_i - A \boldsymbol{v}_j\Vert^2 \leq (1+\varepsilon)\Vert \boldsymbol{v}_i - \boldsymbol{v}_j\Vert^2
$$

内积版 JL 引理：给定 $N$ 个单位向量 $\boldsymbol{v}_1,\boldsymbol{v}_2,⋯,\boldsymbol{v}_N \in \mathbb{R}^m$ 和 $n > \frac{24\log N}{\varepsilon^2}$ ，而随机矩阵 $A\in\mathbb{R}^{n\times m}$ 独立重复采样自 $\mathcal{N}(0,1/n)$ ，$\varepsilon\in(0,1)$ 是给定常数，那么至少有 $N-1/N$ 的概率，使得对于所有的 $i \neq j$ ，都成立

$$
|\langle A\boldsymbol{v}_i, A\boldsymbol{v}_j\rangle - \langle \boldsymbol{v}_i, \boldsymbol{v}_j\rangle| \leq \varepsilon
$$

参考上面的数学版 JL 引理证明，用单位向量 $\boldsymbol{u}=\frac{\boldsymbol{v}_i + \boldsymbol{v}_j}{\Vert \boldsymbol{v}_i + \boldsymbol{v}_j\Vert}$ 过一遍，保距失败的概率也为 $\frac{N-1}{N}$，那么

$$
\begin{aligned} 
(1-\varepsilon)\Vert \boldsymbol{v}_i - \boldsymbol{v}_j\Vert^2 \leq \Vert A\boldsymbol{v}_i - A \boldsymbol{v}_j\Vert^2 \leq (1+\varepsilon)\Vert \boldsymbol{v}_i - \boldsymbol{v}_j\Vert^2 \\ 
(1-\varepsilon)\Vert \boldsymbol{v}_i + \boldsymbol{v}_j\Vert^2 \leq \Vert A\boldsymbol{v}_i + A \boldsymbol{v}_j\Vert^2 \leq (1+\varepsilon)\Vert \boldsymbol{v}_i + \boldsymbol{v}_j\Vert^2 
\end{aligned}
$$

同时成立的概率至少为 $\frac{N-2}{N}$ 的概率使得两者同时成立。将第一个式子乘以 -1 加到第二个式子，并利用内积展开公式

$$
\begin{aligned}
&\Vert{}\boldsymbol{x} + \boldsymbol{y}\Vert{}^2 = \Vert{}\boldsymbol{x}\Vert{}^2 + \Vert{}\boldsymbol{y}\Vert{}^2 + 2\langle \boldsymbol{x}, \boldsymbol{y} \rangle \\
&\Vert{}\boldsymbol{x} - \boldsymbol{y}\Vert{}^2 = \Vert{}\boldsymbol{x}\Vert{}^2 + \Vert{}\boldsymbol{y}\Vert{}^2 - 2\langle \boldsymbol{x}, \boldsymbol{y} \rangle
\end{aligned}
$$

中间项：

$$
\Vert{}A\boldsymbol{v}_i + A\boldsymbol{v}_j\Vert{}^2 - \Vert{}A\boldsymbol{v}_i - A\boldsymbol{v}_j\Vert{}^2 = 4\langle A\boldsymbol{v}_i, A\boldsymbol{v}_j \rangle
$$

左边项：

$$
\begin{aligned} &(1-\varepsilon)\Vert{}\boldsymbol{v}_i + \boldsymbol{v}_j\Vert{}^2 - (1+\varepsilon)\Vert{}\boldsymbol{v}_i - \boldsymbol{v}_j\Vert{}^2 \\ &= \underbrace{\left(\Vert{}\boldsymbol{v}_i + \boldsymbol{v}_j\Vert{}^2 - \Vert{}\boldsymbol{v}_i - \boldsymbol{v}_j\Vert{}^2\right)}_{4\langle \boldsymbol{v}_i, \boldsymbol{v}_j \rangle} - \varepsilon\underbrace{\left(\Vert{}\boldsymbol{v}_i + \boldsymbol{v}_j\Vert{}^2 + \Vert{}\boldsymbol{v}_i - \boldsymbol{v}_j\Vert{}^2\right)}_{2(\Vert{}\boldsymbol{v}_i\Vert{}^2 + \Vert{}\boldsymbol{v}_j\Vert{}^2) \text{（平行四边形法则）}} \\ &= 4\langle \boldsymbol{v}_i, \boldsymbol{v}_j \rangle - 2\varepsilon(\Vert{}\boldsymbol{v}_i\Vert{}^2 + \Vert{}\boldsymbol{v}_j\Vert{}^2) \end{aligned}
$$


右边项（同理，只是 $-\varepsilon$ 变成 $+\varepsilon$）：

$$
(1+\varepsilon)\Vert{}\boldsymbol{v}_i + \boldsymbol{v}_j\Vert{}^2 - (1-\varepsilon)\Vert{}\boldsymbol{v}_i - \boldsymbol{v}_j\Vert{}^2 = 4\langle \boldsymbol{v}_i, \boldsymbol{v}_j \rangle + 2\varepsilon(\Vert{}\boldsymbol{v}_i\Vert{}^2 + \Vert{}\boldsymbol{v}_j\Vert{}^2)
$$

引理前提中限定了 $v_1, v_2, \dots, v_N$ 都是**单位向量**，因此：


$$\Vert{}v_i\Vert{}^2 = 1, \quad \Vert{}v_j\Vert{}^2 = 1 \implies \Vert{}v_i\Vert{}^2 + \Vert{}v_j\Vert{}^2 = 2$$

把 $\Vert{}v_i\Vert{}^2 + \Vert{}v_j\Vert{}^2 = 2$ 代入左右两边：


$$
4\langle v_i, v_j \rangle - 4\varepsilon \le 4\langle Av_i, Av_j \rangle \le 4\langle v_i, v_j \rangle + 4\varepsilon
$$

不等式两边同时除以 $4$，再减去 $\langle v_i, v_j \rangle$：


$$
-\varepsilon \le \langle Av_i, Av_j \rangle - \langle v_i, v_j \rangle \le \varepsilon
$$

写成绝对值形式，即证得：

$$
\left\vert{}\langle Av_i, Av_j \rangle - \langle v_i, v_j \rangle\right\vert{} \le \varepsilon
$$
