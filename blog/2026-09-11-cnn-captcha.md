---
title: 记一次训练 CNN 用于验证码识别
tags: ['AI']
---

学校选课系统登录需要验证码，是很简单的很有 2000s 风格的验证码：

![验证码](https://files.seeusercontent.com/2026/09/11/w7kY/005473.jpg)

为了自动选课以及确保选课期间被踢下线能自动重登录，因此花了一天折腾验证码识别。

<!-- truncate -->

## 数据准备

首先需要搞清楚验证码用了什么字体，这期间我发现多模态 LLM 在识别字体这方面表现非常非常差。似乎是因为模型太过专注于学习文字的语义了，而需要识别特定字体的场景是很少很少的。

最后是怎么找出来字体的呢？我打开电脑上的 WPS，敲了几个字符然后点开字体列表一个个试出来的......

总之找出了字体，接下来就是让 Codex 写脚本批量生成训练数据集了。

## 模型架构

模型也是让 LLM 写的，我的 prompt 大概是这样：

> 我造好了数据，在images目录下有000001.jpg 等，还有一个jsonl内容为{"file": "000001.jpg", "text": "ZNU3", "transforms": [...]}，请写代码训练模型，预期是模型可以在没有GPU的设备上跑，几秒内能出结果最好，我本机有一张 GPU 可用。

然后 Codex 就端上来了这么个东西：

```python
class ConvBNAct(nn.Sequential):
    """最普通的 CNN 基础块"""
    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseSeparableBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1
    ) -> None:
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.pointwise(self.depthwise(inputs))
        if self.use_residual:
            outputs = outputs + inputs
        return outputs


class CaptchaNet(nn.Module):
    """为固定四位验证码设计的轻量级多头分类网络。"""

    def __init__(
        self,
        num_characters: int = len(CHARACTERS),
        text_length: int = TEXT_LENGTH,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.num_characters = num_characters
        self.text_length = text_length
        self.features = nn.Sequential(
            ConvBNAct(1, 32),
            DepthwiseSeparableBlock(32, 48, stride=2),
            DepthwiseSeparableBlock(48, 48),
            DepthwiseSeparableBlock(48, 80, stride=2),
            DepthwiseSeparableBlock(80, 80),
            DepthwiseSeparableBlock(80, 128, stride=2),
            DepthwiseSeparableBlock(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d((2, 9))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 9, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, text_length * num_characters),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.features(inputs))
        logits = self.classifier(features)
        return logits.reshape(-1, self.text_length, self.num_characters)
```

整个网络的尺寸变化：

```
Input
[B, 1, 30, 70]

↓ 普通 3×3 Conv
[B, 32, 30, 70]

↓ DepthwiseSeparable stride=2
[B, 48, 15, 35]

↓ DepthwiseSeparable
[B, 48, 15, 35]

↓ DepthwiseSeparable stride=2
[B, 80, 8, 18]

↓ DepthwiseSeparable
[B, 80, 8, 18]

↓ DepthwiseSeparable stride=2
[B, 128, 4, 9]

↓ DepthwiseSeparable
[B, 128, 4, 9]

↓ AdaptiveAvgPool
[B, 128, 2, 9]

↓ Flatten
[B, 2304]

↓ Linear
[B, 256]

↓ Linear
[B, 128]

↓ reshape
[B, 4, 32]
```

从 `CaptchaNet` 内部可以看到，首先是一个最普通的 CNN 基础块 `ConvBNAct`（Convolution、Batch Normalization、Activation），从通道数为 1 的灰度图得到了 32 个通道的初步 feature map.

计算 CNN 卷积层输出大小 $N$ 的公式如下，其中 $W$ 为输入大小，$K$ 为卷积核大小(kernel size)，$P$ 为填充大小(padding)，$S$ 为步长(stride)：

$$
N=\left\lfloor \frac{W-K+2P}{S} \right\rfloor +1
$$

对于 `ConvBNAct`，若 stride 和 padding 都为 1：
$$
N=\frac{W-3+2}{1}+1=W
$$
故大小不变。

### Depthwise Separable Conv

随后连续用了 6 个`DepthwiseSeparableBlock`。这个结构的优势在哪呢？在于参数量小。

以 48 通道扩为 80 通道并尺寸减半为例，也就是上面的

````
[B, 48, 15, 35]
↓ DepthwiseSeparable stride=2
[B, 80, 8, 18]
````

这一步。

如果直接用 `Conv2d`，会写成

```python
nn.Conv2d(
    in_channels=48,
    out_channels=80,
    kernel_size=3,
    stride=2, # 尺寸减半
    padding=1,
    bias=False,
),
```

参数量是：

$$
48\times80\times3\times3 =34560
$$

而 Depthwise Separable Conv 把它拆成两步。

#### 第一步：Depthwise Conv

这里的核心要点在于 `groups` 参数：

```
nn.Conv2d(..., groups=in_channels, ...)
```

如果不加 `groups=in_channels` ，默认是一个输出通道与 48 个输入通道连接：

```
in channel 0..47 → 48 个 3×3 卷积核 → out channel 0
in channel 0..47 → 48 个 3×3 卷积核 → out channel 1
...
in channel 0..47 → 48 个 3×3 卷积核 → out channel 47
```

加上 `groups=in_channels` 后，每个输入通道自己做一个 3×3 卷积，不同通道之间互不影响。比如 48 个输入通道，就是：

```
in channel 0 → 一个 3×3 卷积核 → out channel 0
in channel 1 → 一个 3×3 卷积核 → out channel 1
...
in channel 47 → 一个 3×3 卷积核 → out channel 47
```

参数：
$$
48\times3\times3=432
$$

#### 第二步：Pointwise Conv

再通过一个 `1×1 Conv` 混合通道：

```
nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=1
)
```

参数：

$$
48\times80=3840
$$

#### Refinement block

在每次 stride=2 尺寸减半的 Downsampling 块后，都跟着一个输入输出通道数相同的 Refinement 块。当输入输出通道数相同且 stride=1 时会启用残差连接：

```python
self.use_residual = stride == 1 and in_channels == out_channels
```

若启用了残差连接

```python
if self.use_residual:
    outputs = outputs + inputs
```

也即
$$
y=F(x)+x
$$
残差的主要意义不是让模型“看之前的图片”，而是让网络更容易学习：

$$
F(x)
$$

只需要学习“相比原输入还需要修改什么”，而不是重新构造整个特征。

在 Downsampling 块后接 Refinement 块的好处在于，每次降采样之后网络都有机会先“消化”一下刚获得的新 feature representation.

- Downsampling 块负责把更大的图像区域压缩成一个 feature.
- Refinement 块负责在这个已经较大的感受野上融合邻域信息。

### Classifier

后面的结构比较常规

```python
self.pool = nn.AdaptiveAvgPool2d((2, 9))
self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(128 * 2 * 9, 256),
    nn.SiLU(inplace=True),
    nn.Dropout(dropout),
    nn.Linear(256, text_length * num_characters),
)
```

首先自适应平均池化到 `[B, 128, 2, 9]` 这个大小（所谓自适应也就是能通过目标大小 `(2,9)` 自动计算核大小而已），然后进入 classifier 展平，过一层线性层、SiLU 非线性层、Dropout, 然后就线性投影到最终结果了。最终的输出是模型认为第 $i$ 个字符为字符集中第 $j$ 个字符的概率。

### 参数量

总参数：
$$
432+3840=4272
$$

相比普通 CNN 的 $34560$ 差了大约 **8 倍**。

整个模型只有大约：

$$
669,632
$$

个参数，也就是约 **67 万参数**。

FP32 权重理论大小只有大约：

$$
669632\times4\approx2.68\text{ MB}
$$

所以非常小。

## 训练

训练也很常规。主要的发现是模型效果和 batch size 有很大关系。一开始用的 batch size 是 1024, 模型在生成出来的测试集上的准确率只能到 94% 左右，到真实环境中准确率会更低一点。后面把 batch size 改为 128, 测试集准确率来到了 97%.

另外还发现一个对准确率有影响的是训练数据的分布是否尽量和真实数据一致。比如训练数据里 4 个字符在图片中的绝对位置如果过于固定，那么模型可能不能适应字符位置偏移大、两个字符部分重叠的情况。似乎一般来说训练数据做得难一些效果会更好。
