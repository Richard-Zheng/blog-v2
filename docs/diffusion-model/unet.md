---
sidebar_position: 5
---

# UNet 代码整理

pipeline_stable_diffusion.py 中 `__call__` 调用unet的地方位于Denoising loop

```python
# 7. Denoising loop
num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
self._num_timesteps = len(timesteps)
with self.progress_bar(total=num_inference_steps) as progress_bar:
    for i, t in enumerate(timesteps):
        if self.interrupt:
            continue

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
        if hasattr(self.scheduler, "scale_model_input"):
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=self.cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
```

查看`self.unet`的定义，来到 unet_2d_condition.py , `UNet2DConditionModel`的forward方法：

```py
# 1. time
t_emb = self.get_time_embed(sample=sample, timestep=timestep)
emb = self.time_embedding(t_emb, timestep_cond)

class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
if class_emb is not None:
    if self.config.class_embeddings_concat:
        emb = torch.cat([emb, class_emb], dim=-1)
    else:
        emb = emb + class_emb

aug_emb = self.get_aug_embed(
    emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
)
if self.config.addition_embed_type == "image_hint":
    aug_emb, hint = aug_emb
    sample = torch.cat([sample, hint], dim=1)

emb = emb + aug_emb if aug_emb is not None else emb

if self.time_embed_act is not None:
    emb = self.time_embed_act(emb)
```

好多看上去是为了兼容性的boilerplate……普通的SD1.5就只是把time embedding拿到，然后time_embedding方法把dimension对好，然后这个就是emb，也是后面的temb。（为什么要搞这么多个名字啊！）因为我们用的Classifier-free guidance, 所以class embedding就是None, 不用管。

```
encoder_hidden_states = self.process_encoder_hidden_states(
    encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
)
```

`process_encoder_hidden_states` prepares the text/image conditioning (the `encoder_hidden_states`) before they are used in the UNet's cross-attention layers.

Its primary job is to **resize** or **transform** the conditioning features to match the UNet's internal dimensions using a projection layer (encoder_hid_proj).

```
# 2. pre-process
sample = self.conv_in(sample)
```

```
> print(self.conv_in)
Conv2d(4, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
```

That is the **very first layer** of the UNet. Here is what those numbers tell you about how the model starts processing data:

- **`4` (Input Channels)**: This matches the standard latent space used by Stable Diffusion. Instead of 3 channels (RGB), the VAE encodes images into 4 latent channels.
- **`320` (Output Channels)**: This is the feature width of the first UNet block. The model takes those 4 latent channels and "projects" them into 320 deeper features to start looking for patterns.
- `kernel_size=(3, 3), padding=(1, 1)`: These ensure that the spatial resolution (height and width) stays exactly the same during this first step.

This layer is what turns the "noisy pixels" into high-dimensional "features".

还有一些奇奇怪怪的看不懂，也用不上

```py
down_block_res_samples = (sample,)
for downsample_block in self.down_blocks:
    if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
        # For t2i-adapter CrossAttnDownBlock2D
        additional_residuals = {}
        if is_adapter and len(down_intrablock_additional_residuals) > 0:
            additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

        sample, res_samples = downsample_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            encoder_attention_mask=encoder_attention_mask,
            **additional_residuals,
        )
    else:
        sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
        if is_adapter and len(down_intrablock_additional_residuals) > 0:
            sample += down_intrablock_additional_residuals.pop(0)

    down_block_res_samples += res_samples
```

这里是重点。遍历down_blocks然后一个个跑，会根据是不是cross_attention输入不同的参数。

## 初始化阶段(`__init__`)

blocks的定义都在 unet_2d_blocks.py 中，在初始化`UNet2DConditionModel`的时候会调用这里的`get_down_block`函数

SD1.5的架构和默认参数里的一样：

```
down_block_types: Tuple[str, ...] = (
    "CrossAttnDownBlock2D",
    "CrossAttnDownBlock2D",
    "CrossAttnDownBlock2D",
    "DownBlock2D",
),
mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
up_block_types: Tuple[str, ...] = (
    "UpBlock2D",
    "CrossAttnUpBlock2D",
    "CrossAttnUpBlock2D",
    "CrossAttnUpBlock2D",
),
```

所以先看`CrossAttnDownBlock2D`，它的实例在get_down_block函数中被创建：

```py
elif down_block_type == "CrossAttnDownBlock2D":
    if cross_attention_dim is None:
        raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock2D")
    return CrossAttnDownBlock2D(
        num_layers=num_layers,
        transformer_layers_per_block=transformer_layers_per_block,
        in_channels=in_channels,
        out_channels=out_channels,
        temb_channels=temb_channels,
        dropout=dropout,
        add_downsample=add_downsample,
        resnet_eps=resnet_eps,
        resnet_act_fn=resnet_act_fn,
        resnet_groups=resnet_groups,
        downsample_padding=downsample_padding,
        cross_attention_dim=cross_attention_dim,
        num_attention_heads=num_attention_heads,
        dual_cross_attention=dual_cross_attention,
        use_linear_projection=use_linear_projection,
        only_cross_attention=only_cross_attention,
        upcast_attention=upcast_attention,
        resnet_time_scale_shift=resnet_time_scale_shift,
        attention_type=attention_type,
    )
```

num_layers 为 2, num_attention_heads 为8, 见[config](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/unet/config.json#L24)。

来到`CrossAttnDownBlock2D`的`__init__`

```
for i in range(num_layers):
    in_channels = in_channels if i == 0 else out_channels
    resnets.append(
        ResnetBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            eps=resnet_eps,
            groups=resnet_groups,
            dropout=dropout,
            time_embedding_norm=resnet_time_scale_shift,
            non_linearity=resnet_act_fn,
            output_scale_factor=output_scale_factor,
            pre_norm=resnet_pre_norm,
        )
    )
    if not dual_cross_attention:
        attentions.append(
            Transformer2DModel(
                num_attention_heads,
                out_channels // num_attention_heads,
                in_channels=out_channels,
                num_layers=transformer_layers_per_block[i],
                cross_attention_dim=cross_attention_dim,
                norm_num_groups=resnet_groups,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention,
                upcast_attention=upcast_attention,
                attention_type=attention_type,
            )
        )
    else:
        attentions.append(
            DualTransformer2DModel(
                num_attention_heads,
                out_channels // num_attention_heads,
                in_channels=out_channels,
                num_layers=1,
                cross_attention_dim=cross_attention_dim,
                norm_num_groups=resnet_groups,
            )
        )
self.attentions = nn.ModuleList(attentions)
self.resnets = nn.ModuleList(resnets)
```

可以看出，每个 num_layers 有一个resnet一个attention，都会被分别放到两个数组attentions和resnets里。每层的输入通道数等于上一层的输出通道数，而输出通道数一直是不变的。

`dual_cross_attention` is a boolean flag used to switch between a standard transformer block and a specialized **Dual Transformer** block.It is primarily used in models like **UniDiffuser**. Unlike standard Stable Diffusion which mostly focuses on text-to-image, UniDiffuser is a "bi-modal" model.

所以对于SD1.5来说`dual_cross_attention == False`。`transformer_layers_per_block`似乎没有指定，所以按照默认参数就是1。Transformer2DModel前两个参数是注意力头数量和每个注意力头的通道数。输出通道数没有指定，默认和输入通道数一样。

点进Transformer2DModel会来到 transformer_2d.py ，——init——函数：

```
self.is_input_continuous = (in_channels is not None) and (patch_size is None)
```

故is_input_continuous为True. norm_type未指定默认为layer_norm.

```
# 2. Initialize the right blocks.
# These functions follow a common structure:
# a. Initialize the input blocks. b. Initialize the transformer blocks.
# c. Initialize the output blocks and other projection blocks when necessary.
if self.is_input_continuous:
    self._init_continuous_input(norm_type=norm_type)
elif self.is_input_vectorized:
    self._init_vectorized_inputs(norm_type=norm_type)
elif self.is_input_patches:
    self._init_patched_inputs(norm_type=norm_type)
```

```
def _init_continuous_input(self, norm_type):
    self.norm = torch.nn.GroupNorm(
        num_groups=self.config.norm_num_groups, num_channels=self.in_channels, eps=1e-6, affine=True
    )
    if self.use_linear_projection:
        self.proj_in = torch.nn.Linear(self.in_channels, self.inner_dim)
    else:
        self.proj_in = torch.nn.Conv2d(self.in_channels, self.inner_dim, kernel_size=1, stride=1, padding=0)

    self.transformer_blocks = nn.ModuleList(
        [
            BasicTransformerBlock(
                self.inner_dim,
                self.config.num_attention_heads,
                self.config.attention_head_dim,
                dropout=self.config.dropout,
                cross_attention_dim=self.config.cross_attention_dim,
                activation_fn=self.config.activation_fn,
                num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                attention_bias=self.config.attention_bias,
                only_cross_attention=self.config.only_cross_attention,
                double_self_attention=self.config.double_self_attention,
                upcast_attention=self.config.upcast_attention,
                norm_type=norm_type,
                norm_elementwise_affine=self.config.norm_elementwise_affine,
                norm_eps=self.config.norm_eps,
                attention_type=self.config.attention_type,
            )
            for _ in range(self.config.num_layers)
        ]
    )

    if self.use_linear_projection:
        self.proj_out = torch.nn.Linear(self.inner_dim, self.out_channels)
    else:
        self.proj_out = torch.nn.Conv2d(self.inner_dim, self.out_channels, kernel_size=1, stride=1, padding=0)
```

点进BasicTransformerBlock进入 attention.py

结构为 norm1(Layernorm) -> attn1 -> norm2 -> attn2 ->  norm3 -> ff(FeedForward)

norm1 attn1 是自注意力，norm2, attn2 是交叉注意力

在Attention_processor.py里，Attention的init：

qk_norm为None, cross_attention_norm也为None

```
self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)

if not self.only_cross_attention:
    # only relevant for the `AddedKVProcessor` classes
    self.to_k = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
    self.to_v = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
else:
    self.to_k = None
    self.to_v = None
```

**Standard Mode (`only_cross_attention=False`):** This is what 99% of models use. **Added-KV Mode (`only_cross_attention=True`):** This is a specialized mode used in models like **UnCLIP**. These models use a more complex setup where they have "added" projection layers (`add_k_proj` and `add_v_proj`). If this flag is set to `True`, the model skips creating these "standard" `to_k`/`to_v` layers because it intends to use those specialized "added" layers instead.

所有dimension都是320。

```py
# set attention processor
# We use the AttnProcessor2_0 by default when torch 2.x is used which uses
# torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
# but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
if processor is None:
    processor = (
        AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
    )
self.set_processor(processor)
```

This block of code is a **performance optimizer**. It automatically selects the fastest available method to calculate the "Attention" scores based on your environment.

`hasattr(F, "scaled_dot_product_attention")` checks if you are running **PyTorch 2.0 or newer**.

- PyTorch 2.0 introduced a built-in function called **SDPA** (Scaled Dot-Product Attention).
- SDPA is highly optimized and can automatically use "Flash Attention" or "Memory Efficient Attention" kernels, which are significantly faster and use less VRAM than standard math.

然后还有AttnProcessor2_0，建议复制给Gemini让它写注释。
