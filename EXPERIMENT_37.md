# 实验 37：在 FastWAM 中以 MiniMax-H3 替换 Wan 视频骨干

## 0. 文档说明

本文档记录实验 37 的完整技术背景、方案选择、代码改动、模型结构、权重处理、数据准备、训练过程、产物位置、复现命令，以及目前仍未验证或无法确定的事项。

本文所说的“替换 backbone”需要准确理解：

- Wan2.2 视频 DiT 和 Wan VAE 已不再参与视觉特征提取。
- 视觉输入改由 MiniMax-H3 Visual VAE 和 MiniMax-H3 FL2VA Transformer 处理。
- MiniMax-H3 视觉分支在当前实验中被冻结，只作为逐层视觉 K/V 提供者。
- 可训练部分是一个约 0.973B 参数、与 H3 注意力几何兼容的窄 ActionDiT，以及 proprio encoder。
- 这不是把 action 作为 H3 官方的第四种 modality 直接塞进原始 5376 维 residual stream，而是 H3-shaped MoT：视频和动作保留不同宽度，通过相同的 head 数和 head dimension 在注意力空间交互。
- 当前实验没有保留 FastWAM 原论文中的 video co-training loss，`loss_video` 恒为 0。

因此，实验 37 是“MiniMax-H3 视觉骨干 + FastWAM 风格动作专家”的可运行首版，而不是“完整 H3 Omni Transformer 端到端微调版”。

本文档状态时间：2026-08-16。100k continuation run 正在运行，文中的动态 step 和 ETA 仅是该时间点快照。

---

## 1. 实验目标

原始 FastWAM 使用 Wan2.2-TI2V-5B 作为视频世界模型骨干，并通过 Mixture-of-Transformer 方式将视频专家和动作专家在每层注意力中结合。

实验 37 的目标是：

1. 在 FastWAM 框架内去掉 Wan 视频骨干。
2. 使用 MiniMax-H3 FL2VA 作为视觉世界模型。
3. 保留 FastWAM 的动作 diffusion / flow-matching 训练接口和 LIBERO 数据链路。
4. 在单机 8×NVIDIA H20 96GB 上实现可实际训练的版本。
5. 使用 FastWAM release 的 LIBERO 数据进行训练。
6. 首先验证模型能加载、前向、反向和分布式训练，再扩展到总计 100k optimizer steps。

---

## 2. 为什么不能直接把 Wan 类名改成 H3

Wan2.2 和 MiniMax-H3 在多个关键维度上不兼容。

### 2.1 VAE latent 不兼容

原始 FastWAM/Wan 视频分支使用 48 通道 latent；MiniMax-H3 Visual VAE 使用 24 通道 latent。

H3 的空间压缩倍率为 16，实验输入首帧为 `224×448`，编码后得到：

```text
[B, 3, 224, 448]
-> [B, 24, 1, 14, 28]
```

H3 patch size 为 `(1,2,2)`，因此 patch 后得到：

```text
[B, 24, 1, 14, 28]
-> [B, 98, 96]
```

其中：

- token 数：`1 × 7 × 14 = 98`
- 每个 patch 的输入维度：`24 × 1 × 2 × 2 = 96`

### 2.2 Transformer 主宽度不兼容

MiniMax-H3 FL2VA 的主要参数为：

- hidden size：5376
- FFN hidden size：14336
- 主层数：50
- attention heads：56
- head dimension：128
- QKV inner dimension：`56 × 128 = 7168`
- video latent channels：24
- time embedding dimension：2688
- 每轴 RoPE frequency 数：16
- RoPE 实际旋转维度：96

原始 FastWAM 的 Wan video expert 和 action expert 使用另一套层数、宽度和 head 配置，不能直接共享注意力张量。

### 2.3 AdaLN modality 语义不兼容

H3 原始主 block 对 video、text、audio 三种 modality 使用不同的 AdaLN slice。每层 AdaLN 输出规模为：

```text
3 modalities × 6 modulation values × 5376 hidden
= 96768
```

如果严格新增 action modality tag，需要把所有 50 层 AdaLN 从三种 modality 扩成四种，并决定第四个 slice 的初始化方式。这会改变官方权重形状，也会增加显存和训练复杂度。

### 2.4 文本编码器不兼容

H3 官方使用 Qwen3-VL 系列文本/视觉编码器，H3 context dimension 是 5120。

当前 FastWAM 数据链路缓存的是 Wan UMT5 embedding：

```text
[B, 128, 4096]
```

实验 37 没有重新构建 H3 Qwen3-VL embedding 数据链路，而是保留 UMT5 cache，在 action branch 中通过新的 `context_encoder` 映射到 512 维 action hidden space。

### 2.5 资源约束

H3 FL2VA Transformer 原始权重约 33B 参数，BF16 权重约 61.7 GiB。若使用 AdamW 全量训练，参数、梯度和 optimizer states 无法在当前 FastWAM ZeRO-2 构造方式下合理容纳。

原始 trainer 会先在每张 GPU 上实例化模型，再交给 Accelerate/DeepSpeed。ZeRO-2 只切 optimizer state，不切模型参数。因此首版必须冻结 H3，或进一步重写为 ZeRO-3 分片加载。

---

## 3. 最终采用的方案

实验 37 采用以下结构：

```text
当前首帧
  -> H3 Visual VAE
  -> H3 video patch tokens
  -> 冻结的 H3 50 层视觉 Transformer
  -> 每层导出 video K/V cache

噪声 action
  -> 512 维 H3ActionDiT
  -> 每层 action Q 读取：
       当前层 H3 video K/V
       当前层 language K/V
       当前层 action K/V
  -> action velocity
  -> flow-matching loss
```

核心设计原则：

- H3 视频 residual width 保持 5376。
- ActionDiT residual width 降为 512。
- 两者都使用 50 层、56 heads、128 head dimension。
- 两者不共享 projection/MLP 参数。
- 交互发生在 `[heads, head_dim] = [56,128]` 的注意力空间。
- 视频分支不读取 action，所以可以在 `torch.no_grad()` 下计算。
- action 每层读取对应层的 H3 video K/V。

这与 FastWAM 的 MoT 思路一致，但当前实现进一步简化成单向交互：

```text
video -> action
```

而不是原始 MoT 中可能存在的双向：

```text
video <-> action
```

---

## 4. 代码文件与职责

### 4.1 新增 H3 模型目录

路径：

```text
src/fastwam/models/minimax_h3/
├── __init__.py
├── action_dit.py
├── fastwam.py
├── video_dit.py
└── video_vae.py
```

职责：

- `video_dit.py`
  - 定义与 FL2VA 原始 checkpoint key 对齐的 H3 video-only Transformer。
  - 实现 H3 RMSNorm、time embedding、3D RoPE、attention、SwiGLU MLP、AdaLN 和 final layer。
  - 逐层输出视觉 K/V。
  - 加载 13 个 safetensors shard。
  - 缓存 `t=0` 的 video AdaLN 并删除原始大矩阵。

- `video_vae.py`
  - 动态导入 H3 release 自带的 `video_vae` Python bundle。
  - 将 FastWAM 的 `[-1,1]` RGB 输入转换为 H3 processor 所需输入。
  - 实现 H3 的逐通道 latent mean/std normalization。
  - 提供 `encode_image`、`encode` 和 `decode`。

- `action_dit.py`
  - 定义 512 维、50 层、56×128 attention geometry 的 ActionDiT。
  - 实现动作 RoPE、AdaLN、SwiGLU 和 gradient checkpointing。
  - 每层读取 H3 visual K/V、language K/V 和 action K/V。

- `fastwam.py`
  - 组合 H3 VAE、H3 video expert 和 ActionDiT。
  - 冻结 H3 VAE 和 video expert。
  - 准备训练输入。
  - 实现 action flow-matching loss。
  - 实现 action denoising inference。
  - 保存/加载 action expert 和 proprio encoder 权重。

### 4.2 Runtime factory

`src/fastwam/runtime.py` 新增：

```text
create_fastwam_h3
```

该 factory：

1. 解析 Hydra `DictConfig`。
2. 校验 action scheduler 配置。
3. 调用 `FastWAMH3.from_pretrained`。
4. 将每个进程的模型放到其 `LOCAL_RANK` 对应 GPU。

### 4.3 Trainer 改动

`src/fastwam/trainer.py` 的主要改动：

- 不再假设只有 `model.dit` 可训练。
- 如果模型实现 `trainable_modules()`，optimizer 只接收这些模块。
- 实验 37 返回：
  - `action_expert`
  - `proprio_encoder`
- 增加 SwanLab logger。
- 增加 weights-only resume 的 `initial_step`。
- 当使用 2k 权重续训 100k 时：
  - 只恢复模型权重；
  - 不恢复旧 optimizer；
  - 不恢复旧 2k scheduler；
  - 创建新的 100k scheduler；
  - 将 scheduler 前进到 global step 2000。

### 4.4 配置文件

新增：

```text
configs/model/fastwam_h3.yaml
configs/task/libero_h3_uncond_2cam224_1e-4.yaml
```

新增预处理脚本：

```text
scripts/preprocess_h3_action_dit_backbone.py
```

新增启动脚本：

```text
scripts/train_experiment37.sh
```

---

## 5. 完整训练张量流

### 5.1 Dataset 输出

单样本的实测 shape：

```text
video             [3, 5, 224, 448]
action            [32, 7]
proprio           [32, 8]
context           [128, 4096]
context_mask      [128]
image_is_pad      [5]
action_is_pad     [32]
proprio_is_pad    [33]
```

batch 后：

```text
video             [B, 3, 5, 224, 448]
action            [B, 32, 7]
proprio           [B, 32, 8]
context           [B, 128, 4096]
context_mask      [B, 128]
```

重要事实：当前训练实际上只使用 `video[:,:,0]` 和 `proprio[:,0]`。

虽然 dataset 为 H3 帧族采样了 5 帧，但当前 action loss 路径没有把后 4 帧送入 H3，也没有计算 video loss。

### 5.2 双相机图像

LIBERO 配置包含：

- 外部相机 `image`
- 腕部相机 `wrist_image`

每路 resize 为 `224×224`，然后水平拼接：

```text
224×224 + 224×224 -> 224×448
```

### 5.3 H3 VAE

首帧：

```text
[B,3,224,448], range [-1,1]
```

适配器先转换为 `[0,1]`，再调用 H3 processor 做其模型所需的 pixel normalization。

输出：

```text
[B,24,1,14,28]
```

随后使用 H3 config 中的 24 维 `latents_mean` 和 `latents_std` 做逐通道：

```text
z_normalized = (z - mean) / std
```

没有使用单一 scalar `scaling_factor`。

### 5.4 H3 video tokens

patch size：

```text
(1,2,2)
```

因此：

```text
[B,24,1,14,28]
-> [B,98,96]
-> Linear(96,5376)
-> [B,98,5376]
```

每层 video Q/K/V：

```text
[B,98,56,128]
```

50 层 cache 形式：

```text
list[50] {
  "k": [B,98,56,128],
  "v": [B,98,56,128]
}
```

### 5.5 Language 和 proprio

UMT5 context：

```text
[B,128,4096]
```

proprio 只取第一个时间步：

```text
[B,8]
-> Linear(8,4096)
-> [B,1,4096]
```

拼接后：

```text
[B,129,4096]
```

ActionDiT 的 context encoder：

```text
Linear(4096,512)
-> SiLU
-> Linear(512,512)
```

得到：

```text
[B,129,512]
```

每个 ActionDiT block 用自己的 fused QKV projection 产生 context K/V：

```text
[B,129,56,128]
```

### 5.6 Action tokens

动作输入：

```text
[B,32,7]
-> Linear(7,512)
-> [B,32,512]
```

Action Q/K/V：

```text
[B,32,56,128]
```

### 5.7 每层 mixed attention

action query 长度为 32。

key/value 按以下顺序拼接：

```text
video K/V     98 tokens
context K/V  129 tokens
action K/V    32 tokens
总长度        259 tokens
```

因此 SDPA 的逻辑 shape 为：

```text
Q  [B,56,32,128]
K  [B,56,259,128]
V  [B,56,259,128]
```

输出经过 action expert 自己的：

```text
out_proj: 7168 -> 512
residual
RMSNorm
SwiGLU MLP: 512 -> 4096 -> 512
residual
```

重复 50 层，最后：

```text
RMSNorm
Linear(512,7)
-> [B,32,7]
```

---

## 6. 注意力 mask 和因果性

### 6.1 Video self-attention

配置为：

```text
video_attention_mask_mode = first_frame_causal
```

其语义是首帧 token 不允许看到后续帧，后续帧可以看到全部 video token。

但当前训练只编码一张首帧，H3 latent 时间长度为 1。因此实际 98 个 video token 全部属于同一帧，该 mask 等效为单帧内双向 attention。

### 6.2 Action 读取 video

ActionDiT 每层只取：

```python
cache["k"][:, :video_tokens_per_frame]
cache["v"][:, :video_tokens_per_frame]
```

当前 `video_tokens_per_frame=98`，恰好是全部首帧 token。

这保证 action 不读取未来观测；同时也意味着后四个 dataset video frame 完全没有参与训练。

### 6.3 Context mask

ActionDiT 的接口支持由 `context_mask` 屏蔽文本 padding，video token 和 action token 默认全部有效。但当前 dataset 实现先把 padding embedding 清零，随后把 `context_mask` 强制改为全 true；由于 `context_encoder` 带 bias，清零的 padding token 经 MLP 后会重新成为非零向量。

因此正式实验中 context mask 实际没有屏蔽文本 padding，ActionDiT 会读取全部 128 个文本位置以及追加的 proprio token。这是已确认的实现风险，而不是预期设计。

action query 可以读取：

- 全部当前帧 visual token
- 全部 128 个 language 位置和 1 个 proprio token
- 全部 action token

Action token 之间不是 causal mask，而是双向 diffusion token attention。这符合整段 action chunk 去噪的常见做法。

---

## 7. RoPE 设计

### 7.1 H3 video RoPE

video 使用真实 `(t,h,w)` 网格。

每轴 16 个 frequency，三轴合计 48，再复制为 cos/sin 配对后的 96 个旋转维度。head dimension 为 128，因此剩余 32 维不旋转。

### 7.2 Action RoPE

action position 使用：

```text
t = 0..31
h = 0
w = 0
```

同样得到 96 维旋转位置。

### 7.3 Context

当前 context K/V 不应用 RoPE。

### 7.4 尚未验证的点

action 的时间坐标没有对齐 H3 官方 text/media clock，也没有根据机器人控制频率换算为 H3 的媒体时间坐标。当前只是把 action index 直接作为 temporal position。

video K 使用 H3 空间 RoPE，action Q 使用 action temporal RoPE，context K 无 RoPE；三者混合在一次 attention 中。这在数学上可运行，但是否是最佳位置编码语义尚未通过消融实验验证。

---

## 8. H3 权重加载和 13B AdaLN 缓存

### 8.1 权重来源

H3 路径：

```text
/root/wuqingman/models/MiniMax-H3/FL2VA
```

完整下载约 135GB，实验使用：

```text
FL2VA/transformer/
FL2VA/video_vae/
```

audio VAE 和 H3 text encoder 不进入当前模型前向。

### 8.2 分片加载

Transformer 权重由 13 个 safetensors shard 组成。

加载过程：

1. 从 `config.json` 构建 meta model。
2. 在 CPU 上 `to_empty`，避免先在 GPU 分配约 65GB 空模型。
3. 根据 `model.safetensors.index.json` 按 shard 读取当前 video-only model 需要的 key。
4. 使用 `assign=True` 替换参数。
5. 严格检查所有目标 key 都已加载。
6. 缓存 video AdaLN。
7. 将剩余模型转到 GPU BF16。

### 8.3 为什么可以缓存 AdaLN

当前 frozen video branch 永远在：

```text
timestep = 0
```

H3 每层 video modality 的 AdaLN 输出因此是常量。代码对每层计算一次：

```text
shift_msa
scale_msa
gate_msa
shift_mlp
scale_mlp
gate_mlp
```

随后保存为非持久 buffer，并删除巨大的 `adaln_proj`。

实测：

- 缓存前 H3 video-only 参数约 32.324B。
- 缓存后常驻参数约 19.314B。
- 单独 H3 video backbone GPU allocation 约 36.05GB。
- 完整 FastWAMH3 加载后约 42.78GB。
- 单卡完整 loss/backward 峰值约 44.8GB。
- 原始 `B=1` 训练约 49GB/卡；调优后的 `B=16` 正式训练约 72.2GB/卡。

### 8.4 限制

缓存后 video branch 只适用于 `t=0` visual prefill，不能再用于任意 timestep 的 H3 video denoising。

这也是当前无法恢复 video flow loss 的原因之一。

---

## 9. ActionDiT 初始化

### 9.1 目标结构

ActionDiT 配置：

- action dimension：7
- context dimension：4096
- hidden size：512
- FFN hidden size：2048
- layers：50
- heads：56
- head dimension：128
- QKV inner dimension：7168
- time embedding input：256
- time embedding hidden：512
- time embedding output：512
- RoPE frequencies per axis：16
- gradient checkpointing：模型支持；调优后的正式训练关闭

总参数量实测：

```text
972,913,159
```

约 0.973B。

### 9.2 哪些参数从 H3 初始化

从 H3 video modality 初始化：

- 50 层 RMSNorm
- 50 层 fused QKV projection
- 50 层 Q/K norm
- 50 层 attention output projection
- 50 层 SwiGLU MLP
- 50 层 AdaLN
- time embedder

不从 H3 初始化、保持新模型随机初始化：

- `action_encoder`
- `context_encoder`
- `final_norm`
- `action_head`

### 9.3 AdaLN 选择

H3 AdaLN 原始 shape 包含 video/text/audio 三个 slice。预处理脚本只选择：

```text
source_modality_tag = 0
```

即 video slice，然后将 5376 宽参数插值为 512 宽 action 参数。

### 9.4 插值规则

对 shape 不匹配的每一个维度依次做：

```text
1D linear interpolation
align_corners = true
```

如果最后输入维度发生变化，再乘：

```text
sqrt(source_input_dim / target_input_dim)
```

即 alpha scaling。

checkpoint 记录：

```text
copied tensors       100
interpolated tensors 404
dtype                bfloat16
```

生成文件：

```text
checkpoints/H3ActionDiT_video_interp_512hdim.pt
```

大小约 1.9GB。

### 9.5 预处理命令

```bash
cd /root/wuqingman/FastWAM

PYTHONPATH=src python scripts/preprocess_h3_action_dit_backbone.py \
  --model-config configs/model/fastwam_h3.yaml \
  --h3-transformer-dir /root/wuqingman/models/MiniMax-H3/FL2VA/transformer \
  --output checkpoints/H3ActionDiT_video_interp_512hdim.pt \
  --dtype bfloat16
```

### 9.6 尚未验证的点

- 多维顺序线性插值并不是 H3 官方提供的初始化方法。
- alpha scaling 是否适合 H3 后期层中数值较大的权重，没有理论或实验保证。
- ActionDiT 宽度只有 512，但每个 token 投影到 7168 维 QKV，参数分配非常不均衡。
- 没有与随机初始化、无 alpha scaling、SVD/低秩投影或复制子空间初始化做对比。
- 训练 loss 能下降只能说明该初始化可优化，不能证明它优于其他初始化。

---

## 10. Flow-matching 训练目标

实验沿用 FastWAM 的 `WanContinuousFlowMatchScheduler`。

action scheduler：

```text
train shift         5.0
inference shift     5.0
train timesteps     1000
```

训练时：

```text
u ~ Uniform(0,1)
sigma = shift * u / (1 + (shift - 1) * u)
t = sigma * 1000
```

加噪：

```text
x_t = (1-sigma) * action + sigma * noise
```

目标 velocity：

```text
target = noise - action
```

loss：

1. 对 7 个 action dimension 做 MSE。
2. 使用 `action_is_pad` 去掉 padding action。
3. 对每个 sample 聚合。
4. 乘 scheduler 的 timestep training weight。
5. batch mean。

最终：

```text
loss = lambda_action * loss_action
lambda_action = 1.0
lambda_video = 0.0
```

配置中虽然保留 H3 video scheduler shift 12.0，但当前 `FastWAMH3.training_loss` 不使用 video scheduler。

---

## 11. 冻结和可训练参数

冻结：

- H3 Visual VAE
- H3 video Transformer
- H3 final video head

训练：

- H3ActionDiT
- proprio encoder `Linear(8,4096)`

训练前 trainer 执行：

```text
model.eval()
model.requires_grad_(False)
```

然后仅对 `trainable_modules()` 返回的模块重新：

```text
train()
requires_grad_(True)
```

optimizer：

```text
AdamW
lr           1e-4
betas        (0.9, 0.95)
weight_decay 1e-2
```

最大 gradient norm：

```text
1.0
```

精度：

```text
BF16
```

---

## 12. LIBERO 数据

### 12.1 来源

FastWAM 官方 release：

```text
https://huggingface.co/datasets/yuanty/LIBERO-fastwam
```

Hugging Face 主站连接失败后，实际通过：

```text
HF_ENDPOINT=https://hf-mirror.com
```

下载成功。

### 12.2 四个子集

```text
libero_spatial_no_noops_lerobot  10 tasks
libero_object_no_noops_lerobot   10 tasks
libero_goal_no_noops_lerobot     10 tasks
libero_10_no_noops_lerobot       10 tasks
```

共 40 个任务。

压缩包大小：

```text
libero_10      约 1.5GB
libero_goal    约 799MB
libero_object  约 1.3GB
libero_spatial 约 924MB
```

压缩包与解压数据当前合计约 8.8GB。

### 12.3 数据路径

```text
data/libero_mujoco3.3.2/
├── libero_10_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
└── libero_spatial_no_noops_lerobot/
```

数据格式为 LeRobot v2.1，来源环境对应 MuJoCo 3.3.2。

### 12.4 Dataset 实测规模

四个分集合计：

```text
1,712 episodes
277,713 frames/transitions
40 tasks
3,424 AV1 videos
1,712 parquet files
```

各分集：

```text
spatial    434 episodes   53,229 frames   868 videos
object     457 episodes   67,309 frames   914 videos
goal       433 episodes   52,895 frames   866 videos
libero_10  388 episodes  104,280 frames   776 videos
```

训练 dataset 长度为 `277,713 samples`。数据根目录实测约 8.78GiB，其中包含 Hugging Face 下载缓存导致的重复占用。

`val_set_proportion=0.0`，因此没有独立 validation split。trainer 中的 `val_dataset` 实际引用 train dataset，但 `eval_every=0`，训练期间不会运行 eval。

### 12.5 Action 和状态

action：

```text
7D = 6D end-effector delta pose + 1D gripper
```

state/proprio：

```text
8D = 6D end-effector pose + 2D gripper state
```

前 6 个 action dimension 使用 delta 语义，gripper 不使用 delta。

action/state normalization：

```text
min/max
```

dataset stats 在每个 run 目录生成：

```text
dataset_stats.json
```

---

## 13. 视频采样策略

原始 LIBERO config：

```text
num_frames = 33
action_video_freq_ratio = 4
```

实验 37 override：

```text
action_video_freq_ratio = 8
```

因此 33 个机器人时间步产生：

```text
32 action steps
5 sampled video frames: index 0,8,16,24,32
```

这样满足代码对 `(num_frames-1)/ratio` 可被 4 整除的约束，也得到 H3 适配时计划使用的 5 帧输入。

但是当前模型只使用第 0 帧。因此“5 帧满足 H3 frame family”的设计在当前 action-only 实现中没有实际参与 H3 video encoding。

---

## 14. 文本 embedding

FastWAM release 不包含预计算 task embedding，因此本地读取四个子集的：

```text
meta/tasks.jsonl
```

共得到 40 条唯一 instruction。

使用 Wan UMT5 text encoder 生成：

```text
40 files
总大小约 41MB
每个 context shape [128,4096]
```

cache 路径：

```text
data/text_embeds_cache/libero/
```

UMT5 权重通过 ModelScope intra-cloud 下载，约 10.6GB，位置：

```text
/root/wuqingman/models/wan/
```

生成命令：

```bash
DIFFSYNTH_MODEL_BASE_PATH=/root/wuqingman/models/wan \
PYTHONPATH=src \
/root/wuqingman/.venv-fastwam/bin/python \
scripts/precompute_text_embeds.py \
task=libero_h3_uncond_2cam224_1e-4
```

重要偏离：这不是 H3 官方 Qwen3-VL embedding。当前只能称作“H3 visual backbone + Wan UMT5 task conditioning”。

---

## 15. 软件和硬件环境

### 15.1 硬件

```text
8 × NVIDIA H20
每卡显存约 96GB
Host RAM 约 1.5TiB
Root 可用磁盘最初约 4.1TB
```

### 15.2 Python 环境

```text
/root/wuqingman/.venv-fastwam
```

关键版本：

```text
Python         3.11
PyTorch        2.7.1+cu128
CUDA runtime   12.8
torchvision    0.22.1+cu128
accelerate     1.12.0
DeepSpeed      0.18.5
diffusers      0.32.2
transformers   4.49.0
safetensors    0.5.3
datasets       3.6.0
ModelScope     1.34.0
SwanLab        0.9.4
```

系统原有 PyTorch 2.3.1+cu121 在 H20 BF16 SDPA 中出现过进程 `SIGFPE`。升级到 PyTorch 2.7.1+cu128 后，完整 H3 forward 正常。

PyTorch 官方源在该机器下载极慢，最终通过阿里云 PyTorch wheel mirror 安装。

---

## 16. 验证过程

### 16.1 小模型结构测试

完成：

- 小尺寸 H3ActionDiT mixed attention forward。
- 输出 shape 正确且 finite。
- 小尺寸 H3 video patchify/unpatchify。
- 逐层 K/V cache 数量和 shape 正确。

### 16.2 完整 H3 video backbone

实测：

```text
加载后参数       19.314B
GPU allocation   36.05GB
输入 latent      [1,24,1,14,28]
输出 prediction  [1,24,1,14,28]
K/V cache layers 50
输出 finite      true
```

### 16.3 H3 Visual VAE

实测：

```text
输入  [1,3,224,448]
输出  [1,24,1,14,28]
finite true
GPU peak 约 5GB
```

需要补充两个实现事实：

- H3 release wrapper 的 image encode 最终使用 posterior `.sample()`，不是 posterior mean。因此同一图像重复编码并非严格确定；当前没有固定这部分采样随机性，也没有比较 sample 与 mean 对策略训练的影响。
- 初版 adapter 把整个 batch tensor 作为一个 list element 传给官方 `encode_images/encode_videos`，当 `B>1` 时会多出额外 batch 维。性能调优前已改为按 batch 维拆成 list，并实测 `B=1/2/4` 均得到 `[B,24,1,14,28]` 且 finite；之后 8 卡 `B=8/16` 完整训练也通过。

上述 VAE、完整 H3 forward 和单卡 backward 的结果来自开发期测试记录；除两卡 smoke 外，没有保存独立原始测试日志，因而目前只能复现检查，不能从 run 产物反向证明当时的逐项数值。

### 16.4 完整单卡 loss/backward

合成输入下：

```text
loss finite
513 个有梯度 tensor 全部 finite
0 个 non-finite gradient tensor
GPU peak 约 44.8GB
```

曾出现一次 gradient NaN，根因是 gradient checkpoint closure 在 Python 循环中晚绑定到了最后一个 block。通过把当前 block 绑定为 closure 默认参数修复。修复后完整检查无 NaN。

### 16.5 两卡 ZeRO-2 smoke

运行：

```text
2 GPUs
max_steps=1
gradient_accumulation_steps=1
num_workers=0
```

结果：

```text
loss 1.4547
约 44.5GB/GPU
optimizer step 成功
weights checkpoint 成功
DeepSpeed state 成功
```

---

## 17. 第一阶段：2k 验证训练

run ID：

```text
experiment37_h3_libero_v2
```

输出：

```text
runs/libero_h3_uncond_2cam224_1e-4/experiment37_h3_libero_v2
```

主要配置：

```text
8 GPUs
batch_size per GPU             1
gradient_accumulation_steps   16
effective global batch       128
max_steps                   2000
save_every                   500
eval_every                     0
learning_rate              1e-4
weight_decay               1e-2
cosine scheduler
5% warmup
BF16
ZeRO-2
```

关键训练点：

```text
step 1      loss 1.7027  lr 1.99e-6
step 500    loss 0.2562  lr 8.96e-5
step 1000   loss 0.2073  lr 5.46e-5
step 1500   loss 0.0662  lr 1.70e-5
step 2000   loss 0.0694  lr 1.00e-6
```

运行时间：

```text
开始 2026-08-16 01:17
完成 2026-08-16 13:46
约 12.5 小时
```

平均速度约：

```text
0.04 optimizer step/s
0.36 samples/s（日志口径）
```

权重：

```text
step_000500.pt   约 1.9GB
step_001000.pt   约 1.9GB
step_001500.pt   约 1.9GB
step_002000.pt   约 1.9GB
```

每个 DeepSpeed full state 实测约 101.3GB（十进制，约 94.3GiB）。整个 2k run 目录约 384.6GiB。

注意：单点 loss 受随机 timestep、noise 和 batch 影响，不能把上述几个数字当作严格单调曲线。

当前 task YAML 已改为 100k continuation 默认配置。因此若要重新运行原始 2k 实验，必须显式覆盖 `resume` 和 `initial_step`，不能只复制 task 名称：

```bash
cd /root/wuqingman/FastWAM

PATH=/root/wuqingman/.venv-fastwam/bin:$PATH \
PYTHONPATH=src \
DIFFSYNTH_MODEL_BASE_PATH=/root/wuqingman/models/wan \
TOKENIZERS_PARALLELISM=false \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
RUN_ID=experiment37_h3_libero_v2 \
bash scripts/train_zero2.sh 8 \
  task=libero_h3_uncond_2cam224_1e-4 \
  max_steps=2000 \
  save_every=500 \
  resume=null \
  initial_step=0 \
  swanlab.mode=local
```

---

## 18. 第二阶段：总计 100k 训练

用户确认目标为总计 100k optimizer steps，每 10k 保存一次。

run ID：

```text
experiment37_h3_libero_100k_fast
```

输出：

```text
runs/libero_h3_uncond_2cam224_1e-4/experiment37_h3_libero_100k_fast
```

最初的 `experiment37_h3_libero_100k` 使用 `batch_size=1`、gradient accumulation 16，在 step 2159 主动停止；该 run 未达到 10k，因而没有新 checkpoint。性能调优后，正式 run 重新从同一个 2k weights checkpoint 启动。

### 18.1 为什么没有直接恢复 2k full state

2k run 的 cosine scheduler 总 horizon 就是 2000，step 2000 时 LR 已到 `1e-6`。

如果直接恢复完整 state：

- optimizer moments 可以恢复；
- 但 scheduler 也会恢复成已经结束的 2k scheduler；
- 后续 98k step 的 LR 轨迹不正确。

因此采用：

```text
恢复 step_002000.pt 模型权重
不恢复 optimizer
不恢复旧 scheduler
新建 100k optimizer/scheduler
把 global_step 设置为 2000
把新 scheduler 前进到 step 2000
```

恢复后实测：

```text
global_step 2000
lr          4.0012e-5
```

100k scheduler 有 5000 step warmup，因此 step 2000 仍处于 warmup 中。

### 18.2 当前配置

```text
max_steps                    100000
initial_step                   2000
实际新增 optimizer steps      98000
save_every                    10000
batch_size per GPU               16
GPUs                              8
gradient accumulation             1
effective global batch          128
ActionDiT gradient checkpointing 关闭
DeepSpeed overlap_comm            开启
DeepSpeed contiguous_gradients    开启
num_workers per rank             16
eval_every                        0
```

### 18.3 启动命令

```bash
cd /root/wuqingman/FastWAM

PATH=/root/wuqingman/.venv-fastwam/bin:$PATH \
PYTHONPATH=src \
DIFFSYNTH_MODEL_BASE_PATH=/root/wuqingman/models/wan \
TOKENIZERS_PARALLELISM=false \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
SWANLAB_SKIP_SWANBOARD_VERSION_CHECK=1 \
RUN_ID=experiment37_h3_libero_100k_fast \
bash scripts/train_zero2.sh 8 \
  task=libero_h3_uncond_2cam224_1e-4 \
  swanlab.mode=local \
  +data.train.pretrained_norm_stats=./runs/libero_h3_uncond_2cam224_1e-4/experiment37_h3_libero_v2/dataset_stats.json
```

### 18.4 启动后实测

```text
step 2010 loss 1.5261 lr 4.02e-5
step 2020 loss 1.3923 lr 4.04e-5
step 2030 loss 1.4150 lr 4.06e-5
step 2040 loss 1.2453 lr 4.08e-5
step 2050 loss 1.2082 lr 4.10e-5
```

恢复后 loss 明显波动，原因至少包括：

- 新 optimizer 没有继承 2k run 的 Adam moments；
- 新 scheduler 的 LR 从旧 run 末尾 `1e-6` 跳到约 `4e-5`；
- flow-matching 的随机 timestep 和 noise 导致单步 loss 方差较大；
- 100k 阶段重新从 step 2000 权重开始。

这需要观察更长窗口的 moving average，不能只用前几个 step 判断训练退化。

### 18.5 预计耗时

调优前约为 22–23 秒/optimizer step。调优后 step 2010–2050 的稳定区间约为 2.2 秒/optimizer step：

```text
剩余约 98k steps
纯训练计算约 60 小时
考虑 checkpoint 和波动，预计约 2.5–3 天
```

ETA 会随数据加载、checkpoint 保存和机器负载变化。

### 18.6 性能调优测试

所有主要对比均保持 effective global batch 为 128：

```text
设置                               稳态耗时/step    结果
B=1,  accumulation=16, GC on      约 22–23 秒      原始配置
B=8,  accumulation=2,  GC off     约 3.3–3.5 秒    稳定
B=16, accumulation=1,  GC on      约 3 秒          稳定
B=16, accumulation=1,  GC off     约 2.2–2.7 秒    最快
```

最终设置实测：

```text
平均 GPU SM utilization   84.0%（12秒采样）
平均 GPU power            218.3W
平均显存                  72.2GB/卡
有效吞吐                  约 58 samples/s
```

相对于原设置，optimizer step 约加速 9–10 倍，同时 global batch 不变。显存仍保留约 24GB/卡余量。

### 18.7 Epoch 换算

dataset 长度为 277,713，effective global batch 为 128。

近似：

```text
optimizer steps per epoch ≈ 277713 / 128 ≈ 2170
2k steps  ≈ 0.92 epoch
20k steps ≈ 9.2 epochs
100k steps ≈ 46 epochs
```

配置中的 `num_epochs=10` 在设置了 `max_steps=100000` 后不会限制训练总步数。

---

## 19. Checkpoint 和磁盘规划

weights-only checkpoint：

```text
约 1.9GB
包含 action_expert、proprio_encoder、step、backbone 标记
不包含冻结 H3 权重
```

DeepSpeed state：

```text
约 101.3GB/checkpoint（十进制）
约 94.3GiB/checkpoint
包含分布式 optimizer/scheduler/model state
```

100k run 每 10k 保存，理论上可能产生约：

```text
10 × 101.3GB ≈ 1.01TB full states
10 × 1.9GB ≈ 19GB weights
```

再加 2k run 已使用约 385GB，总磁盘压力可能超过 1.3TB。当前 root 盘容量足够，但需要持续监控。

训练到 `max_steps` 时 trainer 会再次保存最终 state；如果 final step 同时满足 `save_every`，可能发生同一路径重复保存，增加结束阶段耗时但通常不会翻倍占用最终目录。

---

## 20. SwanLab

代码已经增加 SwanLab 原生日志入口。

源 task config 写的是：

```text
swanlab.mode=online
```

实际 run command override 为：

```text
swanlab.mode=local
```

原因是没有把 API key 写入仓库、配置或终端命令。当前日志保存在 run 目录的 `.swanlab`/run 数据文件中，可以后续认证后同步。

当前文档不记录 API key。

---

## 21. Inference 和评测现状

`infer_action` 已实现 action flow-matching 采样，可以从随机 action noise 迭代得到 32×7 action chunk。

但 `FastWAMH3.infer()` 当前返回的视频不是 H3 rollout，而是简单重复输入首帧，用于满足原 trainer 接口：

```text
video = [input_frame] × num_frames
```

因此：

- 当前不能用 rollout PSNR/SSIM 评价 H3 world model。
- `eval_every=0` 是有意设置。
- 尚未运行 LIBERO simulator success-rate evaluation。
- 当前训练 loss 降低不等价于机器人任务成功率提高。

正式判断实验 37 是否有效，必须补充：

1. Action-only LIBERO simulator evaluation。
2. 40 个任务、多个 seeds 的 success rate。
3. 与原 FastWAM/Wan checkpoint 的同设置对比。
4. 至少按四个 suite 分开统计。

---

## 22. 已知事实、设计决策和不确定点

### 22.1 已确认事实

- Wan VAE 和 Wan video DiT 不参与视觉前向。
- 视觉输入由 H3 Visual VAE 和 H3 FL2VA Transformer 处理。
- H3 visual branch 冻结。
- ActionDiT 和 proprio encoder 可训练。
- 完整前向、反向、2 卡和 8 卡训练已通过。
- 2k run 无 OOM、无最终 NaN，并成功保存。
- 100k run 已从 global step 2000 启动。
- 当前训练只使用首帧，不使用未来 4 帧。
- 当前没有 video loss。
- 当前 language embedding 仍来自 Wan UMT5。

### 22.2 主动做出的工程决策

- 选择 H3-shaped MoT，而不是扩展 H3 第四 modality。
- 选择 512 维 ActionDiT，以降低可训练参数和 optimizer state。
- 保持 56 heads、128 head dimension 和 50 层，以便逐层读取 H3 K/V。
- 冻结 H3，以适配 ZeRO-2 和 96GB H20。
- 缓存 `t=0` video AdaLN，减少约 26GB BF16 权重显存。
- 关闭 eval，避免把重复首帧伪 rollout 当成有效视频预测。
- 100k continuation 使用 weight-only resume 和新 scheduler。

### 22.3 高优先级不确定点

1. **是否可以严格称为“完整 backbone 替换”**
   - 视觉 backbone 确实是 H3。
   - 但文本仍是 Wan UMT5，action 不是 H3 官方第四 modality，H3 也被冻结。
   - 更准确名称是“FastWAM with frozen MiniMax-H3 visual backbone”。

2. **没有 video co-training 的影响**
   - FastWAM 论文表明去掉 video co-training 可能降低 action performance。
   - 当前 H3 预训练能否完全替代 embodied video co-training 未验证。

3. **只使用首帧**
   - dataset 后四帧未参与训练。
   - H3 只提供静态视觉表征，而不是在机器人轨迹上学习动态世界模型。

4. **ActionDiT 插值初始化**
   - 方法可运行，但不是官方方法。
   - 尚无随机初始化对照。

5. **Action RoPE**
   - 未对齐 H3 媒体时间轴和机器人控制频率。

6. **UMT5 与 H3 表征不一致**
   - task text 没有进入 H3 residual stream。
   - context 仅通过 action branch 的随机初始化 adapter 使用。

7. **VAE dtype**
   - 当前 H3 VAE 被转成 BF16。
   - 官方现代 H3 VAE 通常建议保留 FP32 或特定 autocast 策略。
   - 已验证 finite，但重建质量尚未系统测量。

8. **H3 raw FL2VA 与现代 diffusers 实现差异**
   - 当前代码按下载的 FL2VA fused-QKV checkpoint 和 DiffSynth 风格实现加载。
   - 没有逐层与最新 diffusers `MiniMaxH3Transformer3DModel` 做数值对齐测试。

9. **100k resume 的 optimizer discontinuity**
   - 2k optimizer moments 丢失。
   - LR 从 `1e-6` 跳到约 `4e-5`。
   - 这使 2k→100k 不是严格连续训练。

10. **100k 是否过度训练**
    - 约 46 epochs，远高于 FastWAM 官方约 20k step/9 epochs 的口径。
    - 是否带来收益或过拟合必须靠 simulator evaluation 判断。

11. **Checkpoint 成本**
    - 每 10k 保存完整 DeepSpeed state 约 101.3GB（94.3GiB）。
    - 长训练存在接近 TB 级磁盘占用。

12. **SwanLab 云端**
    - 当前只确认本地数据写入。
    - 尚未确认云端 run 和远程曲线。

13. **VAE posterior 随机采样**
    - 当前不是确定性的 posterior mean 编码。
    - 同一首帧可能得到不同 latent 和 H3 K/V。
    - 这可能是正则化，也可能只是额外训练方差，尚无对照。

14. **文本 padding mask 实际失效**
    - dataset 会先把 padding embedding 清零，随后把 `context_mask` 强制改成全 true。
    - `context_encoder` 带 bias，零 padding 经 MLP 后会重新成为非零 token。
    - 因而当前 ActionDiT 会关注全部 128 个文本位置；这不是文档前面所描述的“只关注有效 token”，属于需要修正或消融的实现问题。

15. **单卡 batch size 大于 1 的 VAE shape 问题（已修复）**
    - 初版 adapter 的 list/batch 包装在 `B>1` 时会多出一维。
    - 现已按 batch 拆分输入，并通过独立 `B=1/2/4` shape test 以及 `B=8/16` 训练验证。

16. **配置校验不完整**
    - H3 hidden size、层数等主要结构实际从磁盘 `config.json` 读取。
    - YAML 中对应字段没有全部与官方配置做一致性检查。
    - `video_scheduler`、`lambda_video` 和旧 MoT checkpoint 参数在 H3 factory 中实际不参与训练逻辑。

17. **Weights-only checkpoint 校验偏弱**
    - 加载时没有严格核对 checkpoint 内的 `step`、`backbone` 与配置 `initial_step`。
    - 若缺少 proprio encoder，当前逻辑可能保留随机参数继续运行。

### 22.4 尚未完成的验证

- H3 VAE PSNR/SSIM 全数据统计。
- H3 VAE FP32 与 BF16 对比。
- H3 VAE posterior sample 与 posterior mean 对比。
- 单卡 `batch_size>1` 的 H3 VAE encode shape 测试和修复。
- H3 visual features 与官方 pipeline 数值对齐。
- AdaLN 缓存前后逐层 K/V 和最终输出数值等价性。
- 保留真实 text mask 与当前全 true mask 的对比。
- LIBERO simulator 安装和 MuJoCo 3.3.2 evaluation。
- 2k/10k/20k/50k/100k checkpoints 的 success-rate 曲线。
- 原 Wan FastWAM 同 batch、同 step 对照。
- random-init ActionDiT 对照。
- 不同 action width 对照。
- 恢复 video co-training 的实验。
- H3 Qwen3-VL text embedding 替换 UMT5。
- action RoPE 时间对齐消融。
- full H3 fourth-modality 或 LoRA/ZeRO-3 方案。

---

## 23. 推荐的后续实验

优先级 1：尽早做 action-only evaluation，而不是等待 100k 全部结束。

建议在以下 checkpoint 评测：

```text
2k
10k
20k
50k
100k
```

优先级 2：建立最小对照组。

至少需要：

```text
原 FastWAM/Wan
H3 visual + random ActionDiT
H3 visual + interpolated ActionDiT
H3 visual + interpolated ActionDiT + video co-training
```

优先级 3：修正语义不一致。

- 改用 H3 Qwen3-VL task embedding。
- 将 action time 映射到 H3 media clock。
- 判断是否需要把 action 作为 H3 第四 modality。

优先级 4：提高训练效率。

- 缓存冻结 H3 对首帧的逐层 K/V；当前每个 epoch 对同一观测仍重复计算。
- 不计算当前 loss 路径未使用的 H3 final video prediction。
- 评估 sequence/data parallel。
- 若要训练 H3，改用 ZeRO-3/FSDP 分片初始化。

---

## 24. 复现检查清单

训练前确认：

```text
[ ] /root/wuqingman/.venv-fastwam 可用
[ ] PyTorch 为 2.7.1+cu128
[ ] 8 张 H20 空闲
[ ] H3 FL2VA transformer 13 个 shard 完整
[ ] H3 video_vae/source/model.safetensors 存在
[ ] H3 ActionDiT 1.9GB 初始化权重存在
[ ] 四个 LIBERO 子集已解压
[ ] 40 个 text embedding cache 存在
[ ] dataset_stats 可生成
[ ] root 盘至少预留 1TB
```

训练中确认：

```text
[ ] 8 个 rank 均启动
[ ] 每卡显存约 72GB（当前 B=16 配置）
[ ] loss finite
[ ] grad norm finite
[ ] global step 连续
[ ] LR 与 100k scheduler 一致
[ ] 每 10k weights/state 均保存
[ ] SwanLab 本地文件持续增长
[ ] 磁盘空间充足
```

训练后确认：

```text
[ ] step_100000.pt 存在
[ ] final DeepSpeed state 可加载
[ ] action inference finite
[ ] LIBERO simulator success rate 已评测
[ ] 与 Wan baseline 同设置对比
```

---

## 25. 最重要的结论

实验 37 已经证明：

- MiniMax-H3 Visual VAE 和 FL2VA Transformer 可以在 FastWAM 数据和训练框架内实际运行。
- 通过保留 56×128 注意力几何，可以让一个窄 ActionDiT 逐层读取 H3 visual K/V。
- 通过冻结 H3 和缓存 `t=0` AdaLN，可以将模型放入 96GB H20，并完成 8 卡 ZeRO-2 训练。
- 2k run 的 action flow loss 能从约 1.7 降到约 0.07。

实验 37 尚未证明：

- H3 比 Wan 更好。
- 100k 训练会提高 LIBERO success rate。
- H3 预训练可以替代 FastWAM 的 video co-training。
- 当前插值 ActionDiT、UMT5 conditioning 和 action RoPE 是最优设计。
- 当前模型具备真正的视频 imagination/rollout 能力。

因此，当前阶段应定义为：

```text
工程可行性和训练可行性已验证；
任务性能和方法有效性仍待 simulator evaluation 与对照实验验证。
```
