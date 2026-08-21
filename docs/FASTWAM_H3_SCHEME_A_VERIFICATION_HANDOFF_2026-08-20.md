# FastWAM-H3 Scheme A 修复与真实验证 Handoff

日期：2026-08-20  
用途：交给另一名 AI 或工程师进行独立代码审查  
状态：第一轮修复已在 `f8d4dd0c012101deab2a77150b378f35c252fd4b`
推送；第 22 节生产收口修改正在本地验证，尚未再次提交

## 1. 审查对象和工作区

本轮工作以以下远端分支和固定提交为基线：

- 仓库：`Qingman-Wu/fastwam-minimax-h3`
- 分支：`codex/fastwam-h3-scheme-a-2026-08-19`
- 基线提交：`c336c384bae444b676c2956a46d92bfd53a685d9`
- 本地验证工作区：
  `/root/wuqingman/FastWAM-H3-scheme-a-verify`
- MiniMax-H3 FL2VA 权重：
  `/root/wuqingman/models/MiniMax-H3/FL2VA`

验证环境：

- PyTorch：`2.7.1+cu128`
- Python：3.11
- Transformers：隔离安装的 `4.57.6`
- Tokenizers：`0.22.2`
- GPU：H20 96 GiB
- 运行验证时 8 张卡上存在用户启动的约 30% 人工计算负载；没有停止该负载

仓库当前仍位于基线 commit，但工作树包含本轮未提交修改。

当前代码变更规模：

```text
13 files changed
372 insertions
34 deletions
```

另有以下本地 artifacts：

- `checkpoints/H3ActionDiT_video_interp_1024hdim.pt`
  - 大小约 4.5 GiB
  - 2.415B 参数
- 一个真实 LIBERO Qwen cache v2 样本：
  - `data/h3_condition_cache/libero_2cam224/ee45f6cd8d1ca2fc30efb500f1f72324f123ae2e58069ed690c02d043a54a216.h3-qwen-prenorm-layer50-v2.pt`
  - 大小约 1.4 MiB
- `data` 是指向 `/root/wuqingman/FastWAM/data` 的未跟踪符号链接，不应提交

## 2. 总体结论

固定提交 `c336c384` 的 Scheme A 设计方向基本正确，但不能直接开始训练。
真实权重验证确认至少存在以下启动阻断：

1. 5 帧 full-video VAE 编码未使用 `encode_prefix=True`。
2. 未正确处理 `encode_prefix=True` 的 tuple 返回。
3. 5 帧对应的 2 个 latent 不能直接被官方 decoder 解码。
4. Qwen 截断 50 层后仍保留 final RMSNorm，得到的是错误的 post-norm 表示。
5. Qwen cache 仍是 schema v1，不能区分错误实现与 prenorm 新实现。
6. 官方 H3 checkpoint 的 FP32/BF16 混合参数在无 autocast 时发生 dtype mismatch。
7. H3 没有 LoRA wiring，因此 `freeze_video_expert=true` 时视频损失没有可训练目标。
8. 缺少 evaluator 使用的 `infer_joint()` 和 `infer_action()`。
9. 1024-width ActionDiT 初始化 artifact 不存在。
10. 默认 `B=16、GC off` 与真实 33B 显存需求不兼容。
11. condition cache 首次生成时，dataset stats 目录不存在会直接失败。

这些问题已在验证工作区修复。修复后完成了：

- 真实 VAE encode/decode
- 真实 Qwen-32B layer49 reference
- 真实 33B H3 加载
- 真实 LIBERO 单样本 forward
- backward
- AdamW optimizer step
- 2 次联合 video/action 更新
- 5 帧 VAE decode
- LIBERO evaluator action denormalization 路径

最终单元测试结果：

```text
75 passed, 1 warning
```

`compileall`、IDE lint 和 `git diff --check` 均通过。

## 3. VAE 验证和修复

### 3.1 原问题

基线 full-video 路径调用：

```python
self.vae.encode_videos(..., transform_input=False)
```

官方 VAE 会按 chunk granularity 处理普通视频。5 帧不满足普通路径要求，真实调用报错：

```text
Cannot trim 5 frames to valid length 22: not enough frames
```

### 3.2 真实 prefix 编码结果

使用官方 FL2VA VAE 权重和 224×448 输入验证：

```text
5 frames,  prefix=True  -> [24, 2, 14, 28], prefix_pad_frames=[0]
22 frames, prefix=True  -> [24, 7, 14, 28], prefix_pad_frames=[0]
39 frames, prefix=True  -> [24,12, 14, 28], prefix_pad_frames=[0]
```

因此设计文档中的：

```text
5  -> 2
22 -> 7
39 -> 12
```

被真实权重确认。

`encode_prefix=True` 的真实返回类型是：

```python
(video_latents, prefix_pad_frames)
```

而不是普通的 `List[Tensor]`。

### 3.3 22/39 prefix 数值一致性

官方 VAE 的 `encode_base()` 默认从 posterior 随机采样。普通路径和 prefix 路径
内部 posterior temporal shape 不同，所以仅设置同一个随机 seed 仍会给各通道分配
不同噪声，不能直接据此判断 encoder 是否一致。

将 posterior sampling 临时替换为 posterior mean 后，结果为：

```text
22 frames:
  max_abs_diff  = 0.0
  mean_abs_diff = 0.0

39 frames:
  max_abs_diff  = 0.0
  mean_abs_diff = 0.0
```

这证明对本来就合法的 22/39 帧输入，prefix 路径保留 latent 与普通路径严格一致。

### 3.4 编码修复

修改：

`src/fastwam/models/minimax_h3/video_vae.py`

full-video 路径现在：

1. 显式传入 `encode_prefix=True`。
2. 检查返回值必须是二元 tuple。
3. 解包 `latents, prefix_pad_frames`。
4. 检查 pad count 数量等于 batch size。
5. 对当前 FastWAM LIBERO contract 要求所有 `prefix_pad_frames == 0`。
6. 非零 leading pad 会明确报错，不会静默产生时间错位。

image/keyframe 路径仍走 `encode_images()`，没有改成 prefix video 路径。

### 3.5 新发现：2-latent decode 阻断

第一次真实联合推理已经完成两次 H3/Action 去噪更新，但最终 decode 报错：

```text
ValueError:
decode_temporal streaming planned non-positive output_frames=0
total_frames=0 pad_frames=0
```

原因是官方 decoder 的 chunk 规划不能直接从 2 个 temporal latent 建立一个可解码
窗口。当前 checkpoint：

```text
tokens_chunk_size = 5
token_overlap     = 2
```

一个可解码窗口至少需要：

```text
5 + 2 = 7 temporal latent rows
```

### 3.6 decode 修复

当 `frame_num == 5` 且输入少于完整 decoder window 时：

1. 使用最后一个 latent 向尾部补齐。
2. 补至 `tokens_chunk_size + token_overlap`，当前即 7。
3. 调用官方 `decode_base(..., frame_num=5)`。
4. 由官方 causal `trim_output()` 裁回 5 帧。

修复后真实联合推理输出：

```text
frames     = 5
frame size = 448 x 224
action     = [32, 7]
action finite = true
peak allocated memory = 71.45 GiB
```

2026-08-20 的后续 gold parity 实验已经证明：这段修复只解决了 runtime shape
和 chunk planner 阻断，**没有解决短序列的正确解码语义**。它不应被视为正式通过；
详见第 21 节。

### 3.7 VAE FP32/BF16 fidelity

使用真实 LIBERO 样本、posterior mean、同一短 decode 修复，对比 FP32 和 BF16：

```text
latent max_abs_diff  = 0.5522571
latent mean_abs_diff = 0.0066516
decode PSNR          = 47.725 dB
decode SSIM          = 0.996733
BF16 latent/decode finite = true
```

当前证据支持 BF16 VAE 可用于 smoke 和初始训练，不支持把它定为 P0 blocker。

注意：此 fidelity 只证明 FP32 与 BF16 对同一 workaround 的一致性。后续 gold parity
已经否定该 workaround 的语义正确性，因此这里的高 PSNR/SSIM 不能用于证明 5 帧
decode 正确。

## 4. Qwen layer-50 prenorm 验证和修复

### 4.1 原实现问题

基线逻辑：

1. 将 language layers 截断为前 50 层。
2. 保留 `language_model.norm`。
3. 请求 `output_hidden_states=True`。
4. 读取 `hidden_states[50]`。

在 Transformers 4.57.6 中，截断后的：

```text
last_hidden_state
hidden_states[50]
```

均已经过 final RMSNorm。

因此基线没有得到 H3 需要的 decoder layer49 output / layer-50 prenorm state。

### 4.2 真实 Qwen-32B gold reference

加载真实 64 层 Qwen3-VL-32B，使用真实图像 presentation，在 decoder layer 49
注册 forward hook。

测试序列：

```text
shape = [1, 111, 5120]
finite = true
Qwen load peak = 62.15 GiB
```

比较结果：

```text
50 layers + final norm retained + last_hidden_state:
  max_abs_diff  = 23424.0
  mean_abs_diff = 1.2053654

50 layers + final norm retained + hidden_states[50]:
  max_abs_diff  = 23424.0
  mean_abs_diff = 1.2053654

50 layers + final norm Identity + last_hidden_state:
  max_abs_diff  = 0.0
  mean_abs_diff = 0.0
```

结论非常明确：

```text
生产实现：
layers[0:50] + final norm Identity + last_hidden_state

gold reference：
完整 64 层 + layer49 forward hook
```

两者在真实 32B 权重上严格一致。

### 4.3 Qwen 修复

修改：

`src/fastwam/models/minimax_h3/text_encoder.py`

具体变化：

- 保留 decoder layers 0..49。
- 将 `model.language_model.norm` 替换为 `nn.Identity()`。
- 不再请求全部 hidden states。
- 直接读取 `output.last_hidden_state[0]`。
- 定义固定 encoder signature：

```text
qwen3-vl-4.57.6:layers-0-49:final-norm-identity:last-hidden-state
```

减少 `output_hidden_states=True` 也避免保存不需要的 50 层输出。

## 5. Qwen condition cache v2

修改：

`src/fastwam/datasets/h3_condition_cache.py`

变化：

- schema version：1 -> 2
- cache digest 现在包含：
  - schema version
  - Qwen encoder implementation signature
  - instruction
  - post-transform first-frame pixels和 shape
- 文件名后缀：

```text
.h3-qwen-prenorm-layer50-v2.pt
```

- payload 新增：

```python
"encoder_signature": H3_QWEN_ENCODER_SIGNATURE
```

- loader 同时校验：
  - schema version
  - hidden layer
  - encoder signature

旧 v1 cache 不会被新实现误读。

### 5.1 真实 cache smoke

`scripts/precompute_h3_conditions.py` 增加可选 `max_samples`，用于真实小规模 smoke：

```bash
python scripts/precompute_h3_conditions.py \
  task=libero_h3_uncond_2cam224_1e-4 \
  +overwrite=true \
  +max_samples=1
```

已使用真实 LIBERO 第 0 个样本成功生成一个 cache v2 文件。

尚未生成完整数据集 cache。

## 6. H3 checkpoint mixed precision

### 6.1 官方真实权重 dtype

真实 checkpoint 中：

```text
video_patch_proj      FP32
time_embedder         FP32
final_layer.video_out FP32
condition_proj        BF16
H3 block qkv/out      BF16
```

完整 H3 Video Expert 单卡加载：

```text
peak allocated = 61.805 GiB
```

### 6.2 基线失败

关闭 autocast 后，基线的三个真实模块都报：

```text
mat1 and mat2 must have the same dtype,
but got BFloat16 and Float
```

失败路径：

- video patch projection
- timestep embedder
- final video output projection

打开 autocast 时它们可以运行，但 trainer 和 evaluator 不应依赖外部偶然 autocast。

### 6.3 显式 dtype boundary 修复

修改：

`src/fastwam/models/minimax_h3/video_dit.py`

行为：

- patch 输入先转为 `video_patch_proj.weight.dtype`，即 FP32。
- patch projection 后转回 H3 activation dtype，当前为 BF16。
- timestep sinusoidal embedding 进入每个 linear 前匹配该 linear 的 weight dtype。
- time embedder 输出转回请求的 activation dtype。
- final hidden 在进入 FP32 `video_out` 前转 FP32。
- final logits 再转回 H3 hidden dtype。
- Qwen condition 输入显式匹配 `condition_proj.weight.dtype`。

真实 batch=1 training smoke 没有使用外部 autocast，证明这些边界在完整模型路径可运行。

## 7. H3 attention LoRA

### 7.1 原问题

基线配置：

```text
freeze_video_expert = true
lambda_video        = 1.0
```

因为 H3 不读取 Action/State，且整个 H3 frozen：

```text
loss_video
```

不会更新任何参数，只是一个 diagnostic metric。

### 7.2 实现

新增：

- `H3LoRABranch`
- `H3LoRALinear`
- `MiniMaxH3VideoBackbone.inject_attention_lora()`
- `MiniMaxH3VideoBackbone.lora_branches()`

目标模块仅为每个主 H3 block 的：

```text
attn.qkv_proj
attn.out_proj
```

没有注入 TokenRefiner、MLP、AdaLN 或 final output。

默认配置：

```yaml
h3_lora_rank: 32
h3_lora_alpha: 32.0
h3_lora_dropout: 0.0
```

LoRA 初始化：

- A：Kaiming uniform
- B：全零
- 初始 LoRA 对 base 输出严格为 no-op
- base linear 保持 frozen

共 50 层、每层 qkv/out 两个 LoRA branch：

```text
100 LoRA branches
```

### 7.3 Trainer wiring

`FastWAMH3.trainable_modules()` 现在返回：

- 完整 Action Expert
- 当 H3 frozen 时，单独返回所有 H3 LoRA branches

不会因为把整个 wrapper 作为 trainable module 而误解冻 base linear。

`train()` 保持 H3 base eval，同时单独设置 LoRA branch 的 train/eval 状态。

### 7.4 Checkpoint wiring

现有 Scheme A checkpoint schema version 仍为 2。

新增 payload：

```python
"h3_lora_rank": ...
"h3_lora": [branch.state_dict(), ...]
```

load 时校验 branch 数量，然后逐 branch strict load。

Reviewer 应重点检查：

1. LoRA state 当前按 branch 顺序保存，不按稳定模块名保存。
2. 如果未来 block/module 遍历顺序变化，旧 checkpoint 可能错误对应。
3. 更稳健的设计可能是保存带完整 module path 的 named state dict。
4. schema version 仍为 2；旧 schema-2 checkpoint 没有 LoRA payload 时会保留零初始化
   LoRA，而不是明确拒绝。是否应升级到 schema 3 需要决定。

## 8. 梯度路径验证

### 8.1 Tiny 两层结构验证

视频损失：

```text
video loss -> H3 parameters：有梯度
video loss -> Action Expert：无梯度
```

动作损失：

```text
action loss -> Action Expert：有梯度
action loss -> H3 qkv：有梯度
```

两层模型中，action loss 对 H3 attention LoRA：

```text
layer 0 qkv: non-zero
layer 0 out: non-zero
layer 1 qkv: non-zero
layer 1 out: zero
```

最后一层 H3 out-proj 不影响同层 Action 读取的 H3 K/V，所以 action loss 对最后一层
out-proj 为零是结构预期，不是 wiring bug。video loss 仍会训练该 out-proj。

### 8.2 非对称性和 sensitivity

tiny real implementation：

```text
改变 noisy action -> H3 video prediction max diff = 0.0
改变 state        -> action prediction max diff = 0.02821
改变 noisy video  -> action prediction max diff = 0.02612
```

这确认：

- H3 不读取 Action。
- Action 读取 State。
- Action 读取 H3 world representation。

这些只是局部 sensitivity 证据，不证明训练后学会正确 action following。

## 9. Evaluation API

基线只有：

```python
infer(...)
```

但 evaluator 会调用：

```python
infer_joint(...)
infer_action(...)
```

新增兼容 wrapper：

- `infer()`：唯一 joint sampler。
- `infer_joint()`：参数转换后调用 `infer()`。
- `infer_action()`：仍然完整 joint denoise，只返回 action 主输出。

没有增加第二套 action-only sampler，也没有引入静态 H3 KV cache。

`infer_joint()` 会忽略旧 evaluator 的：

```text
test_action_with_infer_action
```

并继续执行唯一 joint sampler。

## 10. ActionDiT 初始化 artifact

执行：

```bash
PYTHONPATH=src python scripts/preprocess_h3_action_dit_backbone.py \
  --model-config configs/model/fastwam_h3.yaml \
  --h3-transformer-dir /root/wuqingman/models/MiniMax-H3/FL2VA/transformer \
  --output checkpoints/H3ActionDiT_video_interp_1024hdim.pt \
  --dtype bfloat16
```

结果：

```text
parameters   = 2.415B
copied       = 101 tensors
interpolated = 404 tensors
file size    = approximately 4.5 GiB
```

artifact 当前只是本地文件，没有 commit、上传、checksum 或 artifact manifest。

Reviewer 应检查 interpolation 和 alpha scaling 是否仍符合期望初始化策略。

## 11. 真实 batch=1 training smoke

### 11.1 输入

真实 LIBERO 样本：

```text
video = [1, 3, 5, 224, 448]
action = [1, 32, 7]
Qwen cache = [1, 140, 5120]
```

配置：

```text
batch size = 1
gradient checkpointing = on
H3 qkv/out LoRA rank = 32
Action Expert = full train
external autocast = off
optimizer = AdamW
base progress = 0.5
```

### 11.2 模型和显存

```text
trainable parameters = 2.477613591B
LoRA branches        = 100
model load peak      = 71.2839 GiB
forward peak         = 72.7728 GiB
backward peak        = 76.1104 GiB
optimizer-step peak  = 89.8907 GiB
```

### 11.3 Loss 和 sigma

```text
total loss  = 2.0339127
video loss  = 0.3542247
action loss = 1.6796880

base progress = 0.5
sigma video   = 0.924
sigma action  = 0.832
```

这确认共享 base progress 经过两个 shift 后产生不同 sigma。

### 11.4 梯度

```text
first H3 qkv LoRA-B grad norm = 44.7045
first H3 out LoRA-B grad norm = 1.76943
Action Expert grad tensors    = 514
all Action gradients finite   = true
```

随后成功执行：

```text
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

没有 OOM。

### 11.5 配置调整

原 task 默认：

```text
B=16 per GPU
GC off
gradient accumulation=1
```

真实 B=1 optimizer step 已峰值约 89.9 GiB，因此原配置不可信。

当前改为：

```text
B=1 per GPU
GC on
num_workers=4
gradient accumulation=16
```

8 GPU 时理论 effective global batch 仍为：

```text
1 * 8 * 16 = 128
```

注意：89.9 GiB 是单进程 `torch.cuda.max_memory_allocated()`，没有包含完整 8-GPU
DeepSpeed/DDP/NCCL 运行的所有额外 reserved memory 和通信 buffer。96 GiB H20 的余量
不大，多卡生产训练仍可能 OOM。

## 12. 真实联合推理 smoke

参数：

```text
num_frames = 5
action_horizon = 32
num_inference_steps = 3 sigma points
```

H3 scheduler 语义下执行 2 次 update。

修复短 decode 后：

```text
output video frames = 5
output frame size   = 448 x 224
output action shape = [32, 7]
action finite       = true
peak allocated      = 71.4527 GiB
```

### 12.1 Denormalization

LIBERO evaluator 实际使用：

```python
normalizer = processor.normalizer.normalizers["action"][action_key]
denorm = normalizer.backward(action)
```

该真实 evaluator 路径验证：

```text
input shape  = [1, 32, 7]
output shape = [1, 32, 7]
finite       = true
```

零归一化 action 对当前 stats 的输出范围：

```text
min = 0.0
max = 0.5
```

一次 smoke 脚本曾错误调用不存在的 `dataset.processor`，产生
`AttributeError`；真实 processor 位于 `dataset.lerobot_dataset.processor`。
该错误发生在 joint infer 和 VAE decode 已成功之后，是验证脚本错误，不是模型错误。

## 13. Dataset/cache 启动修复

首次 precompute cache 时，dataset 会计算 normalization stats 并写：

```text
./runs/dataset_stats.json
```

基线没有创建 `./runs`，真实报错：

```text
FileNotFoundError: ./runs/dataset_stats.json
```

修改：

`src/fastwam/datasets/lerobot/robot_video_dataset.py`

保存前执行：

```python
os.makedirs(work_dir, exist_ok=True)
```

注意：当前配置没有使用 `pretrained_norm_stats`，多次 dataset instantiate 仍会重复计算
stats。这是性能问题，本轮只修复了首次启动阻断。

## 14. 测试变更

新增或增强的 regression coverage：

- VAE video 路径必须使用 `encode_prefix=True`。
- prefix 返回必须包含 pad counts。
- 5 帧 decode 必须把 2 latent 补成 7-row decoder window。
- mixed checkpoint dtype 下 patch/time/final projection 无 autocast 可运行。
- LoRA 初始输出严格等于 base。
- LoRA base 保持 frozen。
- LoRA-B 能收到梯度。
- Qwen conditioner 使用 `last_hidden_state` 而不是 hidden state tuple。
- cache 文件必须是 prenorm schema v2。
- `infer_joint()` 和 `infer_action()` 必须复用同一 sampler。

最终：

```text
75 passed, 1 warning
```

warning 仅为环境中的 `pynvml` deprecation。

## 15. 修改文件清单

生产代码：

- `configs/model/fastwam_h3.yaml`
- `configs/task/libero_h3_uncond_2cam224_1e-4.yaml`
- `scripts/precompute_h3_conditions.py`
- `src/fastwam/datasets/h3_condition_cache.py`
- `src/fastwam/datasets/lerobot/robot_video_dataset.py`
- `src/fastwam/models/minimax_h3/fastwam.py`
- `src/fastwam/models/minimax_h3/text_encoder.py`
- `src/fastwam/models/minimax_h3/video_dit.py`
- `src/fastwam/models/minimax_h3/video_vae.py`
- `src/fastwam/runtime.py`

测试：

- `tests/models/minimax_h3/test_condition_cache.py`
- `tests/models/minimax_h3/test_conditioning.py`
- `tests/models/minimax_h3/test_inference_contract.py`

## 16. 明确尚未完成或尚未证明的事项

以下事项不能宣称完成：

1. 没有生成完整 LIBERO Qwen cache v2；只生成了一个真实样本。
2. 没有运行 8×H20 DeepSpeed/Accelerate 生产 trainer。
3. 没有验证 DDP/NCCL/ZeRO 下的真实峰值显存。
4. 没有运行长训练或观察 loss curve。
5. 没有生成训练 checkpoint 后再做 save/load/resume smoke。
6. 没有运行真实 LIBERO rollout success rate。
7. 没有证明 action following，只证明了梯度、局部 sensitivity 和接口可运行。
8. 没有做 ActionSynth 风格的训练后 action/world/state swap 判别实验。
9. 没有测试 B=2 或更高 batch。
10. 没有实现 NF4、CPU offload 或 optimizer sharding 优化。
11. 没有 commit 或 push 本轮修改。
12. 5-frame latent decode padding 策略没有官方短 decode API 作为语义 gold reference。

## 17. 建议 reviewer 重点审查

### P0

1. 5 帧 decode 通过重复最后 latent 补至 7 rows 是否是最佳语义。
2. 自定义 LoRA wrapper 是否与 checkpoint、DDP、DeepSpeed ZeRO 完全兼容。
3. `trainable_modules()` 返回 100 个 LoRA branch 是否会被 optimizer/Accelerate
   正确去重并处理。
4. checkpoint 是否应从 schema 2 升级 schema 3。
5. LoRA state 是否应按稳定模块名保存，而不是按 branch 顺序保存。
6. 89.9 GiB 单卡 peak 在生产多卡栈中是否仍有足够余量。

### P1

1. `infer_action()` 返回 action-only dict 是否满足所有 evaluator 调用方。
2. `infer_joint()` 对旧参数的兼容范围是否完整。
3. cache signature 是否还应加入 checkpoint hash、processor hash 或 git revision。
4. dataset stats 是否应成为固定 artifact，避免每次启动重新计算。
5. ActionDiT 初始化 artifact 是否需要 safetensors、checksum 和 manifest。

### 训练质量

1. `loss_action` 会更新 H3 qkv LoRA。
2. `loss_action` 会更新非最后层 H3 out-proj LoRA。
3. `loss_video` 会更新全部目标 H3 LoRA。
4. 这属于 joint representation adaptation，不违反“H3 不读取 Action”的非对称约束。
5. 但 gradient/sensitivity 不能证明 action following。

## 18. 建议下一步执行顺序

1. 独立 review 本文列出的代码 diff。
2. 根据 review 修正 LoRA checkpoint schema 和 5-frame decode 策略。
3. 为完整数据集生成 Qwen cache v2。
4. 执行 8×H20、B=1、GC-on、accumulation=16 的 1-step trainer smoke。
5. 保存 checkpoint。
6. 新进程 load/resume 1 step。
7. 跑 2–3 step joint inference 和 VAE decode。
8. 跑最小 LIBERO rollout。
9. 训练后执行：
   - 固定 instruction/state/action noise，改变 world representation。
   - 固定 world/instruction/action noise，改变 state。
   - 改变 noisy action，确认 H3 video branch 严格不变。
10. 最终以 LIBERO success rate 判断 action following。

## 19. 证据日志

本地终端日志可用于复核：

- VAE 5/22/39：
  `/root/.cursor/projects/root-wuqingman/terminals/537836.txt`
- VAE deterministic parity：
  `/root/.cursor/projects/root-wuqingman/terminals/309775.txt`
- H3 mixed dtype：
  `/root/.cursor/projects/root-wuqingman/terminals/956681.txt`
- 真实 Qwen layer49 reference：
  `/root/.cursor/projects/root-wuqingman/terminals/428879.txt`
- ActionDiT artifact：
  `/root/.cursor/projects/root-wuqingman/terminals/104183.txt`
- 真实 batch=1 training smoke：
  `/root/.cursor/projects/root-wuqingman/terminals/499382.txt`
- 首次推理发现 2-latent decode blocker：
  `/root/.cursor/projects/root-wuqingman/terminals/64925.txt`
- 修复后真实 joint sample：
  `/root/.cursor/projects/root-wuqingman/terminals/331514.txt`
- FP32/BF16 VAE fidelity：
  `/root/.cursor/projects/root-wuqingman/terminals/172998.txt`

## 20. 最终状态

本轮目标中的真实训练闭环已经完成，但后续 gold parity 发现 5 帧 auxiliary video
decode 仍有一个未解决的 P0 语义问题。

可以准确宣称：

```text
真实 33B H3 + 2.415B Action Expert + H3 LoRA
在单张 H20 上完成了：
load -> real LIBERO forward -> backward -> AdamW step
以及 joint video/action sampling。
当前 5-frame VAE decode 只做到运行不崩，输出语义没有通过 gold parity。
```

不能宣称：

```text
已经完成生产多卡训练
已经学会 action following
已经达到 LIBERO 成功率
2-latent 可以正确恢复 5-frame auxiliary video
```

所有修改目前仍是未提交工作树，等待独立 review 后再决定 commit/push。

## 21. Review 后补充实验（2026-08-20）

### 21.1 Reviewer 提议的 decode gold parity：失败

首先严格执行 reviewer 提议的实验。使用真实 FL2VA VAE、真实 LIBERO 像素、
posterior mean，并只编码一次 22 帧得到：

```text
z7 shape = [1, 24, 7, 14, 28]
```

比较：

```text
A: z0...z6 -> decode_base(frame_num=5)
B: z0,z1 -> repeat z1 to 7 rows -> decode_base(frame_num=5)
```

结果：

```text
FP32:
  max_abs  = 1.7059225
  mean_abs = 0.1164824
  PSNR     = 19.7521 dB
  equal    = false

BF16:
  max_abs  = 1.7031250
  mean_abs = 0.1158855
  PSNR     = 19.7791 dB
  equal    = false
```

这不是浮点误差级差异。

### 21.2 原 gold 定义还包含一个时序误区

官方 `trim_output()` 在 `causal_encoder=True` 时执行：

```python
dec = dec[:, :, -target_frames:, :, :]
```

所以：

```text
decode_base(z7, frame_num=5)
```

得到的是完整 reconstruction 的**最后 5 帧**，不是 reviewer 描述的“前 5 帧”。

修正后的 reference 应为：

```text
z7 -> decode_base(frame_num=22) -> 显式取 [:5]
```

该 reference 对输入 5 帧的 reconstruction：

```text
mean_abs = 0.0283114
PSNR     = 30.3389 dB
```

而当前 2-latent repeat workaround 对同一输入：

```text
mean_abs = 0.1318395
PSNR     = 18.3757 dB
```

当前 workaround 与正确 22-frame reconstruction 的前 5 帧：

```text
max_abs  = 1.7731919
mean_abs = 0.1299718
PSNR     = 18.4851 dB
```

### 21.3 进一步候选策略也失败

确认：

```text
encode_prefix(22)[:2] == encode_prefix(5)
max_abs = 0.0
```

因此差异不是两次 encode 或前两个 latent 不一致导致。

另外测试了：

1. 直接 `_adaptive_decode(z0,z1)` 后取前 5 帧。
2. 直接 `_adaptive_decode(z0,z1)` 后取后 5 帧。
3. 临时把 token/frame drop 和 overlap 设为 0 后 decode。
4. repeat 到 7、decode 22 帧后显式取前 5 帧。

它们对正确 reference 的 PSNR 分别约为：

```text
12.08 dB
10.55 dB
11.24 dB
11.21 dB
```

均不能作为正确短 decoder。

### 21.4 根因结论

`encode_prefix=True` 为 5 帧输入补黑帧到合法窗口，然后删除 3 个尾部 padding
latent，只保留 2 个 prefix latent。删除的 rows 对 H3 prefix conditioning 是合理的，
但官方 temporal decoder 的 receptive field、token_drop 和 chunk alignment 仍依赖
完整窗口。2 latent 本身不足以通过现有官方 decoder 重建正确的 5 帧视频。

所以需要区分：

```text
5 frames -> 2 latents 作为 H3 prefix/target 表示：已验证
2 generated latents -> 官方 decoder -> 正确 5 frames：已被实验否定
```

当前 repeat workaround 应视为 P0 待决，不应在正式提交中被描述成正确 decoder。

候选决策包括：

1. 5 帧训练保持 2 latent，但 inference 不返回 auxiliary video。
2. 训练/predict 完整可解码 latent window，包括被 prefix encoder 删除的尾 rows。
3. 使用至少 22 帧/7 latent 的视频 target。
4. 增加专门的 short-prefix decoder。

这些选项会改变 inference contract 或视频 target contract，需要架构 owner 明确选择。

### 21.5 真实 H3 LoRA 双 loss 梯度关系

使用真实 33B H3、真实 Action Expert、真实 LIBERO 样本，固定完全相同 RNG 和
`base_progress=0.5`，分别执行：

```text
L_video only
L_action only
```

代表层 LoRA-B 梯度：

```text
block 0 qkv:
  ||g_video||  = 0.027599
  ||g_action|| = 55.490509
  ratio        = 2010.59
  cosine       = 0.035743

block 0 out:
  ||g_video||  = 0.009887
  ||g_action|| = 1.810418
  ratio        = 183.11
  cosine       = 0.095738

block 24 qkv:
  ||g_video||  = 0.009031
  ||g_action|| = 0.771978
  ratio        = 85.48
  cosine       = 0.133989

block 24 out:
  ||g_video||  = 0.004075
  ||g_action|| = 0.339120
  ratio        = 83.22
  cosine       = 0.104497

block 49 qkv:
  ||g_video||  = 0.008093
  ||g_action|| = 0.038348
  ratio        = 4.74
  cosine       = 0.002446

block 49 out:
  ||g_video||  = 0.001400
  ||g_action|| = 0.0
```

所有实际存在的梯度均 finite。最后一层 H3 out-proj 的 action gradient 为零符合
当前 layer ordering：Action 读取该层 H3 K/V，而最后的 H3 out-proj 不再影响 Action。

### 21.6 梯度诊断结论

单个真实样本的初始状态下，action loss 对 H3 LoRA 的梯度在早中层显著大于 video
loss；block 0 qkv 甚至相差约 2011 倍。两者 cosine 为小正值，当前样本上不是明显
反向冲突，但 `lambda_video=1, lambda_action=1` 并不意味着 H3 收到平衡监督。

这不是启动 blocker，也不能由一个样本直接确定最终 loss 权重，但正式长训练前至少
应选择以下一种策略：

1. 对 H3 LoRA 单独降低 action-loss 权重。
2. 采用 per-loss gradient normalization/balancing。
3. 暂时 stop-gradient action -> H3，作为对照实验。
4. 在多个样本和多个 step 上统计 norm/cosine 后再定权重。

当前证据支持保留 joint optimization 作为设计方向，但不支持未经监控地直接使用
`1:1` 权重跑长训练。

## 22. Production integration 收口（2026-08-20 晚）

### 22.1 5-frame pixel decode 已从生产默认路径移除

真实 gold parity 已证明 2 latent repeat 到 7 的策略错误，因此现在：

```text
infer(..., decode_video=False)
```

默认返回：

```text
video_latents
action
```

5 帧 latent 仍完整联合去噪，Action 仍读取 H3 world representation，但不会再执行
昂贵且语义错误的 pixel decoder。

显式请求：

```text
num_frames=5
decode_video=true
```

会在采样开始前抛出 `NotImplementedError`。22/39 等具备原生可解码窗口的输出仍可
显式设置 `decode_video=true`。

`infer_action()` 强制 `decode_video=false`；CLI 会保存：

```text
*.action.pt
*.video_latents.pt
```

仅在输出确实包含 pixel video 时才保存 MP4。

### 22.2 第一轮 baseline 恢复 FastWAM-like joint gradient

新增配置：

```yaml
stop_action_gradient_to_h3: false
```

默认不对 Action attention 使用的 H3 K/V 做 detach：

```text
forward:
  H3 -> Action 保持不变

backward:
  L_video  -> H3 LoRA
  L_action -> Action Expert
  L_action -> H3 LoRA
  H3 base 33B remains frozen
```

真实 33B、真实 LIBERO 样本验证：

```text
Action-only loss:
  H3 LoRA grad tensors = 200
  H3 non-zero grads     = 0
  Action grad tensors   = 514
  Action non-zero grads = 514

Video-only loss:
  H3 LoRA grad tensors = 200
  H3 non-zero grads     = 100
  Action grad tensors   = 514
  Action non-zero grads = 0

all gradients finite = true
```

以上零梯度结果验证的是仍然保留的可选 stop-gradient 路径。需要注意，zero tensor
并不保证 AdamW 完全不更新参数，decoupled weight decay 仍可能生效；要求绝对冻结时
应将 H3 LoRA 移出 optimizer。

### 22.3 Checkpoint schema 3

weights checkpoint 现在严格保存：

```text
schema_version = 3
Action Expert config + named state_dict
H3 LoRA rank/alpha/dropout
H3 LoRA full target module names
base H3 config/index fingerprint
named LoRA state_dict
```

LoRA key示例：

```text
blocks.0.attn.qkv_proj
blocks.0.attn.out_proj
...
blocks.49.attn.out_proj
```

load 会严格比较：

- schema version
- Action Expert config
- rank/alpha/dropout
- base H3 fingerprint
- 完整 module name 集合
- tensor shape/state

schema 2 会明确拒绝，不再静默恢复 zero-init LoRA。

### 22.4 ActionDiT artifact contract

初始化脚本现在自动生成相邻 manifest：

```text
H3ActionDiT_video_interp_1024hdim.pt.manifest.json
```

记录：

- filename
- exact byte size
- artifact SHA256
- source H3 config/index SHA256
- Action Expert config
- 完整生成命令
- dtype
- interpolation/scaling 策略
- copied/interpolated tensor 数量

加载 ActionDiT 前会强制检查 manifest、size 和 SHA256。缺失或 checksum 不一致会拒绝
启动。

当前 artifact：

```text
size   = 4,827,088,347 bytes
SHA256 = 6b0a3de516f67bc2d1c1e92712ead856c68b4cdaa067f78667dbacbc357230e6
```

### 22.5 Qwen cache schema 3

cache directory 现在必须包含 manifest，并记录：

- encoder implementation signature
- Qwen checkpoint config/index fingerprint
- processor/tokenizer fingerprint
- hidden layer
- schema version

每个 sample digest 和 payload 都绑定 Qwen checkpoint manifest/index 与 processor
fingerprint。不同 manifest/index、processor 或 tokenizer 不能误读旧 cache；当前尚未
逐字节 hash `.safetensors` shards，因此训练依赖固定且不可修改的官方 release 目录。

已用真实 Qwen-32B 和真实 LIBERO 样本成功生成一个 schema-3 cache sample。完整 cache
正在重新生成。

### 22.6 Action MM-RoPE 时间 contract

effective Action RoPE fps 现在由实际 shape 计算：

```text
video_fps * action_horizon / (num_frames - 1)
```

如果 task 仍显式配置 `action_fps`，它只作为 assertion；与实际 frame/action layout
不一致会立即报错。基础 model config 使用 `null`，避免隐藏一个错误的通用默认值；
当前 LIBERO task 的 5-frame/32-action override 仍严格检查为 192 Hz。

### 22.7 自动测试

本轮修改后的 H3 测试：

```text
88 passed, 1 warning
```

warning 仍只是环境中的 `pynvml` deprecation。

### 22.8 真实 8×H20 训练与恢复验证

正式 ZeRO-2 smoke 已完成：

```text
GPU                = 8x H20 96 GiB
micro batch        = 1 / GPU
gradient accumulation = 16
global batch       = 128
optimizer step     = success
weights checkpoint = 4.7 GiB
full state          = 166 GiB
```

每卡初始化观测：

```text
allocated = 72.4 GiB
reserved  = 74.94 GiB
```

随后在全新 8-rank 进程中加载：

- model weights
- ZeRO-2 optimizer shards
- scheduler
- random states
- `global_step=1`
- `epoch=0`
- `batch_in_epoch=16`
- dataloader sample offset 128

并成功完成 `step=2`。因此“save → 新进程 load → resume one optimizer step”链路已
通过真实权重与真实数据验证。

### 22.9 Cache 可复现性修复

多卡 smoke 首次暴露了 cache 与 padded-sample retry 的隐含随机性：

```text
相同 dataset index
-> 随机选择不同 unpadded replacement
-> f0 digest 改变
-> offline cache miss
```

现在每个初始 index 使用由该 index 固定 seed 的 retry RNG；precompute 和 train 对同一
index 必然选择同一 replacement。`FileNotFoundError` 也不再被 `__getitem__` 吞掉并
随机换样本，避免用随机替代掩盖不完整 cache。

precompute 新增：

- 多样本 batch 编码
- 指定 sampler seed 的 deterministic smoke subset
- batch 内 cache path 去重

真实完整 dataset 长度是 `277,713` 个窗口，不是 normalization 日志里的 1,712 个
episode。按真实速度估计，全量 cache 约需 19 小时，预计占用约 0.9 TiB。

### 22.10 FastWAM-like joint-gradient 最快稳定 setting

8×H20 sustained benchmark：

```text
B=1, no gradient checkpoint:
  1.75 processed samples/s (10 steps)

B=1, gradient checkpoint:
  1.50 processed samples/s (3 steps)

B=2, no gradient checkpoint:
  2.38 processed samples/s (10 steps)
  GPU memory = 95.4--96.2 GiB / 97.9 GiB

B=2, no gradient checkpoint, accumulation=8:
  2.37 processed samples/s (1 full optimizer step / 8 microsteps)
```

前三组 benchmark 显式设置 accumulation=1；最后一组使用正式 accumulation=8。
Trainer 的通用统计公式现已正确计入 accumulation。

结论：

```text
batch_size = 2
mot_checkpoint_mixed_attn = false
gradient_accumulation_steps = 8
```

是当前实测最快的正式训练 setting，比 B=1 no-GC 快约 36%，同时保持 global batch
128。B=2 只剩约 1.7--2.4 GiB 显存余量；10-step 和 accumulation=8 smoke 均成功，
长训练仍需监控，若出现 OOM 则回退到 B=1/accumulation=16。

正式预算保持 `max_steps=100000`。`EXPERIMENT_37.md` 已记录用户明确确认总计
100k optimizer steps、每 10k 保存一次；约 46.1 次数据遍历是有意设计，而不是
`num_epochs=10` 应当生效但被意外覆盖。

### 22.11 Checkpoint 磁盘保留策略

完整 ZeRO-2 state 每次约 166 GiB。新增：

```yaml
max_checkpoints: 2
```

每次 checkpoint 完成后只保留最新两个 full states 和最新两个 named schema-3 weights，
防止长实验因 checkpoint 累积耗尽磁盘。最终 step 若已命中周期 checkpoint，也不再
重复写入同一份约 166 GiB state。

### 22.12 正式实验状态

原 stop-gradient 流水线已停止，防止按过期配置自动启动。joint-gradient smoke test
通过后将恢复已有 cache 并启动：

```text
full schema-3 Qwen cache
  -> cache complete sentinel
  -> 8xH20 large Scheme-A training
```

大训练 run id：

```text
scheme-a-large-jointgrad-b2-nogc-20260821
```
