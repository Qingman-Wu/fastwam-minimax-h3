# FastWAM-H3：Review 后修复、官方 Parity 与 Cache 重建记录

日期：2026-08-23  
工作目录：`/root/wuqingman/FastWAM-H3-scheme-a-verify`  
状态：未提交、未推送、未启动正式 100k

## 1. 文档范围

本文只记录以下 review 结论之后发生的工作：

> 先修复 metadata/hash/strict-skip/eval fail-fast；增加 reviewer 要求的官方 parity；暂停全量 VAE cache，Post-Refiner cache 可继续；完成严格全量 audit 和 B=2 cache-only step；再提交并推送明确 commit；通过后才启动 100k。

本文不是对旧审查文档的重复，而是后续实际修改、实验结果、失败诊断、cache 失效与重建状态的增量 handoff。

当前最重要的结论是：

1. review 非常有效，最终通过真实 50 层 parity 找到了一个之前未识别的数值缺口：本地 `H3RMSNorm` 强制 FP32 归一化，而 pinned 官方 Diffusers 使用原生 BF16 `RMSNorm` kernel。
2. 该差异在单层看似较小，但 50 层后累积为最终 velocity 约 `7.59%` relative RMS。
3. 改为官方 RMSNorm 路径后，真实 checkpoint 的 50 层 hidden state 和最终 velocity 与官方实现达到逐元素零差异。
4. 同一个 RMSNorm 实现也被 TokenRefiner 使用，因此此前生成的 post-Refiner cache 数值不再有效，不能仅升级 metadata 后保留。
5. 已将 Refiner 和 schema-4 implementation signature 升级，并启动 8 卡全量重建。
6. VAE cache 仍未启动全量生成；正式 100k 仍未获准启动。

## 2. 当前总体状态

已经完成：

- logical tensor hash 修复；
- schema-4 Qwen/Refiner fingerprint 补强；
- cache strict-skip；
- `load_vae=false` 与周期评测的训练启动期 fail-fast；
- keyframe 官方 helper parity；
- prefix posterior moments/sample parity；
- timestep/AdaLN 官方 parity；
- scheduler 完整 trajectory 官方 parity；
- synthetic single-block 官方 parity；
- 真实 checkpoint 50-block + final velocity 官方 parity；
- 真实 checkpoint TokenRefiner 官方 parity；
- H3 回归测试，共 `108 passed`。

正在进行：

- 8 卡重建修正后的 post-Refiner schema-4 cache。

尚未完成：

- 重建完成后的 direct Qwen+Refiner vs schema-4 payload parity；
- 新 cache 的严格全量 audit；
- B=2 cache-only optimizer step；
- 审查文档最终更新；
- commit 和 push；
- VAE 全量 cache；
- 1k/5k canary；
- 正式 100k。

## 3. Cache metadata、hash 与 strict-skip 修复

### 3.1 VAE logical tensor hash

文件：

- `src/fastwam/datasets/h3_vae_cache.py`
- `tests/models/minimax_h3/test_vae_cache.py`

旧实现使用 tensor storage 级别字节。对于 view/slice，这可能把逻辑 tensor 之外的共享 backing storage 一并纳入 hash，与此前 2.6 TB shared-storage 序列化问题属于同一风险类别。

修改后先执行：

- `detach()`
- CPU
- contiguous
- `view(torch.uint8).numpy()`
- `memoryview(...)`

hash 只覆盖 tensor 的逻辑内容。

新增回归测试验证：

- 对 backing tensor 的 slice 计算 digest；
- 修改 slice 外的 backing storage；
- digest 不应变化。

### 3.2 Condition cache logical checksum

文件：

- `src/fastwam/datasets/h3_condition_cache.py`

对图像 digest 和 cache payload checksum 使用 logical contiguous tensor bytes，不再读取整个 untyped storage。

### 3.3 Schema-4 同时绑定 Qwen 与 Refiner

涉及文件：

- `src/fastwam/models/minimax_h3/text_encoder.py`
- `src/fastwam/models/minimax_h3/video_dit.py`
- `src/fastwam/datasets/h3_condition_cache.py`
- `scripts/convert_h3_post_refiner_cache.py`
- `scripts/precompute_h3_post_refiner_complete.py`
- `scripts/upgrade_h3_post_refiner_manifest.py`

schema-4 manifest 现在校验：

- Qwen checkpoint fingerprint；
- Qwen 实际 weight shard content fingerprint；
- Qwen weight shard size/mtime stat signature；
- processor fingerprint；
- layer-50 prenorm encoder signature；
- presentation / token-tag signature；
- Refiner 实际 artifact content fingerprint；
- Refiner artifact size/mtime stat signature；
- Refiner implementation signature；
- schema-4 implementation signature；
- BF16 embedding dtype；
- width `5376`；
- source schema-3 manifest contract。

Refiner fingerprint 修复了一个实际 bug：早期代码对 `config.json` 和 `model.safetensors.index.json` 的处理没有正确纳入文件内容。现实现直接 hash 文件内容，再 hash index 指向的实际 shard 内容。

### 3.4 Strict-skip

涉及文件：

- `scripts/convert_h3_post_refiner_cache.py`
- `scripts/precompute_h3_post_refiner_complete.py`

旧行为：

```text
目标路径存在 -> skip
```

新行为：

```text
目标路径存在
  -> 使用当前 manifest 严格 load
  -> 验证 fingerprint、dtype、shape、checksum、finite、source/key
  -> 全部通过才 skip
  -> 任一失败则重建
```

因此损坏文件、旧 manifest 文件和错误 dtype 文件不会永久被误认为完成。

### 3.5 Manifest upgrade 工具

新增：

- `scripts/upgrade_h3_post_refiner_manifest.py`

该工具只适用于数值 payload 没有变化、仅 metadata contract 加强的情况。它曾用于升级旧 schema-4 manifest。

但是本轮发现 RMSNorm 数值实现发生变化，因此不能再靠 manifest upgrade 保留旧 embedding；必须重新运行 Refiner。

## 4. Evaluation fail-fast

涉及文件：

- `src/fastwam/trainer.py`
- `tests/test_trainer_h3_eval.py`

新增启动期检查：

```text
eval_every > 0
+ val_dataset 存在
+ H3 model 未加载 VAE
=> 启动时立即 ValueError
```

这样不会在 cache-only 训练运行若干 step 后才进入 `model.infer()` 并因缺少 VAE 失败。

已验证：

- `load_vae=false + periodic inference eval` 被拒绝；
- `eval_every=0` 允许；
- 没有 validation dataset 时允许。

## 5. VAE posterior clamp-order 验证

文件：

- `src/fastwam/models/minimax_h3/video_vae.py`
- `tests/models/minimax_h3/test_vae_cache.py`
- `scripts/verify_h3_prefix_posterior_parity.py`

当前语义：

1. released `DiagonalGaussianDistribution` 在 raw latent space 执行：

```text
raw_logvar = clamp(raw_logvar, -30, 20)
```

2. cache normalization：

```text
mean_norm   = (mean_raw - latent_mean) / latent_std
logvar_norm = raw_logvar - 2 * log(latent_std)
```

3. 在线采样：

```text
sample_norm = mean_norm + exp(0.5 * logvar_norm) * epsilon
```

4. 不会对 `logvar_norm` 再 clamp。

新增 synthetic regression 特意构造：

- raw logvar 小于 `-30`；
- raw logvar 大于 `20`；
- 不同 channel latent std；
- normalized logvar 可以超出 `[-30, 20]`；
- cached normalized sample 与“raw-space sample 后除以 std”一致。

测试 tolerance：

- `atol=1e-6`
- `rtol=1e-6`

## 6. Keyframe 官方 parity

新增：

- `scripts/verify_h3_keyframe_official_parity.py`

官方 reference：

- Diffusers revision `2f7e0154a9db246e95c9ede43edba7db5b130805`
- `diffusers.modular_pipelines.minimax_h3.encoders.encode_vae_condition`

隔离 parity 环境：

- Diffusers pinned archive：`/root/wuqingman/.deps-diffusers-h3-archive`
- venv：`/root/wuqingman/.venv-diffusers-h3`
- Torch `2.5.1+cu124`
- torchvision `0.20.1+cu124`
- Transformers `4.57.6`

对比覆盖：

- 输入从 `[-1, 1]` 严格还原并 round 到 uint8；
- uint8 后除以 255；
- ImageNet mean/std；
- released VAE posterior；
- 独立 CPU generator；
- seed `42`；
- CPU 随机数后移到 CUDA 的语义；
- raw latent FP16 rounding；
- FP32 latent normalization。

结果：

```text
shape=(1, 24, 1, 14, 28)
max_abs_diff=0
mean_abs_diff=0
official_keyframe_parity=PASS
```

结论：FastWAM keyframe condition 与 pinned 官方 helper 逐元素一致。

## 7. Prefix posterior moments/sample parity

新增：

- `scripts/verify_h3_prefix_posterior_parity.py`

该测试准确描述为：

> released bundle prefix posterior 与 cached normalized moments 的同噪声代数 parity。

它不是声称公开 Diffusers 有独立的训练 prefix cache API。

测试步骤：

1. 用 released wrapper 的 `encode_videos(..., encode_prefix=True)` 计算原始 posterior sample；
2. 用 FastWAM `encode_video_posterior()` 得到 normalized mean/logvar；
3. 向两条路径显式注入相同 retained posterior noise；
4. 比较 normalized sample。

一个重要细节：

- released 5-frame prefix 路径先产生 5 个 posterior latent slots，再 drop 3 个，只保留 2 个；
- cached moments 直接保存保留后的 2 个 slots；
- 因为 `torch.randn` 的完整 shape 不同，“相同 seed”不保证每个 channel 的 retained noise 相同；
- reviewer 要求的是相同随机噪声，因此测试显式从 5-slot noise 中取 retained 2-slot noise 注入 cache 路径。

这证明的是分布和同噪声代数等价，而不是 seed 到具体样本的 bitwise 映射保持不变。训练语义仍是每次在线重新采样。

结果：

```text
shape=(1, 24, 2, 14, 28)
max_abs_diff=3.57627869e-07
mean_abs_diff=2.87806667e-08
released_prefix_posterior_parity=PASS
```

## 8. Timestep 与 AdaLN 官方 parity

新增：

- `scripts/verify_h3_timestep_scheduler_official_parity.py`

对比：

- FastWAM `H3TimeEmbedder`
- 官方 `Timesteps + TimestepEmbedding`
- FastWAM `H3AdaLNProjection`
- 官方 `MiniMaxH3AdaLayerNormModulation`

覆盖的关键 dtype contract：

- sinusoidal timestep input FP32；
- timestep MLP FP32；
- SiLU 在 FP32；
- 仅在进入 BF16 AdaLN projection 前 cast；
- 三 modality table row layout；
- 六组 modulation 参数顺序。

结果：

```text
official_timestep_adaln_parity=PASS
```

synthetic small-model 对比达到逐元素一致。

## 9. Scheduler trajectory parity 与实现修复

涉及文件：

- `src/fastwam/models/wan22/schedulers/scheduler_continuous.py`
- `src/fastwam/models/minimax_h3/fastwam.py`
- `tests/models/minimax_h3/test_scheduler_parity.py`
- `scripts/verify_h3_timestep_scheduler_official_parity.py`

最初 local additive Euler：

```text
sample_next = sample + local_velocity * delta_sigma
```

与官方公式代数等价，但 BF16 下 operation ordering 不完全相同，完整 trajectory 有约一个 BF16 ULP 的差异。

新增 H3 video 专用：

```python
step_h3_video(...)
```

它严格复现官方：

1. 从模型实际 conditioning progress 恢复 sigma；
2. 构造 data-ward denoised；
3. 使用 FP32 `x_t / x0` blend；
4. 最后 cast 回 sample dtype。

只修改 H3 video inference。Action scheduler 和其他 Wan/FastWAM 路径仍使用原 generic `step()`，避免无关语义变化。

结果：

```text
official_scheduler_trajectory_parity=PASS
```

完整 20 sigma-point trajectory 逐 step 一致。

## 10. Synthetic single-block parity

新增：

- `scripts/verify_h3_block_official_parity.py`

关键 checkpoint/layout 映射：

- 官方分离 Q/K/V；
- FastWAM fused QKV 使用 head-interleaved storage；
- 官方 SwiGLU checkpoint order 与本地 fused `fc1` half order互换；
- norm、AdaLN、output projection 直接映射；
- 本地 raw RoPE angle 与官方 `(cos, sin)` 等价；
- 两边强制 native SDPA。

结果：

```text
shape=(1, 7, 16)
max_abs_diff=0
mean_abs_diff=0
official_single_block_parity=PASS
```

## 11. 真实 50-block parity 暴露 RMSNorm 缺口

新增：

- `scripts/verify_h3_50block_official_parity.py`

测试使用真实 H3 checkpoint weights，并逐层把本地 native checkpoint layout 映射到 pinned 官方 block。

测试输入为短 synthetic packed video sequence，目的不是评估生成质量，而是隔离 50 层数值实现。

### 11.1 修复前结果

最初 local `H3RMSNorm`：

```python
value = x.float()
value = value * rsqrt(mean(value ** 2) + eps)
return value.to(input_dtype) * weight
```

这会强制 FP32 reduction，再转 BF16。

pinned 官方实现使用原生 `nn.RMSNorm`，block stack 为 BF16 kernel contract。

修复前真实 50 层结果：

```text
block 00 relative_rms = 0.00442367233
block 26 relative_rms = 0.0115080271
block 30 relative_rms = 0.0235515647
block 40 relative_rms = 0.0269473027
block 45 relative_rms = 0.0411957018
block 49 relative_rms = 0.0687463135

final_velocity_relative_rms = 0.0758855119
final_velocity_cosine = 0.997318387
```

这证明：

- 单层误差约 `0.44%` relative RMS；
- 误差经过 50 层持续累积；
- 最终 velocity 约 `7.59%` relative RMS；
- 旧实现不能通过官方数值 gate。

尝试把官方 QKV 也切换为 fused GEMM 后结果不变，排除了 QKV fused/unfused 是主因。

独立 BF16 RMSNorm 对比显示：

```text
max_abs_diff=0.03125
relative_rms=0.003049694...
```

与第一层误差量级吻合。

### 11.2 RMSNorm 修复

文件：

- `src/fastwam/models/minimax_h3/video_dit.py`

生产/cache 环境现在使用 PyTorch native `F.rms_norm`，与官方 kernel contract 一致。

仓库的系统 pytest 环境仍是旧 Torch `2.3.1`，没有 `nn.RMSNorm` 和 `F.rms_norm`。为了让纯 CPU legacy test environment 可运行，保留一个只在缺少 native API 时启用的 dtype-local fallback。

重要边界：

- 正式 cache 和训练必须使用项目 venv；
- 项目 venv 当前报告 Torch `2.7.1+cu128`，会走 native `F.rms_norm`；
- pinned parity venv 使用 Torch `2.5.1+cu124`，也走 native `F.rms_norm`；
- legacy Torch 2.3 fallback 不是正式数值基线。

### 11.3 修复后结果

真实 checkpoint 50 层全部：

```text
max_abs_diff=0
relative_rms=0
```

最终：

```text
worst_block_max_abs_diff=0
worst_block_relative_rms=0
final_velocity_max_abs_diff=0
final_velocity_relative_rms=0
final_velocity_cosine=1
official_50block_final_velocity_parity=PASS
```

结论：修复后的 H3 block stack 和 final velocity 与 pinned 官方实现逐元素一致。

## 12. 真实 TokenRefiner parity

新增：

- `scripts/verify_h3_refiner_official_parity.py`

使用真实：

- condition projection weights；
- 两层 TokenRefiner weights；
- final norm；
- BF16；
- native SDPA；
- 官方 Q/K/V、SwiGLU 和 norm 语义。

结果：

```text
shape=(7, 5376)
max_abs_diff=0
mean_abs_diff=0
official_token_refiner_parity=PASS
```

这项测试同时证明 RMSNorm 修复后 post-Refiner embedding 的计算与 pinned 官方实现逐元素一致。

## 13. 为什么旧 post-Refiner cache 必须失效

旧 post-Refiner cache 是在强制 FP32 `H3RMSNorm` 下生成的。

TokenRefiner 中包含：

- block norm1；
- Q norm；
- K norm；
- block norm2；
- final norm。

这些 norm 都受实现变化影响。因此旧 `[L, 5376]` embedding 不是“metadata 旧、payload 仍正确”，而是 payload 本身数值不同。

已执行：

- Refiner implementation signature 从 `v2` 升为 `v3`；
- 新 signature 明确包含 `torch-rmsnorm` contract；
- schema-4 implementation signature 从 `fastwam-h3-post-refiner-v2` 升为 `fastwam-h3-post-refiner-v3`；
- strict loader 自动拒绝旧 manifest/cache；
- 已停止旧 cache 的全量 audit，因为继续审计已知失效 payload 没有价值。

cache schema version 和文件 suffix 仍保持 schema-4：

```text
*.h3-post-refiner-v4.pt
```

变化的是实现签名，而不是 tensor shape/schema layout。

## 14. 8 卡 post-Refiner cache 重建

启动时间：

```text
2026-08-23 00:18 UTC+8
```

命令分两阶段串行：

### 阶段 A：已有 schema-3 Qwen cache

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/convert_h3_post_refiner_cache.py \
  --source-dir data/h3_condition_cache/libero_2cam224 \
  --output-dir data/h3_condition_cache/libero_2cam224_post_refiner_v4 \
  --transformer-dir /root/wuqingman/models/MiniMax-H3/FL2VA/transformer \
  --batch-size 128 \
  --overwrite
```

`--overwrite` 是必要的，因为旧 cache payload 数值已失效。

阶段 A 每 rank 分配约 `11,733` 个 schema-3 文件，总计约 `93,865`。

本文写入时进度：

```text
每 rank 已写入至少 7,680
8 ranks 均正常推进
skipped=0
```

### 阶段 B：没有 schema-3 Qwen cache 的样本

阶段 A 完成后自动执行：

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/precompute_h3_post_refiner_complete.py \
  task=libero_h3_uncond_2cam224_1e-4 \
  +source_qwen_cache_dir=data/h3_condition_cache/libero_2cam224 \
  +post_refiner_cache_dir=data/h3_condition_cache/libero_2cam224_post_refiner_v4 \
  +cache_batch_size=8
```

该阶段对剩余样本运行完整 Qwen + condition projection + TokenRefiner。

此前目标 dataset sample count：

```text
277,713
```

最终 completion 不能只按路径存在判断，必须由新 manifest 下的 strict load 和全量 audit 确认。

## 15. Test 状态

RMSNorm 修复后运行：

```bash
pytest -q tests/models/minimax_h3 tests/test_trainer_h3_eval.py
```

结果：

```text
108 passed
1 warning
```

warning 仅为 `pynvml` deprecation。

较早的 cache/VAE/scheduler focused regression：

```text
17 passed
```

官方 parity 独立结果：

- keyframe：PASS，逐元素零差异；
- prefix posterior：PASS，最大差 `3.57627869e-07`；
- timestep/AdaLN：PASS；
- scheduler trajectory：PASS；
- synthetic single block：PASS，逐元素零差异；
- real 50 blocks：PASS，逐元素零差异；
- real final velocity：PASS，逐元素零差异；
- real TokenRefiner：PASS，逐元素零差异。

## 16. 尚未关闭的 gate

### 16.1 Direct Qwen+Refiner vs schema-4 payload

需要在新 cache 重建后：

1. 固定真实 dataset samples；
2. 直接运行完整 Qwen + condition projection + TokenRefiner；
3. 严格加载对应 schema-4；
4. 比较 `[L, 5376]` BF16 payload；
5. 要求逐元素一致，或记录任何非零差异。

TokenRefiner 本体官方 parity 已通过，但 cache serialization/data-key 路径仍需要独立 gate。

### 16.2 严格全量 audit

旧 audit 被主动停止，因为它审计的是已知失效的 RMSNorm-v2 payload。

新 cache 完成后必须重新运行：

```text
expected_sample_count=277713
verify_dataset_references=true
allow_partial=false
8 ranks
```

需要检查：

- manifest；
- 全部 payload strict load；
- dtype；
- width；
- checksum；
- finite；
- source/key；
- dataset references；
- missing；
- orphan；
- duplicate。

### 16.3 B=2 cache-only optimizer step

要求：

- `load_text_encoder=false`；
- `load_vae=false`；
- 两个真实样本；
- schema-4 post-Refiner condition；
- cached keyframe latent；
- cached full-video moments；
- action/state 仍在线通过 trainable encoder；
- forward；
- video/action loss；
- backward；
- optimizer step；
- 验证 H3 LoRA 和 Action Expert 梯度；
- 验证没有读取旧 schema cache。

该 gate 依赖 VAE cache artifact。目前全量 VAE cache 仍暂停，因此需先完成 VAE cache 的正式样本 smoke 或明确构造小规模严格 artifact。

### 16.4 Commit/push

当前仍是未提交工作区。

遵守用户要求：

- 未经明确要求不 commit；
- 未经明确要求不 push。

## 17. 对 reviewer 的重点问题

请 reviewer 优先核对以下内容：

1. 将 block/refiner RMSNorm 改为 native BF16 kernel，是否准确对应 pinned Diffusers mixed-dtype contract？
2. legacy Torch 2.3 fallback 是否应保留，还是应直接提高项目最低 Torch 版本并删除 fallback？
3. `step_h3_video()` 是否逐项复现官方 `x_t/x0` FP32 blend，且只用于 H3 video、没有错误影响 Action/Wan 路径？
4. prefix posterior parity 使用“相同 retained noise”而不是“相同 seed”是否是正确 gate 定义？
5. schema version 仍为 4、只升级 implementation signature 到 v3，是否足够，还是应升级 cache schema/file suffix？
6. Refiner implementation signature 是否应进一步绑定 PyTorch major/minor、CUDA 或 attention backend？
7. 真实 TokenRefiner parity 使用 native SDPA 且短序列逐元素一致，是否足以批准 cache 重建？
8. 真实 50-block parity 使用 synthetic packed hidden、真实 weights、真实 BF16 block stack，是否足以关闭 Transformer 数值 gate？
9. 是否还要求从官方完整 model `forward()` 到 FastWAM `forward_joint()` 的 end-to-end video-only parity，而不仅是逐模块和 50-block stream parity？
10. 在 direct Qwen+Refiner cache parity、strict full audit、B=2 cache-only step 完成前，是否同意继续保持 100k 禁止启动？

## 18. 当前启动建议

仍然不要启动正式 100k。

推荐顺序：

1. 等待 8 卡 corrected post-Refiner cache 重建完成；
2. direct Qwen+Refiner vs schema-4 parity；
3. strict full schema-4 audit；
4. 完成 VAE cache 正式 smoke；
5. B=2 cache-only optimizer step；
6. 更新本文最终 artifact 路径和 checksum；
7. 用户明确要求后 commit/push；
8. 外部 AI 对明确 commit 做逐行审核；
9. 通过后运行 1k canary；
10. 完整 state resume 到 5k；
11. canary 稳定后再考虑 100k。

## 19. 最终结论

reviewer 提出的“必须做 full 50-block parity”是正确的，并实际找到了会在 50 层累积的 RMSNorm 数值问题。

修复后，以下核心数值路径已经达到 pinned 官方实现 parity：

- keyframe helper；
- timestep/AdaLN；
- TokenRefiner；
- single block；
- 50-block stack；
- final velocity；
- scheduler trajectory；
- prefix posterior moments/sample algebra。

但永久 cache 的最终 gate 尚未完成。旧 post-Refiner payload 已失效，新 cache 正在 8 卡重建。正式训练仍应保持暂停，直到：

```text
corrected cache complete
+ direct cache parity
+ strict full audit
+ B=2 cache-only optimizer step
+ review commit approved
```

## 20. 2026-08-23 单样本 cache-only 全流程 smoke

按用户要求，8 卡 post-Refiner 全量重建已暂停；该任务可依靠 strict-skip
恢复。随后对 dataset index 0 生成一个独立 VAE cache：

```text
artifacts/h3_one_sample_vae_cache
written=1
skipped=0
exit_code=0
```

训练读取 corrected schema-4 post-Refiner condition cache，并读取上述 VAE
cache；`load_text_encoder=false`、`load_vae=false`，因此训练设备上没有加载
Qwen 或 VAE。新增可复现入口：

```text
scripts/smoke_h3_one_sample_cache_training.py
```

第一次正式 ZeRO-2 启动暴露出 cache-only 生命周期 bug：
`FastWAMH3.train()` 在 `self.vae is None` 时仍无条件执行
`self.vae.eval()`。现已改为仅在 VAE 已加载时设置 eval mode，并增加
`test_cache_only_h3_train_mode_allows_absent_frozen_encoders` 回归测试。
相关 focused tests 结果为 `15 passed`。

最终使用 8 卡 Accelerate ZeRO-2 对同一个 cached sample 完成一个真实
optimizer step，结果：

```text
condition_shape=[140, 5376]
keyframe_shape=[24, 1, 14, 28]
posterior_shape=[24, 2, 14, 28]
loss=1.7608
loss_action=1.4818
loss_video=0.2791
peak_allocated=76.47 GiB/GPU
peak_reserved=81.56 GiB/GPU
H3_ONE_SAMPLE_CACHE_TRAINING_SMOKE=PASS
exit_code=0
```

该 gate 已覆盖 strict cache load、posterior sampling、joint video/action
forward、loss、backward、ZeRO-2 gradient synchronization、gradient
clipping、optimizer step 和 scheduler step。它证明单样本 cache-only
训练链路可执行，但不替代“两个不同真实样本、每 rank B=2”的最终 memory
gate，也不替代 corrected schema-4 全量 strict audit。

