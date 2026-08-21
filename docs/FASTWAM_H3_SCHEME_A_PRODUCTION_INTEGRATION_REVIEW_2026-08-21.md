# FastWAM-H3 Scheme A Production Integration Review

日期：2026-08-21  
用途：交给另一名 AI / 工程师独立审查本轮生产收口修改  
仓库：`Qingman-Wu/fastwam-minimax-h3`  
分支：`codex/fastwam-h3-scheme-a-2026-08-19`  
上一版远端提交：`f8d4dd0c012101deab2a77150b378f35c252fd4b`

## 1. 本轮目标

上一轮真实验证确认主体架构已经成立，但留下以下生产 blocker：

1. 5-frame H3 rollout 的 2 个 prefix latent 被重复到 7 latent 后 decode，运行上可行，
   但 gold parity 只有约 18 dB，语义错误。
2. Action loss 对早中层 H3 LoRA 的梯度远大于 video loss；FastWAM-like joint
   optimization 保留该路径，但长训练必须监控 loss 和 gradient balance。
3. LoRA checkpoint 按 list 顺序保存，缺少 module path、alpha、dropout 和 base
   checkpoint fingerprint。
4. Qwen cache 只绑定实现版本，没有绑定真实 Qwen 权重和 processor。
5. ActionDiT 初始化 artifact 没有 checksum / source manifest。
6. `action_fps` 容易被误解为真实数据采样频率，缺少 layout-level contract。
7. 还没有真实 8×H20 save → new process load → resume step 证据。
8. 正式训练参数尚未做真实吞吐和显存比较。

本轮已逐项处理，并启动了全量 cache → 大规模训练流水线。

## 2. 5-frame pixel decode 隔离

涉及文件：

- `src/fastwam/models/minimax_h3/video_vae.py`
- `src/fastwam/models/minimax_h3/fastwam.py`
- `src/fastwam/runtime.py`
- `tests/models/minimax_h3/test_conditioning.py`
- `tests/models/minimax_h3/test_inference_contract.py`
- `tests/models/minimax_h3/test_runtime_config.py`

### 2.1 删除错误 workaround

删除了：

```text
2 temporal latents
-> repeat last latent
-> pad to 7 latents
-> decode 5 pixels frames
```

`MiniMaxH3VAEAdapter.decode()` 遇到 `frame_num=5` 且输入是 2 temporal latents 时，
现在明确抛出 `NotImplementedError`。

### 2.2 推理接口

`FastWAMH3.infer()` 新增：

```text
decode_video: bool = False
```

默认输出：

```text
video_latents
action
```

含义：

- 5-frame full-video latent 仍然参与每一步 joint denoising。
- Action stream 仍然读取 H3 的 world representation。
- 默认不执行像素 decoder。
- `infer_action()` 强制不 decode。
- 显式请求 5-frame pixel decode 会在采样前拒绝。
- 22/39 等原生 decoder window 仍可显式 `decode_video=true`。

CLI 现在默认保存：

```text
*.action.pt
*.video_latents.pt
```

只有 `infer_out` 实际包含 `video` 时才写 MP4。

## 3. 默认采用 FastWAM-like joint gradient

涉及文件：

- `configs/model/fastwam_h3.yaml`
- `src/fastwam/runtime.py`
- `src/fastwam/models/minimax_h3/fastwam.py`
- `src/fastwam/models/minimax_h3/video_dit.py`
- `src/fastwam/models/minimax_h3/mixed_attention.py`
- `tests/models/minimax_h3/test_mixed_attention.py`

新增配置：

```yaml
stop_action_gradient_to_h3: false
```

默认实现：

```text
H3 self-attention:
  使用原始 H3 K/V

Action asymmetric attention:
  使用原始 H3 K/V，不 detach
```

因此 forward coupling 不变：

```text
H3 -> Action
```

baseline backward contract 是：

```text
L_video  -> H3 LoRA
L_action -> Action Expert
L_action -> H3 LoRA
H3 33B base weights remain frozen
```

这与原始 FastWAM 允许 Action loss 更新 Video Expert 的机制一致。保留
`stop_action_gradient_to_h3=true` 作为稳定性 ablation，而不是默认 baseline。

### 3.1 真实 33B 梯度验证

环境：

- 真实 MiniMax-H3 FL2VA 33B
- 真实 LIBERO sample
- BF16
- 50 层 H3
- rank-32 H3 attention LoRA

结果：

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

all observed gradients finite = true
```

以上 action-only 零梯度结果验证了可选 stop-gradient 路径。需要注意：zero tensor
不等于 AdamW 绝对不更新，decoupled weight decay 仍可能改变进入 optimizer 的非零
LoRA 参数；若要求 H3 绝对冻结，应将对应参数移出 optimizer。

## 4. Checkpoint schema 3

涉及文件：

- `src/fastwam/models/minimax_h3/fastwam.py`
- `src/fastwam/models/minimax_h3/video_dit.py`
- `tests/models/minimax_h3/test_checkpoint_contract.py`

### 4.1 Payload

weights checkpoint 现在包含：

```text
schema_version = 3
backbone identity
global step
Action Expert config
Action Expert state_dict
H3 LoRA rank
H3 LoRA alpha
H3 LoRA dropout
完整 LoRA target module names
base H3 config/index fingerprint
按完整 module path 保存的 LoRA state_dict
```

LoRA key 形如：

```text
blocks.0.attn.qkv_proj
blocks.0.attn.out_proj
...
blocks.49.attn.qkv_proj
blocks.49.attn.out_proj
```

### 4.2 Strict load

load 时严格比较：

- schema version
- Action Expert geometry/config
- rank / alpha / dropout
- base H3 fingerprint
- 完整 target name 集合
- state_dict keys 和 shape

schema 2 会明确拒绝，不能再静默恢复为 zero-init LoRA。

单元测试覆盖：

- schema-3 named-path round trip
- alpha mismatch 拒绝
- schema-2 拒绝

## 5. ActionDiT artifact manifest

涉及文件：

- `scripts/preprocess_h3_action_dit_backbone.py`
- `src/fastwam/models/minimax_h3/action_dit.py`
- `tests/models/minimax_h3/test_action_expert.py`

生成 ActionDiT backbone 时自动写相邻 manifest：

```text
H3ActionDiT_video_interp_1024hdim.pt.manifest.json
```

记录：

- artifact filename
- exact byte size
- artifact SHA256
- source H3 config/index SHA256
- Action Expert config
- generation command
- output dtype
- alpha scaling policy
- copied/interpolated tensor 数量

真实 artifact：

```text
size   = 4,827,088,347 bytes
SHA256 = 6b0a3de516f67bc2d1c1e92712ead856c68b4cdaa067f78667dbacbc357230e6
```

`H3ActionDiT.from_pretrained()` 在读 4.7 GiB payload 前强制检查：

- manifest 存在
- schema 正确
- filename 正确
- byte size 正确
- SHA256 正确

## 6. Qwen condition cache schema 3

涉及文件：

- `src/fastwam/models/minimax_h3/text_encoder.py`
- `src/fastwam/datasets/h3_condition_cache.py`
- `scripts/precompute_h3_conditions.py`
- `tests/models/minimax_h3/test_condition_cache.py`

cache directory 现在必须有：

```text
h3-qwen-cache-manifest.json
```

manifest 绑定：

- encoder implementation signature
- hidden layer 50 prenorm contract
- Qwen checkpoint config/index fingerprint
- processor/tokenizer/chat-template fingerprint
- cache schema version

每个 sample 的 filename digest 和 payload 同时绑定 Qwen checkpoint manifest/index 与
processor fingerprint。它能拒绝不同 manifest/index 或 processor，但当前并未逐字节
hash `.safetensors` shards；固定官方 release 目录是本轮训练的额外前提。

sample suffix 从：

```text
*.h3-qwen-prenorm-layer50-v2.pt
```

升级为：

```text
*.h3-qwen-prenorm-layer50-v3.pt
```

### 6.1 Precompute 能力

precompute 脚本新增：

- multi-sample Qwen encode
- batch 内 path 去重
- deterministic sampler-seed subset
- overwrite-safe manifest 初始化
- 多 GPU 分片

真实 benchmark 表明 batch=8 没有带来吞吐收益，因此正式全量 cache 使用 batch=1。

## 7. Dataset cache 可复现性修复

涉及文件：

- `src/fastwam/datasets/lerobot/robot_video_dataset.py`

真实多卡训练发现：

```text
同一个初始 dataset index
-> padded sample
-> np.random 选择不同 replacement
-> 得到不同 f0
-> offline cache miss
```

现在：

- retry RNG 由初始 dataset index 固定 seed。
- precompute 与 train 对同一个 index 选择相同 unpadded replacement。
- cache `FileNotFoundError` 不再被 `__getitem__` 捕获后随机换样本。
- 其他 sample 处理异常的 replacement 也由原始 index 确定。

这样 cache 缺失会 fail fast，而不是被随机采样掩盖。

真实训练窗口数为：

```text
277,713
```

此前日志中的 1,712 是 normalization episode 数，不是 cache sample 数。

## 8. Action MM-RoPE clock contract

涉及文件：

- `configs/model/fastwam_h3.yaml`
- `src/fastwam/runtime.py`
- `src/fastwam/models/minimax_h3/fastwam.py`
- `tests/models/minimax_h3/test_inference_contract.py`

effective Action RoPE fps 由实际 layout 计算：

```text
video_fps * action_horizon / (num_frames - 1)
```

基础 model config 的 `action_fps` 改为 `null`，运行时自动推导。

如果 task 显式配置 `action_fps`，它只作为 assertion；不匹配实际 frame/action layout
会立即报错。

当前 LIBERO：

```text
24 video fps * 32 actions / 4 visual intervals = 192 Hz
```

因此 task override `action_fps: 192.0` 会被严格验证。

## 9. 真实 8×H20 1-step

配置：

```text
8x H20 96 GiB
ZeRO-2
BF16
B=1 / GPU
gradient accumulation=16
global batch=128
50-layer H3 + Action Expert
rank-32 H3 LoRA
```

结果：

```text
optimizer step 1 success
weights checkpoint written
full ZeRO-2 state written
```

每卡初始化：

```text
allocated = 72.4 GiB
reserved  = 74.94 GiB
```

产物：

```text
schema-3 weights = 4.7 GiB
full state       = 166 GiB
```

## 10. 新进程 resume 验证

在全新 8-rank 进程中加载 step 1：

```text
all model weights loaded successfully
all optimizer states loaded successfully
all scheduler states loaded successfully
all random states loaded successfully
global_step=1
epoch=0
batch_in_epoch=16
sample_offset=128
```

随后成功完成：

```text
max_steps reached step=2
exit code 0
```

因此真实链路：

```text
save
-> terminate
-> new distributed process
-> load model/optimizer/scheduler/random/dataloader progress
-> next optimizer step
```

已通过。

## 11. FastWAM-like joint-gradient 最快稳定 setting

8×H20 sustained comparison：

```text
B=1, no gradient checkpoint:
  1.75 processed samples/s (10 steps)

B=1, gradient checkpoint:
  1.50 processed samples/s (3 steps)

B=2, no gradient checkpoint:
  2.38 processed samples/s (10 steps)
  memory = 95.4--96.2 GiB / 97.9 GiB

B=2, no gradient checkpoint, accumulation=8:
  2.37 processed samples/s (1 full optimizer step / 8 microsteps)
```

前三组 benchmark 显式使用 `gradient_accumulation_steps=1`；最后一组使用正式
accumulation=8。Trainer 的通用公式现已正确计入 accumulation。

选择：

```yaml
batch_size: 2
model:
  mot_checkpoint_mixed_attn: false
gradient_accumulation_steps: 8
```

理由：

- no-GC 比 checkpointing 快。
- B=2 比 B=1 no-GC 快约 36%，且 10 steps 与正式 accumulation=8 都通过。
- B=2 仅剩约 1.7--2.4 GiB 显存，长训练必须监控 OOM；若出现显存碎片或异常峰值，
  应回退到 B=1/accumulation=16。

训练预算保持 `max_steps=100000`。`EXPERIMENT_37.md` 已记录用户明确确认“总计
100k optimizer steps，每 10k 保存一次”；因此约 46.1 次数据遍历是有意实验预算，
不是把 `num_epochs=10` 误当成生效上限。长训练需要据验证指标监控后期 overfitting。

## 12. Checkpoint 磁盘控制

涉及文件：

- `src/fastwam/trainer.py`
- `configs/task/libero_h3_uncond_2cam224_1e-4.yaml`
- `tests/test_trainer_checkpoint_retention.py`

完整 ZeRO-2 state 约 166 GiB。长实验若永久保留每个 checkpoint，会耗尽
本地磁盘。

新增：

```yaml
max_checkpoints: 2
```

每次成功保存后：

- 只保留最新两个 full state directories。
- 只保留最新两个 schema-3 weights files。
- 删除发生在所有 rank 完成保存之后。

当前磁盘预算：

```text
available                ~= 2.5 TiB
estimated full Qwen cache ~= 0.9 TiB
two full states           ~= 0.33 TiB
```

保留策略满足预算。

## 12.1 Review follow-up engineering fixes

- Scheme A internal eval 显式 `decode_video=false`，只报告 val loss 以及 normalized /
  denormalized action L1/L2；5-frame 路径不再访问不存在的 `pred["video"]`，也不调用
  unsupported VAE pixel decode。原生可解码 window 仍保留 pixel PSNR/SSIM/MP4。
- `samples/s` 现在乘入 `gradient_accumulation_steps`。
- final step 若已命中周期 checkpoint，不再重复写同一份 full state。
- dataset fallback 记录 replacement count/rate 与 exception histogram；每个 worker
  warmup 1000 次后若 replacement rate 超过 0.1%，直接终止而不是继续掩盖系统性错误。

## 13. 自动测试

最终命令：

```bash
PYTHONPATH="/root/wuqingman/.deps-transformers-4.57.6:src" \
python3 -m pytest tests -q
```

结果：

```text
88 passed, 1 warning
```

warning 仅为环境中的 `pynvml` deprecation。

附加检查：

```text
python compileall = pass
git diff --check = pass
IDE diagnostics = no new code errors
```

`trainer.py` 的 `wandb` / `swanlab` basedpyright warning 是已有 optional dependency
静态解析 warning；真实 venv 中 `swanlab` import 已验证成功。

## 14. 正式流水线状态

原 stop-gradient 串联任务已停止，避免 cache 完成后按过期配置启动训练。新流水线将在
joint-gradient 真实 8×H20 smoke test 通过后启动：

```text
8-GPU full Qwen schema-3 cache
-> FULL_CACHE_DONE sentinel
-> 8xH20 large training
```

大训练 run id：

```text
scheme-a-large-jointgrad-b2-nogc-20260821
```

正式 cache 使用：

```text
world_size=8
cache_batch_size=1
overwrite=false
```

cache 使用 `overwrite=false` 恢复，因此停止旧进程不会丢失已完成样本。新任务中若
cache 进程失败，shell 的 `&&` 会阻止训练启动，避免在不完整 cache 上训练。

## 15. 建议 reviewer 重点检查

1. joint-gradient baseline 中 `L_action -> H3 LoRA` 是否在长训练保持稳定。
2. schema-3 base fingerprint 是否需要进一步升级为完整 shard checksum。
3. ActionDiT 每次 load 都计算 4.7 GiB SHA256 的启动成本是否可接受。
4. `video_latents` 的 batch dimension contract 是否应统一保留。
5. deterministic padding replacement 是否会引入可接受的数据采样偏差。
6. full cache 接近 0.9 TiB 是否需要后续改为 shard/container 格式以降低 inode 压力。
7. 100k optimizer steps（约 46.1 次数据遍历）的后期 overfitting 与 checkpoint 指标。

## 16. 当前结论

已确认：

- 5-frame 错误 pixel decode 不再出现在默认生产路径。
- baseline 恢复原始 FastWAM-like joint gradient；H3 base 冻结，仅 LoRA 接收
  `L_video + L_action`。
- named LoRA schema-3 checkpoint 可严格恢复。
- ActionDiT 绑定真实字节 SHA256；Qwen cache 绑定 checkpoint manifest/index 与
  processor fingerprint。
- 192 Hz 是显式且可验证的等效 RoPE clock。
- 真实 8×H20 optimizer step 成功。
- 真实新进程 full-state resume step 成功。
- 最快稳定设置已经通过对照 benchmark。
- checkpoint 磁盘增长已受控。

仍在进行：

- joint-gradient 真实 8×H20 smoke test。
- 277,713-window 全量 Qwen cache（从已有样本恢复）。
- cache 完成后自动启动的大规模训练。
