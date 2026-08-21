# FastWAM-H3 Scheme A Joint-Gradient Handoff

Date: 2026-08-21  
Base reviewed commit: `7ce2815d8ef404a7bed05ae0208184f5c515acee`  
Branch: `codex/fastwam-h3-scheme-a-2026-08-19`

## 1. Executive summary

This change turns the Scheme A baseline back into a FastWAM-like optimization
scheme:

```text
Forward:
  H3 -> Action Expert
  Action -/-> H3

Backward:
  L_video  -> H3 attention LoRA
  L_action -> Action Expert
  L_action -> H3 attention LoRA
```

The 33B H3 base model remains frozen. Only rank-32 LoRA branches on each H3
attention `qkv_proj` and `out_proj` are trainable. With 50 H3 blocks, this is
100 named LoRA branches. The Action Expert remains fully trainable.

The production default is now:

```yaml
freeze_video_expert: true
h3_lora_rank: 32
h3_lora_alpha: 32.0
h3_lora_dropout: 0.0
stop_action_gradient_to_h3: false
```

The optional stop-gradient implementation is retained for ablations, but it is
no longer the baseline.

All review follow-up engineering fixes were also completed:

1. H3 five-frame internal evaluation no longer accesses a nonexistent pixel
   video or invokes the unsupported two-latent-to-five-frame decoder.
2. Throughput logging now includes gradient accumulation.
3. The final optimizer step no longer writes the same 166 GiB ZeRO-2 state
   twice when it also hits `save_every`.
4. Dataset fallback now records replacement count/rate and exception types and
   fails if replacement becomes systematic.
5. Fingerprint documentation now distinguishes manifest/index fingerprints
   from actual shard-byte SHA256.
6. The 100k optimizer-step budget is explicitly documented as intentional.

Final automated result:

```text
88 passed, 1 environment warning
python compileall: pass
git diff --check: pass
```

The warning is the existing `pynvml` deprecation warning.

## 2. Why joint gradient is the correct baseline

The original FastWAM mixed attention concatenates Video K/V and Action K/V
without detaching Video K/V. Action queries can read Video K/V, so the action
loss naturally backpropagates through the Video Expert.

Scheme A previously added:

```python
action_h3_k = h3_k.detach()
action_h3_v = h3_v.detach()
```

That implementation was correct as an isolation mechanism, but it was an
additional design choice relative to original FastWAM.

For a baseline whose goal is “replace Wan Video Expert with H3 while preserving
the FastWAM optimization pattern,” the appropriate setting is:

```yaml
stop_action_gradient_to_h3: false
```

This does not introduce Action-to-H3 forward conditioning. The asymmetric mask
still prevents H3 queries from reading Action tokens. It only restores the
backward path from `L_action` through the H3 K/V used by Action attention.

## 3. Exact trainable parameter contract

H3:

```text
33B base parameters: frozen
50 blocks x qkv_proj LoRA: trainable
50 blocks x out_proj LoRA: trainable
total named H3 LoRA branches: 100
```

Action Expert:

```text
H3ActionDiT parameters: fully trainable
```

Gradient ownership:

```text
L_video:
  updates H3 LoRA
  does not update Action Expert

L_action:
  updates Action Expert
  updates H3 LoRA

Neither loss:
  updates frozen H3 base weights
```

The low-level attention tests cover both modes:

- `detach_h3_for_action=true`: Action still reads H3 values, but Action loss
  produces no H3 gradient.
- default `detach_h3_for_action=false`: Action loss produces a nonzero H3
  gradient.

## 4. Known gradient-balance risk

The earlier real 33B diagnostic found that early H3 LoRA layers can receive a
much larger gradient from action loss than from video loss. The most extreme
observed block-0 ratio was approximately:

```text
|grad from L_action| / |grad from L_video| ~= 2010
```

This evidence is not being ignored. The current decision is to preserve the
original FastWAM-like path and monitor it rather than delete it preemptively.

The production run should monitor:

- `loss_video`
- `loss_action`
- total gradient norm
- non-finite gradients
- evidence that action loss falls while video loss degrades
- checkpoint-level downstream LIBERO metrics

If joint optimization is unstable, the preferred next ablation is gradient
scaling rather than an immediate architectural rewrite:

```text
grad_H3 = grad(L_video) + beta * grad(L_action)

beta = 1.0   current FastWAM-like baseline
beta = 0.1   candidate
beta = 0.01  candidate
beta = 0.0   stop-gradient ablation
```

No gradient scaling is implemented in this commit.

## 5. H3 latent-only evaluation fix

The new H3 inference contract defaults to:

```python
decode_video = False
```

For the five-frame Scheme A window, inference returns:

```text
video_latents
action
```

It does not return `video`, because H3 VAE cannot natively decode the
five-frame/two-latent representation. The old Trainer evaluator still assumed:

```python
pred_video = pred["video"]
```

and later attempted VAE pixel reconstruction. That would fail as soon as
`eval_every > 0`.

The evaluator now:

1. Detects the H3 latent/action inference contract.
2. Calls inference with `decode_video=false`.
3. Reports validation loss.
4. Reports normalized action L1/L2 over valid, non-padding entries.
5. Reports denormalized action L1/L2.
6. Omits pixel PSNR, SSIM, VAE reconstruction, and MP4 for the unsupported
   five-frame path.
7. Preserves pixel metrics and MP4 generation when a model/window actually
   returns a natively decodable pixel video.

`tests/test_trainer_h3_eval.py` verifies that latent-only H3 evaluation no
longer accesses pixel-only fields.

## 6. Throughput accounting correction

Trainer `global_step` counts optimizer steps, not microsteps. The correct
processed-sample throughput is:

```python
samples_per_sec = (
    optimizer_steps_per_sec
    * batch_size
    * world_size
    * gradient_accumulation_steps
)
```

The logger and experiment tracking payload now use this formula.

Important correction to the review discussion: the earlier B=1/B=2 benchmark
commands explicitly set:

```text
gradient_accumulation_steps=1
```

Therefore their original absolute sample-rate logs were not missing a factor of
16. In the formal production setting, accumulation increases samples per
optimizer step while optimizer steps per second decrease correspondingly.

## 7. New joint-gradient speed benchmarks

All runs used:

```text
8 x H20
DeepSpeed ZeRO-2
real H3 33B
real Action Expert
real LIBERO samples
stop_action_gradient_to_h3=false
```

### 7.1 B=1, no gradient checkpointing

Run:

```text
scheme-a-jointgrad-b1-no-gc-10step
```

At step 10:

```text
1.75 processed samples/s
```

### 7.2 B=1, gradient checkpointing

Run:

```text
scheme-a-jointgrad-gc-3step
```

At step 3:

```text
1.50 processed samples/s
```

No gradient checkpointing remains faster.

### 7.3 B=2, no gradient checkpointing

Run:

```text
scheme-a-jointgrad-b2-no-gc-10step
```

At step 10:

```text
2.38 processed samples/s
```

This is approximately 36% faster than B=1/no-GC.

The previously measured memory range is:

```text
95.4--96.2 GiB used / 97.9 GiB
```

### 7.4 Final production accumulation smoke

Final production-equivalent setting:

```yaml
batch_size: 2
gradient_accumulation_steps: 8
model:
  mot_checkpoint_mixed_attn: false
```

Run:

```text
scheme-a-jointgrad-production-setting-smoke
```

One complete optimizer step, containing eight accumulated microsteps, completed
successfully:

```text
loss        = 2.0086
loss_action = 1.7363
loss_video  = 0.2723
throughput  = 2.37 processed samples/s
```

The global effective batch remains:

```text
2 samples/GPU x 8 GPUs x accumulation 8 = 128
```

The selected fastest verified setting is therefore B=2/no-GC/accumulation=8.

Risk: B=2 has only approximately 1.7--2.4 GiB of memory headroom. It passed the
10-step benchmark and the full accumulated optimizer-step smoke test. If a long
run encounters OOM due to fragmentation or an unexpected allocation peak, the
safe fallback is:

```yaml
batch_size: 1
gradient_accumulation_steps: 16
```

which preserves global batch 128.

## 8. Checkpoint duplicate-save fix

Before this change, step 100000 could execute both:

```text
periodic save because 100000 % 10000 == 0
final save because global_step >= max_steps
```

Both writes targeted the same checkpoint, causing a redundant write of an
approximately 166 GiB full ZeRO-2 state.

The loop now records whether the current step was already saved and skips the
final save in that case.

Checkpoint retention remains:

```yaml
max_checkpoints: 2
```

## 9. Dataset replacement telemetry

Cache misses remain fatal:

```python
except FileNotFoundError:
    raise
```

Other sample failures can still use deterministic replacement, but each dataset
worker now records:

```text
sample_attempt_count
replacement_count
replacement rate
exception-type histogram
```

Defaults:

```text
warmup attempts: 1000
maximum replacement rate: 0.1%
```

After warmup, exceeding 0.1% raises a `RuntimeError` with the counts and
exception histogram. This prevents a systematic shape/data bug from being
silently converted into repeated replacement samples.

The counters are worker-local, which is conservative enough to detect a
systematic worker-visible failure but is not a globally aggregated telemetry
system.

## 10. Fingerprint wording correction

ActionDiT initialization is bound to the actual 4.7 GiB artifact bytes through
SHA256.

Qwen and H3 base fingerprints currently hash checkpoint config/index manifests,
plus processor/tokenizer/chat-template files for Qwen. They do not hash every
`.safetensors` shard byte.

The exact current guarantee is:

```text
different checkpoint manifest/index or processor cannot reuse the old cache
```

It is not yet:

```text
arbitrarily replaced shard bytes with identical index are always detected
```

The present run assumes the official MiniMax-H3 release directory is fixed and
immutable. Paper-level artifact integrity should later add one precomputed
SHA256 manifest covering all shard files.

## 11. Training budget decision

Dataset windows:

```text
277,713
```

Global effective batch:

```text
128
```

Approximate optimizer steps per data traversal:

```text
277,713 / 128 ~= 2,170
```

The configured budget remains:

```yaml
num_epochs: 10
max_steps: 100000
```

`max_steps` intentionally overrides `num_epochs`, so this is approximately:

```text
100,000 / 2,170 ~= 46.1 data traversals
```

This is not an unresolved choice. `EXPERIMENT_37.md` records that the user
explicitly confirmed a total budget of 100k optimizer steps with a checkpoint
every 10k steps. Long-run validation should still monitor late overfitting.

## 12. Automated and real-system verification

Automated:

```text
88 passed, 1 warning
python compileall: pass
git diff --check: pass
```

Real 8-GPU evidence added in this change:

```text
joint-gradient B=1/no-GC 10 steps: pass
joint-gradient B=1/GC 3 steps: pass
joint-gradient B=2/no-GC 10 steps: pass
joint-gradient B=2/accumulation=8 optimizer step: pass
all observed losses finite
```

Previously established evidence still applies:

```text
real H3 33B load/forward/backward
real 2.4B Action Expert
DeepSpeed ZeRO-2 optimizer step
schema-3 named LoRA checkpoint
full state save
new 8-rank process restore
optimizer/scheduler/random/dataloader restore
continue to next optimizer step
```

## 13. Current cache and large-run state

The obsolete stop-gradient cache-to-training shell chain was stopped before it
could launch training with the old baseline.

Full Qwen schema-3 cache generation resumed with:

```text
world size: 8
cache batch size: 1
overwrite: false
```

At 2026-08-21 14:36 UTC+8:

```text
cache process: running
cache directory size: approximately 44 GiB
GPU memory: approximately 50.5 GiB on each of 8 GPUs
```

`overwrite=false` preserves completed entries and resumes missing cache entries.

The intended formal training run is:

```text
scheme-a-large-jointgrad-b2-nogc-20260821
```

It must start only after the full cache process exits successfully. The formal
configuration is:

```yaml
max_steps: 100000
save_every: 10000
max_checkpoints: 2
batch_size: 2
gradient_accumulation_steps: 8
eval_every: 0
model:
  stop_action_gradient_to_h3: false
  mot_checkpoint_mixed_attn: false
```

## 14. Files changed in this handoff

Configuration:

- `configs/model/fastwam_h3.yaml`
- `configs/task/libero_h3_uncond_2cam224_1e-4.yaml`

Model/runtime:

- `src/fastwam/models/minimax_h3/fastwam.py`
- `src/fastwam/runtime.py`

Trainer/data:

- `src/fastwam/trainer.py`
- `src/fastwam/datasets/padding.py`
- `src/fastwam/datasets/lerobot/robot_video_dataset.py`

Tests:

- `tests/models/minimax_h3/test_runtime_config.py`
- `tests/models/minimax_h3/test_mixed_attention.py`
- `tests/models/minimax_h3/test_dataset_padding.py`
- `tests/test_trainer_h3_eval.py`

Updated prior handoffs:

- `docs/FASTWAM_H3_SCHEME_A_PRODUCTION_INTEGRATION_REVIEW_2026-08-21.md`
- `docs/FASTWAM_H3_SCHEME_A_VERIFICATION_HANDOFF_2026-08-20.md`

## 15. Reviewer checklist

1. Confirm that `stop_action_gradient_to_h3=false` is propagated from Hydra
   config through runtime construction into every mixed-attention block.
2. Confirm that H3 still cannot read Action tokens in the forward pass.
3. Confirm that the default action-loss backward path reaches H3 LoRA but not
   frozen H3 base parameters.
4. Review whether the approximately 2010x early-layer gradient ratio requires a
   `beta` scaling ablation before or during the first long run.
5. Review the B=2 memory risk and whether 1.7--2.4 GiB headroom is acceptable
   for a 100k-step run.
6. Confirm that H3 latent-only evaluation never calls a five-frame pixel
   decoder.
7. Confirm the dataset replacement threshold semantics and whether worker-local
   telemetry is sufficient.
8. Confirm that manifest/index fingerprinting is described accurately and is
   acceptable for this run.
9. Confirm that 100k optimizer steps, approximately 46.1 traversals, is the
   intended experiment budget recorded in `EXPERIMENT_37.md`.
