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
7. Optimizer-step loss logging is now the exact sample-weighted mean over every
   microbatch and every rank, rather than only the final microbatch.
8. CUDA peak allocated/reserved memory is logged at each optimizer step.
9. A reusable real-H3 diagnostic now reports per-loss LoRA gradients, beta
   sweeps, robust local LoRA perturbation, and five-frame latent rollout error.
10. A distributed memory-smoke entry point can conservatively pad cached Qwen
    rows to test condition-length headroom.
11. H3 inference now accepts unbatched cached Qwen rows by batching embeddings,
    tags, and masks consistently; the diagnostic exposed the previous mismatch.

Final automated result:

```text
96 passed, 1 environment warning
python compileall: pass
git diff --check: pass
```

The warning is the existing `pynvml` deprecation warning. Running unscoped
`pytest` also discovers two vendored RoboTwin scripts that require optional
`sapien`; the repository-owned command is `pytest tests`.

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

The loss payload was also corrected. Previously, with accumulation 8, one
logged optimizer step contained only the final microbatch loss. Training
gradients were correct, but `train/loss`, `loss_video`, and `loss_action` were
noisier than their labels implied. The trainer now accumulates each scalar
weighted by local batch size, gathers sums and sample counts across ranks, and
logs:

```text
sum(metric * local_microbatch_size over every microbatch and rank)
-----------------------------------------------------------------
sum(local_microbatch_size over every microbatch and rank)
```

This remains exact for a short final batch and is covered by
`tests/test_trainer_accumulation_metrics.py`.

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

The earlier NVML numbers mixed decimal GB and binary GiB. The trainer now
reports PyTorch peak memory directly in binary GiB.

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

The pre-100k rerun used the current code, real cache, and an explicit
`--num_processes 8` launcher override. It completed one full accumulated step:

```text
world size              = 8
cache row range scanned = 125--140
loss                    = 2.0511
loss_action             = 1.7643
loss_video              = 0.2868
peak allocated          = 90.08 GiB
peak reserved           = 92.97 GiB
device capacity         = 95.08 GiB
```

The one-step 3.95 samples/s value includes startup and is not a replacement for
the sustained 2.38 samples/s benchmark.

A deliberately out-of-distribution 192-row condition smoke failed in backward
with only about 10 MiB free. This is useful boundary evidence, but it does not
describe the current dataset: all 39,677 cache files present during the scan
were between 125 and 140 rows, and the observed maximum was 140. Therefore B=2
is accepted for the current immutable cache schema/data distribution, but any
future prompt template, image tokenization, or cache row-length increase must
repeat the memory smoke.

An initial set of memory-smoke commands accidentally inherited
`num_processes: 1` from the Accelerate config. Their single-GPU OOM results were
discarded. Only the explicit world-size-8 run above is used for the production
decision.

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

## 12. Pre-100k joint-gradient diagnostics

### 12.1 Reproducible diagnostic entry point

`scripts/diagnose_h3_joint_gradient.py` loads the real frozen H3 33B base, all
100 rank-32 LoRA branches, the real Action Expert checkpoint, and cached native
Qwen conditions. It fixes model initialization and diffusion-noise seeds before
measuring:

1. `g_video` and `g_action` separately on H3 LoRA parameters in blocks 0, 24,
   and 49;
2. gradient norm ratio and cosine;
3. analytic beta sweeps for 1, 0.1, 0.01, and 0.001;
4. local LoRA-output/base-output RMS after one reversible LoRA-only AdamW probe;
5. full 20-point five-frame latent denoising rollout error.

The production model is restored after the reversible probe. Diagnostic JSON
artifacts are stored under `artifacts/`.

### 12.2 Three-sample gradient result

The deterministic beta=1 probes produced:

```text
sample  block 0 ratio  block 24 ratio  block 49 ratio
0       4155.8x        77.1x           4.57x
1       2856.5x        67.2x           7.80x
2       2759.2x        98.2x           8.11x
```

The corresponding cosines were:

```text
sample  block 0      block 24     block 49
0        0.00027     -0.05546      0.00085
1       -0.01345     -0.01022     -0.00001
2        0.00475      0.06229      0.00500
```

Therefore the action and video gradients are consistently near-orthogonal at
the selected layers. The early-layer action gradient is not just larger; it
points in a largely independent direction.

For sample 0, scaling only the Action-to-H3 path would give:

```text
beta   block 0 scaled ratio  block 24 scaled ratio  block 49 scaled ratio
1      4155.8x               77.11x                 4.566x
0.1     415.6x                7.711x                0.457x
0.01     41.6x                0.771x                0.0457x
0.001     4.16x               0.077x                0.00457x
```

No single global beta balances every depth: beta 0.01 balances the middle block
but still leaves block 0 action-dominated, while beta 0.001 still leaves block
0 at 2.8--4.2x across the three samples and nearly removes Action-to-H3 in late
blocks.

### 12.3 Representation probe and a rejected metric

Comparing hidden states from two separate BF16 H3 forwards initially appeared
to show about 12% relative drift at block 49 after one update. An LR=0 no-op
control showed the same magnitude. This means repeated-forward hidden
difference is dominated by BF16/attention numerical non-determinism and is not
a valid update-drift metric in this environment.

The robust replacement measures each LoRA projection's output and frozen base
projection output inside the same forward. After one beta=1, LR=1e-4 LoRA-only
AdamW probe, all six selected projection ratios were only:

```text
3.04e-5 to 4.31e-5
```

The LR=0 no-op control was exactly zero. Beta 0 and beta 0.001 produced similar
first-step magnitudes because Adam's first update largely normalizes gradient
scale; beta primarily changes direction at this point. This is why raw norm
ratios alone are insufficient to predict immediate representation damage.

### 12.4 Five-frame latent rollout

The full 20-point latent-only denoising result for deterministic sample 0 was:

```text
latent shape = [24, 2, 14, 28]
L1           = 0.30362
MSE/L2       = 0.29163
relative L2  = 0.56162
cosine       = 0.84087
```

This is a pre-training baseline, not a quality target. It verifies that the
complete five-frame latent rollout path runs without invoking the unsupported
pixel decoder and gives a checkpoint-comparable metric.

The first rollout attempt exposed an inference API bug: two-dimensional cached
embeddings were automatically batched, but one-dimensional token tags and
masks were not. `_prepare_text_condition()` now batches all three consistently,
and `test_inference_accepts_unbatched_cached_qwen_rows` covers the contract.

### 12.5 Beta decision

The production configuration remains beta=1 implicitly through:

```yaml
stop_action_gradient_to_h3: false
```

This preserves the explicitly requested original FastWAM optimization
topology. The diagnostics do not justify silently changing the baseline to a
new beta: the gradient imbalance is real, but one-step local LoRA perturbation
is small, and a global beta cannot balance all depths. The appropriate next
step is a short beta=1 training canary with the corrected accumulated loss and
peak-memory logs, followed by checkpoint diagnostics. A beta or layer-wise
scaling mechanism remains a research ablation if the video objective degrades.

## 13. Automated and real-system verification

Automated:

```text
96 passed, 1 warning
python compileall: pass
git diff --check: pass
```

Real 8-GPU evidence added in this change:

```text
joint-gradient B=1/no-GC 10 steps: pass
joint-gradient B=1/GC 3 steps: pass
joint-gradient B=2/no-GC 10 steps: pass
joint-gradient B=2/accumulation=8 optimizer step: pass
current-code real-cache B=2/accumulation=8 optimizer step: pass
synthetic 192-row B=2 boundary smoke: expected OOM
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

## 14. Current cache and large-run state

The obsolete stop-gradient cache-to-training shell chain was stopped before it
could launch training with the old baseline.

Full Qwen schema-3 cache generation uses:

```text
world size: 8
cache batch size: 1
overwrite: false
```

It was intentionally paused while the GPUs ran the diagnostics in this
handoff. At the row-length scan:

```text
completed cache files: 39,677
row length range: 125--140
maximum row length: 140
```

`overwrite=false` preserves completed entries and resumes missing cache entries.
The eight-rank cache process was resumed at 2026-08-21 16:38 UTC+8 after the
diagnostics completed. No formal 100k training process was launched during
these diagnostics.

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

## 15. Files changed in this handoff

Configuration:

- `configs/model/fastwam_h3.yaml`
- `configs/task/libero_h3_uncond_2cam224_1e-4.yaml`

Model/runtime:

- `src/fastwam/models/minimax_h3/fastwam.py`
- `src/fastwam/models/minimax_h3/video_dit.py`
- `src/fastwam/runtime.py`

Trainer/data:

- `src/fastwam/trainer.py`
- `src/fastwam/datasets/h3_condition_cache.py`
- `src/fastwam/datasets/padding.py`
- `src/fastwam/datasets/lerobot/robot_video_dataset.py`

Diagnostic scripts and artifacts:

- `scripts/diagnose_h3_joint_gradient.py`
- `scripts/audit_h3_condition_cache.py`
- `scripts/smoke_h3_b2_memory.py`
- `artifacts/h3_joint_diagnostic_sample0_beta1.json`
- `artifacts/h3_joint_local_probe_sample1_beta1.json`
- `artifacts/h3_joint_local_probe_sample2_beta1.json`
- `artifacts/h3_joint_local_probe_beta0.001.json`
- `artifacts/h3_joint_local_probe_beta0.json`
- `artifacts/h3_joint_local_probe_noop.json`

Tests:

- `tests/models/minimax_h3/test_runtime_config.py`
- `tests/models/minimax_h3/test_mixed_attention.py`
- `tests/models/minimax_h3/test_dataset_padding.py`
- `tests/models/minimax_h3/test_inference_contract.py`
- `tests/models/minimax_h3/test_joint_gradient_diagnostics.py`
- `tests/models/minimax_h3/test_training_contract.py`
- `tests/test_trainer_accumulation_metrics.py`
- `tests/test_trainer_h3_eval.py`

Updated prior handoffs:

- `docs/FASTWAM_H3_SCHEME_A_PRODUCTION_INTEGRATION_REVIEW_2026-08-21.md`
- `docs/FASTWAM_H3_SCHEME_A_VERIFICATION_HANDOFF_2026-08-20.md`

## 16. Reviewer checklist

1. Confirm that `stop_action_gradient_to_h3=false` is propagated from Hydra
   config through runtime construction into every mixed-attention block.
2. Confirm that H3 still cannot read Action tokens in the forward pass.
3. Confirm that the default action-loss backward path reaches H3 LoRA but not
   frozen H3 base parameters.
4. Review the three-sample block-0 ratios of 2759--4156x, near-zero gradient
   cosines, and the conclusion to keep beta=1 for the FastWAM-like baseline.
5. Confirm the current-cache B=2 peak of 90.08 GiB allocated / 92.97 GiB
   reserved on 95.08-GiB devices and the requirement to rerun if cache rows
   exceed the observed maximum of 140.
6. Confirm that H3 latent-only evaluation never calls a five-frame pixel
   decoder.
7. Confirm the dataset replacement threshold semantics and whether worker-local
   telemetry is sufficient.
8. Confirm that manifest/index fingerprinting is described accurately and is
   acceptable for this run.
9. Confirm that 100k optimizer steps, approximately 46.1 traversals, is the
   intended experiment budget recorded in `EXPERIMENT_37.md`.
10. Confirm accumulation-window losses are reduced using exact sample-weighted
    sums/counts across every microbatch and rank.

## 17. Post-`f14ea0d` reviewer gates

AI-A and AI-C independently accepted the beta=1 joint-gradient baseline and the
current B=2 evidence, but required two hard gates and two operational additions
before an unattended 100k continuation. These are now implemented in code.

### 17.1 Checkpoint-aware diagnostics

`scripts/diagnose_h3_joint_gradient.py` now accepts:

```yaml
diagnostic:
  checkpoint_path: /absolute/path/to/step_001000.pt
  hash_checkpoint: true
```

The checkpoint is loaded through the model's strict schema-3
`load_checkpoint()` path before any sample is evaluated. The JSON schema is now
version 2 and records checkpoint path, size, SHA256, schema, saved step,
backbone, and base-H3 fingerprint.

Selected H3 QKV/out projections capture two distinct measurements:

1. `current_lora_output_before_probe`: the LoRA/base RMS ratio of the loaded
   checkpoint before any synthetic update;
2. `local_lora_output_after_one_probe_step`: the prior reversible one-step
   AdamW stress probe.

The original LoRA state is restored after the probe. Therefore rollout metrics
evaluate the loaded checkpoint, not the temporary probe update.

The same deterministic rollout now records padding-aware normalized action
metrics in addition to five-frame latent metrics:

- valid-element count;
- L1, MSE, and RMSE;
- relative L2;
- cosine.

Time padding and padded action dimensions are both excluded.

### 17.2 Full-cache audit

`scripts/audit_h3_condition_cache.py` performs both required audit layers:

1. enumerate every unique schema-3 cache file and strictly validate that it is
   readable, matches the manifest, has width 5120, and has valid modality tags;
2. instantiate the production dataset and visit every one of its 277,713
   indices, proving that each training sample resolves to a valid cache entry.

The report contains unique-file and sample-reference row-length histograms,
both maxima, unique referenced-file count, missing/orphan counts, replacement
telemetry, and separate `passed` and `formal_gate_passed` fields. The formal
gate is true only when both scans are complete, not when a smoke-test subset
passes.

The intended post-cache command is:

```bash
PATH="/root/wuqingman/.venv-fastwam/bin:$PATH" \
PYTHONPATH="/root/wuqingman/.deps-transformers-4.57.6:src" \
torchrun --standalone --nproc_per_node=8 \
  scripts/audit_h3_condition_cache.py \
  task=libero_h3_uncond_2cam224_1e-4 \
  +cache_audit.output_path=artifacts/h3_condition_cache_full_audit.json
```

If either reported maximum exceeds 140, B=2 is blocked until
`scripts/smoke_h3_b2_memory.py` passes with that exact maximum row length.

A live partial smoke audited four unique files and two dataset references. It
correctly found dataset length 277,713, schema 3, valid 140-row references, no
errors, `passed=true`, and `formal_gate_passed=false`.

### 17.3 Continuous 1k/5k canary boundaries

Trainer now supports `stop_after_step` independently of `max_steps`. The formal
scheduler remains constructed with `max_steps=100000`, including its 5000-step
warmup, while the process stops and writes both weights and complete ZeRO-2
state at the requested boundary.

The continuous sequence is:

```text
max_steps=100000 stop_after_step=1000
  -> diagnose step 1000
resume full state, max_steps=100000 stop_after_step=5000
  -> diagnose step 5000
resume full state, max_steps=100000 stop_after_step=null
  -> continue to 100000 with save_every=10000
```

Resuming with a boundary already reached is rejected instead of silently
training one extra step. This is not a separate short-scheduler canary and does
not reset optimizer, scheduler, sampler, epoch, or accumulation state.

### 17.4 Audio-AdaLN trimming is not part of this baseline

The proposed `[video | text | audio] -> [video | text]` frozen-weight trimming
could materially increase the B=2 margin, but it changes the instantiated H3
base and therefore its fingerprint/checkpoint contract. It is being treated as
an isolated follow-up optimization requiring numerical tag-0/tag-1 equivalence
tests and a new real 8xH20 memory smoke. It is not mixed into the accepted
beta=1 baseline or used to bypass the full-cache maximum-row gate.

The local implementation confirms the opportunity and the required mechanics.
Every one of 50 blocks has an AdaLN linear projection
`2688 -> 5376 * 6 * 3`; this is approximately 13.01B weights in total. The
third modality accounts for approximately 4.337B parameters, or 8.08 GiB at
BF16. Current indexing is `inverse_indices * 3 + tag`, and the video-only path
uses stride 3. A two-slot variant would have to:

- copy the first contiguous `5376 * 6 * 2` output rows and biases from every
  released projection;
- reshape with two modalities, use `inverse_indices * 2 + tag`, and use video
  stride 2;
- reject every tag greater than 1;
- teach the safetensors loader to perform this explicit shape conversion,
  because its current `assign=True` load is strict about target tensor shapes;
- include the AdaLN variant in the base fingerprint and schema-3 LoRA
  checkpoint config.

Those changes should be exactly equivalent for tags 0 and 1 in real arithmetic,
but the equivalence must be asserted block-by-block against the unchanged
three-slot model before adopting it. The current cache contains only tags 0/1,
so no data migration would be required.

The final H3 layer is already video-only and must remain unchanged. Qwen cache
schema 3 is also unaffected because it fingerprints Qwen/processor artifacts
and stores only tags 0/1. In contrast, existing Accelerate full-state
directories contain the old model geometry and cannot safely resume into the
trimmed model. Frozen-base schema-3 weights contain only Action Expert and LoRA
tensors, so they could be supported through an explicit metadata migration;
silent compatibility under the old fingerprint is forbidden. The preferred
contract is a transformed-base fingerprint such as
`adaln-layout:vta-to-vt-v1` plus either checkpoint schema 4 or an explicit,
tested schema-3 migration path.
