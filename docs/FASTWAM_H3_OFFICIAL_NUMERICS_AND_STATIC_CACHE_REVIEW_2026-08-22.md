# FastWAM-H3 official numerics alignment and static-cache implementation review

Date: 2026-08-22  
Repository: `/root/wuqingman/FastWAM-H3-scheme-a-verify`  
Branch baseline discussed in the review: `codex/fastwam-h3-scheme-a-2026-08-19`  
Pre-change reviewed commit: `f4e2491`  
Status of the changes in this document: local working tree, not committed  

## 1. Purpose of this document

This document records the implementation work performed after the following
decision:

> The goal is not to redesign Scheme A. The goal is to align FastWAM-H3 with
> the released H3 numerical execution contract before producing permanent
> training caches.

The reason for doing numerical alignment first is that the proposed caches are
static inputs to every optimizer step. A wrong dtype boundary, wrong VAE
posterior sample, or wrong Refiner output would otherwise be serialized once
and then reused throughout the planned 100k run.

This handoff is intended for an independent AI reviewer. It therefore records:

- the accepted architecture that was intentionally not changed;
- the defects reported by AI-A and checked against the released H3 code;
- the exact implementation changes;
- the cache semantics and schemas;
- tests and real-model smoke results;
- failures encountered during large-scale cache generation;
- the current live cache-generation state;
- claims that are not yet proven and gates that remain open.

## 2. External baselines used for reasoning

The review supplied the following baselines:

- FastWAM-H3 baseline commit: `f4e2491`;
- MiniMax H3 model revision: `42ed227ee7df40d41602854ae760620d6eb651fe`;
- Hugging Face Diffusers H3 implementation revision:
  `2f7e0154a9db246e95c9ede43edba7db5b130805`.

Important limitation:

The public H3 code is primarily an inference implementation. It provides strong
evidence for Transformer dtype boundaries, VAE execution, FL2VA keyframe
conditioning, scheduler updates, packed layout, and decoding. It does not
provide the original MiniMax training loop. Therefore:

- seed-42 keyframe conditioning is aligned to the released FL2VA inference
  contract;
- full-video posterior resampling is a FastWAM training-semantics decision;
- the current flow-matching timestep distribution and loss weighting cannot be
  described as the original unpublished H3 training objective.

## 3. Scheme A architecture deliberately preserved

No redesign of the accepted Scheme A topology was intended.

The preserved forward topology is:

```text
f0 + instruction
        |
        v
Qwen3-VL layer-50 PRENORM condition
        |
        v
condition_proj + TokenRefiner
        |
        v
H3 video expert
        |
        v
Action Expert reads H3/state/action rows
```

The preserved asymmetry is:

```text
H3 video rows do not read action rows.
Action rows can read H3 rows.
State is an Action Expert condition row.
```

The preserved backward behavior is:

```text
video loss  -> H3 attention LoRA
action loss -> Action Expert
            -> H3 attention LoRA through the H3 features read by Action Expert

H3 base weights remain frozen.
```

The following were not intentionally changed:

- beta=1 Scheme A topology;
- independent Action Expert;
- action/state online trainable encoders;
- H3-to-Action forward visibility;
- Action-to-H3 forward isolation;
- packed sequence semantics;
- keyframe and noisy full-video row separation;
- video loss applying only to noisy full-video rows;
- action loss masking;
- H3 LoRA rank or placement;
- optimizer policy;
- direct-100k production policy.

No 100k training job has been started by this work.

## 4. Static-cache boundary decision

The cache-boundary discussion established the following rule:

> Outputs of frozen deterministic components may be cached. Outputs of
> trainable components must remain online.

### 4.1 Safe and useful static outputs

The selected static outputs are:

```text
1. Post-Refiner Qwen condition
   shape: [L, 5376]
   dtype: BF16

2. Full-video VAE posterior moments
   mean:   [24, T_latent, 14, 28]
   logvar: [24, T_latent, 14, 28]
   dtype: FP32

3. Deterministic keyframe condition latent
   shape: [24, 1, 14, 28]
   dtype: FP32
```

### 4.2 Outputs intentionally not cached

The following remain online:

```text
normalized action [32, 7]
    -> trainable action_encoder

state_f0 [8]
    -> trainable state_encoder
```

Caching action/state encoder embeddings would bypass the evolving encoder
weights and remove those encoders from the effective training graph. This work
does not do that.

### 4.3 Why keyframe and full-video cache semantics differ

Keyframe:

- follows released FL2VA conditioning;
- posterior sampling uses an independent generator with seed 42;
- the sampled raw latent is rounded through FP16;
- normalization is performed in FP32;
- the final normalized latent is deterministic and is cached directly.

Full video:

- is the diffusion target used during training;
- caching one sampled latent would freeze posterior noise for the entire run;
- caching only the mean would remove posterior sampling;
- therefore mean and logvar are cached;
- a new posterior sample is drawn online during each training call.

This distinction is intentional.

## 5. H3 FP32/BF16 boundary corrections

Primary file:

`src/fastwam/models/minimax_h3/video_dit.py`

Scheduler file:

`src/fastwam/models/wan22/schedulers/scheduler_continuous.py`

### 5.1 Timestep embedding

The previous path converted the timestep MLP result back to the model BF16
dtype too early.

The corrected semantic path is:

```text
timestep / Fourier embedding
    -> FP32
FP32 linear projection
    -> FP32 SiLU
FP32 output projection
    -> FP32 timestep embedding
```

The BF16 conversion now occurs only at the boundary of the BF16 AdaLN
projection weight.

Affected implementation:

- `H3TimeEmbedder.forward`;
- `H3AdaLayerNormModulation.forward`;
- related final/output AdaLN handling.

### 5.2 AdaLN activation boundary

The intended order is:

```text
FP32 timestep embedding
    -> SiLU in FP32
    -> cast to AdaLN linear weight dtype
    -> BF16 AdaLN projection
```

This prevents SiLU from being executed on an already-truncated BF16 timestep
embedding.

### 5.3 Final video velocity

The previous final layer converted the FP32 output-head result back to the
hidden-state dtype.

The corrected final layer preserves FP32 logits/velocity.

This matters because the output is consumed by flow/scheduler arithmetic and
because the released H3 output projection is an FP32 boundary.

### 5.4 Scheduler construction and Euler update

The scheduler now:

- builds inference timesteps and deltas in FP32;
- converts sample and model output to FP32 for the Euler update;
- returns the updated sample in the caller's original sample dtype.

This avoids performing the numerically sensitive update itself in BF16.

### 5.5 RoPE

The review confirmed the existing intent that RoPE computation is FP32. This
work did not redesign the position layout or Action/H3 clock.

## 6. Video VAE precision and posterior changes

Primary file:

`src/fastwam/models/minimax_h3/video_vae.py`

### 6.1 FP32 VAE weights

The adapter previously allowed the entire VAE to be moved to the requested
model dtype, which was BF16 in production.

The adapter now keeps the VAE model weights in FP32 regardless of the H3
Transformer BF16 dtype.

This affects:

- encoder;
- decoder;
- quant convolution;
- post-quant convolution.

### 6.2 FP32 RGB input

Primary integration file:

`src/fastwam/models/minimax_h3/fastwam.py`

Video and keyframe image tensors are now supplied to the VAE as FP32.

Casting a previously truncated BF16 image back to FP32 cannot restore lost
precision, so this correction was made before the VAE call rather than only
inside the VAE.

The PIL conversion helpers also round and clamp to uint8 explicitly before
permuting/converting image arrays.

### 6.3 Full-video posterior moments

The VAE adapter now exposes:

```python
encode_video_posterior(video) -> (normalized_mean, normalized_logvar)
```

The implementation:

- executes the official prefix encoding mechanics;
- obtains raw posterior moments without sampling;
- converts mean and logvar to FP32;
- transforms the distribution into normalized latent space;
- returns normalized FP32 moments.

For an affine normalization:

```text
z_normalized = (z_raw - latent_mean) / latent_std
```

the distribution transformation is:

```text
mean_normalized   = (mean_raw - latent_mean) / latent_std
logvar_normalized = logvar_raw - 2 * log(latent_std)
```

Training samples from these cached normalized moments online.

### 6.4 Deterministic FL2VA keyframe path

The adapter now exposes:

```python
encode_keyframe_condition(image, seed=42)
```

The intended released FL2VA recipe is:

```text
RGB keyframe
    -> ImageNet normalization
    -> FP32 VAE posterior
    -> sample with independent generator(seed=42)
    -> raw latent cast to FP16
    -> cast back to FP32
    -> FP32 latent mean/std normalization
    -> deterministic normalized keyframe latent
```

The seed-42 generator is local to this operation. It does not consume or depend
on the request/training generator state.

### 6.5 Decode autocast

VAE weights remain FP32, while the CUDA decode call is wrapped in FP16
autocast, matching the released decoding execution style.

Latent denormalization first converts latent values to FP32.

### 6.6 Noise draw order

The model-side random draw order was changed to align with the released FL2VA
conditioning sequence:

```text
keyframe augmentation noise
    -> video noise
    -> action noise
```

The keyframe posterior itself is not drawn from this generator because the
keyframe VAE sample uses the independent seed-42 generator.

## 7. Training and inference integration

Primary file:

`src/fastwam/models/minimax_h3/fastwam.py`

### 7.1 Training with online VAE

When no VAE cache is supplied:

- full video is encoded to posterior mean/logvar;
- training samples the full-video latent online;
- the keyframe is encoded with the deterministic keyframe method.

### 7.2 Training with VAE cache

When the dataset supplies:

```text
clean_keyframe_latents
video_posterior_mean
video_posterior_logvar
```

the model:

- does not require the VAE to be loaded;
- samples video latents online from cached moments;
- directly uses the deterministic cached keyframe latent.

This permits cache-only training startup with `load_vae=False`.

### 7.3 Conditional VAE loading

Files:

- `src/fastwam/models/minimax_h3/fastwam.py`;
- `src/fastwam/runtime.py`.

`FastWAMH3.from_pretrained` and the runtime factory now accept a `load_vae`
control. This is needed to avoid loading a large frozen FP32 VAE when all
required VAE static values are present in the dataset payload.

### 7.4 Inference compatibility

Inference uses deterministic keyframe encoding when the adapter provides the
new method.

A legacy fallback to `encode_video(..., process_image=True)` remains for older
test doubles/adapters. Production correctness therefore depends on loading the
new `MiniMaxH3VAEAdapter`, not the fallback.

### 7.5 Post-Refiner condition input

The H3 forward path accepts either:

```text
pre-Refiner Qwen rows: [L, 5120]
post-Refiner rows:     [L, hidden_size] = [L, 5376]
```

For 5120-wide input, the online frozen condition projection and TokenRefiner
execute.

For 5376-wide input, these modules are skipped.

The implementation rejects bypassing trainable projection/Refiner modules. The
post-Refiner path is only valid because these components are frozen.

## 8. Isolated H3 condition Refiner

Primary file:

`src/fastwam/models/minimax_h3/video_dit.py`

New abstraction:

```text
H3ConditionRefiner
    = condition_proj
    + TokenRefiner
```

New loader:

```python
load_h3_condition_refiner(...)
```

The loader reads only the condition projection and TokenRefiner weights rather
than instantiating all 50 H3 blocks.

This is used for:

- converting old Qwen cache rows;
- completing missing Qwen+Refiner rows;
- direct parity smoke against the complete H3 condition path.

Config reads for normalization epsilons use explicit defaults when those keys
are absent.

## 9. Post-Refiner cache schema

Primary loader/schema file:

`src/fastwam/datasets/h3_condition_cache.py`

Converter:

`scripts/convert_h3_post_refiner_cache.py`

Completion script:

`scripts/precompute_h3_post_refiner_complete.py`

### 9.1 Schema compatibility

The loader supports:

```text
schema 3:
    Qwen layer-50 PRENORM [L, 5120]

schema 4:
    Post-Refiner condition [L, 5376]
```

When a schema-4 manifest is present, the cache path suffix is:

```text
.h3-post-refiner-v4.pt
```

### 9.2 Schema-4 payload

The schema-4 payload contains:

```text
schema_version = 4
source filename/mode
refiner fingerprint
prompt_embeds [L, 5376], BF16
prompt_token_tags [L], integer
payload SHA256
```

The strict loader validates:

- schema version;
- Refiner fingerprint;
- exact embedding width;
- exact BF16 embedding dtype;
- finite values;
- tag shape/type;
- payload checksum.

### 9.3 Refiner fingerprint

The converter fingerprints:

- Transformer config;
- safetensors index;
- all actual shard files containing `condition_proj.*`;
- all actual shard files containing `token_refiner.*`.

This closes the prior weakness where hashing only config/index metadata would
not detect modified shard contents.

### 9.4 Existing schema-3 conversion

For old cache entries:

```text
cached BF16 Qwen [L, 5120]
    -> condition_proj
    -> TokenRefiner with original sequence boundary
    -> BF16 [L, 5376]
    -> schema-4 file
```

This is expected to be numerically identical to direct Qwen followed by the
same frozen Refiner when:

- Qwen weights and processor match;
- layer-50 PRENORM extraction matches;
- cached Qwen dtype is BF16 without additional quantization;
- tags and sequence boundaries match;
- Refiner weights and implementation match.

### 9.5 Missing schema-3 completion

The completion script scans strict dataset samples.

For each sample:

- derive the cache key from the first frame and instruction;
- skip if schema-4 output already exists;
- use schema-3 Qwen rows if available;
- otherwise run Qwen from the first frame and instruction;
- apply the isolated Refiner;
- atomically save schema-4 output.

### 9.6 Resume support

Large-scale generation revealed the need for resumable dataset scanning.

The completion script now accepts:

```text
+start_index=<global dataset index>
```

Each distributed rank computes the first index at or after `start_index` that
belongs to that rank's modulo shard.

This is an operational resume mechanism. It is not a substitute for the final
full strict audit; an overly aggressive start index could skip an unfilled
sample.

### 9.7 Removal of final NCCL barrier

The schema-4 files are independent per sample. A final collective barrier is
not required for correctness.

It was removed after a real run showed heavily unequal missing work per rank:

- some ranks completed and entered the barrier;
- other ranks were still running Qwen/Refiner work;
- the completed ranks timed out waiting in NCCL;
- torchrun terminated all ranks despite valid files already being saved.

Each rank now prints its own completion result and exits independently.
`torchrun` remains responsible for waiting for all child processes.

## 10. VAE cache schema

Schema file:

`src/fastwam/datasets/h3_vae_cache.py`

Precompute script:

`scripts/precompute_h3_vae_cache.py`

Dataset integration:

`src/fastwam/datasets/lerobot/robot_video_dataset.py`

### 10.1 Manifest

Schema version: 1.

The manifest records:

```text
VAE fingerprint
dataset/processor signature
implementation signature
keyframe semantic signature
video posterior semantic signature
exact cache dtype = torch.float32
```

Semantic labels currently include:

```text
keyframe:
seed42-sample-fp16-round-fp32-normalize

video:
fp32-normalized-posterior-mean-logvar
```

### 10.2 Cache key

The key includes:

- exact FP32 video tensor content digest;
- VAE fingerprint;
- processor signature;
- implementation signature;
- schema version.

Using the FP32 dataset tensor avoids silently conflating values after an
additional uint8 conversion.

### 10.3 Payload

Each payload stores:

```text
clean_keyframe_latents
video_posterior_mean
video_posterior_logvar
video content digest
manifest fingerprints
payload checksum
```

All three tensors must be FP32, rank 4, finite, and shape-compatible.

The keyframe temporal dimension must be one. Video mean/logvar shapes must
match.

Writes are atomic through a temporary file followed by `os.replace`.

### 10.4 Dataset loading

`RobotVideoDataset` accepts an optional `h3_vae_cache_dir`.

After constructing the normal sample, it loads and adds:

```text
clean_keyframe_latents
video_posterior_mean
video_posterior_logvar
```

Action and state values remain the normal dataset values and continue through
their online trainable encoders.

## 11. Tests and real-model evidence

### 11.1 Automated tests

The directed suite reported:

```text
113 passed
```

No linter errors were reported for the edited files.

The full repository collection was blocked by the unrelated optional RoboTwin
dependency `sapien`; this was not caused by the H3 changes.

Relevant test changes include:

- dtype expectations for timestep, patch projection, and final output;
- isolated `H3ConditionRefiner` equivalence;
- training without a loaded VAE when static VAE cache values are supplied;
- FP32 video/action timestep assertions;
- schema-4 BF16 and checksum validation;
- VAE cache round-trip and checksum tamper rejection.

### 11.2 Post-Refiner real-model smoke

Four real schema-3 samples were converted.

The isolated partial Refiner output was compared with the complete H3
condition-refinement path:

```text
max_abs_diff = 0
```

The produced embeddings had:

```text
width = 5376
dtype = BF16
finite = true
checksum = valid
```

Smoke directory:

`artifacts/h3_post_refiner_cache_smoke/`

### 11.3 VAE real-model smoke

Two real dataset samples were used.

Observed shapes:

```text
keyframe latent: [24, 1, 14, 28], FP32
video mean:      [24, 2, 14, 28], FP32
video logvar:    [24, 2, 14, 28], FP32
```

The keyframe encoding was reproducible with seed 42.

Repeated sampling from the same video posterior moments produced different
samples, confirming that posterior randomness remains online rather than being
frozen in the cache.

Smoke directory:

`artifacts/h3_vae_cache_smoke/`

## 12. Large-scale Post-Refiner cache execution

### 12.1 Intended two-stage run

The 8-GPU job performs:

```text
stage 1:
convert all existing schema-3 Qwen files to schema 4

stage 2:
strictly scan the full dataset
reuse schema-3 where available
run Qwen from scratch where schema-3 is missing
produce complete schema-4 coverage
```

### 12.2 Initial performance problems fixed

Refiner fingerprint:

- hashing large safetensor shards repeatedly on every process caused heavy
  startup I/O;
- the existing manifest fingerprint is now reused;
- rank 0 computes a new fingerprint only when initializing a new manifest.

Tensor checksum:

- Python conversion through `bytes(untyped_storage())` was a bottleneck in the
  converter;
- it was changed to a NumPy-backed byte memoryview for C-level hashing.

### 12.3 Critical shared-storage serialization bug

The converter initially saved a per-sample slice of a packed batch without
cloning it.

Although the slice shape was correct, the slice shared the full packed batch
storage. `torch.save` serialized the entire underlying storage for each sample.

Observed consequence:

```text
approximately 23,251 output files
approximately 2.6 TB consumed
root filesystem filled
PytorchStreamWriter file write failure
```

Fix:

```python
refined = refined_packed[start:end].clone()
```

All oversized schema-4 outputs from that failed run were removed, reclaiming
the disk space. The compact conversion was restarted.

### 12.4 Existing Qwen conversion result

The corrected converter completed all existing unique schema-3 files:

```text
93,865 schema-4 files
approximately 126.457 GiB
```

The compact conversion completed quickly after the storage and checksum fixes.

### 12.5 Full completion scan and generation

The full scan showed that the old schema-3 set did not cover every strict
dataset sample.

The job transitioned from skip-only scanning into direct Qwen+Refiner
generation.

Before the first long process ended, the output reached:

```text
221,846 files
approximately 296.599 GiB
```

No CUDA OOM or disk-full event occurred in this corrected run.

### 12.6 First interruption

The first corrected long shell task ended after roughly 13 hours with exit code
1 but without a Python/CUDA/disk traceback in its output.

Existing atomic cache files remained valid.

Resume support was added and the job restarted conservatively from global
dataset index 180,000.

### 12.7 NCCL barrier failure during resume

The resumed job completed the assigned range on ranks 0, 1, and 2, with logs
including:

```text
rank 0: generated=0,   skipped=12215
rank 1: generated=367, skipped=11847
rank 2: generated=264, skipped=11950
```

Other ranks had not yet completed when an NCCL collective timeout occurred at
the final barrier.

The run still atomically added valid files and reached:

```text
222,889 files
approximately 298.024 GiB
```

The final barrier was then removed.

### 12.8 Current live resume

At the time this document was written:

```text
resume start_index = 240000
world size         = 8 GPUs
cache batch size   = 8
files present      = 222,889
disk usage         = approximately 298.024 GiB
free filesystem    = approximately 2.0 TiB
```

The resumed process was running and scanning/skipping the already completed
part of the suffix range. It had not reported a new error.

This number is a live operational snapshot, not the final audited count.

## 13. Files changed or added

### 13.1 Model and runtime

`src/fastwam/models/minimax_h3/video_dit.py`

- corrected timestep/AdaLN/final-output dtype boundaries;
- added `H3ConditionRefiner`;
- added isolated Refiner loader;
- accepted pre- and post-Refiner condition widths;
- guarded post-Refiner bypass against trainable Refiner components.

`src/fastwam/models/minimax_h3/video_vae.py`

- preserved FP32 VAE weights;
- added posterior-moment encoding;
- added deterministic keyframe condition encoding;
- added FP16 decode autocast;
- made normalization/denormalization FP32.

`src/fastwam/models/minimax_h3/fastwam.py`

- integrated cached VAE values;
- sampled full-video posterior online;
- integrated deterministic keyframe path;
- corrected image precision;
- corrected noise draw order;
- supported cache-only training without a VAE;
- accepted post-Refiner conditions.

`src/fastwam/models/minimax_h3/__init__.py`

- exported new condition/VAE-related components as required.

`src/fastwam/models/wan22/schedulers/scheduler_continuous.py`

- preserved FP32 schedule and Euler arithmetic.

`src/fastwam/runtime.py`

- added conditional VAE loading.

### 13.2 Dataset and cache

`src/fastwam/datasets/h3_condition_cache.py`

- added schema-4 manifest/path/loading;
- added exact BF16/width/fingerprint/checksum validation;
- preserved schema-3 compatibility.

`src/fastwam/datasets/h3_vae_cache.py`

- new strict VAE cache schema, keys, fingerprints, checksums, and atomic I/O.

`src/fastwam/datasets/lerobot/robot_video_dataset.py`

- added optional VAE cache loading.

### 13.3 Scripts

`scripts/convert_h3_post_refiner_cache.py`

- new distributed schema-3 to schema-4 converter;
- includes batching, full Refiner shard fingerprint, checksum, and compact
  cloned saves.

`scripts/precompute_h3_post_refiner_complete.py`

- new strict full-dataset completion script;
- reuses old Qwen cache when available;
- runs direct Qwen for missing entries;
- supports `start_index`;
- no final NCCL barrier.

`scripts/precompute_h3_vae_cache.py`

- new distributed FP32 VAE cache precomputation script.

### 13.4 Tests

`tests/models/minimax_h3/test_condition_cache.py`

- schema-4 exact dtype/checksum validation.

`tests/models/minimax_h3/test_conditioning.py`

- corrected dtype assertions;
- isolated Refiner equivalence coverage.

`tests/models/minimax_h3/test_training_contract.py`

- new VAE cache-only training path;
- updated VAE test double;
- FP32 timestep assertions.

`tests/models/minimax_h3/test_vae_cache.py`

- new VAE cache round-trip and tamper test.

## 14. Important unresolved items

The following must not be described as completed.

### 14.1 No complete official 50-block parity gate yet

The work includes:

- component-level dtype contract tests;
- isolated Refiner vs complete local H3 Refiner exact smoke;
- scheduler contract changes;
- real VAE semantic smoke.

It does not yet include a single test that loads the pinned official Diffusers
50-block implementation and the FastWAM implementation with identical packed
inputs and compares final full-H3 output.

Still desirable:

- timestep/AdaLN direct official parity;
- one Transformer block direct official parity;
- complete 50-block frozen H3 direct official parity;
- scheduler trajectory direct official parity;
- full end-to-end direct-Qwen-vs-schema-4 sampled parity.

### 14.2 Post-Refiner cache is not yet fully audited

The current file count is not proof of strict full-dataset coverage.

Required final gate:

- strict dataset traversal;
- no random sample substitution;
- zero missing schema-4 keys;
- zero invalid dtype/shape/fingerprint/checksum payloads;
- duplicate-key accounting;
- sampled direct Qwen+Refiner parity;
- a persisted audit artifact with an explicit pass field.

### 14.3 Full VAE cache has not been generated

Only the VAE implementation, schema, tests, and two-sample real smoke are
complete.

The full FP32 VAE cache has not yet been generated or fully audited.

### 14.4 VAE precompute still contains distributed barriers

`scripts/precompute_h3_vae_cache.py` uses a barrier after manifest
initialization and a final barrier after processing.

The initial barrier is useful to ensure the manifest exists.

The final barrier may deserve removal or a longer timeout before a large
resume run, because the Post-Refiner run demonstrated that uneven skip/work
distribution can make a final barrier fail even when independent outputs are
correct.

### 14.5 No B=2 optimizer-step production smoke after all caches

A full cache-only model load and B=2 optimizer-step smoke remains required to
prove:

- `load_vae=False` production wiring;
- post-Refiner bypass in the real model;
- online full-video posterior sampling;
- online trainable action/state encoders;
- expected memory use;
- finite losses and gradients.

### 14.6 No 100k launch

No training should be launched until:

- complete schema-4 condition-cache audit passes;
- full VAE cache generation and audit pass;
- B=2 optimizer-step/memory gate passes;
- the reviewer accepts the remaining parity limitations or requests the full
  official implementation parity test.

## 15. Reviewer focus questions

The independent reviewer should specifically answer:

1. Are the timestep MLP, AdaLN activation/cast, final output, and scheduler
   FP32/BF16 boundaries now identical to the pinned released implementation?

2. Is normalized posterior logvar transformed correctly when moving the VAE
   Gaussian distribution into normalized latent space?

3. Does `encode_keyframe_condition` exactly reproduce ImageNet normalization,
   seed-42 posterior sampling, FP16 rounding, and FP32 normalization in the
   correct order?

4. Is the keyframe/video/action RNG draw order correct in both training and
   inference, and is any global RNG still consumed before the intended draws?

5. Can `[L, 5376]` input bypass `condition_proj + TokenRefiner` under every
   production model configuration without silently bypassing trainable
   parameters?

6. Does the schema-4 key/fingerprint cover every artifact that can change the
   Refiner output, including implementation code and all relevant weights?

7. Is the VAE processor signature stable and sufficiently complete, or does it
   accidentally include irrelevant runtime fields / omit relevant transforms?

8. Should schema-4 direct-Qwen outputs also store a source Qwen tensor digest
   for consistency with converted schema-3 outputs?

9. Should the VAE cache tensor hashing be changed from
   `bytes(untyped_storage())` to the faster memoryview approach already used in
   the condition converter?

10. Is `start_index` resume safe enough operationally if and only if followed
    by a complete strict audit?

11. Should the final VAE precompute barrier be removed before full-scale
    generation?

12. Is a pinned official full 50-block parity test mandatory before the 100k
    run, or are the component tests and real-model smokes sufficient for the
    accepted risk level?

## 16. Current conclusion

The main Scheme A architecture was not rebuilt.

The implementation now has:

- corrected H3 mixed-precision boundaries;
- FP32 VAE weights and FP32 RGB inputs;
- deterministic released-style keyframe conditioning;
- resampleable full-video posterior-moment caching;
- BF16 Post-Refiner condition caching;
- strict payload checksums and weight fingerprints;
- cache-only training support without loading the VAE;
- real-model component smokes and 113 passing directed tests.

However, the project is not yet ready to claim formal production readiness.
The Post-Refiner full scan is still running, the full VAE cache has not been
generated, the final strict audits and B=2 optimizer smoke remain open, and a
direct pinned official complete 50-block parity gate has not yet been run.
