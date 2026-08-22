# FastWAM-H3 Scheme A nested strict audit and direct-100k handoff

Date: 2026-08-21  
Branch: `codex/fastwam-h3-scheme-a-2026-08-19`  
Current reviewed implementation commit: `f4e2491`  
Commit title: `Close nested cache audit substitution`

## 1. Executive summary

This handoff records the work performed after AI-A and AI-C reviewed commit
`08f34b1`.

The reviewers accepted the main Scheme A implementation and most of the cache
gate, but found a nested replacement path that the first strict audit fix did
not bypass:

```text
RobotVideoDataset.get_strict()
    -> RobotVideoDataset._get()
    -> BaseLerobotDataset.__getitem__()
    -> lower read exception
    -> random retry index
```

The resulting issue was subtle:

- the outer `RobotVideoDataset.__getitem__()` replacement was bypassed;
- legal temporal-padding remap was still required;
- but the lower `BaseLerobotDataset.__getitem__()` could still hide a genuine
  read/decode/split exception by randomly selecting another index.

Commit `f4e2491` closes that path while preserving the production temporal
padding policy.

The user also made a final production-policy decision:

> Once the full cache-integrity gate and B=2 memory gate pass, do not stop at
> 1k or 5k. Start one continuous beta=1 run to 100k optimizer steps.

The code, configuration comments, handoff, tests, and queued launch automation
now reflect that decision.

No Scheme A model architecture was changed.

## 2. Current accepted Scheme A topology

The model remains:

```text
Forward:

Qwen(f0, instruction)
        |
        v
       H3
        |
        v
   Action Expert

Action tokens do not enter H3.
State tokens do not enter H3 or Qwen.
H3 cannot attend to Action.
Action Expert can attend to H3/state/action.
```

Backward remains:

```text
L_video  -> H3 attention LoRA

L_action -> Action Expert
         -> H3 attention LoRA

H3 base parameters remain frozen.
```

Production optimization remains:

```text
beta = 1
stop_action_gradient_to_h3 = false
h3_lora_rank = 32
batch_size = 2
gradient_accumulation_steps = 8
mot_checkpoint_mixed_attn = false
max_steps = 100000
```

This commit did not modify:

- packed token layout;
- H3/Action asymmetric attention masks;
- H3-to-Action forward visibility;
- Action-to-H3 forward isolation;
- joint-gradient behavior;
- diffusion targets;
- action/state alignment;
- loss coefficients;
- LoRA modules or rank;
- optimizer;
- gradient clipping;
- learning rate;
- scheduler implementation;
- checkpoint schema;
- Audio AdaLN layout.

## 3. Why the first strict getter was insufficient

### 3.1 Outer replacement

`RobotVideoDataset.__getitem__()` is a tolerant training path. For non-cache
exceptions it may record replacement telemetry and return a deterministic
random sample.

Commit `08f34b1` added:

```python
def get_strict(self, idx):
    return self._get(idx)
```

This bypassed the outer `RobotVideoDataset.__getitem__()` replacement.

### 3.2 Nested lower replacement

However, `RobotVideoDataset._get()` fetched lower samples through:

```python
self.lerobot_dataset.__getitem__
```

`BaseLerobotDataset.__getitem__()` contains a separate tolerant retry:

```text
requested lower index
    -> try multi_dataset[index]
    -> split sample
    -> on exception choose random lower index
    -> retry up to five attempts
```

Therefore the old strict path could still do:

```text
requested index i
    -> legal temporal-padding policy resolves effective index j
    -> lower read of j fails
    -> BaseLerobotDataset silently retries random index k
    -> k succeeds
    -> audit appears successful
```

This violates the intended formal audit semantics because a genuine artifact,
decode, or lower-dataset error was hidden by exception-driven substitution.

## 4. Required distinction: legal remap versus illegal substitution

Two index changes must not be conflated.

### 4.1 Legal production temporal-padding remap

Production configuration uses:

```text
skip_padding_as_possible = true
max_padding_retry = 50
```

If a requested temporal window reaches episode padding, production deliberately
selects another deterministic candidate until it obtains an unpadded target.

This behavior is part of the production sampling contract:

```text
requested index i
    -> padded temporal masks
    -> deterministic production padding policy
    -> effective unpadded index j
```

The formal audit must preserve this policy.

`padding_remap_count > 0` is not itself an error.

### 4.2 Illegal exception-driven lower substitution

If lower index `j` raises a real exception:

```text
read failure
decode failure
split failure
processor failure
shape/pathology error
```

the formal audit must fail:

```text
effective index j
    -> real exception
    -> raise
    -> reference_errors
    -> strict_lower_fetch_error_count > 0
    -> formal cache gate false
```

It must never choose random index `k`.

## 5. BaseLerobotDataset one-shot strict path

File changed:

```text
src/fastwam/datasets/lerobot/base_lerobot_dataset.py
```

The lower dataset was refactored into reusable operations.

### 5.1 One-shot raw load

```python
def _load_lerobot_sample_once(self, idx):
    lerobot_sample = self.multi_dataset[idx]
    return self._split_lerobot_sample(lerobot_sample)
```

This function performs exactly one lower access and split.

### 5.2 Shared sample construction

The original state/action/image extraction and processor logic was moved into:

```python
def _build_sample(self, sample_idx, lerobot_sample):
    ...
```

Both tolerant training and strict audit use this same construction logic.

This avoids semantic drift between the two paths.

### 5.3 New strict getter

```python
def get_strict(self, idx):
    if not 0 <= idx < len(self):
        raise IndexError(...)
    lerobot_sample = self._load_lerobot_sample_once(idx)
    return self._build_sample(idx, lerobot_sample)
```

Properties:

- validates the requested index;
- performs one lower read;
- performs one split;
- performs normal sample construction and preprocessing;
- directly propagates every exception;
- never calls `np.random.randint`;
- never retries another index.

### 5.4 Training tolerant behavior retained

`BaseLerobotDataset.__getitem__()` still:

- retries lower read/split exceptions;
- chooses a random index after a failed lower attempt;
- keeps the existing five-attempt limit;
- raises only after the retry budget is exhausted.

The production training tolerance was not removed.

The refactor preserves the original retry scope: the retry loop remains around
the lower load/split operation, as before.

## 6. RobotVideoDataset nested strict path

File changed:

```text
src/fastwam/datasets/lerobot/robot_video_dataset.py
```

### 6.1 Explicit lower fetch selection

The temporal fetch path now chooses:

```text
training:
    self.lerobot_dataset.__getitem__

strict audit:
    self.lerobot_dataset.get_strict
```

### 6.2 Production padding policy retained

Both paths use the configured temporal-padding policy.

The strict path conceptually executes:

```text
requested index i
    -> BaseLerobotDataset.get_strict(i)
    -> inspect temporal padding masks
    -> if padded, deterministic candidate j
    -> BaseLerobotDataset.get_strict(j)
    -> repeat within max_padding_retry
```

Any lower exception immediately escapes.

### 6.3 Final strict API

```python
def get_strict(self, idx):
    """Apply production padding remap without exception substitution."""
    return self._get(idx, strict_lower_fetch=True)
```

The strict audit therefore preserves legal padding behavior while bypassing
exception substitution at both dataset layers.

## 7. Padding resolver now returns the effective index

File changed:

```text
src/fastwam/datasets/padding.py
```

The existing compatibility function remains:

```python
fetch_unpadded_temporal_sample(...)
    -> sample
```

A new indexed variant was added:

```python
fetch_unpadded_temporal_sample_with_index(...)
    -> (sample, effective_index)
```

The old helper delegates to the indexed helper and discards the index.

This preserves all existing callers while allowing the formal audit to report
requested-versus-effective sampling behavior.

## 8. New strict sampling telemetry

The strict path records:

```text
strict_requested_sample_count
strict_resolved_sample_count
padding_remap_count
strict_lower_fetch_error_count
padding_remap_records
```

### 8.1 Requested count

`strict_requested_sample_count` increments once for every outer requested
dataset index presented to `RobotVideoDataset.get_strict()`.

The full audit expectation is:

```text
requested_sample_count = 277713
```

### 8.2 Resolved count

`strict_resolved_sample_count` increments after an unpadded effective sample is
successfully obtained through the lower strict path.

The formal full audit expectation is:

```text
resolved_sample_count = 277713
```

### 8.3 Legal remap count and rate

If:

```text
effective_index != requested_index
```

the audit records a legal temporal-padding remap.

The report contains:

```text
padding_remap_count
padding_remap_rate
```

The rate is:

```text
padding_remap_count / requested_sample_count
```

This count may be nonzero without failing the gate.

### 8.4 Limited debug records

The report stores a bounded number of examples:

```json
{
  "requested_index": 12340,
  "resolved_index": 87652,
  "reason": "temporal_padding"
}
```

This makes the effective production selection policy reviewable without
writing all remap pairs into the JSON.

### 8.5 Strict lower errors

Any exception raised by `BaseLerobotDataset.get_strict()` increments:

```text
strict_lower_fetch_error_count
```

The formal cache-integrity gate requires:

```text
strict_lower_fetch_error_count = 0
```

## 9. Audit gate integration

File changed:

```text
scripts/audit_h3_condition_cache.py
```

The distributed audit gathers the new counters and debug records across every
rank.

The report now includes:

```text
requested_sample_count
resolved_sample_count
padding_remap_count
padding_remap_rate
padding_remap_records
strict_lower_fetch_error_count
```

Cache integrity is:

```text
scan_passed
and complete_file_audit
and complete_reference_audit
and replacement_count == 0
and strict_lower_fetch_error_count == 0
```

The outer `replacement_count` remains as a defensive invariant even though the
strict path bypasses outer `__getitem__`.

## 10. Cache tensor validation retained

The prior `08f34b1` validation remains active.

Every unique cache file must have:

```text
schema_version = 3
hidden_layer = 50
matching encoder signature
matching Qwen fingerprint
matching processor fingerprint
prompt_embeds shape [L,5120]
L > 0
floating-point embeddings
finite embeddings
integer prompt_token_tags before conversion
tag shape [L]
tag values exactly 0 or 1
```

Missing references expose the exact missing filename/path.

No BF16-only restriction was added. Finite FP16/BF16/FP32 embeddings remain
accepted by the current schema.

No “must contain both tag 0 and tag 1” rule was added to the generic loader.

## 11. Cache integrity and B=2 memory remain separate

The report retains:

```text
formal_cache_gate_passed
b2_memory_gate_passed
requires_memory_resmoke
formal_gate_passed
```

### 11.1 Cache gate

`formal_cache_gate_passed` means:

- every unique cache file was scanned and strictly validated;
- all 277,713 requested indices were processed;
- legal padding remaps were explicitly accounted for;
- every resulting effective sample was strictly readable;
- no outer exception replacement occurred;
- no lower strict-fetch exception occurred;
- every successful effective sample resolved to a valid H3 cache;
- no required cache filename was missing.

### 11.2 B=2 memory gate

The current verified maximum is:

```text
b2_verified_max_rows = 140
```

The audit computes:

```text
max_rows = max(unique_file_max_rows, reference_max_rows)
```

Then:

```text
b2_memory_gate_passed =
    max_rows is known
    and max_rows <= 140
```

### 11.3 Exact-max resmoke

If:

```text
formal_cache_gate_passed = true
max_rows > 140
```

the report must show:

```text
b2_memory_gate_passed = false
requires_memory_resmoke = true
formal_gate_passed = false
```

The exact final maximum must be used for a new full optimizer-step:

```text
B=2
8xH20
gradient_accumulation_steps=8
no gradient checkpointing
exact maximum Qwen rows
```

The 100k launch remains blocked until that smoke passes and the verified limit
is deliberately updated.

## 12. Direct-100k production decision

The previous operating plan used:

```text
stop_after_step=1000
diagnose
resume to 5000
diagnose
resume to 100000
```

The user explicitly canceled that mandatory 1k/5k sequence.

The production configuration after all gates pass is now:

```text
beta = 1
batch_size = 2
gradient_accumulation_steps = 8
max_steps = 100000
stop_after_step = null
save_every = 10000
max_checkpoints = 2
eval_every = 0
model.stop_action_gradient_to_h3 = false
model.mot_checkpoint_mixed_attn = false
```

This gives:

```text
global batch = 2 samples/GPU * 8 GPUs * 8 accumulation
             = 128
```

The scheduler remains:

```text
total horizon = 100000 optimizer steps
warmup = 5000 optimizer steps
cosine schedule
```

There is no restart at step 1000 or 5000.

The optional `stop_after_step` infrastructure remains implemented for future
diagnostic use, but Experiment 37 production leaves it null.

Checkpoint-aware diagnostics remain available for normal saved checkpoints;
they are no longer mandatory continuation gates at 1k/5k.

## 13. Configuration and documentation changes

Files changed:

```text
configs/train.yaml
configs/task/libero_h3_uncond_2cam224_1e-4.yaml
docs/FASTWAM_H3_SCHEME_A_JOINT_GRADIENT_HANDOFF_2026-08-21.md
docs/FASTWAM_H3_STRICT_CACHE_GATE_FIX_2026-08-21.md
```

The production task comment now states that after strict cache and B=2 gates:

```text
stop_after_step = null
run continuously to 100000
native 5000-step warmup
```

Statements claiming mandatory 1k/5k canary stops were removed or changed to
describe optional diagnostic infrastructure.

## 14. Tests added

New file:

```text
tests/test_lerobot_strict_fetch.py
```

Additional gate coverage:

```text
tests/test_h3_cache_audit.py
```

### 14.1 Legal padding-remap test

The test constructs:

```text
requested index 0 -> padded
deterministic resolved index -> unpadded
```

It verifies:

- lower strict getter is used;
- lower tolerant `__getitem__` is never called;
- resolution succeeds;
- requested and resolved indices differ;
- `padding_remap_count` increments;
- the exact debug record is stored.

### 14.2 Resolved lower-error test

The test constructs:

```text
requested index -> padded
resolved lower index -> decode ValueError
```

It verifies:

- the ValueError propagates;
- no random tolerant lower call occurs;
- `strict_lower_fetch_error_count` increments;
- no resolved success is recorded.

This is the main regression test for the nested correctness issue.

### 14.3 Training tolerant behavior test

The test verifies that ordinary:

```python
dataset[index]
```

still retries a failed lower load using the existing random selection policy.

It then verifies that:

```python
dataset.get_strict(index)
```

performs exactly one attempt and raises.

### 14.4 Gate test

A nonzero:

```text
strict_lower_fetch_error_count
```

is verified to make:

```text
cache_integrity_passed = false
formal_gate_passed = false
```

## 15. Verification results

Complete maintained test suite:

```text
110 passed
1 unrelated pynvml deprecation warning
```

Additional checks:

```text
python -m compileall -q scripts src tests
    PASS

git diff --check
    PASS
```

No new IDE diagnostics were introduced. The audit script still has only the
pre-existing editor source-root import-resolution warnings.

## 16. Nested strict partial smoke

A real partial audit was executed after the nested strict fix:

```text
max_samples = 2
max_cache_files = 4
allow_partial = true
```

Observed fields:

```text
schema_version = 2
dataset_length = 277713
audited_sample_count = 2
requested_sample_count = 2
resolved_sample_count = 2
padding_remap_count = 0
padding_remap_rate = 0.0
padding_remap_records = []
strict_lower_fetch_error_count = 0
replacement_count = 0
reference_errors = []
file_errors = []
missing_referenced_files = []
max_rows = 140
b2_verified_max_rows = 140
passed = true
cache_integrity_passed = false
formal_cache_gate_passed = false
b2_memory_gate_passed = true
requires_memory_resmoke = false
formal_gate_passed = false
```

The formal gates are correctly false because the scan was deliberately
partial.

At the time of this smoke, the active precompute process had created 53,056
unique cache files. This was not the final count.

## 17. Commit and branch state

Implementation commit:

```text
f4e2491 Close nested cache audit substitution
```

Pushed branch:

```text
origin/codex/fastwam-h3-scheme-a-2026-08-19
```

The local branch and remote branch were identical after push.

The only unrelated untracked workspace item was the pre-existing:

```text
data -> /root/wuqingman/FastWAM/data
```

symlink. It was not committed.

## 18. Automated gate and launch chain

Two background stages are armed.

### 18.1 Stage one: precompute then strict full audit

The first stage waits for the currently running cache precompute process to
exit.

It then runs:

```text
torchrun --standalone --nproc_per_node=8
scripts/audit_h3_condition_cache.py
task=libero_h3_uncond_2cam224_1e-4
cache_audit.output_path=
    artifacts/h3_condition_cache_full_audit.json
```

The stage asserts:

```text
formal_gate_passed = true
```

### 18.2 Stage two: gate validation then direct 100k

The second stage waits for the audit stage to exit and reads:

```text
artifacts/h3_condition_cache_full_audit.json
```

It explicitly requires:

```text
formal_cache_gate_passed = true
b2_memory_gate_passed = true
formal_gate_passed = true
strict_lower_fetch_error_count = 0
```

Only then does it emit:

```text
DIRECT_100K_GATE_PASSED
STARTING_DIRECT_100K
```

and launch:

```text
RUN_ID=scheme-a-large-jointgrad-b2-nogc-100k-20260821

bash scripts/train_zero2.sh 8
task=libero_h3_uncond_2cam224_1e-4
max_steps=100000
stop_after_step=null
batch_size=2
gradient_accumulation_steps=8
model.stop_action_gradient_to_h3=false
model.mot_checkpoint_mixed_attn=false
```

If the audit fails, the direct run does not launch.

If final rows exceed 140, the B=2 gate fails and the direct run does not launch.

## 19. Current live runtime state

At approximately 21:38 UTC+8 on 2026-08-21:

```text
cache worker runtime: about 5 hours
worker count: 8
worker CPU utilization: approximately 100% each
GPU utilization: 0% on all 8 H20 GPUs
resident GPU memory: approximately 50,511 MiB per GPU
formal full audit: not started yet
direct 100k training: not started yet
```

All cache workers were alive in running state.

The workload remains CPU-bound because each requested dataset window must be
decoded and transformed before the content-addressed f0/instruction cache key
can be determined.

No hung worker or CUDA failure was observed.

## 20. Required final audit JSON

Before direct training, the production report must contain:

```text
schema_version = 2
dataset_length = 277713
expected_sample_count = 277713
requested_sample_count = 277713
resolved_sample_count = 277713
audited_sample_count = 277713

strict_getter_used = true
complete_file_audit = true
complete_reference_audit = true

replacement_count = 0
strict_lower_fetch_error_count = 0

file_errors = []
reference_errors = []
missing_referenced_files = []

formal_cache_gate_passed = true
b2_memory_gate_passed = true
requires_memory_resmoke = false
formal_gate_passed = true

max_rows <= 140
```

`padding_remap_count` may be nonzero. It must be reported but is not a failure
condition.

## 21. Reviewer checklist

AI-A and AI-C should verify:

1. `BaseLerobotDataset.get_strict()` performs one lower attempt;
2. lower strict fetch never calls `np.random.randint`;
3. tolerant `BaseLerobotDataset.__getitem__()` retains the prior retry policy;
4. strict and tolerant paths share the same sample-construction logic;
5. `RobotVideoDataset.get_strict()` uses the lower strict getter;
6. production temporal-padding remap remains enabled;
7. requested and effective indices are both observable;
8. a lower exception at a remapped index is propagated;
9. lower strict errors fail cache integrity;
10. legal padding remaps do not fail cache integrity;
11. the full audit still validates every unique cache file;
12. embedding finite/nonempty and exact tag checks remain active;
13. the B=2 row gate remains separate from cache correctness;
14. `max_rows > 140` blocks the combined launch gate;
15. production now uses `stop_after_step=null`;
16. no mandatory 1k/5k stop remains in the production policy;
17. the 100k scheduler horizon and 5000-step warmup remain unchanged;
18. no Scheme A model architecture changed;
19. all 110 maintained tests pass;
20. formal training is not claimed to have started before the live gate passes.

## 22. Accurate current conclusion

The accurate state is:

> Scheme A architecture, joint-gradient flow, B=2 optimizer-step evidence,
> checkpoint-aware diagnostics, strict cache tensor validation, machine B=2
> row gating, nested one-shot lower audit semantics, and direct-100k launch
> automation are implemented. Legal production padding remap is preserved and
> distinguished from forbidden exception-driven substitution. The cache
> precompute is still running; the full schema-2 audit and direct 100k training
> have not started. Once the strict cache and B=2 gates both pass, the system
> will launch one uninterrupted beta=1 100k run without mandatory 1k/5k stops.

