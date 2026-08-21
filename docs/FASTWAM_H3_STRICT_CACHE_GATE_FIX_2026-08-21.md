# FastWAM-H3 Scheme A strict cache gate fix

Date: 2026-08-21  
Branch: `codex/fastwam-h3-scheme-a-2026-08-19`  
Implementation commit: `08f34b1` (`Make the H3 cache gate strict`)  
Previous reviewed commits: `07d0a10`, `9f6117e`

## 1. Purpose of this document

This document records the work performed after AI-A and AI-C reviewed the
checkpoint-aware diagnostic, optional stop-boundary mechanism, full-cache
audit, and Audio AdaLN analysis.

The reviewers accepted:

- the beta=1 FastWAM-like joint-gradient topology;
- checkpoint-aware H3 diagnostics;
- pre-probe and post-probe LoRA/base RMS measurements;
- padding-aware action rollout metrics;
- the scheduler-preserving optional `stop_after_step` mechanism;
- separation of the Audio AdaLN optimization from the production baseline.

They found one remaining production gate correctness issue:

> The cache audit must preserve the production temporal-padding remap policy,
> but it must bypass exception-driven substitution at both
> `RobotVideoDataset` and `BaseLerobotDataset`.

They also requested:

1. stricter validation of cached tensors;
2. an explicit, machine-readable B=2 maximum-row memory gate;
3. nonzero production exit status when any formal gate fails;
4. more accurate reporting of missing cache filenames.

This document describes the resulting fix. No model architecture, loss,
optimizer, scheduler, beta, AdaLN layout, or training topology was changed.

## 2. Original audit correctness problem

### 2.1 Training dataset behavior

`RobotVideoDataset.__getitem__()` is intentionally tolerant during training.
Its behavior is conceptually:

```text
sample_attempt_count += 1

try:
    return _get(original_index)
except FileNotFoundError:
    raise
except other sample error:
    record replacement telemetry
    replacement_index = deterministic_random(original_index)
    return _get(replacement_index)
```

This behavior is useful in a long training process because an occasional
non-cache sample pathology does not immediately terminate the run, subject to
the configured replacement-rate safety threshold.

It is not appropriate for a formal artifact audit. In addition,
`BaseLerobotDataset.__getitem__()` has its own five-attempt random retry around
the lower dataset read. Bypassing only the outer replacement is insufficient.

### 2.2 How the old audit could produce a false proof

The old audit used:

```python
sample = dataset[sample_index]
```

Suppose original index 12,345 raised a video processing, shape, padding, or
processor exception. Training `__getitem__()` could substitute index 87,654.
The audit would then compute the cache digest and row count for index 87,654.

The report could therefore contain:

```text
dataset_length = 277713
audited_sample_count = 277713
reference_errors = []
missing_referenced_files = []
replacement_count = 5
formal_gate_passed = true
```

That only proves that 277,713 calls eventually returned some sample. It does
not prove that every effective sample selected by the production padding policy
was read without exception substitution.

The old formal gate also reported `replacement_count` but did not require it to
be zero.

## 3. Nested strict API with legal padding remap

File changed:

```text
src/fastwam/datasets/lerobot/base_lerobot_dataset.py
src/fastwam/datasets/lerobot/robot_video_dataset.py
```

`BaseLerobotDataset` now has a one-shot `get_strict(idx)` path. It performs the
normal lower read, split, state/action/image extraction, and processor
preprocessing, but raises every exception directly and never retries another
index.

`RobotVideoDataset` exposes:

```python
def get_strict(self, idx):
    """Apply production padding remap without exception substitution."""
    return self._get(idx, strict_lower_fetch=True)
```

The distinction is now explicit:

```text
Training:
    dataset[index]
    -> __getitem__()
    -> replacement policy remains available

Formal audit:
    RobotVideoDataset.get_strict(requested_index)
    -> production deterministic temporal-padding selection
    -> BaseLerobotDataset.get_strict(effective_index)
    -> no exception-driven replacement at either layer
```

Legal padding remap remains enabled. A padded requested index `i` may resolve
under the exact production policy to unpadded index `j`. The audit strictly
validates `j` and records `i -> j`. A real read/decode failure at `j` is raised;
it cannot be hidden by selecting random index `k`.

The audit requires a callable outer `get_strict(index)`. It fails immediately
if the configured dataset does not provide that API and never silently falls
back to `__getitem__`.

This change does not alter normal training data loading.

## 4. Strict cache tensor validation

File changed:

```text
src/fastwam/datasets/h3_condition_cache.py
```

The independent cache-file loader already validated:

- cache schema version 3;
- H3 hidden layer 50;
- Qwen encoder signature;
- Qwen checkpoint fingerprint;
- processor fingerprint;
- embedding rank and width;
- token-tag shape;
- modality tags restricted to video/text.

The following checks were added.

### 4.1 Nonempty embedding rows

The embedding must have:

```text
shape = [L, 5120]
L > 0
```

An empty `[0, 5120]` embedding is rejected. Such a tensor previously satisfied
the rank/width check but would produce an invalid empty Qwen condition.

### 4.2 Floating-point embedding dtype

`prompt_embeds` must be a floating-point tensor. Integer and other invalid
representations are rejected before training.

### 4.3 Finite embeddings

Every embedding element must be finite:

```python
torch.isfinite(embeddings).all()
```

Any NaN or positive/negative infinity makes the cache file invalid.

### 4.4 Exact integer modality tags

The old loader converted tags to `torch.long` before validating their values.
A corrupt floating tag such as `0.5` could therefore become integer `0`.

Validation now occurs before conversion:

```text
raw dtype must be one of:
uint8, int8, int16, int32, int64

raw value must be exactly:
0 or 1
```

Only after these checks are the tags normalized to `torch.long`.

### 4.5 Exact missing-cache filename

`load_h3_condition_cache()` now attaches the resolved missing cache path to the
raised `FileNotFoundError.filename`.

The audit records both:

- the sample index;
- the exact missing cache filename/path.

This fixes the prior reporting limitation where `reference_errors` failed the
gate but `missing_referenced_files` could remain empty.

## 5. Strict full-cache reference audit

File changed:

```text
scripts/audit_h3_condition_cache.py
```

For every requested dataset index, the audit calls:

```python
sample = dataset.get_strict(sample_index)
```

Successful samples contribute:

- row-length histogram entries;
- exact content-addressed cache digest;
- unique referenced filename;
- requested and resolved sample counts;
- legal temporal-padding remap count/rate;
- limited `requested_index -> resolved_index` debug records.

Failures contribute:

- requested sample index;
- exception type and message;
- exact missing cache path when available.

The resulting semantics are:

```text
277,713 requested training indices
-> exact production padding-selection policy
-> effective unpadded samples
-> strict one-shot lower reads
-> zero exception-driven substitutions
```

The audit still distinguishes:

```text
277,713 dataset references
from
the smaller set of unique content-addressed cache files
```

This distinction is required because cache identity is based on:

```text
normalized f0 pixels
+ instruction
+ cache schema
+ Qwen encoder signature
+ Qwen checkpoint fingerprint
+ processor fingerprint
```

Multiple training windows may legitimately reference the same cache file.

## 6. New machine-readable gate model

The audit report schema was increased from version 1 to version 2.

The old single gate was insufficient because cache correctness and B=2 memory
compatibility are different facts.

### 6.1 `passed`

`passed` describes the scanned material:

- no unique-file validation errors;
- no strict reference errors;
- no missing referenced files;
- configured expected dataset length matches.

For a deliberately limited smoke subset, `passed` may be true while all formal
gates remain false.

### 6.2 `cache_integrity_passed`

This is true only when:

```text
passed
and complete_file_audit
and complete_reference_audit
and replacement_count == 0
and strict_lower_fetch_error_count == 0
```

### 6.3 `formal_cache_gate_passed`

This is currently identical to `cache_integrity_passed`. It is included as an
explicit production-facing field:

```text
all unique files strictly valid
and all 277,713 requested indices resolved under production padding policy
and every effective sample was strictly readable
and no exception-driven replacements occurred
```

An orphan cache file does not fail this gate. Orphans are valid
content-addressed files not referenced by the current dataset and do not affect
training correctness.

### 6.4 `b2_memory_gate_passed`

The audit computes:

```text
unique_file_max_rows
reference_max_rows
max_rows = max(the two available maxima)
```

The default verified B=2 boundary is:

```text
b2_verified_max_rows = 140
```

The B=2 gate is:

```text
max_rows is known
and max_rows <= b2_verified_max_rows
```

This converts the prior documentation-only rule into a machine-readable gate.

### 6.5 `requires_memory_resmoke`

This is true when:

```text
cache_integrity_passed
and not b2_memory_gate_passed
```

For example, if the final cache is fully correct but `max_rows=141`:

```text
formal_cache_gate_passed = true
b2_memory_gate_passed = false
requires_memory_resmoke = true
formal_gate_passed = false
```

The cache itself is not declared corrupt. Instead, the current B=2 memory
evidence is declared insufficient.

### 6.6 Combined `formal_gate_passed`

The final production launch gate is:

```text
formal_gate_passed =
    formal_cache_gate_passed
    and b2_memory_gate_passed
```

The queued training chain must not continue unless this field is true.

## 7. Production and partial-audit exit behavior

Production audit now exits nonzero unless:

```text
formal_gate_passed == true
```

This includes:

- incomplete scans;
- strict original-index failures;
- corrupt or nonfinite cache tensors;
- nonzero replacement telemetry;
- missing cache references;
- final maximum rows above the verified B=2 limit.

Partial smoke is still supported, but it requires:

```yaml
cache_audit:
  allow_partial: true
  max_samples: <finite value>
  # and/or
  max_cache_files: <finite value>
```

`allow_partial=true` is rejected unless at least one explicit scan limit is
present. It cannot be applied to an unlimited production scan to bypass the
formal gate.

## 8. Missing-file reporting

The audit now maintains a distributed set of explicit missing filenames.

Each rank gathers:

```text
local_missing_referenced_files
```

The final report combines:

```text
explicit FileNotFoundError filenames
union
referenced filenames absent from the enumerated cache directory
```

The report therefore provides actionable missing artifact names rather than
only generic sample exceptions.

## 9. Tests added

Files changed or added:

```text
tests/models/minimax_h3/test_condition_cache.py
tests/test_h3_cache_audit.py
tests/test_lerobot_strict_fetch.py
```

New coverage verifies:

1. a missing cache exposes its exact filename;
2. empty `[0,5120]` embeddings are rejected;
3. NaN embeddings are rejected;
4. floating-point tags are rejected before integer conversion;
5. legal temporal-padding remap succeeds and records requested/resolved indices;
6. a real error at the padding-resolved lower sample is raised without random
   substitution;
7. ordinary training `__getitem__` retains its tolerant retry behavior while
   lower `get_strict()` remains one-shot;
8. a nonzero strict lower-fetch error count fails cache integrity;
9. `get_strict()` propagates an outer sample failure without replacement;
10. `max_rows=141` passes cache integrity but fails the B=2 gate and requires a
   memory resmoke;
11. any nonzero replacement count fails the formal cache gate.

The complete maintained test suite result is:

```text
110 passed
1 unrelated pynvml deprecation warning
```

Additional verification:

```text
python -m compileall -q scripts src tests
    PASS

git diff --check
    PASS
```

## 10. Strict partial smoke evidence

A live partial smoke was run with:

```text
max_samples = 2
max_cache_files = 4
allow_partial = true
```

Observed report fields:

```text
schema_version = 2
dataset_length = 277713
strict_getter_used = true
audited_sample_count = 2
requested_sample_count = 2
resolved_sample_count = 2
padding_remap_count = 0
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

The formal cache gate is correctly false because the smoke was intentionally
partial, even though all scanned material and the observed row length were
valid.

At the time of this smoke, the concurrently running precompute process had
increased the unique cache-file count to 53,056. This confirms that the current
process is still creating missing entries rather than merely scanning already
complete cache contents. This is not the final cache count.

## 11. Files changed across the strict cache fixes

Implementation:

- `scripts/audit_h3_condition_cache.py`
- `src/fastwam/datasets/h3_condition_cache.py`
- `src/fastwam/datasets/lerobot/base_lerobot_dataset.py`
- `src/fastwam/datasets/lerobot/robot_video_dataset.py`
- `src/fastwam/datasets/padding.py`

Tests:

- `tests/models/minimax_h3/test_condition_cache.py`
- `tests/test_h3_cache_audit.py`
- `tests/test_lerobot_strict_fetch.py`

Documentation:

- `docs/FASTWAM_H3_SCHEME_A_JOINT_GRADIENT_HANDOFF_2026-08-21.md`

## 12. What was deliberately not changed

This fix does not modify:

- Scheme A packed token layout;
- H3-to-Action asymmetric attention;
- H3 inability to read Action tokens;
- Action-loss gradient flow into H3 LoRA;
- frozen H3 base weights;
- H3 LoRA targets or rank;
- beta or either loss coefficient;
- optimizer or gradient clipping;
- 100k optimizer-step budget;
- 5,000-step warmup;
- cosine scheduler;
- B=2, accumulation=8, no-gradient-checkpointing setting;
- checkpoint schema 3;
- checkpoint-aware diagnostic behavior;
- optional `stop_after_step` infrastructure;
- three-slot H3 AdaLN baseline;
- Qwen cache schema or fingerprint derivation.

## 13. Current runtime state

At the time this document was written:

- the eight-rank Qwen cache precompute process was still running;
- all eight workers were active and CPU-bound;
- formal 100k training had not started;
- the corrected strict audit was queued to run after precompute exits;
- the queued audit command will load the latest working-tree implementation,
  including commit `08f34b1`;
- no approval is implied until the final JSON is inspected.

The production audit output path is:

```text
artifacts/h3_condition_cache_full_audit.json
```

## 14. Required final JSON before direct 100k launch

The following fields must hold:

```text
schema_version = 2
dataset_length = 277713
audited_sample_count = 277713
strict_getter_used = true
complete_file_audit = true
complete_reference_audit = true
file_errors = []
reference_errors = []
missing_referenced_files = []
replacement_count = 0
requested_sample_count = 277713
resolved_sample_count = 277713
padding_remap_count = <recorded nonnegative value>
strict_lower_fetch_error_count = 0
formal_cache_gate_passed = true
b2_memory_gate_passed = true
requires_memory_resmoke = false
formal_gate_passed = true
max_rows <= 140
```

If cache integrity passes but `max_rows > 140`, the next step is not training.
The exact reported maximum must be used for a new:

```text
B=2
8xH20
gradient_accumulation_steps=8
no gradient checkpointing
real/exact-length memory smoke
```

Only after that smoke passes may `b2_verified_max_rows` be deliberately raised
for a new formal audit/launch decision.

## 15. Direct training sequence after both gates pass

The user has explicitly chosen a direct continuous production run without
mandatory 1k/5k stop boundaries:

```text
beta = 1
batch_size = 2
gradient_accumulation_steps = 8
max_steps = 100000
stop_after_step = null
save_every = 10000
```

The run uses the native 5000-step warmup and 100k cosine horizon without
restart. `stop_after_step` remains available as a diagnostic mechanism but is
not enabled for Experiment 37 production. Checkpoint-aware diagnostics may be
run against normal saved checkpoints without gating continuation at 1k/5k.

## 16. Reviewer checklist

AI-A and AI-C should verify:

1. legal deterministic temporal-padding remap is retained;
2. both outer and lower exception-driven random substitution are bypassed;
3. the lower strict fetch performs one attempt and propagates decode/read errors;
4. requested/resolved/remap telemetry matches the effective sampling policy;
5. `strict_lower_fetch_error_count == 0` is part of cache integrity;
6. the audit refuses datasets without a strict getter;
7. `replacement_count == 0` is part of cache integrity;
8. empty and nonfinite embeddings cannot pass;
9. floating modality tags cannot be truncated into valid integer tags;
10. missing cache reports include exact filenames;
11. cache integrity and B=2 memory are distinct report fields;
12. `max_rows > 140` makes the combined formal gate false;
13. production mode exits nonzero when the combined gate is false;
14. partial mode requires both `allow_partial=true` and an explicit scan limit;
15. no model, loss, scheduler, beta, AdaLN, or checkpoint topology changed;
16. the full audit was not represented as complete before its final JSON
    exists.

## 17. Final status

Commit `08f34b1` introduced the first strict audit gate. The follow-up recorded
in this revision closes the nested `BaseLerobotDataset` retry path while
preserving legal padding remap.

```text
origin/codex/fastwam-h3-scheme-a-2026-08-19
```

The code-level nested strict audit issue identified by AI-A and AI-C is fixed.

The remaining gate is operational:

```text
finish cache generation
-> run strict 277,713-index audit
-> inspect final schema-2 JSON
-> require cache and B=2 gates
-> start the direct beta=1 100k production run
```

Formal 100k training remains unstarted.
