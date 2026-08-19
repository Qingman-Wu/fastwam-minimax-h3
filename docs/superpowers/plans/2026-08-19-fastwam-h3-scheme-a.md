# FastWAM-H3 Scheme A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Replace the current H3 single-frame action-only cache with Scheme A full-video joint flow matching, H3-native first-frame dual conditioning, and a state-prefixed independent Action Expert.

**Architecture:** H3 owns one packed stream containing Qwen text/vision, near-clean keyframe, and noisy full-video target rows. A width-reduced Action Expert owns one state row followed by noisy action rows. Aligned layers use asymmetric joint attention so Action reads H3 while H3 remains numerically isolated from state/action.

**Tech Stack:** Python 3.10+, PyTorch 2.7, scaled-dot-product attention, safetensors, Hugging Face Qwen3-VL processor/model, Hydra/OmegaConf, pytest.

**Spec:** docs/superpowers/specs/2026-08-19-fastwam-h3-scheme-a.md

## Global Constraints

- Full-video target includes every H3 temporal latent, including the first.
- Default keyframe augmentation is 0.999 clean plus 0.001 noise.
- H3 tags are only video/vision 0 and text 1; state/action never enter H3 combined_indices.
- Audio sequence length is exactly zero and no zero audio tensor is constructed.
- H3 queries cannot see state/action keys in version one.
- State input is exactly one condition token prefixed to noisy action tokens.
- Video scheduler shift defaults to 12.0 and action shift defaults to 5.0.
- Action positions use H3 MM-RoPE with 96 rotated and 32 pass-through dimensions.
- All output recovery and losses use explicit indices/masks, never implicit padded slices.

---

### Task 1: Packed sequence and MM-RoPE contracts

**Files:**
- Create: src/fastwam/models/minimax_h3/packed_sequence.py
- Test: tests/models/minimax_h3/test_packed_sequence.py

**Interfaces:**
- Produces: H3TokenType, H3PackedSample, build_h3_packed_sample(), build_batch_cu_seqlens(), h3_temporal_positions(), action_mm_position_ids().
- Consumes: Literal Qwen embeddings/tags, keyframe/video row counts, latent grids, video/action rates, and optional valid lengths.

- [x] **Step 1: Write failing layout tests**

    def test_scheme_a_layout_keeps_keyframe_and_full_video_disjoint():
        packed = build_h3_packed_sample(
            qwen_tags=torch.tensor([1, 0, 0]),
            latent_t=2, latent_h=14, latent_w=28,
            keyframe_count=1,
        )
        assert packed.keyframe_indices.numel() == 98
        assert packed.video_target_indices.numel() == 196
        assert not torch.isin(
            packed.keyframe_indices, packed.video_target_indices
        ).any()
        assert packed.video_loss_mask[packed.video_target_indices].all()
        assert not packed.video_loss_mask[packed.keyframe_indices].any()

    def test_action_positions_share_h3_mm_rope_clock():
        pos = action_mm_position_ids(
            action_length=3, text_origin=7,
            video_fps=24.0, action_fps=8.0,
        )
        assert torch.equal(pos[:, 1:], torch.zeros(3, 2, dtype=pos.dtype))
        assert torch.allclose(pos[:, 0], torch.tensor([7.0, 12.0, 17.0]))

- [x] **Step 2: Run tests and verify RED**

    .venv/bin/python -m pytest tests/models/minimax_h3/test_packed_sequence.py -q

  Expected: import failure because packed_sequence.py does not exist.

- [x] **Step 3: Implement immutable metadata builders**

  H3PackedSample stores text_indices, keyframe_indices, video_target_indices,
  token_tags, position_ids, video_loss_mask, sequence_length, and cu_seqlens.
  build_h3_packed_sample lays out exactly text, keyframe, full-video target with
  no audio region and validates latent_h/latent_w divisibility by patch (2,2).

- [x] **Step 4: Verify GREEN and boundary cases**

    .venv/bin/python -m pytest tests/models/minimax_h3/test_packed_sequence.py -q

  Add literal assertions for 5x224x448 (98/196), two-sample cu_seqlens, invalid
  tags, invalid rates, and action timestamp overrides.

- [x] **Step 5: Commit**

    git add src/fastwam/models/minimax_h3/packed_sequence.py tests/models/minimax_h3/test_packed_sequence.py
    git commit -m "Add H3 Scheme A packing contracts"

### Task 2: H3-native text and keyframe conditioning

**Files:**
- Create: src/fastwam/models/minimax_h3/text_encoder.py
- Modify: src/fastwam/models/minimax_h3/video_vae.py
- Modify: src/fastwam/models/minimax_h3/video_dit.py
- Test: tests/models/minimax_h3/test_conditioning.py

**Interfaces:**
- Produces: MiniMaxH3TextConditioner.encode(images, instructions), augment_keyframe_latents(), MiniMaxH3VideoBackbone.embed_h3_stream().
- Consumes: H3 processor/text_encoder directories, Qwen input tensors, H3 condition_proj and TokenRefiner weights, VAE image encoding.

- [x] **Step 1: Write failing condition tests**

    def test_keyframe_augmentation_uses_h3_native_ratio():
        clean = torch.full((1, 24, 1, 2, 2), 2.0)
        noise = torch.full_like(clean, -3.0)
        got = augment_keyframe_latents(clean, noise, strength=0.999)
        assert torch.allclose(got, torch.full_like(clean, 1.995))

    def test_text_and_vision_tags_survive_condition_projection(tiny_video_dit):
        qwen = torch.randn(3, tiny_video_dit.text_dim)
        tags = torch.tensor([1, 0, 1])
        embedded = tiny_video_dit.refine_text_condition(
            qwen, tags, torch.tensor([0, 3], dtype=torch.int32)
        )
        assert embedded.shape == (3, tiny_video_dit.hidden_size)
        assert torch.equal(tags, torch.tensor([1, 0, 1]))

- [x] **Step 2: Run and verify RED**

    .venv/bin/python -m pytest tests/models/minimax_h3/test_conditioning.py -q

- [x] **Step 3: Add Qwen3-VL wrapper with lazy imports**

  The wrapper loads AutoProcessor and the H3 Qwen3-VL model only when requested,
  builds an FL2VA first-frame presentation, returns flattened hidden states and
  literal tags 0/1, and supports precomputed prompt_embeds/tags for training.
  It rejects context tensors with the old 4096-wide Wan contract.

- [x] **Step 4: Restore H3 condition projection and two-layer TokenRefiner**

  MiniMaxH3VideoBackbone gains condition_proj and token_refiner modules whose
  checkpoint names and shapes match released H3 weights. Audio projection/head
  remain absent from the instantiated model and are omitted by the loader.

- [x] **Step 5: Make image encoding use the native process_image route**

  MiniMaxH3VAEAdapter.encode_image calls encode_video(..., process_image=True)
  when the released wrapper exposes it, and validates one temporal latent.

- [x] **Step 6: Verify GREEN and commit**

    .venv/bin/python -m pytest tests/models/minimax_h3/test_conditioning.py -q
    git add src/fastwam/models/minimax_h3/text_encoder.py src/fastwam/models/minimax_h3/video_vae.py src/fastwam/models/minimax_h3/video_dit.py tests/models/minimax_h3/test_conditioning.py
    git commit -m "Add H3 native first-frame conditioning"

### Task 3: Independent state-prefixed Action Expert

**Files:**
- Modify: src/fastwam/models/minimax_h3/action_dit.py
- Test: tests/models/minimax_h3/test_action_expert.py

**Interfaces:**
- Produces: ActionExpertState, H3ActionDiT.pre_dit(actions, state, timestep, positions), H3ActionDiT.post_dit(tokens).
- Consumes: normalized state [B,Ds], noisy actions [B,N,Da], action sigma/timestep, H3 layer Q/K/V.

- [x] **Step 1: Write failing prefix/output tests**

    def test_state_is_one_prefix_row_and_head_returns_only_actions(tiny_action_dit):
        state = torch.randn(2, 5)
        action = torch.randn(2, 4, 3)
        prepared = tiny_action_dit.pre_dit(action, state, torch.tensor([0.2, 0.8]))
        assert prepared.tokens.shape[:2] == (2, 5)
        output = tiny_action_dit.post_dit(prepared.tokens)
        assert output.shape == (2, 4, 3)

    def test_state_row_is_not_action_timestep_modulated(tiny_action_dit):
        state = torch.randn(1, 5)
        action = torch.randn(1, 2, 3)
        low = tiny_action_dit.pre_dit(action, state, torch.tensor([0.1])).tokens[:, 0]
        high = tiny_action_dit.pre_dit(action, state, torch.tensor([0.9])).tokens[:, 0]
        assert torch.allclose(low, high)

- [x] **Step 2: Run and verify RED**

    .venv/bin/python -m pytest tests/models/minimax_h3/test_action_expert.py -q

- [x] **Step 3: Replace context/proprio concatenation with StateEncoder**

  H3ActionDiT receives state_dim and owns a two-layer StateEncoder to hidden_size.
  pre_dit concatenates one state token before action_encoder output. It returns a
  boolean state mask and action output indices. post_dit slices only action rows.

- [x] **Step 4: Apply action AdaLN only to action rows**

  H3ActionBlock computes modulation from the action timestep, updates action
  rows with it, and leaves the state row unmodulated except for normal residual
  attention/MLP processing. No separate State AdaLN is introduced.

- [x] **Step 5: Replace standalone action RoPE with explicit H3 MM positions**

  pre_dit accepts [B,1+N,3] positions and uses the same MiniMaxH3Rope frequency
  construction as the video expert.

- [x] **Step 6: Verify GREEN and commit**

    .venv/bin/python -m pytest tests/models/minimax_h3/test_action_expert.py -q
    git add src/fastwam/models/minimax_h3/action_dit.py tests/models/minimax_h3/test_action_expert.py
    git commit -m "Add state-prefixed H3 Action Expert"

### Task 4: Asymmetric per-layer joint attention

**Files:**
- Create: src/fastwam/models/minimax_h3/mixed_attention.py
- Modify: src/fastwam/models/minimax_h3/video_dit.py
- Modify: src/fastwam/models/minimax_h3/action_dit.py
- Test: tests/models/minimax_h3/test_mixed_attention.py

**Interfaces:**
- Produces: asymmetric_joint_attention(h3_qkv, action_qkv, regions), H3/Action aligned block runner.
- Consumes: H3 rows and state/action rows with explicit region boundaries.

- [x] **Step 1: Write failing visibility tests**

    def test_action_changes_cannot_change_h3_output(tiny_joint_layer):
        h3 = torch.randn(1, 6, 8)
        state = torch.randn(1, 1, 4)
        a0 = torch.zeros(1, 2, 4)
        a1 = torch.ones(1, 2, 4) * 100
        h3_a, _ = tiny_joint_layer(h3, state, a0)
        h3_b, _ = tiny_joint_layer(h3, state, a1)
        assert torch.allclose(h3_a, h3_b, atol=1e-6)

    def test_action_output_reads_video_and_state(tiny_joint_layer):
        h3 = torch.randn(1, 6, 8)
        state = torch.randn(1, 1, 4)
        action = torch.randn(1, 2, 4)
        _, baseline = tiny_joint_layer(h3, state, action)
        _, changed = tiny_joint_layer(h3 + 1, state + 1, action)
        assert not torch.allclose(baseline, changed)

- [x] **Step 2: Run and verify RED**

    .venv/bin/python -m pytest tests/models/minimax_h3/test_mixed_attention.py -q

- [x] **Step 3: Implement three query visibility paths**

  H3 query attends only H3 keys. State query attends Qwen/keyframe plus itself.
  Action query attends Qwen/keyframe/noisy-video/state/action keys. Masks are
  derived from explicit indices and support a per-sample valid-length mask.

- [x] **Step 4: Run aligned H3/Action blocks without a video KV cache**

  Each H3 block computes its own native attention and update. The aligned Action
  block then reads that layer's H3 K/V and updates the state/action stream. H3
  does not read Action K/V. The forward returns both final H3 and action states.

- [x] **Step 5: Verify GREEN and commit**

    .venv/bin/python -m pytest tests/models/minimax_h3/test_mixed_attention.py -q
    git add src/fastwam/models/minimax_h3/mixed_attention.py src/fastwam/models/minimax_h3/video_dit.py src/fastwam/models/minimax_h3/action_dit.py tests/models/minimax_h3/test_mixed_attention.py
    git commit -m "Add asymmetric H3 action joint attention"

### Task 5: Scheme A training loss

**Files:**
- Rewrite: src/fastwam/models/minimax_h3/fastwam.py
- Test: tests/models/minimax_h3/test_training_contract.py

**Interfaces:**
- Produces: FastWAMH3.prepare_conditions(), prepare_noisy_targets(), training_loss().
- Consumes: video, instruction or precomputed Qwen condition, proprio masks, action masks, two schedulers.

- [x] **Step 1: Write failing end-to-end contract tests with tiny fakes**

  Tests inject tiny deterministic VAE/text/video/action components and assert:
  full video is encoded once; image is encoded separately; all video latent
  positions have loss; state has no loss; action_dim_is_pad and action_is_pad
  remove elements; the same base progress creates separate video/action sigma.

- [x] **Step 2: Run and verify RED**

    .venv/bin/python -m pytest tests/models/minimax_h3/test_training_contract.py -q

- [x] **Step 3: Implement condition and target preparation**

  prepare_conditions obtains Qwen rows/tags, image latent, 0.999 augmentation,
  and aligned normalized state. prepare_noisy_targets encodes complete video once,
  samples one base progress, maps it through both shifts, and noises full-video
  and action targets independently.

- [x] **Step 4: Implement explicit masked losses**

  video prediction is gathered only at video_target_indices and compared to
  epsilon_video-clean_video. Action prediction contains only action rows and is
  masked by time and dimension masks. Both denominators clamp at one and metrics
  report unweighted loss_video/loss_action plus total loss.

- [x] **Step 5: Verify GREEN and commit**

    .venv/bin/python -m pytest tests/models/minimax_h3/test_training_contract.py -q
    git add src/fastwam/models/minimax_h3/fastwam.py tests/models/minimax_h3/test_training_contract.py
    git commit -m "Train full H3 video and action targets"

### Task 6: Joint video/action inference

**Files:**
- Modify: src/fastwam/models/minimax_h3/fastwam.py
- Test: tests/models/minimax_h3/test_inference_contract.py

**Interfaces:**
- Produces: FastWAMH3.infer() returning action and decoded auxiliary video.
- Consumes: f0, instruction/Qwen rows, state, num_frames, action_horizon, two inference schedules.

- [x] **Step 1: Write failing inference loop tests**

  Use deterministic fake schedulers and assert every step calls both experts,
  video and action both change, no future ground-truth video is accepted, the
  keyframe/Qwen/state conditions are computed once, and decoded video is not a
  repeated input-frame placeholder.

- [x] **Step 2: Run and verify RED**

    .venv/bin/python -m pytest tests/models/minimax_h3/test_inference_contract.py -q

- [x] **Step 3: Implement synchronized dual-schedule denoising**

  Initialize complete video and action Gaussian noise. For each shared progress
  index, run the complete aligned H3/Action stack and apply each schedule's own
  sigma delta. Decode the final complete video and denormalize actions.

- [x] **Step 4: Verify GREEN and commit**

    .venv/bin/python -m pytest tests/models/minimax_h3/test_inference_contract.py -q
    git add src/fastwam/models/minimax_h3/fastwam.py tests/models/minimax_h3/test_inference_contract.py
    git commit -m "Jointly denoise H3 video and actions"

### Task 7: Runtime, config, checkpoint, and dependency wiring

**Files:**
- Modify: src/fastwam/runtime.py
- Modify: src/fastwam/models/minimax_h3/__init__.py
- Modify: configs/model/fastwam_h3.yaml
- Modify: configs/task/libero_h3_uncond_2cam224_1e-4.yaml
- Modify: pyproject.toml
- Test: tests/models/minimax_h3/test_runtime_config.py

**Interfaces:**
- Produces: create_fastwam_h3() with text/video/action scheduler and loss config validation; checkpoint schema version 2.
- Consumes: H3 FL2VA component path and data-derived state/action dimensions.

- [ ] **Step 1: Write failing config validation tests**

  Assert video_scheduler is required, lambda_video defaults to 1.0, load_text_encoder
  is not silently discarded, action hidden size is 1024, attention geometry is
  56x128, and video_attention_mask_mode defaults to bidirectional H3 packed attention.

- [ ] **Step 2: Run and verify RED**

    .venv/bin/python -m pytest tests/models/minimax_h3/test_runtime_config.py -q

- [ ] **Step 3: Wire all constructors and checkpoint state**

  Runtime passes video/action schedule settings, keyframe strength, text encoder
  toggle, state dimension, rates, and both loss weights. Checkpoints save/load
  state encoder, Action Expert, any trainable H3 adapters, optimizer, step, and
  schema metadata; they reject the old action-only schema with a clear message.

- [ ] **Step 4: Update dependencies and configs**

  Replace the incompatible transformers 4.49 pin with transformers==4.57.6,
  which exposes Qwen3VLModel and Qwen3VLConfig. Set lambda_video=1.0, lambda_action=1.0,
  keyframe_condition_strength=0.999, video shift 12.0, action shift 5.0, and
  remove comments describing a frozen action-only visual cache.

- [ ] **Step 5: Verify GREEN and commit**

    .venv/bin/python -m pytest tests/models/minimax_h3/test_runtime_config.py -q
    git add src/fastwam/runtime.py src/fastwam/models/minimax_h3/__init__.py configs/model/fastwam_h3.yaml configs/task/libero_h3_uncond_2cam224_1e-4.yaml pyproject.toml tests/models/minimax_h3/test_runtime_config.py
    git commit -m "Wire FastWAM H3 Scheme A runtime"

### Task 8: Local and RTX 5090 verification

**Files:**
- Modify only if a failing test exposes a production defect; each defect gets a failing regression test first.

**Interfaces:**
- Consumes: complete repository and target GitHub branch.
- Produces: clean test output, CUDA smoke result, and pushed commits.

- [ ] **Step 1: Run local CPU suite**

    .venv/bin/python -m pytest tests/models/minimax_h3 -q
    .venv/bin/python -m compileall -q src scripts tests

- [ ] **Step 2: Push the reviewed branch commits**

    git push minimax-h3 codex/fastwam-h3-scheme-a-2026-08-19

- [ ] **Step 3: Update the RTX 5090 checkout**

    ssh rtx5090-180 'git -C /root/codex-workspaces/fastwam-h3-scheme-a-2026-08-19 pull --ff-only'

- [ ] **Step 4: Build the remote test environment and run CPU tests**

    ssh rtx5090-180 'cd /root/codex-workspaces/fastwam-h3-scheme-a-2026-08-19 && uv venv --python 3.10 .venv && uv pip install --python .venv/bin/python -e . pytest && .venv/bin/python -m pytest tests/models/minimax_h3 -q'

- [ ] **Step 5: Run a small CUDA smoke test on an idle GPU**

  Instantiate one-layer tiny H3/Action experts in bf16, run forward/backward for
  batch two with unequal valid lengths, assert finite video/action gradients,
  and record peak allocated memory. Do not start the full 33B model until an
  idle multi-GPU allocation and actual FL2VA path are confirmed.

- [ ] **Step 6: Final mutation and spec coverage check**

  Mutate each mask/visibility boundary mentally and confirm at least one test
  fails: keyframe in loss, first video latent omitted, state output included,
  H3 seeing action, padding visible, same sigma forced, and audio row inserted.

## Self-review

- Spec coverage: every approved data path, visibility rule, scheduler, loss,
  inference behavior, shape invariant, and excluded feature maps to Tasks 1-7.
- Placeholder scan: the plan contains no deferred implementation decisions and
  pins the Qwen3-VL dependency to transformers==4.57.6.
- Type consistency: H3PackedSample indices feed both embedding and loss recovery;
  H3ActionDiT always consumes one state row plus action rows and returns actions only.
