# FastWAM-H3 Scheme A Design Specification

## Status

Approved on 2026-08-19. This document is the implementation baseline. The old
single-frame visual-cache/action-only implementation is not the target design.

## Training contract

Each sample provides a full video F=[f0,...,fT], an instruction, the state
aligned with f0, and an action target. The first frame follows two independent
condition paths:

1. f0 is encoded through the H3 video VAE image path with process_image=True,
   patchified, and used as a near-clean keyframe condition.
2. f0 and the instruction are encoded by H3's native Qwen3-VL presentation.
   Qwen vision tokens keep H3 tag 0 and text tokens keep H3 tag 1.

The complete video is encoded exactly once by the H3 video VAE. Every latent
position, including the first temporal latent, is noised and supervised by the
video flow-matching loss. The keyframe condition never replaces the first
full-video latent and never contributes to video loss.

State follows exactly this action-stream path:

    state -> normalize -> StateEncoder ->
    [state condition token | noisy action tokens] -> ActionExpert

The state token is not noised, has no diffusion loss, does not pass through
Qwen, does not receive an H3 modality tag, and is excluded from the action
output head.

## Packed layout

The logical sample layout is:

    H3 stream:
      [Qwen text/vision condition | near-clean keyframe condition |
       noisy full-video target]

    Action stream:
      [state condition token | noisy action target]

Audio length is always zero. No placeholder audio tensor or zero audio token is
created. Structural padding, when required by a kernel, is an isolated varlen
segment and is excluded from output and loss masks.

## Per-layer interaction

The H3 Video Expert and Action Expert have aligned layers and compatible
56-head by 128-dimension Q/K/V geometry. Visibility is asymmetric:

| Query | Qwen | Keyframe | Noisy video | State | Noisy action |
|---|---:|---:|---:|---:|---:|
| H3 | yes | yes | yes | no | no |
| State | yes | yes | no | yes | no |
| Action | yes | yes | yes | yes | yes |

H3 queries never include state/action keys, so introducing the Action Expert
does not change H3's attention softmax denominator. Action-to-H3 feedback is a
later gated ablation, not a version-one feature.

## H3-native conditioning

Qwen hidden states have width 5120 and must pass through H3 condition_proj
(5120 to 5376) and the two-layer H3 TokenRefiner before H3 blocks. Keyframe and
video patch rows have width 96 for a 24-channel latent and patch (1,2,2), then
pass through video_patch_proj (96 to 5376).

The default keyframe augmentation follows the H3 FL2VA pretrained distribution:

    z_keyframe_aug = 0.999 * z_keyframe + 0.001 * epsilon

This row is fixed across the denoising trajectory and carries no loss. Strict
1.0 clean conditioning is an ablation.

## Time and position contract

One base denoising progress u is sampled per sample. Video and action use
separate shifted schedules:

    sigma_video = phi(u, 12.0)
    sigma_action = phi(u, action_shift)

The two experts use separate timestep embeddings and AdaLN modulation. For a
clean target x0 and noise epsilon:

    x_sigma = (1 - sigma) * x0 + sigma * epsilon
    velocity_target = epsilon - x0

Action and state positions use H3 MM-RoPE rather than a standalone 1D RoPE.
H3 rotates 96 of each 128 head dimensions and passes the remaining 32 through.
State is located at the current-frame time. Action times are mapped from their
real timestamps, or from i * video_fps / action_fps when timestamps are absent.
For the current 32-action/five-frame clips, eight action rows occupy each
visual interval, so H3's 24-fps RoPE clock uses an effective action rate of
192 rather than the physical controller rate. The result is then scaled by
H3's 5/3 temporal position convention.

## Losses

Video loss covers every valid full-video target patch and no condition row.
Action loss covers every valid action time/dimension and excludes the state row.
Both losses are normalized by their own valid-element counts before weighting:

    loss = lambda_video * loss_video + lambda_action * loss_action

## Inference contract

Inference receives f0, instruction, state, random video noise, and random action
noise. Video and action are jointly denoised at every step using shared progress
and separate sigma schedules. Action is the primary output; decoded full video
is auxiliary. Because Scheme A predicts the entire video latent, the decoded
first frame is generated and need not match f0 pixel-for-pixel. Version one has
no static video KV cache or action-only fast path.

## Shape invariants

H3-native frame counts are 5+17k and spatial dimensions are divisible by 32.
For a 5x224x448 video:

- f0 image latent: [B,24,1,14,28], producing 98 keyframe rows.
- full-video latent: [B,24,2,14,28], producing 196 target rows.

These are distinct token regions.

## Out of scope for version one

- Future-only video targets.
- Replacing the first video latent with the keyframe latent.
- ActionSynth-style action injection into Video AdaLN.
- State tokens inside Qwen or H3 modality indices.
- Audio placeholders or audio loss.
- Bidirectional Action-to-H3 attention.
- Action-only inference with a static H3 cache.
