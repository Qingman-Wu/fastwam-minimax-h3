"""Packed-sequence metadata for FastWAM-H3 Scheme A.

This module contains no model weights.  It is the single source of truth for
the logical H3 row layout and for mapping robot action time onto H3 MM-RoPE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


H3_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
H3_FRAME_RESCALE = 5.0 / 3.0
H3_SPATIAL_INTERPOLATION = 32.0
H3_VIDEO_TAG = 0
H3_TEXT_TAG = 1


@dataclass(frozen=True)
class H3PackedSample:
    """Indices and metadata for one unpadded H3 sample."""

    text_indices: torch.Tensor
    keyframe_indices: torch.Tensor
    video_target_indices: torch.Tensor
    token_tags: torch.Tensor
    position_ids: torch.Tensor
    video_loss_mask: torch.Tensor
    sequence_length: int
    cu_seqlens: torch.Tensor


def build_batch_cu_seqlens(lengths: Sequence[int]) -> torch.Tensor:
    """Return int32 cumulative lengths without creating padding segments."""

    if not lengths:
        raise ValueError("lengths must contain at least one sample")
    normalized = [int(length) for length in lengths]
    if any(length <= 0 for length in normalized):
        raise ValueError(f"all sequence lengths must be positive, got {normalized}")
    return torch.tensor(
        [0, *torch.tensor(normalized, dtype=torch.int64).cumsum(0).tolist()],
        dtype=torch.int32,
    )


def h3_temporal_positions(
    length: int,
    *,
    origin: float = 0.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build H3's native 1/4/4/4/4 temporal grid with 5/3 rescaling."""

    length = int(length)
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    spans = torch.tensor(
        [
            H3_FRAME_RESCALE * H3_FRAME_PER_TOKEN[index % len(H3_FRAME_PER_TOKEN)]
            for index in range(length)
        ],
        dtype=torch.float64,
        device=device,
    )
    starts = torch.cat((torch.zeros(1, dtype=torch.float64, device=device), spans[:-1]))
    return starts.cumsum(0) + float(origin)


def _spatial_position_grid(
    latent_h: int,
    latent_w: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    sqrt_area = float(latent_h * latent_w) ** 0.5

    def axis(dim: int) -> torch.Tensor:
        ratio = dim / sqrt_area
        left = (1.0 - ratio) * 0.5
        count = dim // 2
        return (
            left
            + torch.arange(count, dtype=torch.float64, device=device) * ratio / count
        ) * H3_SPATIAL_INTERPOLATION

    height, width = torch.meshgrid(axis(latent_h), axis(latent_w), indexing="ij")
    return torch.stack((height.reshape(-1), width.reshape(-1)), dim=-1)


def build_h3_packed_sample(
    *,
    qwen_tags: torch.Tensor,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    keyframe_count: int = 1,
) -> H3PackedSample:
    """Build [Qwen | keyframe | full-video target] metadata for one sample."""

    if qwen_tags.ndim != 1 or qwen_tags.numel() == 0:
        raise ValueError("Qwen tags must be a non-empty one-dimensional tensor")
    qwen_tags = qwen_tags.to(dtype=torch.long)
    if not torch.logical_or(qwen_tags == H3_VIDEO_TAG, qwen_tags == H3_TEXT_TAG).all():
        raise ValueError("Qwen tags must contain only H3 video tag 0 or text tag 1")

    latent_t = int(latent_t)
    latent_h = int(latent_h)
    latent_w = int(latent_w)
    keyframe_count = int(keyframe_count)
    if latent_t <= 0:
        raise ValueError(f"latent_t must be positive, got {latent_t}")
    if latent_h <= 0 or latent_w <= 0 or latent_h % 2 or latent_w % 2:
        raise ValueError(
            f"latent_h and latent_w must be positive and divisible by 2, "
            f"got {(latent_h, latent_w)}"
        )
    if keyframe_count != 1:
        raise ValueError(
            f"Scheme A requires keyframe_count=1 for f0, got {keyframe_count}"
        )

    device = qwen_tags.device
    text_length = int(qwen_tags.numel())
    frame_rows = (latent_h // 2) * (latent_w // 2)
    keyframe_rows = keyframe_count * frame_rows
    video_rows = latent_t * frame_rows
    sequence_length = text_length + keyframe_rows + video_rows

    text_indices = torch.arange(text_length, device=device, dtype=torch.long)
    keyframe_indices = torch.arange(
        text_length,
        text_length + keyframe_rows,
        device=device,
        dtype=torch.long,
    )
    video_target_indices = torch.arange(
        text_length + keyframe_rows,
        sequence_length,
        device=device,
        dtype=torch.long,
    )

    token_tags = torch.full(
        (sequence_length,), H3_VIDEO_TAG, device=device, dtype=torch.long
    )
    token_tags[text_indices] = qwen_tags

    position_ids = torch.zeros(
        (sequence_length, 3), device=device, dtype=torch.float64
    )
    position_ids[text_indices, 0] = torch.arange(
        text_length, device=device, dtype=torch.float64
    )
    spatial = _spatial_position_grid(latent_h, latent_w, device=device)
    position_ids[keyframe_indices, 0] = float(text_length)
    position_ids[keyframe_indices, 1:] = spatial.repeat(keyframe_count, 1)

    temporal = h3_temporal_positions(
        latent_t, origin=float(text_length), device=device
    )
    position_ids[video_target_indices, 0] = temporal.repeat_interleave(frame_rows)
    position_ids[video_target_indices, 1:] = spatial.repeat(latent_t, 1)

    video_loss_mask = torch.zeros(
        sequence_length, device=device, dtype=torch.bool
    )
    video_loss_mask[video_target_indices] = True

    return H3PackedSample(
        text_indices=text_indices,
        keyframe_indices=keyframe_indices,
        video_target_indices=video_target_indices,
        token_tags=token_tags,
        position_ids=position_ids,
        video_loss_mask=video_loss_mask,
        sequence_length=sequence_length,
        cu_seqlens=build_batch_cu_seqlens([sequence_length]).to(device),
    )


def action_mm_position_ids(
    *,
    action_length: int,
    text_origin: int | float,
    video_fps: float,
    action_fps: float,
    action_timestamps: torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Map action times to the temporal axis of H3's three-axis MM-RoPE."""

    action_length = int(action_length)
    if action_length <= 0:
        raise ValueError(f"action_length must be positive, got {action_length}")
    if float(video_fps) <= 0 or float(action_fps) <= 0:
        raise ValueError(
            f"video_fps and action_fps must be positive, got "
            f"{(video_fps, action_fps)}"
        )
    if action_timestamps is None:
        timestamps = (
            torch.arange(action_length, dtype=torch.float64, device=device)
            / float(action_fps)
        )
    else:
        if action_timestamps.ndim != 1 or action_timestamps.numel() != action_length:
            raise ValueError(
                f"action_timestamps must have shape [{action_length}], got "
                f"{tuple(action_timestamps.shape)}"
            )
        timestamps = action_timestamps.to(device=device, dtype=torch.float64)
    positions = torch.zeros((action_length, 3), dtype=torch.float64, device=device)
    positions[:, 0] = (
        float(text_origin)
        + H3_FRAME_RESCALE * timestamps * float(video_fps)
    )
    return positions


def state_mm_position_ids(
    *,
    batch_size: int,
    text_origin: int | float | torch.Tensor,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Place one state condition row at each sample's current-frame origin."""

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    origin = torch.as_tensor(text_origin, dtype=torch.float64, device=device)
    if origin.ndim == 0:
        origin = origin.expand(batch_size)
    if origin.shape != (batch_size,):
        raise ValueError(
            f"text_origin must be scalar or [{batch_size}], got {tuple(origin.shape)}"
        )
    positions = torch.zeros((batch_size, 1, 3), dtype=torch.float64, device=device)
    positions[:, 0, 0] = origin
    return positions
