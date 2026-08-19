import pytest
import torch

from fastwam.models.minimax_h3.packed_sequence import (
    action_mm_position_ids,
    build_batch_cu_seqlens,
    build_h3_packed_sample,
    h3_temporal_positions,
    state_mm_position_ids,
)


def test_scheme_a_layout_keeps_keyframe_and_full_video_disjoint():
    packed = build_h3_packed_sample(
        qwen_tags=torch.tensor([1, 0, 0]),
        latent_t=2,
        latent_h=14,
        latent_w=28,
        keyframe_count=1,
    )

    assert packed.keyframe_indices.numel() == 98
    assert packed.video_target_indices.numel() == 196
    assert not torch.isin(
        packed.keyframe_indices, packed.video_target_indices
    ).any()
    assert packed.video_loss_mask[packed.video_target_indices].all()
    assert not packed.video_loss_mask[packed.keyframe_indices].any()


def test_packed_layout_has_only_h3_text_and_video_tags():
    packed = build_h3_packed_sample(
        qwen_tags=torch.tensor([1, 0, 1]),
        latent_t=1,
        latent_h=4,
        latent_w=4,
        keyframe_count=1,
    )

    assert packed.sequence_length == 3 + 4 + 4
    assert packed.text_indices.tolist() == [0, 1, 2]
    assert packed.keyframe_indices.tolist() == [3, 4, 5, 6]
    assert packed.video_target_indices.tolist() == [7, 8, 9, 10]
    assert set(packed.token_tags.tolist()) == {0, 1}
    assert packed.cu_seqlens.tolist() == [0, 11]


def test_packed_layout_rejects_audio_and_invalid_qwen_tags_by_construction():
    with pytest.raises(ValueError, match="Qwen tags"):
        build_h3_packed_sample(
            qwen_tags=torch.tensor([1, 2]),
            latent_t=1,
            latent_h=4,
            latent_w=4,
            keyframe_count=1,
        )


def test_batch_cu_seqlens_uses_real_lengths_without_padding_segment():
    cu = build_batch_cu_seqlens([11, 7, 19])

    assert cu.dtype == torch.int32
    assert cu.tolist() == [0, 11, 18, 37]


def test_h3_temporal_positions_follow_native_five_token_pattern():
    positions = h3_temporal_positions(6, origin=10.0)

    assert torch.allclose(
        positions,
        torch.tensor(
            [10.0, 35.0 / 3.0, 55.0 / 3.0, 25.0, 95.0 / 3.0, 115.0 / 3.0],
            dtype=torch.float64,
        ),
    )


def test_action_positions_share_h3_mm_rope_clock():
    positions = action_mm_position_ids(
        action_length=3,
        text_origin=7,
        video_fps=24.0,
        action_fps=8.0,
    )

    assert positions.dtype == torch.float64
    assert torch.equal(positions[:, 1:], torch.zeros(3, 2, dtype=torch.float64))
    assert torch.allclose(
        positions[:, 0], torch.tensor([7.0, 12.0, 17.0], dtype=torch.float64)
    )


def test_action_positions_prefer_explicit_timestamps():
    positions = action_mm_position_ids(
        action_length=3,
        text_origin=4,
        video_fps=24.0,
        action_fps=8.0,
        action_timestamps=torch.tensor([0.0, 0.25, 0.75]),
    )

    assert torch.allclose(
        positions[:, 0], torch.tensor([4.0, 14.0, 34.0], dtype=torch.float64)
    )


def test_state_position_is_aligned_with_current_frame():
    assert torch.equal(
        state_mm_position_ids(batch_size=2, text_origin=torch.tensor([3, 9])),
        torch.tensor([[[3.0, 0.0, 0.0]], [[9.0, 0.0, 0.0]]], dtype=torch.float64),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"latent_t": 0}, "latent_t"),
        ({"latent_h": 5}, "divisible"),
        ({"latent_w": 5}, "divisible"),
        ({"keyframe_count": 0}, "keyframe_count"),
    ],
)
def test_packed_layout_rejects_invalid_latent_geometry(kwargs, message):
    values = {
        "qwen_tags": torch.tensor([1]),
        "latent_t": 1,
        "latent_h": 4,
        "latent_w": 4,
        "keyframe_count": 1,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        build_h3_packed_sample(**values)
