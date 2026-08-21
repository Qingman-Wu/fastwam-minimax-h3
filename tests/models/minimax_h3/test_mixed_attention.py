import torch

from fastwam.models.minimax_h3.action_dit import H3ActionBlock
from fastwam.models.minimax_h3.mixed_attention import (
    AsymmetricAttentionMasks,
    asymmetric_joint_attention,
    run_asymmetric_joint_block,
)
from fastwam.models.minimax_h3.video_dit import H3VideoBlock


def as_qkv(hidden):
    value = hidden.unsqueeze(2)
    return value, value, value


def run_attention(h3, state, action):
    action_stream = torch.cat((state, action), dim=1)
    h3_q, h3_k, h3_v = as_qkv(h3)
    action_q, action_k, action_v = as_qkv(action_stream)
    masks = AsymmetricAttentionMasks(
        h3_valid=torch.ones(h3.shape[:2], dtype=torch.bool),
        h3_condition=torch.tensor([[True, True, False, False]]),
        action_valid=torch.ones(action_stream.shape[:2], dtype=torch.bool),
    )
    return asymmetric_joint_attention(
        h3_q=h3_q,
        h3_k=h3_k,
        h3_v=h3_v,
        action_q=action_q,
        action_k=action_k,
        action_v=action_v,
        masks=masks,
    )


def test_action_changes_cannot_change_h3_attention_output():
    h3 = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]]]
    )
    state = torch.tensor([[[0.5, -0.5]]])
    action_a = torch.zeros(1, 2, 2)
    action_b = torch.ones(1, 2, 2) * 100.0

    h3_a, _ = run_attention(h3, state, action_a)
    h3_b, _ = run_attention(h3, state, action_b)

    assert torch.allclose(h3_a, h3_b, atol=1e-6)


def test_action_attention_reads_h3_conditions_video_and_state():
    h3 = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]]]
    )
    state = torch.tensor([[[0.5, -0.5]]])
    action = torch.tensor([[[0.2, 0.3], [-0.4, 0.1]]])

    _, baseline = run_attention(h3, state, action)
    _, changed_h3 = run_attention(h3 + 3.0, state, action)
    _, changed_state = run_attention(h3, state + 3.0, action)

    assert not torch.allclose(baseline[:, 1:], changed_h3[:, 1:])
    assert not torch.allclose(baseline[:, 1:], changed_state[:, 1:])


def test_action_reads_h3_values_without_backpropagating_into_h3():
    h3 = torch.randn(1, 4, 2, requires_grad=True)
    state = torch.randn(1, 1, 2)
    action = torch.randn(1, 2, 2, requires_grad=True)
    action_stream = torch.cat((state, action), dim=1)
    h3_q, h3_k, h3_v = as_qkv(h3)
    action_q, action_k, action_v = as_qkv(action_stream)
    masks = AsymmetricAttentionMasks(
        h3_valid=torch.ones(1, 4, dtype=torch.bool),
        h3_condition=torch.tensor([[True, True, False, False]]),
        action_valid=torch.ones(1, 3, dtype=torch.bool),
    )

    _, action_out = asymmetric_joint_attention(
        h3_q=h3_q,
        h3_k=h3_k,
        h3_v=h3_v,
        action_q=action_q,
        action_k=action_k,
        action_v=action_v,
        masks=masks,
        detach_h3_for_action=True,
    )
    action_out.square().mean().backward()

    assert h3.grad is None
    assert action.grad is not None


def test_action_loss_backpropagates_into_h3_by_default():
    h3 = torch.randn(1, 4, 2, requires_grad=True)
    state = torch.randn(1, 1, 2)
    action = torch.randn(1, 2, 2, requires_grad=True)
    action_stream = torch.cat((state, action), dim=1)
    h3_q, h3_k, h3_v = as_qkv(h3)
    action_q, action_k, action_v = as_qkv(action_stream)
    masks = AsymmetricAttentionMasks(
        h3_valid=torch.ones(1, 4, dtype=torch.bool),
        h3_condition=torch.tensor([[True, True, False, False]]),
        action_valid=torch.ones(1, 3, dtype=torch.bool),
    )

    _, action_out = asymmetric_joint_attention(
        h3_q=h3_q,
        h3_k=h3_k,
        h3_v=h3_v,
        action_q=action_q,
        action_k=action_k,
        action_v=action_v,
        masks=masks,
    )
    action_out.square().mean().backward()

    assert h3.grad is not None
    assert torch.count_nonzero(h3.grad) > 0
    assert action.grad is not None


def test_state_query_cannot_read_noisy_video_or_noisy_action():
    condition = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    video_a = torch.zeros(1, 2, 2)
    video_b = torch.ones(1, 2, 2) * 100.0
    state = torch.tensor([[[0.5, -0.5]]])
    action_a = torch.zeros(1, 2, 2)
    action_b = torch.ones(1, 2, 2) * 100.0

    _, stream_a = run_attention(
        torch.cat((condition, video_a), dim=1), state, action_a
    )
    _, stream_b = run_attention(
        torch.cat((condition, video_b), dim=1), state, action_b
    )

    assert torch.allclose(stream_a[:, :1], stream_b[:, :1], atol=1e-6)
    assert not torch.allclose(stream_a[:, 1:], stream_b[:, 1:])


def test_padding_keys_are_invisible_to_both_experts():
    h3 = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [999.0, 999.0]]]
    )
    action_stream = torch.tensor(
        [[[0.5, -0.5], [0.2, 0.3], [999.0, 999.0]]]
    )
    h3_q, h3_k, h3_v = as_qkv(h3)
    action_q, action_k, action_v = as_qkv(action_stream)
    masks = AsymmetricAttentionMasks(
        h3_valid=torch.tensor([[True, True, True, False]]),
        h3_condition=torch.tensor([[True, True, False, False]]),
        action_valid=torch.tensor([[True, True, False]]),
    )

    h3_out, action_out = asymmetric_joint_attention(
        h3_q=h3_q,
        h3_k=h3_k,
        h3_v=h3_v,
        action_q=action_q,
        action_k=action_k,
        action_v=action_v,
        masks=masks,
    )

    assert torch.isfinite(h3_out[:, :3]).all()
    assert torch.isfinite(action_out[:, :2]).all()
    assert h3_out[:, :3].abs().max() < 10
    assert action_out[:, :2].abs().max() < 10


def test_aligned_blocks_preserve_h3_isolation_and_update_action():
    torch.manual_seed(11)
    h3_block = H3VideoBlock(
        hidden_size=8,
        num_attention_heads=2,
        attention_head_dim=8,
        ffn_hidden_size=16,
        time_embed_dim=8,
        norm_eps=1e-5,
        qk_norm_eps=1e-5,
    ).eval()
    action_block = H3ActionBlock(
        hidden_size=8,
        num_attention_heads=2,
        attention_head_dim=8,
        ffn_hidden_size=16,
        time_embed_dim=8,
        norm_eps=1e-5,
        qk_norm_eps=1e-5,
    ).eval()
    h3 = torch.randn(1, 4, 8)
    action_a = torch.randn(1, 3, 8)
    action_b = action_a + 10.0
    masks = AsymmetricAttentionMasks(
        h3_valid=torch.ones(1, 4, dtype=torch.bool),
        h3_condition=torch.tensor([[True, True, False, False]]),
        action_valid=torch.ones(1, 3, dtype=torch.bool),
    )
    kwargs = {
        "h3_block": h3_block,
        "action_block": action_block,
        "h3_time_embedding": torch.randn(1, 8),
        "h3_combined_indices": torch.zeros(1, 4, dtype=torch.long),
        "action_time_embedding": torch.randn(1, 8),
        "h3_rope_freqs": torch.zeros(1, 4, 6),
        "action_rope_freqs": torch.zeros(1, 3, 6),
        "action_target_mask": torch.tensor([[False, True, True]]),
        "masks": masks,
    }

    h3_a, updated_a = run_asymmetric_joint_block(
        h3_hidden=h3, action_hidden=action_a, **kwargs
    )
    h3_b, updated_b = run_asymmetric_joint_block(
        h3_hidden=h3, action_hidden=action_b, **kwargs
    )

    assert torch.allclose(h3_a, h3_b, atol=1e-6)
    assert not torch.allclose(updated_a, updated_b)
