"""Compare one FastWAM H3 block with pinned Diffusers MiniMax-H3."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3RotaryPosEmbed,
    MiniMaxH3TransformerBlock,
)
from fastwam.models.minimax_h3.video_dit import H3RoPE, H3VideoBlock


def main() -> None:
    torch.manual_seed(20260822)
    hidden_size = 16
    heads = 2
    head_dim = 8
    ffn_dim = 24
    time_dim = 10
    official = MiniMaxH3TransformerBlock(
        hidden_size=hidden_size,
        num_attention_heads=heads,
        attention_head_dim=head_dim,
        ffn_dim=ffn_dim,
        time_embed_dim=time_dim,
        norm_eps=1e-5,
        qk_norm_eps=1e-5,
    )
    local = H3VideoBlock(
        hidden_size=hidden_size,
        num_attention_heads=heads,
        attention_head_dim=head_dim,
        ffn_hidden_size=ffn_dim,
        time_embed_dim=time_dim,
        norm_eps=1e-5,
        qk_norm_eps=1e-5,
    )
    with torch.no_grad():
        local.norm1.weight.copy_(official.norm1.weight)
        local.norm2.weight.copy_(official.norm2.weight)
        local.attn.q_norm.weight.copy_(official.attn.norm_q.weight)
        local.attn.k_norm.weight.copy_(official.attn.norm_k.weight)
        local.attn.out_proj.weight.copy_(official.attn.to_out[0].weight)
        local.adaln_proj.linear.load_state_dict(
            official.adaln_proj.linear.state_dict()
        )
        q, k, v = [
            weight.view(heads, head_dim, hidden_size)
            for weight in (
                official.attn.to_q.weight,
                official.attn.to_k.weight,
                official.attn.to_v.weight,
            )
        ]
        local.attn.qkv_proj.weight.copy_(
            torch.stack((q, k, v), dim=1).reshape(
                3 * heads * head_dim, hidden_size
            )
        )
        official_up, official_gate = official.ff.net[0].proj.weight.chunk(
            2, dim=0
        )
        local.mlp.fc1.weight.copy_(
            torch.cat((official_gate, official_up), dim=0)
        )
        local.mlp.fc2.weight.copy_(official.ff.net[2].weight)

    batch, sequence, timesteps = 1, 7, 2
    hidden = torch.randn(batch, sequence, hidden_size)
    time_embedding = torch.randn(timesteps, time_dim)
    indices = torch.tensor([0, 1, 3, 4, 5, 2, 0], dtype=torch.long)
    position_ids = torch.randint(0, 8, (sequence, 3)).float()
    official_rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=1)
    local_rope = H3RoPE(1)
    with torch.no_grad():
        local_rope.inv_freq.copy_(official_rope.inv_freq)
    official.attn.processor._attention_backend = "native"

    with torch.no_grad():
        expected = official(
            hidden,
            time_embedding,
            indices,
            official_rope(position_ids),
        )
        q, k, v, modulation = local.pre_attention(
            hidden,
            time_embedding,
            local_rope(position_ids),
            indices.unsqueeze(0),
        )
        attended = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        ).transpose(1, 2)
        actual = local.post_attention(hidden, attended, modulation)

    difference = (actual - expected).abs()
    print(
        f"shape={tuple(actual.shape)} "
        f"max_abs_diff={float(difference.max()):.9g} "
        f"mean_abs_diff={float(difference.mean()):.9g}"
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    print("official_single_block_parity=PASS")


if __name__ == "__main__":
    main()
