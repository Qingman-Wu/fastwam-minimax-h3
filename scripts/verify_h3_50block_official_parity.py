"""Stream real H3 weights through local and official 50-block implementations."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3AdaLayerNormOut,
    MiniMaxH3TransformerBlock,
)
from fastwam.models.minimax_h3.video_dit import load_h3_video_backbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transformer-dir",
        default="/root/wuqingman/models/MiniMax-H3/FL2VA/transformer",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--max-relative-rms", type=float, default=0.02)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    return parser.parse_args()


@torch.no_grad()
def map_block(local, official) -> None:
    official.norm1.weight.copy_(local.norm1.weight)
    official.norm2.weight.copy_(local.norm2.weight)
    official.attn.norm_q.weight.copy_(local.attn.q_norm.weight)
    official.attn.norm_k.weight.copy_(local.attn.k_norm.weight)
    official.attn.to_out[0].weight.copy_(local.attn.out_proj.weight)
    official.adaln_proj.linear.load_state_dict(
        local.adaln_proj.linear.state_dict()
    )
    fused = local.attn.qkv_proj.weight.view(
        local.attn.num_heads,
        3,
        local.attn.head_dim,
        local.attn.qkv_proj.in_features,
    )
    official.attn.to_q.weight.copy_(fused[:, 0].flatten(0, 1))
    official.attn.to_k.weight.copy_(fused[:, 1].flatten(0, 1))
    official.attn.to_v.weight.copy_(fused[:, 2].flatten(0, 1))
    if official.attn.fused_projections:
        official.attn.to_qkv.weight.copy_(
            torch.cat(
                (
                    official.attn.to_q.weight,
                    official.attn.to_k.weight,
                    official.attn.to_v.weight,
                ),
                dim=0,
            )
        )
    local_gate, local_up = local.mlp.fc1.weight.chunk(2, dim=0)
    official.ff.net[0].proj.weight.copy_(
        torch.cat((local_up, local_gate), dim=0)
    )
    official.ff.net[2].weight.copy_(local.mlp.fc2.weight)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    local_model = load_h3_video_backbone(
        args.transformer_dir,
        device=device,
        dtype=torch.bfloat16,
    )
    official_block = MiniMaxH3TransformerBlock(
        hidden_size=local_model.hidden_size,
        num_attention_heads=local_model.num_attention_heads,
        attention_head_dim=local_model.attention_head_dim,
        ffn_dim=local_model.blocks[0].mlp.fc2.in_features,
        time_embed_dim=local_model.blocks[0].adaln_proj.linear.in_features,
        norm_eps=local_model.blocks[0].norm1.eps,
        qk_norm_eps=local_model.blocks[0].attn.q_norm.eps,
    ).to(device=device, dtype=torch.bfloat16)
    official_block.attn.fuse_projections()
    official_block.attn.processor._attention_backend = "native"

    generator = torch.Generator(device=device).manual_seed(20260822)
    local_hidden = torch.randn(
        (1, args.sequence_length, local_model.hidden_size),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    official_hidden = local_hidden.clone()
    progress = torch.tensor([0.375], device=device, dtype=torch.float32)
    time_embedding = local_model.time_embedder(progress)
    positions = torch.stack(
        (
            torch.arange(args.sequence_length, device=device),
            torch.arange(args.sequence_length, device=device) % 2,
            torch.arange(args.sequence_length, device=device) % 4,
        ),
        dim=-1,
    )
    rope_angles = local_model.rope(positions)
    official_rope = (rope_angles.cos(), rope_angles.sin())
    combined_indices = torch.zeros(
        (1, args.sequence_length), device=device, dtype=torch.long
    )
    official_indices = combined_indices[0]
    worst_block_diff = 0.0
    worst_relative_rms = 0.0
    worst_cosine = 1.0

    with torch.no_grad():
        for index, local_block in enumerate(local_model.blocks):
            map_block(local_block, official_block)
            q, k, v, modulation = local_block.pre_attention(
                local_hidden,
                time_embedding,
                rope_angles,
                combined_indices,
            )
            attended = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
            ).transpose(1, 2)
            local_hidden = local_block.post_attention(
                local_hidden, attended, modulation
            )
            official_hidden = official_block(
                official_hidden,
                time_embedding,
                official_indices,
                official_rope,
            )
            block_diff = float(
                (local_hidden.float() - official_hidden.float()).abs().max()
            )
            difference = local_hidden.float() - official_hidden.float()
            relative_rms = float(
                difference.square().mean().sqrt()
                / official_hidden.float().square().mean().sqrt().clamp(min=1e-12)
            )
            cosine = float(
                F.cosine_similarity(
                    local_hidden.float().flatten(),
                    official_hidden.float().flatten(),
                    dim=0,
                )
            )
            worst_block_diff = max(worst_block_diff, block_diff)
            worst_relative_rms = max(worst_relative_rms, relative_rms)
            worst_cosine = min(worst_cosine, cosine)
            print(
                f"block={index:02d} max_abs_diff={block_diff:.9g} "
                f"relative_rms={relative_rms:.9g} cosine={cosine:.9g}",
                flush=True,
            )

        official_norm = MiniMaxH3AdaLayerNormOut(
            hidden_size=local_model.hidden_size,
            time_embed_dim=local_model.final_layer.adaln_proj.linear.in_features,
            eps=local_model.final_layer.norm.eps,
        ).to(device=device, dtype=torch.bfloat16)
        official_norm.norm.weight.copy_(local_model.final_layer.norm.weight)
        official_norm.linear.load_state_dict(
            local_model.final_layer.adaln_proj.linear.state_dict()
        )
        official_out = torch.nn.Linear(
            local_model.hidden_size,
            local_model.final_layer.video_out.out_features,
            bias=True,
            device=device,
            dtype=local_model.final_layer.video_out.weight.dtype,
        )
        official_out.load_state_dict(
            local_model.final_layer.video_out.state_dict()
        )
        local_logits = local_model.final_layer(
            local_hidden, time_embedding, combined_indices
        )
        official_logits = official_out(
            official_norm(
                official_hidden,
                time_embedding,
                torch.zeros(
                    args.sequence_length,
                    device=device,
                    dtype=torch.long,
                ),
            ).to(official_out.weight.dtype)
        )
        final_diff = float(
            (local_logits.float() - official_logits.float()).abs().max()
        )
        final_difference = local_logits.float() - official_logits.float()
        final_relative_rms = float(
            final_difference.square().mean().sqrt()
            / official_logits.float().square().mean().sqrt().clamp(min=1e-12)
        )
        final_cosine = float(
            F.cosine_similarity(
                local_logits.float().flatten(),
                official_logits.float().flatten(),
                dim=0,
            )
        )
    print(
        f"worst_block_max_abs_diff={worst_block_diff:.9g} "
        f"worst_block_relative_rms={worst_relative_rms:.9g} "
        f"worst_block_cosine={worst_cosine:.9g} "
        f"final_velocity_max_abs_diff={final_diff:.9g} "
        f"final_velocity_relative_rms={final_relative_rms:.9g} "
        f"final_velocity_cosine={final_cosine:.9g}"
    )
    if worst_relative_rms > args.max_relative_rms:
        raise SystemExit(
            f"Block relative RMS {worst_relative_rms} exceeds "
            f"{args.max_relative_rms}"
        )
    if final_relative_rms > args.max_relative_rms:
        raise SystemExit(
            f"Final relative RMS {final_relative_rms} exceeds "
            f"{args.max_relative_rms}"
        )
    if min(worst_cosine, final_cosine) < args.min_cosine:
        raise SystemExit(
            f"Cosine {min(worst_cosine, final_cosine)} below "
            f"{args.min_cosine}"
        )
    print("official_50block_final_velocity_parity=PASS")


if __name__ == "__main__":
    main()
