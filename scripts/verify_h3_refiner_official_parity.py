"""Compare real-weight FastWAM TokenRefiner with pinned Diffusers H3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3TokenRefiner,
)
from fastwam.models.minimax_h3.video_dit import load_h3_condition_refiner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transformer-dir",
        default="/root/wuqingman/models/MiniMax-H3/FL2VA/transformer",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequence-length", type=int, default=7)
    return parser.parse_args()


@torch.no_grad()
def map_block(local, official) -> None:
    official.norm1.weight.copy_(local.norm1.weight)
    official.norm2.weight.copy_(local.norm2.weight)
    official.attn.norm_q.weight.copy_(local.attn.q_norm.weight)
    official.attn.norm_k.weight.copy_(local.attn.k_norm.weight)
    official.attn.to_out[0].weight.copy_(local.attn.out_proj.weight)
    fused = local.attn.qkv_proj.weight.view(
        local.attn.num_heads,
        3,
        local.attn.head_dim,
        local.attn.qkv_proj.in_features,
    )
    official.attn.to_q.weight.copy_(fused[:, 0].flatten(0, 1))
    official.attn.to_k.weight.copy_(fused[:, 1].flatten(0, 1))
    official.attn.to_v.weight.copy_(fused[:, 2].flatten(0, 1))
    if not official.attn.fused_projections:
        official.attn.fuse_projections()
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
    official.attn.processor._attention_backend = "native"
    local_gate, local_up = local.mlp.fc1.weight.chunk(2, dim=0)
    official.ff.net[0].proj.weight.copy_(
        torch.cat((local_up, local_gate), dim=0)
    )
    official.ff.net[2].weight.copy_(local.mlp.fc2.weight)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = json.loads(
        (Path(args.transformer_dir) / "config.json").read_text()
    )
    local = load_h3_condition_refiner(
        args.transformer_dir, device=device, dtype=torch.bfloat16
    )
    official_projection = torch.nn.Linear(
        local.text_dim,
        local.hidden_size,
        bias=True,
        device=device,
        dtype=local.condition_proj.weight.dtype,
    )
    official_projection.load_state_dict(local.condition_proj.state_dict())
    official = MiniMaxH3TokenRefiner(
        hidden_size=local.hidden_size,
        num_attention_heads=config["num_attention_heads"],
        attention_head_dim=config["attention_head_dim"],
        ffn_dim=config["ffn_hidden_size"],
        num_layers=config["token_refiner_num_layers"],
        norm_eps=config["norm_eps"],
        qk_norm_eps=config["qk_norm_eps"],
        final_norm_eps=config["final_norm_eps"],
    ).to(device=device, dtype=torch.bfloat16)
    for local_block, official_block in zip(
        local.token_refiner.blocks, official.refiner_blocks
    ):
        map_block(local_block, official_block)
    official.final_norm.weight.data.copy_(
        local.token_refiner.final_norm.weight
    )

    generator = torch.Generator(device=device).manual_seed(20260822)
    qwen = torch.randn(
        (args.sequence_length, local.text_dim),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    with torch.no_grad():
        actual = local(
            qwen,
            torch.tensor(
                [0, args.sequence_length], device=device, dtype=torch.int32
            ),
        )
        expected = official(
            official_projection(qwen).unsqueeze(0)
        ).squeeze(0)
    difference = (actual.float() - expected.float()).abs()
    print(
        f"shape={tuple(actual.shape)} "
        f"max_abs_diff={float(difference.max()):.9g} "
        f"mean_abs_diff={float(difference.mean()):.9g}"
    )
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    print("official_token_refiner_parity=PASS")


if __name__ == "__main__":
    main()
