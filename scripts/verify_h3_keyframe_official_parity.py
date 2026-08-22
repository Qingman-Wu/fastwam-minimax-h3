"""Compare FastWAM keyframe encoding with pinned Diffusers MiniMax-H3.

Run this from the isolated parity environment containing Diffusers revision
2f7e0154a9db246e95c9ede43edba7db5b130805.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch

from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
from fastwam.models.minimax_h3.video_vae import MiniMaxH3VAEAdapter


class _ReleasedVAEProxy:
    """Expose the released VAE through the official helper's small interface."""

    def __init__(self, adapter: MiniMaxH3VAEAdapter) -> None:
        self.adapter = adapter
        self.config = SimpleNamespace(
            latents_mean=adapter.latents_mean.flatten().tolist(),
            latents_std=adapter.latents_std.flatten().tolist(),
        )

    def encode(
        self, pixels: torch.Tensor, return_dict: bool = False
    ) -> tuple[object]:
        if return_dict:
            raise ValueError("Parity proxy supports return_dict=False only")
        moments = self.adapter.vae._adaptive_encode(pixels)
        return (self.adapter._distribution_cls(moments),)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component-dir",
        default="/root/wuqingman/models/MiniMax-H3/FL2VA/video_vae",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--image-seed", type=int, default=20260822)
    parser.add_argument("--encode-seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    image_generator = torch.Generator(device="cpu").manual_seed(args.image_seed)
    pixels_uint8 = torch.randint(
        0,
        256,
        (1, 3, args.height, args.width),
        dtype=torch.uint8,
        generator=image_generator,
    )
    image = pixels_uint8.float().div(127.5).sub(1.0)

    adapter = MiniMaxH3VAEAdapter(
        args.component_dir, device=device, dtype=torch.float32
    )
    official = encode_vae_condition(
        _ReleasedVAEProxy(adapter),
        pixels_uint8.unsqueeze(2).to(device),
        pixel_mean=(0.485, 0.456, 0.406),
        pixel_std=(0.229, 0.224, 0.225),
        encode_seed=args.encode_seed,
    )
    fastwam = adapter.encode_keyframe_condition(
        image, device="cpu", seed=args.encode_seed
    )
    difference = (fastwam - official).abs()
    max_abs_diff = float(difference.max())
    mean_abs_diff = float(difference.mean())
    print(
        f"shape={tuple(fastwam.shape)} "
        f"max_abs_diff={max_abs_diff:.9g} "
        f"mean_abs_diff={mean_abs_diff:.9g}"
    )
    if max_abs_diff > args.atol:
        raise SystemExit(
            f"Official keyframe parity failed: {max_abs_diff} > {args.atol}"
        )
    print("official_keyframe_parity=PASS")


if __name__ == "__main__":
    main()
