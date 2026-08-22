"""Verify cached H3 prefix moments preserve released posterior sampling."""

from __future__ import annotations

import argparse

import torch

from fastwam.models.minimax_h3.video_vae import MiniMaxH3VAEAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component-dir",
        default="/root/wuqingman/models/MiniMax-H3/FL2VA/video_vae",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--video-seed", type=int, default=20260822)
    parser.add_argument("--posterior-seed", type=int, default=123)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    video_generator = torch.Generator(device="cpu").manual_seed(args.video_seed)
    pixels_uint8 = torch.randint(
        0,
        256,
        (1, 3, args.num_frames, args.height, args.width),
        dtype=torch.uint8,
        generator=video_generator,
    )
    video = pixels_uint8.float().div(127.5).sub(1.0)

    adapter = MiniMaxH3VAEAdapter(
        args.component_dir, device=device, dtype=torch.float32
    )
    pixels = pixels_uint8.to(device=device, dtype=torch.float32).div(255.0)
    transformed = adapter.vae.processor.transform_tensor(pixels).float()

    cached_latent_shape = (
        1,
        adapter.z_dim,
        adapter.latent_temporal_length(args.num_frames),
        args.height // adapter.upsampling_factor,
        args.width // adapter.upsampling_factor,
    )
    released_latent_shape = (
        cached_latent_shape[0],
        cached_latent_shape[1],
        adapter.vae.tokens_chunk_size,
        cached_latent_shape[3],
        cached_latent_shape[4],
    )
    noise_generator = torch.Generator(device="cpu").manual_seed(
        args.posterior_seed
    )
    released_posterior_noise = torch.randn(
        released_latent_shape, generator=noise_generator
    )
    posterior_noise = released_posterior_noise[
        :, :, : cached_latent_shape[2]
    ].contiguous()
    distribution_class = adapter._distribution_cls
    original_sample = distribution_class.sample

    def fixed_sample(distribution, generator=None):
        del generator
        if tuple(distribution.mean.shape) != released_latent_shape:
            raise ValueError(
                "Unexpected released posterior shape "
                f"{tuple(distribution.mean.shape)} != {released_latent_shape}"
            )
        return distribution.mean + distribution.std * (
            released_posterior_noise.to(distribution.mean.device)
        )

    distribution_class.sample = fixed_sample
    try:
        released_raw, prefix_pad_frames = adapter.vae.encode_videos(
            [transformed[0]],
            transform_input=False,
            use_fp16_latent=False,
            encode_prefix=True,
        )
    finally:
        distribution_class.sample = original_sample
    if prefix_pad_frames != [0]:
        raise SystemExit(f"Unexpected prefix padding: {prefix_pad_frames}")
    released_normalized = adapter._normalize(
        released_raw[0].unsqueeze(0).float()
    ).cpu()

    cached_mean, cached_logvar = adapter.encode_video_posterior(
        video, device="cpu"
    )
    if tuple(cached_mean.shape) != cached_latent_shape:
        raise SystemExit(
            f"Unexpected cached posterior shape {tuple(cached_mean.shape)}"
        )
    cached_sample = cached_mean + torch.exp(0.5 * cached_logvar) * posterior_noise

    difference = (cached_sample - released_normalized).abs()
    max_abs_diff = float(difference.max())
    mean_abs_diff = float(difference.mean())
    print(
        f"shape={tuple(cached_sample.shape)} "
        f"max_abs_diff={max_abs_diff:.9g} "
        f"mean_abs_diff={mean_abs_diff:.9g}"
    )
    torch.testing.assert_close(
        cached_sample,
        released_normalized,
        atol=args.atol,
        rtol=args.rtol,
    )
    print("released_prefix_posterior_parity=PASS")


if __name__ == "__main__":
    main()
