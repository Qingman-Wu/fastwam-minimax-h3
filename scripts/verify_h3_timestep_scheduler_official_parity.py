"""Check FastWAM timestep/AdaLN and scheduler against pinned Diffusers H3."""

from __future__ import annotations

import torch

from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3AdaLayerNormModulation,
    MiniMaxH3Transformer3DModel,
)
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from fastwam.models.minimax_h3.video_dit import (
    H3AdaLNProjection,
    H3TimeEmbedder,
)
from fastwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)


def _copy_linear(target: torch.nn.Linear, source: torch.nn.Linear) -> None:
    target.weight.data.copy_(source.weight.data)
    target.bias.data.copy_(source.bias.data)


def verify_timestep_and_adaln() -> None:
    torch.manual_seed(20260822)
    official_model = MiniMaxH3Transformer3DModel(
        num_attention_heads=2,
        attention_head_dim=8,
        hidden_size=16,
        num_layers=0,
        num_refiner_layers=0,
        ffn_dim=32,
        in_channels=4,
        audio_in_channels=4,
        patch_size=(1, 1, 1),
        text_dim=12,
        freq_dim=8,
        time_embed_hidden_dim=24,
        time_embed_dim=10,
        rope_freq_dim=2,
    )
    local_time = H3TimeEmbedder(8, 24, 10)
    _copy_linear(local_time.proj_in, official_model.time_embedder.linear_1)
    _copy_linear(local_time.proj_out, official_model.time_embedder.linear_2)
    timesteps = torch.tensor([0.0, 0.125, 0.5, 0.999], dtype=torch.float32)
    official_time = official_model.time_embedder(
        official_model.time_proj(timesteps).float()
    )
    local_time_output = local_time(timesteps)
    torch.testing.assert_close(
        local_time_output, official_time, atol=0.0, rtol=0.0
    )

    official_adaln = MiniMaxH3AdaLayerNormModulation(
        time_embed_dim=10, hidden_size=16
    ).to(dtype=torch.bfloat16)
    local_adaln = H3AdaLNProjection(
        hidden_size=16, time_embed_dim=10
    ).to(dtype=torch.bfloat16)
    _copy_linear(local_adaln.linear, official_adaln.linear)
    official_values = official_adaln(official_time)
    local_values = local_adaln(local_time_output)
    for local_value, official_value in zip(local_values, official_values):
        torch.testing.assert_close(
            local_value, official_value, atol=0.0, rtol=0.0
        )
    print("official_timestep_adaln_parity=PASS")


def verify_scheduler() -> None:
    sigma_points = 20
    shift = 12.0
    official = MiniMaxH3Scheduler(shift=shift)
    official.set_timesteps(sigma_points, device="cpu")
    local = WanContinuousFlowMatchScheduler(
        num_train_timesteps=1000, shift=shift
    )
    local_timesteps, local_deltas = local.build_inference_schedule(
        sigma_points - 1, torch.device("cpu"), torch.bfloat16
    )
    local_sigmas = torch.cat(
        (
            local_timesteps.div(1000.0),
            (local_timesteps[-1].div(1000.0) + local_deltas[-1]).view(1),
        )
    )
    torch.testing.assert_close(
        local_sigmas, official.sigmas, atol=1e-7, rtol=1e-7
    )

    generator = torch.Generator(device="cpu").manual_seed(20260822)
    local_sample = torch.randn(
        (2, 4, 3, 5), generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    official_sample = local_sample.clone()
    for index, (local_timestep, local_delta) in enumerate(
        zip(local_timesteps, local_deltas)
    ):
        local_velocity = torch.randn(
            local_sample.shape, generator=generator, dtype=torch.float32
        )
        local_sample = local.step_h3_video(
            local_velocity,
            local_timestep,
            local_delta,
            local_sample,
            timestep_scale=1000.0,
        )
        official_sample = official.step(
            -local_velocity,
            official.timesteps[index],
            official_sample,
        ).prev_sample
        torch.testing.assert_close(
            local_sample, official_sample, atol=0.0, rtol=0.0
        )
    print("official_scheduler_trajectory_parity=PASS")


def main() -> None:
    verify_timestep_and_adaln()
    verify_scheduler()


if __name__ == "__main__":
    main()
