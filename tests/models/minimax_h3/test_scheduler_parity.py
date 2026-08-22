import torch

from fastwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)


def test_h3_video_step_uses_official_fp32_blend_order():
    scheduler = WanContinuousFlowMatchScheduler(
        num_train_timesteps=1000, shift=12.0
    )
    sample = torch.tensor(
        [[[[0.25, -0.75], [1.5, -2.0]]]], dtype=torch.bfloat16
    )
    local_velocity = torch.tensor(
        [[[[0.125, 0.5], [-0.25, 1.0]]]], dtype=torch.float32
    )
    timestep = torch.tensor(875.0, dtype=torch.float32)
    delta = torch.tensor(-0.0625, dtype=torch.float32)

    actual = scheduler.step_h3_video(
        local_velocity, timestep, delta, sample
    )

    sigma = timestep.float() / 1000.0
    sigma_next = sigma + delta
    progress = 1.0 - sigma
    sigma_from_progress = 1.0 - progress.to(torch.bfloat16)
    denoised = sample - sigma_from_progress * local_velocity
    ratio = sigma_next / sigma
    expected = (
        ratio * sample.float() + (1.0 - ratio) * denoised.float()
    ).to(torch.bfloat16)
    assert torch.equal(actual, expected)

