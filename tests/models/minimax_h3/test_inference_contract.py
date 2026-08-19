import pytest
import torch
import torch.nn as nn

from fastwam.models.minimax_h3.fastwam import FastWAMH3
from fastwam.models.minimax_h3.text_encoder import H3TextConditionBatch


class InferenceVAE(nn.Module):
    temporal_downsample_factor = 4
    upsampling_factor = 16
    z_dim = 2

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.model = type("VAEInfo", (), {"z_dim": self.z_dim})()
        self.calls = []

    def encode_video(self, video, device=None, process_image=False):
        self.calls.append(("encode", process_image, tuple(video.shape)))
        if not process_image:
            raise AssertionError("inference must not encode a ground-truth full video")
        return torch.full(
            (video.shape[0], self.z_dim, 1, video.shape[-2] // 16, video.shape[-1] // 16),
            2.0,
            device=video.device,
        )

    def decode(self, latents, device=None, frame_num=None):
        self.calls.append(("decode", frame_num, tuple(latents.shape)))
        frames = int(frame_num)
        values = torch.linspace(-1.0, 1.0, frames, device=latents.device)
        return values.view(1, 1, frames, 1, 1).expand(
            latents.shape[0], 3, frames, latents.shape[-2] * 16, latents.shape[-1] * 16
        )


class InferenceActionExpert(nn.Module):
    action_dim = 3
    state_dim = 4
    num_layers = 1
    num_heads = 1
    attn_head_dim = 2

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))


class RecordingJointVideoExpert(nn.Module):
    num_layers = 1
    num_heads = 1
    attn_head_dim = 2

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.calls = []

    def forward_joint(self, **kwargs):
        self.calls.append(
            {
                "video": kwargs["noisy_video_latents"].detach().clone(),
                "action": kwargs["noisy_action_tokens"].detach().clone(),
                "keyframe": kwargs["clean_keyframe_latents"].detach().clone(),
                "state": kwargs["state_tokens"].detach().clone(),
                "video_timestep": kwargs["video_timestep"].detach().clone(),
                "action_timestep": kwargs["action_timestep"].detach().clone(),
            }
        )
        return {
            "video_prediction": torch.ones_like(kwargs["noisy_video_latents"]),
            "action_prediction": torch.full_like(kwargs["noisy_action_tokens"], 2.0),
        }


class CountingTextConditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.calls = 0

    def encode(self, images, instructions):
        self.calls += 1
        assert len(images) == len(instructions) == 1
        return H3TextConditionBatch.from_precomputed(
            embeddings=torch.zeros(3, 5120),
            token_tags=torch.tensor([1, 0, 1]),
            cu_seqlens=torch.tensor([0, 3], dtype=torch.int32),
        )


def make_model(*, text_conditioner=None):
    return FastWAMH3(
        video_expert=RecordingJointVideoExpert(),
        action_expert=InferenceActionExpert(),
        vae=InferenceVAE(),
        text_conditioner=text_conditioner,
        device="cpu",
        torch_dtype=torch.float32,
        video_train_shift=12.0,
        video_infer_shift=12.0,
        video_num_train_timesteps=1000,
        action_train_shift=5.0,
        action_infer_shift=5.0,
        action_num_train_timesteps=1000,
        keyframe_condition_strength=0.999,
        loss_lambda_video=1.0,
        loss_lambda_action=1.0,
        video_fps=24.0,
        action_fps=8.0,
        freeze_video_expert=True,
    )


def infer_kwargs():
    return {
        "input_image": torch.zeros(1, 3, 32, 32),
        "num_frames": 5,
        "action_horizon": 4,
        "proprio": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "prompt_embeds": torch.zeros(1, 3, 5120),
        "prompt_token_tags": torch.tensor([[1, 0, 1]]),
        "prompt_attention_mask": torch.tensor([[True, True, True]]),
        "num_inference_steps": 3,
        "video_noise": torch.zeros(1, 2, 2, 2, 2),
        "action_noise": torch.zeros(1, 4, 3),
        "keyframe_noise": torch.zeros(1, 2, 1, 2, 2),
    }


def test_joint_inference_updates_full_video_and_action_at_every_step():
    model = make_model()

    output = model.infer(**infer_kwargs())

    assert len(model.video_expert.calls) == 3
    assert not torch.equal(
        model.video_expert.calls[0]["video"], model.video_expert.calls[1]["video"]
    )
    assert not torch.equal(
        model.video_expert.calls[0]["action"], model.video_expert.calls[1]["action"]
    )
    assert output["action"].shape == (4, 3)
    assert not torch.equal(output["action"], torch.zeros_like(output["action"]))


def test_inference_conditions_are_computed_once_and_never_replace_video_noise():
    model = make_model()

    model.infer(**infer_kwargs())

    assert [call[0] for call in model.vae.calls] == ["encode", "decode"]
    assert model.vae.calls[0][1] is True
    assert torch.equal(
        model.video_expert.calls[0]["video"], torch.zeros(1, 2, 2, 2, 2)
    )
    assert torch.allclose(
        model.video_expert.calls[0]["keyframe"], torch.full((1, 2, 1, 2, 2), 1.998)
    )
    assert torch.equal(
        model.video_expert.calls[0]["state"], torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    )


def test_native_first_frame_qwen_condition_is_computed_once():
    conditioner = CountingTextConditioner()
    model = make_model(text_conditioner=conditioner)
    kwargs = infer_kwargs()
    kwargs.pop("prompt_embeds")
    kwargs.pop("prompt_token_tags")
    kwargs.pop("prompt_attention_mask")
    kwargs["prompt"] = "move the gripper"

    model.infer(**kwargs)

    assert conditioner.calls == 1


def test_video_and_action_keep_separate_shifted_schedules():
    model = make_model()

    model.infer(**infer_kwargs())

    first = model.video_expert.calls[0]
    second = model.video_expert.calls[1]
    assert first["video_timestep"].item() == 1000.0
    assert first["action_timestep"].item() == 1000.0
    assert second["video_timestep"].item() != second["action_timestep"].item()


def test_decoded_auxiliary_video_is_generated_not_repeated_input_frame():
    model = make_model()

    output = model.infer(**infer_kwargs())

    assert len(output["video"]) == 5
    assert output["video"][0].tobytes() != output["video"][1].tobytes()
    assert model.vae.calls[-1][1] == 5


def test_inference_rejects_ground_truth_action_conditioning():
    model = make_model()
    kwargs = infer_kwargs()
    kwargs["action"] = torch.zeros(4, 3)

    with pytest.raises(ValueError, match="ground-truth action"):
        model.infer(**kwargs)
