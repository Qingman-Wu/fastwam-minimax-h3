import torch
import torch.nn as nn

from fastwam.models.minimax_h3.action_dit import H3ActionDiT
from fastwam.models.minimax_h3.fastwam import FastWAMH3
from fastwam.models.minimax_h3.video_dit import MiniMaxH3VideoBackbone


class TinyVAE(nn.Module):
    def __init__(self, first_latent_value=1.0):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.first_latent_value = float(first_latent_value)
        self.calls = []

    def encode_video(self, video, device=None, process_image=False):
        self.calls.append(process_image)
        batch = video.shape[0]
        if process_image:
            return torch.full((batch, 2, 1, 2, 2), 3.0, device=video.device)
        latent = torch.full((batch, 2, 2, 2, 2), 2.0, device=video.device)
        latent[:, :, 0] = self.first_latent_value
        return latent


class TinyActionExpert(nn.Module):
    action_dim = 3
    state_dim = 4
    num_layers = 1
    num_heads = 1
    attn_head_dim = 2

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))


class TinyJointVideoExpert(nn.Module):
    num_layers = 1
    num_heads = 1
    attn_head_dim = 2

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.last_inputs = None

    def forward_joint(self, **kwargs):
        self.last_inputs = kwargs
        return {
            "video_prediction": torch.zeros_like(kwargs["noisy_video_latents"]),
            "action_prediction": torch.zeros_like(kwargs["noisy_action_tokens"]),
        }


def make_model(first_latent_value=1.0):
    return FastWAMH3(
        video_expert=TinyJointVideoExpert(),
        action_expert=TinyActionExpert(),
        vae=TinyVAE(first_latent_value),
        text_conditioner=None,
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


def make_sample():
    action = torch.ones(1, 4, 3)
    action[:, 2:, 1] = 1000.0
    return {
        "video": torch.zeros(1, 3, 5, 32, 32),
        "action": action,
        "action_is_pad": torch.tensor([[False, False, True, True]]),
        "action_dim_is_pad": torch.tensor([[False, True, False]]),
        "proprio": torch.tensor([[[1.0, 2.0, 300.0, 4.0]]]),
        "proprio_is_pad": torch.tensor([[False]]),
        "proprio_dim_is_pad": torch.tensor([[False, False, True, False]]),
        "prompt_embeds": torch.zeros(1, 3, 5120),
        "prompt_token_tags": torch.tensor([[1, 0, 1]]),
        "prompt_attention_mask": torch.tensor([[True, True, True]]),
    }


def deterministic_training_loss(model, sample):
    return model.training_loss(
        sample,
        base_progress=torch.tensor([0.5]),
        video_noise=torch.zeros(1, 2, 2, 2, 2),
        action_noise=torch.zeros(1, 4, 3),
        keyframe_noise=torch.zeros(1, 2, 1, 2, 2),
    )


def test_training_encodes_full_video_once_and_keyframe_separately():
    model = make_model()

    deterministic_training_loss(model, make_sample())

    assert model.vae.calls == [False, True]
    inputs = model.video_expert.last_inputs
    assert inputs["clean_keyframe_latents"].shape == (1, 2, 1, 2, 2)
    assert inputs["noisy_video_latents"].shape == (1, 2, 2, 2, 2)
    assert inputs["clean_keyframe_latents"].data_ptr() != inputs[
        "noisy_video_latents"
    ].data_ptr()


def test_video_loss_includes_first_temporal_latent():
    first_small = make_model(first_latent_value=1.0)
    first_large = make_model(first_latent_value=10.0)

    _, metrics_small = deterministic_training_loss(first_small, make_sample())
    _, metrics_large = deterministic_training_loss(first_large, make_sample())

    assert metrics_large["loss_video"] > metrics_small["loss_video"]


def test_action_loss_excludes_padded_times_and_dimensions():
    model = make_model()
    sample = make_sample()
    _, metrics = deterministic_training_loss(model, sample)

    changed = make_sample()
    changed["action"][:, 2:] = 1_000_000.0
    changed["action"][:, :, 1] = -1_000_000.0
    _, changed_metrics = deterministic_training_loss(model, changed)

    assert metrics["loss_action"] == changed_metrics["loss_action"]


def test_state_is_current_aligned_condition_and_invalid_dimensions_are_zero():
    model = make_model()

    deterministic_training_loss(model, make_sample())

    state = model.video_expert.last_inputs["state_tokens"]
    assert torch.equal(state, torch.tensor([[1.0, 2.0, 0.0, 4.0]]))


def test_state_condition_only_uses_f0_even_if_observation_mask_is_longer():
    model = make_model()
    sample = make_sample()
    sample["proprio_is_pad"] = torch.tensor(
        [[False, False, False, False, False]]
    )

    deterministic_training_loss(model, sample)

    assert torch.equal(
        model.video_expert.last_inputs["state_tokens"],
        torch.tensor([[1.0, 2.0, 0.0, 4.0]]),
    )


def test_shared_progress_produces_separate_video_and_action_sigmas():
    model = make_model()

    _, metrics = deterministic_training_loss(model, make_sample())

    assert metrics["base_progress_mean"] == 0.5
    assert metrics["sigma_video_mean"] != metrics["sigma_action_mean"]
    assert metrics["sigma_video_mean"] > metrics["sigma_action_mean"]


def test_keyframe_is_near_clean_and_has_no_loss_row():
    model = make_model()

    deterministic_training_loss(model, make_sample())

    keyframe = model.video_expert.last_inputs["clean_keyframe_latents"]
    assert torch.allclose(keyframe, torch.full_like(keyframe, 2.997))
    assert "keyframe_loss_mask" not in model.video_expert.last_inputs


def test_real_tiny_experts_run_scheme_a_checkpointed_forward_and_backward():
    torch.manual_seed(17)
    video_expert = MiniMaxH3VideoBackbone(
        hidden_size=8,
        ffn_hidden_size=16,
        num_layers=1,
        token_refiner_num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        latents_dim=2,
        patch_size=(1, 2, 2),
        text_dim=5,
        timestep_input_dim=4,
        time_embed_hidden_size=8,
        time_embed_dim=8,
        rope_inv_freq_len=1,
    ).train()
    action_expert = H3ActionDiT(
        action_dim=3,
        state_dim=4,
        hidden_size=8,
        ffn_hidden_size=16,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        timestep_input_dim=4,
        time_embed_hidden_size=8,
        time_embed_dim=8,
        rope_inv_freq_len=1,
        use_gradient_checkpointing=True,
    ).train()

    output = video_expert.forward_joint(
        action_expert=action_expert,
        qwen_embeddings=torch.randn(2, 3, 5),
        qwen_tags=torch.tensor([[1, 0, 1], [1, 0, 0]]),
        qwen_valid=torch.tensor([[True, True, True], [True, True, False]]),
        clean_keyframe_latents=torch.randn(2, 2, 1, 4, 4),
        noisy_video_latents=torch.randn(2, 2, 2, 4, 4),
        video_timestep=torch.tensor([500.0, 700.0]),
        noisy_action_tokens=torch.randn(2, 3, 3),
        action_timestep=torch.tensor([300.0, 600.0]),
        state_tokens=torch.randn(2, 4),
        action_valid=torch.tensor([[True, True, True], [True, True, False]]),
        keyframe_condition_strength=0.999,
        video_fps=24.0,
        action_fps=8.0,
        return_debug=True,
    )

    assert output["video_prediction"].shape == (2, 2, 2, 4, 4)
    assert output["action_prediction"].shape == (2, 3, 3)
    assert torch.isfinite(output["video_prediction"]).all()
    assert torch.isfinite(output["action_prediction"]).all()
    assert output["debug"]["keyframe_rows"] == 4
    assert output["debug"]["video_target_rows"] == 8
    assert output["debug"]["audio_rows"] == 0

    loss = (
        output["video_prediction"].float().square().mean()
        + output["action_prediction"].float().square().mean()
    )
    loss.backward()
    video_grads = [
        parameter.grad
        for parameter in video_expert.parameters()
        if parameter.grad is not None
    ]
    action_grads = [
        parameter.grad
        for parameter in action_expert.parameters()
        if parameter.grad is not None
    ]
    assert video_grads and action_grads
    assert all(torch.isfinite(grad).all() for grad in video_grads + action_grads)
