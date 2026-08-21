import torch
import torch.nn as nn

from fastwam.models.minimax_h3.action_dit import H3ActionDiT
from fastwam.models.minimax_h3.fastwam import FastWAMH3
from fastwam.models.minimax_h3.video_dit import MiniMaxH3VideoBackbone


class CheckpointVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)


def make_model(*, alpha=4.0):
    video = MiniMaxH3VideoBackbone(
        hidden_size=8,
        ffn_hidden_size=16,
        num_layers=1,
        token_refiner_num_layers=0,
        num_attention_heads=2,
        attention_head_dim=8,
        latents_dim=2,
        patch_size=(1, 2, 2),
        text_dim=5,
        timestep_input_dim=4,
        time_embed_hidden_size=8,
        time_embed_dim=8,
        rope_inv_freq_len=1,
    )
    video.inject_attention_lora(rank=2, alpha=alpha, dropout=0.1)
    action = H3ActionDiT(
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
    )
    return FastWAMH3(
        video_expert=video,
        action_expert=action,
        vae=CheckpointVAE(),
        text_conditioner=None,
        device="cpu",
        torch_dtype=torch.float32,
        video_train_shift=12.0,
        video_infer_shift=12.0,
        video_num_train_timesteps=1000,
        action_train_shift=5.0,
        action_infer_shift=5.0,
        action_num_train_timesteps=1000,
        freeze_video_expert=True,
        h3_lora_rank=2,
        h3_lora_alpha=alpha,
        h3_lora_dropout=0.1,
        base_h3_fingerprint="sha256:test-h3",
    )


def test_schema3_checkpoint_round_trip_uses_named_lora_paths(tmp_path):
    model = make_model()
    with torch.no_grad():
        model.action_expert.action_head.weight.fill_(0.25)
        for index, branch in enumerate(model._named_h3_lora_branches().values()):
            branch.lora_b.weight.fill_(index + 1)
    expected_action = model.action_expert.action_head.weight.detach().clone()
    expected_lora = {
        name: branch.lora_b.weight.detach().clone()
        for name, branch in model._named_h3_lora_branches().items()
    }
    path = tmp_path / "scheme-a-v3.pt"

    model.save_checkpoint(path, step=17)
    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert payload["schema_version"] == 3
    assert set(payload["h3_lora"]) == {
        "blocks.0.attn.qkv_proj",
        "blocks.0.attn.out_proj",
    }
    assert payload["h3_lora_config"] == {
        "rank": 2,
        "alpha": 4.0,
        "dropout": 0.1,
        "targets": [
            "blocks.0.attn.qkv_proj",
            "blocks.0.attn.out_proj",
        ],
        "base_h3_fingerprint": "sha256:test-h3",
    }

    restored = make_model()
    restored.load_checkpoint(path)
    assert torch.equal(restored.action_expert.action_head.weight, expected_action)
    for name, branch in restored._named_h3_lora_branches().items():
        assert torch.equal(branch.lora_b.weight, expected_lora[name])


def test_schema3_rejects_lora_semantic_config_mismatch(tmp_path):
    path = tmp_path / "scheme-a-v3.pt"
    make_model(alpha=4.0).save_checkpoint(path)

    mismatched = make_model(alpha=2.0)
    try:
        mismatched.load_checkpoint(path)
    except ValueError as error:
        assert "LoRA config" in str(error)
    else:
        raise AssertionError("LoRA alpha mismatch must reject resume")


def test_schema2_checkpoint_is_rejected_instead_of_zero_initializing_lora(tmp_path):
    path = tmp_path / "scheme-a-v2.pt"
    torch.save(
        {
            "schema_version": 2,
            "action_expert": make_model().action_expert.state_dict(),
        },
        path,
    )

    try:
        make_model().load_checkpoint(path)
    except ValueError as error:
        assert "schema-3" in str(error)
    else:
        raise AssertionError("Schema-2 resume must not silently keep zero LoRA")
