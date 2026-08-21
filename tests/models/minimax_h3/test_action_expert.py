import hashlib
import json

import pytest
import torch

from fastwam.models.minimax_h3.action_dit import H3ActionDiT


def tiny_action_config(num_layers=0):
    return {
        "action_dim": 3,
        "state_dim": 5,
        "hidden_size": 8,
        "ffn_hidden_size": 16,
        "num_layers": num_layers,
        "num_attention_heads": 2,
        "attention_head_dim": 8,
        "timestep_input_dim": 4,
        "time_embed_hidden_size": 8,
        "time_embed_dim": 8,
        "rope_inv_freq_len": 1,
        "use_gradient_checkpointing": False,
    }


def make_tiny_action_dit(num_layers=0):
    return H3ActionDiT(**tiny_action_config(num_layers)).eval()


def write_artifact_with_manifest(path):
    model = make_tiny_action_dit()
    state = model.state_dict()
    payload = {
        "meta": {key: getattr(model, key) for key in model.META_KEYS},
        "backbone_state_dict": {
            key: state[key] for key in model.backbone_key_set(state)
        },
    }
    torch.save(payload, path)
    manifest = {
        "schema_version": 1,
        "artifact": {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    }
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(manifest)
    )
    return model


def test_state_is_one_prefix_row_and_head_returns_only_actions():
    model = make_tiny_action_dit()
    state = torch.randn(2, 5)
    action = torch.randn(2, 4, 3)

    prepared = model.pre_dit(action, state, torch.tensor([0.2, 0.8]))
    output = model.post_dit(prepared.tokens)

    assert prepared.tokens.shape == (2, 5, 8)
    assert prepared.state_mask.tolist() == [
        [True, False, False, False, False],
        [True, False, False, False, False],
    ]
    assert prepared.action_mask.tolist() == [
        [False, True, True, True, True],
        [False, True, True, True, True],
    ]
    assert prepared.action_output_indices.tolist() == [1, 2, 3, 4]
    assert output.shape == (2, 4, 3)


def test_state_prefix_is_encoded_without_action_timestep_modulation():
    model = make_tiny_action_dit()
    state = torch.randn(1, 5)
    action = torch.randn(1, 2, 3)

    low = model.pre_dit(action, state, torch.tensor([0.1]))
    high = model.pre_dit(action, state, torch.tensor([0.9]))

    expected = model.state_encoder(state).unsqueeze(1)
    assert torch.allclose(low.tokens[:, :1], expected)
    assert torch.allclose(high.tokens[:, :1], expected)
    assert not torch.allclose(low.time_embedding, high.time_embedding)


def test_action_expert_uses_h3_three_axis_positions():
    model = make_tiny_action_dit()
    state = torch.randn(1, 5)
    action = torch.randn(1, 2, 3)
    positions = torch.tensor(
        [[[7.0, 0.0, 0.0], [7.0, 0.0, 0.0], [12.0, 0.0, 0.0]]],
        dtype=torch.float64,
    )

    prepared = model.pre_dit(
        action, state, torch.tensor([0.5]), position_ids=positions
    )

    assert prepared.position_ids.shape == (1, 3, 3)
    assert prepared.rope_freqs.shape == (1, 3, 6)
    assert torch.allclose(prepared.rope_freqs[0, :, 0], positions[0, :, 0].float())
    assert torch.allclose(prepared.rope_freqs[0, :, 3], positions[0, :, 0].float())
    assert torch.equal(
        prepared.rope_freqs[0, :, [1, 2, 4, 5]], torch.zeros(3, 4)
    )


def test_action_expert_rejects_missing_or_misaligned_state():
    model = make_tiny_action_dit()
    action = torch.randn(2, 4, 3)

    with pytest.raises(ValueError, match="state_tokens"):
        model.pre_dit(action, None, torch.tensor([0.2, 0.8]))
    with pytest.raises(ValueError, match=r"\[B,5\]"):
        model.pre_dit(action, torch.randn(2, 4), torch.tensor([0.2, 0.8]))


def test_action_expert_forward_returns_no_state_prediction():
    model = make_tiny_action_dit(num_layers=1)
    action = torch.randn(2, 4, 3)
    state = torch.randn(2, 5)

    prediction = model(action, state, torch.tensor([0.2, 0.8]))

    assert prediction.shape == action.shape


def test_action_artifact_requires_and_validates_manifest(tmp_path):
    path = tmp_path / "action.pt"
    expected = write_artifact_with_manifest(path)

    loaded = H3ActionDiT.from_pretrained(
        tiny_action_config(),
        path,
        dtype=torch.float32,
    )

    for key in expected.backbone_key_set(expected.state_dict()):
        assert torch.equal(loaded.state_dict()[key], expected.state_dict()[key])


def test_action_artifact_rejects_missing_manifest(tmp_path):
    path = tmp_path / "action.pt"
    model = make_tiny_action_dit()
    torch.save(
        {
            "meta": {key: getattr(model, key) for key in model.META_KEYS},
            "backbone_state_dict": {
                key: model.state_dict()[key]
                for key in model.backbone_key_set(model.state_dict())
            },
        },
        path,
    )

    with pytest.raises(FileNotFoundError, match="artifact manifest"):
        H3ActionDiT.from_pretrained(
            tiny_action_config(),
            path,
            dtype=torch.float32,
        )
