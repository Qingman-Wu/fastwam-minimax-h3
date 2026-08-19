from pathlib import Path
import tomllib

import pytest
import torch
from omegaconf import OmegaConf

from fastwam.models.minimax_h3.fastwam import FastWAMH3
from fastwam.runtime import create_fastwam_h3


ROOT = Path(__file__).resolve().parents[3]


def minimal_kwargs():
    return {
        "model_path": "/models/MiniMax-H3/FL2VA",
        "video_dit_config": {
            "num_layers": 50,
            "num_attention_heads": 56,
            "attention_head_dim": 128,
            "video_attention_mask_mode": "bidirectional",
        },
        "action_dit_config": {
            "action_dim": 7,
            "hidden_size": 1024,
            "num_layers": 50,
            "num_attention_heads": 56,
            "attention_head_dim": 128,
        },
        "proprio_dim": 8,
        "video_scheduler": {
            "train_shift": 12.0,
            "infer_shift": 12.0,
            "num_train_timesteps": 1000,
        },
        "action_scheduler": {
            "train_shift": 5.0,
            "infer_shift": 5.0,
            "num_train_timesteps": 1000,
        },
        "loss": {"lambda_video": 1.0, "lambda_action": 1.0},
        "load_text_encoder": True,
        "keyframe_condition_strength": 0.999,
        "video_fps": 24.0,
        "action_fps": 8.0,
        "freeze_video_expert": True,
        "mot_checkpoint_mixed_attn": False,
        "model_dtype": torch.float32,
        "device": "cpu",
    }


def test_runtime_wires_both_schedulers_state_and_native_text(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_from_pretrained(cls, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        FastWAMH3, "from_pretrained", classmethod(fake_from_pretrained)
    )

    assert create_fastwam_h3(**minimal_kwargs()) is sentinel
    assert captured["load_text_encoder"] is True
    assert captured["video_train_shift"] == 12.0
    assert captured["action_train_shift"] == 5.0
    assert captured["keyframe_condition_strength"] == 0.999
    assert captured["loss_lambda_video"] == 1.0
    assert captured["loss_lambda_action"] == 1.0
    assert captured["action_dit_config"]["state_dim"] == 8
    assert captured["action_dit_config"]["use_gradient_checkpointing"] is False


def test_runtime_requires_complete_video_scheduler():
    kwargs = minimal_kwargs()
    kwargs["video_scheduler"] = None

    with pytest.raises(ValueError, match="video_scheduler"):
        create_fastwam_h3(**kwargs)


def test_runtime_rejects_attention_geometry_mismatch():
    kwargs = minimal_kwargs()
    kwargs["action_dit_config"]["attention_head_dim"] = 64

    with pytest.raises(ValueError, match="attention geometry"):
        create_fastwam_h3(**kwargs)


def test_h3_yaml_is_scheme_a_not_legacy_action_only():
    config = OmegaConf.load(ROOT / "configs/model/fastwam_h3.yaml")
    raw = OmegaConf.to_container(config, resolve=False)

    assert config.load_text_encoder is True
    assert config.keyframe_condition_strength == 0.999
    assert config.video_dit_config.video_attention_mask_mode == "bidirectional"
    assert "audio_latents_dim" not in config.video_dit_config
    assert config.action_dit_config.hidden_size == 1024
    assert config.action_dit_config.num_attention_heads == 56
    assert config.action_dit_config.attention_head_dim == 128
    assert raw["action_dit_config"]["state_dim"] == (
        "${data.train.processor.proprio_output_dim}"
    )
    assert "context_dim" not in config.action_dit_config
    assert config.loss.lambda_video == 1.0
    assert config.loss.lambda_action == 1.0


def test_h3_task_uses_native_prompt_path_and_no_t5_cache():
    config = OmegaConf.load(
        ROOT / "configs/task/libero_h3_uncond_2cam224_1e-4.yaml"
    )
    data_config = OmegaConf.load(ROOT / "configs/data/libero_2cam.yaml")

    assert config.data.train.text_embedding_cache_dir is None
    assert data_config.train.num_frames == 33
    assert config.data.train.action_video_freq_ratio == 8


def test_transformers_pin_exposes_qwen3_vl_release():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    dependencies = pyproject["project"]["dependencies"]
    assert "transformers==4.57.6" in dependencies
