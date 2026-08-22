from contextlib import nullcontext

import torch
import pytest

from fastwam.trainer import Wan22Trainer, _validate_evaluation_vae_contract


class _DummyDit:
    training = False


class _DummyH3:
    inference_accepts_ground_truth_action = False
    dit = _DummyDit()

    def eval(self):
        return self

    def training_loss(self, sample):
        return torch.tensor(1.25), {}

    def infer(self, **kwargs):
        assert kwargs["decode_video"] is False
        return {
            "video_latents": torch.zeros(1, 2, 2, 2, 2),
            "action": None,
        }


class _DummyAccelerator:
    process_index = 0
    device = torch.device("cpu")
    is_main_process = True

    def unwrap_model(self, model):
        return model

    def autocast(self):
        return nullcontext()

    def gather_for_metrics(self, value):
        return value


def test_h3_evaluate_skips_unsupported_pixel_metrics():
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.val_dataset = [
        {
            "video": torch.zeros(3, 5, 4, 4),
            "prompt": "move",
        }
    ]
    trainer.model = _DummyH3()
    trainer.accelerator = _DummyAccelerator()
    trainer.global_step = 0
    trainer.eval_num_inference_steps = 1

    metrics = trainer.evaluate()

    assert metrics == {
        "val_loss": 1.25,
        "video_path": None,
    }


def test_cache_only_h3_rejects_periodic_inference_without_vae():
    model = type("CacheOnlyH3", (), {"vae": None})()

    with pytest.raises(ValueError, match="Periodic evaluation requires a loaded VAE"):
        _validate_evaluation_vae_contract(model, val_dataset=[object()], eval_every=10)


def test_cache_only_h3_allows_disabled_or_absent_validation():
    model = type("CacheOnlyH3", (), {"vae": None})()

    _validate_evaluation_vae_contract(model, val_dataset=[object()], eval_every=0)
    _validate_evaluation_vae_contract(model, val_dataset=None, eval_every=10)
