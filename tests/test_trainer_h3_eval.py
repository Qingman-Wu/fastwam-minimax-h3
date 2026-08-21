from contextlib import nullcontext

import torch

from fastwam.trainer import Wan22Trainer


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
