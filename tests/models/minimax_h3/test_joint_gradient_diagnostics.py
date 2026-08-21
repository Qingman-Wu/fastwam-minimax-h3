import pytest
import torch

from scripts.diagnose_h3_joint_gradient import (
    _action_rollout_metrics,
    _beta_sweep,
    _gradient_stats,
    _representation_drift,
)
from scripts.smoke_h3_b2_memory import _PaddedH3ConditionDataset
from fastwam.models.minimax_h3.video_dit import H3LoRALinear


def test_gradient_stats_report_norm_ratio_and_cosine():
    stats = _gradient_stats(
        [torch.tensor([3.0, 4.0])],
        [torch.tensor([6.0, 8.0])],
    )

    assert stats["video_grad_norm"] == pytest.approx(5.0)
    assert stats["action_grad_norm"] == pytest.approx(10.0)
    assert stats["action_to_video_norm_ratio"] == pytest.approx(2.0)
    assert stats["cosine"] == pytest.approx(1.0)


def test_beta_sweep_reports_scaled_gradient_balance():
    sweep = _beta_sweep(
        {
            "video_grad_norm": 2.0,
            "action_grad_norm": 20.0,
            "cosine": 0.0,
        },
        [0.1],
    )

    assert sweep["0.1"]["scaled_action_to_video_norm_ratio"] == pytest.approx(1.0)
    assert sweep["0.1"]["combined_to_video_norm_ratio"] == pytest.approx(2**0.5)
    assert sweep["0.1"]["combined_video_cosine"] == pytest.approx(2**-0.5)


def test_representation_drift_is_zero_for_identical_hidden_states():
    hidden = torch.randn(2, 3, 4)

    drift = _representation_drift(hidden, hidden.clone())

    assert drift["rms"] == pytest.approx(0.0)
    assert drift["relative_l2"] == pytest.approx(0.0)
    assert drift["cosine"] == pytest.approx(1.0)


def test_lora_linear_captures_local_output_ratio():
    base = torch.nn.Linear(2, 2, bias=False)
    torch.nn.init.eye_(base.weight)
    projection = H3LoRALinear(
        base,
        rank=1,
        alpha=1.0,
        dropout=0.0,
    )
    projection.capture_diagnostics = True
    projection(torch.ones(1, 1, 2))

    assert projection.diagnostic_stats["lora_to_base_rms_ratio"] == 0.0

    torch.nn.init.ones_(projection.lora.lora_a.weight)
    torch.nn.init.ones_(projection.lora.lora_b.weight)
    projection(torch.ones(1, 1, 2))

    assert projection.diagnostic_stats["lora_to_base_rms_ratio"] == pytest.approx(2.0)


class _TinyConditionDataset:
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return index

    def collate_fn(self, samples):
        del samples
        return {
            "prompt_embeds": torch.ones(2, 3, 4),
            "prompt_token_tags": torch.tensor([[1, 1, 1], [1, 1, 0]]),
            "prompt_attention_mask": torch.tensor(
                [[True, True, True], [True, True, False]]
            ),
        }


def test_memory_smoke_padding_creates_contiguous_valid_rows():
    dataset = _PaddedH3ConditionDataset(_TinyConditionDataset(), pad_to_length=5)

    batch = dataset.collate_fn([0, 1])

    assert batch["prompt_embeds"].shape == (2, 5, 4)
    assert batch["prompt_attention_mask"].all()


def test_action_rollout_metrics_ignore_padded_times_and_dimensions():
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    predicted = target.clone()
    predicted[0, 0] = 2.0
    predicted[0, 1] = 1_000.0
    predicted[1] = -1_000.0

    metrics = _action_rollout_metrics(
        predicted,
        target,
        action_is_pad=torch.tensor([False, True]),
        action_dim_is_pad=torch.tensor([False, True]),
    )

    assert metrics["valid_element_count"] == 1
    assert metrics["l1"] == pytest.approx(1.0)
    assert metrics["mse"] == pytest.approx(1.0)
