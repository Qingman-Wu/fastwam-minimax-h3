import pytest
import torch

from fastwam.trainer import (
    _accumulate_sample_weighted_metrics,
    _reduce_accumulated_metrics,
)


class _GatherWithRemoteRank:
    def gather(self, local):
        # keys are sorted as loss, loss_video, followed by sample count.
        remote = torch.tensor([9.0, 3.0, 3.0], device=local.device)
        return torch.cat((local, remote))


def test_accumulation_window_metrics_are_global_sample_weighted_means():
    sums = {}
    sample_count = 0
    sample_count += _accumulate_sample_weighted_metrics(
        sums,
        loss=torch.tensor(1.0),
        loss_dict={"loss_video": 0.2},
        batch_size=2,
    )
    sample_count += _accumulate_sample_weighted_metrics(
        sums,
        loss=torch.tensor(4.0),
        loss_dict={"loss_video": 1.1},
        batch_size=1,
    )

    metrics = _reduce_accumulated_metrics(
        _GatherWithRemoteRank(),
        sums,
        sample_count,
    )

    assert metrics["loss"] == pytest.approx(2.5)
    assert metrics["loss_video"] == pytest.approx(0.75)
