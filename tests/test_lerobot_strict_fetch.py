from types import SimpleNamespace

import pytest
import torch

import fastwam.datasets.lerobot.base_lerobot_dataset as base_module
from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset


def _temporal_sample(*, padded: bool):
    mask = torch.tensor([False, padded])
    return {
        "action_is_pad": mask,
        "image_is_pad": mask,
        "proprio_is_pad": mask,
    }


class _NestedDataset:
    def __init__(self, *, fail_resolved: bool = False):
        self.calls = []
        self.tolerant_calls = 0
        self.fail_resolved = fail_resolved

    def __len__(self):
        return 3

    def get_strict(self, index):
        self.calls.append(index)
        if index == 0:
            return _temporal_sample(padded=True)
        if self.fail_resolved:
            raise ValueError(f"decode failure at resolved index {index}")
        return _temporal_sample(padded=False)

    def __getitem__(self, index):
        self.tolerant_calls += 1
        return _temporal_sample(padded=False)


def _strict_robot_dataset(lower_dataset):
    dataset = RobotVideoDataset.__new__(RobotVideoDataset)
    dataset.lerobot_dataset = lower_dataset
    dataset.skip_padding_as_possible = True
    dataset.max_padding_retry = 1
    dataset.strict_requested_sample_count = 0
    dataset.strict_resolved_sample_count = 0
    dataset.padding_remap_count = 0
    dataset.strict_lower_fetch_error_count = 0
    dataset.padding_remap_records = []
    return dataset


def test_strict_path_preserves_legal_padding_remap_and_records_resolution():
    lower = _NestedDataset()
    dataset = _strict_robot_dataset(lower)

    _, resolved_index = dataset._fetch_temporal_sample(
        0,
        strict_lower_fetch=True,
    )

    assert lower.calls[0] == 0
    assert resolved_index != 0
    assert lower.calls[-1] == resolved_index
    assert lower.tolerant_calls == 0
    assert dataset.padding_remap_count == 1
    assert dataset.padding_remap_records == [
        {
            "requested_index": 0,
            "resolved_index": resolved_index,
            "reason": "temporal_padding",
        }
    ]


def test_strict_path_propagates_resolved_decode_error_without_substitution():
    lower = _NestedDataset(fail_resolved=True)
    dataset = _strict_robot_dataset(lower)

    with pytest.raises(ValueError, match="decode failure at resolved index"):
        dataset._fetch_temporal_sample(0, strict_lower_fetch=True)

    assert len(lower.calls) == 2
    assert lower.tolerant_calls == 0
    assert dataset.strict_lower_fetch_error_count == 1
    assert dataset.strict_resolved_sample_count == 0


def test_base_training_fetch_retries_but_strict_fetch_is_one_shot(monkeypatch):
    dataset = BaseLerobotDataset.__new__(BaseLerobotDataset)
    dataset.multi_dataset = SimpleNamespace(num_frames=2)
    load_calls = []

    def load_once(index):
        load_calls.append(index)
        if index == 0:
            raise ValueError("corrupt frame")
        return {"loaded_index": index}

    monkeypatch.setattr(dataset, "_load_lerobot_sample_once", load_once)
    monkeypatch.setattr(
        dataset,
        "_build_sample",
        lambda index, sample: {"idx": index, **sample},
    )
    monkeypatch.setattr(base_module.np.random, "randint", lambda size: 1)

    tolerant = dataset[0]

    assert tolerant["idx"] == 1
    assert load_calls == [0, 1]

    load_calls.clear()
    with pytest.raises(ValueError, match="corrupt frame"):
        dataset.get_strict(0)
    assert load_calls == [0]
