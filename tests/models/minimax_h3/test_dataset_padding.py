import pytest
import torch

from fastwam.datasets.padding import (
    fetch_unpadded_temporal_sample,
    validate_replacement_rate,
)


def padded_sample(is_padded):
    mask = torch.tensor([False, bool(is_padded)])
    return {
        "action_is_pad": mask,
        "image_is_pad": mask,
        "proprio_is_pad": mask,
    }


def test_padding_retry_returns_only_a_fully_unpadded_sample():
    samples = {
        0: padded_sample(True),
        1: padded_sample(False),
    }

    sample = fetch_unpadded_temporal_sample(
        samples.__getitem__,
        initial_index=0,
        dataset_size=2,
        max_retries=1,
        random_index=lambda _: 1,
    )

    assert sample is samples[1]


def test_padding_retry_never_silently_returns_padding_after_budget_exhausted():
    calls = 0

    def fetch(_):
        nonlocal calls
        calls += 1
        return padded_sample(True)

    with pytest.raises(RuntimeError, match="unpadded"):
        fetch_unpadded_temporal_sample(
            fetch,
            initial_index=0,
            dataset_size=3,
            max_retries=2,
            random_index=lambda _: 1,
        )

    assert calls == 3


def test_dataset_replacement_rate_fails_above_safety_limit():
    with pytest.raises(RuntimeError, match="replacement rate exceeded"):
        validate_replacement_rate(
            replacement_count=2,
            sample_attempt_count=1000,
            max_replacement_rate=0.001,
            replacement_rate_warmup=1000,
            exception_types={"ValueError": 1, "TypeError": 1},
        )
