"""Dependency-light temporal padding selection helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


TEMPORAL_PADDING_KEYS = (
    "action_is_pad",
    "image_is_pad",
    "proprio_is_pad",
)


def _has_temporal_padding(sample: Mapping[str, Any]) -> bool:
    missing = [key for key in TEMPORAL_PADDING_KEYS if key not in sample]
    if missing:
        raise KeyError(f"sample is missing temporal padding masks: {missing}")
    return any(bool(sample[key].any().item()) for key in TEMPORAL_PADDING_KEYS)


def fetch_unpadded_temporal_sample(
    fetch: Callable[[int], Mapping[str, Any]],
    *,
    initial_index: int,
    dataset_size: int,
    max_retries: int,
    random_index: Callable[[int], int],
) -> Mapping[str, Any]:
    """Fetch a sample without padding or fail instead of returning a bad target."""

    dataset_size = int(dataset_size)
    max_retries = int(max_retries)
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    sample_index = int(initial_index)
    for attempt in range(max_retries + 1):
        sample = fetch(sample_index)
        if not _has_temporal_padding(sample):
            return sample
        if attempt < max_retries:
            sample_index = int(random_index(dataset_size))
            if not 0 <= sample_index < dataset_size:
                raise IndexError(
                    f"random_index returned {sample_index} for size {dataset_size}"
                )
    raise RuntimeError(
        "Unable to fetch an unpadded temporal sample after "
        f"{max_retries + 1} attempts"
    )
