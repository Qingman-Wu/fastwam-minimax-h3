"""Dependency-light temporal padding selection helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


TEMPORAL_PADDING_KEYS = (
    "action_is_pad",
    "image_is_pad",
    "proprio_is_pad",
)


def validate_replacement_rate(
    *,
    replacement_count: int,
    sample_attempt_count: int,
    max_replacement_rate: float,
    replacement_rate_warmup: int,
    exception_types: Mapping[str, int],
) -> float:
    """Return the replacement rate or fail when corruption is systematic."""

    replacement_rate = replacement_count / sample_attempt_count
    if (
        sample_attempt_count >= replacement_rate_warmup
        and replacement_rate > max_replacement_rate
    ):
        raise RuntimeError(
            "Dataset replacement rate exceeded the configured safety limit: "
            f"{replacement_count}/{sample_attempt_count} "
            f"({replacement_rate:.4%}) > {max_replacement_rate:.4%}; "
            f"exceptions={dict(exception_types)}"
        )
    return replacement_rate


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

    sample, _ = fetch_unpadded_temporal_sample_with_index(
        fetch,
        initial_index=initial_index,
        dataset_size=dataset_size,
        max_retries=max_retries,
        random_index=random_index,
    )
    return sample


def fetch_unpadded_temporal_sample_with_index(
    fetch: Callable[[int], Mapping[str, Any]],
    *,
    initial_index: int,
    dataset_size: int,
    max_retries: int,
    random_index: Callable[[int], int],
) -> tuple[Mapping[str, Any], int]:
    """Fetch an unpadded sample and return its effective dataset index."""

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
            return sample, sample_index
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
