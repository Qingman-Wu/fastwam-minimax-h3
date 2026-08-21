import pytest

from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from scripts.audit_h3_condition_cache import _evaluate_gates


def test_strict_dataset_getter_does_not_replace_a_failing_original_index(
    monkeypatch,
):
    dataset = RobotVideoDataset.__new__(RobotVideoDataset)
    dataset.replacement_count = 0
    dataset.sample_attempt_count = 0

    def fail_original(index, **kwargs):
        assert kwargs == {"strict_lower_fetch": True}
        raise ValueError(f"invalid original index {index}")

    monkeypatch.setattr(dataset, "_get", fail_original)

    with pytest.raises(ValueError, match="original index 7"):
        dataset.get_strict(7)

    assert dataset.replacement_count == 0
    assert dataset.sample_attempt_count == 0


def test_row_141_passes_cache_integrity_but_blocks_b2_memory_gate():
    gates = _evaluate_gates(
        scan_passed=True,
        complete_file_audit=True,
        complete_reference_audit=True,
        replacement_count=0,
        strict_lower_fetch_error_count=0,
        max_rows=141,
        b2_verified_max_rows=140,
    )

    assert gates["cache_integrity_passed"]
    assert gates["formal_cache_gate_passed"]
    assert not gates["b2_memory_gate_passed"]
    assert gates["requires_memory_resmoke"]
    assert not gates["formal_gate_passed"]


def test_replacement_count_blocks_formal_cache_gate():
    gates = _evaluate_gates(
        scan_passed=True,
        complete_file_audit=True,
        complete_reference_audit=True,
        replacement_count=1,
        strict_lower_fetch_error_count=0,
        max_rows=140,
        b2_verified_max_rows=140,
    )

    assert not gates["cache_integrity_passed"]
    assert not gates["formal_gate_passed"]


def test_lower_fetch_error_blocks_formal_cache_gate():
    gates = _evaluate_gates(
        scan_passed=True,
        complete_file_audit=True,
        complete_reference_audit=True,
        replacement_count=0,
        strict_lower_fetch_error_count=1,
        max_rows=140,
        b2_verified_max_rows=140,
    )

    assert not gates["cache_integrity_passed"]
    assert not gates["formal_gate_passed"]
