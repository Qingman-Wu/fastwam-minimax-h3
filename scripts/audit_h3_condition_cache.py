"""Audit H3 cache files and every training-dataset cache reference."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import hydra
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig

from fastwam.datasets.h3_condition_cache import (
    H3_CACHE_MANIFEST,
    _cache_digest,
    _load_cache_manifest,
    load_h3_condition_cache_file,
)
from fastwam.utils.config_resolvers import register_default_resolvers


register_default_resolvers()


def _distributed_context() -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("gloo")
    return rank, world_size


def _gather_objects(value, world_size: int):
    if world_size == 1:
        return [value]
    gathered = [None] * world_size
    dist.all_gather_object(gathered, value)
    return gathered


def _merge_histograms(values) -> dict[str, int]:
    merged = Counter()
    for value in values:
        merged.update({int(key): int(count) for key, count in value.items()})
    return {str(key): merged[key] for key in sorted(merged)}


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    audit_cfg = cfg.get("cache_audit", {})
    rank, world_size = _distributed_context()
    cache_dir = Path(str(cfg.data.train.h3_condition_cache_dir)).expanduser()
    output_path = Path(
        str(
            audit_cfg.get(
                "output_path",
                "artifacts/h3_condition_cache_audit.json",
            )
        )
    )
    verify_dataset_references = bool(
        audit_cfg.get("verify_dataset_references", True)
    )
    max_samples_value = audit_cfg.get("max_samples")
    max_samples = (
        None if max_samples_value is None else int(max_samples_value)
    )
    max_cache_files_value = audit_cfg.get("max_cache_files")
    max_cache_files = (
        None
        if max_cache_files_value is None
        else int(max_cache_files_value)
    )
    expected_sample_count_value = audit_cfg.get("expected_sample_count", 277713)
    expected_sample_count = (
        None
        if expected_sample_count_value is None
        else int(expected_sample_count_value)
    )
    max_recorded_errors = int(audit_cfg.get("max_recorded_errors", 20))

    manifest = _load_cache_manifest(cache_dir)
    cache_files = sorted(cache_dir.glob("*.h3-qwen-prenorm-layer50-v3.pt"))
    scanned_cache_files = (
        cache_files
        if max_cache_files is None
        else cache_files[:max_cache_files]
    )
    local_file_histogram = Counter()
    local_file_errors = []
    for file_index in range(rank, len(scanned_cache_files), world_size):
        path = scanned_cache_files[file_index]
        try:
            payload = load_h3_condition_cache_file(path, manifest=manifest)
            local_file_histogram[int(payload["prompt_embeds"].shape[0])] += 1
        except Exception as error:
            if len(local_file_errors) < max_recorded_errors:
                local_file_errors.append(
                    {
                        "path": str(path),
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )

    file_histograms = _gather_objects(dict(local_file_histogram), world_size)
    file_errors = [
        error
        for rank_errors in _gather_objects(local_file_errors, world_size)
        for error in rank_errors
    ]
    file_histogram = _merge_histograms(file_histograms)

    reference_histogram = {}
    reference_errors = []
    referenced_files: set[str] = set()
    dataset_length = None
    audited_sample_count = 0
    replacement_count = 0
    sample_attempt_count = 0
    if verify_dataset_references:
        dataset = instantiate(cfg.data.train)
        dataset_length = len(dataset)
        audit_length = (
            dataset_length
            if max_samples is None
            else min(dataset_length, max_samples)
        )
        local_reference_histogram = Counter()
        local_reference_errors = []
        local_referenced_files = set()
        local_sample_count = 0
        for sample_index in range(rank, audit_length, world_size):
            try:
                sample = dataset[sample_index]
                row_count = int(sample["prompt_embeds"].shape[0])
                local_reference_histogram[row_count] += 1
                digest = _cache_digest(
                    sample["video"][:, 0],
                    sample["prompt"],
                    manifest,
                )
                local_referenced_files.add(
                    f"{digest}.h3-qwen-prenorm-layer50-v3.pt"
                )
                local_sample_count += 1
            except Exception as error:
                if len(local_reference_errors) < max_recorded_errors:
                    local_reference_errors.append(
                        {
                            "sample_index": sample_index,
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )
        reference_histogram = _merge_histograms(
            _gather_objects(dict(local_reference_histogram), world_size)
        )
        reference_errors = [
            error
            for rank_errors in _gather_objects(local_reference_errors, world_size)
            for error in rank_errors
        ]
        for rank_files in _gather_objects(local_referenced_files, world_size):
            referenced_files.update(rank_files)
        audited_sample_count = sum(
            _gather_objects(local_sample_count, world_size)
        )
        replacement_count = sum(
            _gather_objects(
                int(getattr(dataset, "replacement_count", 0)),
                world_size,
            )
        )
        sample_attempt_count = sum(
            _gather_objects(
                int(getattr(dataset, "sample_attempt_count", 0)),
                world_size,
            )
        )

    all_cache_names = {path.name for path in cache_files}
    missing_referenced_files = sorted(referenced_files - all_cache_names)
    complete_reference_audit = (
        verify_dataset_references
        and max_samples is None
        and audited_sample_count == dataset_length
    )
    complete_file_audit = max_cache_files is None
    orphan_cache_file_count = (
        len(all_cache_names - referenced_files)
        if complete_reference_audit
        else None
    )
    passed = not file_errors and not reference_errors and not missing_referenced_files
    if expected_sample_count is not None and dataset_length is not None:
        passed = passed and dataset_length == expected_sample_count
    if complete_reference_audit:
        passed = passed and audited_sample_count == dataset_length
    formal_gate_passed = (
        passed
        and complete_file_audit
        and complete_reference_audit
    )

    report = {
        "schema_version": 1,
        "passed": passed,
        "formal_gate_passed": formal_gate_passed,
        "world_size": world_size,
        "cache_dir": str(cache_dir.resolve()),
        "manifest_path": str((cache_dir / H3_CACHE_MANIFEST).resolve()),
        "manifest": manifest,
        "unique_cache_file_count": len(cache_files),
        "audited_unique_cache_file_count": len(scanned_cache_files),
        "complete_file_audit": complete_file_audit,
        "unique_file_row_length_histogram": file_histogram,
        "unique_file_max_rows": (
            max(map(int, file_histogram)) if file_histogram else None
        ),
        "file_errors": file_errors,
        "dataset_reference_audit_enabled": verify_dataset_references,
        "dataset_length": dataset_length,
        "expected_sample_count": expected_sample_count,
        "audited_sample_count": audited_sample_count,
        "complete_reference_audit": complete_reference_audit,
        "referenced_unique_file_count": len(referenced_files),
        "reference_row_length_histogram": reference_histogram,
        "reference_max_rows": (
            max(map(int, reference_histogram)) if reference_histogram else None
        ),
        "missing_referenced_files": missing_referenced_files[
            :max_recorded_errors
        ],
        "orphan_cache_file_count": orphan_cache_file_count,
        "reference_errors": reference_errors,
        "replacement_count": replacement_count,
        "sample_attempt_count": sample_attempt_count,
    }
    if rank == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"Wrote cache audit to {output_path}")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    if not passed:
        raise RuntimeError("H3 condition cache audit failed; inspect the JSON report")


if __name__ == "__main__":
    main()
