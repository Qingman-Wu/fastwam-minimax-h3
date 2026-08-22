import hashlib
import json

import pytest
import torch

from fastwam.datasets.h3_condition_cache import (
    collate_h3_condition_samples,
    h3_condition_cache_path,
    initialize_h3_condition_cache,
    load_h3_condition_cache,
    load_h3_condition_cache_file,
    save_h3_condition_cache,
)
from fastwam.models.minimax_h3.text_encoder import (
    H3_QWEN_ENCODER_SIGNATURE,
    H3_QWEN_PRESENTATION_SIGNATURE,
)
from fastwam.models.minimax_h3.video_dit import (
    H3_CONDITION_REFINER_IMPLEMENTATION_SIGNATURE,
)


def initialize(cache_dir, *, qwen="sha256:qwen-a", processor="sha256:processor-a"):
    initialize_h3_condition_cache(
        cache_dir,
        qwen_checkpoint_fingerprint=qwen,
        processor_fingerprint=processor,
    )


def test_h3_condition_cache_is_keyed_by_first_frame_and_instruction(tmp_path):
    frame = torch.zeros(3, 4, 4)
    initialize(tmp_path)

    first = h3_condition_cache_path(tmp_path, frame, "pick up cup")
    changed_text = h3_condition_cache_path(tmp_path, frame, "open drawer")
    changed_frame = h3_condition_cache_path(
        tmp_path, frame.index_fill(1, torch.tensor([0]), 1.0), "pick up cup"
    )

    assert first != changed_text
    assert first != changed_frame
    assert first.name.endswith(".h3-qwen-prenorm-layer50-v3.pt")


def test_h3_condition_cache_round_trip_preserves_native_rows(tmp_path):
    frame = torch.zeros(3, 4, 4)
    embeddings = torch.randn(3, 5120, dtype=torch.bfloat16)
    tags = torch.tensor([1, 0, 1])
    initialize(tmp_path)

    save_h3_condition_cache(
        tmp_path,
        first_frame=frame,
        instruction="move",
        embeddings=embeddings,
        token_tags=tags,
    )
    loaded = load_h3_condition_cache(
        tmp_path, first_frame=frame, instruction="move"
    )

    assert torch.equal(loaded["prompt_embeds"], embeddings)
    assert torch.equal(loaded["prompt_token_tags"], tags)
    assert loaded["prompt_attention_mask"].tolist() == [True, True, True]


def test_h3_condition_cache_miss_exposes_exact_missing_filename(tmp_path):
    initialize(tmp_path)

    with pytest.raises(FileNotFoundError) as captured:
        load_h3_condition_cache(
            tmp_path,
            first_frame=torch.zeros(3, 4, 4),
            instruction="missing",
        )

    assert captured.value.filename.endswith(
        ".h3-qwen-prenorm-layer50-v3.pt"
    )


def test_h3_condition_cache_manifest_rejects_different_qwen_weights(tmp_path):
    initialize(tmp_path)

    try:
        initialize_h3_condition_cache(
            tmp_path,
            qwen_checkpoint_fingerprint="sha256:qwen-b",
            processor_fingerprint="sha256:processor-a",
        )
    except ValueError as error:
        assert "manifest" in str(error)
    else:
        raise AssertionError("Different Qwen weights must invalidate the cache")


def test_direct_cache_file_loader_strictly_validates_manifest(tmp_path):
    frame = torch.zeros(3, 4, 4)
    manifest = initialize_h3_condition_cache(
        tmp_path,
        qwen_checkpoint_fingerprint="sha256:qwen-a",
        processor_fingerprint="sha256:processor-a",
    )
    path = save_h3_condition_cache(
        tmp_path,
        first_frame=frame,
        instruction="move",
        embeddings=torch.zeros(2, 5120),
        token_tags=torch.tensor([1, 0]),
    )
    incompatible = {**manifest, "processor_fingerprint": "sha256:other"}

    try:
        load_h3_condition_cache_file(path, manifest=incompatible)
    except ValueError as error:
        assert "schema" in str(error)
    else:
        raise AssertionError("Direct file audit must reject a manifest mismatch")


def _make_cache_file(tmp_path):
    manifest = initialize_h3_condition_cache(
        tmp_path,
        qwen_checkpoint_fingerprint="sha256:qwen-a",
        processor_fingerprint="sha256:processor-a",
    )
    path = save_h3_condition_cache(
        tmp_path,
        first_frame=torch.zeros(3, 4, 4),
        instruction="move",
        embeddings=torch.zeros(2, 5120),
        token_tags=torch.tensor([1, 0]),
    )
    return path, manifest


def test_direct_cache_file_loader_rejects_empty_embeddings(tmp_path):
    path, manifest = _make_cache_file(tmp_path)
    payload = torch.load(path, weights_only=True)
    payload["prompt_embeds"] = torch.empty(0, 5120)
    payload["prompt_token_tags"] = torch.empty(0, dtype=torch.long)
    torch.save(payload, path)

    with pytest.raises(ValueError, match="embedding shape"):
        load_h3_condition_cache_file(path, manifest=manifest)


def test_direct_cache_file_loader_rejects_nonfinite_embeddings(tmp_path):
    path, manifest = _make_cache_file(tmp_path)
    payload = torch.load(path, weights_only=True)
    payload["prompt_embeds"][0, 0] = float("nan")
    torch.save(payload, path)

    with pytest.raises(ValueError, match="non-finite"):
        load_h3_condition_cache_file(path, manifest=manifest)


def test_direct_cache_file_loader_rejects_float_tags_before_cast(tmp_path):
    path, manifest = _make_cache_file(tmp_path)
    payload = torch.load(path, weights_only=True)
    payload["prompt_token_tags"] = torch.tensor([1.0, 0.5])
    torch.save(payload, path)

    with pytest.raises(ValueError, match="integer dtype"):
        load_h3_condition_cache_file(path, manifest=manifest)


def test_h3_condition_collate_pads_variable_qwen_lengths_without_valid_padding():
    common = {
        "video": torch.zeros(3, 5, 4, 4),
        "action": torch.zeros(4, 2),
    }
    first = {
        **common,
        "prompt_embeds": torch.ones(2, 5120),
        "prompt_token_tags": torch.tensor([1, 0]),
        "prompt_attention_mask": torch.tensor([True, True]),
    }
    second = {
        **common,
        "prompt_embeds": torch.full((3, 5120), 2.0),
        "prompt_token_tags": torch.tensor([1, 0, 1]),
        "prompt_attention_mask": torch.tensor([True, True, True]),
    }

    batch = collate_h3_condition_samples([first, second])

    assert batch["prompt_embeds"].shape == (2, 3, 5120)
    assert batch["prompt_attention_mask"].tolist() == [
        [True, True, False],
        [True, True, True],
    ]
    assert torch.equal(batch["prompt_embeds"][0, 2], torch.zeros(5120))


def test_post_refiner_cache_requires_bfloat16_and_content_checksum(tmp_path):
    source_manifest = initialize_h3_condition_cache(
        tmp_path / "source",
        qwen_checkpoint_fingerprint="sha256:qwen-a",
        processor_fingerprint="sha256:processor-a",
    )
    refined_dir = tmp_path / "refined"
    refined_dir.mkdir()
    manifest = {
        "schema_version": 4,
        "source_manifest": source_manifest,
        "source_manifest_sha256": "sha256:source",
        "refiner_fingerprint": "sha256:refiner",
        "refiner_artifact_stat_signature": "sha256:refiner-stat",
        "refiner_implementation_signature": (
            H3_CONDITION_REFINER_IMPLEMENTATION_SIGNATURE
        ),
        "qwen_weight_shards_fingerprint": "sha256:qwen-shards",
        "qwen_weight_shards_stat_signature": "sha256:qwen-stat",
        "qwen_encoder_signature": H3_QWEN_ENCODER_SIGNATURE,
        "qwen_presentation_signature": H3_QWEN_PRESENTATION_SIGNATURE,
        "embedding_width": 5376,
        "embedding_dtype": "torch.bfloat16",
        "implementation_signature": "fastwam-h3-post-refiner-v3",
    }
    (refined_dir / "h3-post-refiner-cache-manifest.json").write_text(
        json.dumps(manifest)
    )
    embeddings = torch.randn(2, 5376, dtype=torch.bfloat16)
    tags = torch.tensor([0, 1])
    digest = hashlib.sha256()
    for tensor in (embeddings, tags):
        value = tensor.contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(memoryview(value.view(torch.uint8).numpy()))
    path = refined_dir / "sample.h3-post-refiner-v4.pt"
    torch.save(
        {
            "schema_version": 4,
            "refiner_fingerprint": "sha256:refiner",
            "prompt_embeds": embeddings,
            "prompt_token_tags": tags,
            "payload_sha256": f"sha256:{digest.hexdigest()}",
        },
        path,
    )

    loaded = load_h3_condition_cache_file(path, manifest=manifest)
    assert loaded["prompt_embeds"].shape == (2, 5376)

    payload = torch.load(path, weights_only=True)
    payload["prompt_embeds"][0, 0] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="checksum"):
        load_h3_condition_cache_file(path, manifest=manifest)
