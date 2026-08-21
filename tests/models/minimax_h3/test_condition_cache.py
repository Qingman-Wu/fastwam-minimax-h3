import torch

from fastwam.datasets.h3_condition_cache import (
    collate_h3_condition_samples,
    h3_condition_cache_path,
    initialize_h3_condition_cache,
    load_h3_condition_cache,
    save_h3_condition_cache,
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
