"""On-disk H3 layer-50 Qwen condition cache and variable-length collation."""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import default_collate


H3_QWEN_WIDTH = 5120
H3_QWEN_LAYER = 50
H3_CACHE_KEYS = (
    "prompt_embeds",
    "prompt_token_tags",
    "prompt_attention_mask",
)


def _cache_digest(first_frame: torch.Tensor, instruction: str) -> str:
    if first_frame.ndim != 3 or first_frame.shape[0] != 3:
        raise ValueError(
            f"H3 cache first_frame must be [3,H,W], got {tuple(first_frame.shape)}"
        )
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("H3 cache instruction must be a non-empty string")
    pixels = (
        ((first_frame.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
        .contiguous()
    )
    digest = hashlib.sha256()
    digest.update(f"h3-qwen-layer-{H3_QWEN_LAYER}\0".encode())
    digest.update(instruction.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(tuple(pixels.shape)).encode())
    digest.update(bytes(pixels.untyped_storage()))
    return digest.hexdigest()


def h3_condition_cache_path(
    cache_dir: str | Path,
    first_frame: torch.Tensor,
    instruction: str,
) -> Path:
    digest = _cache_digest(first_frame, instruction)
    return Path(cache_dir).expanduser() / f"{digest}.h3-qwen-layer50.pt"


def save_h3_condition_cache(
    cache_dir: str | Path,
    *,
    first_frame: torch.Tensor,
    instruction: str,
    embeddings: torch.Tensor,
    token_tags: torch.Tensor,
) -> Path:
    if embeddings.ndim != 2 or embeddings.shape[-1] != H3_QWEN_WIDTH:
        raise ValueError(
            f"H3 cached embeddings must be [L,{H3_QWEN_WIDTH}], got "
            f"{tuple(embeddings.shape)}"
        )
    if token_tags.shape != embeddings.shape[:1]:
        raise ValueError("H3 cached token_tags must match embedding rows")
    token_tags = token_tags.detach().cpu().to(torch.long)
    if not torch.logical_or(token_tags == 0, token_tags == 1).all():
        raise ValueError("H3 cached token_tags must contain only 0 or 1")
    path = h3_condition_cache_path(cache_dir, first_frame, instruction)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "hidden_layer": H3_QWEN_LAYER,
        "prompt_embeds": embeddings.detach().cpu(),
        "prompt_token_tags": token_tags,
    }
    temporary = path.parent / f".{path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_h3_condition_cache(
    cache_dir: str | Path,
    *,
    first_frame: torch.Tensor,
    instruction: str,
) -> dict[str, torch.Tensor]:
    path = h3_condition_cache_path(cache_dir, first_frame, instruction)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing native H3 condition cache: {path}. "
            "Run scripts/precompute_h3_conditions.py first."
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1 or payload.get("hidden_layer") != 50:
        raise ValueError(f"Incompatible H3 condition cache schema in {path}")
    embeddings = payload["prompt_embeds"]
    token_tags = payload["prompt_token_tags"].to(torch.long)
    if embeddings.ndim != 2 or embeddings.shape[-1] != H3_QWEN_WIDTH:
        raise ValueError(f"Invalid H3 cached embedding shape in {path}")
    if token_tags.shape != embeddings.shape[:1]:
        raise ValueError(f"Invalid H3 cached tag shape in {path}")
    if not torch.logical_or(token_tags == 0, token_tags == 1).all():
        raise ValueError(f"Invalid H3 cached modality tags in {path}")
    return {
        "prompt_embeds": embeddings,
        "prompt_token_tags": token_tags,
        "prompt_attention_mask": torch.ones(
            embeddings.shape[0], dtype=torch.bool
        ),
    }


def collate_h3_condition_samples(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Default-collate a batch while padding only cached Qwen row sequences."""

    if not samples:
        raise ValueError("cannot collate an empty sample list")
    present = [all(key in sample for key in H3_CACHE_KEYS) for sample in samples]
    if not any(present):
        return default_collate(samples)
    if not all(present):
        raise ValueError("every batch sample must contain all H3 condition cache keys")

    ordinary = [
        {key: value for key, value in sample.items() if key not in H3_CACHE_KEYS}
        for sample in samples
    ]
    batch = default_collate(ordinary)
    batch["prompt_embeds"] = pad_sequence(
        [sample["prompt_embeds"] for sample in samples], batch_first=True
    )
    batch["prompt_token_tags"] = pad_sequence(
        [sample["prompt_token_tags"] for sample in samples],
        batch_first=True,
        padding_value=0,
    )
    batch["prompt_attention_mask"] = pad_sequence(
        [sample["prompt_attention_mask"] for sample in samples],
        batch_first=True,
        padding_value=False,
    )
    return batch
