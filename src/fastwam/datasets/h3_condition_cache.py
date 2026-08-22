"""On-disk H3 layer-50 Qwen condition cache and variable-length collation."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import default_collate

from fastwam.models.minimax_h3.text_encoder import (
    H3_QWEN_ENCODER_SIGNATURE,
    H3_QWEN_PRESENTATION_SIGNATURE,
)
from fastwam.models.minimax_h3.video_dit import (
    H3_CONDITION_REFINER_IMPLEMENTATION_SIGNATURE,
)


H3_QWEN_WIDTH = 5120
H3_REFINED_WIDTH = 5376
H3_QWEN_LAYER = 50
H3_CACHE_SCHEMA_VERSION = 3
H3_CACHE_MANIFEST = "h3-qwen-cache-manifest.json"
H3_REFINED_CACHE_SCHEMA_VERSION = 4
H3_REFINED_CACHE_MANIFEST = "h3-post-refiner-cache-manifest.json"
H3_CACHE_KEYS = (
    "prompt_embeds",
    "prompt_token_tags",
    "prompt_attention_mask",
)


def initialize_h3_condition_cache(
    cache_dir: str | Path,
    *,
    qwen_checkpoint_fingerprint: str,
    processor_fingerprint: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / H3_CACHE_MANIFEST
    manifest = {
        "schema_version": H3_CACHE_SCHEMA_VERSION,
        "hidden_layer": H3_QWEN_LAYER,
        "encoder_signature": H3_QWEN_ENCODER_SIGNATURE,
        "qwen_checkpoint_fingerprint": str(qwen_checkpoint_fingerprint),
        "processor_fingerprint": str(processor_fingerprint),
    }
    if manifest_path.is_file():
        current = json.loads(manifest_path.read_text())
        if current == manifest:
            return manifest
        if not overwrite:
            raise ValueError(
                "H3 condition cache manifest does not match the requested "
                "Qwen/processor artifacts"
            )
    temporary = cache_dir / f".{H3_CACHE_MANIFEST}.tmp.{uuid.uuid4().hex}"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    return manifest


def _load_cache_manifest(cache_dir: str | Path) -> dict[str, Any]:
    cache_dir = Path(cache_dir).expanduser()
    refined_path = cache_dir / H3_REFINED_CACHE_MANIFEST
    manifest_path = (
        refined_path if refined_path.is_file() else cache_dir / H3_CACHE_MANIFEST
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing H3 condition cache manifest: {manifest_path}. "
            "Regenerate the cache with scripts/precompute_h3_conditions.py."
        )
    manifest = json.loads(manifest_path.read_text())
    schema = manifest.get("schema_version")
    schema3_valid = (
        schema == H3_CACHE_SCHEMA_VERSION
        and manifest.get("hidden_layer") == H3_QWEN_LAYER
        and manifest.get("encoder_signature") == H3_QWEN_ENCODER_SIGNATURE
        and bool(manifest.get("qwen_checkpoint_fingerprint"))
        and bool(manifest.get("processor_fingerprint"))
    )
    source = manifest.get("source_manifest", {})
    schema4_valid = (
        schema == H3_REFINED_CACHE_SCHEMA_VERSION
        and manifest.get("embedding_width") == H3_REFINED_WIDTH
        and manifest.get("embedding_dtype") == "torch.bfloat16"
        and bool(manifest.get("refiner_fingerprint"))
        and bool(manifest.get("refiner_artifact_stat_signature"))
        and manifest.get("refiner_implementation_signature")
        == H3_CONDITION_REFINER_IMPLEMENTATION_SIGNATURE
        and bool(manifest.get("qwen_weight_shards_fingerprint"))
        and bool(manifest.get("qwen_weight_shards_stat_signature"))
        and manifest.get("qwen_encoder_signature") == H3_QWEN_ENCODER_SIGNATURE
        and manifest.get("qwen_presentation_signature")
        == H3_QWEN_PRESENTATION_SIGNATURE
        and manifest.get("implementation_signature")
        == "fastwam-h3-post-refiner-v3"
        and source.get("schema_version") == H3_CACHE_SCHEMA_VERSION
        and source.get("hidden_layer") == H3_QWEN_LAYER
        and source.get("encoder_signature") == H3_QWEN_ENCODER_SIGNATURE
        and bool(source.get("qwen_checkpoint_fingerprint"))
        and bool(source.get("processor_fingerprint"))
    )
    if not (schema3_valid or schema4_valid):
        raise ValueError(f"Incompatible H3 condition cache manifest: {manifest_path}")
    return manifest


def _cache_digest(
    first_frame: torch.Tensor,
    instruction: str,
    manifest: dict[str, Any],
) -> str:
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
    digest.update(
        (
            f"h3-cache-v{H3_CACHE_SCHEMA_VERSION}\0"
            f"{H3_QWEN_ENCODER_SIGNATURE}\0"
            f"{manifest['qwen_checkpoint_fingerprint']}\0"
            f"{manifest['processor_fingerprint']}\0"
        ).encode()
    )
    digest.update(instruction.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(tuple(pixels.shape)).encode())
    digest.update(memoryview(pixels.view(torch.uint8).numpy()))
    return digest.hexdigest()


def h3_condition_cache_path(
    cache_dir: str | Path,
    first_frame: torch.Tensor,
    instruction: str,
) -> Path:
    manifest = _load_cache_manifest(cache_dir)
    if manifest["schema_version"] == H3_REFINED_CACHE_SCHEMA_VERSION:
        digest = _cache_digest(first_frame, instruction, manifest["source_manifest"])
        suffix = ".h3-post-refiner-v4.pt"
    else:
        digest = _cache_digest(first_frame, instruction, manifest)
        suffix = ".h3-qwen-prenorm-layer50-v3.pt"
    return Path(cache_dir).expanduser() / f"{digest}{suffix}"


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
    manifest = _load_cache_manifest(cache_dir)
    path = h3_condition_cache_path(cache_dir, first_frame, instruction)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": H3_CACHE_SCHEMA_VERSION,
        "hidden_layer": H3_QWEN_LAYER,
        "encoder_signature": H3_QWEN_ENCODER_SIGNATURE,
        "qwen_checkpoint_fingerprint": manifest["qwen_checkpoint_fingerprint"],
        "processor_fingerprint": manifest["processor_fingerprint"],
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
    manifest = _load_cache_manifest(cache_dir)
    path = h3_condition_cache_path(cache_dir, first_frame, instruction)
    if not path.is_file():
        error = FileNotFoundError(
            f"Missing native H3 condition cache: {path}. "
            "Run scripts/precompute_h3_conditions.py first."
        )
        error.filename = str(path)
        raise error
    return load_h3_condition_cache_file(path, manifest=manifest)


def load_h3_condition_cache_file(
    path: str | Path,
    *,
    manifest: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Load and validate one cache file without recomputing its content key."""

    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if manifest["schema_version"] == H3_REFINED_CACHE_SCHEMA_VERSION:
        if (
            payload.get("schema_version") != H3_REFINED_CACHE_SCHEMA_VERSION
            or payload.get("refiner_fingerprint")
            != manifest["refiner_fingerprint"]
        ):
            raise ValueError(f"Incompatible H3 post-Refiner cache schema in {path}")
        embeddings = payload["prompt_embeds"]
        raw_token_tags = payload["prompt_token_tags"]
        if embeddings.dtype != torch.bfloat16:
            raise ValueError(
                f"H3 post-Refiner embeddings must be bfloat16 in {path}"
            )
        digest = hashlib.sha256()
        for tensor in (embeddings, raw_token_tags):
            value = tensor.detach().cpu().contiguous()
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(memoryview(value.view(torch.uint8).numpy()))
        if payload.get("payload_sha256") != f"sha256:{digest.hexdigest()}":
            raise ValueError(f"H3 post-Refiner payload checksum mismatch in {path}")
        expected_width = H3_REFINED_WIDTH
    else:
        expected_width = H3_QWEN_WIDTH
        embeddings = payload["prompt_embeds"]
        raw_token_tags = payload["prompt_token_tags"]
        if (
        payload.get("schema_version") != H3_CACHE_SCHEMA_VERSION
        or payload.get("hidden_layer") != H3_QWEN_LAYER
        or payload.get("encoder_signature") != H3_QWEN_ENCODER_SIGNATURE
        or payload.get("qwen_checkpoint_fingerprint")
        != manifest["qwen_checkpoint_fingerprint"]
        or payload.get("processor_fingerprint")
        != manifest["processor_fingerprint"]
        ):
            raise ValueError(f"Incompatible H3 condition cache schema in {path}")
    if (
        not isinstance(embeddings, torch.Tensor)
        or embeddings.ndim != 2
        or embeddings.shape[0] <= 0
        or embeddings.shape[-1] != expected_width
    ):
        raise ValueError(f"Invalid H3 cached embedding shape in {path}")
    if not torch.is_floating_point(embeddings):
        raise ValueError(f"H3 cached embeddings must be floating point in {path}")
    if not torch.isfinite(embeddings).all():
        raise ValueError(f"H3 cached embeddings contain non-finite values in {path}")
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if (
        not isinstance(raw_token_tags, torch.Tensor)
        or raw_token_tags.dtype not in integer_dtypes
    ):
        raise ValueError(f"H3 cached modality tags must use an integer dtype in {path}")
    if raw_token_tags.shape != embeddings.shape[:1]:
        raise ValueError(f"Invalid H3 cached tag shape in {path}")
    if not torch.logical_or(raw_token_tags == 0, raw_token_tags == 1).all():
        raise ValueError(f"Invalid H3 cached modality tags in {path}")
    token_tags = raw_token_tags.to(torch.long)
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
