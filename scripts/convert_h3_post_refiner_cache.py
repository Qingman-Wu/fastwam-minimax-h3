"""Convert schema-3 Qwen rows into frozen H3 post-Refiner conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path

import torch

from fastwam.datasets.h3_condition_cache import load_h3_condition_cache_file
from fastwam.models.minimax_h3.text_encoder import (
    H3_QWEN_ENCODER_SIGNATURE,
    H3_QWEN_PRESENTATION_SIGNATURE,
    h3_qwen_artifact_fingerprints,
)
from fastwam.models.minimax_h3.video_dit import (
    H3_CONDITION_REFINER_IMPLEMENTATION_SIGNATURE,
    load_h3_condition_refiner,
)


SCHEMA_VERSION = 4
OUTPUT_SUFFIX = ".h3-post-refiner-v4.pt"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _tensor_digest(embeddings: torch.Tensor, tags: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in (embeddings, tags):
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(memoryview(value.view(torch.uint8).numpy()))
    return f"sha256:{digest.hexdigest()}"


def _refiner_artifact_paths(transformer_dir: Path) -> tuple[Path, ...]:
    index_path = transformer_dir / "model.safetensors.index.json"
    config_path = transformer_dir / "config.json"
    index = json.loads(index_path.read_text())["weight_map"]
    prefixes = ("condition_proj.", "token_refiner.")
    shard_names = sorted(
        {shard for key, shard in index.items() if key.startswith(prefixes)}
    )
    return (config_path, index_path, *(transformer_dir / name for name in shard_names))


def _artifact_stat_signature(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(f"{path.name}\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode())
    return f"sha256:{digest.hexdigest()}"


def _refiner_fingerprint(transformer_dir: Path) -> str:
    paths = _refiner_artifact_paths(transformer_dir)
    digest = hashlib.sha256()
    for path in paths[:2]:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    for path in paths[2:]:
        digest.update(path.name.encode())
        digest.update(_sha256_file(path).encode())
    return f"sha256:{digest.hexdigest()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transformer-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    source_manifest_path = args.source_dir / "h3-qwen-cache-manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_files = sorted(
        args.source_dir.glob("*.h3-qwen-prenorm-layer50-v3.pt")
    )
    if args.max_files is not None:
        source_files = source_files[: args.max_files]
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(f"Invalid rank/world_size: {rank}/{world_size}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    source_files = source_files[rank::world_size]
    if not source_files:
        raise FileNotFoundError(
            f"No schema-3 cache files assigned to rank {rank} in {args.source_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "h3-post-refiner-cache-manifest.json"
    existing_manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    )
    refiner_paths = _refiner_artifact_paths(args.transformer_dir)
    if existing_manifest is not None:
        refiner_fingerprint = existing_manifest["refiner_fingerprint"]
        qwen_weight_shards_fingerprint = existing_manifest[
            "qwen_weight_shards_fingerprint"
        ]
        qwen_weight_shards_stat_signature = existing_manifest[
            "qwen_weight_shards_stat_signature"
        ]
    elif rank == 0:
        refiner_fingerprint = _refiner_fingerprint(args.transformer_dir)
        qwen_fingerprints = h3_qwen_artifact_fingerprints(
            args.transformer_dir.parent, include_weight_shards=True
        )
        qwen_weight_shards_fingerprint = qwen_fingerprints[
            "qwen_weight_shards_fingerprint"
        ]
        qwen_weight_shards_stat_signature = qwen_fingerprints[
            "qwen_weight_shards_stat_signature"
        ]
    else:
        raise RuntimeError(
            "Rank 0 must initialize the post-Refiner manifest before "
            "multi-process conversion starts"
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest": source_manifest,
        "source_manifest_sha256": _sha256_file(source_manifest_path),
        "refiner_fingerprint": refiner_fingerprint,
        "refiner_artifact_stat_signature": _artifact_stat_signature(refiner_paths),
        "refiner_implementation_signature": (
            H3_CONDITION_REFINER_IMPLEMENTATION_SIGNATURE
        ),
        "qwen_weight_shards_fingerprint": qwen_weight_shards_fingerprint,
        "qwen_weight_shards_stat_signature": (
            qwen_weight_shards_stat_signature
        ),
        "qwen_encoder_signature": H3_QWEN_ENCODER_SIGNATURE,
        "qwen_presentation_signature": H3_QWEN_PRESENTATION_SIGNATURE,
        "embedding_width": 5376,
        "embedding_dtype": "torch.bfloat16",
        "implementation_signature": "fastwam-h3-post-refiner-v3",
    }
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        if not args.overwrite:
            raise ValueError("Existing post-Refiner manifest is incompatible")
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.tmp.{uuid.uuid4().hex}"
    )
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_manifest, manifest_path)

    refiner = load_h3_condition_refiner(
        args.transformer_dir,
        device=(
            f"cuda:{local_rank}"
            if args.device.startswith("cuda") and world_size > 1
            else args.device
        ),
        dtype=torch.bfloat16,
    )
    device = next(refiner.parameters()).device
    written = 0
    skipped = 0
    for start in range(0, len(source_files), args.batch_size):
        pending = []
        for source_path in source_files[start : start + args.batch_size]:
            digest_name = source_path.name.split(".", 1)[0]
            output_path = args.output_dir / f"{digest_name}{OUTPUT_SUFFIX}"
            if output_path.exists() and not args.overwrite:
                try:
                    load_h3_condition_cache_file(output_path, manifest=manifest)
                except (KeyError, RuntimeError, ValueError) as error:
                    print(
                        f"rank={rank} regenerating invalid cache "
                        f"path={output_path} error={error}",
                        flush=True,
                    )
                else:
                    skipped += 1
                    continue
            source = torch.load(source_path, map_location="cpu", weights_only=True)
            embeddings = source["prompt_embeds"]
            tags = source["prompt_token_tags"].to(torch.long)
            if embeddings.ndim != 2 or embeddings.shape[-1] != 5120:
                raise ValueError(f"Invalid source embedding shape in {source_path}")
            pending.append((source_path, output_path, embeddings, tags))
        if not pending:
            continue
        lengths = [item[2].shape[0] for item in pending]
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + int(length))
        packed = torch.cat([item[2] for item in pending]).to(device)
        cu = torch.tensor(offsets, device=device, dtype=torch.int32)
        refined_packed = refiner(packed, cu).to(torch.bfloat16).cpu()
        for item_index, (source_path, output_path, _, tags) in enumerate(pending):
            # A contiguous slice can still share the full packed batch storage.
            # Clone so torch.save writes only this sample, not the ~128x batch.
            refined = refined_packed[
                offsets[item_index] : offsets[item_index + 1]
            ].clone()
            payload = {
                "schema_version": SCHEMA_VERSION,
                "source_filename": source_path.name,
                "source_tensor_sha256": _tensor_digest(
                    pending[item_index][2], tags
                ),
                "refiner_fingerprint": refiner_fingerprint,
                "prompt_embeds": refined,
                "prompt_token_tags": tags.cpu(),
                "payload_sha256": _tensor_digest(refined, tags),
            }
            temporary = output_path.with_name(
                f".{output_path.name}.tmp.{uuid.uuid4().hex}"
            )
            torch.save(payload, temporary)
            os.replace(temporary, output_path)
            written += 1
        if written and written % max(args.batch_size * 20, 100) == 0:
            print(
                f"rank={rank} progress written={written} skipped={skipped} "
                f"assigned={len(source_files)}",
                flush=True,
            )
    print(
        f"POST_REFINER_CACHE_COMPLETE rank={rank} written={written} "
        f"skipped={skipped} assigned={len(source_files)} output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
