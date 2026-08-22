"""Upgrade a schema-4 manifest without rewriting valid embedding payloads."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from convert_h3_post_refiner_cache import (
    _artifact_stat_signature,
    _refiner_artifact_paths,
    _refiner_fingerprint,
    _sha256_file,
)
from fastwam.models.minimax_h3.text_encoder import (
    H3_QWEN_ENCODER_SIGNATURE,
    H3_QWEN_PRESENTATION_SIGNATURE,
    h3_qwen_artifact_fingerprints,
)
from fastwam.models.minimax_h3.video_dit import (
    H3_CONDITION_REFINER_IMPLEMENTATION_SIGNATURE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_manifest_path = args.source_dir / "h3-qwen-cache-manifest.json"
    output_manifest_path = args.output_dir / "h3-post-refiner-cache-manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    current = json.loads(output_manifest_path.read_text())
    if current.get("schema_version") != 4:
        raise ValueError("Only schema-4 post-Refiner manifests can be upgraded")
    if current.get("source_manifest") != source_manifest:
        raise ValueError("Post-Refiner source manifest does not match source cache")

    transformer_dir = args.model_dir / "transformer"
    actual_refiner_fingerprint = _refiner_fingerprint(transformer_dir)
    if current.get("refiner_fingerprint") != actual_refiner_fingerprint:
        raise ValueError(
            "Existing cache Refiner fingerprint does not match current weights"
        )
    qwen = h3_qwen_artifact_fingerprints(
        args.model_dir, include_weight_shards=True
    )
    if (
        qwen["qwen_checkpoint_fingerprint"]
        != source_manifest["qwen_checkpoint_fingerprint"]
        or qwen["processor_fingerprint"] != source_manifest["processor_fingerprint"]
    ):
        raise ValueError("Current Qwen/processor does not match source manifest")

    upgraded = {
        "schema_version": 4,
        "source_manifest": source_manifest,
        "source_manifest_sha256": _sha256_file(source_manifest_path),
        "refiner_fingerprint": actual_refiner_fingerprint,
        "refiner_artifact_stat_signature": _artifact_stat_signature(
            _refiner_artifact_paths(transformer_dir)
        ),
        "refiner_implementation_signature": (
            H3_CONDITION_REFINER_IMPLEMENTATION_SIGNATURE
        ),
        "qwen_weight_shards_fingerprint": qwen[
            "qwen_weight_shards_fingerprint"
        ],
        "qwen_weight_shards_stat_signature": qwen[
            "qwen_weight_shards_stat_signature"
        ],
        "qwen_encoder_signature": H3_QWEN_ENCODER_SIGNATURE,
        "qwen_presentation_signature": H3_QWEN_PRESENTATION_SIGNATURE,
        "embedding_width": 5376,
        "embedding_dtype": "torch.bfloat16",
        "implementation_signature": "fastwam-h3-post-refiner-v3",
    }
    temporary = output_manifest_path.with_name(
        f".{output_manifest_path.name}.tmp.{uuid.uuid4().hex}"
    )
    temporary.write_text(json.dumps(upgraded, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_manifest_path)
    print(
        f"H3_POST_REFINER_MANIFEST_UPGRADED path={output_manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
