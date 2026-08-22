"""Build complete H3 post-Refiner cache, reusing Qwen cache when available."""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from convert_h3_post_refiner_cache import (
    OUTPUT_SUFFIX,
    SCHEMA_VERSION,
    _artifact_stat_signature,
    _refiner_artifact_paths,
    _tensor_digest,
)
from fastwam.datasets.h3_condition_cache import (
    h3_condition_cache_path,
    load_h3_condition_cache_file,
)
from fastwam.models.minimax_h3.text_encoder import (
    H3_QWEN_ENCODER_SIGNATURE,
    H3_QWEN_PRESENTATION_SIGNATURE,
    MiniMaxH3TextConditioner,
    h3_qwen_artifact_fingerprints,
    h3_qwen_weight_shards_stat_signature,
)
from fastwam.models.minimax_h3.video_dit import (
    H3_CONDITION_REFINER_IMPLEMENTATION_SIGNATURE,
    load_h3_condition_refiner,
)
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import setup_logging


register_default_resolvers()


def _distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank() if dist.is_initialized() else 0
    return rank, world_size, local_rank


def _to_pil(first_frame: torch.Tensor) -> Image.Image:
    array = (
        ((first_frame.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array)


def _save_refined(
    output_path: Path,
    *,
    refined: torch.Tensor,
    tags: torch.Tensor,
    refiner_fingerprint: str,
    source_path: Path | None,
) -> None:
    refined = refined.detach().cpu().to(torch.bfloat16).contiguous()
    tags = tags.detach().cpu().to(torch.long).contiguous()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_filename": None if source_path is None else source_path.name,
        "source_mode": "direct-qwen" if source_path is None else "schema3-reuse",
        "refiner_fingerprint": refiner_fingerprint,
        "prompt_embeds": refined,
        "prompt_token_tags": tags,
        "payload_sha256": _tensor_digest(refined, tags),
    }
    temporary = output_path.with_name(
        f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    )
    torch.save(payload, temporary)
    os.replace(temporary, output_path)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging(log_level=logging.INFO)
    rank, world_size, local_rank = _distributed_context()
    dataset_config = OmegaConf.to_container(cfg.data.train, resolve=True)
    if not isinstance(dataset_config, dict):
        raise ValueError("data.train must resolve to a dataset config")
    source_dir = Path(str(cfg.source_qwen_cache_dir))
    output_dir = Path(str(cfg.post_refiner_cache_dir))
    output_manifest_path = output_dir / "h3-post-refiner-cache-manifest.json"
    if not output_manifest_path.is_file():
        raise FileNotFoundError(
            f"Run convert_h3_post_refiner_cache.py once to initialize "
            f"{output_manifest_path}"
        )
    output_manifest = json.loads(output_manifest_path.read_text())
    source_manifest = output_manifest["source_manifest"]
    model_path = Path(str(cfg.model.model_path))
    actual_fingerprints = h3_qwen_artifact_fingerprints(
        model_path
    )
    if (
        actual_fingerprints["qwen_checkpoint_fingerprint"]
        != source_manifest["qwen_checkpoint_fingerprint"]
        or actual_fingerprints["processor_fingerprint"]
        != source_manifest["processor_fingerprint"]
    ):
        raise ValueError("Current Qwen/processor does not match source cache manifest")
    if (
        output_manifest.get("qwen_encoder_signature") != H3_QWEN_ENCODER_SIGNATURE
        or output_manifest.get("qwen_presentation_signature")
        != H3_QWEN_PRESENTATION_SIGNATURE
        or output_manifest.get("refiner_implementation_signature")
        != H3_CONDITION_REFINER_IMPLEMENTATION_SIGNATURE
        or output_manifest.get("qwen_weight_shards_stat_signature")
        != h3_qwen_weight_shards_stat_signature(model_path)
        or output_manifest.get("refiner_artifact_stat_signature")
        != _artifact_stat_signature(
            _refiner_artifact_paths(model_path / "transformer")
        )
    ):
        raise ValueError(
            "Current Qwen/Refiner implementation or artifact metadata does not "
            "match the post-Refiner cache manifest"
        )

    dataset_config["h3_condition_cache_dir"] = None
    dataset_config["text_embedding_cache_dir"] = None
    dataset_config["h3_vae_cache_dir"] = None
    dataset = instantiate(dataset_config)
    if not torch.cuda.is_available():
        raise RuntimeError("Post-Refiner precompute requires CUDA")
    device = torch.device(f"cuda:{local_rank}")
    conditioner = MiniMaxH3TextConditioner.from_pretrained(
        model_path, device=device, dtype=torch.bfloat16
    )
    refiner = load_h3_condition_refiner(
        model_path / "transformer",
        device=device,
        dtype=torch.bfloat16,
    )
    cache_batch_size = int(cfg.get("cache_batch_size", 8))
    if cache_batch_size <= 0:
        raise ValueError("cache_batch_size must be positive")
    start_index = int(cfg.get("start_index", 0))
    if start_index < 0 or start_index > len(dataset):
        raise ValueError(
            f"start_index must be between 0 and {len(dataset)}, got {start_index}"
        )
    first_rank_index = start_index + ((rank - start_index) % world_size)
    indices = list(range(first_rank_index, len(dataset), world_size))
    max_samples = cfg.get("max_samples")
    if max_samples is not None:
        indices = indices[: int(max_samples)]
    progress = tqdm(
        total=len(indices),
        disable=rank != 0,
        desc="Complete H3 post-Refiner cache",
    )
    seen: set[Path] = set()
    reused = 0
    generated = 0
    skipped = 0
    for start in range(0, len(indices), cache_batch_size):
        pending = []
        for index in indices[start : start + cache_batch_size]:
            sample = dataset.get_strict(index)
            first_frame = sample["video"][:, 0]
            instruction = str(sample["prompt"])
            source_path = h3_condition_cache_path(
                source_dir, first_frame, instruction
            )
            digest_name = source_path.name.split(".", 1)[0]
            output_path = output_dir / f"{digest_name}{OUTPUT_SUFFIX}"
            if output_path in seen:
                skipped += 1
                continue
            if output_path.is_file():
                try:
                    load_h3_condition_cache_file(
                        output_path, manifest=output_manifest
                    )
                except (KeyError, RuntimeError, ValueError) as error:
                    logging.getLogger(__name__).warning(
                        "Regenerating invalid post-Refiner cache %s: %s",
                        output_path,
                        error,
                    )
                else:
                    seen.add(output_path)
                    skipped += 1
                    continue
            seen.add(output_path)
            pending.append(
                {
                    "first_frame": first_frame,
                    "instruction": instruction,
                    "source_path": source_path if source_path.is_file() else None,
                    "output_path": output_path,
                }
            )
        if pending:
            direct_items = [item for item in pending if item["source_path"] is None]
            direct_rows: dict[Path, tuple[torch.Tensor, torch.Tensor]] = {}
            if direct_items:
                encoded = conditioner.encode(
                    [_to_pil(item["first_frame"]) for item in direct_items],
                    [item["instruction"] for item in direct_items],
                )
                offsets = encoded.cu_seqlens.detach().cpu().tolist()
                for item_index, item in enumerate(direct_items):
                    row_start, row_end = offsets[item_index : item_index + 2]
                    direct_rows[item["output_path"]] = (
                        encoded.embeddings[row_start:row_end],
                        encoded.token_tags[row_start:row_end],
                    )
            rows = []
            tags = []
            lengths = []
            for item in pending:
                if item["source_path"] is None:
                    embeddings, token_tags = direct_rows[item["output_path"]]
                    generated += 1
                else:
                    cached = load_h3_condition_cache_file(
                        item["source_path"], manifest=source_manifest
                    )
                    embeddings = cached["prompt_embeds"].to(device)
                    token_tags = cached["prompt_token_tags"].to(device)
                    reused += 1
                rows.append(embeddings)
                tags.append(token_tags)
                lengths.append(embeddings.shape[0])
            offsets = [0]
            for length in lengths:
                offsets.append(offsets[-1] + int(length))
            refined = refiner(
                torch.cat(rows).to(device),
                torch.tensor(offsets, device=device, dtype=torch.int32),
            ).to(torch.bfloat16)
            for item_index, item in enumerate(pending):
                _save_refined(
                    item["output_path"],
                    refined=refined[offsets[item_index] : offsets[item_index + 1]],
                    tags=tags[item_index],
                    refiner_fingerprint=output_manifest["refiner_fingerprint"],
                    source_path=item["source_path"],
                )
        progress.update(min(cache_batch_size, len(indices) - start))
        if rank == 0:
            progress.set_postfix(
                reused=reused, generated=generated, skipped=skipped
            )
    progress.close()
    print(
        f"POST_REFINER_COMPLETE_PASS rank={rank} reused={reused} "
        f"generated={generated} skipped={skipped} scanned={len(indices)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
