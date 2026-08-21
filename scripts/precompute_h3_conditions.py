"""Precompute exact H3 Qwen layer-50 conditions for f0 plus instruction."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from fastwam.datasets.h3_condition_cache import (
    h3_condition_cache_path,
    initialize_h3_condition_cache,
    save_h3_condition_cache,
)
from fastwam.models.minimax_h3.text_encoder import (
    MiniMaxH3TextConditioner,
    h3_qwen_artifact_fingerprints,
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
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging(log_level=logging.INFO)
    rank, world_size, local_rank = _distributed_context()
    dataset_config = OmegaConf.to_container(cfg.data.train, resolve=True)
    if not isinstance(dataset_config, dict):
        raise ValueError("data.train must resolve to a dataset config")
    cache_dir = dataset_config.get("h3_condition_cache_dir")
    if cache_dir is None:
        raise ValueError("data.train.h3_condition_cache_dir is required")
    dataset_config["h3_condition_cache_dir"] = None
    dataset_config["text_embedding_cache_dir"] = None
    dataset = instantiate(dataset_config)
    overwrite = bool(cfg.get("overwrite", False))
    fingerprints = h3_qwen_artifact_fingerprints(
        Path(str(cfg.model.model_path))
    )
    if rank == 0:
        initialize_h3_condition_cache(
            cache_dir,
            overwrite=overwrite,
            **fingerprints,
        )
    if dist.is_initialized():
        dist.barrier()

    if not torch.cuda.is_available():
        raise RuntimeError("H3 Qwen condition precomputation requires CUDA")
    device = torch.device(f"cuda:{local_rank}")
    conditioner = MiniMaxH3TextConditioner.from_pretrained(
        Path(str(cfg.model.model_path)),
        device=device,
        dtype=torch.bfloat16,
    )
    max_samples = cfg.get("max_samples")
    sample_count = len(dataset)
    if max_samples is not None:
        sample_count = min(sample_count, int(max_samples))
    sampler_seed = cfg.get("sampler_seed")
    if sampler_seed is None:
        indices = list(range(rank, sample_count, world_size))
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(sampler_seed))
        smoke_indices = torch.randperm(
            len(dataset), generator=generator
        )[:sample_count].tolist()
        indices = smoke_indices[rank::world_size]
    cache_batch_size = int(cfg.get("cache_batch_size", 1))
    if cache_batch_size <= 0:
        raise ValueError("cache_batch_size must be positive")
    progress = tqdm(
        total=len(indices),
        disable=rank != 0,
        desc="H3 Qwen layer-50 cache",
    )
    written = 0
    skipped = 0
    for start in range(0, len(indices), cache_batch_size):
        pending = []
        seen_paths: set[Path] = set()
        batch_indices = indices[start : start + cache_batch_size]
        for index in batch_indices:
            sample = dataset[index]
            first_frame = sample["video"][:, 0]
            instruction = str(sample["prompt"])
            path = h3_condition_cache_path(cache_dir, first_frame, instruction)
            if (path.is_file() and not overwrite) or path in seen_paths:
                skipped += 1
                continue
            seen_paths.add(path)
            pending.append((first_frame, instruction))
        if pending:
            batch = conditioner.encode(
                [_to_pil(item[0]) for item in pending],
                [item[1] for item in pending],
            )
            offsets = batch.cu_seqlens.detach().cpu().tolist()
            for item_index, (first_frame, instruction) in enumerate(pending):
                row_start = int(offsets[item_index])
                row_end = int(offsets[item_index + 1])
                save_h3_condition_cache(
                    cache_dir,
                    first_frame=first_frame,
                    instruction=instruction,
                    embeddings=batch.embeddings[row_start:row_end],
                    token_tags=batch.token_tags[row_start:row_end],
                )
                written += 1
        progress.update(len(batch_indices))
        if rank == 0:
            progress.set_postfix(written=written, skipped=skipped)
    progress.close()

    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        logging.getLogger(__name__).info(
            "H3 condition cache complete: written=%d skipped=%d directory=%s",
            written,
            skipped,
            cache_dir,
        )


if __name__ == "__main__":
    main()
