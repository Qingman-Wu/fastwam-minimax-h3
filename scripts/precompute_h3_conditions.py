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
    save_h3_condition_cache,
)
from fastwam.models.minimax_h3.text_encoder import MiniMaxH3TextConditioner
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

    if not torch.cuda.is_available():
        raise RuntimeError("H3 Qwen condition precomputation requires CUDA")
    device = torch.device(f"cuda:{local_rank}")
    conditioner = MiniMaxH3TextConditioner.from_pretrained(
        Path(str(cfg.model.model_path)),
        device=device,
        dtype=torch.bfloat16,
    )
    overwrite = bool(cfg.get("overwrite", False))
    indices = range(rank, len(dataset), world_size)
    progress = tqdm(
        indices,
        total=(len(dataset) + world_size - 1 - rank) // world_size,
        disable=rank != 0,
        desc="H3 Qwen layer-50 cache",
    )
    written = 0
    skipped = 0
    for index in progress:
        sample = dataset[index]
        first_frame = sample["video"][:, 0]
        instruction = str(sample["prompt"])
        path = h3_condition_cache_path(cache_dir, first_frame, instruction)
        if path.is_file() and not overwrite:
            skipped += 1
            continue
        batch = conditioner.encode([_to_pil(first_frame)], [instruction])
        save_h3_condition_cache(
            cache_dir,
            first_frame=first_frame,
            instruction=instruction,
            embeddings=batch.embeddings,
            token_tags=batch.token_tags,
        )
        written += 1
        if rank == 0:
            progress.set_postfix(written=written, skipped=skipped)

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
