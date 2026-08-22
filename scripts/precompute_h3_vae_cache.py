"""Precompute deterministic keyframes and resampleable video VAE posteriors."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from fastwam.datasets.h3_vae_cache import (
    h3_vae_cache_path,
    initialize_h3_vae_cache,
    save_h3_vae_cache,
)
from fastwam.models.minimax_h3.video_vae import MiniMaxH3VAEAdapter
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


def _fingerprint_directory(path: Path) -> str:
    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
        and (
            item.suffix in {".json", ".safetensors"}
            or item.name.endswith(".py")
        )
    )
    digest = hashlib.sha256()
    for item in files:
        digest.update(str(item.relative_to(path)).encode())
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging(log_level=logging.INFO)
    rank, world_size, local_rank = _distributed_context()
    dataset_config = OmegaConf.to_container(cfg.data.train, resolve=True)
    if not isinstance(dataset_config, dict):
        raise ValueError("data.train must resolve to a dataset config")
    cache_dir = cfg.get("vae_cache_dir") or dataset_config.get("h3_vae_cache_dir")
    if cache_dir is None:
        raise ValueError(
            "Set +vae_cache_dir=... or data.train.h3_vae_cache_dir"
        )
    processor_config = {
        key: value
        for key, value in dataset_config.items()
        if key
        not in {
            "h3_condition_cache_dir",
            "h3_vae_cache_dir",
            "text_embedding_cache_dir",
        }
    }
    processor_signature = "sha256:" + hashlib.sha256(
        json.dumps(
            processor_config, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
    ).hexdigest()
    dataset_config["h3_condition_cache_dir"] = None
    dataset_config["text_embedding_cache_dir"] = None
    dataset_config["h3_vae_cache_dir"] = None
    dataset = instantiate(dataset_config)

    component_dir = Path(str(cfg.model.model_path)) / "video_vae"
    fingerprint = _fingerprint_directory(component_dir)
    overwrite = bool(cfg.get("overwrite", False))
    if rank == 0:
        initialize_h3_vae_cache(
            cache_dir,
            vae_fingerprint=fingerprint,
            processor_signature=processor_signature,
            overwrite=overwrite,
        )
    if dist.is_initialized():
        dist.barrier()
    if not torch.cuda.is_available():
        raise RuntimeError("H3 VAE cache precomputation requires CUDA")
    device = torch.device(f"cuda:{local_rank}")
    vae = MiniMaxH3VAEAdapter(
        component_dir, device=device, dtype=torch.float32
    )

    sample_count = len(dataset)
    max_samples = cfg.get("max_samples")
    if max_samples is not None:
        sample_count = min(sample_count, int(max_samples))
    indices = list(range(rank, sample_count, world_size))
    progress = tqdm(
        total=len(indices), disable=rank != 0, desc="H3 FP32 VAE cache"
    )
    written = 0
    skipped = 0
    for index in indices:
        sample = dataset[index]
        video = sample["video"]
        path = h3_vae_cache_path(cache_dir, video)
        if path.is_file() and not overwrite:
            skipped += 1
            progress.update()
            continue
        batched = video.unsqueeze(0).to(device=device, dtype=torch.float32)
        keyframe = vae.encode_keyframe_condition(
            batched[:, :, 0], seed=42
        )[0]
        mean, logvar = vae.encode_video_posterior(batched)
        save_h3_vae_cache(
            cache_dir,
            video=video,
            clean_keyframe_latents=keyframe,
            video_posterior_mean=mean[0],
            video_posterior_logvar=logvar[0],
        )
        written += 1
        progress.update()
        if rank == 0:
            progress.set_postfix(written=written, skipped=skipped)
    progress.close()
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        logging.getLogger(__name__).info(
            "H3 VAE cache complete: written=%d skipped=%d directory=%s",
            written,
            skipped,
            cache_dir,
        )


if __name__ == "__main__":
    main()
