"""Distributed B=2 H3 memory smoke with conservatively padded Qwen rows."""

from __future__ import annotations

import logging

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from fastwam.runtime import (
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
    _resolve_train_device,
)
from fastwam.trainer import Wan22Trainer
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import setup_logging


register_default_resolvers()


class _PaddedH3ConditionDataset:
    def __init__(self, dataset, pad_to_length: int) -> None:
        self.dataset = dataset
        self.pad_to_length = int(pad_to_length)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def collate_fn(self, samples):
        batch = self.dataset.collate_fn(samples)
        current_length = int(batch["prompt_embeds"].shape[1])
        if current_length > self.pad_to_length:
            raise ValueError(
                f"Cached condition length {current_length} exceeds smoke target "
                f"{self.pad_to_length}"
            )
        extra = self.pad_to_length - current_length
        if extra == 0:
            return batch
        batch_size, _, width = batch["prompt_embeds"].shape
        batch["prompt_embeds"] = torch.cat(
            (
                batch["prompt_embeds"],
                torch.zeros(
                    batch_size,
                    extra,
                    width,
                    dtype=batch["prompt_embeds"].dtype,
                ),
            ),
            dim=1,
        )
        batch["prompt_token_tags"] = torch.cat(
            (
                batch["prompt_token_tags"],
                torch.ones(batch_size, extra, dtype=torch.long),
            ),
            dim=1,
        )
        batch["prompt_attention_mask"] = torch.cat(
            (
                batch["prompt_attention_mask"],
                torch.ones(batch_size, extra, dtype=torch.bool),
            ),
            dim=1,
        )
        # Treat any original collate padding plus the synthetic suffix as valid
        # zero-valued rows so every sample has one contiguous 192-row prefix.
        batch["prompt_attention_mask"].fill_(True)
        return batch


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging(log_level=logging.INFO)
    smoke_cfg = cfg.get("memory_smoke", {})
    pad_to_length = int(smoke_cfg.get("pad_to_length", 192))
    smoke_steps = int(smoke_cfg.get("max_steps", 1))

    with open_dict(cfg):
        cfg.max_steps = smoke_steps
        cfg.save_final_checkpoint = False
        cfg.save_every = 0
        cfg.eval_every = 0
        cfg.log_every = 1
        cfg.wandb.enabled = False
        cfg.swanlab.enabled = False
        cfg.output_dir = str(
            smoke_cfg.get("output_dir", "./runs/h3-b2-memory-smoke")
        )

    precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(precision)
    device = _resolve_train_device()
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    dataset = _PaddedH3ConditionDataset(
        instantiate(cfg.data.train),
        pad_to_length=pad_to_length,
    )
    trainer = Wan22Trainer(
        model=model,
        train_dataset=dataset,
        val_dataset=None,
        cfg=cfg,
    )
    if trainer.accelerator.is_main_process:
        print(
            OmegaConf.to_yaml(
                {
                    "memory_smoke": {
                        "batch_size": cfg.batch_size,
                        "gradient_accumulation_steps": (
                            cfg.gradient_accumulation_steps
                        ),
                        "pad_to_length": pad_to_length,
                        "max_steps": smoke_steps,
                    }
                }
            )
        )
    trainer.train()


if __name__ == "__main__":
    main()
