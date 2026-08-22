"""Run one real cache-only H3 optimizer step on one dataset sample."""

from __future__ import annotations

import logging

import hydra
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


class _OneSampleDataset:
    def __init__(self, dataset, index: int) -> None:
        self.dataset = dataset
        self.index = int(index)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError(index)
        return self.dataset[self.index]

    def collate_fn(self, samples):
        return self.dataset.collate_fn(samples)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging(log_level=logging.INFO)
    sample_index = int(cfg.get("sample_index", 0))
    with open_dict(cfg):
        cfg.batch_size = 1
        cfg.num_workers = 0
        cfg.gradient_accumulation_steps = 1
        cfg.max_steps = 1
        cfg.stop_after_step = None
        cfg.save_final_checkpoint = False
        cfg.save_every = 0
        cfg.eval_every = 0
        cfg.log_every = 1
        cfg.resume = None
        cfg.wandb.enabled = False
        cfg.swanlab.enabled = False
        cfg.output_dir = str(
            cfg.get(
                "one_sample_output_dir",
                "./runs/h3-one-sample-cache-training-smoke",
            )
        )

    if bool(cfg.model.get("load_text_encoder", True)):
        raise ValueError("One-sample cache smoke requires model.load_text_encoder=false")
    if bool(cfg.model.get("load_vae", True)):
        raise ValueError("One-sample cache smoke requires model.load_vae=false")
    if not cfg.data.train.get("h3_condition_cache_dir"):
        raise ValueError("One-sample cache smoke requires h3_condition_cache_dir")
    if not cfg.data.train.get("h3_vae_cache_dir"):
        raise ValueError("One-sample cache smoke requires h3_vae_cache_dir")

    precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(precision)
    device = _resolve_train_device()
    dataset = instantiate(cfg.data.train)
    one_sample = _OneSampleDataset(dataset, sample_index)
    cached_sample = one_sample[0]
    required = {
        "prompt_embeds",
        "clean_keyframe_latents",
        "video_posterior_mean",
        "video_posterior_logvar",
    }
    missing = sorted(required - set(cached_sample))
    if missing:
        raise ValueError(f"One-sample cache smoke is missing fields: {missing}")

    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    trainer = Wan22Trainer(
        model=model,
        train_dataset=one_sample,
        val_dataset=None,
        cfg=cfg,
    )
    if trainer.accelerator.is_main_process:
        print(
            OmegaConf.to_yaml(
                {
                    "one_sample_cache_smoke": {
                        "sample_index": sample_index,
                        "condition_shape": list(
                            cached_sample["prompt_embeds"].shape
                        ),
                        "keyframe_shape": list(
                            cached_sample["clean_keyframe_latents"].shape
                        ),
                        "posterior_shape": list(
                            cached_sample["video_posterior_mean"].shape
                        ),
                    }
                }
            ),
            flush=True,
        )
    trainer.train()
    if trainer.accelerator.is_main_process:
        print("H3_ONE_SAMPLE_CACHE_TRAINING_SMOKE=PASS", flush=True)


if __name__ == "__main__":
    main()
