import logging
import json
import inspect
import os
import re
import shutil
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


def _accumulate_sample_weighted_metrics(
    sums: dict[str, torch.Tensor],
    *,
    loss: torch.Tensor,
    loss_dict: dict,
    batch_size: int,
) -> int:
    """Accumulate local metric sums over one gradient-accumulation window."""

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"`batch_size` must be positive, got {batch_size}")
    values = {"loss": loss.detach()}
    values.update(loss_dict)
    for key, value in values.items():
        metric = torch.as_tensor(
            value,
            device=loss.device,
            dtype=torch.float32,
        ).detach()
        if metric.numel() != 1:
            raise ValueError(
                f"Accumulated metric {key!r} must be scalar, got {tuple(metric.shape)}"
            )
        weighted = metric.reshape(()) * batch_size
        sums[key] = sums.get(key, torch.zeros_like(weighted)) + weighted
    return batch_size


def _reduce_accumulated_metrics(
    accelerator,
    sums: dict[str, torch.Tensor],
    sample_count: int,
) -> dict[str, float]:
    """Reduce an accumulation window into exact global sample-weighted means."""

    if sample_count <= 0 or not sums:
        raise ValueError("Cannot reduce an empty accumulation window")
    keys = sorted(sums)
    local = torch.stack(
        [sums[key] for key in keys]
        + [
            torch.tensor(
                float(sample_count),
                device=next(iter(sums.values())).device,
                dtype=torch.float32,
            )
        ]
    )
    gathered = accelerator.gather(local).reshape(-1, len(keys) + 1)
    global_totals = gathered.sum(dim=0)
    global_count = global_totals[-1].clamp(min=1.0)
    return {
        key: float((global_totals[index] / global_count).item())
        for index, key in enumerate(keys)
    }


def _validate_evaluation_vae_contract(model, val_dataset, eval_every: int) -> None:
    if (
        int(eval_every) > 0
        and val_dataset is not None
        and hasattr(model, "vae")
        and model.vae is None
    ):
        raise ValueError(
            "Periodic evaluation requires a loaded VAE. Set eval_every=0 "
            "for cache-only H3 training with load_vae=false, or load the VAE."
        )


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        stop_after_step = cfg.get("stop_after_step")
        self.stop_after_step = (
            None if stop_after_step is None else int(stop_after_step)
        )
        if self.stop_after_step is not None and self.stop_after_step <= 0:
            raise ValueError(
                f"`stop_after_step` must be positive, got {self.stop_after_step}"
            )
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.save_final_checkpoint = bool(cfg.get("save_final_checkpoint", True))
        self.max_checkpoints = int(cfg.get("max_checkpoints", 0))
        if self.max_checkpoints < 0:
            raise ValueError("max_checkpoints must be non-negative")
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        _validate_evaluation_vae_contract(model, val_dataset, self.eval_every)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        
        self.resume = cfg.resume
        self.initial_step = int(cfg.get("initial_step", 0))
        if self.initial_step < 0:
            raise ValueError(f"`initial_step` must be non-negative, got {self.initial_step}.")
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)
        self.swanlab_enabled = bool(cfg.get("swanlab", {}).get("enabled", False))

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)
        trainable_params = [
            parameter
            for module in self._trainable_modules(self.model)
            for parameter in module.parameters()
        ]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        if (
            self.stop_after_step is not None
            and self.stop_after_step > self.max_steps
        ):
            raise ValueError(
                f"`stop_after_step` ({self.stop_after_step}) cannot exceed "
                f"`max_steps` ({self.max_steps})"
            )
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self.swanlab_run = None
        self._init_swanlab()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _init_swanlab(self):
        if not self.swanlab_enabled or not self.accelerator.is_main_process:
            return
        try:
            import swanlab
            from omegaconf import OmegaConf
        except ImportError as e:
            raise ImportError(
                "SwanLab logging is enabled but `swanlab` is not installed."
            ) from e
        self.swanlab_run = swanlab.init(
            project=str(self.cfg.swanlab.project),
            name=str(self.cfg.swanlab.experiment_name),
            config=OmegaConf.to_container(self.cfg, resolve=True),
            mode=str(self.cfg.swanlab.mode),
            log_dir=self.output_dir,
        )
        logger.info(
            "Initialized SwanLab run: project=%s experiment=%s",
            self.cfg.swanlab.project,
            self.cfg.swanlab.experiment_name,
        )

    def _swanlab_log(self, payload: dict):
        if self.swanlab_run is None:
            return
        import swanlab

        swanlab.log(payload, step=self.global_step)

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
            collate_fn=getattr(dataset, "collate_fn", None),
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            if self.initial_step:
                raise ValueError("`initial_step` requires a weights-only `resume` checkpoint.")
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            if self.initial_step:
                raise ValueError(
                    "`initial_step` must be zero when resuming a full training-state directory."
                )
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2.")
        if self.initial_step:
            self.global_step = self.initial_step
            # The previous 2k run used a scheduler whose entire horizon was
            # 2k, so restoring it would keep LR at eta_min. Rebuild the 100k
            # scheduler and advance it to the matching global step instead.
            for _ in range(self.initial_step):
                self.scheduler.step()
            logger.info(
                "Initialized fresh optimizer/scheduler at global_step=%d lr=%.6g.",
                self.global_step,
                self.optimizer.param_groups[0]["lr"],
            )

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _trainable_modules(model):
        provider = getattr(model, "trainable_modules", None)
        if callable(provider):
            modules = list(provider())
            if not modules:
                raise ValueError("model.trainable_modules() returned no modules.")
            return modules
        modules = [model.dit]
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            modules.append(proprio_encoder)
        return modules

    @staticmethod
    def _apply_dit_only_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        for module in Wan22Trainer._trainable_modules(model):
            module.train()
            module.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        batched = {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }
        for key in (
            "image_is_pad",
            "action_is_pad",
            "action_dim_is_pad",
            "proprio_is_pad",
            "proprio_dim_is_pad",
        ):
            value = sample.get(key)
            if value is not None:
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"`sample[{key!r}]` must be a tensor")
                if value.ndim == 1:
                    value = value.unsqueeze(0)
                if value.ndim != 2 or value.shape[0] != video.shape[0]:
                    raise ValueError(
                        f"`sample[{key!r}]` must be [B,L], got {tuple(value.shape)}"
                    )
                batched[key] = value
        return batched

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        is_h3_latent_eval = bool(
            getattr(model, "inference_accepts_ground_truth_action", True) is False
        )
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if bool(getattr(model, "inference_accepts_ground_truth_action", True)):
            infer_kwargs["action"] = action
        if is_h3_latent_eval:
            infer_kwargs["decode_video"] = False
        if sample.get("proprio_dim_is_pad") is not None:
            infer_kwargs["proprio_dim_is_pad"] = sample["proprio_dim_is_pad"]
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred.get("video")
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()
        pred_video_tensor = None
        psnr_rollout_vs_gt = None
        ssim_rollout_vs_gt = None
        if pred_video is not None:
            pred_video_tensor = pil_frames_to_video_tensor(pred_video)
            assert pred_video_tensor.shape == gt_video_tensor.shape, (
                "Eval infer prediction/GT shape mismatch: "
                f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
            )
            psnr_rollout_vs_gt = video_psnr(
                pred=pred_video_tensor, target=gt_video_tensor
            )
            ssim_rollout_vs_gt = video_ssim(
                pred=pred_video_tensor, target=gt_video_tensor
            )

        action_l1 = None
        action_l2 = None
        action_l1_normalized = None
        action_l2_normalized = None
        if action is not None and pred_action is not None:
            pred_action_normalized = pred_action
            if pred_action_normalized.ndim == 2:
                pred_action_normalized = pred_action_normalized.unsqueeze(0)
            gt_action_normalized = action
            if gt_action_normalized.ndim == 2:
                gt_action_normalized = gt_action_normalized.unsqueeze(0)
            pred_action_normalized = pred_action_normalized.detach().to(
                device="cpu", dtype=torch.float32
            )
            gt_action_normalized = gt_action_normalized.detach().to(
                device="cpu", dtype=torch.float32
            )
            if pred_action_normalized.shape != gt_action_normalized.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch before denormalization: "
                    f"pred={tuple(pred_action_normalized.shape)} vs "
                    f"gt={tuple(gt_action_normalized.shape)}"
                )
            valid_action = torch.ones_like(gt_action_normalized, dtype=torch.bool)
            if sample.get("action_is_pad") is not None:
                valid_action &= ~sample["action_is_pad"].detach().cpu().unsqueeze(-1)
            if sample.get("action_dim_is_pad") is not None:
                valid_action &= ~sample["action_dim_is_pad"].detach().cpu().unsqueeze(1)
            normalized_diff = (
                pred_action_normalized - gt_action_normalized
            )[valid_action]
            if normalized_diff.numel() > 0:
                action_l1_normalized = normalized_diff.abs().mean().item()
                action_l2_normalized = normalized_diff.pow(2).mean().item()

            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. Pixel metrics are valid only when inference actually returned a
        # natively decodable video. Five-frame H3 eval remains latent/action-only.
        psnr_decode_vs_gt = None
        ssim_decode_vs_gt = None
        psnr_rollout_vs_decode = None
        ssim_rollout_vs_decode = None
        video_path = None
        if pred_video_tensor is not None:
            gt_video_batch = video0.unsqueeze(0).to(
                device=model.device, dtype=model.torch_dtype
            )
            vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
            vae_recon_video = model._decode_latents(vae_latents, tiled=False)
            vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

            assert vae_video_tensor.shape == gt_video_tensor.shape, (
                "Eval VAE reconstruction/GT shape mismatch: "
                f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
            )

            psnr_decode_vs_gt = video_psnr(
                pred=vae_video_tensor, target=gt_video_tensor
            )
            ssim_decode_vs_gt = video_ssim(
                pred=vae_video_tensor, target=gt_video_tensor
            )
            psnr_rollout_vs_decode = video_psnr(
                pred=pred_video_tensor, target=vae_video_tensor
            )
            ssim_rollout_vs_decode = video_ssim(
                pred=pred_video_tensor, target=vae_video_tensor
            )

            stitched_video_tensor = torch.cat(
                [pred_video_tensor, vae_video_tensor, gt_video_tensor],
                dim=2,
            ).contiguous()
            stitched_frames = []
            for t in range(stitched_video_tensor.shape[1]):
                frame = (
                    stitched_video_tensor[:, t]
                    .permute(1, 2, 0)
                    .clamp(0.0, 1.0)
                    .numpy()
                    * 255.0
                ).astype(np.uint8)
                stitched_frames.append(Image.fromarray(frame))

            video_path = os.path.join(
                self.eval_dir,
                f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
            )
            save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt) if psnr_rollout_vs_gt is not None else float("nan"),
                float(ssim_rollout_vs_gt) if ssim_rollout_vs_gt is not None else float("nan"),
                float(psnr_rollout_vs_decode) if psnr_rollout_vs_decode is not None else float("nan"),
                float(ssim_rollout_vs_decode) if ssim_rollout_vs_decode is not None else float("nan"),
                float(psnr_decode_vs_gt) if psnr_decode_vs_gt is not None else float("nan"),
                float(ssim_decode_vs_gt) if ssim_decode_vs_gt is not None else float("nan"),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
                float(action_l2_normalized) if action_l2_normalized is not None else -1.0,
                float(action_l1_normalized) if action_l1_normalized is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = torch.nanmean(gathered_metrics[:, :7], dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None
        action_l2_normalized_mean = (
            gathered_metrics[:, 9].mean().item()
            if action_l2_normalized is not None
            else None
        )
        action_l1_normalized_mean = (
            gathered_metrics[:, 10].mean().item()
            if action_l1_normalized is not None
            else None
        )

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "video_path": video_path,
        }
        if pred_video_tensor is not None:
            result.update(
                {
                    "psnr_rg": float(mean_metrics[1].item()),
                    "ssim_rg": float(mean_metrics[2].item()),
                    "psnr_rd": float(mean_metrics[3].item()),
                    "ssim_rd": float(mean_metrics[4].item()),
                    "psnr_dg": float(mean_metrics[5].item()),
                    "ssim_dg": float(mean_metrics[6].item()),
                }
            )
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        if action_l2_normalized_mean is not None:
            result["action_l2_normalized"] = float(action_l2_normalized_mean)
        if action_l1_normalized_mean is not None:
            result["action_l1_normalized"] = float(action_l1_normalized_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def _prune_checkpoints(self):
        if self.max_checkpoints <= 0:
            return
        state_paths = sorted(Path(self.state_dir).glob("step_*"))
        weight_paths = sorted(Path(self.weights_dir).glob("step_*.pt"))
        for path in state_paths[: -self.max_checkpoints]:
            shutil.rmtree(path)
            logger.info("Pruned old training state: %s", path)
        for path in weight_paths[: -self.max_checkpoints]:
            path.unlink()
            logger.info("Pruned old weights checkpoint: %s", path)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self._prune_checkpoints()
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def _reached_canary_stop(self) -> bool:
        return (
            self.stop_after_step is not None
            and self.global_step >= self.stop_after_step
        )

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                sampler_state = Path(state_dir) / "sampler.bin"
                if sampler_state.is_file():
                    # Accelerate restores the prepared dataloader cursor from
                    # sampler.bin. Applying our sample offset as well would
                    # skip the consumed global batch range a second time.
                    self.train_sampler.clear_resume_batch_offset()
                    resume_source = "accelerate sampler state"
                else:
                    self.train_sampler.set_resume_batch_offset(
                        self.batch_in_epoch
                    )
                    resume_source = "trainer sample offset"
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d "
                    "sample_offset=%d source=%s",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                    resume_source,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")
        if self._reached_canary_stop():
            raise ValueError(
                f"Training is already at step {self.global_step}, which meets "
                f"`stop_after_step={self.stop_after_step}`. Increase or clear the "
                "canary boundary before resuming."
            )

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()
        accumulated_metric_sums: dict[str, torch.Tensor] = {}
        accumulated_sample_count = 0
        if self.accelerator.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.accelerator.device)

        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                sample_video = sample.get("video")
                if not isinstance(sample_video, torch.Tensor) or sample_video.ndim < 1:
                    raise TypeError(
                        "Training samples must contain a batched `video` tensor "
                        "for accumulation-window metric weighting"
                    )
                accumulated_sample_count += _accumulate_sample_weighted_metrics(
                    accumulated_metric_sums,
                    loss=loss,
                    loss_dict=loss_dict,
                    batch_size=int(sample_video.shape[0]),
                )
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    global_window_metrics = _reduce_accumulated_metrics(
                        self.accelerator,
                        accumulated_metric_sums,
                        accumulated_sample_count,
                    )
                    global_loss = global_window_metrics.pop("loss")
                    global_loss_metrics = global_window_metrics
                    accumulated_metric_sums = {}
                    accumulated_sample_count = 0
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())
                    if self.accelerator.device.type == "cuda":
                        peak_allocated = torch.tensor(
                            torch.cuda.max_memory_allocated(self.accelerator.device)
                            / (1024**3),
                            device=loss.device,
                            dtype=torch.float32,
                        ).reshape(1)
                        peak_reserved = torch.tensor(
                            torch.cuda.max_memory_reserved(self.accelerator.device)
                            / (1024**3),
                            device=loss.device,
                            dtype=torch.float32,
                        ).reshape(1)
                        global_peak_allocated_gib = float(
                            self.accelerator.gather(peak_allocated).max().item()
                        )
                        global_peak_reserved_gib = float(
                            self.accelerator.gather(peak_reserved).max().item()
                        )
                    else:
                        global_peak_allocated_gib = 0.0
                        global_peak_reserved_gib = 0.0

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s peak=%.2f/%.2f GiB eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            steps_per_sec
                            * self.batch_size
                            * self.accelerator.num_processes
                            * self.gradient_accumulation_steps,
                            global_peak_allocated_gib,
                            global_peak_reserved_gib,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec
                            * self.batch_size
                            * self.accelerator.num_processes
                            * self.gradient_accumulation_steps,
                            "performance/peak_allocated_gib": global_peak_allocated_gib,
                            "performance/peak_reserved_gib": global_peak_reserved_gib,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)
                        self._swanlab_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                            )
                            if "psnr_rd" in metrics:
                                description += " infer_psnr=%.4f infer_ssim=%.4f" % (
                                    metrics["psnr_rd"],
                                    metrics["ssim_rd"],
                                )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            if "action_l2_normalized" in metrics:
                                description += " action_l2_norm=%.4f" % metrics["action_l2_normalized"]
                            if "action_l1_normalized" in metrics:
                                description += " action_l1_norm=%.4f" % metrics["action_l1_normalized"]
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                            }
                            for key in (
                                "psnr_rg",
                                "ssim_rg",
                                "psnr_rd",
                                "ssim_rd",
                                "psnr_dg",
                                "ssim_dg",
                            ):
                                if key in metrics:
                                    eval_payload[f"eval/{key}"] = float(metrics[key])
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            if "action_l2_normalized" in metrics:
                                eval_payload["eval/action_l2_normalized"] = float(
                                    metrics["action_l2_normalized"]
                                )
                            if "action_l1_normalized" in metrics:
                                eval_payload["eval/action_l1_normalized"] = float(
                                    metrics["action_l1_normalized"]
                                )
                            self._wandb_log(eval_payload)
                            self._swanlab_log(eval_payload)

                    saved_this_step = False
                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        saved_this_step = True
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self._reached_canary_stop():
                        if not saved_this_step:
                            ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[canary] stop_after_step reached step=%d "
                                "weights=%s state=%s; max_steps=%d scheduler "
                                "horizon is unchanged",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                                self.max_steps,
                            )
                        return

                    if self.global_step >= self.max_steps:
                        if self.save_final_checkpoint and not saved_this_step:
                            ckpt_info = self.save_checkpoint()
                            if self.accelerator.is_main_process:
                                logger.info(
                                    "[done] max_steps reached step=%d weights=%s state=%s",
                                    self.global_step,
                                    ckpt_info["weights_path"],
                                    ckpt_info["state_path"],
                                )
                        elif self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d without final checkpoint",
                                self.global_step,
                            )
                        return

        if self.save_final_checkpoint:
            ckpt_info = self.save_checkpoint()
            if self.accelerator.is_main_process:
                logger.info(
                    "[done] training finished step=%d weights=%s state=%s",
                    self.global_step,
                    ckpt_info["weights_path"],
                    ckpt_info["state_path"],
                )
        
