"""FastWAM policy using MiniMax-H3 as the frozen visual world backbone."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)

from .action_dit import H3ActionDiT
from .video_dit import MiniMaxH3VideoBackbone, load_h3_video_backbone
from .video_vae import MiniMaxH3VAEAdapter


class FastWAMH3(nn.Module):
    """H3 visual prefill plus a trainable width-reduced action expert.

    H3 is already video-pretrained, so the 33B visual backbone and visual VAE
    remain frozen. The action expert is initialized from H3's video blocks and
    learns through layer-wise attention to H3's current-frame K/V features.
    """

    def __init__(
        self,
        *,
        video_expert: MiniMaxH3VideoBackbone,
        action_expert: H3ActionDiT,
        vae: MiniMaxH3VAEAdapter,
        proprio_dim: int | None,
        context_dim: int,
        device: str | torch.device,
        torch_dtype: torch.dtype,
        action_train_shift: float,
        action_infer_shift: float,
        action_num_train_timesteps: int,
        loss_lambda_action: float = 1.0,
    ) -> None:
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.vae = vae
        self.dit = action_expert
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.context_dim = int(context_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        self.proprio_encoder = (
            None
            if self.proprio_dim is None
            else nn.Linear(self.proprio_dim, self.context_dim).to(
                device=self.device, dtype=self.torch_dtype
            )
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        self.loss_lambda_action = float(loss_lambda_action)

        self.video_expert.requires_grad_(False).eval()
        self.vae.requires_grad_(False).eval()
        self.action_expert.to(device=self.device, dtype=self.torch_dtype)

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_path: str | Path,
        video_dit_config: dict[str, Any],
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | Path | None,
        skip_dit_load_from_pretrain: bool,
        proprio_dim: int | None,
        device: str | torch.device,
        torch_dtype: torch.dtype,
        action_train_shift: float,
        action_infer_shift: float,
        action_num_train_timesteps: int,
        loss_lambda_action: float,
    ) -> "FastWAMH3":
        model_path = Path(model_path)
        video_expert = load_h3_video_backbone(
            model_path / "transformer",
            device=device,
            dtype=torch_dtype,
            video_attention_mask_mode=video_dit_config.get(
                "video_attention_mask_mode", "first_frame_causal"
            ),
        )
        action_expert = H3ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            pretrained_path=action_dit_pretrained_path,
            skip_load=skip_dit_load_from_pretrain,
            device=device,
            dtype=torch_dtype,
        )
        if action_expert.num_layers != video_expert.num_layers:
            raise ValueError("H3 action and video experts must have the same layer count.")
        if action_expert.num_heads != video_expert.num_heads:
            raise ValueError("H3 action and video experts must have the same head count.")
        if action_expert.attn_head_dim != video_expert.attn_head_dim:
            raise ValueError("H3 action and video experts must have the same head dimension.")
        vae = MiniMaxH3VAEAdapter(
            model_path / "video_vae", device=device, dtype=torch_dtype
        )
        return cls(
            video_expert=video_expert,
            action_expert=action_expert,
            vae=vae,
            proprio_dim=proprio_dim,
            context_dim=int(action_dit_config.get("context_dim", 4096)),
            device=device,
            torch_dtype=torch_dtype,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_action=loss_lambda_action,
        )

    def trainable_modules(self) -> list[nn.Module]:
        modules: list[nn.Module] = [self.action_expert]
        if self.proprio_encoder is not None:
            modules.append(self.proprio_encoder)
        return modules

    def train(self, mode: bool = True):
        super().train(mode)
        self.video_expert.eval()
        self.vae.eval()
        return self

    def _append_proprio(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None:
            return context, context_mask
        if proprio is None:
            raise ValueError("proprio is required when proprio_dim is configured.")
        if proprio.ndim != 2 or proprio.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"proprio must be [B,{self.proprio_dim}], got {tuple(proprio.shape)}"
            )
        token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=self.torch_dtype)
        ).unsqueeze(1)
        mask = torch.ones(
            (context.shape[0], 1), dtype=torch.bool, device=context_mask.device
        )
        return torch.cat((context, token), dim=1), torch.cat((context_mask, mask), dim=1)

    @torch.no_grad()
    def _encode_current_frame(self, image: torch.Tensor) -> torch.Tensor:
        return self.vae.encode_image(
            image.to(device=self.device, dtype=self.torch_dtype),
            device=self.device,
        ).to(dtype=self.torch_dtype)

    @torch.no_grad()
    def _video_cache(self, image: torch.Tensor) -> dict[str, Any]:
        latents = self._encode_current_frame(image)
        timestep = torch.zeros(
            (latents.shape[0],), device=self.device, dtype=self.torch_dtype
        )
        return self.video_expert.prefill(latents, timestep)

    def _prepare_training_inputs(self, sample: dict[str, Any]) -> dict[str, Any]:
        video = sample["video"]
        action = sample["action"]
        context = sample.get("context")
        context_mask = sample.get("context_mask")
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"video must be [B,3,T,H,W], got {tuple(video.shape)}")
        if action.ndim != 3 or action.shape[-1] != self.action_expert.action_dim:
            raise ValueError(
                f"action must be [B,T,{self.action_expert.action_dim}], "
                f"got {tuple(action.shape)}"
            )
        if context is None or context_mask is None:
            raise ValueError("H3 FastWAM requires precomputed context/context_mask.")
        if context.ndim != 3 or context.shape[-1] != self.context_dim:
            raise ValueError(
                f"context must be [B,L,{self.context_dim}], got {tuple(context.shape)}"
            )
        context = context.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        context_mask = context_mask.to(
            device=self.device, dtype=torch.bool, non_blocking=True
        )
        proprio = sample.get("proprio")
        if proprio is not None:
            proprio = proprio[:, 0].to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
        context, context_mask = self._append_proprio(
            context, context_mask, proprio
        )
        return {
            "image": video[:, :, 0].to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            ),
            "action": action.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            ),
            "action_is_pad": (
                None
                if sample.get("action_is_pad") is None
                else sample["action_is_pad"].to(
                    device=self.device, dtype=torch.bool, non_blocking=True
                )
            ),
            "context": context,
            "context_mask": context_mask,
        }

    def training_loss(self, sample: dict[str, Any], tiled: bool = False):
        del tiled
        inputs = self._prepare_training_inputs(sample)
        action = inputs["action"]
        batch_size = action.shape[0]
        with torch.no_grad():
            cache = self._video_cache(inputs["image"])

        noise = torch.randn_like(action)
        timestep = self.train_action_scheduler.sample_training_t(
            batch_size, self.device, action.dtype
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise, timestep)
        target = self.train_action_scheduler.training_target(action, noise, timestep)
        prediction = self.action_expert.forward_with_video_cache(
            noisy_action,
            timestep,
            inputs["context"],
            inputs["context_mask"],
            cache["kv_cache"],
            int(cache["meta"]["tokens_per_frame"]),
        )
        token_loss = F.mse_loss(
            prediction.float(), target.float(), reduction="none"
        ).mean(dim=-1)
        action_is_pad = inputs["action_is_pad"]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(token_loss.dtype)
            per_sample = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(1)
        else:
            per_sample = token_loss.mean(dim=1)
        weight = self.train_action_scheduler.training_weight(timestep).to(
            device=per_sample.device, dtype=per_sample.dtype
        )
        loss_action = (per_sample * weight).mean()
        loss = self.loss_lambda_action * loss_action
        return loss, {
            "loss_action": self.loss_lambda_action * float(loss_action.detach()),
            "loss_video": 0.0,
        }

    @torch.no_grad()
    def infer_action(
        self,
        *,
        input_image: torch.Tensor,
        action_horizon: int,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor | None = None,
        num_inference_steps: int = 20,
        sigma_shift: float | None = None,
        seed: int | None = None,
        rand_device: str = "cpu",
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        context = context.unsqueeze(0) if context.ndim == 2 else context
        context_mask = (
            context_mask.unsqueeze(0) if context_mask.ndim == 1 else context_mask
        )
        context = context.to(device=self.device, dtype=self.torch_dtype)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        if proprio is not None and proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        context, context_mask = self._append_proprio(
            context,
            context_mask,
            None if proprio is None else proprio.to(self.device),
        )
        cache = self._video_cache(
            input_image.to(device=self.device, dtype=self.torch_dtype)
        )
        generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        action = torch.randn(
            (input_image.shape[0], action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        timesteps, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps,
            self.device,
            action.dtype,
            shift_override=sigma_shift,
        )
        for timestep, delta in zip(timesteps, deltas):
            step = timestep.expand(action.shape[0])
            prediction = self.action_expert.forward_with_video_cache(
                action,
                step,
                context,
                context_mask,
                cache["kv_cache"],
                int(cache["meta"]["tokens_per_frame"]),
            )
            action = self.infer_action_scheduler.step(prediction, delta, action)
        return {"action": action[0].float().cpu()}

    @torch.no_grad()
    def infer(
        self,
        *,
        input_image: torch.Tensor,
        num_frames: int,
        action_horizon: int,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if context is None or context_mask is None:
            raise ValueError(
                "H3 FastWAM currently requires precomputed context; prompt-only "
                "inference is not supported."
            )
        action = self.infer_action(
            input_image=input_image,
            action_horizon=int(action_horizon),
            context=context,
            context_mask=context_mask,
            **kwargs,
        )["action"]
        image = input_image[0] if input_image.ndim == 4 else input_image
        array = (
            ((image.detach().float().cpu().clamp(-1, 1) + 1) * 127.5)
            .byte()
            .permute(1, 2, 0)
            .numpy()
        )
        frame = Image.fromarray(array)
        return {"video": [frame.copy() for _ in range(num_frames)], "action": action}

    @torch.no_grad()
    def _encode_video_latents(self, video: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.vae.encode(video, device=self.device).to(dtype=self.torch_dtype)

    @torch.no_grad()
    def _decode_latents(self, latents: torch.Tensor, **_: Any) -> list[Image.Image]:
        decoded = self.vae.decode(latents, device=self.device)
        decoded = decoded[0].detach().float().cpu().clamp(-1, 1)
        frames = []
        for index in range(decoded.shape[1]):
            array = (
                ((decoded[:, index] + 1) * 127.5)
                .byte()
                .permute(1, 2, 0)
                .numpy()
            )
            frames.append(Image.fromarray(array))
        return frames

    def save_checkpoint(self, path: str | Path, optimizer=None, step=None):
        payload = {
            "action_expert": self.action_expert.state_dict(),
            "step": step,
            "backbone": "MiniMax-H3-FL2VA",
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path, optimizer=None):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.action_expert.load_state_dict(payload["action_expert"], strict=True)
        if self.proprio_encoder is not None and "proprio_encoder" in payload:
            self.proprio_encoder.load_state_dict(payload["proprio_encoder"])
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args: Any, **kwargs: Any):
        return self.training_loss(*args, **kwargs)
