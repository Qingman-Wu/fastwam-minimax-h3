"""FastWAM-H3 Scheme A training and joint inference orchestration."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)

from .action_dit import H3ActionDiT
from .text_encoder import H3TextConditionBatch, MiniMaxH3TextConditioner
from .video_dit import MiniMaxH3VideoBackbone, load_h3_video_backbone
from .video_vae import MiniMaxH3VAEAdapter, augment_keyframe_latents


def _h3_checkpoint_fingerprint(model_path: str | Path) -> str:
    """Fingerprint the immutable H3 config/index manifest without hashing shards."""

    transformer_path = Path(model_path) / "transformer"
    manifests = sorted(transformer_path.glob("*.json"))
    if not manifests:
        return "unavailable"
    digest = hashlib.sha256()
    for manifest in manifests:
        digest.update(manifest.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(manifest.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


class FastWAMH3(nn.Module):
    """H3 full-video target plus a state-prefixed independent Action Expert."""

    inference_accepts_ground_truth_action = False

    def __init__(
        self,
        *,
        video_expert: MiniMaxH3VideoBackbone,
        action_expert: H3ActionDiT,
        vae: MiniMaxH3VAEAdapter,
        text_conditioner: MiniMaxH3TextConditioner | None,
        device: str | torch.device,
        torch_dtype: torch.dtype,
        video_train_shift: float,
        video_infer_shift: float,
        video_num_train_timesteps: int,
        action_train_shift: float,
        action_infer_shift: float,
        action_num_train_timesteps: int,
        keyframe_condition_strength: float = 0.999,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        video_fps: float = 24.0,
        action_fps: float | None = None,
        freeze_video_expert: bool = True,
        h3_lora_rank: int = 0,
        h3_lora_alpha: float = 32.0,
        h3_lora_dropout: float = 0.0,
        stop_action_gradient_to_h3: bool = False,
        base_h3_fingerprint: str = "unavailable",
    ) -> None:
        super().__init__()
        if action_expert.num_layers != video_expert.num_layers:
            raise ValueError("H3 action and video experts must have the same layer count")
        if action_expert.num_heads != video_expert.num_heads:
            raise ValueError("H3 action and video experts must have the same head count")
        if action_expert.attn_head_dim != video_expert.attn_head_dim:
            raise ValueError("H3 action and video experts must have the same head dimension")
        if not 0.0 <= float(keyframe_condition_strength) <= 1.0:
            raise ValueError("keyframe_condition_strength must be in [0, 1]")
        if float(video_fps) <= 0:
            raise ValueError("video_fps must be positive")
        if action_fps is not None and float(action_fps) <= 0:
            raise ValueError("action_fps must be positive when configured")

        self.video_expert = video_expert
        self.action_expert = action_expert
        self.vae = vae
        self.text_conditioner = text_conditioner
        self.dit = action_expert
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.keyframe_condition_strength = float(keyframe_condition_strength)
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.video_fps = float(video_fps)
        self.action_fps = None if action_fps is None else float(action_fps)
        self.freeze_video_expert = bool(freeze_video_expert)
        self.h3_lora_rank = int(h3_lora_rank)
        self.h3_lora_alpha = float(h3_lora_alpha)
        self.h3_lora_dropout = float(h3_lora_dropout)
        self.stop_action_gradient_to_h3 = bool(stop_action_gradient_to_h3)
        self.base_h3_fingerprint = str(base_h3_fingerprint)

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )

        self.vae.requires_grad_(False).eval()
        if self.text_conditioner is not None:
            self.text_conditioner.requires_grad_(False).eval()
        self.video_expert.requires_grad_(not self.freeze_video_expert)
        if self.freeze_video_expert:
            for branch in self._h3_lora_branches():
                branch.requires_grad_(True)
        self.video_expert.eval() if self.freeze_video_expert else self.video_expert.train()
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
        load_text_encoder: bool,
        text_encoder_device: str | torch.device,
        device: str | torch.device,
        torch_dtype: torch.dtype,
        video_train_shift: float,
        video_infer_shift: float,
        video_num_train_timesteps: int,
        action_train_shift: float,
        action_infer_shift: float,
        action_num_train_timesteps: int,
        keyframe_condition_strength: float,
        loss_lambda_video: float,
        loss_lambda_action: float,
        video_fps: float,
        action_fps: float | None,
        freeze_video_expert: bool,
        h3_lora_rank: int = 0,
        h3_lora_alpha: float = 32.0,
        h3_lora_dropout: float = 0.0,
        stop_action_gradient_to_h3: bool = False,
    ) -> "FastWAMH3":
        model_path = Path(model_path)
        video_expert = load_h3_video_backbone(
            model_path / "transformer",
            device=device,
            dtype=torch_dtype,
            video_attention_mask_mode=video_dit_config.get(
                "video_attention_mask_mode", "bidirectional"
            ),
        )
        video_expert.inject_attention_lora(
            rank=h3_lora_rank,
            alpha=h3_lora_alpha,
            dropout=h3_lora_dropout,
        )
        action_expert = H3ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            pretrained_path=action_dit_pretrained_path,
            skip_load=skip_dit_load_from_pretrain,
            device=device,
            dtype=torch_dtype,
        )
        vae = MiniMaxH3VAEAdapter(
            model_path / "video_vae", device=device, dtype=torch_dtype
        )
        text_conditioner = (
            MiniMaxH3TextConditioner.from_pretrained(
                model_path, device=text_encoder_device, dtype=torch_dtype
            )
            if load_text_encoder
            else None
        )
        return cls(
            video_expert=video_expert,
            action_expert=action_expert,
            vae=vae,
            text_conditioner=text_conditioner,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            keyframe_condition_strength=keyframe_condition_strength,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            video_fps=video_fps,
            action_fps=action_fps,
            freeze_video_expert=freeze_video_expert,
            h3_lora_rank=h3_lora_rank,
            h3_lora_alpha=h3_lora_alpha,
            h3_lora_dropout=h3_lora_dropout,
            stop_action_gradient_to_h3=stop_action_gradient_to_h3,
            base_h3_fingerprint=_h3_checkpoint_fingerprint(model_path),
        )

    def _named_h3_lora_branches(self) -> dict[str, nn.Module]:
        getter = getattr(self.video_expert, "named_lora_branches", None)
        if getter is None:
            return {}
        return dict(getter())

    def _h3_lora_branches(self) -> list[nn.Module]:
        named = self._named_h3_lora_branches()
        if named:
            return list(named.values())
        getter = getattr(self.video_expert, "lora_branches", None)
        return [] if getter is None else list(getter())

    def _effective_action_rope_fps(
        self, *, num_frames: int, action_horizon: int
    ) -> float:
        if int(num_frames) <= 1 or int(action_horizon) <= 0:
            raise ValueError("Action RoPE timing requires frames>1 and actions>0")
        effective = (
            self.video_fps * float(action_horizon) / float(num_frames - 1)
        )
        if self.action_fps is not None and not math.isclose(
            self.action_fps, effective, rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError(
                "Configured action_fps does not match the Scheme-A equivalent "
                "RoPE clock: "
                f"{self.action_fps} != {effective} "
                f"for {action_horizon} actions across {num_frames} frames"
            )
        return effective

    def trainable_modules(self) -> list[nn.Module]:
        modules: list[nn.Module] = [self.action_expert]
        if not self.freeze_video_expert:
            modules.append(self.video_expert)
        else:
            modules.extend(self._h3_lora_branches())
        return modules

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        if self.text_conditioner is not None:
            self.text_conditioner.eval()
        if self.freeze_video_expert:
            self.video_expert.eval()
            for branch in self._h3_lora_branches():
                branch.train(mode)
        return self

    @staticmethod
    def _normalize_batch_mask(
        mask: torch.Tensor | None,
        *,
        batch_size: int,
        length: int,
        name: str,
        device: torch.device,
    ) -> torch.Tensor:
        if mask is None:
            return torch.zeros((batch_size, length), dtype=torch.bool, device=device)
        mask = mask.to(device=device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).expand(batch_size, -1)
        if mask.shape != (batch_size, length):
            raise ValueError(
                f"{name} must be [B,{length}] or [{length}], got {tuple(mask.shape)}"
            )
        return mask

    @staticmethod
    def _tensor_images_to_pil(images: torch.Tensor) -> list[Image.Image]:
        output: list[Image.Image] = []
        for image in images:
            array = (
                ((image.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5)
                .byte()
                .permute(1, 2, 0)
                .numpy()
            )
            output.append(Image.fromarray(array))
        return output

    @staticmethod
    def _dense_text_batch(
        batch: H3TextConditionBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths = batch.lengths
        batch_size = len(lengths)
        max_length = max(lengths)
        embeddings = batch.embeddings.new_zeros(
            (batch_size, max_length, batch.embeddings.shape[-1])
        )
        tags = torch.zeros(
            (batch_size, max_length),
            dtype=torch.long,
            device=batch.embeddings.device,
        )
        valid = torch.zeros(
            (batch_size, max_length),
            dtype=torch.bool,
            device=batch.embeddings.device,
        )
        for index, (start, end) in enumerate(
            zip(batch.cu_seqlens[:-1].tolist(), batch.cu_seqlens[1:].tolist())
        ):
            length = int(end - start)
            embeddings[index, :length] = batch.embeddings[start:end]
            tags[index, :length] = batch.token_tags[start:end]
            valid[index, :length] = True
        return embeddings, tags, valid

    def _prepare_text_condition(
        self,
        sample: dict[str, Any],
        first_frame: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = sample.get("prompt_embeds")
        if embeddings is None and sample.get("context") is not None:
            embeddings = sample["context"]
        if embeddings is not None:
            if embeddings.ndim == 2:
                embeddings = embeddings.unsqueeze(0)
            if embeddings.ndim != 3 or embeddings.shape[-1] != 5120:
                raise ValueError(
                    "Precomputed H3 Qwen embeddings must be [B,L,5120]; "
                    "legacy Wan/T5 context width is not accepted"
                )
            tags = sample.get("prompt_token_tags")
            valid = sample.get("prompt_attention_mask")
            if tags is None or valid is None:
                raise ValueError(
                    "prompt_token_tags and prompt_attention_mask are required with "
                    "precomputed H3 Qwen embeddings"
                )
            if tags.ndim == 1:
                tags = tags.unsqueeze(0)
            if valid.ndim == 1:
                valid = valid.unsqueeze(0)
            embeddings = embeddings.to(device=self.device, dtype=self.torch_dtype)
            tags = tags.to(device=self.device, dtype=torch.long)
            valid = valid.to(device=self.device, dtype=torch.bool)
            if tags.shape != embeddings.shape[:2] or valid.shape != embeddings.shape[:2]:
                raise ValueError("Qwen tags/mask must match prompt_embeds [B,L]")
            valid_tags = tags[valid]
            if not torch.logical_or(valid_tags == 0, valid_tags == 1).all():
                raise ValueError("valid Qwen tags must contain only video 0 or text 1")
            return embeddings, tags, valid

        if self.text_conditioner is None:
            raise ValueError(
                "Either native prompt_embeds/tags/mask or a loaded H3 Qwen3-VL "
                "text conditioner is required"
            )
        instructions = sample.get("instruction", sample.get("prompt"))
        if isinstance(instructions, str):
            instructions = [instructions]
        if not isinstance(instructions, Sequence) or len(instructions) != first_frame.shape[0]:
            raise ValueError("instruction/prompt must contain one string per sample")
        batch = self.text_conditioner.encode(
            self._tensor_images_to_pil(first_frame), list(instructions)
        )
        embeddings, tags, valid = self._dense_text_batch(batch)
        return (
            embeddings.to(device=self.device, dtype=self.torch_dtype),
            tags.to(device=self.device, dtype=torch.long),
            valid.to(device=self.device, dtype=torch.bool),
        )

    def _prepare_state(self, sample: dict[str, Any], batch_size: int) -> torch.Tensor:
        proprio = sample.get("proprio")
        if proprio is None:
            raise ValueError("proprio state is required for FastWAM-H3")
        if proprio.ndim == 2:
            proprio = proprio.unsqueeze(1)
        if proprio.ndim != 3 or proprio.shape[0] != batch_size:
            raise ValueError("proprio must be [B,T,Ds]")
        if proprio.shape[-1] != self.action_expert.state_dim:
            raise ValueError(
                f"proprio state width must be {self.action_expert.state_dim}, got "
                f"{proprio.shape[-1]}"
            )
        state_is_pad = sample.get("proprio_is_pad")
        if state_is_pad is None:
            state_is_pad = torch.zeros(
                (batch_size, 1), dtype=torch.bool, device=self.device
            )
        else:
            state_is_pad = state_is_pad.to(device=self.device, dtype=torch.bool)
            if state_is_pad.ndim == 1:
                state_is_pad = state_is_pad.unsqueeze(0).expand(batch_size, -1)
            if state_is_pad.ndim != 2 or state_is_pad.shape[0] != batch_size:
                raise ValueError("proprio_is_pad must be [B,T] or [T]")
            if state_is_pad.shape[1] < 1:
                raise ValueError("proprio_is_pad must include the f0-aligned state")
        if state_is_pad[:, 0].any():
            raise ValueError("state aligned with f0 cannot be padded")
        state = proprio[:, 0].to(device=self.device, dtype=self.torch_dtype)
        dim_is_pad = self._normalize_batch_mask(
            sample.get("proprio_dim_is_pad"),
            batch_size=batch_size,
            length=state.shape[-1],
            name="proprio_dim_is_pad",
            device=self.device,
        )
        return state.masked_fill(dim_is_pad, 0.0)

    def training_loss(
        self,
        sample: dict[str, Any],
        tiled: bool = False,
        *,
        base_progress: torch.Tensor | None = None,
        video_noise: torch.Tensor | None = None,
        action_noise: torch.Tensor | None = None,
        keyframe_noise: torch.Tensor | None = None,
        return_diagnostics: bool = False,
        debug_block_indices: Sequence[int] | None = None,
    ):
        del tiled
        video = sample["video"].to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        action = sample["action"].to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"video must be [B,3,T,H,W], got {tuple(video.shape)}")
        if video.shape[2] % 17 != 5:
            raise ValueError(f"H3 video frame count must be 5+17k, got {video.shape[2]}")
        if video.shape[-2] % 32 or video.shape[-1] % 32:
            raise ValueError("H3 video height and width must be divisible by 32")
        if action.ndim != 3 or action.shape[-1] != self.action_expert.action_dim:
            raise ValueError(
                f"action must be [B,N,{self.action_expert.action_dim}], got "
                f"{tuple(action.shape)}"
            )
        batch_size = video.shape[0]
        if action.shape[0] != batch_size:
            raise ValueError("video and action batch sizes must match")
        image_is_pad = sample.get("image_is_pad")
        if image_is_pad is not None and image_is_pad.to(torch.bool).any():
            raise ValueError("Scheme A does not train video loss on padded frames")

        first_frame = video[:, :, 0]
        qwen_embeddings, qwen_tags, qwen_valid = self._prepare_text_condition(
            sample, first_frame
        )
        state = self._prepare_state(sample, batch_size)

        with torch.no_grad():
            clean_video = self.vae.encode_video(
                video, device=self.device, process_image=False
            ).to(dtype=self.torch_dtype)
            clean_keyframe = self.vae.encode_video(
                first_frame.unsqueeze(2),
                device=self.device,
                process_image=True,
            ).to(dtype=self.torch_dtype)

        if base_progress is None:
            base_progress = torch.rand(
                batch_size, device=self.device, dtype=torch.float32
            )
        else:
            base_progress = base_progress.to(device=self.device, dtype=torch.float32)
            if base_progress.shape != (batch_size,):
                raise ValueError(f"base_progress must be [{batch_size}]")
        video_timestep = self.train_video_scheduler.timestep_from_progress(
            base_progress, dtype=self.torch_dtype
        )
        action_timestep = self.train_action_scheduler.timestep_from_progress(
            base_progress, dtype=self.torch_dtype
        )

        video_noise = (
            torch.randn_like(clean_video)
            if video_noise is None
            else video_noise.to(device=self.device, dtype=self.torch_dtype)
        )
        action_noise = (
            torch.randn_like(action)
            if action_noise is None
            else action_noise.to(device=self.device, dtype=self.torch_dtype)
        )
        keyframe_noise = (
            torch.randn_like(clean_keyframe)
            if keyframe_noise is None
            else keyframe_noise.to(device=self.device, dtype=self.torch_dtype)
        )
        if video_noise.shape != clean_video.shape:
            raise ValueError("video_noise must match the complete video latent shape")
        if action_noise.shape != action.shape:
            raise ValueError("action_noise must match action")
        if keyframe_noise.shape != clean_keyframe.shape:
            raise ValueError("keyframe_noise must match the image latent shape")

        noisy_video = self.train_video_scheduler.add_noise(
            clean_video, video_noise, video_timestep
        )
        noisy_action = self.train_action_scheduler.add_noise(
            action, action_noise, action_timestep
        )
        keyframe_condition = augment_keyframe_latents(
            clean_keyframe,
            keyframe_noise,
            strength=self.keyframe_condition_strength,
        )
        video_target = self.train_video_scheduler.training_target(
            clean_video, video_noise, video_timestep
        )
        action_target = self.train_action_scheduler.training_target(
            action, action_noise, action_timestep
        )

        action_is_pad = self._normalize_batch_mask(
            sample.get("action_is_pad"),
            batch_size=batch_size,
            length=action.shape[1],
            name="action_is_pad",
            device=self.device,
        )
        action_dim_is_pad = self._normalize_batch_mask(
            sample.get("action_dim_is_pad"),
            batch_size=batch_size,
            length=action.shape[2],
            name="action_dim_is_pad",
            device=self.device,
        )
        action_valid = ~action_is_pad
        action_input_valid = (
            action_valid.unsqueeze(-1) & ~action_dim_is_pad.unsqueeze(1)
        )
        noisy_action = noisy_action.masked_fill(~action_input_valid, 0.0)

        predictions = self.video_expert.forward_joint(
            action_expert=self.action_expert,
            qwen_embeddings=qwen_embeddings,
            qwen_tags=qwen_tags,
            qwen_valid=qwen_valid,
            clean_keyframe_latents=keyframe_condition,
            noisy_video_latents=noisy_video,
            video_timestep=video_timestep,
            noisy_action_tokens=noisy_action,
            action_timestep=action_timestep,
            state_tokens=state,
            action_valid=action_valid,
            keyframe_condition_strength=self.keyframe_condition_strength,
            video_fps=self.video_fps,
            action_fps=self._effective_action_rope_fps(
                num_frames=video.shape[2], action_horizon=action.shape[1]
            ),
            video_timestep_scale=float(
                self.train_video_scheduler.num_train_timesteps
            ),
            action_timestep_scale=float(
                self.train_action_scheduler.num_train_timesteps
            ),
            detach_h3_for_action=self.stop_action_gradient_to_h3,
            return_debug=bool(return_diagnostics),
            debug_block_indices=debug_block_indices,
        )
        video_prediction = predictions["video_prediction"]
        action_prediction = predictions["action_prediction"]
        if video_prediction.shape != video_target.shape:
            raise ValueError("video prediction must cover every full-video latent")
        if action_prediction.shape != action_target.shape:
            raise ValueError("action prediction must cover only action target rows")

        video_element_loss = F.mse_loss(
            video_prediction.float(), video_target.float(), reduction="none"
        )
        video_per_sample = video_element_loss.flatten(1).mean(dim=1)
        video_weight = self.train_video_scheduler.training_weight(video_timestep)
        loss_video = (video_per_sample * video_weight.float()).mean()

        action_element_loss = F.mse_loss(
            action_prediction.float(), action_target.float(), reduction="none"
        )
        action_element_valid = action_input_valid
        action_per_sample = (
            (action_element_loss * action_element_valid).flatten(1).sum(dim=1)
            / action_element_valid.flatten(1).sum(dim=1).clamp(min=1)
        )
        action_weight = self.train_action_scheduler.training_weight(action_timestep)
        loss_action = (action_per_sample * action_weight.float()).mean()

        loss = (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_action * loss_action
        )
        sigma_video = video_timestep.float() / float(
            self.train_video_scheduler.num_train_timesteps
        )
        sigma_action = action_timestep.float() / float(
            self.train_action_scheduler.num_train_timesteps
        )
        metrics = {
            "loss_video": float(loss_video.detach()),
            "loss_action": float(loss_action.detach()),
            "base_progress_mean": float(base_progress.mean()),
            "sigma_video_mean": float(sigma_video.mean()),
            "sigma_action_mean": float(sigma_action.mean()),
        }
        if not return_diagnostics:
            return loss, metrics
        return loss, metrics, {
            "loss_video": loss_video,
            "loss_action": loss_action,
            "base_progress": base_progress.detach(),
            "video_noise": video_noise.detach(),
            "action_noise": action_noise.detach(),
            "keyframe_noise": keyframe_noise.detach(),
            "h3_hidden_by_block": predictions["debug"]["h3_hidden_by_block"],
        }

    @torch.no_grad()
    def _encode_video_latents(self, video: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.vae.encode_video(
            video, device=self.device, process_image=False
        ).to(dtype=self.torch_dtype)

    @torch.no_grad()
    def _decode_latents(self, latents: torch.Tensor, **_: Any) -> list[Image.Image]:
        frame_num = _.get("frame_num")
        decoded = self.vae.decode(
            latents,
            device=self.device,
            frame_num=None if frame_num is None else int(frame_num),
        )
        decoded = decoded[0].detach().float().cpu().clamp(-1, 1)
        frames: list[Image.Image] = []
        for index in range(decoded.shape[1]):
            array = (
                ((decoded[:, index] + 1) * 127.5)
                .byte()
                .permute(1, 2, 0)
                .numpy()
            )
            frames.append(Image.fromarray(array))
        return frames

    @torch.no_grad()
    def infer(
        self,
        *,
        input_image: torch.Tensor,
        num_frames: int,
        action_horizon: int,
        proprio: torch.Tensor,
        prompt: str | Sequence[str] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_token_tags: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        proprio_dim_is_pad: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        negative_prompt: str | None = None,
        text_cfg_scale: float = 1.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: float | None = None,
        action_sigma_shift: float | None = None,
        seed: int | None = None,
        rand_device: str | torch.device = "cpu",
        tiled: bool = False,
        video_noise: torch.Tensor | None = None,
        action_noise: torch.Tensor | None = None,
        keyframe_noise: torch.Tensor | None = None,
        decode_video: bool = False,
    ) -> dict[str, Any]:
        """Jointly sample video latents and actions, with optional pixel decode."""

        del negative_prompt, text_cfg_scale, action_cfg_scale, tiled
        if decode_video and int(num_frames) == 5:
            raise NotImplementedError(
                "Five-frame H3 latent rollout is supported, but the released "
                "VAE cannot faithfully decode its two retained prefix latents."
            )
        if action is not None:
            raise ValueError(
                "FastWAM-H3 inference does not accept ground-truth action conditioning"
            )
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if (
            input_image.ndim != 4
            or input_image.shape[0] != 1
            or input_image.shape[1] != 3
        ):
            raise ValueError(
                "input_image must be [1,3,H,W] or [3,H,W] for H3 inference"
            )
        batch_size, _, height, width = input_image.shape
        num_frames = int(num_frames)
        action_horizon = int(action_horizon)
        num_inference_steps = int(num_inference_steps)
        if num_frames < 5 or num_frames % 17 != 5:
            raise ValueError(f"H3 num_frames must be 5+17k, got {num_frames}")
        if height % 32 or width % 32:
            raise ValueError("H3 input height and width must be divisible by 32")
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if num_inference_steps < 2:
            raise ValueError(
                "H3 num_inference_steps counts sigma points and must be at least 2"
            )

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        if proprio.ndim == 1:
            proprio = proprio.view(1, 1, -1)
        elif proprio.ndim == 2:
            proprio = proprio.unsqueeze(1)
        infer_sample: dict[str, Any] = {
            "proprio": proprio,
            "proprio_dim_is_pad": proprio_dim_is_pad,
        }
        if prompt_embeds is not None:
            infer_sample.update(
                {
                    "prompt_embeds": prompt_embeds,
                    "prompt_token_tags": prompt_token_tags,
                    "prompt_attention_mask": prompt_attention_mask,
                }
            )
        elif context is not None:
            infer_sample.update(
                {
                    "context": context,
                    "prompt_token_tags": prompt_token_tags,
                    "prompt_attention_mask": (
                        prompt_attention_mask
                        if prompt_attention_mask is not None
                        else context_mask
                    ),
                }
            )
        else:
            infer_sample["prompt"] = prompt

        qwen_embeddings, qwen_tags, qwen_valid = self._prepare_text_condition(
            infer_sample, input_image
        )
        state = self._prepare_state(infer_sample, batch_size)
        clean_keyframe = self.vae.encode_video(
            input_image.unsqueeze(2),
            device=self.device,
            process_image=True,
        ).to(dtype=self.torch_dtype)

        latent_t = MiniMaxH3VAEAdapter.latent_temporal_length(num_frames)
        latent_h = height // int(self.vae.upsampling_factor)
        latent_w = width // int(self.vae.upsampling_factor)
        latent_channels = int(
            getattr(self.vae, "z_dim", getattr(self.vae.model, "z_dim"))
        )
        video_shape = (
            batch_size,
            latent_channels,
            latent_t,
            latent_h,
            latent_w,
        )
        action_shape = (batch_size, action_horizon, self.action_expert.action_dim)
        generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(int(seed))
        )

        def prepare_noise(
            supplied: torch.Tensor | None,
            shape: tuple[int, ...],
            name: str,
        ) -> torch.Tensor:
            if supplied is None:
                value = torch.randn(
                    shape,
                    generator=generator,
                    device=rand_device,
                    dtype=torch.float32,
                )
            else:
                if tuple(supplied.shape) != shape:
                    raise ValueError(f"{name} must have shape {shape}")
                value = supplied
            return value.to(device=self.device, dtype=self.torch_dtype)

        latents_video = prepare_noise(video_noise, video_shape, "video_noise")
        latents_action = prepare_noise(action_noise, action_shape, "action_noise")
        keyframe_noise = prepare_noise(
            keyframe_noise, tuple(clean_keyframe.shape), "keyframe_noise"
        )
        keyframe_condition = augment_keyframe_latents(
            clean_keyframe,
            keyframe_noise,
            strength=self.keyframe_condition_strength,
        )
        action_valid = torch.ones(
            (batch_size, action_horizon), dtype=torch.bool, device=self.device
        )

        video_timesteps, video_deltas = (
            self.infer_video_scheduler.build_inference_schedule(
                num_inference_steps - 1,
                self.device,
                latents_video.dtype,
                shift_override=sigma_shift,
            )
        )
        action_timesteps, action_deltas = (
            self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps - 1,
                self.device,
                latents_action.dtype,
                shift_override=action_sigma_shift,
            )
        )
        if video_timesteps.shape != action_timesteps.shape:
            raise ValueError("video and action inference schedules must align by step")

        for video_t, video_delta, action_t, action_delta in zip(
            video_timesteps,
            video_deltas,
            action_timesteps,
            action_deltas,
        ):
            predictions = self.video_expert.forward_joint(
                action_expert=self.action_expert,
                qwen_embeddings=qwen_embeddings,
                qwen_tags=qwen_tags,
                qwen_valid=qwen_valid,
                clean_keyframe_latents=keyframe_condition,
                noisy_video_latents=latents_video,
                video_timestep=video_t.expand(batch_size),
                noisy_action_tokens=latents_action,
                action_timestep=action_t.expand(batch_size),
                state_tokens=state,
                action_valid=action_valid,
                keyframe_condition_strength=self.keyframe_condition_strength,
                video_fps=self.video_fps,
                action_fps=self._effective_action_rope_fps(
                    num_frames=num_frames, action_horizon=action_horizon
                ),
                video_timestep_scale=float(
                    self.infer_video_scheduler.num_train_timesteps
                ),
                action_timestep_scale=float(
                    self.infer_action_scheduler.num_train_timesteps
                ),
                detach_h3_for_action=self.stop_action_gradient_to_h3,
            )
            latents_video = self.infer_video_scheduler.step(
                predictions["video_prediction"], video_delta, latents_video
            )
            latents_action = self.infer_action_scheduler.step(
                predictions["action_prediction"], action_delta, latents_action
            )

        output: dict[str, Any] = {
            "video_latents": latents_video[0].detach().float().cpu(),
            "action": latents_action[0].detach().float().cpu(),
        }
        if decode_video:
            output["video"] = self._decode_latents(
                latents_video, frame_num=num_frames
            )
        return output

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: str | Sequence[str] | None,
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        decode_video: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compatibility wrapper; `infer` remains the only joint sampler."""

        kwargs.pop("test_action_with_infer_action", None)
        if proprio is None:
            raise ValueError("FastWAM-H3 infer_joint requires proprio")
        return self.infer(
            prompt=prompt,
            input_image=input_image,
            num_frames=num_video_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            decode_video=decode_video,
            **kwargs,
        )

    @torch.no_grad()
    def infer_action(
        self,
        prompt: str | Sequence[str] | None,
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
        proprio: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Evaluator-compatible action result backed by joint denoising."""

        kwargs.pop("decode_video", None)
        output = self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            decode_video=False,
            **kwargs,
        )
        return {"action": output["action"]}

    def _action_expert_checkpoint_config(self) -> dict[str, int]:
        return {
            "action_dim": int(self.action_expert.action_dim),
            "state_dim": int(self.action_expert.state_dim),
            "hidden_size": int(self.action_expert.hidden_size),
            "num_layers": int(self.action_expert.num_layers),
            "num_heads": int(self.action_expert.num_heads),
            "attention_head_dim": int(self.action_expert.attn_head_dim),
        }

    def _h3_lora_checkpoint_config(self) -> dict[str, Any]:
        branches = self._named_h3_lora_branches()
        return {
            "rank": self.h3_lora_rank,
            "alpha": self.h3_lora_alpha,
            "dropout": self.h3_lora_dropout,
            "targets": list(branches),
            "base_h3_fingerprint": self.base_h3_fingerprint,
        }

    def save_checkpoint(self, path: str | Path, optimizer=None, step=None):
        branches = self._named_h3_lora_branches()
        if self.h3_lora_rank > 0 and not branches:
            raise RuntimeError("Configured H3 LoRA branches are missing")
        payload = {
            "schema_version": 3,
            "action_expert": {
                "config": self._action_expert_checkpoint_config(),
                "state_dict": self.action_expert.state_dict(),
            },
            "step": step,
            "backbone": "MiniMax-H3-FL2VA-Scheme-A",
            "h3_lora_config": self._h3_lora_checkpoint_config(),
            "h3_lora": {
                name: branch.state_dict() for name, branch in branches.items()
            },
        }
        if not self.freeze_video_expert:
            payload["video_expert"] = self.video_expert.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path, optimizer=None):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != 3:
            raise ValueError(
                "Checkpoint is not a FastWAM-H3 Scheme A schema-3 checkpoint"
            )
        action_payload = payload.get("action_expert")
        if not isinstance(action_payload, dict):
            raise ValueError("Checkpoint is missing the Action Expert payload")
        expected_action_config = self._action_expert_checkpoint_config()
        if action_payload.get("config") != expected_action_config:
            raise ValueError(
                "Checkpoint Action Expert config does not match the model: "
                f"{action_payload.get('config')} != {expected_action_config}"
            )
        expected_lora_config = self._h3_lora_checkpoint_config()
        if payload.get("h3_lora_config") != expected_lora_config:
            raise ValueError(
                "Checkpoint H3 LoRA config does not match the model: "
                f"{payload.get('h3_lora_config')} != {expected_lora_config}"
            )
        branches = self._named_h3_lora_branches()
        lora_payload = payload.get("h3_lora")
        if not isinstance(lora_payload, dict) or set(lora_payload) != set(branches):
            raise ValueError(
                "Checkpoint H3 LoRA module names do not match the model"
            )
        self.action_expert.load_state_dict(
            action_payload["state_dict"], strict=True
        )
        for name, branch in branches.items():
            branch.load_state_dict(lora_payload[name], strict=True)
        if "video_expert" in payload:
            self.video_expert.load_state_dict(payload["video_expert"], strict=True)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args: Any, **kwargs: Any):
        return self.training_loss(*args, **kwargs)
