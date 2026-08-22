"""Adapter from the official MiniMax-H3 visual VAE to FastWAM's VAE API."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn


def augment_keyframe_latents(
    clean_latents: torch.Tensor,
    noise: torch.Tensor,
    *,
    strength: float = 0.999,
) -> torch.Tensor:
    """Apply H3's near-clean FL2VA keyframe augmentation."""

    if clean_latents.shape != noise.shape:
        raise ValueError(
            f"clean_latents and noise must have the same shape, got "
            f"{tuple(clean_latents.shape)} and {tuple(noise.shape)}"
        )
    strength = float(strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {strength}")
    return strength * clean_latents + (1.0 - strength) * noise


class MiniMaxH3VAEAdapter(nn.Module):
    temporal_downsample_factor = 4
    upsampling_factor = 16

    @staticmethod
    def latent_temporal_length(num_frames: int) -> int:
        """Return H3's native temporal latent length for a 5+17k clip."""

        num_frames = int(num_frames)
        if num_frames < 5 or (num_frames - 5) % 17:
            raise ValueError(f"H3 num_frames must be 5+17k, got {num_frames}")
        return 2 + 5 * ((num_frames - 5) // 17)

    def __init__(
        self,
        component_dir: str | Path,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.component_dir = Path(component_dir)
        config = json.loads((self.component_dir / "config.json").read_text())
        self.z_dim = int(config["latent_channels"])
        self.model_info = type("H3VAEInfo", (), {"z_dim": self.z_dim})()
        self.model = self.model_info

        # The released component is a self-contained Python package. Import it
        # as the `video_vae` namespace so its relative imports remain valid.
        component_parent = str(self.component_dir.parent)
        if component_parent not in sys.path:
            sys.path.insert(0, component_parent)
        module = importlib.import_module("video_vae.minimax_h3_video_vae")
        distribution_module = importlib.import_module("video_vae.vae_module")
        wrapper = module.MiniMaxH3VideoVAE.from_pretrained(str(self.component_dir))
        # The released VAE checkpoint is FP32. Lower precision is used only by
        # the official decoder autocast recipe, never by downcasting weights.
        del dtype
        self.vae = wrapper.model.to(device=device, dtype=torch.float32).eval()
        self._distribution_cls = distribution_module.DiagonalGaussianDistribution

        self.register_buffer(
            "latents_mean",
            torch.tensor(config["latents_mean"], dtype=torch.float32).view(
                1, self.z_dim, 1, 1, 1
            ),
            persistent=False,
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(config["latents_std"], dtype=torch.float32).view(
                1, self.z_dim, 1, 1, 1
            ),
            persistent=False,
        )

    def _normalize(self, latents: torch.Tensor) -> torch.Tensor:
        return (
            (latents.float() - self.latents_mean.to(latents.device))
            / self.latents_std.to(latents.device)
        ).to(latents.dtype)

    def _denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        return (
            latents.float() * self.latents_std.to(latents.device)
            + self.latents_mean.to(latents.device)
        ).to(latents.dtype)

    def _normalize_posterior(
        self, mean: torch.Tensor, logvar: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        std = self.latents_std.to(device=mean.device, dtype=torch.float32)
        normalized_mean = (
            mean.float() - self.latents_mean.to(mean.device)
        ) / std
        normalized_logvar = logvar.float() - 2.0 * torch.log(std)
        return normalized_mean, normalized_logvar

    def _encode_prefix_moments(
        self, transformed_video: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        """Return one prefix video's raw posterior moments without sampling."""

        if transformed_video.ndim == 4:
            transformed_video = transformed_video.unsqueeze(0)
        _, _, _, height, width = transformed_video.shape
        new_h, new_w = self.vae.processor._align_to_total_patch_size(height, width)
        video = self.vae.processor._crop_to_align(
            transformed_video, new_h, new_w, is_video=True
        )
        model_alignment = (
            self.vae.token_drop,
            self.vae.frame_drop,
            self.vae.token_overlap,
            self.vae.frame_overlap,
        )
        processor_alignment = (
            self.vae.processor.token_overlap,
            self.vae.processor.frame_overlap,
        )
        self.vae.token_drop = 0
        self.vae.frame_drop = 0
        self.vae.token_overlap = 0
        self.vae.frame_overlap = 0
        self.vae.processor.token_overlap = 0
        self.vae.processor.frame_overlap = 0
        try:
            leading, trailing, drop_tokens = (
                self.vae.processor.align_video_length_2pass(video.shape[2])
            )
            if leading > 0:
                black = self.vae.processor.transform(
                    video.new_zeros(leading, 3, new_h, new_w)
                )
                video = torch.cat(
                    [black.unsqueeze(0).permute(0, 2, 1, 3, 4), video], dim=2
                )
            if trailing > 0:
                black = self.vae.processor.transform(
                    video.new_zeros(trailing, 3, new_h, new_w)
                )
                video = torch.cat(
                    [video, black.unsqueeze(0).permute(0, 2, 1, 3, 4)], dim=2
                )
            moments = self.vae.encode_temporal(video)
            if drop_tokens > 0:
                moments = moments[:, :, :-drop_tokens]
            return moments, int(leading)
        finally:
            (
                self.vae.token_drop,
                self.vae.frame_drop,
                self.vae.token_overlap,
                self.vae.frame_overlap,
            ) = model_alignment
            (
                self.vae.processor.token_overlap,
                self.vae.processor.frame_overlap,
            ) = processor_alignment

    @torch.no_grad()
    def encode_video_posterior(
        self, video: torch.Tensor, device: str | torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode full-video prefix posterior moments for online resampling."""

        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"video must be [B,3,T,H,W], got {tuple(video.shape)}")
        target_device = next(self.vae.parameters()).device
        pixels = ((video.to(target_device, dtype=torch.float32) + 1.0) * 0.5).clamp(
            0, 1
        )
        transformed = self.vae.processor.transform_tensor(pixels).to(torch.float32)
        means = []
        logvars = []
        for sample in transformed.unbind(dim=0):
            moments, leading = self._encode_prefix_moments(sample)
            if leading != 0:
                raise ValueError(
                    "FastWAM clips must start on an H3 token boundary; "
                    f"got prefix_pad_frames={leading}"
                )
            distribution = self._distribution_cls(moments)
            mean, logvar = self._normalize_posterior(
                distribution.mean, distribution.logvar
            )
            means.append(mean.squeeze(0))
            logvars.append(logvar.squeeze(0))
        mean = torch.stack(means)
        logvar = torch.stack(logvars)
        if device is not None:
            mean = mean.to(device)
            logvar = logvar.to(device)
        return mean, logvar

    @torch.no_grad()
    def encode_keyframe_condition(
        self,
        image: torch.Tensor,
        device: str | torch.device | None = None,
        *,
        seed: int = 42,
    ) -> torch.Tensor:
        """Apply the official deterministic FL2VA keyframe encode recipe."""

        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"image must be [B,3,H,W], got {tuple(image.shape)}")
        target_device = next(self.vae.parameters()).device
        # The released pipeline converts the PIL keyframe to uint8 before
        # dividing by 255 and applying ImageNet normalization.
        pixels_uint8 = (
            ((image.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
        )
        pixels = pixels_uint8.to(target_device, dtype=torch.float32).div(255.0)
        transformed = self.vae.processor.transform_tensor(pixels).to(torch.float32)
        moments = self.vae._adaptive_encode(transformed.unsqueeze(2))
        distribution = self._distribution_cls(moments)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        latents = distribution.sample(generator=generator)
        latents = self.vae.trim_code(latents, 1)
        # Official FL2VA rounds the sampled raw condition to FP16 before
        # applying FP32 latent normalization.
        latents = latents.to(torch.float16).float()
        normalized = self._normalize(latents).float()
        if device is not None:
            normalized = normalized.to(device)
        return normalized

    @torch.no_grad()
    def encode(
        self,
        video: torch.Tensor,
        device: str | torch.device | None = None,
        **_: object,
    ) -> torch.Tensor:
        return self.encode_video(video, device=device, process_image=False)

    @torch.no_grad()
    def encode_video(
        self,
        video: torch.Tensor,
        device: str | torch.device | None = None,
        *,
        process_image: bool = False,
    ) -> torch.Tensor:
        """Run the released H3 image or video encoder through one explicit API."""

        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"video must be [B,3,T,H,W], got {tuple(video.shape)}")
        if process_image and video.shape[2] != 1:
            raise ValueError(
                f"process_image=True requires one frame, got {video.shape[2]}"
            )
        target_device = next(self.vae.parameters()).device
        # FastWAM supplies [-1, 1]; H3's processor expects pixels in [0, 1].
        pixels = ((video.to(target_device).float() + 1.0) * 0.5).clamp(0, 1)
        model_dtype = next(self.vae.parameters()).dtype
        if process_image:
            transformed = self.vae.processor.transform_tensor(pixels[:, :, 0]).to(
                model_dtype
            )
            latents = self.vae.encode_images(
                list(transformed.unbind(dim=0)),
                transform_input=False,
            )
            latents = [
                latent if latent.ndim == 4 else latent.unsqueeze(1)
                for latent in latents
            ]
        else:
            transformed = self.vae.processor.transform_tensor(pixels).to(model_dtype)
            encoded = self.vae.encode_videos(
                list(transformed.unbind(dim=0)),
                transform_input=False,
                encode_prefix=True,
            )
            if not isinstance(encoded, tuple) or len(encoded) != 2:
                raise TypeError(
                    "H3 encode_prefix=True must return "
                    "(video_latents, prefix_pad_frames)"
                )
            latents, prefix_pad_frames = encoded
            if len(prefix_pad_frames) != video.shape[0]:
                raise ValueError(
                    "H3 VAE returned one prefix pad count per input video"
                )
            if any(int(value) != 0 for value in prefix_pad_frames):
                raise ValueError(
                    "FastWAM clips must already start on an H3 token boundary; "
                    f"got prefix_pad_frames={prefix_pad_frames}"
                )
        stacked = torch.stack(latents, dim=0)
        stacked = self._normalize(stacked)
        if device is not None:
            stacked = stacked.to(device)
        return stacked

    @torch.no_grad()
    def encode_image(
        self,
        image: torch.Tensor,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"image must be [B,3,H,W], got {tuple(image.shape)}")
        return self.encode_video(
            image.unsqueeze(2), device=device, process_image=True
        )

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        device: str | torch.device | None = None,
        frame_num: int | None = None,
        **_: object,
    ) -> torch.Tensor:
        target_device = next(self.vae.parameters()).device
        model_dtype = next(self.vae.parameters()).dtype
        raw = self._denormalize(latents.to(target_device)).to(model_dtype)
        if frame_num is not None and int(frame_num) == 5 and raw.shape[2] == 2:
            raise NotImplementedError(
                "The released H3 VAE cannot faithfully decode a five-frame "
                "rollout from its two retained prefix latents. Use the latent "
                "rollout directly or request a natively decodable frame count."
            )
        with torch.autocast(
            device_type=target_device.type,
            dtype=torch.float16,
            enabled=target_device.type == "cuda",
        ):
            decoded = self.vae.decode_base(raw, frame_num=frame_num)
        pixels = self.vae.processor.revert_tensor(decoded)
        # FastWAM's decoder contract is [-1, 1].
        value = pixels * 2.0 - 1.0
        if device is not None:
            value = value.to(device)
        return value

    def train(self, mode: bool = True):
        # H3 VAE remains frozen regardless of the outer model's train mode.
        super().train(False)
        self.vae.eval()
        return self
