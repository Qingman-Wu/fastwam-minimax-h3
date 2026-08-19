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
        wrapper = module.MiniMaxH3VideoVAE.from_pretrained(str(self.component_dir))
        self.vae = wrapper.model.to(device=device, dtype=dtype).eval()

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
            latents = self.vae.encode_videos(
                list(transformed.unbind(dim=0)),
                transform_input=False,
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
        raw = self._denormalize(latents.to(target_device))
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
