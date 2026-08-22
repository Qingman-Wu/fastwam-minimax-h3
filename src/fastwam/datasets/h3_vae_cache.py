"""Strict static cache for H3 keyframe latents and video posterior moments."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import torch


H3_VAE_CACHE_SCHEMA_VERSION = 1
H3_VAE_CACHE_MANIFEST = "h3-vae-cache-manifest.json"
H3_VAE_IMPLEMENTATION_SIGNATURE = "fastwam-h3-vae-fp32-v2"
H3_VAE_PIXEL_CONTRACT = (
    "dataset-fp32-minus1-plus1:to-0-1:released-transform-tensor"
)
H3_VAE_LATENT_CONTRACT = "config-latents-mean-std:fp32-affine-normalization"
H3_VAE_PREFIX_CONTRACT = "encode-prefix:true:5-to-2:22-to-7:39-to-12"
H3_VAE_CACHE_KEYS = (
    "clean_keyframe_latents",
    "video_posterior_mean",
    "video_posterior_logvar",
)


def _tensor_digest(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(memoryview(value.view(torch.uint8).numpy()))
    return f"sha256:{digest.hexdigest()}"


def initialize_h3_vae_cache(
    cache_dir: str | Path,
    *,
    vae_fingerprint: str,
    processor_signature: str = "unspecified",
    implementation_signature: str = H3_VAE_IMPLEMENTATION_SIGNATURE,
    overwrite: bool = False,
) -> dict[str, Any]:
    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": H3_VAE_CACHE_SCHEMA_VERSION,
        "vae_fingerprint": str(vae_fingerprint),
        "processor_signature": str(processor_signature),
        "implementation_signature": str(implementation_signature),
        "keyframe_semantics": "seed42-sample-fp16-round-fp32-normalize",
        "video_semantics": "fp32-normalized-posterior-mean-logvar",
        "pixel_contract": H3_VAE_PIXEL_CONTRACT,
        "latent_contract": H3_VAE_LATENT_CONTRACT,
        "prefix_contract": H3_VAE_PREFIX_CONTRACT,
        "dtype": "torch.float32",
    }
    path = cache_dir / H3_VAE_CACHE_MANIFEST
    if path.is_file() and json.loads(path.read_text()) != manifest and not overwrite:
        raise ValueError("Existing H3 VAE cache manifest is incompatible")
    temporary = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return manifest


def load_h3_vae_cache_manifest(cache_dir: str | Path) -> dict[str, Any]:
    path = Path(cache_dir).expanduser() / H3_VAE_CACHE_MANIFEST
    if not path.is_file():
        raise FileNotFoundError(f"Missing H3 VAE cache manifest: {path}")
    manifest = json.loads(path.read_text())
    if (
        manifest.get("schema_version") != H3_VAE_CACHE_SCHEMA_VERSION
        or not manifest.get("vae_fingerprint")
        or not manifest.get("processor_signature")
        or manifest.get("implementation_signature")
        != H3_VAE_IMPLEMENTATION_SIGNATURE
        or manifest.get("pixel_contract") != H3_VAE_PIXEL_CONTRACT
        or manifest.get("latent_contract") != H3_VAE_LATENT_CONTRACT
        or manifest.get("prefix_contract") != H3_VAE_PREFIX_CONTRACT
        or manifest.get("dtype") != "torch.float32"
    ):
        raise ValueError(f"Incompatible H3 VAE cache manifest: {path}")
    return manifest


def _video_content_digest(video: torch.Tensor) -> str:
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError(f"H3 VAE cache video must be [3,T,H,W], got {video.shape}")
    value = video.detach().cpu().to(torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(memoryview(value.view(torch.uint8).numpy()))
    return f"sha256:{digest.hexdigest()}"


def h3_vae_cache_path(cache_dir: str | Path, video: torch.Tensor) -> Path:
    manifest = load_h3_vae_cache_manifest(cache_dir)
    video_digest = _video_content_digest(video)
    digest = hashlib.sha256()
    digest.update(
        (
            f"h3-vae-cache-v{H3_VAE_CACHE_SCHEMA_VERSION}\0"
            f"{manifest['vae_fingerprint']}\0"
            f"{manifest['processor_signature']}\0"
            f"{manifest['implementation_signature']}\0"
        ).encode()
    )
    digest.update(video_digest.encode())
    return Path(cache_dir).expanduser() / f"{digest.hexdigest()}.h3-vae-v1.pt"


def save_h3_vae_cache(
    cache_dir: str | Path,
    *,
    video: torch.Tensor,
    clean_keyframe_latents: torch.Tensor,
    video_posterior_mean: torch.Tensor,
    video_posterior_logvar: torch.Tensor,
) -> Path:
    manifest = load_h3_vae_cache_manifest(cache_dir)
    tensors = (
        clean_keyframe_latents.detach().cpu().float().contiguous(),
        video_posterior_mean.detach().cpu().float().contiguous(),
        video_posterior_logvar.detach().cpu().float().contiguous(),
    )
    path = h3_vae_cache_path(cache_dir, video)
    payload = {
        "schema_version": H3_VAE_CACHE_SCHEMA_VERSION,
        "vae_fingerprint": manifest["vae_fingerprint"],
        "processor_signature": manifest["processor_signature"],
        "video_content_sha256": _video_content_digest(video),
        **dict(zip(H3_VAE_CACHE_KEYS, tensors, strict=True)),
        "payload_sha256": _tensor_digest(*tensors),
    }
    temporary = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_h3_vae_cache(
    cache_dir: str | Path, *, video: torch.Tensor
) -> dict[str, torch.Tensor]:
    manifest = load_h3_vae_cache_manifest(cache_dir)
    path = h3_vae_cache_path(cache_dir, video)
    if not path.is_file():
        error = FileNotFoundError(f"Missing H3 VAE cache: {path}")
        error.filename = str(path)
        raise error
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != H3_VAE_CACHE_SCHEMA_VERSION
        or payload.get("vae_fingerprint") != manifest["vae_fingerprint"]
        or payload.get("processor_signature") != manifest["processor_signature"]
        or payload.get("video_content_sha256") != _video_content_digest(video)
    ):
        raise ValueError(f"Incompatible H3 VAE cache payload: {path}")
    tensors = tuple(payload[key] for key in H3_VAE_CACHE_KEYS)
    if any(
        not isinstance(tensor, torch.Tensor)
        or tensor.dtype != torch.float32
        or tensor.ndim != 4
        or not torch.isfinite(tensor).all()
        for tensor in tensors
    ):
        raise ValueError(f"Invalid H3 VAE cache tensors: {path}")
    keyframe, mean, logvar = tensors
    if keyframe.shape[1] != 1 or mean.shape != logvar.shape:
        raise ValueError(f"Invalid H3 VAE cache temporal shapes: {path}")
    if payload.get("payload_sha256") != _tensor_digest(*tensors):
        raise ValueError(f"H3 VAE cache checksum mismatch: {path}")
    return dict(zip(H3_VAE_CACHE_KEYS, tensors, strict=True))
