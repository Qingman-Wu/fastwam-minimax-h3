"""MiniMax-H3 packed video/text backbone primitives used by FastWAM."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    left, right = torch.chunk(x, 2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply H3's partial 3D RoPE to [B, S, heads, head_dim]."""
    rot_dim = int(freqs.shape[-1])
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    cos = freqs.cos().to(dtype=x.dtype)[None, :, None, :]
    sin = freqs.sin().to(dtype=x.dtype)[None, :, None, :]
    return torch.cat((x_rot * cos + _rotate_half(x_rot) * sin, x_pass), dim=-1)


class H3RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.eps)
        return value.to(dtype=dtype) * self.weight


class H3TimeEmbedder(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, output_dim: int) -> None:
        super().__init__()
        self.frequency_embedding_size = int(input_dim)
        self.proj_in = nn.Linear(input_dim, hidden_size, bias=True)
        self.proj_out = nn.Linear(hidden_size, output_dim, bias=True)

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / half
        )
        args = timestep.float()[:, None] * freqs[None]
        embedding = torch.cat((args.cos(), args.sin()), dim=-1).to(dtype)
        return self.proj_out(F.silu(self.proj_in(embedding)))


class H3RoPE(nn.Module):
    def __init__(self, inv_freq_len: int) -> None:
        super().__init__()
        self.inv_freq_len = int(inv_freq_len)
        steps = torch.arange(inv_freq_len, dtype=torch.float32)
        self.inv_freq = nn.Parameter(1.0 / (10000.0 ** (steps / inv_freq_len)))

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        # position_ids: [S, 3] in (time, height, width) order.
        per_axis = position_ids.float().unsqueeze(-1) * self.inv_freq.float()[None, None]
        temporal, height, width = per_axis.unbind(dim=1)
        half = torch.cat((temporal, height, width), dim=-1)
        return torch.cat((half, half), dim=-1)


class H3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm_eps: float,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_attention_heads)
        self.head_dim = int(attention_head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.qkv_proj = nn.Linear(hidden_size, self.inner_dim * 3, bias=False)
        self.q_norm = H3RMSNorm(self.head_dim, qk_norm_eps)
        self.k_norm = H3RMSNorm(self.head_dim, qk_norm_eps)
        self.out_proj = nn.Linear(self.inner_dim, hidden_size, bias=False)

    def project_qkv(
        self, x: torch.Tensor, rope_freqs: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x).view(
            batch, seq_len, self.num_heads, 3, self.head_dim
        )
        q = self.q_norm(qkv[:, :, :, 0])
        k = self.k_norm(qkv[:, :, :, 1])
        v = qkv[:, :, :, 2]
        if rope_freqs is not None:
            q = apply_rope(q, rope_freqs)
            k = apply_rope(k, rope_freqs)
        return q, k, v

    def output(self, attention_output: torch.Tensor) -> torch.Tensor:
        return self.out_proj(attention_output.flatten(2))

    def forward(
        self, x: torch.Tensor, rope_freqs: torch.Tensor | None = None
    ) -> torch.Tensor:
        q, k, v = self.project_qkv(x, rope_freqs)
        attended = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        ).transpose(1, 2)
        return self.output(attended)


class H3MLP(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size * 2, bias=False)
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * up)


class H3TokenRefinerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_hidden_size: int,
        norm_eps: float,
        qk_norm_eps: float,
    ) -> None:
        super().__init__()
        self.norm1 = H3RMSNorm(hidden_size, norm_eps)
        self.norm2 = H3RMSNorm(hidden_size, norm_eps)
        self.attn = H3Attention(
            hidden_size, num_attention_heads, attention_head_dim, qk_norm_eps
        )
        self.mlp = H3MLP(hidden_size, ffn_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_freqs=None)
        return x + self.mlp(self.norm2(x))


class H3TokenRefiner(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_hidden_size: int,
        norm_eps: float,
        qk_norm_eps: float,
        final_norm_eps: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                H3TokenRefinerBlock(
                    hidden_size,
                    num_attention_heads,
                    attention_head_dim,
                    ffn_hidden_size,
                    norm_eps,
                    qk_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = H3RMSNorm(hidden_size, final_norm_eps)

    def forward(
        self, x: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"TokenRefiner input must be [S,H], got {tuple(x.shape)}")
        bounds = [int(value) for value in cu_seqlens.tolist()]
        hidden = x
        for block in self.blocks:
            hidden = torch.cat(
                [block(hidden[start:end].unsqueeze(0)).squeeze(0) for start, end in zip(bounds[:-1], bounds[1:])],
                dim=0,
            )
        return self.final_norm(hidden)


class H3AdaLNProjection(nn.Module):
    """Checkpoint-compatible three-modality H3 AdaLN projection."""

    def __init__(self, hidden_size: int, time_embed_dim: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.linear = nn.Linear(time_embed_dim, hidden_size * 6 * 3, bias=True)

    def video(self, time_embedding: torch.Tensor) -> tuple[torch.Tensor, ...]:
        batch = time_embedding.shape[0]
        values = self.linear(F.silu(time_embedding))
        values = values.view(batch, 3, 6, self.hidden_size)[:, 0]
        return values.unbind(dim=1)


class H3VideoBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_hidden_size: int,
        time_embed_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
    ) -> None:
        super().__init__()
        self.norm1 = H3RMSNorm(hidden_size, norm_eps)
        self.norm2 = H3RMSNorm(hidden_size, norm_eps)
        self.attn = H3Attention(
            hidden_size, num_attention_heads, attention_head_dim, qk_norm_eps
        )
        self.mlp = H3MLP(hidden_size, ffn_hidden_size)
        self.adaln_proj = H3AdaLNProjection(hidden_size, time_embed_dim)
        self.register_buffer(
            "cached_video_modulation", None, persistent=False
        )

    @torch.no_grad()
    def cache_zero_timestep_adaln(self, time_embedding: torch.Tensor) -> None:
        if self.adaln_proj is None:
            return
        values = self.adaln_proj.video(time_embedding)
        self.cached_video_modulation = torch.stack(
            [value[0].detach().contiguous() for value in values], dim=0
        )
        # H3 documents that its ~13B AdaLN branch can be precomputed and
        # omitted for inference-only deployment. FastWAM always prefills the
        # frozen visual branch at t=0, so keeping these weights wastes 26GB.
        self.adaln_proj = None

    def pre_attention(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
        rope_freqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        if self.cached_video_modulation is None:
            modulation = self.adaln_proj.video(time_embedding)
        else:
            modulation = tuple(
                value.unsqueeze(0).expand(time_embedding.shape[0], -1)
                for value in self.cached_video_modulation
            )
        shift_msa, scale_msa, *_ = modulation
        hidden = self.norm1(x)
        hidden = hidden * (1 + scale_msa[:, None]) + shift_msa[:, None]
        q, k, v = self.attn.project_qkv(hidden, rope_freqs)
        return q, k, v, modulation

    def post_attention(
        self,
        x: torch.Tensor,
        attention_output: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        _, _, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation
        x = x + gate_msa[:, None] * self.attn.output(attention_output)
        hidden = self.norm2(x)
        hidden = hidden * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        return x + gate_mlp[:, None] * self.mlp(hidden)


class H3FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        time_embed_dim: int,
        latents_dim: int,
        patch_size: tuple[int, int, int],
        final_norm_eps: float,
    ) -> None:
        super().__init__()
        self.norm = H3RMSNorm(hidden_size, final_norm_eps)
        # The released final AdaLN contains only one modality.
        self.adaln_proj = nn.Module()
        self.adaln_proj.linear = nn.Linear(time_embed_dim, hidden_size * 2, bias=True)
        patch_dim = latents_dim * math.prod(patch_size)
        self.video_out = nn.Linear(hidden_size, patch_dim, bias=True)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaln_proj.linear(F.silu(time_embedding)).chunk(2, dim=-1)
        hidden = self.norm(x)
        hidden = hidden * (1 + scale[:, None]) + shift[:, None]
        return self.video_out(hidden)


class MiniMaxH3VideoBackbone(nn.Module):
    """Video-only view of the released H3 Omni Transformer."""

    def __init__(
        self,
        *,
        hidden_size: int = 5376,
        ffn_hidden_size: int = 14336,
        num_layers: int = 50,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        latents_dim: int = 24,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_dim: int = 5120,
        token_refiner_num_layers: int = 2,
        timestep_input_dim: int = 256,
        time_embed_hidden_size: int = 5376,
        time_embed_dim: int = 2688,
        rope_inv_freq_len: int = 16,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
        video_attention_mask_mode: str = "first_frame_causal",
        **_: Any,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_heads = self.num_attention_heads
        self.attention_head_dim = int(attention_head_dim)
        self.attn_head_dim = self.attention_head_dim
        self.latents_dim = int(latents_dim)
        self.text_dim = int(text_dim)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.video_attention_mask_mode = str(video_attention_mask_mode)
        patch_dim = self.latents_dim * math.prod(self.patch_size)
        self.video_patch_proj = nn.Linear(patch_dim, hidden_size, bias=True)
        self.condition_proj = nn.Linear(text_dim, hidden_size, bias=True)
        self.token_refiner = H3TokenRefiner(
            token_refiner_num_layers,
            hidden_size,
            num_attention_heads,
            attention_head_dim,
            ffn_hidden_size,
            norm_eps,
            qk_norm_eps,
            final_norm_eps,
        )
        self.time_embedder = H3TimeEmbedder(
            timestep_input_dim, time_embed_hidden_size, time_embed_dim
        )
        self.rope = H3RoPE(rope_inv_freq_len)
        self.blocks = nn.ModuleList(
            [
                H3VideoBlock(
                    hidden_size,
                    num_attention_heads,
                    attention_head_dim,
                    ffn_hidden_size,
                    time_embed_dim,
                    norm_eps,
                    qk_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer = H3FinalLayer(
            hidden_size,
            time_embed_dim,
            latents_dim,
            self.patch_size,
            final_norm_eps,
        )

    def refine_text_condition(
        self,
        qwen_embeddings: torch.Tensor,
        token_tags: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Project native Qwen width and run H3's two-layer TokenRefiner."""

        if qwen_embeddings.ndim != 2 or qwen_embeddings.shape[-1] != self.text_dim:
            raise ValueError(
                f"qwen_embeddings must be [S,{self.text_dim}], got "
                f"{tuple(qwen_embeddings.shape)}"
            )
        if token_tags.shape != qwen_embeddings.shape[:1]:
            raise ValueError("token_tags must match the flattened Qwen sequence")
        tags = token_tags.to(device=qwen_embeddings.device, dtype=torch.long)
        if not torch.logical_or(tags == 0, tags == 1).all():
            raise ValueError("Qwen tags must contain only video 0 or text 1")
        cu_seqlens = cu_seqlens.to(device=qwen_embeddings.device, dtype=torch.int32)
        if (
            cu_seqlens.ndim != 1
            or cu_seqlens.numel() < 2
            or int(cu_seqlens[0]) != 0
            or int(cu_seqlens[-1]) != qwen_embeddings.shape[0]
            or not (cu_seqlens[1:] > cu_seqlens[:-1]).all()
        ):
            raise ValueError("cu_seqlens must delimit every non-empty Qwen sample")
        return self.token_refiner(self.condition_proj(qwen_embeddings), cu_seqlens)

    def patchify(self, latents: torch.Tensor) -> tuple[torch.Tensor, dict[str, int]]:
        batch, channels, frames, height, width = latents.shape
        pt, ph, pw = self.patch_size
        if channels != self.latents_dim:
            raise ValueError(f"Expected {self.latents_dim} latent channels, got {channels}.")
        if frames % pt or height % ph or width % pw:
            raise ValueError(
                f"Latent shape {(frames, height, width)} is not divisible by {self.patch_size}."
            )
        ft, fh, fw = frames // pt, height // ph, width // pw
        value = latents.reshape(batch, channels, ft, pt, fh, ph, fw, pw)
        value = torch.einsum("bctrhpwq->bthwcrpq", value)
        tokens = value.reshape(batch, ft * fh * fw, channels * pt * ph * pw)
        return tokens, {
            "frames": frames,
            "height": height,
            "width": width,
            "ft": ft,
            "fh": fh,
            "fw": fw,
            "tokens_per_frame": fh * fw,
        }

    def unpatchify(self, rows: torch.Tensor, meta: dict[str, int]) -> torch.Tensor:
        batch = rows.shape[0]
        pt, ph, pw = self.patch_size
        value = rows.reshape(
            batch,
            meta["ft"],
            meta["fh"],
            meta["fw"],
            self.latents_dim,
            pt,
            ph,
            pw,
        )
        value = torch.einsum("bthwcrpq->bctrhpwq", value)
        return value.reshape(
            batch,
            self.latents_dim,
            meta["frames"],
            meta["height"],
            meta["width"],
        )

    def position_ids(self, meta: dict[str, int], device: torch.device) -> torch.Tensor:
        t, h, w = torch.meshgrid(
            torch.arange(meta["ft"], device=device),
            torch.arange(meta["fh"], device=device),
            torch.arange(meta["fw"], device=device),
            indexing="ij",
        )
        return torch.stack((t, h, w), dim=-1).reshape(-1, 3)

    def build_video_attention_mask(
        self, seq_len: int, tokens_per_frame: int, device: torch.device
    ) -> torch.Tensor:
        if self.video_attention_mask_mode == "bidirectional":
            return torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
        if self.video_attention_mask_mode == "first_frame_causal":
            mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
            first = min(tokens_per_frame, seq_len)
            mask[:first, first:] = False
            return mask
        if self.video_attention_mask_mode == "per_frame_causal":
            frames = seq_len // tokens_per_frame
            mask = torch.tril(torch.ones((frames, frames), dtype=torch.bool, device=device))
            return mask.repeat_interleave(tokens_per_frame, 0).repeat_interleave(
                tokens_per_frame, 1
            )
        raise ValueError(
            f"Unsupported video_attention_mask_mode={self.video_attention_mask_mode!r}."
        )

    @torch.no_grad()
    def prefill(
        self, latents: torch.Tensor, timestep: torch.Tensor
    ) -> dict[str, Any]:
        patches, meta = self.patchify(latents)
        x = self.video_patch_proj(patches)
        time_embedding = self.time_embedder(timestep, x.dtype)
        rope_freqs = self.rope(self.position_ids(meta, x.device)).to(x.device)
        mask = self.build_video_attention_mask(
            x.shape[1], meta["tokens_per_frame"], x.device
        )
        sdpa_mask = mask[None, None]
        kv_cache: list[dict[str, torch.Tensor]] = []
        for block in self.blocks:
            q, k, v, modulation = block.pre_attention(x, time_embedding, rope_freqs)
            attended = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=sdpa_mask,
            ).transpose(1, 2)
            kv_cache.append({"k": k, "v": v})
            x = block.post_attention(x, attended, modulation)
        logits = self.final_layer(x, time_embedding)
        return {
            "kv_cache": kv_cache,
            "prediction": self.unpatchify(logits, meta),
            "meta": meta,
        }

    @torch.no_grad()
    def cache_zero_timestep_adaln(self) -> None:
        parameter = next(self.parameters())
        timestep = torch.zeros((1,), device=parameter.device, dtype=parameter.dtype)
        embedding = self.time_embedder(timestep, parameter.dtype)
        for block in self.blocks:
            block.cache_zero_timestep_adaln(embedding)


def load_h3_video_backbone(
    transformer_dir: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype = torch.bfloat16,
    video_attention_mask_mode: str = "first_frame_causal",
) -> MiniMaxH3VideoBackbone:
    """Load only H3 parameters required by the visual FastWAM backbone."""
    transformer_dir = Path(transformer_dir)
    config = json.loads((transformer_dir / "config.json").read_text())
    config["video_attention_mask_mode"] = video_attention_mask_mode

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.device("meta"):
            model = MiniMaxH3VideoBackbone(**config)
    finally:
        torch.set_default_dtype(previous_dtype)
    # Materialize on host first. Allocating an empty 65GB GPU model and then
    # replacing every parameter with CPU-backed checkpoint tensors causes
    # severe allocator fragmentation on H20.
    model.to_empty(device="cpu")

    target_keys = set(model.state_dict())
    index = json.loads(
        (transformer_dir / "model.safetensors.index.json").read_text()
    )["weight_map"]
    jobs: dict[str, list[str]] = defaultdict(list)
    for key in target_keys:
        if key not in index:
            raise KeyError(f"H3 checkpoint is missing required key {key!r}.")
        jobs[index[key]].append(key)

    loaded: set[str] = set()
    for shard_name, keys in sorted(jobs.items()):
        with safe_open(transformer_dir / shard_name, framework="pt", device="cpu") as handle:
            shard = {key: handle.get_tensor(key).to(dtype=dtype) for key in keys}
        incompatible = model.load_state_dict(shard, strict=False, assign=True)
        unexpected = set(incompatible.unexpected_keys)
        if unexpected:
            raise RuntimeError(f"Unexpected H3 keys while loading {shard_name}: {unexpected}")
        loaded.update(shard)
        del shard
    missing = target_keys - loaded
    if missing:
        raise RuntimeError(f"Did not load all H3 visual parameters: {sorted(missing)[:10]}")
    model.cache_zero_timestep_adaln()
    return model.to(device=device, dtype=dtype).eval()
