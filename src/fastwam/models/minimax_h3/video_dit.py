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
import torch.utils.checkpoint
from safetensors import safe_open

from .mixed_attention import AsymmetricAttentionMasks, run_asymmetric_joint_block
from .packed_sequence import (
    action_mm_position_ids,
    build_batch_cu_seqlens,
    build_h3_packed_sample,
    state_mm_position_ids,
)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    left, right = torch.chunk(x, 2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply H3's partial 3D RoPE to [B, S, heads, head_dim]."""
    rot_dim = int(freqs.shape[-1])
    if rot_dim > x.shape[-1]:
        raise ValueError(
            f"RoPE rotates {rot_dim} dimensions but attention head has {x.shape[-1]}"
        )
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    if freqs.ndim == 2:
        cos = freqs.cos().to(dtype=x.dtype)[None, :, None, :]
        sin = freqs.sin().to(dtype=x.dtype)[None, :, None, :]
    elif freqs.ndim == 3 and freqs.shape[:2] == x.shape[:2]:
        cos = freqs.cos().to(dtype=x.dtype)[:, :, None, :]
        sin = freqs.sin().to(dtype=x.dtype)[:, :, None, :]
    else:
        raise ValueError(
            f"freqs must be [S,R] or [B,S,R], got {tuple(freqs.shape)}"
        )
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
        embedding = torch.cat((args.cos(), args.sin()), dim=-1).to(
            self.proj_in.weight.dtype
        )
        hidden = self.proj_in(embedding)
        hidden = F.silu(hidden).to(self.proj_out.weight.dtype)
        return self.proj_out(hidden).to(dtype)


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


class H3LoRABranch(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.dropout_probability = float(dropout)
        self.scaling = float(alpha) / int(rank)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.lora_b = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


class H3LoRALinear(nn.Module):
    """Frozen checkpoint linear plus an independently trainable LoRA branch."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.base = base.requires_grad_(False)
        self.lora = H3LoRABranch(
            base.in_features,
            base.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )

    @property
    def weight(self) -> nn.Parameter:
        return self.base.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora(x)


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

    def forward(self, time_embedding: torch.Tensor) -> tuple[torch.Tensor, ...]:
        values = self.linear(F.silu(time_embedding))
        values = values.view(time_embedding.shape[0] * 3, 6, self.hidden_size)
        return values.unbind(dim=1)

    def select(
        self,
        time_embedding: torch.Tensor,
        combined_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        values = self(time_embedding)
        flat_indices = combined_indices.reshape(-1).to(dtype=torch.long)
        return tuple(
            value.index_select(0, flat_indices).view(*combined_indices.shape, -1)
            for value in values
        )

    def video(self, time_embedding: torch.Tensor) -> tuple[torch.Tensor, ...]:
        values = self(time_embedding)
        video_indices = torch.arange(
            time_embedding.shape[0], device=time_embedding.device, dtype=torch.long
        ) * 3
        return tuple(value.index_select(0, video_indices) for value in values)


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
    def pre_attention(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
        rope_freqs: torch.Tensor,
        combined_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        if combined_indices is None:
            if time_embedding.shape[0] != x.shape[0]:
                raise ValueError(
                    "batch-aligned H3 modulation requires one timestep per sample"
                )
            modulation = tuple(
                value[:, None].expand(-1, x.shape[1], -1)
                for value in self.adaln_proj.video(time_embedding)
            )
        else:
            if combined_indices.shape != x.shape[:2]:
                raise ValueError("combined_indices must match H3 [B,S] rows")
            modulation = self.adaln_proj.select(time_embedding, combined_indices)
        shift_msa, scale_msa, *_ = modulation
        hidden = self.norm1(x)
        hidden = hidden * (1 + scale_msa) + shift_msa
        q, k, v = self.attn.project_qkv(hidden, rope_freqs)
        return q, k, v, modulation

    def post_attention(
        self,
        x: torch.Tensor,
        attention_output: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        _, _, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation
        x = x + gate_msa * self.attn.output(attention_output)
        hidden = self.norm2(x)
        hidden = hidden * (1 + scale_mlp) + shift_mlp
        return x + gate_mlp * self.mlp(hidden)


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

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
        inverse_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift, scale = self.adaln_proj.linear(F.silu(time_embedding)).chunk(2, dim=-1)
        if inverse_indices is not None:
            flat_indices = inverse_indices.reshape(-1).to(dtype=torch.long)
            shift = shift.index_select(0, flat_indices).view(*x.shape[:2], -1)
            scale = scale.index_select(0, flat_indices).view(*x.shape[:2], -1)
        else:
            shift = shift[:, None]
            scale = scale[:, None]
        hidden = self.norm(x)
        hidden = hidden * (1 + scale) + shift
        logits = self.video_out(hidden.to(self.video_out.weight.dtype))
        return logits.to(x.dtype)


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
        video_attention_mask_mode: str = "bidirectional",
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

    def inject_attention_lora(
        self, *, rank: int, alpha: float, dropout: float = 0.0
    ) -> None:
        rank = int(rank)
        if rank <= 0:
            return
        for block in self.blocks:
            if isinstance(block.attn.qkv_proj, H3LoRALinear):
                raise ValueError("H3 attention LoRA has already been injected")
            block.attn.qkv_proj = H3LoRALinear(
                block.attn.qkv_proj,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            block.attn.out_proj = H3LoRALinear(
                block.attn.out_proj,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )

    def lora_branches(self) -> list[H3LoRABranch]:
        return list(self.named_lora_branches().values())

    def named_lora_branches(self) -> dict[str, H3LoRABranch]:
        branches: dict[str, H3LoRABranch] = {}
        for block_index, block in enumerate(self.blocks):
            for projection_name in ("qkv_proj", "out_proj"):
                projection = getattr(block.attn, projection_name)
                if isinstance(projection, H3LoRALinear):
                    branches[
                        f"blocks.{block_index}.attn.{projection_name}"
                    ] = projection.lora
        return branches

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
        projected = self.condition_proj(
            qwen_embeddings.to(self.condition_proj.weight.dtype)
        )
        return self.token_refiner(projected, cu_seqlens)

    def project_video_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """Run the checkpoint's FP32 patch projection, then enter H3 BF16."""

        projected = self.video_patch_proj(
            patches.to(self.video_patch_proj.weight.dtype)
        )
        return projected.to(self.condition_proj.weight.dtype)

    def forward_joint(
        self,
        *,
        action_expert: nn.Module,
        qwen_embeddings: torch.Tensor,
        qwen_tags: torch.Tensor,
        qwen_valid: torch.Tensor,
        clean_keyframe_latents: torch.Tensor,
        noisy_video_latents: torch.Tensor,
        video_timestep: torch.Tensor,
        noisy_action_tokens: torch.Tensor,
        action_timestep: torch.Tensor,
        state_tokens: torch.Tensor,
        action_valid: torch.Tensor,
        keyframe_condition_strength: float,
        video_fps: float,
        action_fps: float,
        video_timestep_scale: float = 1000.0,
        action_timestep_scale: float = 1000.0,
        detach_h3_for_action: bool = False,
        return_debug: bool = False,
    ) -> dict[str, Any]:
        """Run all aligned H3/Action layers for Scheme A."""

        if qwen_embeddings.ndim != 3 or qwen_embeddings.shape[-1] != self.text_dim:
            raise ValueError(
                f"qwen_embeddings must be [B,L,{self.text_dim}], got "
                f"{tuple(qwen_embeddings.shape)}"
            )
        batch_size, max_text_length = qwen_embeddings.shape[:2]
        if qwen_tags.shape != (batch_size, max_text_length):
            raise ValueError("qwen_tags must match qwen_embeddings [B,L]")
        if qwen_valid.shape != (batch_size, max_text_length):
            raise ValueError("qwen_valid must match qwen_embeddings [B,L]")
        qwen_valid = qwen_valid.to(device=qwen_embeddings.device, dtype=torch.bool)
        qwen_tags = qwen_tags.to(device=qwen_embeddings.device, dtype=torch.long)
        text_lengths = qwen_valid.sum(dim=1).tolist()
        if any(length <= 0 for length in text_lengths):
            raise ValueError("every sample must contain at least one Qwen token")
        if any(
            not qwen_valid[index, : int(length)].all()
            or qwen_valid[index, int(length) :].any()
            for index, length in enumerate(text_lengths)
        ):
            raise ValueError("qwen_valid must be a contiguous prefix per sample")

        keyframe_patches, keyframe_meta = self.patchify(clean_keyframe_latents)
        video_patches, video_meta = self.patchify(noisy_video_latents)
        if keyframe_meta["ft"] != 1:
            raise ValueError("Scheme A requires exactly one keyframe latent slice")
        if (keyframe_meta["fh"], keyframe_meta["fw"]) != (
            video_meta["fh"],
            video_meta["fw"],
        ):
            raise ValueError("keyframe and full-video latent grids must align")
        keyframe_rows = keyframe_patches.shape[1]
        video_rows = video_patches.shape[1]

        flat_qwen = qwen_embeddings[qwen_valid]
        flat_tags = qwen_tags[qwen_valid]
        refiner_cu = build_batch_cu_seqlens(
            [int(length) for length in text_lengths]
        ).to(qwen_embeddings.device)
        refined_flat = self.refine_text_condition(flat_qwen, flat_tags, refiner_cu)
        qwen_hidden = qwen_embeddings.new_zeros(
            (batch_size, max_text_length, self.hidden_size)
        )
        qwen_hidden[qwen_valid] = refined_flat
        h3_hidden = torch.cat(
            (
                qwen_hidden,
                self.project_video_patches(keyframe_patches),
                self.project_video_patches(video_patches),
            ),
            dim=1,
        )
        h3_length = h3_hidden.shape[1]
        h3_valid = torch.cat(
            (
                qwen_valid,
                torch.ones(
                    (batch_size, keyframe_rows + video_rows),
                    dtype=torch.bool,
                    device=h3_hidden.device,
                ),
            ),
            dim=1,
        )
        h3_condition = torch.cat(
            (
                qwen_valid,
                torch.ones(
                    (batch_size, keyframe_rows),
                    dtype=torch.bool,
                    device=h3_hidden.device,
                ),
                torch.zeros(
                    (batch_size, video_rows),
                    dtype=torch.bool,
                    device=h3_hidden.device,
                ),
            ),
            dim=1,
        )
        h3_tags = torch.zeros(
            (batch_size, h3_length), dtype=torch.long, device=h3_hidden.device
        )
        h3_tags[:, :max_text_length] = qwen_tags.masked_fill(~qwen_valid, 0)

        h3_position_ids = torch.zeros(
            (batch_size, h3_length, 3),
            dtype=torch.float64,
            device=h3_hidden.device,
        )
        action_position_parts: list[torch.Tensor] = []
        for batch_index, text_length_value in enumerate(text_lengths):
            text_length = int(text_length_value)
            packed = build_h3_packed_sample(
                qwen_tags=qwen_tags[batch_index, :text_length],
                latent_t=video_meta["ft"],
                latent_h=video_meta["height"],
                latent_w=video_meta["width"],
                keyframe_count=1,
            )
            h3_position_ids[batch_index, :text_length] = packed.position_ids[
                :text_length
            ]
            h3_position_ids[
                batch_index,
                max_text_length : max_text_length + keyframe_rows,
            ] = packed.position_ids[
                text_length : text_length + keyframe_rows
            ]
            h3_position_ids[
                batch_index,
                max_text_length + keyframe_rows :,
            ] = packed.position_ids[text_length + keyframe_rows :]
            state_position = state_mm_position_ids(
                batch_size=1,
                text_origin=text_length,
                device=h3_hidden.device,
            )[0]
            action_positions = action_mm_position_ids(
                action_length=noisy_action_tokens.shape[1],
                text_origin=text_length,
                video_fps=video_fps,
                action_fps=action_fps,
                device=h3_hidden.device,
            )
            action_position_parts.append(
                torch.cat((state_position, action_positions), dim=0)
            )
        h3_rope_freqs = torch.stack(
            [self.rope(position_ids) for position_ids in h3_position_ids], dim=0
        )

        if video_timestep.shape != (batch_size,):
            raise ValueError(f"video_timestep must be [{batch_size}]")
        if float(video_timestep_scale) <= 0:
            raise ValueError("video_timestep_scale must be positive")
        target_progress = (
            1.0 - video_timestep.float() / float(video_timestep_scale)
        ).clamp(0.0, 1.0)
        token_progress = target_progress[:, None].expand(-1, h3_length).clone()
        token_progress[:, max_text_length : max_text_length + keyframe_rows] = (
            torch.maximum(
                target_progress,
                torch.full_like(target_progress, float(keyframe_condition_strength)),
            )[:, None]
        )
        unique_progress, inverse_indices = torch.unique(
            token_progress.reshape(-1), sorted=True, return_inverse=True
        )
        inverse_indices = inverse_indices.view(batch_size, h3_length)
        h3_time_embedding = self.time_embedder(
            unique_progress, h3_hidden.dtype
        )
        h3_combined_indices = inverse_indices * 3 + h3_tags

        if action_timestep.shape != (batch_size,):
            raise ValueError(f"action_timestep must be [{batch_size}]")
        if float(action_timestep_scale) <= 0:
            raise ValueError("action_timestep_scale must be positive")
        action_progress = (
            1.0 - action_timestep.float() / float(action_timestep_scale)
        ).clamp(0.0, 1.0)

        action_state = action_expert.pre_dit(
            noisy_action_tokens,
            state_tokens,
            action_progress,
            position_ids=torch.stack(action_position_parts, dim=0),
        )
        action_stream_valid = torch.cat(
            (
                torch.ones(
                    (batch_size, 1), dtype=torch.bool, device=h3_hidden.device
                ),
                action_valid.to(device=h3_hidden.device, dtype=torch.bool),
            ),
            dim=1,
        )
        masks = AsymmetricAttentionMasks(
            h3_valid=h3_valid,
            h3_condition=h3_condition,
            action_valid=action_stream_valid,
        )
        action_hidden = action_state.tokens
        for h3_block, action_block in zip(self.blocks, action_expert.blocks):
            def layer_forward(
                current_h3: torch.Tensor,
                current_action: torch.Tensor,
                _h3_block=h3_block,
                _action_block=action_block,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return run_asymmetric_joint_block(
                    h3_block=_h3_block,
                    action_block=_action_block,
                    h3_hidden=current_h3,
                    action_hidden=current_action,
                    h3_time_embedding=h3_time_embedding,
                    h3_combined_indices=h3_combined_indices,
                    action_time_embedding=action_state.time_embedding,
                    h3_rope_freqs=h3_rope_freqs,
                    action_rope_freqs=action_state.rope_freqs,
                    action_target_mask=action_state.action_mask,
                    masks=masks,
                    detach_h3_for_action=detach_h3_for_action,
                )

            if (
                action_expert.use_gradient_checkpointing
                and action_expert.training
                and torch.is_grad_enabled()
            ):
                h3_hidden, action_hidden = torch.utils.checkpoint.checkpoint(
                    layer_forward,
                    h3_hidden,
                    action_hidden,
                    use_reentrant=False,
                )
            else:
                h3_hidden, action_hidden = layer_forward(
                    h3_hidden, action_hidden
                )

        video_logits = self.final_layer(
            h3_hidden, h3_time_embedding, inverse_indices
        )[:, max_text_length + keyframe_rows :]
        video_prediction = -self.unpatchify(video_logits, video_meta)
        action_prediction = action_expert.post_dit(action_hidden)
        output: dict[str, Any] = {
            "video_prediction": video_prediction,
            "action_prediction": action_prediction,
        }
        if return_debug:
            output["debug"] = {
                "keyframe_rows": keyframe_rows,
                "video_target_rows": video_rows,
                "audio_rows": 0,
                "h3_valid": h3_valid,
                "h3_condition": h3_condition,
                "refiner_cu_seqlens": refiner_cu,
                "action_progress": action_progress,
            }
        return output

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
        x = self.project_video_patches(patches)
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

def load_h3_video_backbone(
    transformer_dir: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype = torch.bfloat16,
    video_attention_mask_mode: str = "bidirectional",
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
            # The official H3 checkpoint intentionally keeps a small set of
            # projections in fp32. Preserve every stored dtype instead of
            # flattening the entire visual expert to the requested AMP dtype.
            shard = {key: handle.get_tensor(key) for key in keys}
        incompatible = model.load_state_dict(shard, strict=False, assign=True)
        unexpected = set(incompatible.unexpected_keys)
        if unexpected:
            raise RuntimeError(f"Unexpected H3 keys while loading {shard_name}: {unexpected}")
        loaded.update(shard)
        del shard
    missing = target_keys - loaded
    if missing:
        raise RuntimeError(f"Did not load all H3 visual parameters: {sorted(missing)[:10]}")
    return model.to(device=device).eval()
