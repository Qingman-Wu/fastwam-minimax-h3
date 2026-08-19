from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .video_dit import H3RoPE


class H3RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        normalized = x.float() * torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(dtype=dtype) * self.weight


def _rms_norm(size: int, eps: float) -> H3RMSNorm:
    return H3RMSNorm(size, eps)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply H3 RoPE to tensors shaped [B, S, H, Dh]."""
    rot_dim = int(freqs.shape[-1])
    if rot_dim > x.shape[-1]:
        raise ValueError(
            f"RoPE rotates {rot_dim} dimensions but attention head has {x.shape[-1]}"
        )
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    if freqs.ndim == 2:
        cos = freqs.cos().to(dtype=x.dtype).unsqueeze(0).unsqueeze(2)
        sin = freqs.sin().to(dtype=x.dtype).unsqueeze(0).unsqueeze(2)
    elif freqs.ndim == 3 and freqs.shape[:2] == x.shape[:2]:
        cos = freqs.cos().to(dtype=x.dtype).unsqueeze(2)
        sin = freqs.sin().to(dtype=x.dtype).unsqueeze(2)
    else:
        raise ValueError(
            f"freqs must be [S,R] or [B,S,R], got {tuple(freqs.shape)}"
        )
    return torch.cat((x_rot * cos + _rotate_half(x_rot) * sin, x_pass), dim=-1)


class H3ActionAttention(nn.Module):
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
        self.q_norm = _rms_norm(self.head_dim, qk_norm_eps)
        self.k_norm = _rms_norm(self.head_dim, qk_norm_eps)
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

    def project_kv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x).view(
            batch, seq_len, self.num_heads, 3, self.head_dim
        )
        return self.k_norm(qkv[:, :, :, 1]), qkv[:, :, :, 2]

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q, k, v = self.project_qkv(x, rope_freqs)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=x.device, dtype=torch.bool)
            attention_mask = attention_mask.view(1, 1, x.shape[1], x.shape[1])
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.inner_dim)
        return self.out_proj(out)


class H3ActionMLP(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size * 2, bias=False)
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * up)


class H3ActionBlock(nn.Module):
    """Width-reduced H3 block with the original 56x128 attention geometry."""

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
        self.hidden_size = int(hidden_size)
        self.norm1 = _rms_norm(hidden_size, norm_eps)
        self.norm2 = _rms_norm(hidden_size, norm_eps)
        self.attn = H3ActionAttention(
            hidden_size, num_attention_heads, attention_head_dim, qk_norm_eps
        )
        self.mlp = H3ActionMLP(hidden_size, ffn_hidden_size)
        self.adaln_proj = nn.Linear(time_embed_dim, hidden_size * 6)

    def modulation(
        self, time_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        values = self.adaln_proj(F.silu(time_embedding))
        return values.chunk(6, dim=-1)

    @staticmethod
    def modulate_action_rows(
        hidden: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        modulated = hidden * (1 + scale[:, None]) + shift[:, None]
        return torch.where(action_mask.unsqueeze(-1), modulated, hidden)

    @staticmethod
    def gate_action_rows(
        update: torch.Tensor,
        gate: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        gate_per_row = torch.where(
            action_mask.unsqueeze(-1),
            gate[:, None],
            torch.ones_like(gate[:, None]),
        )
        return gate_per_row * update

    def pre_attention(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
        rope_freqs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        modulation = self.modulation(time_embedding)
        shift_msa, scale_msa, *_ = modulation
        hidden = self.modulate_action_rows(
            self.norm1(x), shift_msa, scale_msa, action_mask
        )
        q, k, v = self.attn.project_qkv(hidden, rope_freqs)
        return q, k, v, modulation

    def post_attention(
        self,
        x: torch.Tensor,
        attention_output: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        _, _, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation
        x = x + self.gate_action_rows(
            self.attn.out_proj(attention_output.flatten(2)), gate_msa, action_mask
        )
        hidden = self.modulate_action_rows(
            self.norm2(x), shift_mlp, scale_mlp, action_mask
        )
        return x + self.gate_action_rows(self.mlp(hidden), gate_mlp, action_mask)

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
        rope_freqs: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if action_mask is None:
            action_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        q, k, v, modulation = self.pre_attention(
            x, time_embedding, rope_freqs, action_mask
        )
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=x.device, dtype=torch.bool)
            if attention_mask.ndim == 2:
                attention_mask = attention_mask[None, None]
        attended = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attention_mask,
        ).transpose(1, 2)
        return self.post_attention(x, attended, modulation, action_mask)


@dataclass(frozen=True)
class ActionExpertState:
    tokens: torch.Tensor
    time_embedding: torch.Tensor
    position_ids: torch.Tensor
    rope_freqs: torch.Tensor
    state_mask: torch.Tensor
    action_mask: torch.Tensor
    action_output_indices: torch.Tensor


class H3ActionDiT(nn.Module):
    """~1B action expert initialized by width interpolation from MiniMax H3."""

    BACKBONE_SKIP_PREFIXES = (
        "action_encoder.",
        "state_encoder.",
        "final_norm.",
        "action_head.",
    )
    META_KEYS = (
        "hidden_size",
        "ffn_hidden_size",
        "num_layers",
        "num_attention_heads",
        "attention_head_dim",
        "time_embed_dim",
        "timestep_input_dim",
        "time_embed_hidden_size",
        "rope_inv_freq_len",
        "norm_eps",
        "qk_norm_eps",
        "state_dim",
    )

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        context_dim: int | None = None,
        hidden_size: int = 512,
        ffn_hidden_size: int = 2048,
        num_layers: int = 50,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        timestep_input_dim: int = 256,
        time_embed_hidden_size: int = 512,
        time_embed_dim: int = 512,
        rope_inv_freq_len: int = 16,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        del context_dim
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.hidden_size = int(hidden_size)
        self.ffn_hidden_size = int(ffn_hidden_size)
        self.num_layers = int(num_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_heads = self.num_attention_heads
        self.attention_head_dim = int(attention_head_dim)
        self.attn_head_dim = self.attention_head_dim
        self.timestep_input_dim = int(timestep_input_dim)
        self.time_embed_hidden_size = int(time_embed_hidden_size)
        self.time_embed_dim = int(time_embed_dim)
        self.norm_eps = float(norm_eps)
        self.qk_norm_eps = float(qk_norm_eps)
        self.rope_inv_freq_len = int(rope_inv_freq_len)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        self.action_encoder = nn.Linear(action_dim, hidden_size)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.time_embedder = nn.Sequential(
            nn.Linear(timestep_input_dim, time_embed_hidden_size),
            nn.SiLU(),
            nn.Linear(time_embed_hidden_size, time_embed_dim),
        )
        self.rope = H3RoPE(rope_inv_freq_len)
        self.blocks = nn.ModuleList(
            [
                H3ActionBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    ffn_hidden_size=ffn_hidden_size,
                    time_embed_dim=time_embed_dim,
                    norm_eps=norm_eps,
                    qk_norm_eps=qk_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = _rms_norm(hidden_size, norm_eps)
        self.action_head = nn.Linear(hidden_size, action_dim)

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        return {
            key
            for key in keys
            if not any(key.startswith(prefix) for prefix in cls.BACKBONE_SKIP_PREFIXES)
        }

    @classmethod
    def from_pretrained(
        cls,
        action_dit_config: dict[str, Any],
        pretrained_path: str | Path | None,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        skip_load: bool = False,
    ) -> "H3ActionDiT":
        model = cls(**action_dit_config).to(device=device, dtype=dtype)
        if skip_load or pretrained_path is None:
            return model
        payload = torch.load(Path(pretrained_path), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("backbone_state_dict"), dict
        ):
            raise ValueError(f"Invalid H3 ActionDiT payload: {pretrained_path}")
        meta = payload.get("meta", {})
        for key in cls.META_KEYS:
            expected = getattr(model, key)
            got = meta.get(key)
            if got is None or (
                isinstance(expected, float)
                and abs(float(got) - expected) > 1e-12
            ) or (not isinstance(expected, float) and int(got) != expected):
                raise ValueError(
                    f"H3 ActionDiT meta mismatch for {key}: expected {expected}, got {got}"
                )
        state = model.state_dict()
        expected_keys = cls.backbone_key_set(state)
        loaded = payload["backbone_state_dict"]
        if set(loaded) != expected_keys:
            missing = sorted(expected_keys - set(loaded))
            unexpected = sorted(set(loaded) - expected_keys)
            raise ValueError(
                f"H3 ActionDiT backbone keys mismatch: missing={missing[:5]}, "
                f"unexpected={unexpected[:5]}"
            )
        for key, value in loaded.items():
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(
                    f"H3 ActionDiT shape mismatch for {key}: "
                    f"expected {tuple(state[key].shape)}, got {tuple(value.shape)}"
                )
            state[key] = value.to(device=device, dtype=dtype)
        model.load_state_dict(state, strict=True)
        return model

    def timestep_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.timestep_input_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / half
        )
        args = timestep.float()[:, None] * freqs[None]
        return torch.cat((args.cos(), args.sin()), dim=-1)

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        state_tokens: torch.Tensor | None,
        timestep: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
    ) -> ActionExpertState:
        if action_tokens.ndim != 3 or action_tokens.shape[-1] != self.action_dim:
            raise ValueError(
                f"action_tokens must be [B,S,{self.action_dim}], got {tuple(action_tokens.shape)}"
            )
        if timestep.ndim != 1 or timestep.shape[0] not in (1, action_tokens.shape[0]):
            raise ValueError("timestep must be [1] or [B].")
        if timestep.shape[0] == 1 and action_tokens.shape[0] > 1:
            timestep = timestep.expand(action_tokens.shape[0])
        if state_tokens is None:
            raise ValueError("state_tokens are required for the Action Expert")
        if state_tokens.ndim != 2 or state_tokens.shape != (
            action_tokens.shape[0],
            self.state_dim,
        ):
            raise ValueError(
                f"state_tokens must be [B,{self.state_dim}], got "
                f"{tuple(state_tokens.shape)}"
            )
        action_hidden = self.action_encoder(action_tokens)
        state_hidden = self.state_encoder(
            state_tokens.to(device=action_hidden.device, dtype=action_hidden.dtype)
        ).unsqueeze(1)
        tokens = torch.cat((state_hidden, action_hidden), dim=1)
        time_embedding = self.time_embedder(
            self.timestep_embedding(timestep).to(dtype=tokens.dtype)
        )
        if position_ids is None:
            position_ids = torch.zeros(
                (*tokens.shape[:2], 3),
                dtype=torch.float64,
                device=tokens.device,
            )
            position_ids[:, 1:, 0] = torch.arange(
                action_tokens.shape[1],
                dtype=torch.float64,
                device=tokens.device,
            )
        else:
            if position_ids.shape != (*tokens.shape[:2], 3):
                raise ValueError(
                    f"position_ids must be [B,1+N,3], got {tuple(position_ids.shape)}"
                )
            position_ids = position_ids.to(device=tokens.device, dtype=torch.float64)
        rope_freqs = torch.stack(
            [self.rope(sample_positions) for sample_positions in position_ids], dim=0
        )
        state_mask = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        state_mask[:, 0] = True
        action_mask = ~state_mask
        return ActionExpertState(
            tokens=tokens,
            time_embedding=time_embedding,
            position_ids=position_ids,
            rope_freqs=rope_freqs,
            state_mask=state_mask,
            action_mask=action_mask,
            action_output_indices=torch.arange(
                1, tokens.shape[1], device=tokens.device, dtype=torch.long
            ),
        )

    def post_dit(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1] < 2:
            raise ValueError("Action Expert output must contain state plus action rows")
        return self.action_head(self.final_norm(tokens[:, 1:]))

    def forward(
        self,
        action_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        timestep: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state = self.pre_dit(
            action_tokens,
            state_tokens,
            timestep,
            position_ids=position_ids,
        )
        x = state.tokens
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    state.time_embedding,
                    state.rope_freqs,
                    attention_mask,
                    state.action_mask,
                    use_reentrant=False,
                )
            else:
                x = block(
                    x,
                    state.time_embedding,
                    state.rope_freqs,
                    attention_mask,
                    state.action_mask,
                )
        return self.post_dit(x)
