from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    cos = freqs.cos().to(dtype=x.dtype).unsqueeze(0).unsqueeze(2)
    sin = freqs.sin().to(dtype=x.dtype).unsqueeze(0).unsqueeze(2)
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

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
        rope_freqs: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation(time_embedding)
        )
        h = self.norm1(x)
        h = h * (1 + scale_msa[:, None]) + shift_msa[:, None]
        x = x + gate_msa[:, None] * self.attn(h, rope_freqs, attention_mask)
        h = self.norm2(x)
        h = h * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        return x + gate_mlp[:, None] * self.mlp(h)

    def forward_mixed(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
        rope_freqs: torch.Tensor,
        video_k: torch.Tensor,
        video_v: torch.Tensor,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Action queries attend to frozen H3 visual K/V, language, and actions."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation(time_embedding)
        )
        hidden = self.norm1(x)
        hidden = hidden * (1 + scale_msa[:, None]) + shift_msa[:, None]
        q, action_k, action_v = self.attn.project_qkv(hidden, rope_freqs)
        context_k, context_v = self.attn.project_kv(context_tokens)
        keys = torch.cat((video_k, context_k, action_k), dim=1)
        values = torch.cat((video_v, context_v, action_v), dim=1)

        batch, action_len = x.shape[:2]
        prefix_len = video_k.shape[1]
        action_valid = torch.ones(
            (batch, action_len), dtype=torch.bool, device=x.device
        )
        video_valid = torch.ones(
            (batch, prefix_len), dtype=torch.bool, device=x.device
        )
        key_valid = torch.cat(
            (video_valid, context_mask.to(torch.bool), action_valid), dim=1
        )
        attention_mask = key_valid[:, None, None, :].expand(
            -1, 1, action_len, -1
        )
        attended = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            keys.transpose(1, 2),
            values.transpose(1, 2),
            attn_mask=attention_mask,
        )
        attended = attended.transpose(1, 2).reshape(
            batch, action_len, self.attn.inner_dim
        )
        x = x + gate_msa[:, None] * self.attn.out_proj(attended)
        hidden = self.norm2(x)
        hidden = hidden * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        return x + gate_mlp[:, None] * self.mlp(hidden)


class H3ActionDiT(nn.Module):
    """~1B action expert initialized by width interpolation from MiniMax H3."""

    BACKBONE_SKIP_PREFIXES = (
        "action_encoder.",
        "context_encoder.",
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
        "context_dim",
    )

    def __init__(
        self,
        action_dim: int,
        context_dim: int = 4096,
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
        self.action_dim = int(action_dim)
        self.context_dim = int(context_dim)
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
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.time_embedder = nn.Sequential(
            nn.Linear(timestep_input_dim, time_embed_hidden_size),
            nn.SiLU(),
            nn.Linear(time_embed_hidden_size, time_embed_dim),
        )
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

    def action_rope(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        inv_steps = torch.arange(
            self.rope_inv_freq_len, device=device, dtype=torch.float32
        )
        inv_freq = 1.0 / (10000.0 ** (inv_steps / self.rope_inv_freq_len))
        axis = positions[:, None] * inv_freq[None]
        # H3 rotates 96 dims: 16 frequencies for each of t/h/w, duplicated.
        half = torch.cat((axis, torch.zeros_like(axis), torch.zeros_like(axis)), dim=-1)
        return torch.cat((half, half), dim=-1)

    def pre_dit(
        self, action_tokens: torch.Tensor, timestep: torch.Tensor
    ) -> dict[str, Any]:
        if action_tokens.ndim != 3 or action_tokens.shape[-1] != self.action_dim:
            raise ValueError(
                f"action_tokens must be [B,S,{self.action_dim}], got {tuple(action_tokens.shape)}"
            )
        if timestep.ndim != 1 or timestep.shape[0] not in (1, action_tokens.shape[0]):
            raise ValueError("timestep must be [1] or [B].")
        if timestep.shape[0] == 1 and action_tokens.shape[0] > 1:
            timestep = timestep.expand(action_tokens.shape[0])
        tokens = self.action_encoder(action_tokens)
        time_embedding = self.time_embedder(
            self.timestep_embedding(timestep).to(dtype=tokens.dtype)
        )
        return {
            "tokens": tokens,
            "time_embedding": time_embedding,
            "freqs": self.action_rope(tokens.shape[1], tokens.device),
            "meta": {"batch_size": tokens.shape[0], "seq_len": tokens.shape[1]},
        }

    def post_dit(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.action_head(self.final_norm(tokens))

    def forward_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        video_tokens_per_frame: int,
    ) -> torch.Tensor:
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} H3 video cache layers, "
                f"got {len(video_kv_cache)}."
            )
        state = self.pre_dit(action_tokens, timestep)
        x = state["tokens"]
        if context.ndim != 3 or context.shape[-1] != self.context_dim:
            raise ValueError(
                f"context must be [B,L,{self.context_dim}], got {tuple(context.shape)}"
            )
        if context_mask.shape != context.shape[:2]:
            raise ValueError(
                f"context_mask shape {tuple(context_mask.shape)} does not match "
                f"context {tuple(context.shape[:2])}."
            )
        context_tokens = self.context_encoder(context.to(dtype=x.dtype))
        for layer_idx, block in enumerate(self.blocks):
            cache = video_kv_cache[layer_idx]
            # Structured FastWAM mask: actions can see only the current frame.
            video_k = cache["k"][:, :video_tokens_per_frame]
            video_v = cache["v"][:, :video_tokens_per_frame]

            def layer_forward(
                action_hidden: torch.Tensor,
                action_time: torch.Tensor,
                action_rope: torch.Tensor,
                cached_k: torch.Tensor,
                cached_v: torch.Tensor,
                language_tokens: torch.Tensor,
                language_mask: torch.Tensor,
                _block=block,
            ) -> torch.Tensor:
                return _block.forward_mixed(
                    action_hidden,
                    action_time,
                    action_rope,
                    cached_k,
                    cached_v,
                    language_tokens,
                    language_mask,
                )

            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer_forward,
                    x,
                    state["time_embedding"],
                    state["freqs"],
                    video_k,
                    video_v,
                    context_tokens,
                    context_mask,
                    use_reentrant=False,
                )
            else:
                x = layer_forward(
                    x,
                    state["time_embedding"],
                    state["freqs"],
                    video_k,
                    video_v,
                    context_tokens,
                    context_mask,
                )
        return self.post_dit(x)

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state = self.pre_dit(action_tokens, timestep)
        x = state["tokens"]
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    state["time_embedding"],
                    state["freqs"],
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                x = block(
                    x,
                    state["time_embedding"],
                    state["freqs"],
                    attention_mask,
                )
        return self.post_dit(x)
