"""Asymmetric attention shared by H3 and the independent Action Expert."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AsymmetricAttentionMasks:
    """Dense validity/region masks for one aligned H3/Action layer."""

    h3_valid: torch.Tensor
    h3_condition: torch.Tensor
    action_valid: torch.Tensor

    def validate(self, h3_length: int, action_length: int) -> None:
        if self.h3_valid.ndim != 2 or self.h3_valid.shape[1] != h3_length:
            raise ValueError(
                f"h3_valid must be [B,{h3_length}], got {tuple(self.h3_valid.shape)}"
            )
        if self.h3_condition.shape != self.h3_valid.shape:
            raise ValueError("h3_condition must have the same shape as h3_valid")
        if self.action_valid.ndim != 2 or self.action_valid.shape != (
            self.h3_valid.shape[0],
            action_length,
        ):
            raise ValueError(
                f"action_valid must be [B,{action_length}], got "
                f"{tuple(self.action_valid.shape)}"
            )
        if not self.action_valid[:, 0].all():
            raise ValueError("the state prefix row must be valid for every sample")
        if (self.h3_condition & ~self.h3_valid).any():
            raise ValueError("h3_condition cannot include invalid H3 rows")


def _validate_qkv(name: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            f"{name} q/k/v must share [B,S,H,D], got "
            f"{tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}"
        )


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_mask: torch.Tensor,
) -> torch.Tensor:
    attended = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=key_mask[:, None, None, :],
    )
    return attended.transpose(1, 2)


def asymmetric_joint_attention(
    *,
    h3_q: torch.Tensor,
    h3_k: torch.Tensor,
    h3_v: torch.Tensor,
    action_q: torch.Tensor,
    action_k: torch.Tensor,
    action_v: torch.Tensor,
    masks: AsymmetricAttentionMasks,
    detach_h3_for_action: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the exact version-one visibility matrix.

    H3 queries read only H3.  The state query reads H3 condition rows and its
    own state key.  Action target queries read every valid H3/state/action key.
    """

    _validate_qkv("H3", h3_q, h3_k, h3_v)
    _validate_qkv("Action", action_q, action_k, action_v)
    if h3_q.shape[0] != action_q.shape[0] or h3_q.shape[2:] != action_q.shape[2:]:
        raise ValueError("H3 and Action q/k/v must share batch/head geometry")
    masks.validate(h3_q.shape[1], action_q.shape[1])

    h3_valid = masks.h3_valid.to(device=h3_q.device, dtype=torch.bool)
    h3_condition = masks.h3_condition.to(device=h3_q.device, dtype=torch.bool)
    action_valid = masks.action_valid.to(device=action_q.device, dtype=torch.bool)

    h3_out = _attention(h3_q, h3_k, h3_v, h3_valid)
    h3_out = h3_out * h3_valid[:, :, None, None]

    action_h3_k = h3_k.detach() if detach_h3_for_action else h3_k
    action_h3_v = h3_v.detach() if detach_h3_for_action else h3_v
    all_k = torch.cat((action_h3_k, action_k), dim=1)
    all_v = torch.cat((action_h3_v, action_v), dim=1)

    state_key_valid = torch.zeros_like(action_valid)
    state_key_valid[:, 0] = True
    state_out = _attention(
        action_q[:, :1],
        all_k,
        all_v,
        torch.cat((h3_condition, state_key_valid), dim=1),
    )

    action_target_out = _attention(
        action_q[:, 1:],
        all_k,
        all_v,
        torch.cat((h3_valid, action_valid), dim=1),
    )
    action_target_out = action_target_out * action_valid[:, 1:, None, None]
    action_out = torch.cat((state_out, action_target_out), dim=1)
    action_out = action_out * action_valid[:, :, None, None]
    return h3_out, action_out


def run_asymmetric_joint_block(
    *,
    h3_block: Any,
    action_block: Any,
    h3_hidden: torch.Tensor,
    action_hidden: torch.Tensor,
    h3_time_embedding: torch.Tensor,
    h3_combined_indices: torch.Tensor,
    action_time_embedding: torch.Tensor,
    h3_rope_freqs: torch.Tensor,
    action_rope_freqs: torch.Tensor,
    action_target_mask: torch.Tensor,
    masks: AsymmetricAttentionMasks,
    detach_h3_for_action: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one aligned H3/Action block with asymmetric visibility."""

    h3_q, h3_k, h3_v, h3_modulation = h3_block.pre_attention(
        h3_hidden,
        h3_time_embedding,
        h3_rope_freqs,
        h3_combined_indices,
    )
    action_q, action_k, action_v, action_modulation = action_block.pre_attention(
        action_hidden,
        action_time_embedding,
        action_rope_freqs,
        action_target_mask,
    )
    h3_attended, action_attended = asymmetric_joint_attention(
        h3_q=h3_q,
        h3_k=h3_k,
        h3_v=h3_v,
        action_q=action_q,
        action_k=action_k,
        action_v=action_v,
        masks=masks,
        detach_h3_for_action=detach_h3_for_action,
    )
    h3_hidden = h3_block.post_attention(
        h3_hidden, h3_attended, h3_modulation
    )
    action_hidden = action_block.post_attention(
        action_hidden,
        action_attended,
        action_modulation,
        action_target_mask,
    )
    return h3_hidden, action_hidden
