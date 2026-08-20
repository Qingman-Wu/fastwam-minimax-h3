"""H3-native Qwen3-VL first-frame and instruction conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn


VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
H3_QWEN_HIDDEN_SIZE = 5120
H3_QWEN_HIDDEN_LAYER = 50
H3_QWEN_ENCODER_SIGNATURE = (
    "qwen3-vl-4.57.6:layers-0-49:final-norm-identity:last-hidden-state"
)
H3_VIDEO_TAG = 0
H3_TEXT_TAG = 1


@dataclass(frozen=True)
class H3TextConditionBatch:
    """Flattened Qwen rows and their per-sample boundaries."""

    embeddings: torch.Tensor
    token_tags: torch.Tensor
    cu_seqlens: torch.Tensor

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(
            int(value)
            for value in (self.cu_seqlens[1:] - self.cu_seqlens[:-1]).tolist()
        )

    @classmethod
    def from_precomputed(
        cls,
        *,
        embeddings: torch.Tensor,
        token_tags: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> "H3TextConditionBatch":
        if embeddings.ndim != 2 or embeddings.shape[-1] != H3_QWEN_HIDDEN_SIZE:
            raise ValueError(
                f"H3 Qwen embeddings must be [S,{H3_QWEN_HIDDEN_SIZE}], got "
                f"{tuple(embeddings.shape)}"
            )
        if token_tags.shape != embeddings.shape[:1]:
            raise ValueError(
                f"token_tags must have shape [{embeddings.shape[0]}], got "
                f"{tuple(token_tags.shape)}"
            )
        token_tags = token_tags.to(device=embeddings.device, dtype=torch.long)
        if not torch.logical_or(
            token_tags == H3_VIDEO_TAG, token_tags == H3_TEXT_TAG
        ).all():
            raise ValueError("H3 Qwen token_tags must contain only 0 or 1")
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must contain at least [0, sequence_length]")
        cu_seqlens = cu_seqlens.to(device=embeddings.device, dtype=torch.int32)
        if int(cu_seqlens[0]) != 0 or int(cu_seqlens[-1]) != embeddings.shape[0]:
            raise ValueError(
                "cu_seqlens must start at 0 and end at the flattened sequence length"
            )
        if not (cu_seqlens[1:] > cu_seqlens[:-1]).all():
            raise ValueError("every Qwen sample must contain at least one token")
        return cls(
            embeddings=embeddings,
            token_tags=token_tags,
            cu_seqlens=cu_seqlens,
        )


def _text_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def build_fl2va_presentation(
    tokenizer: Any,
    *,
    instruction: str,
    image_token_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build H3's first-frame FL2VA Qwen presentation and modality tags."""

    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    image_token_count = int(image_token_count)
    if image_token_count <= 0:
        raise ValueError(
            f"image_token_count must be positive, got {image_token_count}"
        )
    prefix = _text_ids(tokenizer, "<Picture 1>: ")
    vision = [
        tokenizer.convert_tokens_to_ids(VISION_START),
        *[tokenizer.convert_tokens_to_ids(IMAGE_PAD)] * image_token_count,
        tokenizer.convert_tokens_to_ids(VISION_END),
    ]
    suffix = _text_ids(tokenizer, instruction)
    input_ids = torch.tensor(prefix + vision + suffix, dtype=torch.long)
    token_tags = torch.tensor(
        [H3_TEXT_TAG] * len(prefix)
        + [H3_VIDEO_TAG] * len(vision)
        + [H3_TEXT_TAG] * len(suffix),
        dtype=torch.long,
    )
    return input_ids, token_tags


class MiniMaxH3TextConditioner(nn.Module):
    """Thin loader around the Qwen3-VL model released with H3 FL2VA."""

    def __init__(
        self,
        *,
        processor: Any,
        model: nn.Module,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.processor = processor
        self.model = model
        self.device = torch.device(device)
        self.dtype = dtype
        self.model.requires_grad_(False).eval()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> "MiniMaxH3TextConditioner":
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as error:
            raise ImportError(
                "MiniMax-H3 Qwen conditioning requires transformers==4.57.6"
            ) from error

        model_path = Path(model_path)
        processor = AutoProcessor.from_pretrained(
            str(model_path / "processor"), trust_remote_code=True
        )
        causal_model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path / "text_encoder"),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model = causal_model.model
        language_layers = model.language_model.layers
        if len(language_layers) < H3_QWEN_HIDDEN_LAYER:
            raise ValueError(
                f"H3 Qwen checkpoint has only {len(language_layers)} language "
                f"layers; layer {H3_QWEN_HIDDEN_LAYER} is required"
            )
        if len(language_layers) > H3_QWEN_HIDDEN_LAYER:
            # H3 consumes the output after language layer 49. Layers 50+ can
            # be released immediately, which materially reduces the offline
            # cache job's resident memory without changing its output.
            model.language_model.layers = nn.ModuleList(
                list(language_layers[:H3_QWEN_HIDDEN_LAYER])
            )
        # H3 consumes decoder layer 49's output before Qwen's final RMSNorm.
        # On transformers 4.57.6 both last_hidden_state and hidden_states[-1]
        # are post-norm, so truncation alone is insufficient.
        model.language_model.norm = nn.Identity()
        del causal_model
        return cls(processor=processor, model=model.to(device), device=device, dtype=dtype)

    @torch.inference_mode()
    def encode(
        self,
        images: Sequence[Any],
        instructions: Sequence[str],
    ) -> H3TextConditionBatch:
        if len(images) != len(instructions) or not images:
            raise ValueError("images and instructions must have the same non-zero length")

        all_embeddings: list[torch.Tensor] = []
        all_tags: list[torch.Tensor] = []
        lengths: list[int] = []
        tokenizer = self.processor.tokenizer
        image_processor = self.processor.image_processor
        image_token_id = int(self.model.config.image_token_id)

        for image, instruction in zip(images, instructions):
            vision = image_processor(images=[image], return_tensors="pt")
            grid = vision["image_grid_thw"]
            merge = int(image_processor.merge_size) ** 2
            image_token_count = int(grid[0].prod().item()) // merge
            input_ids, tags = build_fl2va_presentation(
                tokenizer,
                instruction=instruction,
                image_token_count=image_token_count,
            )
            input_ids = input_ids.unsqueeze(0).to(self.device)
            mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
            mm_token_type_ids[input_ids == image_token_id] = 1
            output = self.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                pixel_values=vision["pixel_values"].to(
                    device=self.device, dtype=self.dtype
                ),
                image_grid_thw=grid.to(self.device),
                mm_token_type_ids=mm_token_type_ids,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
            hidden = output.last_hidden_state[0]
            if hidden.shape[-1] != H3_QWEN_HIDDEN_SIZE:
                raise ValueError(
                    f"H3 Qwen returned hidden width {hidden.shape[-1]}, "
                    f"expected {H3_QWEN_HIDDEN_SIZE}"
                )
            all_embeddings.append(hidden)
            all_tags.append(tags.to(hidden.device))
            lengths.append(hidden.shape[0])

        cu_seqlens = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device=all_embeddings[0].device,
        )
        return H3TextConditionBatch.from_precomputed(
            embeddings=torch.cat(all_embeddings, dim=0),
            token_tags=torch.cat(all_tags, dim=0),
            cu_seqlens=cu_seqlens,
        )

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self
