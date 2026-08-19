import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from fastwam.models.minimax_h3.text_encoder import (
    H3TextConditionBatch,
    MiniMaxH3TextConditioner,
    build_fl2va_presentation,
)
from fastwam.models.minimax_h3.video_dit import MiniMaxH3VideoBackbone
from fastwam.models.minimax_h3.video_vae import (
    MiniMaxH3VAEAdapter,
    augment_keyframe_latents,
)


class LiteralTokenizer:
    vocab = {
        "<|vision_start|>": 101,
        "<|vision_end|>": 102,
        "<|image_pad|>": 103,
    }

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(char) for char in text]}

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]


class IdentityProcessor:
    def transform_tensor(self, value):
        return value


class TinyImageProcessor:
    merge_size = 1

    def __call__(self, images, return_tensors):
        assert len(images) == 1
        assert return_tensors == "pt"
        return {
            "pixel_values": torch.zeros(1, 3),
            "image_grid_thw": torch.tensor([[1, 1, 1]]),
        }


class RecordingQwenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(image_token_id=103)
        self.kwargs = None

    def forward(self, **kwargs):
        self.kwargs = kwargs
        sequence_length = kwargs["input_ids"].shape[1]
        hidden_states = tuple(
            torch.full((1, sequence_length, 5120), float(layer))
            for layer in range(65)
        )
        return SimpleNamespace(
            last_hidden_state=hidden_states[-1],
            hidden_states=hidden_states,
        )


class TinyQwenProcessor:
    tokenizer = LiteralTokenizer()
    image_processor = TinyImageProcessor()


class TinyNativeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.processor = IdentityProcessor()

    def encode_images(self, images, transform_input=False):
        assert transform_input is False
        return [torch.ones(24, 1, 2, 2) for _ in images]

    def encode_videos(self, videos, transform_input=False):
        assert transform_input is False
        return [torch.full((24, 2, 2, 2), 2.0) for _ in videos]


def make_tiny_vae_adapter():
    adapter = MiniMaxH3VAEAdapter.__new__(MiniMaxH3VAEAdapter)
    torch.nn.Module.__init__(adapter)
    adapter.z_dim = 24
    adapter.vae = TinyNativeVAE()
    adapter.register_buffer("latents_mean", torch.zeros(1, 24, 1, 1, 1))
    adapter.register_buffer("latents_std", torch.ones(1, 24, 1, 1, 1))
    return adapter


def make_tiny_video_dit():
    return MiniMaxH3VideoBackbone(
        hidden_size=12,
        ffn_hidden_size=16,
        num_layers=0,
        token_refiner_num_layers=2,
        num_attention_heads=3,
        attention_head_dim=4,
        latents_dim=2,
        patch_size=(1, 2, 2),
        text_dim=5,
        timestep_input_dim=4,
        time_embed_hidden_size=8,
        time_embed_dim=6,
        rope_inv_freq_len=1,
    ).eval()


def test_h3_scheme_a_backbone_defaults_to_bidirectional_attention():
    assert make_tiny_video_dit().video_attention_mask_mode == "bidirectional"


def test_keyframe_augmentation_uses_h3_native_ratio():
    clean = torch.full((1, 24, 1, 2, 2), 2.0)
    noise = torch.full_like(clean, -3.0)

    got = augment_keyframe_latents(clean, noise, strength=0.999)

    assert torch.allclose(got, torch.full_like(clean, 1.995))


def test_keyframe_augmentation_rejects_non_h3_strength():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        augment_keyframe_latents(torch.ones(1), torch.ones(1), strength=1.1)


def test_vae_process_image_route_produces_one_temporal_latent():
    adapter = make_tiny_vae_adapter()
    image_as_video = torch.zeros(2, 3, 1, 2, 2)

    image_latent = adapter.encode_video(image_as_video, process_image=True)
    video_latent = adapter.encode_video(
        torch.zeros(2, 3, 5, 2, 2), process_image=False
    )

    assert image_latent.shape == (2, 24, 1, 2, 2)
    assert video_latent.shape == (2, 24, 2, 2, 2)
    assert torch.equal(image_latent, torch.ones_like(image_latent))
    assert torch.equal(video_latent, torch.full_like(video_latent, 2.0))


@pytest.mark.parametrize(
    ("num_frames", "latent_frames"),
    [(5, 2), (22, 7), (39, 12)],
)
def test_h3_native_temporal_latent_length(num_frames, latent_frames):
    assert MiniMaxH3VAEAdapter.latent_temporal_length(num_frames) == latent_frames


@pytest.mark.parametrize("num_frames", [0, 4, 6, 21, 23])
def test_h3_native_temporal_latent_length_rejects_unsupported_frames(num_frames):
    with pytest.raises(ValueError, match=r"5\+17k"):
        MiniMaxH3VAEAdapter.latent_temporal_length(num_frames)


def test_fl2va_presentation_marks_first_frame_vision_separately_from_text():
    input_ids, tags = build_fl2va_presentation(
        LiteralTokenizer(), instruction="go", image_token_count=2
    )

    prefix = [ord(char) for char in "<Picture 1>: "]
    assert input_ids.tolist() == prefix + [101, 103, 103, 102, ord("g"), ord("o")]
    assert tags.tolist() == [1] * len(prefix) + [0, 0, 0, 0] + [1, 1]


def test_qwen_conditioner_returns_h3_native_layer_50_not_final_layer():
    model = RecordingQwenModel()
    conditioner = MiniMaxH3TextConditioner(
        processor=TinyQwenProcessor(),
        model=model,
        device="cpu",
        dtype=torch.float32,
    )

    batch = conditioner.encode([object()], ["go"])

    assert torch.equal(batch.embeddings, torch.full_like(batch.embeddings, 50.0))
    assert model.kwargs["output_hidden_states"] is True


def test_precomputed_qwen_condition_requires_native_hidden_width_and_tags():
    batch = H3TextConditionBatch.from_precomputed(
        embeddings=torch.randn(5, 5120),
        token_tags=torch.tensor([1, 0, 0, 1, 1]),
        cu_seqlens=torch.tensor([0, 3, 5], dtype=torch.int32),
    )

    assert batch.embeddings.shape == (5, 5120)
    assert batch.lengths == (3, 2)

    with pytest.raises(ValueError, match="5120"):
        H3TextConditionBatch.from_precomputed(
            embeddings=torch.randn(5, 4096),
            token_tags=torch.tensor([1, 0, 0, 1, 1]),
            cu_seqlens=torch.tensor([0, 3, 5], dtype=torch.int32),
        )


def test_text_and_vision_tags_survive_condition_projection_and_refiner():
    model = make_tiny_video_dit()
    qwen = torch.randn(3, model.text_dim)
    tags = torch.tensor([1, 0, 1])

    embedded = model.refine_text_condition(
        qwen, tags, torch.tensor([0, 3], dtype=torch.int32)
    )

    assert embedded.shape == (3, model.hidden_size)
    assert torch.equal(tags, torch.tensor([1, 0, 1]))


def test_token_refiner_does_not_mix_samples_across_cu_seqlens():
    torch.manual_seed(7)
    model = make_tiny_video_dit()
    first = torch.randn(2, model.text_dim)
    second = torch.randn(3, model.text_dim)
    tags = torch.ones(5, dtype=torch.long)
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)

    baseline = model.refine_text_condition(torch.cat((first, second)), tags, cu)
    changed = model.refine_text_condition(
        torch.cat((first, second + 100.0)), tags, cu
    )

    assert torch.allclose(baseline[:2], changed[:2], atol=1e-6)
    assert not torch.allclose(baseline[2:], changed[2:])
