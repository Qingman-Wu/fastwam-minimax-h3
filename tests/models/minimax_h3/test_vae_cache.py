import pytest
import torch
import torch.nn as nn

from fastwam.datasets.h3_vae_cache import (
    _tensor_digest,
    initialize_h3_vae_cache,
    load_h3_vae_cache,
    save_h3_vae_cache,
)
from fastwam.models.minimax_h3.video_vae import MiniMaxH3VAEAdapter


def test_h3_vae_cache_round_trip_and_checksum(tmp_path):
    initialize_h3_vae_cache(tmp_path, vae_fingerprint="sha256:vae")
    video = torch.zeros(3, 5, 8, 8)
    keyframe = torch.randn(24, 1, 2, 2)
    mean = torch.randn(24, 2, 2, 2)
    logvar = torch.randn(24, 2, 2, 2)
    path = save_h3_vae_cache(
        tmp_path,
        video=video,
        clean_keyframe_latents=keyframe,
        video_posterior_mean=mean,
        video_posterior_logvar=logvar,
    )

    loaded = load_h3_vae_cache(tmp_path, video=video)

    assert torch.equal(loaded["clean_keyframe_latents"], keyframe)
    assert torch.equal(loaded["video_posterior_mean"], mean)
    assert torch.equal(loaded["video_posterior_logvar"], logvar)

    payload = torch.load(path, weights_only=True)
    payload["video_posterior_mean"][0, 0, 0, 0] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="checksum"):
        load_h3_vae_cache(tmp_path, video=video)


def test_h3_vae_digest_hashes_logical_slice_not_shared_storage():
    backing = torch.randn(8, 24, 2, 2)
    logical_slice = backing[3]

    assert logical_slice.is_contiguous()
    assert logical_slice.untyped_storage().nbytes() > (
        logical_slice.numel() * logical_slice.element_size()
    )
    assert _tensor_digest(logical_slice) == _tensor_digest(logical_slice.clone())

    original_digest = _tensor_digest(logical_slice)
    backing[0].add_(1)
    assert _tensor_digest(logical_slice) == original_digest


def test_normalized_posterior_preserves_raw_space_clamp_and_sampling():
    adapter = MiniMaxH3VAEAdapter.__new__(MiniMaxH3VAEAdapter)
    nn.Module.__init__(adapter)
    adapter.register_buffer(
        "latents_mean",
        torch.tensor([0.25, -0.5]).view(1, 2, 1, 1, 1),
        persistent=False,
    )
    adapter.register_buffer(
        "latents_std",
        torch.tensor([2.0, 0.5]).view(1, 2, 1, 1, 1),
        persistent=False,
    )
    raw_mean = torch.tensor([1.0, -2.0]).view(1, 2, 1, 1, 1)
    raw_unclamped_logvar = torch.tensor([-100.0, 30.0]).view(1, 2, 1, 1, 1)
    raw_logvar = raw_unclamped_logvar.clamp(-30.0, 20.0)

    cached_mean, cached_logvar = adapter._normalize_posterior(
        raw_mean, raw_logvar
    )

    expected_logvar = raw_logvar - 2.0 * torch.log(adapter.latents_std)
    assert torch.equal(cached_logvar, expected_logvar)
    assert cached_logvar.min() < -30.0
    assert cached_logvar.max() > 20.0

    epsilon = torch.Generator(device="cpu").manual_seed(123)
    epsilon = torch.randn(raw_mean.shape, generator=epsilon)
    raw_sample = raw_mean + torch.exp(0.5 * raw_logvar) * epsilon
    expected_normalized = (
        raw_sample - adapter.latents_mean
    ) / adapter.latents_std
    cached_sample = cached_mean + torch.exp(0.5 * cached_logvar) * epsilon
    torch.testing.assert_close(
        cached_sample, expected_normalized, atol=1e-6, rtol=1e-6
    )
