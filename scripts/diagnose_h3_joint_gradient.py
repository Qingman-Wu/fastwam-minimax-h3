"""Run pre-training diagnostics for FastWAM-H3 joint-gradient Scheme A."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from fastwam.runtime import (
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
)
from fastwam.utils.config_resolvers import register_default_resolvers


register_default_resolvers()


def _gradient_stats(
    left: Iterable[torch.Tensor | None],
    right: Iterable[torch.Tensor | None],
) -> dict[str, float]:
    left_sq = torch.zeros((), dtype=torch.float64)
    right_sq = torch.zeros((), dtype=torch.float64)
    dot = torch.zeros((), dtype=torch.float64)
    for left_grad, right_grad in zip(left, right):
        if left_grad is None and right_grad is None:
            continue
        if left_grad is None:
            left_grad = torch.zeros_like(right_grad)
        if right_grad is None:
            right_grad = torch.zeros_like(left_grad)
        left_cpu = left_grad.detach().double().cpu()
        right_cpu = right_grad.detach().double().cpu()
        left_sq += left_cpu.square().sum()
        right_sq += right_cpu.square().sum()
        dot += (left_cpu * right_cpu).sum()
    left_norm = math.sqrt(float(left_sq))
    right_norm = math.sqrt(float(right_sq))
    denominator = left_norm * right_norm
    return {
        "video_grad_norm": left_norm,
        "action_grad_norm": right_norm,
        "action_to_video_norm_ratio": (
            right_norm / left_norm if left_norm > 0.0 else float("inf")
        ),
        "cosine": float(dot) / denominator if denominator > 0.0 else float("nan"),
    }


def _beta_sweep(
    stats: dict[str, float],
    betas: Iterable[float],
) -> dict[str, dict[str, float]]:
    video_norm = stats["video_grad_norm"]
    action_norm = stats["action_grad_norm"]
    cosine = stats["cosine"]
    report = {}
    for beta_value in betas:
        beta = float(beta_value)
        combined_sq = (
            video_norm**2
            + (beta * action_norm) ** 2
            + 2.0 * beta * video_norm * action_norm * cosine
        )
        combined_norm = math.sqrt(max(combined_sq, 0.0))
        report[f"{beta:g}"] = {
            "scaled_action_to_video_norm_ratio": (
                beta * action_norm / video_norm
                if video_norm > 0.0
                else float("inf")
            ),
            "combined_to_video_norm_ratio": (
                combined_norm / video_norm
                if video_norm > 0.0
                else float("inf")
            ),
            "combined_video_cosine": (
                (video_norm + beta * action_norm * cosine) / combined_norm
                if combined_norm > 0.0
                else float("nan")
            ),
        }
    return report


def _representation_drift(
    baseline: torch.Tensor,
    updated: torch.Tensor,
) -> dict[str, float]:
    baseline = baseline.double()
    updated = updated.double()
    difference = updated - baseline
    baseline_norm = torch.linalg.vector_norm(baseline)
    updated_norm = torch.linalg.vector_norm(updated)
    denominator = baseline_norm * updated_norm
    return {
        "rms": float(difference.square().mean().sqrt()),
        "relative_l2": float(
            torch.linalg.vector_norm(difference) / baseline_norm.clamp(min=1e-12)
        ),
        "cosine": float(
            (baseline.flatten() @ updated.flatten())
            / denominator.clamp(min=1e-12)
        ),
    }


def _clone_lora_state(named_branches) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {
            key: value.detach().cpu().clone()
            for key, value in branch.state_dict().items()
        }
        for name, branch in named_branches.items()
    }


def _restore_lora_state(named_branches, state) -> None:
    for name, branch in named_branches.items():
        device_state = {
            key: value.to(
                device=next(branch.parameters()).device,
                dtype=next(branch.parameters()).dtype,
            )
            for key, value in state[name].items()
        }
        branch.load_state_dict(device_state, strict=True)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    diagnostic_cfg = cfg.get("diagnostic", {})
    sample_index = int(diagnostic_cfg.get("sample_index", 0))
    block_indices = tuple(
        int(index) for index in diagnostic_cfg.get("block_indices", [0, 24, 49])
    )
    probe_lr = float(diagnostic_cfg.get("probe_lr", cfg.learning_rate))
    probe_action_beta = float(diagnostic_cfg.get("probe_action_beta", 1.0))
    beta_candidates = tuple(
        float(value)
        for value in diagnostic_cfg.get(
            "beta_candidates",
            [1.0, 0.1, 0.01, 0.001],
        )
    )
    num_inference_steps = int(diagnostic_cfg.get("num_inference_steps", 20))
    run_latent_rollout = bool(diagnostic_cfg.get("run_latent_rollout", True))
    seed = int(diagnostic_cfg.get("seed", cfg.seed))
    output_path = Path(
        str(
            diagnostic_cfg.get(
                "output_path",
                "artifacts/h3_joint_gradient_diagnostic.json",
            )
        )
    )

    if not torch.cuda.is_available():
        raise RuntimeError("H3 joint-gradient diagnostics require CUDA")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(precision)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = instantiate(cfg.model, model_dtype=model_dtype, device=str(device))
    dataset = instantiate(cfg.data.train)
    if not 0 <= sample_index < len(dataset):
        raise IndexError(f"sample_index {sample_index} is outside dataset")
    sample = dataset.collate_fn([dataset[sample_index]])
    model.eval()

    # Reset after model/dataset construction so diffusion noise is also fixed.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats(device)

    loss, metrics, diagnostics = model.training_loss(
        sample,
        return_diagnostics=True,
        debug_block_indices=block_indices,
    )
    named_branches = model._named_h3_lora_branches()
    if not named_branches:
        raise RuntimeError("The diagnostic requires trainable H3 LoRA branches")

    block_parameters: dict[int, list[torch.nn.Parameter]] = {}
    for block_index in block_indices:
        prefix = f"blocks.{block_index}."
        parameters = [
            parameter
            for name, branch in named_branches.items()
            if name.startswith(prefix)
            for parameter in branch.parameters()
        ]
        if not parameters:
            raise RuntimeError(f"No H3 LoRA parameters found for block {block_index}")
        block_parameters[block_index] = parameters

    selected_parameters = [
        parameter
        for block_index in block_indices
        for parameter in block_parameters[block_index]
    ]
    video_gradients = torch.autograd.grad(
        diagnostics["loss_video"],
        selected_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    action_gradients = torch.autograd.grad(
        diagnostics["loss_action"],
        selected_parameters,
        retain_graph=True,
        allow_unused=True,
    )

    gradient_report = {}
    offset = 0
    for block_index in block_indices:
        count = len(block_parameters[block_index])
        gradient_report[str(block_index)] = _gradient_stats(
            video_gradients[offset : offset + count],
            action_gradients[offset : offset + count],
        )
        gradient_report[str(block_index)]["beta_sweep"] = _beta_sweep(
            gradient_report[str(block_index)],
            beta_candidates,
        )
        offset += count
    gradient_report["selected_blocks_aggregate"] = _gradient_stats(
        video_gradients,
        action_gradients,
    )
    gradient_report["selected_blocks_aggregate"]["beta_sweep"] = _beta_sweep(
        gradient_report["selected_blocks_aggregate"],
        beta_candidates,
    )

    all_lora_parameters = [
        parameter
        for branch in named_branches.values()
        for parameter in branch.parameters()
    ]
    probe_loss = (
        model.loss_lambda_video * diagnostics["loss_video"]
        + probe_action_beta
        * model.loss_lambda_action
        * diagnostics["loss_action"]
    )
    combined_gradients = torch.autograd.grad(
        probe_loss,
        all_lora_parameters,
        allow_unused=True,
    )
    original_lora_state = _clone_lora_state(named_branches)
    optimizer = torch.optim.AdamW(
        all_lora_parameters,
        lr=probe_lr,
        weight_decay=float(cfg.weight_decay),
        betas=(0.9, 0.95),
    )
    for parameter, gradient in zip(all_lora_parameters, combined_gradients):
        parameter.grad = gradient
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    diagnostic_projections = {}
    for block_index in block_indices:
        block = model.video_expert.blocks[block_index]
        for projection_name in ("qkv_proj", "out_proj"):
            projection = getattr(block.attn, projection_name)
            projection.capture_diagnostics = True
            projection.diagnostic_stats = None
            diagnostic_projections[
                f"blocks.{block_index}.attn.{projection_name}"
            ] = projection

    with torch.no_grad():
        _, updated_metrics, updated_diagnostics = model.training_loss(
            sample,
            base_progress=diagnostics["base_progress"],
            video_noise=diagnostics["video_noise"],
            action_noise=diagnostics["action_noise"],
            keyframe_noise=diagnostics["keyframe_noise"],
            return_diagnostics=True,
            debug_block_indices=block_indices,
        )
    repeated_forward_hidden_difference = {
        str(block_index): _representation_drift(
            diagnostics["h3_hidden_by_block"][block_index],
            updated_diagnostics["h3_hidden_by_block"][block_index],
        )
        for block_index in block_indices
    }
    local_lora_output_report = {}
    for name, projection in diagnostic_projections.items():
        if projection.diagnostic_stats is None:
            raise RuntimeError(f"Missing LoRA output diagnostics for {name}")
        local_lora_output_report[name] = projection.diagnostic_stats
        projection.capture_diagnostics = False
        projection.diagnostic_stats = None
    _restore_lora_state(named_branches, original_lora_state)
    del (
        optimizer,
        combined_gradients,
        video_gradients,
        action_gradients,
        probe_loss,
        loss,
    )
    torch.cuda.empty_cache()

    latent_report = None
    if run_latent_rollout:
        video = sample["video"].to(device=device, dtype=model_dtype)
        input_image = video[0, :, 0].unsqueeze(0)
        proprio = sample["proprio"][0, 0]
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": int(video.shape[2]),
            "action_horizon": int(sample["action"].shape[1]),
            "proprio": proprio,
            "prompt_embeds": sample["prompt_embeds"][0],
            "prompt_token_tags": sample["prompt_token_tags"][0],
            "prompt_attention_mask": sample["prompt_attention_mask"][0],
            "proprio_dim_is_pad": sample.get("proprio_dim_is_pad"),
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "decode_video": False,
            "tiled": False,
        }
        rollout = model.infer(**infer_kwargs)
        predicted_latents = rollout["video_latents"].double()
        target_latents = model._encode_video_latents(video)[0].detach().double().cpu()
        if predicted_latents.shape != target_latents.shape:
            raise ValueError(
                "Rollout/target latent shape mismatch: "
                f"{tuple(predicted_latents.shape)} != {tuple(target_latents.shape)}"
            )
        latent_difference = predicted_latents - target_latents
        latent_report = {
            "num_inference_steps": num_inference_steps,
            "shape": list(predicted_latents.shape),
            "l1": float(latent_difference.abs().mean()),
            "l2": float(latent_difference.square().mean()),
            "relative_l2": float(
                torch.linalg.vector_norm(latent_difference)
                / torch.linalg.vector_norm(target_latents).clamp(min=1e-12)
            ),
            "cosine": float(
                torch.nn.functional.cosine_similarity(
                    predicted_latents.flatten(),
                    target_latents.flatten(),
                    dim=0,
                )
            ),
        }

    report = {
        "schema_version": 1,
        "sample_index": sample_index,
        "seed": seed,
        "stop_action_gradient_to_h3": model.stop_action_gradient_to_h3,
        "probe": {
            "optimizer": "AdamW",
            "lora_only": True,
            "action_beta": probe_action_beta,
            "learning_rate": probe_lr,
            "weight_decay": float(cfg.weight_decay),
            "baseline_loss_metrics": metrics,
            "updated_loss_metrics": updated_metrics,
        },
        "gradient_by_block": gradient_report,
        "representation_probe": {
            "local_lora_output_after_one_probe_step": local_lora_output_report,
            "repeated_forward_hidden_difference": repeated_forward_hidden_difference,
            "warning": (
                "Repeated BF16 H3 forwards have a large nonzero no-op floor; "
                "use local_lora_output_after_one_probe_step as the robust metric."
            ),
        },
        "latent_rollout": latent_report,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote diagnostics to {output_path}")


if __name__ == "__main__":
    main()
