import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors import safe_open

from fastwam.models.minimax_h3 import H3ActionDiT


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_dtype(name: str) -> torch.dtype:
    value = str(name).strip().lower()
    choices = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if value not in choices:
        raise ValueError(f"Unsupported dtype {name!r}; choose from {sorted(choices)}.")
    return choices[value]


def _resize_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).float()
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_tensor(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(src.shape) == target_shape:
        return src
    out = src.float()
    if out.ndim != len(target_shape):
        raise ValueError(
            f"Cannot resize rank-{out.ndim} tensor to rank-{len(target_shape)}: "
            f"{tuple(src.shape)} -> {target_shape}"
        )
    for dim, new_size in enumerate(target_shape):
        if out.shape[dim] == new_size:
            continue
        permutation = [i for i in range(out.ndim) if i != dim] + [dim]
        inverse = [0] * out.ndim
        for i, original_dim in enumerate(permutation):
            inverse[original_dim] = i
        out = _resize_last_dim(
            out.permute(*permutation).contiguous(), new_size
        ).permute(*inverse).contiguous()
    if tuple(out.shape) != target_shape:
        raise RuntimeError(f"Resize failed: got {tuple(out.shape)}, expected {target_shape}.")
    return out


def _source_key(target_key: str) -> str:
    if target_key == "time_embedder.0.weight":
        return "time_embedder.proj_in.weight"
    if target_key == "time_embedder.0.bias":
        return "time_embedder.proj_in.bias"
    if target_key == "time_embedder.2.weight":
        return "time_embedder.proj_out.weight"
    if target_key == "time_embedder.2.bias":
        return "time_embedder.proj_out.bias"
    if ".adaln_proj." in target_key:
        return target_key.replace(".adaln_proj.", ".adaln_proj.linear.")
    return target_key


def _select_video_adaln(
    tensor: torch.Tensor, *, source_hidden_size: int, is_bias: bool
) -> torch.Tensor:
    """Select H3 modality tag 0 (video) from [video, text, audio] AdaLN."""
    if is_bias:
        return tensor.view(3, 6, source_hidden_size)[0].reshape(-1)
    return tensor.view(3, 6, source_hidden_size, tensor.shape[1])[0].reshape(
        6 * source_hidden_size, tensor.shape[1]
    )


def _load_action_config(path: Path) -> dict[str, Any]:
    config = OmegaConf.load(path)
    value = OmegaConf.to_container(config.action_dit_config, resolve=False)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must define action_dit_config as a mapping.")
    unresolved = [key for key, item in value.items() if isinstance(item, str) and "${" in item]
    for key in unresolved:
        if key in {"action_dim", "state_dim"}:
            value[key] = 7 if key == "action_dim" else 8
            print(
                f"[WARN] {key} is unresolved; using {value[key]} for "
                "LIBERO backbone preprocessing."
            )
        elif key == "use_gradient_checkpointing":
            value[key] = False
        else:
            raise ValueError(f"Cannot preprocess unresolved action_dit_config.{key}: {value[key]}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize the width-reduced H3 action expert from MiniMax-H3 FL2VA."
    )
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--h3-transformer-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--apply-alpha-scaling", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    model_config_path = Path(args.model_config)
    transformer_dir = Path(args.h3_transformer_dir)
    output_path = Path(args.output)
    index_path = transformer_dir / "model.safetensors.index.json"
    source_config_path = transformer_dir / "config.json"
    if not index_path.is_file() or not source_config_path.is_file():
        raise FileNotFoundError(
            f"Expected config.json and model.safetensors.index.json under {transformer_dir}."
        )

    action_config = _load_action_config(model_config_path)
    source_config = json.loads(source_config_path.read_text())
    weight_index = json.loads(index_path.read_text())["weight_map"]
    output_dtype = _parse_dtype(args.dtype)

    with torch.device("meta"):
        action_model = H3ActionDiT(**action_config)
    target_state = action_model.state_dict()
    target_keys = sorted(H3ActionDiT.backbone_key_set(target_state))

    jobs_by_shard: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for target_key in target_keys:
        source_key = _source_key(target_key)
        if source_key not in weight_index:
            raise KeyError(f"H3 checkpoint does not contain {source_key!r} for {target_key!r}.")
        jobs_by_shard[weight_index[source_key]].append((target_key, source_key))

    backbone_state: dict[str, torch.Tensor] = {}
    copied = interpolated = 0
    source_hidden_size = int(source_config["hidden_size"])
    for shard_name, jobs in sorted(jobs_by_shard.items()):
        shard_path = transformer_dir / shard_name
        print(f"[INFO] Reading {shard_path.name} ({len(jobs)} tensors)")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for target_key, source_key in jobs:
                src = handle.get_tensor(source_key)
                if ".adaln_proj." in target_key:
                    src = _select_video_adaln(
                        src,
                        source_hidden_size=source_hidden_size,
                        is_bias=target_key.endswith(".bias"),
                    )
                target_shape = tuple(target_state[target_key].shape)
                if tuple(src.shape) == target_shape:
                    value = src
                    copied += 1
                else:
                    value = _resize_tensor(src, target_shape)
                    if (
                        args.apply_alpha_scaling
                        and src.ndim >= 2
                        and src.shape[-1] != target_shape[-1]
                    ):
                        value = value * (float(src.shape[-1]) / target_shape[-1]) ** 0.5
                    interpolated += 1
                backbone_state[target_key] = value.to(output_dtype).contiguous()

    parameter_count = sum(
        tensor.numel() for tensor in target_state.values()
    )
    payload = {
        "policy": {
            "source": str(transformer_dir),
            "source_modality": "video",
            "source_modality_tag": 0,
            "skip_prefixes": list(H3ActionDiT.BACKBONE_SKIP_PREFIXES),
            "interpolation": "sequential_1d_linear_align_corners_true",
            "alpha_scaling": bool(args.apply_alpha_scaling),
        },
        "meta": {
            **{
                key: (
                    float(action_config[key])
                    if key in {"norm_eps", "qk_norm_eps"}
                    else int(action_config[key])
                )
                for key in H3ActionDiT.META_KEYS
            },
            "parameter_count": parameter_count,
        },
        "backbone_state_dict": backbone_state,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    source_digest = hashlib.sha256()
    source_digest.update(source_config_path.read_bytes())
    source_digest.update(index_path.read_bytes())
    manifest = {
        "schema_version": 1,
        "artifact": {
            "filename": output_path.name,
            "size_bytes": output_path.stat().st_size,
            "sha256": _sha256_file(output_path),
        },
        "source_h3": {
            "transformer_dir": str(transformer_dir),
            "config_index_sha256": source_digest.hexdigest(),
        },
        "action_config": action_config,
        "generation": {
            "command": list(sys.argv),
            "dtype": args.dtype,
            "apply_alpha_scaling": bool(args.apply_alpha_scaling),
            "copied_tensors": copied,
            "interpolated_tensors": interpolated,
        },
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"[INFO] Saved {output_path}: parameters={parameter_count / 1e9:.3f}B, "
        f"copied={copied}, interpolated={interpolated}, dtype={output_dtype}; "
        f"manifest={manifest_path}."
    )


if __name__ == "__main__":
    main()
