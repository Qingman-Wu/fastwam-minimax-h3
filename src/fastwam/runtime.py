import logging
import os
import inspect
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
import numpy as np
from einops import repeat
from omegaconf import OmegaConf

from .utils.logging_config import get_logger, setup_logging
from .utils.video_io import save_mp4

logger = get_logger(__name__)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    if not isinstance(mixed_precision, str):
        raise ValueError(f"`mixed_precision` must be str, got {type(mixed_precision)}")
    key = mixed_precision.strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def create_wan22_model(
    model_id: str,
    tokenizer_model_id: str,
    dit_config,
    tokenizer_max_len: int = 512,
    train_shift: float = 5.0,
    infer_shift: float = 5.0,
    num_train_timesteps: int = 1000,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.wan22 import Wan22Core

    if isinstance(dit_config, DictConfig):
        dit_config = OmegaConf.to_container(dit_config, resolve=True)
    if not isinstance(dit_config, dict):
        raise ValueError(f"`dit_config` must resolve to a dict, got {type(dit_config)}")

    return Wan22Core.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        redirect_common_files=bool(redirect_common_files),
        dit_config=dit_config,
        train_shift=float(train_shift),
        infer_shift=float(infer_shift),
        num_train_timesteps=int(num_train_timesteps),
    )


def create_fastwam(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.fastwam import FastWAM

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for FastWAM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    return FastWAM.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )


def create_fastwam_h3(
    model_path: str,
    video_dit_config,
    action_dit_config,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    proprio_dim: int | None = None,
    action_scheduler=None,
    video_scheduler=None,
    loss=None,
    load_text_encoder: bool = True,
    text_encoder_device: str | None = None,
    mot_checkpoint_mixed_attn: bool = True,
    keyframe_condition_strength: float = 0.999,
    video_fps: float = 24.0,
    action_fps: float = 8.0,
    freeze_video_expert: bool = True,
    h3_lora_rank: int = 0,
    h3_lora_alpha: float = 32.0,
    h3_lora_dropout: float = 0.0,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Build the H3-native Scheme A video/action flow-matching policy."""
    from .models.minimax_h3.fastwam import FastWAMH3

    def as_dict(value, name: str):
        if isinstance(value, DictConfig):
            value = OmegaConf.to_container(value, resolve=True)
        if not isinstance(value, dict):
            raise ValueError(f"`{name}` must resolve to a dict, got {type(value)}")
        return dict(value)

    video_dit_config = as_dict(video_dit_config, "video_dit_config")
    action_dit_config = as_dict(action_dit_config, "action_dit_config")
    if video_scheduler is None:
        raise ValueError("`video_scheduler` is required for FastWAM-H3 Scheme A")
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for FastWAM-H3 Scheme A")
    video_scheduler = as_dict(video_scheduler, "video_scheduler")
    action_scheduler = as_dict(action_scheduler, "action_scheduler")
    loss = {} if loss is None else as_dict(loss, "loss")
    required = {"train_shift", "infer_shift", "num_train_timesteps"}
    for name, scheduler in (
        ("video_scheduler", video_scheduler),
        ("action_scheduler", action_scheduler),
    ):
        missing = required - set(scheduler)
        if missing:
            raise ValueError(f"`{name}` is missing {sorted(missing)}")

    geometry_keys = ("num_layers", "num_attention_heads", "attention_head_dim")
    geometry_mismatch = {
        key: (video_dit_config.get(key), action_dit_config.get(key))
        for key in geometry_keys
        if video_dit_config.get(key) != action_dit_config.get(key)
    }
    if geometry_mismatch:
        raise ValueError(
            "H3 video/action attention geometry must match: "
            f"{geometry_mismatch}"
        )
    if video_dit_config.get("video_attention_mask_mode", "bidirectional") != "bidirectional":
        raise ValueError(
            "FastWAM-H3 Scheme A requires bidirectional packed H3 attention"
        )
    if proprio_dim is None or int(proprio_dim) <= 0:
        raise ValueError("`proprio_dim` must be a positive state width")
    configured_state_dim = action_dit_config.get("state_dim")
    if configured_state_dim is not None and int(configured_state_dim) != int(proprio_dim):
        raise ValueError(
            "action_dit_config.state_dim must match proprio_dim: "
            f"{configured_state_dim} != {proprio_dim}"
        )
    action_dit_config["state_dim"] = int(proprio_dim)
    action_dit_config["use_gradient_checkpointing"] = bool(
        mot_checkpoint_mixed_attn
    )

    return FastWAMH3.from_pretrained(
        model_path=model_path,
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        load_text_encoder=bool(load_text_encoder),
        text_encoder_device=(
            device if text_encoder_device is None else str(text_encoder_device)
        ),
        device=device,
        torch_dtype=model_dtype,
        video_train_shift=float(video_scheduler["train_shift"]),
        video_infer_shift=float(video_scheduler["infer_shift"]),
        video_num_train_timesteps=int(video_scheduler["num_train_timesteps"]),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        keyframe_condition_strength=float(keyframe_condition_strength),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        video_fps=float(video_fps),
        action_fps=float(action_fps),
        freeze_video_expert=bool(freeze_video_expert),
        h3_lora_rank=int(h3_lora_rank),
        h3_lora_alpha=float(h3_lora_alpha),
        h3_lora_dropout=float(h3_lora_dropout),
    )


def create_fastwam_joint(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.fastwam_joint import FastWAMJoint

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for FastWAM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    return FastWAMJoint.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )


def create_fastwam_idm(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.fastwam_idm import (
        FastWAMIDM,
    )

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for FastWAM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    return FastWAMIDM.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )


def build_datasets(data_cfg: DictConfig):
    from .utils import misc

    train_ds = instantiate(data_cfg.train)
    if data_cfg.get("val") is None:
        val_ds = train_ds
    else:
        train_stats_path = data_cfg.train.get("pretrained_norm_stats")
        default_stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
        val_stats_path = data_cfg.val.get("pretrained_norm_stats")
        pretrained_norm_stats = val_stats_path or train_stats_path or default_stats_path
        logger.info("Building val dataset with pretrained_norm_stats: %s", pretrained_norm_stats)
        val_ds = instantiate(data_cfg.val, pretrained_norm_stats=pretrained_norm_stats)
    return train_ds, val_ds


def _resolve_train_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return "cuda:0"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= device_count:
        return "cuda:0"
    return f"cuda:{local_rank}"


def run_training(cfg: DictConfig):
    from .trainer import Wan22Trainer
    from .utils import misc

    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True,
    )
    misc.register_work_dir(cfg.output_dir)
    config_payload = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(config_payload, f)

    model_device = _resolve_train_device()
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    train_ds, val_ds = build_datasets(cfg.data)

    trainer = Wan22Trainer(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )
    trainer.train()


def _prepare_h3_inference_state(
    inference_cfg: DictConfig,
    *,
    expected_state_dim: int,
) -> dict[str, torch.Tensor | int]:
    """Load the normalized f0 state required by H3 action inference."""

    action_horizon = inference_cfg.get("action_horizon")
    if action_horizon is None or int(action_horizon) <= 0:
        raise ValueError("H3 inference requires a positive `action_horizon`")
    inline_state = inference_cfg.get("proprio")
    state_path = inference_cfg.get("proprio_path")
    if inline_state is not None and state_path is not None:
        raise ValueError("Set only one of `proprio` and `proprio_path`")
    if inline_state is None and state_path is None:
        raise ValueError(
            "H3 inference requires normalized f0 state via `proprio` or "
            "`proprio_path`"
        )
    if state_path is not None:
        path = Path(str(state_path))
        if not path.is_file():
            raise FileNotFoundError(f"H3 proprio_path does not exist: {path}")
        if path.suffix.lower() == ".npy":
            value = torch.from_numpy(np.load(path))
        else:
            value = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(value, dict):
                if "proprio" not in value:
                    raise ValueError(
                        "H3 proprio checkpoint dict must contain a `proprio` tensor"
                    )
                value = value["proprio"]
    else:
        value = OmegaConf.to_container(inline_state, resolve=True)
    proprio = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    expected_state_dim = int(expected_state_dim)
    if proprio.numel() != expected_state_dim:
        raise ValueError(
            f"H3 proprio must contain {expected_state_dim} values, got "
            f"{proprio.numel()}"
        )

    prepared: dict[str, torch.Tensor | int] = {
        "action_horizon": int(action_horizon),
        "proprio": proprio,
    }
    dim_is_pad = inference_cfg.get("proprio_dim_is_pad")
    if dim_is_pad is not None:
        dim_is_pad = torch.as_tensor(
            OmegaConf.to_container(dim_is_pad, resolve=True), dtype=torch.bool
        ).reshape(-1)
        if dim_is_pad.numel() != expected_state_dim:
            raise ValueError(
                "proprio_dim_is_pad must match the H3 state width "
                f"{expected_state_dim}"
            )
        prepared["proprio_dim_is_pad"] = dim_is_pad
    return prepared

def run_inference(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)
    inference_cfg = cfg.inference
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    model = instantiate(cfg.model, model_dtype=model_dtype, device=str(inference_cfg.device))
    checkpoint_path = inference_cfg.get("checkpoint_path")
    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            logger.info("Loading finetuned checkpoint: %s", checkpoint_path)
            model.load_checkpoint(checkpoint_path)
        else:
            logger.warning("Checkpoint not found, skipping load: %s", checkpoint_path)
    model.eval()
    
    def center_crop_resize(img: Image, width: int, height: int) -> Image.Image:
        src_w, src_h = img.size
        scale = max(width / src_w, height / src_h)
        resized = img.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
        rw, rh = resized.size
        left = max((rw - width) // 2, 0)
        top = max((rh - height) // 2, 0)
        return resized.crop((left, top, left + width, top + height))

    input_image = Image.open(str(inference_cfg.input_image_path)).convert("RGB")
    input_image = center_crop_resize(input_image, width=inference_cfg.width, height=inference_cfg.height)
    arr = np.array(input_image, dtype=np.float32)
    x = torch.from_numpy(arr)
    x = x.to(device=model.device, dtype=model.torch_dtype)
    x = x * (2.0 / 255.0) - 1.0
    x = repeat(x, "H W C -> B C H W", B=1)
    output_mp4 = str(inference_cfg.output_mp4)

    infer_kwargs = {
        "prompt": str(inference_cfg.prompt),
        "negative_prompt": str(inference_cfg.get("negative_prompt", "")),
        "text_cfg_scale": float(inference_cfg.get("text_cfg_scale", 1.0)),
        "action_cfg_scale": float(inference_cfg.get("action_cfg_scale", 1.0)),
        "input_image": x,
        "num_frames": int(inference_cfg.num_frames),
        "num_inference_steps": int(inference_cfg.num_inference_steps),
        "sigma_shift": None if inference_cfg.get("sigma_shift") is None else float(inference_cfg.sigma_shift),
        "seed": int(inference_cfg.seed),
        "rand_device": str(inference_cfg.get("rand_device", "cpu")),
        "tiled": bool(inference_cfg.get("tiled", False)),
    }

    is_h3 = bool(
        getattr(model, "inference_accepts_ground_truth_action", True) is False
    )
    if is_h3:
        infer_kwargs.update(
            _prepare_h3_inference_state(
                inference_cfg,
                expected_state_dim=int(model.action_expert.state_dim),
            )
        )
        h3_cache_dir = inference_cfg.get("h3_condition_cache_dir")
        if h3_cache_dir is not None:
            from .datasets.h3_condition_cache import load_h3_condition_cache

            cached = load_h3_condition_cache(
                h3_cache_dir,
                first_frame=x[0],
                instruction=str(inference_cfg.prompt),
            )
            infer_kwargs.update(
                {
                    "prompt_embeds": cached["prompt_embeds"].unsqueeze(0),
                    "prompt_token_tags": cached[
                        "prompt_token_tags"
                    ].unsqueeze(0),
                    "prompt_attention_mask": cached[
                        "prompt_attention_mask"
                    ].unsqueeze(0),
                }
            )
        elif getattr(model, "text_conditioner", None) is None:
            raise ValueError(
                "H3 inference with load_text_encoder=false requires "
                "inference.h3_condition_cache_dir"
            )

    autocast_enabled = (
        model.device.type == "cuda"
        and model_dtype in {torch.float16, torch.bfloat16}
    )
    with torch.autocast(
        device_type=model.device.type,
        dtype=model_dtype,
        enabled=autocast_enabled,
    ):
        infer_out = model.infer(**infer_kwargs)
    video = infer_out["video"]
    output_fps = int(
        inference_cfg.get("fps", getattr(model, "video_fps", 15))
    )
    save_mp4(video, output_mp4, fps=output_fps)
    logger.info("Saved inference video to %s", output_mp4)
    if is_h3:
        output_action = inference_cfg.get("output_action")
        action_path = (
            Path(str(output_action))
            if output_action is not None
            else Path(output_mp4).with_suffix(".action.pt")
        )
        action_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(infer_out["action"], action_path)
        logger.info("Saved primary H3 action prediction to %s", action_path)
        return {
            "video_path": output_mp4,
            "action_path": str(action_path),
            "action": infer_out["action"],
        }
    return output_mp4
