import torch


class WanContinuousFlowMatchScheduler:
    """Continuous-time Flow-Matching scheduler with shift-based sampling."""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0, eps: float = 1e-10):
        if num_train_timesteps <= 0:
            raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}")
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.eps = float(eps)
        self._y_min, self._weight_norm_const = self._precompute_training_weight_stats()

    @staticmethod
    def _phi(u: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * u / (1.0 + (shift - 1.0) * u)

    def _precompute_training_weight_stats(self) -> tuple[float, float]:
        steps = self.num_train_timesteps
        u_grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
        t_grid = self._phi(u_grid, self.shift) * float(steps)
        y_grid = torch.exp(-2.0 * ((t_grid - (steps / 2.0)) / steps) ** 2)
        y_min = float(y_grid.min().item())
        y_shifted_grid = y_grid - y_min
        norm_const = float(y_shifted_grid.mean().item())
        return y_min, norm_const

    def sample_training_t(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}")
        u = torch.rand((batch_size,), device=device, dtype=torch.float32)
        sigma = self._phi(u, self.shift)
        timestep = sigma * float(self.num_train_timesteps)
        return timestep.to(dtype=dtype)

    def timestep_from_progress(
        self, progress: torch.Tensor, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """Map shared base progress to this scheduler's shifted timestep."""

        if not isinstance(progress, torch.Tensor) or progress.ndim != 1:
            raise ValueError("progress must be a one-dimensional tensor")
        if ((progress < 0) | (progress > 1)).any():
            raise ValueError("progress values must be in [0, 1]")
        timestep = self._phi(progress.float(), self.shift) * float(
            self.num_train_timesteps
        )
        return timestep.to(dtype=progress.dtype if dtype is None else dtype)

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        t = timestep.to(dtype=torch.float32)
        steps = float(self.num_train_timesteps)
        y = torch.exp(-2.0 * ((t - (steps / 2.0)) / steps) ** 2)
        y_shifted = y - self._y_min
        weight = y_shifted / (self._weight_norm_const + self.eps)
        if weight.numel() == 1:
            return weight.reshape(())
        return weight

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        sigma = (timestep / float(self.num_train_timesteps)).to(
            original_samples.device, dtype=original_samples.dtype
        )
        if sigma.ndim == 0:
            return (1 - sigma) * original_samples + sigma * noise
        sigma = sigma.view(-1, *([1] * (original_samples.ndim - 1)))
        return (1 - sigma) * original_samples + sigma * noise

    @staticmethod
    def training_target(sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return noise - sample

    def build_inference_schedule(
        self,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        shift_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del dtype
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}")
        shift = self.shift if shift_override is None else float(shift_override)
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")

        u_steps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32)
        sigma_steps = self._phi(u_steps, shift)
        timesteps = sigma_steps[:-1] * float(self.num_train_timesteps)
        deltas = sigma_steps[1:] - sigma_steps[:-1]
        return timesteps, deltas

    @staticmethod
    def step(model_output: torch.Tensor, delta: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        output_dtype = sample.dtype
        compute_dtype = (
            torch.float32
            if sample.dtype in (torch.float16, torch.bfloat16)
            else sample.dtype
        )
        sample_compute = sample.to(dtype=compute_dtype)
        model_output_compute = model_output.to(
            device=sample.device, dtype=compute_dtype
        )
        delta = delta.to(sample.device, dtype=compute_dtype)
        if delta.ndim == 0:
            updated = sample_compute + model_output_compute * delta
        else:
            delta = delta.view(-1, *([1] * (sample.ndim - 1)))
            updated = sample_compute + model_output_compute * delta
        return updated.to(dtype=output_dtype)

    @staticmethod
    def step_h3_video(
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        delta: torch.Tensor,
        sample: torch.Tensor,
        *,
        timestep_scale: float = 1000.0,
    ) -> torch.Tensor:
        """Match MiniMax-H3's released FP32 x_t/x0 Euler update exactly."""

        output_dtype = sample.dtype
        compute_dtype = (
            torch.float32
            if sample.dtype in (torch.float16, torch.bfloat16)
            else sample.dtype
        )
        sigma = timestep.to(sample.device, dtype=compute_dtype) / float(
            timestep_scale
        )
        sigma_next = sigma + delta.to(sample.device, dtype=compute_dtype)
        progress = 1.0 - sigma
        sigma_from_progress = 1.0 - progress.to(dtype=output_dtype)
        while sigma_from_progress.ndim < sample.ndim:
            sigma_from_progress = sigma_from_progress.unsqueeze(-1)
        denoised = sample - sigma_from_progress * model_output.to(
            device=sample.device
        )
        ratio = sigma_next / sigma
        while ratio.ndim < sample.ndim:
            ratio = ratio.unsqueeze(-1)
        updated = ratio * sample.to(dtype=compute_dtype) + (
            1.0 - ratio
        ) * denoised.to(dtype=compute_dtype)
        return updated.to(dtype=output_dtype)
