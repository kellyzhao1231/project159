"""Dataset and collation for variable-length spatial, GP, and image data.

Each spatial .npz file is expected to contain object arrays
``x_train``/``x_test`` (features) and ``pos_train``/``pos_test``
(coordinates).  Individual samples may have different numbers of observation
points.

GP and image datasets generate tasks on-the-fly using the same dict
interface: ``{x_context, y_context, x_target, y_target}``.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.distributions import MultivariateNormal
from torch.utils.data import Dataset


class SpatialDataset(Dataset):
    """Random context/target splits over irregular spatial observations.

    Args:
        features:  Object array of shape ``(N,)``; each element ``(n_i, y_dim)``.
        positions: Object array of shape ``(N,)``; each element ``(n_i, x_dim)``.
        context_frac_range: ``(lo, hi)`` – uniform range for context fraction.
        deterministic: If *True*, seed the RNG with the sample index (for eval).
    """

    def __init__(
        self,
        features: np.ndarray,
        positions: np.ndarray,
        context_frac_range: tuple[float, float] = (0.1, 0.5),
        deterministic: bool = False,
    ):
        assert len(features) == len(positions)
        self.features = features
        self.positions = positions
        self.context_frac_range = context_frac_range
        self.deterministic = deterministic

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        feat = torch.from_numpy(self.features[idx].astype(np.float32))
        pos = torch.from_numpy(self.positions[idx].astype(np.float32))
        n = pos.shape[0]

        rng = np.random.RandomState(idx) if self.deterministic else np.random

        frac = rng.uniform(*self.context_frac_range)
        n_ctx = max(3, int(n * frac))
        perm = torch.from_numpy(rng.permutation(n))

        return {
            "x_context": pos[perm[:n_ctx]],
            "y_context": feat[perm[:n_ctx]],
            "x_target": pos[perm[n_ctx:]],
            "y_target": feat[perm[n_ctx:]],
        }


def spatial_collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Pad variable-length context/target sets and build boolean masks.

    Returns a dict with keys ``x_context``, ``y_context``, ``x_target``,
    ``y_target`` (zero-padded to the maximum length in the batch) and
    ``context_mask``, ``target_mask`` (``True`` for real points).
    """
    B = len(batch)
    result: dict[str, torch.Tensor] = {}

    for role in ("context", "target"):
        x_key, y_key = f"x_{role}", f"y_{role}"
        x_list = [b[x_key] for b in batch]
        y_list = [b[y_key] for b in batch]
        max_n = max(t.shape[0] for t in x_list)
        x_dim = x_list[0].shape[1]
        y_dim = y_list[0].shape[1]

        x_pad = torch.zeros(B, max_n, x_dim)
        y_pad = torch.zeros(B, max_n, y_dim)
        mask = torch.zeros(B, max_n, dtype=torch.bool)

        for i in range(B):
            ni = x_list[i].shape[0]
            x_pad[i, :ni] = x_list[i]
            y_pad[i, :ni] = y_list[i]
            mask[i, :ni] = True

        result[x_key] = x_pad
        result[y_key] = y_pad
        result[f"{role}_mask"] = mask

    return result


# ── GP kernels ───────────────────────────────────────────────────────


class RBFKernel:
    """Squared-exponential kernel with random length-scale and scale."""

    def __init__(self, sigma_eps=2e-2, max_length=0.6, max_scale=1.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        length = 0.1 + (self.max_length - 0.1) * torch.rand(
            [x.shape[0], 1, 1, 1], device=x.device
        )
        scale = 0.1 + (self.max_scale - 0.1) * torch.rand(
            [x.shape[0], 1, 1], device=x.device
        )
        dist = (x.unsqueeze(-2) - x.unsqueeze(-3)) / length  # (B, N, N, D)
        cov = scale.pow(2) * torch.exp(-0.5 * dist.pow(2).sum(-1)) + (
            self.sigma_eps**2 * torch.eye(x.shape[-2], device=x.device)
        )
        return cov


class Matern52Kernel:
    """Matern-5/2 kernel with random length-scale and scale."""

    def __init__(self, sigma_eps=2e-2, max_length=0.6, max_scale=1.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        length = 0.1 + (self.max_length - 0.1) * torch.rand(
            [x.shape[0], 1, 1, 1], device=x.device
        )
        scale = 0.1 + (self.max_scale - 0.1) * torch.rand(
            [x.shape[0], 1, 1], device=x.device
        )
        dist = torch.norm(
            (x.unsqueeze(-2) - x.unsqueeze(-3)) / length, dim=-1
        )
        cov = (
            scale.pow(2)
            * (1 + math.sqrt(5.0) * dist + 5.0 * dist.pow(2) / 3.0)
            * torch.exp(-math.sqrt(5.0) * dist)
            + self.sigma_eps**2 * torch.eye(x.shape[-2], device=x.device)
        )
        return cov


class PeriodicKernel:
    """Periodic kernel with random period, length-scale, and scale."""

    def __init__(self, sigma_eps=2e-2, max_length=0.6, max_scale=1.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        p = 0.1 + 0.4 * torch.rand([x.shape[0], 1, 1], device=x.device)
        length = 0.1 + (self.max_length - 0.1) * torch.rand(
            [x.shape[0], 1, 1], device=x.device
        )
        scale = 0.1 + (self.max_scale - 0.1) * torch.rand(
            [x.shape[0], 1, 1], device=x.device
        )
        dist = x.unsqueeze(-2) - x.unsqueeze(-3)
        cov = scale.pow(2) * torch.exp(
            -2
            * (torch.sin(math.pi * dist.abs().sum(-1) / p) / length).pow(2)
        ) + self.sigma_eps**2 * torch.eye(x.shape[-2], device=x.device)
        return cov


def _make_kernel(name: str):
    """Return a kernel instance by name."""
    kernels = {
        "rbf": RBFKernel,
        "matern": Matern52Kernel,
        "periodic": PeriodicKernel,
    }
    if name not in kernels:
        raise ValueError(f"Unknown kernel {name!r}; choose from {list(kernels)}")
    return kernels[name]()


# ── GP synthetic dataset ─────────────────────────────────────────────


class GPSyntheticDataset(Dataset):
    """On-the-fly GP regression tasks.

    Each ``__getitem__`` call samples a fresh GP regression problem:
    random x locations, kernel covariance, y from a multivariate normal,
    then splits into context/target.

    Args:
        kernel_name: One of ``"rbf"``, ``"matern"``, ``"periodic"``.
        epoch_size: Virtual dataset length (number of tasks per epoch).
        max_num_points: Maximum total number of points per task.
        x_range: Tuple ``(lo, hi)`` for uniform x sampling.
        context_frac_range: ``(lo, hi)`` for random context fraction.
    """

    def __init__(
        self,
        kernel_name: str = "rbf",
        epoch_size: int = 10000,
        max_num_points: int = 50,
        x_range: tuple[float, float] = (-2.0, 2.0),
        context_frac_range: tuple[float, float] = (0.1, 0.5),
    ):
        self.kernel = _make_kernel(kernel_name)
        self.epoch_size = epoch_size
        self.max_num_points = max_num_points
        self.x_range = x_range
        self.context_frac_range = context_frac_range

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        n = torch.randint(10, self.max_num_points + 1, size=[1]).item()
        x = self.x_range[0] + (self.x_range[1] - self.x_range[0]) * torch.rand(
            1, n, 1
        )  # (1, N, 1)
        cov = self.kernel(x)  # (1, N, N)
        mean = torch.zeros(1, n)
        y = MultivariateNormal(mean, cov).rsample().unsqueeze(-1)  # (1, N, 1)

        # Remove batch dim
        x = x.squeeze(0)  # (N, 1)
        y = y.squeeze(0)  # (N, 1)

        frac = np.random.uniform(*self.context_frac_range)
        n_ctx = max(3, int(n * frac))
        perm = torch.randperm(n)

        return {
            "x_context": x[perm[:n_ctx]],
            "y_context": y[perm[:n_ctx]],
            "x_target": x[perm[n_ctx:]],
            "y_target": y[perm[n_ctx:]],
        }


# ── Image dataset ────────────────────────────────────────────────────


class ImageDataset(Dataset):
    """Wraps a torchvision image dataset as a neural process task.

    Each ``__getitem__`` converts an image to a set of ``(x, y)`` pairs where
    ``x`` are 2D pixel coordinates normalised to ``[-1, 1]`` and ``y`` are
    pixel values shifted to ``[-0.5, 0.5]``.  A random subset is chosen as
    context and the rest as targets.

    Args:
        tv_dataset: A torchvision dataset returning ``(image_tensor, label)``.
            Images should be ``(C, H, W)`` tensors in ``[0, 1]``.
        context_frac_range: ``(lo, hi)`` for random context fraction.
        deterministic: If *True*, seed the RNG with the sample index (for eval).
    """

    def __init__(
        self,
        tv_dataset,
        context_frac_range: tuple[float, float] = (0.01, 0.5),
        deterministic: bool = False,
    ):
        self.tv_dataset = tv_dataset
        self.context_frac_range = context_frac_range
        self.deterministic = deterministic

    def __len__(self) -> int:
        return len(self.tv_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img, _label = self.tv_dataset[idx]  # (C, H, W) in [0, 1]
        C, H, W = img.shape
        num_pixels = H * W

        rng = np.random.RandomState(idx) if self.deterministic else np.random

        # Random permutation of all pixel indices
        perm = torch.from_numpy(rng.permutation(num_pixels))

        # Convert flat indices to 2D coords normalised to [-1, 1]
        rows = perm // W
        cols = perm % W
        x = torch.stack(
            [2.0 * rows.float() / (H - 1) - 1.0, 2.0 * cols.float() / (W - 1) - 1.0],
            dim=-1,
        )  # (num_pixels, 2)

        # Gather pixel values and shift to [-0.5, 0.5]
        flat_img = img.reshape(C, -1)  # (C, H*W)
        y = flat_img[:, perm].T - 0.5  # (num_pixels, C)

        # Context / target split
        frac = rng.uniform(*self.context_frac_range)
        n_ctx = max(3, int(num_pixels * frac))

        return {
            "x_context": x[:n_ctx],
            "y_context": y[:n_ctx],
            "x_target": x[n_ctx:],
            "y_target": y[n_ctx:],
        }
