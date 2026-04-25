"""Dataset and collation for variable-length spatial NPZ data.

Each .npz file is expected to contain object arrays ``x_train``/``x_test``
(features) and ``pos_train``/``pos_test`` (coordinates).  Individual samples
may have different numbers of observation points.
"""

from __future__ import annotations

import numpy as np
import torch
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
