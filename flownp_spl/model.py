"""Flow Matching Neural Process (FlowNP).

Clean re-implementation of the method from *Flow Matching Neural Processes*
(NeurIPS 2025), adapted for variable-length irregular spatial data with
proper padding-mask support.

The velocity field is parameterised by a Transformer encoder operating on
Fourier-encoded context and target tokens.  Training uses the conditional
flow-matching (CFM) MSE objective; inference integrates the learned velocity
with an Euler scheme from Gaussian noise.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ── helpers ─────────────────────────────────────────────────────────


def fourier_encode(x: torch.Tensor, n_freq: int) -> torch.Tensor:
    """Sinusoidal positional encoding.

    Args:
        x: ``(..., D)`` input coordinates.
        n_freq: Total features per dimension (must be even).

    Returns:
        ``(..., D * n_freq)`` encoded tensor.
    """
    half = n_freq // 2
    omega = math.pi * (2.0 ** torch.arange(half, dtype=x.dtype, device=x.device).sub(2))
    phase = x.unsqueeze(-1) * omega  # (..., D, half)
    return torch.cat([phase.sin(), phase.cos()], dim=-1).flatten(-2)


def _build_mlp(
    in_dim: int, hid_dim: int, out_dim: int, depth: int,
) -> nn.Sequential:
    """ReLU MLP with ``depth`` linear layers (minimum 2)."""
    assert depth >= 2, "MLP must have at least 2 layers"
    layers: list[nn.Module] = [nn.Linear(in_dim, hid_dim), nn.ReLU(inplace=True)]
    for _ in range(depth - 2):
        layers += [nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True)]
    layers.append(nn.Linear(hid_dim, out_dim))
    return nn.Sequential(*layers)


# ── model ───────────────────────────────────────────────────────────


class FlowNP(nn.Module):
    """Flow Matching Neural Process.

    Args:
        dim_x: Spatial coordinate dimension.
        dim_y: Observation / feature dimension.
        dim_posenc: Fourier features per coordinate dimension (even).
        d_model: Transformer hidden size.
        emb_depth: Depth of the token embedding MLP.
        dim_feedforward: Transformer feed-forward hidden size.
        nhead: Number of attention heads.
        dropout: Dropout rate in the Transformer.
        num_layers: Number of Transformer encoder layers.
        timesteps: Number of Euler steps for sampling / likelihood.
    """

    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        dim_posenc: int = 80,
        d_model: int = 64,
        emb_depth: int = 4,
        dim_feedforward: int = 128,
        nhead: int = 4,
        dropout: float = 0.0,
        num_layers: int = 6,
        timesteps: int = 100,
    ):
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_posenc = dim_posenc
        self.timesteps = timesteps

        # +1 accounts for the scalar flow-time appended to spatial coords
        token_dim = (dim_x + 1) * dim_posenc + dim_y

        self.embedder = _build_mlp(token_dim, d_model, d_model, emb_depth)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.velocity_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, dim_y),
        )

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ── velocity field ──────────────────────────────────────────────

    def _velocity(
        self,
        xc: torch.Tensor,  # (B, Nc, Dx)
        yc: torch.Tensor,  # (B, Nc, Dy)
        xt: torch.Tensor,  # (B, Nt, Dx)
        yt: torch.Tensor,  # (B, Nt, Dy) — noisy targets at flow-time t
        t: torch.Tensor,   # (B, Nt, 1)  — flow-time
        ctx_mask: Optional[torch.Tensor] = None,  # (B, Nc) True=valid
        tar_mask: Optional[torch.Tensor] = None,  # (B, Nt) True=valid
    ) -> torch.Tensor:
        """Predict velocity at target locations conditioned on context."""
        B, Nc = xc.shape[:2]
        Nt = xt.shape[1]

        # Context tokens live at flow-time t=1 (fully observed)
        xc_t = torch.cat([xc, xc.new_ones(B, Nc, 1)], dim=-1)
        xt_t = torch.cat([xt, t], dim=-1)

        xc_enc = fourier_encode(xc_t, self.dim_posenc)
        xt_enc = fourier_encode(xt_t, self.dim_posenc)

        ctx_tok = torch.cat([xc_enc, yc], dim=-1)
        tar_tok = torch.cat([xt_enc, yt], dim=-1)
        tokens = torch.cat([ctx_tok, tar_tok], dim=1)  # (B, Nc+Nt, D_tok)

        pad_mask: Optional[torch.Tensor] = None
        if ctx_mask is not None or tar_mask is not None:
            cm = ctx_mask if ctx_mask is not None else xc.new_ones(B, Nc, dtype=torch.bool)
            tm = tar_mask if tar_mask is not None else xt.new_ones(B, Nt, dtype=torch.bool)
            pad_mask = ~torch.cat([cm, tm], dim=1)  # True ⇒ ignore

        hidden = self.encoder(self.embedder(tokens), src_key_padding_mask=pad_mask)
        return self.velocity_head(hidden[:, Nc:])  # (B, Nt, Dy)

    # ── training ────────────────────────────────────────────────────

    def forward(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_target: torch.Tensor,
        y_target: torch.Tensor,
        ctx_mask: Optional[torch.Tensor] = None,
        tar_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Conditional flow-matching training step.

        Returns:
            Dict with key ``"loss"`` (scalar CFM-MSE).
        """
        B, Nt, Dy = y_target.shape

        y0 = torch.randn_like(y_target)
        t = torch.rand(B, Nt, 1, device=y_target.device)

        yt = t * y_target + (1.0 - t) * y0           # linear interpolation path
        v_pred = self._velocity(
            x_context, y_context, x_target, yt, t,
            ctx_mask=ctx_mask, tar_mask=tar_mask,
        )
        v_true = y_target - y0                         # target velocity field

        sq_err = (v_pred - v_true).pow(2).mean(dim=-1)  # (B, Nt)

        if tar_mask is not None:
            n_valid = tar_mask.float().sum(dim=1).clamp(min=1)
            loss = ((sq_err * tar_mask.float()).sum(dim=1) / n_valid).mean()
        else:
            loss = sq_err.mean()

        return {"loss": loss}

    # ── sampling / prediction ───────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_target: torch.Tensor,
        num_samples: int = 50,
        ctx_mask: Optional[torch.Tensor] = None,
        tar_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate predictions by Euler-integrating the velocity field.

        Uses the stochastic Euler scheme from the FlowNP paper with a slight
        noise injection for diversity.

        Returns:
            ``(mean, std)`` each of shape ``(B, Nt, Dy)`` aggregated over
            *num_samples* independent trajectories.
        """
        B, Nt = x_target.shape[:2]
        Dy = y_context.shape[-1]
        S = num_samples

        xc = x_context.repeat(S, 1, 1)
        yc = y_context.repeat(S, 1, 1)
        xt = x_target.repeat(S, 1, 1)
        cm = ctx_mask.repeat(S, 1) if ctx_mask is not None else None
        tm = tar_mask.repeat(S, 1) if tar_mask is not None else None

        yt = torch.randn(S * B, Nt, Dy, device=x_context.device)
        T = self.timesteps
        dt = 1.0 / T

        for step in range(T):
            s = step / T
            t_vec = yt.new_full((S * B, Nt, 1), s)
            v = self._velocity(xc, yc, xt, yt, t_vec, ctx_mask=cm, tar_mask=tm)

            alpha = 1.0 + s * (1.0 - s)
            sigma = 0.2 * math.sqrt(max(s * (1.0 - s), 0.0))
            yt = yt + (alpha * v + sigma * torch.randn_like(yt)) * dt

        yt = yt.reshape(S, B, Nt, Dy)
        return yt.mean(dim=0), yt.std(dim=0).clamp(min=1e-5)
