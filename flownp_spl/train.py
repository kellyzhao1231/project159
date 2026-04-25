"""Train FlowNP on irregular spatial datasets (PeakWeather / GDP).

Usage:
    python train.py --config configs/weather.yaml [--device cuda:0]
    python train.py --config configs/gdp.yaml     [--device cuda:1]
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data import SpatialDataset, spatial_collate
from model import FlowNP


# ── helpers ─────────────────────────────────────────────────────────


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


# ── training / evaluation loops ─────────────────────────────────────


def train_epoch(
    model: FlowNP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    report_freq: int = 50,
) -> float:
    model.train()
    losses: list[float] = []

    for step, batch in enumerate(loader):
        batch = _to_device(batch, device)

        out = model(
            batch["x_context"], batch["y_context"],
            batch["x_target"], batch["y_target"],
            ctx_mask=batch["context_mask"],
            tar_mask=batch["target_mask"],
        )

        optimizer.zero_grad()
        out["loss"].backward()
        optimizer.step()

        losses.append(out["loss"].item())
        if step % report_freq == 0:
            avg = np.mean(losses[-report_freq:])
            print(f"  step {step:5d}  loss={avg:.6f}")

    return float(np.mean(losses))


@torch.no_grad()
def evaluate_loss(
    model: FlowNP,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Fast evaluation: average CFM loss on a held-out set."""
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = _to_device(batch, device)
        out = model(
            batch["x_context"], batch["y_context"],
            batch["x_target"], batch["y_target"],
            ctx_mask=batch["context_mask"],
            tar_mask=batch["target_mask"],
        )
        total += out["loss"].item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate_prediction(
    model: FlowNP,
    loader: DataLoader,
    device: torch.device,
    num_samples: int = 20,
) -> float:
    """Sample-based evaluation: MSE of the predicted mean vs ground truth.

    Slower than :func:`evaluate_loss` because it runs full Euler integration.
    """
    model.eval()
    total_mse, total_pts = 0.0, 0

    for batch in loader:
        batch = _to_device(batch, device)
        mean, _std = model.predict(
            batch["x_context"], batch["y_context"],
            batch["x_target"],
            num_samples=num_samples,
            ctx_mask=batch["context_mask"],
            tar_mask=batch["target_mask"],
        )
        mask = batch["target_mask"].float()
        mse = ((mean - batch["y_target"]).pow(2).mean(dim=-1) * mask).sum()
        total_mse += mse.item()
        total_pts += mask.sum().item()

    return total_mse / max(total_pts, 1)


# ── main ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FlowNP on spatial data")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Override checkpoint directory")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    save_dir = Path(args.save_dir or f"checkpoints/{cfg['dataset_name']}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── data ──────────────────────────────────────────────────────
    print(f"Loading {cfg['data_path']} …")
    data = np.load(cfg["data_path"], allow_pickle=True)

    train_ds = SpatialDataset(
        data["x_train"], data["pos_train"],
        context_frac_range=tuple(cfg["context_frac_range"]),
    )
    test_ds = SpatialDataset(
        data["x_test"], data["pos_test"],
        context_frac_range=(cfg["eval_context_frac"], cfg["eval_context_frac"]),
        deterministic=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], collate_fn=spatial_collate,
        pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], collate_fn=spatial_collate,
        pin_memory=True,
    )
    print(f"Train: {len(train_ds)} samples · Test: {len(test_ds)} samples")

    # ── model ─────────────────────────────────────────────────────
    model = FlowNP(
        dim_x=cfg["x_dim"],
        dim_y=cfg["y_dim"],
        dim_posenc=cfg["dim_posenc"],
        d_model=cfg["d_model"],
        emb_depth=cfg["emb_depth"],
        dim_feedforward=cfg["dim_feedforward"],
        nhead=cfg["nhead"],
        dropout=cfg.get("dropout", 0.0),
        num_layers=cfg["num_layers"],
        timesteps=cfg["timesteps"],
    ).to(device)

    print(f"Model parameters: {model.num_params / 1e6:.2f}M")

    # ── optimiser & scheduler ─────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"],
    )

    # ── training loop ─────────────────────────────────────────────
    best_loss = math.inf

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        elapsed = time.time() - t0

        do_eval = (epoch % cfg["eval_interval"] == 0) or (epoch == cfg["epochs"])

        if do_eval:
            test_loss = evaluate_loss(model, test_loader, device)
            print(
                f"Epoch {epoch:3d}/{cfg['epochs']}  "
                f"train_loss={train_loss:.6f}  test_loss={test_loss:.6f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  time={elapsed:.1f}s"
            )

            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(
                    {"epoch": epoch, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "test_loss": test_loss, "config": cfg},
                    save_dir / "best_model.pt",
                )
                print(f"  ↑ New best test loss: {best_loss:.6f}")
        else:
            print(
                f"Epoch {epoch:3d}/{cfg['epochs']}  "
                f"train_loss={train_loss:.6f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  time={elapsed:.1f}s"
            )

        torch.save(
            {"epoch": epoch, "model": model.state_dict(),
             "optimizer": optimizer.state_dict(), "config": cfg},
            save_dir / "latest.pt",
        )

    # ── final sample-based evaluation ─────────────────────────────
    print("\nRunning sample-based evaluation (this is slower) …")
    pred_mse = evaluate_prediction(
        model, test_loader, device,
        num_samples=cfg.get("eval_num_samples", 20),
    )
    print(f"Final prediction MSE per point: {pred_mse:.6f}")
    print(f"Best  CFM test loss:            {best_loss:.6f}")
    print(f"Checkpoints saved to {save_dir}")


if __name__ == "__main__":
    main()
