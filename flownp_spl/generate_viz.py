"""Generate predictions from trained FlowNP and visualize image completions."""

import sys
import torch
import numpy as np
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

sys.path.insert(0, "/Users/kellyzhao/project159/flownp_spl")
from model import FlowNP
from data import ImageDataset, spatial_collate

# ── Load checkpoint ──────────────────────────────────────────────────
ckpt = torch.load(
    "/Users/kellyzhao/project159/flownp_spl/checkpoints/mnist/best_model.pt",
    map_location="cpu",
)
cfg = ckpt["config"]

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
)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"Loaded model (epoch {ckpt['epoch']}, test loss {ckpt['test_loss']:.4f})")

# ── Prepare test data ───────────────────────────────────────────────
tv_test = torchvision.datasets.MNIST(
    "/Users/kellyzhao/project159/flownp_spl/data",
    train=False, download=True, transform=T.ToTensor(),
)

# ── Helper ───────────────────────────────────────────────────────────
def coords_to_img(x, y, mask):
    """Map (x, y) pairs back to a 28x28 image grid."""
    img = np.full((28, 28), np.nan)
    n_valid = mask.sum().item()
    for k in range(n_valid):
        r = int(round((x[k, 0].item() + 1.0) / 2.0 * 27))
        c = int(round((x[k, 1].item() + 1.0) / 2.0 * 27))
        r = np.clip(r, 0, 27)
        c = np.clip(c, 0, 27)
        img[r, c] = y[k, 0].item() + 0.5
    return img


def run_batch(indices, context_frac):
    """Run inference on a batch of images at a given context fraction."""
    ds = ImageDataset(tv_test, context_frac_range=(context_frac, context_frac), deterministic=True)
    samples = [ds[i] for i in indices]
    batch = spatial_collate(samples)
    mean, std = model.predict(
        batch["x_context"], batch["y_context"], batch["x_target"],
        num_samples=10,
        ctx_mask=batch["context_mask"],
        tar_mask=batch["target_mask"],
    )
    return batch, mean, std


# ── Figure 1: Main generation grid (5 images, clean layout) ─────────
print("Generating main visualization...")
indices = [0, 3, 5, 15, 61]
batch, mean, std = run_batch(indices, 0.1)

fig, axes = plt.subplots(len(indices), 4, figsize=(10, 2.5 * len(indices)))
col_titles = ["Ground Truth", "Context (10%)", "Generated Mean", "Uncertainty"]

for j, title in enumerate(col_titles):
    axes[0, j].set_title(title, fontsize=12, fontweight="bold", pad=8)

for i in range(len(indices)):
    ctx_mask = batch["context_mask"][i]
    tar_mask = batch["target_mask"][i]

    gt_img, _ = tv_test[indices[i]]
    gt_np = gt_img.squeeze(0).numpy()

    ctx_img = coords_to_img(batch["x_context"][i], batch["y_context"][i], ctx_mask)

    # Full reconstruction: context + predicted targets
    pred_img = np.full((28, 28), np.nan)
    n_ctx = ctx_mask.sum().item()
    for k in range(n_ctx):
        r = int(round((batch["x_context"][i][k, 0].item() + 1.0) / 2.0 * 27))
        c = int(round((batch["x_context"][i][k, 1].item() + 1.0) / 2.0 * 27))
        pred_img[np.clip(r, 0, 27), np.clip(c, 0, 27)] = batch["y_context"][i][k, 0].item() + 0.5
    n_tar = tar_mask.sum().item()
    for k in range(n_tar):
        r = int(round((batch["x_target"][i][k, 0].item() + 1.0) / 2.0 * 27))
        c = int(round((batch["x_target"][i][k, 1].item() + 1.0) / 2.0 * 27))
        pred_img[np.clip(r, 0, 27), np.clip(c, 0, 27)] = mean[i][k, 0].item() + 0.5

    std_img = coords_to_img(batch["x_target"][i], std[i], tar_mask)

    # Ground truth
    axes[i, 0].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
    # Context: show as colored dots on black background
    ctx_display = np.zeros((28, 28))
    ctx_overlay = np.zeros((28, 28, 4))  # RGBA
    for k in range(n_ctx):
        r = int(round((batch["x_context"][i][k, 0].item() + 1.0) / 2.0 * 27))
        c = int(round((batch["x_context"][i][k, 1].item() + 1.0) / 2.0 * 27))
        r, c = np.clip(r, 0, 27), np.clip(c, 0, 27)
        val = batch["y_context"][i][k, 0].item() + 0.5
        ctx_display[r, c] = val
        ctx_overlay[r, c] = [1, 0.4, 0, 1.0]  # orange dot markers
    axes[i, 1].imshow(np.zeros((28, 28)), cmap="gray", vmin=0, vmax=1)
    axes[i, 1].imshow(ctx_overlay, alpha=0.9)
    axes[i, 1].imshow(ctx_display, cmap="gray", vmin=0, vmax=1, alpha=0.7)

    # Generated mean
    axes[i, 2].imshow(np.nan_to_num(np.clip(pred_img, 0, 1)), cmap="gray", vmin=0, vmax=1)

    # Uncertainty
    std_clean = np.nan_to_num(std_img, nan=0)
    axes[i, 3].imshow(std_clean, cmap="magma", vmin=0, vmax=std_clean.max() + 1e-6)

    for j in range(4):
        axes[i, j].axis("off")

plt.suptitle("FlowNP — MNIST Image Completion", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("/Users/kellyzhao/project159/flownp_spl/generated_samples.png", dpi=180, bbox_inches="tight")
print("Saved generated_samples.png")
plt.close()


# ── Figure 2: Varying context % on a single digit ───────────────────
print("Generating context ablation...")
target_idx = 3  # a '0' digit
fracs = [0.02, 0.05, 0.1, 0.2, 0.4]

fig, axes = plt.subplots(2, len(fracs) + 1, figsize=(3 * (len(fracs) + 1), 6))

# Ground truth in top-left
gt_img, _ = tv_test[target_idx]
gt_np = gt_img.squeeze(0).numpy()
axes[0, 0].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
axes[0, 0].set_title("Ground\nTruth", fontsize=11, fontweight="bold")
axes[0, 0].axis("off")
axes[1, 0].axis("off")

for fi, frac in enumerate(fracs):
    col = fi + 1
    ds = ImageDataset(tv_test, context_frac_range=(frac, frac), deterministic=True)
    sample = ds[target_idx]
    batch = spatial_collate([sample])
    mean_pred, std_pred = model.predict(
        batch["x_context"], batch["y_context"], batch["x_target"],
        num_samples=10,
        ctx_mask=batch["context_mask"],
        tar_mask=batch["target_mask"],
    )
    ctx_mask = batch["context_mask"][0]
    tar_mask = batch["target_mask"][0]

    # Reconstruct full image
    pred_img = np.full((28, 28), np.nan)
    n_ctx = ctx_mask.sum().item()
    for k in range(n_ctx):
        r = int(round((batch["x_context"][0][k, 0].item() + 1.0) / 2.0 * 27))
        c = int(round((batch["x_context"][0][k, 1].item() + 1.0) / 2.0 * 27))
        pred_img[np.clip(r, 0, 27), np.clip(c, 0, 27)] = batch["y_context"][0][k, 0].item() + 0.5
    n_tar = tar_mask.sum().item()
    for k in range(n_tar):
        r = int(round((batch["x_target"][0][k, 0].item() + 1.0) / 2.0 * 27))
        c = int(round((batch["x_target"][0][k, 1].item() + 1.0) / 2.0 * 27))
        pred_img[np.clip(r, 0, 27), np.clip(c, 0, 27)] = mean_pred[0][k, 0].item() + 0.5

    std_img = coords_to_img(batch["x_target"][0], std_pred[0], tar_mask)

    pct = int(frac * 100)
    axes[0, col].imshow(np.nan_to_num(np.clip(pred_img, 0, 1)), cmap="gray", vmin=0, vmax=1)
    axes[0, col].set_title(f"{pct}% context\n({n_ctx} px)", fontsize=11, fontweight="bold")
    axes[0, col].axis("off")

    std_clean = np.nan_to_num(std_img, nan=0)
    axes[1, col].imshow(std_clean, cmap="magma", vmin=0, vmax=0.25)
    axes[1, col].axis("off")

axes[0, 0].set_ylabel("Generated\nMean", fontsize=11, fontweight="bold", rotation=0, labelpad=50, va="center")
axes[1, 0].set_ylabel("Uncertainty", fontsize=11, fontweight="bold", rotation=0, labelpad=50, va="center")

plt.suptitle("Effect of Context Size on Generation Quality", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("/Users/kellyzhao/project159/flownp_spl/context_ablation.png", dpi=180, bbox_inches="tight")
print("Saved context_ablation.png")
plt.close()
