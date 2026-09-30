import matplotlib.pyplot as plt
import numpy as np

# Data from training_results.txt
epochs = list(range(1, 51))
train_loss = [
    0.3054, 0.1882, 0.0933, 0.0698, 0.0654,
    0.0604, 0.0569, 0.0543, 0.0528, 0.0513,
    0.0500, 0.0487, 0.0479, 0.0471, 0.0462,
    0.0457, 0.0453, 0.0447, 0.0442, 0.0438,
    0.0434, 0.0431, 0.0428, 0.0425, 0.0422,
    0.0418, 0.0416, 0.0414, 0.0413, 0.0409,
    0.0408, 0.0405, 0.0404, 0.0402, 0.0401,
    0.0398, 0.0398, 0.0395, 0.0393, 0.0393,
    0.0392, 0.0390, 0.0389, 0.0388, 0.0389,
    0.0388, 0.0387, 0.0386, 0.0385, 0.0386,
]

test_epochs = [5, 10, 15, 25, 30, 35, 40, 45, 50]
test_loss = [0.0737, 0.0592, 0.0545, 0.0496, 0.0481, 0.0471, 0.0464, 0.0460, 0.0458]

lr = [
    5.00e-04, 4.98e-04, 4.96e-04, 4.92e-04, 4.88e-04,
    4.82e-04, 4.76e-04, 4.69e-04, 4.61e-04, 4.52e-04,
    4.43e-04, 4.32e-04, 4.21e-04, 4.09e-04, 3.97e-04,
    3.84e-04, 3.70e-04, 3.56e-04, 3.42e-04, 3.27e-04,
    3.12e-04, 2.97e-04, 2.81e-04, 2.66e-04, 2.50e-04,
    2.34e-04, 2.19e-04, 2.03e-04, 1.88e-04, 1.73e-04,
    1.58e-04, 1.44e-04, 1.30e-04, 1.16e-04, 1.03e-04,
    9.06e-05, 7.89e-05, 6.78e-05, 5.74e-05, 4.77e-05,
    3.89e-05, 3.09e-05, 2.38e-05, 1.76e-05, 1.22e-05,
    7.85e-06, 4.43e-06, 1.97e-06, 4.93e-07, 0.00e+00,
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Plot 1: Loss Curves ---
ax1 = axes[0]
ax1.plot(epochs, train_loss, label="Train Loss", color="#2196F3", linewidth=2)
ax1.plot(test_epochs, test_loss, 'o-', label="Test Loss", color="#F44336",
         linewidth=2, markersize=6)
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Loss (CFM)", fontsize=12)
ax1.set_title("FlowNP MNIST — Training & Test Loss", fontsize=13, fontweight="bold")
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1, 50)

# --- Plot 2: Learning Rate Schedule ---
ax2 = axes[1]
ax2.plot(epochs, lr, color="#4CAF50", linewidth=2)
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Learning Rate", fontsize=12)
ax2.set_title("Cosine Annealing LR Schedule", fontsize=13, fontweight="bold")
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1, 50)
ax2.ticklabel_format(axis='y', style='scientific', scilimits=(-4, -4))

plt.tight_layout()
plt.savefig("/Users/kellyzhao/project159/flownp_spl/training_results.png", dpi=150)
print("Saved plot to training_results.png")
plt.close()
