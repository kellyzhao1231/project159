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
    0.0388, 0.0387, 0.0386, 0.0385, 0.0386
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
    7.85e-06, 4.43e-06, 1.97e-06, 4.93e-07, 0.00e+00
]

# Epoch-over-epoch improvement
train_delta = [train_loss[i] - train_loss[i+1] for i in range(len(train_loss)-1)]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('FlowNP MNIST Training Results — All 50 Epochs', fontsize=15, fontweight='bold')

# --- Plot 1: Train Loss per epoch ---
ax1 = axes[0, 0]
ax1.plot(epochs, train_loss, 'b-o', markersize=4, linewidth=1.5, label='Train Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Train Loss (Every Epoch)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# --- Plot 2: Test Loss per evaluation epoch ---
ax2 = axes[0, 1]
ax2.plot(test_epochs, test_loss, 'r-s', markersize=7, linewidth=2, label='Test Loss')
for e, l in zip(test_epochs, test_loss):
    ax2.annotate(f'{l:.4f}', (e, l), textcoords="offset points", xytext=(0, 10),
                 fontsize=8, ha='center', color='red')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Test Loss (Every 5 Epochs)')
ax2.grid(True, alpha=0.3)
ax2.legend()

# --- Plot 3: Train vs Test Loss comparison ---
ax3 = axes[1, 0]
ax3.plot(epochs, train_loss, 'b-o', markersize=3, linewidth=1.5, label='Train Loss', alpha=0.8)
ax3.plot(test_epochs, test_loss, 'r-s', markersize=7, linewidth=2, label='Test Loss')
train_at_test = [train_loss[e-1] for e in test_epochs]
ax3.fill_between(test_epochs, train_at_test, test_loss, alpha=0.15, color='purple', label='Generalization Gap')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.set_title('Train vs Test Loss (Generalization Gap)')
ax3.grid(True, alpha=0.3)
ax3.legend()

# --- Plot 4: Epoch-over-Epoch Improvement ---
ax4 = axes[1, 1]
colors = ['green' if d > 0 else 'red' for d in train_delta]
ax4.bar(epochs[1:], train_delta, color=colors, alpha=0.7, width=0.8)
ax4.axhline(y=0, color='black', linewidth=0.5)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Loss Decrease from Previous Epoch')
ax4.set_title('Per-Epoch Improvement (green=improved, red=regressed)')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/Users/kellyzhao/project159/flownp_spl/training_visualization.png', dpi=150, bbox_inches='tight')
print("Saved: training_visualization.png")

# --- Second figure: detailed epoch-by-epoch comparison ---
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Epoch-by-Epoch Detailed Analysis', fontsize=14, fontweight='bold')

# Heatmap-style bar chart of train loss per epoch
ax5 = axes2[0]
norm_loss = np.array(train_loss)
colors_heat = plt.cm.RdYlGn_r((norm_loss - norm_loss.min()) / (norm_loss.max() - norm_loss.min()))
ax5.bar(epochs, train_loss, color=colors_heat, width=0.9)
ax5.set_xlabel('Epoch')
ax5.set_ylabel('Train Loss')
ax5.set_title('Train Loss per Epoch (color-coded by magnitude)')
ax5.grid(True, alpha=0.3, axis='y')

# Learning rate vs loss (dual axis)
ax6 = axes2[1]
ax6.plot(epochs, train_loss, 'b-o', markersize=3, linewidth=1.5, label='Train Loss')
ax6.set_xlabel('Epoch')
ax6.set_ylabel('Train Loss', color='blue')
ax6.tick_params(axis='y', labelcolor='blue')
ax6_twin = ax6.twinx()
ax6_twin.plot(epochs, lr, 'g--', linewidth=1.5, label='Learning Rate')
ax6_twin.set_ylabel('Learning Rate', color='green')
ax6_twin.tick_params(axis='y', labelcolor='green')
ax6_twin.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
ax6.set_title('Train Loss vs Learning Rate Schedule')
lines1, labels1 = ax6.get_legend_handles_labels()
lines2, labels2 = ax6_twin.get_legend_handles_labels()
ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/kellyzhao/project159/flownp_spl/training_detailed.png', dpi=150, bbox_inches='tight')
print("Saved: training_detailed.png")
plt.show()
