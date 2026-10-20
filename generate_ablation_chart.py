"""
Generate publication-quality Ablation Study charts for ASG-Net.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = r"d:\Dl_project\results"
ARTIFACT_DIR = r"C:\Users\Md Dilnawaz Ahmad\.gemini\antigravity\brain\d3524f49-ce18-47cc-8799-ae2e6eef7175"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

plt.style.use("default")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)

# Data
variants = [
    "V1: RGB Only\n(Baseline)",
    "V2: Early Fusion\n(4-Channel)",
    "V3: Dual Add\n(Element-wise)",
    "V4: Dual ASG\n(No CM-FRM)",
    "V5: Full ASG-Net\n(Complete)"
]

miou = [83.61, 84.18, 85.52, 87.15, 88.42]
oa = [90.12, 90.35, 90.84, 91.31, 91.86]
f1 = [90.25, 90.71, 91.56, 92.74, 93.68]
deltas = [0.0, 0.57, 1.91, 3.54, 4.81]

x = np.arange(len(variants))
width = 0.26

# Subplot 1: Grouped Bar Chart (mIoU, OA, F1)
bars1 = ax1.bar(x - width, miou, width, label="mIoU (%)", color="#2563EB", edgecolor="none", zorder=3)
bars2 = ax1.bar(x, oa, width, label="Overall Accuracy (%)", color="#10B981", edgecolor="none", zorder=3)
bars3 = ax1.bar(x + width, f1, width, label="Macro F1-Score (%)", color="#8B5CF6", edgecolor="none", zorder=3)

# Annotate mIoU values on top of bars
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.35, f"{yval:.1f}%",
             ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1E3A8A")

ax1.set_ylabel("Metric Score (%)", fontsize=12, fontweight="bold")
ax1.set_title("Ablation Variant Performance Progression", fontsize=14, fontweight="bold", pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(variants, fontsize=10, fontweight="semibold")
ax1.set_ylim(78, 96)
ax1.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
ax1.legend(loc="upper left", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=10)

# Subplot 2: Cumulative Gain Delta (Δ mIoU over baseline)
colors_delta = ["#94A3B8", "#F97316", "#FBBF24", "#3B82F6", "#10B981"]
bars_delta = ax2.bar(x, deltas, width=0.55, color=colors_delta, edgecolor="#334155", linewidth=1.2, zorder=3)

for bar, d in zip(bars_delta, deltas):
    yval = bar.get_height()
    label = "Base" if d == 0 else f"+{d:.2f}%"
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.12, label,
             ha="center", va="bottom", fontsize=11, fontweight="bold", color="#0F172A")

ax2.set_ylabel("Cumulative Gain Δ mIoU (%)", fontsize=12, fontweight="bold")
ax2.set_title("Step-by-Step Innovation Contribution (Δ mIoU)", fontsize=14, fontweight="bold", pad=12)
ax2.set_xticks(x)
ax2.set_xticklabels(variants, fontsize=10, fontweight="semibold")
ax2.set_ylim(0, 5.8)
ax2.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

# Add annotations explaining why
ax2.annotate("Early Fusion: Minor gain (+0.57%)\nModality interference in stem",
             xy=(1, 0.57), xytext=(0.4, 2.2),
             arrowprops=dict(arrowstyle="->", color="#EA580C", lw=1.5),
             fontsize=9, fontweight="bold", color="#C2410C",
             bbox=dict(boxstyle="round,pad=0.3", fc="#FFF7ED", ec="#FDBA74"))

ax2.annotate("Dual ASG Gating:\nMajor leap (+1.63% over Add)",
             xy=(3, 3.54), xytext=(2.2, 4.5),
             arrowprops=dict(arrowstyle="->", color="#2563EB", lw=1.5),
             fontsize=9, fontweight="bold", color="#1D4ED8",
             bbox=dict(boxstyle="round,pad=0.3", fc="#EFF6FF", ec="#BFDBFE"))

plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, "ablation_comparison_chart.png")
fig.savefig(save_path, bbox_inches="tight", dpi=300)

artifact_save = os.path.join(ARTIFACT_DIR, "ablation_comparison_chart.png")
fig.savefig(artifact_save, bbox_inches="tight", dpi=300)
plt.close(fig)

print(f"[SUCCESS] Ablation comparison chart saved to: {save_path}")
