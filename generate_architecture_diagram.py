"""
Generate a publication-grade, professional architectural diagram for ASG-Net:
Adaptive Spectral-Spatial Gated Fusion Network for RGB-Multispectral Semantic Segmentation.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch
import matplotlib.patheffects as patheffects

OUTPUT_DIR = r"d:\Dl_project\results"
PAPER_FIG_DIR = r"d:\Dl_project\paper\figures"
ARTIFACT_DIR = r"C:\Users\Md Dilnawaz Ahmad\.gemini\antigravity\brain\d3524f49-ce18-47cc-8799-ae2e6eef7175"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PAPER_FIG_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Create high-res wide figure
fig, ax = plt.subplots(figsize=(22, 11), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
ax.axis("off")
fig.patch.set_facecolor("#FFFFFF")

# Styling constants
FONT_TITLE = {"fontsize": 13, "fontweight": "bold", "color": "#0F172A", "ha": "center", "va": "center"}
FONT_HEADER = {"fontsize": 11, "fontweight": "bold", "color": "#1E293B", "ha": "center", "va": "center"}
FONT_BODY = {"fontsize": 9, "fontweight": "normal", "color": "#334155", "ha": "center", "va": "center"}
FONT_LABEL = {"fontsize": 8, "fontweight": "bold", "color": "#475569", "ha": "center", "va": "center"}

def draw_card(ax, x, y, w, h, bg_color, border_color="#CBD5E1", radius=1.0, lw=1.5):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=lw,
        zorder=2
    )
    ax.add_patch(box)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color="#64748B", lw=1.5, style="-|>", rad=0.0):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=f"{style},head_length=5,head_width=3",
        color=color,
        linewidth=lw,
        connectionstyle=f"arc3,rad={rad}",
        zorder=3
    )
    ax.add_patch(arrow)
    return arrow

# ==========================================
# 0. MAIN TITLE & SUBTITLE BANNER
# ==========================================
ax.text(50, 48.0, "ASG-Net: Adaptive Spectral-Spatial Gated Fusion Network Architecture",
        fontsize=18, fontweight="black", color="#0F172A", ha="center")
ax.text(50, 46.2, "RGB-Multispectral Dual-Stream Hierarchy with Cross-Modal Rectification (CM-FRM) and Multi-Scale Boundary Supervision",
        fontsize=11, fontweight="semibold", color="#475569", ha="center")

# ==========================================
# 1. INPUT MODALITIES
# ==========================================
# RGB Input
draw_card(ax, 2, 33, 9, 8.5, "#EFF6FF", "#3B82F6", radius=0.8, lw=2.0)
ax.text(6.5, 39.5, "RGB Imagery", fontsize=11, fontweight="bold", color="#1D4ED8", ha="center")
ax.text(6.5, 37.5, "True Color\n(Red, Green, Blue)", fontsize=9, color="#1E40AF", ha="center")
ax.text(6.5, 34.5, "H × W × 3", fontsize=9, fontweight="bold", color="#3B82F6", ha="center")

# NIR Input
draw_card(ax, 2, 10.5, 9, 8.5, "#ECFDF5", "#10B981", radius=0.8, lw=2.0)
ax.text(6.5, 17.0, "NIR Imagery", fontsize=11, fontweight="bold", color="#047857", ha="center")
ax.text(6.5, 15.0, "Physical Band 4\n(0.7 - 1.1 μm)", fontsize=9, color="#065F46", ha="center")
ax.text(6.5, 12.0, "H × W × 1", fontsize=9, fontweight="bold", color="#10B981", ha="center")

# Arrows from Input to Encoders
draw_arrow(ax, 11, 37.25, 15, 37.25, color="#3B82F6", lw=2.0)
draw_arrow(ax, 11, 14.75, 15, 14.75, color="#10B981", lw=2.0)

# ==========================================
# 2. DUAL-STREAM MiT-B2 ENCODERS (4 STAGES)
# ==========================================
# Stage coordinates & configs
stage_xs = [15, 23.5, 32, 40.5]
stage_w = 7.0
channels = [64, 128, 320, 512]
scales = ["H/4 × W/4", "H/8 × W/8", "H/16 × W/16", "H/32 × W/32"]

# Enclosing Group Box for RGB Stream
draw_card(ax, 14, 31.5, 34.5, 11.5, "#F8FAFC", "#94A3B8", radius=1.2, lw=1.0)
ax.text(14.8, 41.8, "RGB Stream: MiT-B2 Transformer Encoder (Pretrained ADE20K)", fontsize=9, fontweight="bold", color="#2563EB", ha="left")

# Enclosing Group Box for NIR Stream
draw_card(ax, 14, 9.0, 34.5, 11.5, "#F8FAFC", "#94A3B8", radius=1.2, lw=1.0)
ax.text(14.8, 19.3, "NIR Stream: Adapted MiT-B2 Encoder (Multispectral Weights)", fontsize=9, fontweight="bold", color="#059669", ha="left")

for i, (sx, ch, sc) in enumerate(zip(stage_xs, channels, scales)):
    # RGB Stage Box
    draw_card(ax, sx, 33, stage_w, 7.5, "#DBEAFE", "#2563EB", radius=0.6, lw=1.5)
    ax.text(sx + stage_w/2, 38.5, f"Stage {i+1}", fontsize=10, fontweight="bold", color="#1E3A8A", ha="center")
    ax.text(sx + stage_w/2, 36.8, f"MiT Block", fontsize=8.5, color="#1E40AF", ha="center")
    ax.text(sx + stage_w/2, 35.0, f"C = {ch}", fontsize=8.5, fontweight="bold", color="#2563EB", ha="center")
    ax.text(sx + stage_w/2, 33.8, sc, fontsize=7.5, color="#64748B", ha="center")

    # NIR Stage Box
    draw_card(ax, sx, 10.5, stage_w, 7.5, "#D1FAE5", "#059669", radius=0.6, lw=1.5)
    ax.text(sx + stage_w/2, 16.0, f"Stage {i+1}", fontsize=10, fontweight="bold", color="#064E3B", ha="center")
    ax.text(sx + stage_w/2, 14.3, f"MiT Block", fontsize=8.5, color="#065F46", ha="center")
    ax.text(sx + stage_w/2, 12.5, f"C = {ch}", fontsize=8.5, fontweight="bold", color="#059669", ha="center")
    ax.text(sx + stage_w/2, 11.3, sc, fontsize=7.5, color="#64748B", ha="center")

    # Horizontal intra-stream arrows
    if i < 3:
        draw_arrow(ax, sx + stage_w, 36.75, stage_xs[i+1], 36.75, color="#2563EB", lw=1.5)
        draw_arrow(ax, sx + stage_w, 14.25, stage_xs[i+1], 14.25, color="#059669", lw=1.5)

# ==========================================
# 3. CM-FRM & ASG FUSION BLOCKS (STAGES 1-4)
# ==========================================
# Draw 4 Fusion Nodes between the streams
fusion_xs = [15, 23.5, 32, 40.5]
for i, (fx, ch, sc) in enumerate(zip(fusion_xs, channels, scales)):
    # Vertical arrows from RGB and NIR stages to the fusion block
    draw_arrow(ax, fx + stage_w/2, 33.0, fx + stage_w/2, 27.5, color="#2563EB", lw=1.5)
    draw_arrow(ax, fx + stage_w/2, 18.0, fx + stage_w/2, 23.5, color="#059669", lw=1.5)

    # Fusion block card
    draw_card(ax, fx, 23.5, stage_w, 4.0, "#FEF3C7", "#D97706", radius=0.6, lw=1.5)
    ax.text(fx + stage_w/2, 26.0, f"ASG Block {i+1}", fontsize=9, fontweight="bold", color="#92400E", ha="center")
    ax.text(fx + stage_w/2, 24.5, "CM-FRM + Gate α", fontsize=7.5, color="#B45309", ha="center")

# ==========================================
# 4. ZOOM-IN DETAIL: THE ASG BLOCK CALLOUT
# ==========================================
# Draw a dedicated detailed box below explaining the inner ASG Block
draw_card(ax, 50, 15.5, 23, 16.0, "#F5F3FF", "#7C3AED", radius=1.0, lw=2.0)
ax.text(61.5, 30.2, "Inside the ASG Fusion Block (Detail)", fontsize=11, fontweight="bold", color="#5B21B6", ha="center")

# Step 1: CM-FRM
draw_card(ax, 51.5, 25.5, 9.5, 3.4, "#FFFFFF", "#C4B5FD", radius=0.5, lw=1.0)
ax.text(56.25, 27.5, "1. CM-FRM Rectification", fontsize=8.5, fontweight="bold", color="#6D28D9", ha="center")
ax.text(56.25, 26.3, "Cross-Modal GAP + Depthwise Conv", fontsize=7.5, color="#7C3AED", ha="center")

# Step 2: Channel Attention
draw_card(ax, 62.0, 25.5, 9.5, 3.4, "#FFFFFF", "#C4B5FD", radius=0.5, lw=1.0)
ax.text(66.75, 27.5, "2. Channel Attention", fontsize=8.5, fontweight="bold", color="#6D28D9", ha="center")
ax.text(66.75, 26.3, "Joint SE Bottleneck (2C → 2C/16)", fontsize=7.5, color="#7C3AED", ha="center")

# Step 3: Spatial Gating
draw_card(ax, 51.5, 20.5, 20.0, 3.8, "#EDE9FE", "#7C3AED", radius=0.5, lw=1.2)
ax.text(61.5, 23.0, "3. Adaptive Spatial Gating Module (α)", fontsize=9, fontweight="bold", color="#4C1D95", ha="center")
ax.text(61.5, 21.6, "Gate α = σ( Conv1×1( [ F_rgb ; F_nir ] ) )   ∈  [0, 1] per pixel", fontsize=8.5, fontweight="bold", color="#6D28D9", ha="center")

# Step 4: Formulation
draw_card(ax, 51.5, 16.5, 20.0, 3.0, "#FFFFFF", "#A78BFA", radius=0.5, lw=1.0)
ax.text(61.5, 18.3, "Fused Output:  F_out = α ⊙ F_rgb + (1 - α) ⊙ F_nir + Conv(Res)", fontsize=8.5, fontweight="bold", color="#1E293B", ha="center")
ax.text(61.5, 17.2, "α → 0: Relies on NIR (Shadows/Canopy)  |  α → 1: Relies on RGB (Edges/Roofs)", fontsize=7.5, color="#64748B", ha="center")

# Connect Stage 2 Fusion block to Detail box
draw_arrow(ax, 47.5, 25.5, 50.0, 23.5, color="#7C3AED", lw=1.5, style="-|>", rad=0.1)

# ==========================================
# 5. ALL-MLP SEGFORMER DECODER
# ==========================================
draw_card(ax, 75, 23.0, 11, 18.0, "#ECFEFF", "#0891B2", radius=1.0, lw=2.0)
ax.text(80.5, 39.5, "SegFormer All-MLP Decoder", fontsize=11, fontweight="bold", color="#0E7490", ha="center")

decoder_steps = [
    ("MLP Projections", "Project stages 1-4 to C = 256"),
    ("Bilinear Upsampling", "Upsample all to 1/4 resolution"),
    ("Feature Concatenation", "Fuse 4 scales (4 × 256 = 1024 ch)"),
    ("Unified MLP Head", "Linear(1024, 256) → ReLU → Dropout"),
    ("Classifier Projection", "Linear(256, 6) → Softmax")
]

for i, (title, desc) in enumerate(decoder_steps):
    y = 35.5 - i * 2.8
    draw_card(ax, 76, y, 9.0, 2.2, "#FFFFFF", "#A5F3FC", radius=0.4, lw=1.0)
    ax.text(80.5, y + 1.4, title, fontsize=8, fontweight="bold", color="#155E75", ha="center")
    ax.text(80.5, y + 0.6, desc, fontsize=7, color="#0E7490", ha="center")

# Arrows from 4 fusion blocks to Decoder
for fx in fusion_xs:
    draw_arrow(ax, fx + stage_w, 25.5, 75.0, 32.0, color="#D97706", lw=1.2, rad=0.08)

# ==========================================
# 6. AUXILIARY BOUNDARY/EDGE DECODER
# ==========================================
draw_card(ax, 50, 4.0, 23, 9.5, "#FFF1F2", "#E11D48", radius=1.0, lw=1.8)
ax.text(61.5, 12.0, "Auxiliary Multi-Scale Edge Decoder", fontsize=10, fontweight="bold", color="#9F1239", ha="center")
ax.text(61.5, 10.3, "Supervises high-frequency boundaries (roads & buildings)", fontsize=8, color="#BE123C", ha="center")

draw_card(ax, 51.5, 5.0, 20.0, 4.2, "#FFFFFF", "#FDA4AF", radius=0.5, lw=1.0)
ax.text(61.5, 7.8, "Inputs: Stage 1 (H/4) & Stage 2 (H/8) Fused Features", fontsize=8, fontweight="bold", color="#9F1239", ha="center")
ax.text(61.5, 6.2, "Conv3×3 → BatchNorm → ReLU → Upsample → Conv1×1 → Sigmoid", fontsize=7.5, color="#64748B", ha="center")

# Edge arrows from Stages 1 & 2
draw_arrow(ax, 15 + stage_w/2, 23.5, 50.0, 8.5, color="#E11D48", lw=1.2, rad=0.15)
draw_arrow(ax, 23.5 + stage_w/2, 23.5, 50.0, 8.5, color="#E11D48", lw=1.2, rad=0.10)

# ==========================================
# 7. FINAL MULTI-TASK OUTPUTS & LOSSES
# ==========================================
# Semantic Segmentation Map Output
draw_card(ax, 88.5, 30.0, 9.5, 9.0, "#EFF6FF", "#2563EB", radius=0.8, lw=2.0)
ax.text(93.25, 37.0, "Semantic Mask", fontsize=11, fontweight="bold", color="#1E3A8A", ha="center")
ax.text(93.25, 34.8, "6-Class ISPRS Map\n(H × W × 6)", fontsize=8.5, color="#1D4ED8", ha="center")
ax.text(93.25, 31.8, "Roads, Buildings, Trees,\nLow Veg, Cars, Clutter", fontsize=7.5, color="#64748B", ha="center")

draw_arrow(ax, 86.0, 34.5, 88.5, 34.5, color="#0891B2", lw=2.0)

# Boundary Map Output
draw_card(ax, 75.0, 4.0, 11, 7.0, "#FFF1F2", "#E11D48", radius=0.8, lw=2.0)
ax.text(80.5, 9.2, "Boundary Map", fontsize=10, fontweight="bold", color="#9F1239", ha="center")
ax.text(80.5, 7.2, "Edge Probability Map\n(H × W × 1)", fontsize=8, color="#BE123C", ha="center")
ax.text(80.5, 5.2, "Crisp 90° Corner Outlines", fontsize=7.5, color="#64748B", ha="center")

draw_arrow(ax, 73.0, 7.5, 75.0, 7.5, color="#E11D48", lw=2.0)

# Loss Function Summary Banner
draw_card(ax, 2, 1.0, 96.0, 2.4, "#0F172A", "#334155", radius=0.5, lw=1.0)
ax.text(50, 2.2, "Composite Multi-Task Loss:   L_total  =  1.0 · L_WeightedCE(P_seg, Y)  +  0.75 · L_Lovász(P_seg, Y)  +  0.50 · L_BCE(P_edge, Y_edge)",
        fontsize=9.5, fontweight="bold", color="#38BDF8", ha="center")

plt.tight_layout()

# Save paths
img_out_results = os.path.join(OUTPUT_DIR, "asg_net_architecture.png")
img_out_paper = os.path.join(PAPER_FIG_DIR, "asg_net_architecture.png")
img_out_artifact = os.path.join(ARTIFACT_DIR, "asg_net_architecture.png")

fig.savefig(img_out_results, bbox_inches="tight", dpi=300)
fig.savefig(img_out_paper, bbox_inches="tight", dpi=300)
fig.savefig(img_out_artifact, bbox_inches="tight", dpi=300)
plt.close(fig)

print(f"[SUCCESS] ASG-Net architecture diagram saved to:\n  • {img_out_results}\n  • {img_out_paper}\n  • {img_out_artifact}")
