# Semantic Segmentation using RGB and Multispectral Image Fusion

## Goal

Build a complete **research paper implementation** for semantic segmentation that fuses RGB and multispectral (Near-Infrared) imagery using a novel **Dual-Encoder SegFormer with Adaptive Spectral-Spatial Gated Fusion (ASG-Net)**. The project will:
1. Implement the model architecture in PyTorch
2. Train and evaluate on the **ISPRS Potsdam** benchmark dataset
3. Conduct ablation studies demonstrating the fusion advantage
4. Generate a full research paper with results, figures, and analysis
5. Provide a pathway to adapt the model to your college campus imagery

---

## User Review Required

> [!IMPORTANT]
> **Dataset Choice**: We'll use **ISPRS Potsdam** as the primary benchmark — it has 5cm resolution aerial imagery with RGB + NIR bands and 6 urban classes (buildings, roads, vegetation, trees, cars, clutter). These classes closely match your campus needs. Training on Potsdam first gives you a strong pretrained model that can later be fine-tuned on your own campus images.

> [!IMPORTANT]
> **Compute Environment**: Since you don't have a local NVIDIA GPU, the project is designed for **Google Colab (free T4 GPU)**. We'll use the lighter `MiT-B2` backbone (~25M params per encoder) which trains comfortably on Colab. All code will include Colab-compatible setup cells.

> [!WARNING]
> **Campus Data Collection**: To segment your actual college campus, you'll eventually need to:
> 1. Capture drone/aerial images (RGB + optionally NIR) of your campus
> 2. Annotate them with segmentation masks using [LabelMe](https://github.com/labelmeai/labelme) or [CVAT](https://www.cvat.ai/)
> 3. Fine-tune the Potsdam-pretrained model on your campus data
>
> We can build the fine-tuning pipeline now, but actual campus-specific results depend on your data collection.

---

## Open Questions

> [!IMPORTANT]
> 1. **Paper Name**: Proposed title: *"ASG-Net: Adaptive Spectral-Spatial Gated Fusion Network for RGB-Multispectral Semantic Segmentation"* — Do you like this, or prefer a different name?
> 2. **Google Colab**: Do you have access to Google Colab? (Free version has T4 GPU with ~15GB VRAM, sufficient for this project)
> 3. **Paper Format**: Do you need a specific LaTeX template (IEEE, Springer, ACM) or is a general format okay?

---

## Architecture: ASG-Net (Our Proposed Model)

```
     ┌─────────────────────┐         ┌─────────────────────┐
     │   RGB Image (3ch)   │         │ NIR/MS Image (1-3ch) │
     └──────────┬──────────┘         └──────────┬──────────┘
                │                                │
                ▼                                ▼
     ┌──────────────────┐             ┌──────────────────┐
     │  MiT-B2 Encoder  │             │  MiT-B2 Encoder  │
     │    (Pretrained)   │             │  (Pretrained)    │
     │                   │             │                  │
     │  Stage1: 1/4 res  │             │  Stage1: 1/4 res │
     │  Stage2: 1/8 res  │             │  Stage2: 1/8 res │
     │  Stage3: 1/16 res │             │  Stage3: 1/16 res│
     │  Stage4: 1/32 res │             │  Stage4: 1/32 res│
     └────────┬─────────┘             └────────┬─────────┘
              │                                 │
              │    ┌────────────────────────┐    │
              ├───►│  ASG Fusion Block (×4) │◄───┤
              │    │  at each encoder stage │    │
              │    └───────────┬────────────┘    │
              │                │                 │
              │                ▼                 │
              │  ┌──────────────────────────┐    │
              │  │ Cross-Modal Rectification │   │
              │  │ • Channel calibration     │   │
              │  │ • Spatial attention map    │   │
              │  └────────────┬─────────────┘   │
              │               │                  │
              │               ▼                  │
              │  ┌──────────────────────────┐    │
              │  │ Adaptive Gated Fusion     │   │
              │  │ • α = σ(W·[F_rgb;F_ms])  │   │
              │  │ • F_fused = α⊙F_rgb +    │   │
              │  │   (1-α)⊙F_ms             │   │
              │  └────────────┬─────────────┘   │
              │               │                  │
              └───────────────┼──────────────────┘
                              │
                              ▼
              ┌──────────────────────────────┐
              │  SegFormer All-MLP Decoder    │
              │  • Unify channels → 256      │
              │  • Upsample to 1/4 res       │
              │  • Concat → MLP → Pred Mask  │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │  Auxiliary Edge Decoder       │
              │  (Boundary-aware supervision) │
              └──────────────────────────────┘
```

### Key Novel Contributions (for the Paper)

1. **Adaptive Spectral-Spatial Gating (ASG)**: A learnable gating mechanism that dynamically weights RGB vs. multispectral features based on scene content, replacing static concatenation or addition
2. **Cross-Modal Feature Rectification**: Channel and spatial calibration that aligns feature distributions before fusion, reducing modality gap
3. **Boundary-Aware Auxiliary Loss**: An edge detection branch that supervises boundary pixels separately, improving segmentation sharpness
4. **Comprehensive Ablation**: 5-variant comparison proving each component's contribution

---

## Dataset: ISPRS Potsdam

| Property | Value |
|:---|:---|
| **Resolution** | 5 cm GSD (very high resolution) |
| **Tile Size** | 6000 × 6000 pixels (38 tiles) |
| **Bands** | RGB (3ch) + NIR (1ch) + DSM (1ch) |
| **Classes** | 6: Impervious Surfaces, Building, Low Vegetation, Tree, Car, Clutter |
| **Train / Val / Test** | 18 / 6 / 14 tiles (standard split) |
| **Patch Size** | 512 × 512 crops (stride 256, with augmentation) |
| **Download** | [ISPRS Potsdam](https://seafile.projekt.uni-hannover.de/f/429be50cc79d423ab6c4/) (password: `secret`) |

### Class Mapping to Campus

| Potsdam Class | Campus Equivalent |
|:---|:---|
| Impervious Surfaces | **Pathways + Parking** |
| Building | **Buildings** |
| Low Vegetation | **Lawns / Playground** |
| Tree | **Trees** |
| Car | **Vehicles** |
| Clutter/Background | **Water Bodies + Others** |

---

## Proposed Changes

### Project Structure

```
d:\Dl_project\
├── configs/
│   └── config.yaml                # [NEW] All hyperparameters & paths
├── data/
│   ├── __init__.py                # [NEW] Data module exports
│   ├── potsdam_dataset.py         # [NEW] ISPRS Potsdam dataloader
│   ├── transforms.py              # [NEW] Augmentation pipeline
│   └── download_potsdam.py        # [NEW] Download & preprocess script
├── models/
│   ├── __init__.py                # [NEW] Model exports
│   ├── encoders/
│   │   ├── __init__.py            # [NEW]
│   │   └── mit_encoder.py         # [NEW] Mix-Transformer encoder (MiT-B2)
│   ├── fusion/
│   │   ├── __init__.py            # [NEW]
│   │   ├── asg_block.py           # [NEW] ★ Adaptive Spectral-Spatial Gated Fusion
│   │   └── cm_frm.py              # [NEW] Cross-Modal Feature Rectification Module
│   ├── decoders/
│   │   ├── __init__.py            # [NEW]
│   │   ├── segformer_head.py      # [NEW] All-MLP decoder head
│   │   └── edge_decoder.py        # [NEW] Auxiliary boundary decoder
│   └── asg_net.py                 # [NEW] ★ Main ASG-Net model (dual-encoder fusion)
├── losses/
│   ├── __init__.py                # [NEW]
│   ├── combined_loss.py           # [NEW] CE + Lovász + Boundary Loss
│   └── lovasz_loss.py             # [NEW] Lovász-Softmax for mIoU optimization
├── utils/
│   ├── __init__.py                # [NEW]
│   ├── metrics.py                 # [NEW] mIoU, OA, F1-score computation
│   ├── visualize.py               # [NEW] Prediction visualization & comparison
│   └── lr_scheduler.py            # [NEW] Polynomial LR decay
├── train.py                       # [NEW] Training loop with logging
├── evaluate.py                    # [NEW] Evaluation & metric reporting
├── predict.py                     # [NEW] Inference on new images
├── ablation.py                    # [NEW] Run all ablation experiments
├── requirements.txt               # [NEW] Dependencies
├── setup_colab.sh                 # [NEW] Colab environment setup script
├── notebooks/
│   └── ASGNet_Colab.ipynb         # [NEW] Full Colab training notebook
└── paper/
    ├── paper.md                   # [NEW] Full research paper draft
    ├── figures/                   # [NEW] Generated figures for the paper
    └── references.bib             # [NEW] Bibliography
```

---

### Component 1: Data Pipeline

#### [NEW] [config.yaml](file:///d:/Dl_project/configs/config.yaml)
- All hyperparameters: learning rate (`6e-5`), batch size (`4`), epochs (`100`), patch size (`512`)
- Dataset paths, class definitions, color maps
- Augmentation settings (random flip, rotate, scale, color jitter)
- Model config (backbone type, fusion type, decoder channels)

#### [NEW] [potsdam_dataset.py](file:///d:/Dl_project/data/potsdam_dataset.py)
- Custom `torch.utils.data.Dataset` for ISPRS Potsdam
- Loads RGB (3ch) and NIR (1ch) as separate tensors
- Applies synchronized augmentations to both modalities + mask
- Returns `{'rgb': tensor, 'nir': tensor, 'mask': tensor, 'edge': tensor}`
- Handles 6000×6000 tile cropping into 512×512 patches with configurable stride

#### [NEW] [transforms.py](file:///d:/Dl_project/data/transforms.py)
- Synchronized transforms: `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation(90°)`
- `RandomScale(0.5–2.0)` with bilinear interpolation for images, nearest for masks
- `ColorJitter` (brightness, contrast, saturation) — applied only to RGB
- `Normalize` with ImageNet stats for RGB, separate stats for NIR
- Edge map generation via Canny/Sobel on ground truth masks

#### [NEW] [download_potsdam.py](file:///d:/Dl_project/data/download_potsdam.py)
- Script to download Potsdam dataset tiles
- Extracts RGBIR 4-band TIFFs, splits into RGB and NIR channels
- Generates 512×512 patch crops with overlap
- Creates train/val/test split following standard protocol
- Generates edge maps from ground truth labels

---

### Component 2: Model Architecture (Core)

#### [NEW] [mit_encoder.py](file:///d:/Dl_project/models/encoders/mit_encoder.py)
- Mix-Transformer (MiT-B2) hierarchical encoder
- 4 stages producing features at 1/4, 1/8, 1/16, 1/32 resolution
- Channel dimensions: [64, 128, 320, 512]
- Loads pretrained weights from `segformer-b2-finetuned-ade-512-512`
- Supports 1-channel or 3-channel input (for NIR branch, repeats/adapts first conv)

#### [NEW] [asg_block.py](file:///d:/Dl_project/models/fusion/asg_block.py) ★ Key Contribution
- **Adaptive Spectral-Spatial Gated Fusion Block**
- Input: `F_rgb` and `F_ms` feature maps from same encoder stage
- Steps:
  1. **Channel Attention**: Squeeze-and-Excitation on concatenated features
  2. **Spatial Attention**: Cross-modal spatial attention via depthwise convolution
  3. **Gating**: `α = σ(Conv1x1(Concat(F_rgb, F_ms)))` — learned per-pixel gate
  4. **Fusion**: `F_out = α ⊙ F_rgb + (1 - α) ⊙ F_ms + Residual`
- ~0.5M additional parameters per stage (lightweight)

#### [NEW] [cm_frm.py](file:///d:/Dl_project/models/fusion/cm_frm.py)
- **Cross-Modal Feature Rectification Module** (inspired by CMX)
- Calibrates channel statistics between RGB and MS branches
- Uses cross-modality global average pooling to generate recalibration weights
- Aligns feature distributions before gated fusion

#### [NEW] [segformer_head.py](file:///d:/Dl_project/models/decoders/segformer_head.py)
- Lightweight All-MLP decoder following SegFormer design
- Unifies multi-scale fused features to 256 channels via `Linear` layers
- Upsamples all to 1/4 resolution, concatenates, and predicts via MLP
- Output: class logits at 1/4 resolution (upsampled to full res during eval)

#### [NEW] [edge_decoder.py](file:///d:/Dl_project/models/decoders/edge_decoder.py)
- Auxiliary boundary decoder for edge-aware supervision
- Takes Stage 1 and Stage 2 fused features
- Predicts binary edge map (boundary / non-boundary)
- Weighted Binary Cross-Entropy loss (boundary pixels are rare)

#### [NEW] [asg_net.py](file:///d:/Dl_project/models/asg_net.py) ★ Main Model
- Orchestrates the full ASG-Net architecture
- `__init__`: Creates dual MiT-B2 encoders, 4 ASG fusion blocks, MLP decoder, edge decoder
- `forward(rgb, nir)`:
  1. Extract 4-stage features from each encoder
  2. Apply CM-FRM + ASG fusion at each stage
  3. Decode fused features → semantic prediction
  4. Decode early features → edge prediction (training only)
- Returns `{'seg_pred': logits, 'edge_pred': edge_logits}`

---

### Component 3: Loss Functions

#### [NEW] [combined_loss.py](file:///d:/Dl_project/losses/combined_loss.py)
- **Total Loss** = `CE_loss` + `λ₁ × Lovász_loss` + `λ₂ × Edge_loss`
- `λ₁ = 0.75` (Lovász-Softmax directly optimizes mIoU)
- `λ₂ = 0.5` (boundary-aware supervision, decayed after epoch 50)
- Class-weighted Cross-Entropy to handle class imbalance (vegetation >> cars)

#### [NEW] [lovasz_loss.py](file:///d:/Dl_project/losses/lovasz_loss.py)
- Implementation of Lovász-Softmax loss for multi-class segmentation
- Surrogate optimization of the Jaccard (IoU) metric
- Critical for achieving high mIoU (vs. plain CE which optimizes pixel accuracy)

---

### Component 4: Training & Evaluation

#### [NEW] [train.py](file:///d:/Dl_project/train.py)
- Full training loop with:
  - Mixed precision training (`torch.cuda.amp`) for Colab efficiency
  - AdamW optimizer (`lr=6e-5`, `weight_decay=0.01`)
  - Polynomial LR decay (`power=0.9`)
  - Gradient clipping (`max_norm=1.0`)
  - TensorBoard / Weights & Biases logging
  - Best model checkpointing (by val mIoU)
  - Early stopping (patience=15)
- 100 epochs, batch size 4 (Colab T4 friendly)

#### [NEW] [evaluate.py](file:///d:/Dl_project/evaluate.py)
- Sliding window inference on full 6000×6000 tiles (overlap=256)
- Computes: **mIoU**, **Overall Accuracy (OA)**, **F1-Score**, **per-class IoU**
- Generates confusion matrix and per-class performance table
- Saves colored prediction maps overlaid on RGB images

#### [NEW] [ablation.py](file:///d:/Dl_project/ablation.py)
- Runs 5 ablation experiments:

| Variant | Description | Purpose |
|:---|:---|:---|
| **V1: RGB-Only** | Single MiT-B2 + SegFormer decoder, RGB only | Baseline — no MS |
| **V2: Early Fusion** | Single encoder, 4ch (RGBNIR) input | Simplest fusion baseline |
| **V3: Dual-Encoder + Add** | Two encoders, element-wise addition | Mid-level fusion baseline |
| **V4: Dual-Encoder + ASG** | Two encoders + ASG blocks (no CM-FRM) | Ablate rectification |
| **V5: Full ASG-Net** | Two encoders + CM-FRM + ASG + Edge Loss | **Full proposed model** |

- Auto-generates comparison table and bar charts

#### [NEW] [predict.py](file:///d:/Dl_project/predict.py)
- Inference on new images (your campus photos)
- Loads trained model checkpoint
- Accepts RGB + NIR image pair (or RGB-only with dummy NIR)
- Outputs colored segmentation overlay

---

### Component 5: Visualization & Utilities

#### [NEW] [visualize.py](file:///d:/Dl_project/utils/visualize.py)
- Color-coded segmentation maps with Potsdam palette
- Side-by-side comparison: RGB | Ground Truth | Prediction
- Per-class IoU bar charts
- Confusion matrix heatmap
- Training loss/mIoU curves
- Generates publication-quality figures (300 DPI, matplotlib)

#### [NEW] [metrics.py](file:///d:/Dl_project/utils/metrics.py)
- Streaming metric computation (memory-efficient for large tiles)
- `StreamingConfusionMatrix` class
- Methods: `compute_miou()`, `compute_oa()`, `compute_f1()`, `compute_per_class_iou()`

---

### Component 6: Google Colab Notebook

#### [NEW] [ASGNet_Colab.ipynb](file:///d:/Dl_project/notebooks/ASGNet_Colab.ipynb)
- Self-contained notebook with:
  - GPU detection and environment setup
  - Dataset download and preprocessing
  - Model training with progress visualization
  - Evaluation with result tables
  - Ablation study runner
  - Visualization of all results
- Designed for Google Colab free tier (T4, 15GB VRAM)

---

### Component 7: Research Paper

#### [NEW] [paper.md](file:///d:/Dl_project/paper/paper.md)
Full research paper with sections:

1. **Abstract** (~250 words)
2. **Introduction** — Problem motivation, RGB limitations, multispectral advantages, contribution summary
3. **Related Work** — Semantic segmentation evolution, multimodal fusion strategies, RGB-X methods
4. **Methodology**
   - 3.1 Overall Architecture
   - 3.2 Dual MiT Encoder
   - 3.3 Cross-Modal Feature Rectification (CM-FRM)
   - 3.4 Adaptive Spectral-Spatial Gated Fusion (ASG Block)
   - 3.5 All-MLP Decoder
   - 3.6 Boundary-Aware Auxiliary Loss
   - 3.7 Total Loss Function
5. **Experiments**
   - 4.1 Dataset & Implementation Details
   - 4.2 Comparison with State-of-the-Art
   - 4.3 Ablation Study
   - 4.4 Qualitative Analysis
   - 4.5 Campus Adaptation Case Study
6. **Conclusion & Future Work**
7. **References** (20+ citations)

---

## Verification Plan

### Automated Tests
```bash
# 1. Verify model builds and forward pass works
python -c "from models.asg_net import ASGNet; import torch; m = ASGNet(); out = m(torch.randn(1,3,512,512), torch.randn(1,1,512,512)); print(out['seg_pred'].shape)"

# 2. Verify dataset loads correctly
python -c "from data.potsdam_dataset import PotsdamDataset; ds = PotsdamDataset('data/potsdam', split='train'); print(len(ds), ds[0]['rgb'].shape)"

# 3. Run one training epoch
python train.py --epochs 1 --batch_size 2 --fast_dev_run

# 4. Run evaluation on validation set
python evaluate.py --checkpoint checkpoints/best_model.pth --split val

# 5. Run full ablation study
python ablation.py --epochs 100
```

### Manual Verification
- Visual inspection of prediction maps vs. ground truth
- Verify mIoU scores are in expected range (>80% for Potsdam with SOTA methods)
- Check training curves for convergence (no divergence or overfitting)
- Review paper figures for publication quality

### Expected Performance Targets

| Model Variant | Expected mIoU | Expected OA |
|:---|:---:|:---:|
| V1: RGB-Only | ~82% | ~88% |
| V2: Early Fusion (4ch) | ~84% | ~89% |
| V3: Dual + Addition | ~86% | ~90% |
| V4: Dual + ASG (no CM-FRM) | ~88% | ~91% |
| **V5: Full ASG-Net** | **~90%** | **~92%** |

---

## Implementation Order

1. **Phase 1**: Project setup, dependencies, config → `requirements.txt`, `config.yaml`
2. **Phase 2**: Data pipeline → `download_potsdam.py`, `potsdam_dataset.py`, `transforms.py`
3. **Phase 3**: Model architecture → `mit_encoder.py`, `asg_block.py`, `cm_frm.py`, `segformer_head.py`, `edge_decoder.py`, `asg_net.py`
4. **Phase 4**: Loss functions → `lovasz_loss.py`, `combined_loss.py`
5. **Phase 5**: Training & evaluation → `train.py`, `evaluate.py`, `metrics.py`, `lr_scheduler.py`
6. **Phase 6**: Visualization & inference → `visualize.py`, `predict.py`
7. **Phase 7**: Ablation experiments → `ablation.py`
8. **Phase 8**: Colab notebook → `ASGNet_Colab.ipynb`
9. **Phase 9**: Research paper → `paper.md`, figures, references
