"""
ASG-Net: ISPRS Potsdam Download Instructions and Preprocessing Pipeline
======================================================================
Preprocesses the raw ISPRS Potsdam 6000x6000 aerial tiles into aligned
multimodal 512x512 patches (RGB, NIR, and semantic labels) with configurable stride.

Usage:
------
# 1. View download instructions
python data/download_potsdam.py --instructions

# 2. Run full preprocessing pipeline on downloaded raw data
python data/download_potsdam.py --raw_dir /path/to/potsdam_raw --output_dir data/potsdam

# 3. Create a synthetic dataset for testing without large downloads
python data/download_potsdam.py --generate_synthetic --output_dir data/potsdam
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Generator, List, Optional, Sequence, Tuple, Union

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml


# Color mapping for ISPRS Potsdam Benchmark (RGB -> Class ID)
POTSDAM_COLOR_TO_CLASS: Dict[Tuple[int, int, int], int] = {
    (255, 255, 255): 0,  # Impervious Surface (White)
    (0, 0, 255): 1,      # Building (Blue)
    (0, 255, 255): 2,    # Low Vegetation (Cyan)
    (0, 255, 0): 3,      # Tree (Green)
    (255, 255, 0): 4,    # Car (Yellow)
    (255, 0, 0): 5,      # Clutter / Background (Red)
}

CLASS_NAMES: Sequence[str] = (
    "Impervious Surface",
    "Building",
    "Low Vegetation",
    "Tree",
    "Car",
    "Clutter",
)

# Standard ISPRS Potsdam Train/Val/Test Split
DEFAULT_TRAIN_IDS: List[str] = [
    "2_10", "2_11", "2_12", "3_10", "3_11", "3_12",
    "4_10", "4_11", "4_12", "5_10", "5_11", "5_12",
    "6_7", "6_8", "6_9", "6_10", "6_11", "6_12",
]

DEFAULT_VAL_IDS: List[str] = [
    "7_7", "7_8", "7_9", "7_10", "7_11", "7_12",
]

DEFAULT_TEST_IDS: List[str] = [
    "2_13", "2_14", "3_13", "3_14", "4_13", "4_14",
    "5_13", "5_14", "6_13", "6_14", "7_13",
]


def print_download_instructions() -> None:
    """
    Displays comprehensive step-by-step instructions for downloading
    the official ISPRS Potsdam benchmark dataset.
    """
    banner = "=" * 78
    instructions = f"""
{banner}
ISPRS POTSDAM 2D SEMANTIC LABELING BENCHMARK - DOWNLOAD INSTRUCTIONS
{banner}

The ISPRS Potsdam benchmark contains 38 very-high-resolution (5cm GSD)
orthorectified tiles of size 6000 x 6000 pixels.

1. ACCESS REPOSITORY:
   URL:      https://seafile.projekt.uni-hannover.de/f/429be50cc79d423ab6c4/
   Password: secret

2. DOWNLOAD THE FOLLOWING ARCHIVES / FOLDERS:
   a. '2_Ortho_RGB/'
      - Contains true-color RGB imagery (3 channels: Red, Green, Blue).
      - Files: top_potsdam_X_Y_RGB.tif

   b. '3_Ortho_IRRG/'
      - Contains False-Color Infrared imagery (3 channels: NIR, Red, Green).
      - Channel 0 contains the physical Near-Infrared (NIR) reflectance band.
      - Files: top_potsdam_X_Y_IRRG.tif

   c. '5_Labels_all/' or '5_Labels_for_participants/'
      - Contains ground truth semantic segmentation masks.
      - Files: top_potsdam_X_Y_label.tif (or top_potsdam_X_Y_label_noBoundary.tif)

3. ORGANIZE RAW DIRECTORY:
   Place the extracted folders into a single raw directory structured as:
     /path/to/potsdam_raw/
       ├── 2_Ortho_RGB/
       ├── 3_Ortho_IRRG/
       └── 5_Labels_all/

4. RUN PREPROCESSING PIPELINE:
   python data/download_potsdam.py --raw_dir /path/to/potsdam_raw --output_dir data/potsdam

{banner}
"""
    print(instructions)


def extract_nir_from_irrg(irrg_path: Union[str, Path]) -> np.ndarray:
    """
    Extracts the physical Near-Infrared (NIR) band from an IRRG image.

    In standard ISPRS Potsdam IRRG imagery:
      - Band 0: Near-Infrared (NIR)
      - Band 1: Red (R)
      - Band 2: Green (G)

    Args:
        irrg_path: Path to the IRRG image file (.tif, .png, etc.).

    Returns:
        2D NumPy array of shape (H, W) with uint8 intensity values in [0, 255].
    """
    irrg_path = Path(irrg_path)
    if not irrg_path.is_file():
        raise FileNotFoundError(f"IRRG file not found: {irrg_path}")

    # Try loading with tifffile for robust multi-band GeoTIFF handling
    try:
        import tifffile
        img = tifffile.imread(str(irrg_path))
    except Exception:
        # Fallback to PIL Image
        try:
            pil_img = Image.open(irrg_path)
            img = np.array(pil_img)
        except Exception:
            # Fallback to OpenCV if available
            if HAS_CV2 and cv2 is not None:
                img = cv2.imread(str(irrg_path), cv2.IMREAD_UNCHANGED)
            else:
                img = None

    if img is None:
        raise ValueError(f"Failed to read image from {irrg_path}")

    # Handle shape conventions: (H, W, C) vs (C, H, W)
    if img.ndim == 3:
        if img.shape[0] < img.shape[2] and img.shape[0] in (3, 4):
            # (C, H, W) format
            nir = img[0, :, :]
        else:
            # (H, W, C) format
            nir = img[:, :, 0]
    elif img.ndim == 2:
        nir = img
    else:
        raise ValueError(f"Unexpected IRRG image array shape: {img.shape}")

    # Handle 16-bit / floating-point TIFFs
    if nir.dtype == np.uint16:
        nir = (nir / 256).astype(np.uint8)
    elif np.issubdtype(nir.dtype, np.floating):
        if nir.max() <= 1.0:
            nir = (nir * 255.0).astype(np.uint8)
        else:
            nir = np.clip(nir, 0, 255).astype(np.uint8)
    else:
        nir = nir.astype(np.uint8)

    return nir


def process_labels(label_path: Union[str, Path]) -> np.ndarray:
    """
    Converts RGB color-coded Potsdam ground truth masks into discrete class index masks.

    Class Mappings:
      - White  (255, 255, 255) -> 0: Impervious Surface
      - Blue   (0, 0, 255)     -> 1: Building
      - Cyan   (0, 255, 255)   -> 2: Low Vegetation
      - Green  (0, 255, 0)     -> 3: Tree
      - Yellow (255, 255, 0)   -> 4: Car
      - Red    (255, 0, 0)     -> 5: Clutter / Background

    Args:
        label_path: Path to the label mask image.

    Returns:
        2D NumPy array of shape (H, W) with uint8 class indices in range [0, 5].
    """
    label_path = Path(label_path)
    if not label_path.is_file():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    # Load with PIL to guarantee standard RGB color channel order
    pil_img = Image.open(label_path)

    # Check if image is already a single-channel indexed mask
    if pil_img.mode in ("L", "P", "I"):
        raw_arr = np.array(pil_img)
        if raw_arr.max() < len(POTSDAM_COLOR_TO_CLASS):
            return raw_arr.astype(np.uint8)

    label_rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    h, w = label_rgb.shape[:2]

    # Initialize mask array with clutter (class 5) as default for boundary/unknown pixels
    mask = np.full((h, w), fill_value=5, dtype=np.uint8)

    r = label_rgb[:, :, 0]
    g = label_rgb[:, :, 1]
    b = label_rgb[:, :, 2]

    # Fast vectorized color matching
    for (cr, cg, cb), class_idx in POTSDAM_COLOR_TO_CLASS.items():
        match = (r == cr) & (g == cg) & (b == cb)
        mask[match] = class_idx

    return mask


def create_patches(
    image: np.ndarray,
    patch_size: int = 512,
    stride: int = 256,
) -> Generator[Tuple[int, int, np.ndarray], None, None]:
    """
    Crops large aerial orthotiles into fixed-size overlapping patches.
    Guarantees complete coverage of the tile by appending boundary coordinates.

    Args:
        image: 2D or 3D NumPy array of shape (H, W) or (H, W, C).
        patch_size: Crop window height and width. Default: 512.
        stride: Step size between consecutive patch origins. Default: 256.

    Yields:
        Tuples of (y, x, patch) where (y, x) is the top-left coordinate.
    """
    h, w = image.shape[:2]

    if h < patch_size or w < patch_size:
        raise ValueError(
            f"Image dimensions ({h}, {w}) are smaller than requested patch_size ({patch_size})."
        )

    # Compute vertical start coordinates
    y_starts = list(range(0, h - patch_size + 1, stride))
    if y_starts[-1] != h - patch_size:
        y_starts.append(h - patch_size)

    # Compute horizontal start coordinates
    x_starts = list(range(0, w - patch_size + 1, stride))
    if x_starts[-1] != w - patch_size:
        x_starts.append(w - patch_size)

    for y in y_starts:
        for x in x_starts:
            patch = image[y : y + patch_size, x : x + patch_size].copy()
            yield y, x, patch


def _find_tile_file(search_dir: Path, tile_id: str, keyword: str = "") -> Optional[Path]:
    """
    Locates a tile file in `search_dir` matching the tile ID (e.g. '2_10' or '02_10')
    and an optional keyword (e.g. 'RGB', 'IRRG', 'label').
    """
    if not search_dir.is_dir():
        return None

    # Handle zero-padded tile IDs (e.g., '2_10' vs '02_10')
    parts = tile_id.split("_")
    patterns = [tile_id]
    if len(parts) == 2:
        patterns.append(f"{int(parts[0]):02d}_{parts[1]}")
        patterns.append(f"{int(parts[0])}_{int(parts[1]):02d}")
        patterns.append(f"{int(parts[0]):02d}_{int(parts[1]):02d}")

    valid_exts = {".tif", ".tiff", ".png", ".jpg"}
    for item in search_dir.iterdir():
        if item.is_file() and item.suffix.lower() in valid_exts:
            fname = item.name.lower()
            keyword_match = (keyword.lower() in fname) if keyword else True
            for pat in patterns:
                if pat.lower() in fname and keyword_match:
                    return item

    return None


def prepare_dataset(
    raw_dir: Union[str, Path],
    output_dir: Union[str, Path],
    patch_size: int = 512,
    stride: int = 256,
    config_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Executes the complete ISPRS Potsdam raw data extraction, channel separation,
    patch generation, and dataset split organization.

    Args:
        raw_dir: Path to directory containing raw Potsdam downloads.
        output_dir: Destination directory where processed patches will be stored.
        patch_size: Patch spatial resolution in pixels (default: 512).
        stride: Sliding window stride for patch extraction (default: 256).
        config_path: Optional path to configs/config.yaml for train/val/test splits.
    """
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)

    # Load split IDs from config if provided
    train_ids = DEFAULT_TRAIN_IDS
    val_ids = DEFAULT_VAL_IDS
    test_ids = DEFAULT_TEST_IDS

    if config_path is not None and Path(config_path).is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            dataset_cfg = cfg.get("dataset", {})
            train_ids = dataset_cfg.get("train_ids", train_ids)
            val_ids = dataset_cfg.get("val_ids", val_ids)
            test_ids = dataset_cfg.get("test_ids", test_ids)
            patch_size = dataset_cfg.get("patch_size", patch_size)
            stride = dataset_cfg.get("stride", stride)

    # Identify candidate subdirectories in raw_dir
    rgb_dirs = [raw_path / "2_Ortho_RGB", raw_path / "RGB", raw_path]
    irrg_dirs = [raw_path / "3_Ortho_IRRG", raw_path / "IRRG", raw_path]
    label_dirs = [
        raw_path / "5_Labels_all",
        raw_path / "5_Labels_for_participants",
        raw_path / "Labels",
        raw_path,
    ]

    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    print(f"Preparing Potsdam dataset from: {raw_path}")
    print(f"Destination patch directory:    {out_path}")
    print(f"Patch size: {patch_size}x{patch_size}, Stride: {stride}")

    total_patches = {"train": 0, "val": 0, "test": 0}

    for split_name, tile_list in splits.items():
        print(f"\nProcessing split: {split_name.upper()} ({len(tile_list)} tiles)...")

        out_rgb_dir = out_path / "rgb" / split_name
        out_nir_dir = out_path / "nir" / split_name
        out_lbl_dir = out_path / "labels" / split_name

        out_rgb_dir.mkdir(parents=True, exist_ok=True)
        out_nir_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        for tile_id in tqdm(tile_list, desc=f"Tiles ({split_name})"):
            # 1. Locate RGB file
            rgb_file = None
            for d in rgb_dirs:
                rgb_file = _find_tile_file(d, tile_id, "rgb")
                if rgb_file:
                    break
            if not rgb_file:
                print(f"Warning: RGB tile {tile_id} not found in {raw_path}. Skipping.")
                continue

            # 2. Locate IRRG file (for NIR channel)
            irrg_file = None
            for d in irrg_dirs:
                irrg_file = _find_tile_file(d, tile_id, "irrg")
                if irrg_file:
                    break
            if not irrg_file:
                print(f"Warning: IRRG tile {tile_id} not found in {raw_path}. Skipping.")
                continue

            # 3. Locate Label file
            label_file = None
            for d in label_dirs:
                label_file = _find_tile_file(d, tile_id, "label")
                if label_file:
                    break
            if not label_file:
                print(f"Warning: Label tile {tile_id} not found in {raw_path}. Skipping.")
                continue

            # Read RGB imagery
            if HAS_CV2 and cv2 is not None:
                rgb_img = cv2.cvtColor(cv2.imread(str(rgb_file), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            else:
                rgb_img = np.array(Image.open(rgb_file).convert("RGB"), dtype=np.uint8)

            # Read and extract NIR band from IRRG
            nir_img = extract_nir_from_irrg(irrg_file)

            # Convert RGB labels to class index mask
            mask_img = process_labels(label_file)

            # Assert spatial dimensions match
            if rgb_img.shape[:2] != nir_img.shape[:2] or rgb_img.shape[:2] != mask_img.shape[:2]:
                raise ValueError(
                    f"Dimension mismatch for tile {tile_id}: "
                    f"RGB={rgb_img.shape[:2]}, NIR={nir_img.shape[:2]}, Mask={mask_img.shape[:2]}"
                )

            # Crop and save patches
            tile_patch_count = 0
            for (y, x, rgb_patch), (_, _, nir_patch), (_, _, mask_patch) in zip(
                create_patches(rgb_img, patch_size=patch_size, stride=stride),
                create_patches(nir_img, patch_size=patch_size, stride=stride),
                create_patches(mask_img, patch_size=patch_size, stride=stride),
            ):
                patch_id = f"top_potsdam_{tile_id}_y{y}_x{x}.png"

                # Save RGB (H, W, 3)
                Image.fromarray(rgb_patch).save(out_rgb_dir / patch_id)

                # Save NIR (H, W) grayscale
                Image.fromarray(nir_patch).save(out_nir_dir / patch_id)

                # Save Mask (H, W) class indices 0..5
                Image.fromarray(mask_patch).save(out_lbl_dir / patch_id)

                tile_patch_count += 1

            total_patches[split_name] += tile_patch_count

    print("\nDataset preparation completed successfully!")
    print(f"  Train patches: {total_patches['train']}")
    print(f"  Val patches:   {total_patches['val']}")
    print(f"  Test patches:  {total_patches['test']}")


def generate_synthetic_dataset(
    output_dir: Union[str, Path],
    num_samples_per_split: int = 16,
    patch_size: int = 512,
) -> None:
    """
    Generates synthetic dummy patches for rapid verification of the ASG-Net pipeline.
    """
    out_path = Path(output_dir)
    print(f"Generating synthetic dataset in: {out_path}")
    splits = ["train", "val", "test"]

    for split in splits:
        rgb_dir = out_path / "rgb" / split
        nir_dir = out_path / "nir" / split
        lbl_dir = out_path / "labels" / split

        rgb_dir.mkdir(parents=True, exist_ok=True)
        nir_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_samples_per_split):
            fname = f"synthetic_potsdam_{split}_{i:04d}.png"

            # Synthetic RGB (H, W, 3)
            rgb = np.random.randint(0, 256, (patch_size, patch_size, 3), dtype=np.uint8)
            Image.fromarray(rgb).save(rgb_dir / fname)

            # Synthetic NIR (H, W)
            nir = np.random.randint(0, 256, (patch_size, patch_size), dtype=np.uint8)
            Image.fromarray(nir).save(nir_dir / fname)

            # Synthetic Labels (H, W) in {0, 1, 2, 3, 4, 5}
            mask = np.random.randint(0, 6, (patch_size, patch_size), dtype=np.uint8)
            Image.fromarray(mask).save(lbl_dir / fname)

    print(f"Generated {num_samples_per_split} synthetic patches per split in {out_path}")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="ISPRS Potsdam Benchmark Dataset Download and Preprocessing Pipeline"
    )
    parser.add_argument(
        "--instructions",
        action="store_true",
        help="Print manual download instructions for ISPRS Potsdam dataset.",
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=None,
        help="Path to directory containing raw extracted Potsdam archives.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/potsdam",
        help="Path where processed 512x512 patches will be written (default: data/potsdam).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to project config.yaml containing split IDs (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=512,
        help="Patch resolution height and width in pixels (default: 512).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=256,
        help="Stride for overlapping patch extraction (default: 256).",
    )
    parser.add_argument(
        "--generate_synthetic",
        action="store_true",
        help="Generate synthetic dummy patches for immediate end-to-end pipeline verification.",
    )
    parser.add_argument(
        "--fetch_real",
        action="store_true",
        help="Automatically download and preprocess real ISPRS Potsdam patches from benchmark repository.",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=200,
        help="Number of real training patches to fetch when using --fetch_real (default: 200).",
    )
    parser.add_argument(
        "--num_val",
        type=int,
        default=50,
        help="Number of real validation patches to fetch when using --fetch_real (default: 50).",
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=50,
        help="Number of real test patches to fetch when using --fetch_real (default: 50).",
    )

    args = parser.parse_args()

    if args.instructions:
        print_download_instructions()
        sys.exit(0)

    if args.fetch_real:
        from data.fetch_real_potsdam import main as fetch_main
        sys.argv = [
            sys.argv[0],
            "--output_dir", str(args.output_dir),
            "--num_train", str(args.num_train),
            "--num_val", str(args.num_val),
            "--num_test", str(args.num_test),
            "--overwrite",
        ]
        fetch_main()
        sys.exit(0)

    if args.generate_synthetic:
        generate_synthetic_dataset(output_dir=args.output_dir, patch_size=args.patch_size)
        sys.exit(0)

    if args.raw_dir is None:
        print_download_instructions()
        print("\n[NOTE] No --raw_dir specified. To fetch real benchmark data automatically, run:")
        print("       python data/download_potsdam.py --fetch_real")
        print("\n       Or to process manually downloaded tiles, run:")
        print("       python data/download_potsdam.py --raw_dir /path/to/raw --output_dir data/potsdam")
        print("\n       Or to test the pipeline with synthetic patches, run:")
        print("       python data/download_potsdam.py --generate_synthetic\n")
        sys.exit(0)

    prepare_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
