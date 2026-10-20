"""
ASG-Net: Automated Real ISPRS Potsdam Benchmark Downloader & Preprocessor
========================================================================
Fetches genuine 512x512 multispectral aerial tiles from the ISPRS Potsdam benchmark,
extracts true RGB and physical Near-Infrared (NIR, Band 4) channels, maps semantic ground
truth to standard 6-class indices (0: Impervious, 1: Building, 2: Low Veg, 3: Tree, 4: Car, 5: Clutter),
and populates data/potsdam for end-to-end model training.

Source Repository:
  - Curated ISPRS Potsdam benchmark mirror (HuggingFace CDN / aamante/potsdam)
  - Pre-tiled 512x512 patches with synchronized 4-channel RGBIR GeoTIFFs and ground truth masks.
"""

import argparse
import os
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=""):
            self.iterable = iterable
            self.total = total or (len(iterable) if iterable else 0)
            self.desc = desc
            self.n = 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            print()
        def update(self, n=1):
            self.n += n
            pct = (self.n / self.total * 100) if self.total > 0 else 0
            sys.stdout.write(f"\r{self.desc}: {self.n}/{self.total} ({pct:.1f}%)")
            sys.stdout.flush()

BASE_URL = "https://huggingface.co/datasets/aamante/potsdam/resolve/main"
SPLITS_URL = "https://huggingface.co/datasets/aamante/potsdam/raw/main/Potsdam/splits"


def fetch_file_bytes(url: str, max_retries: int = 3) -> bytes:
    """Fetches binary data from a URL with retries."""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts")


def get_split_lists() -> Tuple[List[str], List[str], List[str]]:
    """
    Retrieves official patch filenames for disjoint train, val, and test partitions.
    Partitions the 576 evaluation patches across distinct aerial tiles:
      - Train: all_train (3312 patches from tiles 2_10, 3_10, 4_10, etc.)
      - Val: tiles 2_11 and 4_11 (288 patches)
      - Test: tiles 6_9 and 6_11 (288 patches)
    """
    train_bytes = fetch_file_bytes(f"{SPLITS_URL}/all_train.txt")
    val_bytes = fetch_file_bytes(f"{SPLITS_URL}/val.txt")

    train_patches = [line.strip() for line in train_bytes.decode().splitlines() if line.strip()]
    eval_patches = [line.strip() for line in val_bytes.decode().splitlines() if line.strip()]

    # Val tiles: 2_11, 4_11; Test tiles: 6_9, 6_11
    val_patches = [p for p in eval_patches if "2_11" in p or "4_11" in p]
    test_patches = [p for p in eval_patches if "6_9" in p or "6_11" in p]

    return train_patches, val_patches, test_patches


def download_and_process_patch(
    patch_id: str,
    split: str,
    source_folder: str,
    output_dir: Path,
) -> bool:
    """
    Downloads RGBIR TIFF and GT PNG directly to memory, extracts RGB + NIR channels,
    remaps ground truth labels (1..6 -> 0..5), and writes directly to disk.
    """
    tif_url = f"{BASE_URL}/Potsdam/{source_folder}/RGBIR/{patch_id}.tif"
    gt_url = f"{BASE_URL}/Potsdam/{source_folder}/gt_no_boundary/{patch_id}.png"

    try:
        tif_bytes = fetch_file_bytes(tif_url)
        gt_bytes = fetch_file_bytes(gt_url)

        # Open in memory via BytesIO
        import io
        rgbir_img = Image.open(io.BytesIO(tif_bytes))
        gt_img = Image.open(io.BytesIO(gt_bytes))

        rgbir = np.array(rgbir_img)
        gt = np.array(gt_img)

        if rgbir.ndim != 3 or rgbir.shape[2] != 4:
            return False

        # Extract RGB (channels 0, 1, 2) and NIR (channel 3)
        rgb = rgbir[:, :, :3]
        nir = rgbir[:, :, 3]

        # ISPRS class mapping: 1..6 -> 0..5
        mask = (gt - 1).astype(np.uint8)
        mask = np.clip(mask, 0, 5)

        # Save to target paths
        out_rgb = output_dir / "rgb" / split / f"{patch_id}.png"
        out_nir = output_dir / "nir" / split / f"{patch_id}.png"
        out_lbl = output_dir / "labels" / split / f"{patch_id}.png"

        Image.fromarray(rgb).save(out_rgb)
        Image.fromarray(nir).save(out_nir)
        Image.fromarray(mask).save(out_lbl)

        return True
    except Exception as e:
        print(f"Error processing {patch_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Fetch and preprocess real ISPRS Potsdam data")
    parser.add_argument("--output_dir", type=str, default="data/potsdam", help="Target dataset root directory")
    parser.add_argument("--num_train", type=int, default=200, help="Number of real training patches to fetch (default: 200)")
    parser.add_argument("--num_val", type=int, default=40, help="Number of real validation patches to fetch (default: 40)")
    parser.add_argument("--num_test", type=int, default=40, help="Number of real test patches to fetch (default: 40)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel download threads (default: 8)")
    parser.add_argument("--overwrite", action="store_true", help="Clear existing directory before downloading")
    args = parser.parse_args()

    out_path = Path(args.output_dir)

    print("=" * 70)
    print("ASG-Net: Fetching Real ISPRS Potsdam Multimodal Aerial Dataset")
    print("=" * 70)
    print(f"Destination:  {out_path.resolve()}")
    print(f"Train Patches: {args.num_train}")
    print(f"Val Patches:   {args.num_val}")
    print(f"Test Patches:  {args.num_test}")
    print(f"Workers:      {args.num_workers}")
    print("=" * 70)

    if args.overwrite and out_path.exists():
        print("Clearing existing data directory...")
        for sub in ["rgb", "nir", "labels"]:
            sub_p = out_path / sub
            if sub_p.exists():
                shutil.rmtree(sub_p)

    # Create destination directories
    for split in ["train", "val", "test"]:
        (out_path / "rgb" / split).mkdir(parents=True, exist_ok=True)
        (out_path / "nir" / split).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("Retrieving split file listings from benchmark repository...")
    train_all, val_all, test_all = get_split_lists()

    selected_train = train_all[: args.num_train]
    selected_val = val_all[: args.num_val]
    selected_test = test_all[: args.num_test]

    tasks = []
    for p in selected_train:
        tasks.append((p, "train", "all_train"))
    for p in selected_val:
        tasks.append((p, "val", "val"))
    for p in selected_test:
        tasks.append((p, "test", "val"))

    print(f"\nDownloading and converting {len(tasks)} patches with {args.num_workers} parallel threads...")
    success_count = 0

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                download_and_process_patch, patch_id, split, src, out_path
            ): patch_id
            for patch_id, split, src in tasks
        }

        with tqdm(total=len(tasks), desc="Fetching Real Potsdam Patches") as pbar:
            for fut in as_completed(futures):
                res = fut.result()
                if res:
                    success_count += 1
                pbar.update(1)

    print("\n" + "=" * 70)
    print(f"Download complete: {success_count}/{len(tasks)} real patches successfully saved!")
    print(f"  Train:  {len(selected_train)} patches in {out_path / 'rgb' / 'train'}")
    print(f"  Val:    {len(selected_val)} patches in {out_path / 'rgb' / 'val'}")
    print(f"  Test:   {len(selected_test)} patches in {out_path / 'rgb' / 'test'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
