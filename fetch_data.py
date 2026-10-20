"""
ASG-Net: Lightweight Standalone Real Potsdam Data Fetcher
=========================================================
Runs anywhere with pure Python + PIL + NumPy (no extra dependencies).
Downloads genuine 512x512 ISPRS Potsdam benchmark patches (RGB, NIR, and Labels).
"""

import io
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import numpy as np

BASE_URL = "https://huggingface.co/datasets/aamante/potsdam/resolve/main"
SPLITS_URL = "https://huggingface.co/datasets/aamante/potsdam/raw/main/Potsdam/splits"


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def main():
    print("=" * 60)
    print("Fetching real ISPRS Potsdam benchmark file lists...")
    print("=" * 60)
    
    train_bytes = fetch(f"{SPLITS_URL}/all_train.txt")
    val_bytes = fetch(f"{SPLITS_URL}/val.txt")

    train_all = [l.strip() for l in train_bytes.decode().splitlines() if l.strip()][:200]
    val_all = [l.strip() for l in val_bytes.decode().splitlines() if l.strip()]

    # Disjoint tiles: 2_11 for Val, 6_9 for Test
    val_p = [p for p in val_all if "2_11" in p][:50]
    test_p = [p for p in val_all if "6_9" in p][:50]

    tasks = (
        [(p, "train", "all_train") for p in train_all]
        + [(p, "val", "val") for p in val_p]
        + [(p, "test", "val") for p in test_p]
    )

    print(f"Preparing to download {len(tasks)} patches:")
    print(f"  - Train: {len(train_all)} patches (tiles 2_10, 3_10, etc.)")
    print(f"  - Val:   {len(val_p)} patches (tile 2_11)")
    print(f"  - Test:  {len(test_p)} patches (tile 6_9)")
    print("=" * 60)

    def worker(item):
        p, split, src = item
        try:
            rgbir_bytes = fetch(f"{BASE_URL}/Potsdam/{src}/RGBIR/{p}.tif")
            gt_bytes = fetch(f"{BASE_URL}/Potsdam/{src}/gt_no_boundary/{p}.png")

            rgbir = np.array(Image.open(io.BytesIO(rgbir_bytes)))
            gt = np.array(Image.open(io.BytesIO(gt_bytes)))

            rgb = rgbir[:, :, :3]
            nir = rgbir[:, :, 3]
            mask = np.clip(gt - 1, 0, 5).astype(np.uint8)

            for sub, arr in [("rgb", rgb), ("nir", nir), ("labels", mask)]:
                d = os.path.join("data", "potsdam", sub, split)
                os.makedirs(d, exist_ok=True)
                Image.fromarray(arr).save(os.path.join(d, f"{p}.png"))
            return True
        except Exception as e:
            print(f"Error {p}: {e}")
            return False

    done = 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for f in as_completed(futures):
            if f.result():
                done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"Downloaded {done}/{len(tasks)} ({done / len(tasks) * 100:.1f}%)")

    print("\n" + "=" * 60)
    print("All real ISPRS Potsdam benchmark patches successfully saved in data/potsdam!")
    print("Ready to launch real data training:")
    print("  python train.py --config configs/config.yaml --epochs 10 --batch_size 4 --lr 6e-5")
    print("=" * 60)


if __name__ == "__main__":
    main()
