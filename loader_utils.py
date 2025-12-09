"""
# === Cell: SMOKE_IMAGE_LOADER ===============================================

Utility helpers to load a SMOKE subset of ImageNet images safely inside
a notebook / interactive environment.

This replaces the previous multiprocessing-based loader which failed with:
"AttributeError: Can't get attribute '_read_one' on <module '__main__' (built-in)>"
due to Python's pickling rules for spawned worker processes.
"""

from pathlib import Path
from typing import List, Union, Optional

from PIL import Image
from tqdm.auto import tqdm
from multiprocessing.dummy import Pool as ThreadPool  # thread-based pool


def _read_one(path: Union[str, Path]) -> Optional[Image.Image]:
    """
    Read a single image from disk and return a PIL.Image.Image instance.

    Returns:
        PIL.Image.Image if successful, or None if the image could not be loaded.
    """
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        # You can log or collect the failing paths here if you like
        # print(f"[WARN] Failed to read {path}: {e}")
        return None


def load_images_smoke(
    paths: List[Union[str, Path]],
    num_workers: int = 8,
):
    """
    Load a SMOKE subset of ImageNet images using a thread pool.

    This is notebook-safe and avoids multiprocessing pickling issues like:
    "AttributeError: Can't get attribute '_read_one' on <module '__main__' (built-in)>".

    Args:
        paths:
            List of image paths (str or Path) to load.
        num_workers:
            Number of threads to use. If <= 1, falls back to single-threaded.

    Returns:
        List of successfully loaded PIL.Image.Image objects.
        Images that fail to load are skipped.
    """
    n = len(paths)
    if n == 0:
        print("[SMOKE] No paths provided; nothing to load.")
        return []

    paths = [str(p) for p in paths]  # ensure simple pickleable/serializable types

    if num_workers is None or num_workers < 1:
        num_workers = 1

    print(f"[SMOKE] Loading {n} images using {num_workers} thread(s)...")

    if num_workers == 1:
        # Simple single-threaded fallback
        imgs = []
        for p in tqdm(paths, desc="[SMOKE] Loading (1 worker)"):
            img = _read_one(p)
            if img is not None:
                imgs.append(img)
    else:
        # Threaded loading (safe in notebooks / REPL)
        with ThreadPool(num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(_read_one, paths),
                    total=len(paths),
                    desc=f"[SMOKE] Loading ({num_workers} threads)",
                )
            )
        # Filter out failed reads
        imgs = [im for im in results if im is not None]

    print(f"[SMOKE] Loaded {len(imgs)} / {n} images successfully.")
    return imgs
