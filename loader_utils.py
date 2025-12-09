# ============================================================
# loader_utils.py — Image loading helpers for SMOKE experiments
# ============================================================
"""
Utility functions to load image data for the SMOKE subset.

Main entry point:
    - load_images_smoke(paths, num_workers=8)

This is designed to be imported from your notebook, e.g.:

    from loader_utils import load_images_smoke
"""

from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Iterable, List, Tuple, Optional

from PIL import Image


# -----------------------------
# Internal helpers (top-level!)
# -----------------------------
def _read_one(path_str: str) -> Tuple[str, Image.Image]:
    """
    Read a single image from disk and convert it to RGB.

    Args:
        path_str: String path to the image.

    Returns:
        (path_str, PIL.Image.Image)
    """
    p = Path(path_str)
    img = Image.open(p).convert("RGB")
    return path_str, img


def _read_one_safe(path_str: str) -> Optional[Tuple[str, Image.Image]]:
    """
    Safe wrapper around _read_one() that catches exceptions.

    Returns:
        (path_str, image) on success, or None if failed.
    """
    try:
        return _read_one(path_str)
    except Exception as e:
        print(f"[WARN] Failed to read {path_str}: {e}")
        return None


# --------------------------------------------------------
# Public API
# --------------------------------------------------------
def load_images_smoke(
    paths: Iterable[Path],
    num_workers: int = 8,
) -> List[Tuple[str, Image.Image]]:
    """
    CELL B — Load SMOKE images into memory

    Purpose:
        Load a list of image files into memory as (path, PIL.Image) tuples.
        Can run single-threaded or with multiprocessing.

    Args:
        paths:
            Iterable of pathlib.Path objects pointing to image files.
        num_workers:
            - 1 or <=0 → single-process (no multiprocessing)
            - >1       → use multiprocessing.Pool with that many workers
                         (capped at cpu_count()).

    Returns:
        List of (path_str, PIL.Image.Image) tuples.
    """
    path_strs = [str(p) for p in paths]

    if not path_strs:
        print("[SMOKE] No paths provided to load_images_smoke().")
        return []

    # Single-process path (easier for debugging)
    if num_workers is None or num_workers <= 1:
        print(f"[SMOKE] Loading {len(path_strs)} images in a single process...")
        out: List[Tuple[str, Image.Image]] = []
        for ps in path_strs:
            try:
                out.append(_read_one(ps))
            except Exception as e:
                print(f"[WARN] Failed to read {ps}: {e}")
        print(f"[SMOKE] Loaded {len(out)}/{len(path_strs)} images successfully.")
        return out

    # Multiprocessing path
    workers = min(num_workers, cpu_count())
    print(f"[SMOKE] Loading {len(path_strs)} images using multiprocessing ({workers} workers)...")

    with Pool(processes=workers) as pool:
        # chunksize can be tuned; 16 is usually reasonable
        results = pool.map(_read_one_safe, path_strs, chunksize=16)

    samples = [r for r in results if r is not None]
    print(f"[SMOKE] Loaded {len(samples)}/{len(path_strs)} images successfully.")

    return samples
