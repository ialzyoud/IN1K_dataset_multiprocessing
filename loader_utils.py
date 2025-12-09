"""
loader_utils.py

Utilities for loading images (e.g., SMOKE subset) with optional multiprocessing.
"""

from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

from PIL import Image


# ---------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------
def _read_one(path: Path) -> Tuple[Path, Image.Image]:
    """
    Read a single image from disk.

    Args:
        path: Path to an image file.

    Returns:
        (path, PIL.Image) tuple. If reading fails, raises the original exception.
    """
    img = Image.open(path).convert("RGB")
    return path, img


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def load_images_smoke(
    image_paths: List[Path],
    num_workers: int | None = None
) -> List[Tuple[Path, Image.Image]]:
    """
    Load a small SMOKE subset of images into memory.

    This is the main entry point used by the notebook.

    Args:
        image_paths:
            List of image file paths (typically a few thousand).
        num_workers:
            Number of worker processes for multiprocessing.
            If None or <= 1, runs single-process.
            If > 1, uses multiprocessing.Pool with that many workers,
            capped at the physical cpu_count().

    Returns:
        List of (path, PIL.Image) tuples in the same order as image_paths.
    """
    if not image_paths:
        return []

    if num_workers is None or num_workers <= 1:
        # --- single-process path (safe everywhere) ---
        out: List[Tuple[Path, Image.Image]] = []
        for p in image_paths:
            out.append(_read_one(p))
        return out

    # --- multiprocessing path ---
    n_workers = min(num_workers, cpu_count())
    print(f"[SMOKE] Loading {len(image_paths)} images using multiprocessing "
          f"with {n_workers} workers...")

    with Pool(processes=n_workers) as pool:
        # imap_unordered is usually faster; we re-order to match input order
        results_unordered = list(pool.imap_unordered(_read_one, image_paths))

    # Reorder results to match the original image_paths order
    path_to_img = {p: img for p, img in results_unordered}
    ordered = [(p, path_to_img[p]) for p in image_paths if p in path_to_img]
    return ordered
