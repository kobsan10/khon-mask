"""Foreground masks for object-centric reconstruction.

Why this module exists
----------------------
The proposal plans to rotate the mask on a turntable under fixed lighting.
That produces a photo set in which the *object* moves and the *background*
stays put. Structure from motion assumes a rigid scene and has no way to know
which of the two is the subject, so it locks onto the larger, better-textured
rigid thing -- the background -- and returns a confidently wrong reconstruction
of the mask. The failure is silent: the reprojection error can look excellent.

Feeding COLMAP a per-image foreground mask confines feature extraction to the
mask itself and removes the ambiguity. COLMAP expects, for an image named
``IMG_0001.jpg``, a file ``IMG_0001.jpg.png`` in the mask directory, where
black (0) pixels are ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import MaskConfig
from .io_utils import get_logger, list_images, mask_path_for

log = get_logger(__name__)


def _largest_component(binary: np.ndarray) -> np.ndarray:
    """Keep only the biggest blob.

    A Khon mask is one connected object; stray islands are background clutter
    or specular flare that the segmenter mistook for foreground.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def _postprocess(mask: np.ndarray, cfg: MaskConfig) -> np.ndarray:
    """Clean up, then erode slightly.

    Eroding rather than dilating is deliberate: a mask that leaks a few pixels
    of background reintroduces exactly the static features we are excluding,
    whereas losing a few pixels of silhouette costs almost nothing.
    """
    mask = (mask > 127).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = _largest_component(mask)
    # Fill interior holes: dark eye sockets and open mouths are part of the
    # object even when the segmenter disagrees.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        mask = filled
    if cfg.erode_px > 0:
        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * cfg.erode_px + 1, 2 * cfg.erode_px + 1)
        )
        mask = cv2.erode(mask, erode_kernel, iterations=1)
    return mask


def _mask_rembg(image_bgr: np.ndarray, session: Any) -> np.ndarray:
    """Segment with rembg's u2net and return the alpha channel."""
    from rembg import remove

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    out = remove(rgb, session=session, only_mask=True)
    if out.ndim == 3:
        out = out[..., -1]
    return out.astype(np.uint8)


def _mask_grabcut(image_bgr: np.ndarray, cfg: MaskConfig) -> np.ndarray:
    """OpenCV GrabCut fallback: no model download, assumes a centred subject."""
    h, w = image_bgr.shape[:2]
    mx, my = int(w * cfg.grabcut_margin), int(h * cfg.grabcut_margin)
    rect = (mx, my, max(w - 2 * mx, 1), max(h - 2 * my, 1))

    gc_mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            image_bgr, gc_mask, rect, bgd, fgd, cfg.grabcut_iterations,
            cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error as exc:
        log.warning("grabcut failed (%s); falling back to the full frame", exc)
        return np.full((h, w), 255, np.uint8)
    foreground = np.isin(gc_mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
    return (foreground.astype(np.uint8)) * 255


def generate_masks(
    images_dir: Path, masks_dir: Path, cfg: MaskConfig, overwrite: bool = False
) -> dict[str, Any]:
    """Write a COLMAP-format mask for every image.

    Returns per-image foreground area fractions plus the list of images whose
    mask looks implausible, so a bad segmentation is caught here rather than
    diagnosed later from a broken mesh.
    """
    paths = list_images(images_dir)
    if not paths:
        raise FileNotFoundError(f"no images in {images_dir}")
    masks_dir.mkdir(parents=True, exist_ok=True)

    session = None
    method = cfg.method
    if method == "rembg":
        try:
            from rembg import new_session

            session = new_session("u2net")
        except Exception as exc:
            log.warning("rembg unavailable (%s); falling back to grabcut", exc)
            method = "grabcut"

    areas: dict[str, float] = {}
    suspect: list[str] = []

    for i, path in enumerate(paths, 1):
        out_path = mask_path_for(path.name, masks_dir)
        if out_path.exists() and not overwrite:
            existing = cv2.imread(str(out_path), cv2.IMREAD_GRAYSCALE)
            if existing is not None:
                areas[path.name] = float((existing > 127).mean())
                continue

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            log.warning("unreadable image, skipping mask: %s", path.name)
            continue

        raw = (
            _mask_rembg(image, session)
            if method == "rembg"
            else _mask_grabcut(image, cfg)
        )
        if raw.shape[:2] != image.shape[:2]:
            raw = cv2.resize(raw, (image.shape[1], image.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        mask = _postprocess(raw, cfg)

        fraction = float((mask > 127).mean())
        areas[path.name] = fraction
        if not (cfg.min_area_fraction <= fraction <= cfg.max_area_fraction):
            suspect.append(path.name)

        cv2.imwrite(str(out_path), mask)
        if i % 20 == 0 or i == len(paths):
            log.info("masked %d/%d images", i, len(paths))

    if suspect:
        log.warning(
            "%d mask(s) cover an implausible image fraction (e.g. %s) -- inspect "
            "them before trusting the reconstruction",
            len(suspect), suspect[:5],
        )

    values = np.array(list(areas.values())) if areas else np.array([0.0])
    return {
        "method": method,
        "n_masks": len(areas),
        "mean_area_fraction": float(values.mean()),
        "min_area_fraction": float(values.min()),
        "max_area_fraction": float(values.max()),
        "suspect_masks": suspect,
        "areas": areas,
    }


def write_mask_previews(
    images_dir: Path, masks_dir: Path, out_dir: Path, limit: int = 8
) -> list[Path]:
    """Save a few image/mask overlays so masking can be eyeballed quickly."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = list_images(images_dir)
    if not paths:
        return []
    step = max(1, len(paths) // max(limit, 1))
    written: list[Path] = []
    for path in paths[::step][:limit]:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path_for(path.name, masks_dir)), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        tint = np.zeros_like(image)
        tint[..., 1] = 255  # green where the object is
        overlay = np.where(
            (mask > 127)[..., None],
            cv2.addWeighted(image, 0.75, tint, 0.25, 0),
            (image * 0.25).astype(np.uint8),
        )
        dest = out_dir / f"maskcheck_{path.stem}.jpg"
        cv2.imwrite(str(dest), overlay)
        written.append(dest)
    return written
