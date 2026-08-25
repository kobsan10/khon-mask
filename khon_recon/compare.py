"""Novel-view comparison: rendered mesh against the real photograph.

Without a ground-truth 3D scan, this is the closest thing to a direct accuracy
measurement the project has. Two choices make it meaningful rather than
decorative:

*Masked comparison.* Scores are computed only inside the rendered silhouette.
A Khon mask occupies a minority of the frame against a plain backdrop, so
scoring the whole image would mostly measure how well we reproduce an empty
background -- inflating PSNR while saying nothing about the mask.

*Held-out views.* The views compared here were withheld from mapping (see
``sfm.split_holdout``), so the geometry was never fitted to them. Comparing
against training views measures self-consistency; comparing against held-out
views measures reconstruction quality.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .io_utils import get_logger

log = get_logger(__name__)


def masked_psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """PSNR over the masked pixels only (8-bit inputs)."""
    if mask.sum() == 0:
        return float("nan")
    diff = a[mask].astype(np.float64) - b[mask].astype(np.float64)
    mse = float((diff**2).mean())
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10((255.0**2) / mse))


def masked_ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Mean SSIM over the masked pixels.

    SSIM is computed densely and then averaged inside the mask, rather than on
    a cropped image: the local windows still need real neighbourhoods, and
    cropping to a bounding box would drag background pixels in anyway.
    """
    from skimage.metrics import structural_similarity

    if mask.sum() == 0:
        return float("nan")
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    _, ssim_map = structural_similarity(gray_a, gray_b, full=True, data_range=255)
    return float(ssim_map[mask].mean())


def compare_pair(
    render: np.ndarray, photo: np.ndarray, mask: np.ndarray, use_mask: bool = True
) -> dict[str, float]:
    """Score one render/photo pair."""
    if render.shape != photo.shape:
        photo = cv2.resize(
            photo, (render.shape[1], render.shape[0]), interpolation=cv2.INTER_AREA
        )
    effective = mask if use_mask else np.ones(mask.shape, dtype=bool)
    return {
        "psnr": masked_psnr(render, photo, effective),
        "ssim": masked_ssim(render, photo, effective),
        "mask_coverage": float(mask.mean()),
    }


def evaluate_views(
    mesh,
    reconstruction,
    images_dir: Path,
    out_dir: Path,
    holdout_names: list[str] | None,
    scale: float = 0.5,
    use_mask: bool = True,
    save_figures: int = 6,
) -> dict[str, Any]:
    """Render every registered view, score it, and save a few side-by-sides.

    Held-out and training views are scored separately: the gap between them is
    itself informative, since a large one means the reconstruction is fitting
    its training views rather than the object.
    """
    from .render import MeshRenderer, undistort_photo

    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = MeshRenderer(mesh)
    holdout = set(holdout_names or [])

    per_view: list[dict[str, Any]] = []
    saved = 0

    for image_id in sorted(reconstruction.reg_image_ids()):
        image = reconstruction.image(image_id)
        camera = reconstruction.camera(image.camera_id)

        photo = cv2.imread(str(images_dir / image.name), cv2.IMREAD_COLOR)
        if photo is None:
            continue
        photo = undistort_photo(photo, camera)

        rendered = renderer.render(camera, image, scale=scale)
        photo = cv2.resize(
            photo,
            (rendered.color.shape[1], rendered.color.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

        if rendered.mask.sum() < 100:  # nothing meaningful in view
            continue

        scores = compare_pair(rendered.color, photo, rendered.mask, use_mask)
        scores["image"] = image.name
        scores["is_holdout"] = image.name in holdout
        per_view.append(scores)

        if saved < save_figures and (not holdout or image.name in holdout):
            _save_side_by_side(
                rendered.color, photo, rendered.mask,
                out_dir / f"compare_{Path(image.name).stem}.jpg",
                title=f"{image.name}  PSNR {scores['psnr']:.1f} dB  SSIM {scores['ssim']:.3f}",
            )
            saved += 1

    return _summarise(per_view)


def _summarise(per_view: list[dict[str, Any]]) -> dict[str, Any]:
    def stats(rows: list[dict[str, Any]]) -> dict[str, float]:
        if not rows:
            return {}
        psnr = np.array([r["psnr"] for r in rows], dtype=float)
        ssim = np.array([r["ssim"] for r in rows], dtype=float)
        psnr = psnr[np.isfinite(psnr)]
        ssim = ssim[np.isfinite(ssim)]
        return {
            "n": len(rows),
            "psnr_mean": float(psnr.mean()) if psnr.size else float("nan"),
            "psnr_std": float(psnr.std()) if psnr.size else float("nan"),
            "ssim_mean": float(ssim.mean()) if ssim.size else float("nan"),
            "ssim_std": float(ssim.std()) if ssim.size else float("nan"),
        }

    holdout_rows = [r for r in per_view if r["is_holdout"]]
    train_rows = [r for r in per_view if not r["is_holdout"]]

    summary = {
        "all": stats(per_view),
        "holdout": stats(holdout_rows),
        "train": stats(train_rows),
        "per_view": per_view,
    }
    if holdout_rows and train_rows:
        gap = summary["train"]["psnr_mean"] - summary["holdout"]["psnr_mean"]
        summary["train_holdout_psnr_gap"] = float(gap)
        log.info(
            "novel-view PSNR: held-out %.2f dB vs training %.2f dB (gap %.2f dB)",
            summary["holdout"]["psnr_mean"], summary["train"]["psnr_mean"], gap,
        )
    return summary


def _save_side_by_side(
    render: np.ndarray, photo: np.ndarray, mask: np.ndarray, path: Path, title: str
) -> None:
    """Photo | render | error map, for the report's qualitative figure."""
    error = cv2.absdiff(render, photo)
    error = cv2.applyColorMap(
        cv2.cvtColor(error, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_INFERNO
    )
    error[~mask] = (30, 30, 30)  # dim outside the silhouette: not scored

    panel = np.hstack([photo, render, error])
    labelled = np.full((panel.shape[0] + 34, panel.shape[1], 3), 255, np.uint8)
    labelled[34:] = panel
    cv2.putText(labelled, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1,
                cv2.LINE_AA)
    third = panel.shape[1] // 3
    for i, name in enumerate(("photograph", "reconstruction", "error")):
        cv2.putText(labelled, name, (8 + i * third, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), labelled)
