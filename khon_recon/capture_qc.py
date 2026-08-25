"""Capture quality control.

Runs before SfM and answers one question: *is this photo set good enough to
reconstruct, or should we reshoot?* The museum session is the one
irreplaceable resource in this project, so the checks here are deliberately
run on trial shots first.

Checks map directly onto the requirements the proposal states:
  - sharp images                  -> variance of Laplacian
  - fixed focus and exposure      -> luminance drift + EXIF consistency
  - 60-70% overlap between shots  -> consecutive-pair inlier match counts
  - plain, non-reflective backdrop-> static-background (turntable) detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import QCConfig
from .io_utils import get_logger, list_images, read_exif

log = get_logger(__name__)

# Working resolution for QC. Small enough to be fast over 100 images, large
# enough that blur and match counts stay meaningful.
_QC_MAX_DIM = 900


def _load_gray(path: Path, max_dim: int = _QC_MAX_DIM) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return _resize_max(img, max_dim)


def _resize_max(img: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------


def blur_scores(paths: list[Path]) -> dict[str, float]:
    """Variance of the Laplacian per image; higher is sharper."""
    scores: dict[str, float] = {}
    for path in paths:
        gray = _load_gray(path)
        if gray is None:
            log.warning("unreadable image: %s", path.name)
            continue
        scores[path.name] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return scores


def exposure_scores(paths: list[Path]) -> dict[str, float]:
    """Mean luminance per image, used to detect exposure drift."""
    scores: dict[str, float] = {}
    for path in paths:
        gray = _load_gray(path)
        if gray is None:
            continue
        scores[path.name] = float(gray.mean())
    return scores


def exif_consistency(paths: list[Path]) -> dict[str, Any]:
    """Report whether exposure/ISO/focal length actually stayed fixed.

    The proposal requires fixed focus and exposure because inconsistent
    exposure introduces texture-blending artifacts in the final mesh. A phone
    left on auto will silently violate this.
    """
    fields = ("ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength")
    observed: dict[str, set] = {f: set() for f in fields}
    models: set[str] = set()
    have_exif = 0

    for path in paths:
        exif = read_exif(path)
        if not exif:
            continue
        if any(f in exif for f in fields):
            have_exif += 1
        for f in fields:
            value = exif.get(f)
            if value is not None:
                try:
                    observed[f].add(round(float(value), 4))
                except (TypeError, ValueError):
                    observed[f].add(str(value))
        if "Model" in exif:
            models.add(str(exif["Model"]))

    return {
        "images_with_exif": have_exif,
        "camera_models": sorted(models),
        "distinct_values": {f: sorted(v)[:10] for f, v in observed.items()},
        "varying_fields": sorted(f for f, v in observed.items() if len(v) > 1),
        "multiple_cameras": len(models) > 1,
    }


def overlap_probe(
    paths: list[Path], n_features: int, min_matches: int
) -> dict[str, Any]:
    """Estimate overlap between consecutive frames via inlier match counts.

    ORB rather than SIFT: this is a fast screening check over the whole set,
    not the actual reconstruction matcher. The absolute counts matter less
    than spotting the *gaps* -- the pairs where the photographer stepped too
    far between shots and broke the chain.
    """
    orb = cv2.ORB_create(nfeatures=n_features)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    prev_kp = prev_des = None
    prev_name = ""
    pairs: list[dict[str, Any]] = []

    for path in paths:
        gray = _load_gray(path)
        if gray is None:
            prev_kp = prev_des = None
            continue
        kp, des = orb.detectAndCompute(gray, None)

        if prev_des is not None and des is not None and len(kp) >= 8 and len(prev_kp) >= 8:
            inliers = _inlier_count(prev_kp, prev_des, kp, des, matcher)
            pairs.append(
                {
                    "from": prev_name,
                    "to": path.name,
                    "inliers": inliers,
                    "weak": inliers < min_matches,
                }
            )
        prev_kp, prev_des, prev_name = kp, des, path.name

    counts = [p["inliers"] for p in pairs]
    return {
        "pairs": pairs,
        "median_inliers": float(np.median(counts)) if counts else 0.0,
        "min_inliers": int(min(counts)) if counts else 0,
        "weak_pairs": [p for p in pairs if p["weak"]],
    }


def _inlier_count(kp1, des1, kp2, des2, matcher) -> int:
    """Lowe-ratio matches confirmed by a fundamental-matrix RANSAC."""
    try:
        knn = matcher.knnMatch(des1, des2, k=2)
    except cv2.error:
        return 0
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return len(good)
    src = np.float32([kp1[m.queryIdx].pt for m in good])
    dst = np.float32([kp2[m.trainIdx].pt for m in good])
    _, mask = cv2.findFundamentalMat(src, dst, cv2.FM_RANSAC, 3.0, 0.99)
    return int(mask.sum()) if mask is not None else len(good)


def static_background_probe(paths: list[Path], sample: int = 24) -> dict[str, Any]:
    """Detect the turntable regime: a background that never moves.

    With a rotating object and a fixed camera, the background is the rigid
    scene and COLMAP will reconstruct *it*, not the mask. Pixels whose value
    barely changes across the set are static; a large static fraction means
    masking is required rather than optional.
    """
    if len(paths) < 4:
        return {"static_fraction": 0.0, "assessed": False}

    idx = np.linspace(0, len(paths) - 1, min(sample, len(paths))).astype(int)
    stack = []
    for i in idx:
        gray = _load_gray(paths[int(i)], max_dim=400)
        if gray is not None:
            stack.append(gray.astype(np.float32))
    if len(stack) < 4:
        return {"static_fraction": 0.0, "assessed": False}

    shapes = {s.shape for s in stack}
    if len(shapes) > 1:  # mixed orientations/aspect ratios
        return {"static_fraction": 0.0, "assessed": False, "note": "mixed image shapes"}

    temporal_std = np.stack(stack, axis=0).std(axis=0)
    static = temporal_std < 6.0  # near-constant pixel across the whole set
    fraction = float(static.mean())

    # Where is the static region? A static *border* with a changing centre is
    # the signature of a turntable; static everywhere means the camera never
    # moved at all.
    h, w = static.shape
    border = np.ones_like(static, dtype=bool)
    border[h // 5 : 4 * h // 5, w // 5 : 4 * w // 5] = False
    return {
        "static_fraction": fraction,
        "static_border_fraction": float(static[border].mean()),
        "static_centre_fraction": float(static[~border].mean()),
        "assessed": True,
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


@dataclass
class QCReport:
    n_images: int
    blur: dict[str, float] = field(default_factory=dict)
    exposure: dict[str, float] = field(default_factory=dict)
    exif: dict[str, Any] = field(default_factory=dict)
    overlap: dict[str, Any] = field(default_factory=dict)
    background: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    blurry_images: list[str] = field(default_factory=list)
    exposure_drift: float = 0.0
    needs_masking: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_images": self.n_images,
            "blurry_images": self.blurry_images,
            "exposure_drift": self.exposure_drift,
            "needs_masking": self.needs_masking,
            "warnings": self.warnings,
            "blur": self.blur,
            "exposure": self.exposure,
            "exif": self.exif,
            "overlap": self.overlap,
            "background": self.background,
        }

    def summary(self) -> str:
        lines = [
            "=" * 68,
            f"CAPTURE QC  --  {self.n_images} images",
            "=" * 68,
        ]
        if self.blur:
            values = np.array(list(self.blur.values()))
            lines.append(
                f"sharpness (var of Laplacian): median {np.median(values):8.1f}  "
                f"min {values.min():8.1f}"
            )
        lines.append(f"exposure drift (luminance range): {self.exposure_drift:.1f}")
        if self.overlap:
            lines.append(
                f"consecutive-pair overlap: median {self.overlap['median_inliers']:.0f} "
                f"inliers, weakest {self.overlap['min_inliers']}"
            )
        if self.background.get("assessed"):
            lines.append(
                f"static background: {self.background['static_fraction']:.0%} of frame "
                f"(border {self.background['static_border_fraction']:.0%})"
            )
        lines.append("-" * 68)
        if self.warnings:
            for warning in self.warnings:
                lines.append(f"  [!] {warning}")
        else:
            lines.append("  no issues detected -- capture looks usable")
        lines.append("=" * 68)
        return "\n".join(lines)


def run_qc(images_dir: Path, cfg: QCConfig) -> QCReport:
    """Run every capture check and collect the warnings."""
    paths = list_images(images_dir)
    report = QCReport(n_images=len(paths))
    if not paths:
        report.warnings.append(f"no images found in {images_dir}")
        return report

    log.info("QC: scoring sharpness and exposure over %d images", len(paths))
    report.blur = blur_scores(paths)
    report.exposure = exposure_scores(paths)

    # --- sharpness ---
    if report.blur:
        values = np.array(list(report.blur.values()))
        names = list(report.blur)
        cutoff = float(np.quantile(values, cfg.blur_bottom_fraction))
        soft = {
            name
            for name, value in zip(names, values)
            if value <= cutoff or value < cfg.blur_abs_min
        }
        report.blurry_images = sorted(soft)
        hard_fails = [n for n in soft if report.blur[n] < cfg.blur_abs_min]
        if hard_fails:
            report.warnings.append(
                f"{len(hard_fails)} image(s) below the absolute sharpness floor "
                f"({cfg.blur_abs_min}); consider removing: {hard_fails[:5]}"
            )

    # --- exposure ---
    if report.exposure:
        values = np.array(list(report.exposure.values()))
        report.exposure_drift = float(values.max() - values.min())
        if report.exposure_drift > cfg.exposure_max_drift:
            report.warnings.append(
                f"luminance varies by {report.exposure_drift:.0f} levels across the "
                "set -- exposure was probably not locked, which causes texture "
                "blending artifacts"
            )

    # --- EXIF ---
    report.exif = exif_consistency(paths)
    if report.exif["varying_fields"]:
        report.warnings.append(
            "EXIF shows these settings changed mid-capture: "
            f"{report.exif['varying_fields']} -- lock focus/exposure/ISO"
        )
    if report.exif["multiple_cameras"]:
        report.warnings.append(
            f"images come from multiple cameras {report.exif['camera_models']}; "
            "set sfm.single_camera=false"
        )

    # --- overlap ---
    log.info("QC: probing consecutive-frame overlap")
    report.overlap = overlap_probe(paths, cfg.overlap_features, cfg.min_pair_matches)
    weak = report.overlap["weak_pairs"]
    if weak:
        report.warnings.append(
            f"{len(weak)} consecutive pair(s) below {cfg.min_pair_matches} inlier "
            f"matches -- insufficient overlap, e.g. {weak[0]['from']} -> "
            f"{weak[0]['to']} ({weak[0]['inliers']}). Shoot more densely here."
        )

    # --- background ---
    log.info("QC: checking for a static background")
    report.background = static_background_probe(paths)
    if report.background.get("assessed"):
        if report.background["static_border_fraction"] > cfg.static_bg_ratio:
            report.needs_masking = True
            report.warnings.append(
                "the background is static while the subject moves (turntable "
                "regime). COLMAP will register the BACKGROUND instead of the "
                "mask unless foreground masks are used -- keep mask.enabled=true"
            )

    return report
