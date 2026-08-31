"""Image ingest: folders, video frames, and the COLMAP sample set.

The mask photographs do not exist yet, so the pipeline has to be developed and
proven against stand-in data. Three inputs are supported:

  folder  -- a directory of photographs (the real capture, eventually)
  video   -- a phone video, sampled to frames with ffmpeg. This is the closest
             proxy to the planned orbit capture and the best way to rehearse
             the shoot.
  sample  -- COLMAP's own south-building set, for a deterministic smoke test
             that exercises the SfM code path with no capture risk.
"""

from __future__ import annotations

import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import Config
from .io_utils import get_logger, list_images, timed

log = get_logger(__name__)

# COLMAP ships its demo datasets as GitHub release assets. The old
# demuc.de/colmap/datasets/*.zip direct links now 404; the index page points here.
SAMPLE_DATASETS = {
    "south-building": (
        "https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip"
    ),
    "gerrard-hall": (
        "https://github.com/colmap/colmap/releases/download/3.11.1/gerrard-hall.zip"
    ),
}


def fetch_sample(name: str, dest_root: Path, limit: int = 0) -> Path:
    """Download and unpack a COLMAP sample dataset; return its image directory."""
    if name not in SAMPLE_DATASETS:
        raise ValueError(f"unknown sample {name!r}; choose from {sorted(SAMPLE_DATASETS)}")

    dest_root.mkdir(parents=True, exist_ok=True)
    archive = dest_root / f"{name}.zip"
    extracted = dest_root / name

    if not extracted.is_dir():
        if not archive.exists():
            url = SAMPLE_DATASETS[name]
            with timed(f"downloading {name} (~100-400 MB)", log):
                urllib.request.urlretrieve(url, archive)
        with timed(f"unpacking {name}", log):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(dest_root)
        archive.unlink(missing_ok=True)

    # The archives nest the images one or two levels down.
    candidates = [p for p in extracted.rglob("images") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no images/ directory inside {extracted}")
    images_dir = candidates[0]

    if limit:
        keep = list_images(images_dir)[:limit]
        subset = extracted / "images_subset"
        subset.mkdir(exist_ok=True)
        for path in keep:
            target = subset / path.name
            if not target.exists():
                shutil.copy2(path, target)
        log.info("using a %d-image subset for the smoke test", len(keep))
        return subset

    return images_dir


def frames_from_video(video: Path, out_dir: Path, fps: float, overwrite: bool = False) -> int:
    """Sample a video into numbered JPEG frames with ffmpeg."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; install it with `brew install ffmpeg`")
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list_images(out_dir)
    if existing and not overwrite:
        log.info("reusing %d frames already in %s", len(existing), out_dir)
        return len(existing)
    for path in existing:
        path.unlink()

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video),
        "-vf", f"fps={fps}",
        "-q:v", "2",  # high-quality JPEG; compression artifacts hurt matching
        str(out_dir / "frame_%04d.jpg"),
    ]
    with timed(f"extracting frames at {fps} fps", log):
        result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-1000:]}")

    count = len(list_images(out_dir))
    log.info("extracted %d frames from %s", count, video.name)
    return count


def _exif_orientation(path: Path) -> int | None:
    """EXIF orientation tag, or None when absent. 1 means upright."""
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.getexif().get(274)
    except Exception:
        return None


def _even_subset(paths: list[Path], max_images: int) -> list[Path]:
    """Evenly spaced subset, preserving angular coverage.

    Evenly spaced rather than truncated: taking the first N frames of an orbit
    would cover only part of the object.
    """
    if max_images <= 0 or len(paths) <= max_images:
        return paths
    idx = np.linspace(0, len(paths) - 1, max_images).round().astype(int)
    return [paths[int(i)] for i in dict.fromkeys(idx)]


def ingest_images(
    source: Path, cfg: Config, overwrite: bool = False
) -> dict[str, Any]:
    """Copy/downscale source images into the subject's image directory."""
    paths = list_images(source)
    if not paths:
        raise FileNotFoundError(f"no images found in {source}")

    dest = cfg.images_dir
    dest.mkdir(parents=True, exist_ok=True)

    # Photographs already sitting in the subject directory are the originals,
    # and they are irreplaceable. Wiping the destination here would delete the
    # very files we are about to read, so an in-place ingest is a no-op rather
    # than a destructive rewrite.
    in_place = source.resolve() == dest.resolve()
    if in_place:
        if overwrite:
            log.warning(
                "--overwrite ignored: %s is already the subject directory, and "
                "clearing it would destroy the source photographs",
                source,
            )
        # Skipping the ingest also skips EXIF-orientation normalisation, which
        # matters more than it sounds: COLMAP on macOS records a rotated image
        # at its stored size while COLMAP on Linux applies the rotation, so a
        # model built here fails on Colab with
        #   Check failed: distorted_camera.Width() == distorted_bitmap.Width()
        # Ingesting from a separate directory rewrites the pixels upright and
        # drops the tag, so every reader agrees.
        rotated = sum(1 for p in paths[:20] if _exif_orientation(p) not in (None, 1))
        if rotated:
            log.warning(
                "%d of the first %d images carry an EXIF rotation tag and are NOT "
                "being normalised, because the source is already the subject "
                "directory. Move them elsewhere and re-ingest, or the dense stage "
                "will fail on a differently-behaving COLMAP build.",
                rotated, min(20, len(paths)),
            )
        log.info("images are already in place (%d files); nothing to ingest", len(paths))
        return {
            "source": str(source),
            "n_source_images": len(paths),
            "n_ingested": len(paths),
            "n_resized": 0,
            "max_dim": cfg.prepare.max_dim,
            "images_dir": str(dest),
            "in_place": True,
        }

    if overwrite:
        for path in list_images(dest):
            path.unlink()

    selected = _even_subset(paths, cfg.prepare.max_images)
    max_dim = cfg.prepare.max_dim
    resized = 0

    with timed(f"ingesting {len(selected)} images into {dest}", log):
        for path in selected:
            target = dest / path.name
            if target.exists() and not overwrite:
                continue
            # Pillow rather than OpenCV, for two reasons that both matter:
            #
            #   * exif_transpose bakes the rotation into the pixels and drops the
            #     orientation tag, so every reader reports the same dimensions.
            #     COLMAP builds disagree otherwise -- macOS records a rotated
            #     image at its stored size while Linux applies the rotation, and
            #     the Colab undistorter then aborts on a width mismatch.
            #   * the remaining EXIF is carried over. cv2.imwrite discards it,
            #     and losing FocalLength stops COLMAP grouping frames by lens:
            #     on this capture that turned 2 cameras into 49, one per image,
            #     each with unconstrained intrinsics.
            try:
                with Image.open(path) as raw:
                    image = ImageOps.exif_transpose(raw)
                    exif_bytes = image.info.get("exif")
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    w, h = image.size
                    if max_dim > 0 and max(w, h) > max_dim:
                        scale = max_dim / max(w, h)
                        image = image.resize(
                            (int(round(w * scale)), int(round(h * scale))),
                            Image.LANCZOS,
                        )
                        resized += 1
                    # Always JPEG so downstream naming is predictable, at high
                    # quality: blocking artifacts degrade feature matching.
                    target = target.with_suffix(".jpg")
                    save_kwargs = {"quality": cfg.prepare.jpeg_quality, "subsampling": 0}
                    if exif_bytes:
                        save_kwargs["exif"] = exif_bytes
                    image.save(target, "JPEG", **save_kwargs)
            except Exception as exc:
                log.warning("skipping unreadable file %s (%s)", path.name, exc)
                continue

    final = list_images(dest)
    if len(final) < 20:
        log.warning(
            "only %d images -- the proposal calls for 60-100 for full coverage",
            len(final),
        )
    return {
        "source": str(source),
        "n_source_images": len(paths),
        "n_ingested": len(final),
        "n_resized": resized,
        "max_dim": max_dim,
        "images_dir": str(dest),
    }
