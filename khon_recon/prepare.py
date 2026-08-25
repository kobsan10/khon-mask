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
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                log.warning("skipping unreadable file: %s", path.name)
                continue
            h, w = image.shape[:2]
            if max_dim > 0 and max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                image = cv2.resize(
                    image,
                    (int(round(w * scale)), int(round(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
                resized += 1
            # Always write JPEG so downstream naming is predictable, and keep
            # quality high: JPEG blocking artifacts degrade feature matching.
            target = target.with_suffix(".jpg")
            cv2.imwrite(
                str(target), image, [int(cv2.IMWRITE_JPEG_QUALITY), cfg.prepare.jpeg_quality]
            )

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
