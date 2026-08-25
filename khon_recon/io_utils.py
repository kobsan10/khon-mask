"""Image discovery, EXIF inspection, logging and run manifests."""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from PIL import Image, ExifTags

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".JPG", ".JPEG"}

_LOG_FORMAT = "%(asctime)s  %(levelname)-7s %(name)s: %(message)s"


def setup_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=_LOG_FORMAT,
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def list_images(directory: Path) -> list[Path]:
    """Sorted image paths in ``directory``.

    Sorted order matters: the overlap probe and the hold-out split both assume
    filename order tracks capture order, which is true for camera-numbered files
    and for the zero-padded names written by the frame extractor.
    """
    if not directory.is_dir():
        return []
    files = [p for p in directory.iterdir() if p.suffix in IMAGE_SUFFIXES]
    return sorted(files, key=lambda p: p.name)


def mask_path_for(image_name: str, masks_dir: Path) -> Path:
    """COLMAP's mask naming convention: ``<image_name>.png``.

    Note the full original filename is kept, extension included, so
    ``IMG_0001.jpg`` maps to ``IMG_0001.jpg.png``.
    """
    return masks_dir / f"{image_name}.png"


def read_exif(path: Path) -> dict[str, Any]:
    """Extract the EXIF fields relevant to capture consistency.

    The proposal requires fixed focus and exposure across the set; this is how
    we check the camera actually honoured that instead of silently auto-adjusting.
    """
    out: dict[str, Any] = {}
    try:
        with Image.open(path) as img:
            out["width"], out["height"] = img.size
            exif = img.getexif()
            if not exif:
                return out
            tags = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}
            # Exposure/aperture/ISO live in the Exif IFD, not the top level.
            try:
                ifd = exif.get_ifd(0x8769)
                tags.update({ExifTags.TAGS.get(k, str(k)): v for k, v in ifd.items()})
            except Exception:
                pass
            for key in (
                "Make",
                "Model",
                "FNumber",
                "ExposureTime",
                "ISOSpeedRatings",
                "FocalLength",
                "FocalLengthIn35mmFilm",
                "LensModel",
                "WhiteBalance",
            ):
                if key in tags:
                    value = tags[key]
                    # EXIF rationals are not JSON-serialisable.
                    out[key] = float(value) if hasattr(value, "numerator") else value
    except Exception:  # corrupt or exotic file: absence of EXIF is not fatal
        pass
    return out


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default))
    return path


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):  # numpy scalars/arrays
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None  # not a git repo yet; the manifest stays valid without it


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for name in ("numpy", "cv2", "open3d", "pycolmap", "skimage", "trimesh"):
        try:
            mod = __import__(name)
            versions[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            versions[name] = "not installed"
    try:
        colmap = subprocess.run(
            ["colmap", "--help"], capture_output=True, text=True, timeout=15
        )
        first = (colmap.stdout or colmap.stderr).splitlines()
        if first:
            versions["colmap_cli"] = first[0].strip()
    except Exception:
        versions["colmap_cli"] = "not found"
    return versions


def update_manifest(run_dir: Path, stage: str, payload: dict[str, Any]) -> Path:
    """Append a stage record to the run manifest.

    The manifest is what lets a number in the report be traced back to the code
    and settings that produced it.
    """
    path = run_dir / "manifest.json"
    manifest: dict[str, Any]
    if path.exists():
        manifest = read_json(path)
    else:
        manifest = {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_sha": _git_sha(),
            "platform": f"{platform.system()} {platform.machine()}",
            "versions": package_versions(),
            "stages": {},
        }
    manifest.setdefault("stages", {})[stage] = {
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        **payload,
    }
    write_json(path, manifest)
    return path


@contextmanager
def timed(label: str, logger: logging.Logger | None = None) -> Iterator[dict]:
    """Time a block and expose the elapsed seconds to the caller."""
    log = logger or get_logger("khon_recon")
    log.info("%s ...", label)
    start = time.perf_counter()
    holder: dict[str, float] = {}
    try:
        yield holder
    finally:
        holder["seconds"] = time.perf_counter() - start
        log.info("%s done in %.1fs", label, holder["seconds"])
