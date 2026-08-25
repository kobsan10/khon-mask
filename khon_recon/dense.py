"""Dense multi-view stereo -- Colab hand-off.

COLMAP's ``patch_match_stereo`` requires CUDA. This project is developed on
Apple Silicon, where the Homebrew build reports "without CUDA" and dense stereo
simply cannot run:

    $ colmap --help
    COLMAP 4.1.1 (Commit Unknown on Unknown without CUDA)

So densification is the one stage that leaves this machine. The split is
mechanical rather than conceptual: SfM produces poses locally, those poses plus
the images go to a Colab GPU, and a fused point cloud comes back.

  local   scripts/02_dense_export.py  -> dense_bundle_<run>.zip
  Colab   notebooks/colmap_dense_colab.ipynb -> fused.ply
  local   scripts/03_dense_import.py <fused.ply>

The backend is deliberately behind a small interface so a local CPU
densifier (OpenMVS) can be added later without touching the stages downstream.
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

from .config import Config
from .io_utils import get_logger, list_images, timed, write_json

log = get_logger(__name__)


def cuda_available() -> bool:
    """Whether the local COLMAP build can do dense stereo at all."""
    try:
        result = subprocess.run(
            ["colmap", "--help"], capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return False
    banner = (result.stdout or "") + (result.stderr or "")
    return "without CUDA" not in banner


def export_bundle(cfg: Config, include_masks: bool = True) -> dict[str, Any]:
    """Package images, masks and the sparse model for the Colab dense run."""
    sparse_dir = cfg.sparse_dir / "0"
    if not (sparse_dir / "cameras.bin").exists():
        raise FileNotFoundError(
            f"no sparse model at {sparse_dir}; run scripts/01_sfm.py first"
        )

    images = list_images(cfg.images_dir)
    if not images:
        raise FileNotFoundError(f"no images in {cfg.images_dir}")

    # Settings the notebook reads, so the dense run is driven by the same
    # config as everything else rather than by hand-edited notebook cells.
    settings = {
        "run_id": cfg.paths.run_id,
        "subject": cfg.paths.subject,
        "max_image_size": cfg.dense.max_image_size,
        "geom_consistency": cfg.dense.geom_consistency,
        "window_radius": cfg.dense.window_radius,
        "num_samples": cfg.dense.num_samples,
        "fusion_min_num_pixels": cfg.dense.fusion_min_num_pixels,
        "fusion_max_reproj_error": cfg.dense.fusion_max_reproj_error,
        "n_images": len(images),
    }

    bundle = cfg.run_dir / f"dense_bundle_{cfg.paths.run_id}.zip"
    with timed(f"packaging {len(images)} images + sparse model", log):
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for image in images:
                zf.write(image, f"images/{image.name}")
            if include_masks and cfg.mask.enabled and cfg.masks_dir.is_dir():
                for mask in sorted(cfg.masks_dir.glob("*.png")):
                    zf.write(mask, f"masks/{mask.name}")
            for item in sparse_dir.iterdir():
                if item.is_file():
                    zf.write(item, f"sparse/{item.name}")
            zf.writestr("dense_settings.json", __import__("json").dumps(settings, indent=2))

    size_mb = bundle.stat().st_size / 1e6
    log.info("wrote %s (%.1f MB)", bundle.name, size_mb)
    log.info(
        "next: open notebooks/colmap_dense_colab.ipynb in Colab (GPU runtime), "
        "upload this bundle, run all cells, then download fused.ply"
    )
    return {"bundle": str(bundle), "size_mb": size_mb, **settings}


def import_fused(cfg: Config, ply_path: Path) -> dict[str, Any]:
    """Ingest ``fused.ply`` from Colab and check it matches the sparse model.

    The frame/scale check is the important part: a bundle downloaded from the
    wrong run produces a plausible-looking file that meshes into nonsense, and
    the cause is very hard to see three stages later.
    """
    import open3d as o3d

    from .metrics import load_reconstruction, sparse_dense_agreement

    ply_path = Path(ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(ply_path)

    cfg.dense_dir.mkdir(parents=True, exist_ok=True)
    target = cfg.dense_dir / "fused.ply"
    if ply_path.resolve() != target.resolve():
        shutil.copy2(ply_path, target)

    with timed("loading fused point cloud", log):
        pcd = o3d.io.read_point_cloud(str(target))
    n_points = len(pcd.points)
    if n_points == 0:
        raise ValueError(f"{target} contains no points")

    reconstruction = load_reconstruction(cfg.sparse_dir)
    agreement = sparse_dense_agreement(reconstruction, pcd)
    if not agreement["ok"]:
        log.error(
            "dense cloud does not line up with the sparse model "
            "(centroid offset %.2f x object size, extent ratio %.2f). "
            "This usually means fused.ply came from a different run.",
            agreement["centroid_offset_relative"], agreement["extent_ratio"],
        )
    else:
        log.info(
            "dense/sparse alignment OK (%d dense points vs %d sparse)",
            agreement["n_dense_points"], agreement["n_sparse_points"],
        )

    stats = {
        "fused_ply": str(target),
        "n_points": n_points,
        "has_colors": bool(pcd.has_colors()),
        "has_normals": bool(pcd.has_normals()),
        "agreement": agreement,
    }
    write_json(cfg.dense_dir / "dense_stats.json", stats)
    return stats


def dense_instructions(cfg: Config) -> str:
    """Printable next-step instructions for the Colab hand-off."""
    return "\n".join(
        [
            "",
            "=" * 70,
            "DENSE STAGE -- runs on Colab, not on this Mac",
            "=" * 70,
            "COLMAP's patch_match_stereo needs CUDA; the local build reports",
            "'without CUDA', so this stage cannot run here.",
            "",
            "  1. open notebooks/colmap_dense_colab.ipynb in Google Colab",
            "  2. Runtime > Change runtime type > GPU (T4 is enough)",
            f"  3. upload {cfg.run_dir.name}/dense_bundle_{cfg.paths.run_id}.zip",
            "  4. run all cells (roughly 15-45 min for 60-100 images)",
            "  5. download fused.ply, then run:",
            "",
            f"     python scripts/03_dense_import.py ~/Downloads/fused.ply "
            f"-s paths.run_id={cfg.paths.run_id}",
            "=" * 70,
            "",
        ]
    )
