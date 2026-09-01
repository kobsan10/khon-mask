"""Per-stage visual results, rendered from the same viewpoints every time.

The metrics say whether a stage succeeded; they do not say what it *looks*
like, and several failures in this pipeline are far easier to see than to
measure -- a dense cloud that reconstructed the table instead of the mask, a
Poisson surface ballooning over the eye sockets, a texture bake that coloured
nothing because the normals were inverted.

Every preview here is rendered from the same evenly-spaced registered cameras
and beside the photograph taken from that pose, so a stage can be compared
against the real object and against the stage before it. That fixed viewpoint
choice is the whole point: renders from arbitrary angles cannot be stacked.

Previews are diagnostics, not paper figures -- ``report.py`` still owns
anything that goes into the write-up, at IEEE column width.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import Config
from .io_utils import get_logger
from .render import camera_to_opencv, extrinsic_matrix

log = get_logger(__name__)

# Rows of the progression sheet, in pipeline order. The stage that produces
# each one is named so a missing row points at the stage to re-run.
STAGE_PREVIEWS: list[tuple[str, str, str]] = [
    ("01_sfm", "1. sparse SfM", "scripts/01_sfm.py"),
    ("03_dense", "3. dense MVS", "scripts/03_dense_import.py"),
    ("04_mesh", "4. Poisson mesh", "scripts/04_mesh.py"),
    ("05_texture", "5. textured", "scripts/05_texture.py"),
]

TILE_HEIGHT = 460
_LABEL_BG = (28, 28, 28)


def previews_dir(cfg: Config) -> Path:
    directory = cfg.figures_dir / "stages"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def select_views(reconstruction, count: int = 3) -> list:
    """Evenly spaced registered images, in name order.

    Sorted by filename and sampled evenly so the same physical viewpoints come
    back on every stage and every re-run -- previews that wander are useless
    for comparison.
    """
    images = sorted(
        (im for im in reconstruction.images.values() if im.has_pose),
        key=lambda im: im.name,
    )
    if not images:
        return []
    if len(images) <= count:
        return images
    idx = np.linspace(0, len(images) - 1, count).round().astype(int)
    return [images[int(i)] for i in dict.fromkeys(idx)]


def _fit(image: np.ndarray, height: int = TILE_HEIGHT) -> np.ndarray:
    scale = height / image.shape[0]
    return cv2.resize(image, (max(1, int(round(image.shape[1] * scale))), height))


def _label(tile: np.ndarray, text: str) -> np.ndarray:
    """Caption a tile with a strip along the bottom, outside the image area."""
    strip = np.full((26, tile.shape[1], 3), _LABEL_BG, np.uint8)
    cv2.putText(strip, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1,
                cv2.LINE_AA)
    return np.vstack([tile, strip])


def _grid(rows: list[list[np.ndarray]]) -> np.ndarray:
    """Stack labelled tiles into one image, padding ragged rows."""
    width = max(sum(t.shape[1] for t in row) for row in rows)
    out = []
    for row in rows:
        strip = np.hstack(row)
        if strip.shape[1] < width:
            pad = np.full((strip.shape[0], width - strip.shape[1], 3), _LABEL_BG, np.uint8)
            strip = np.hstack([strip, pad])
        out.append(strip)
    return np.vstack(out)


def photo_tile(cfg: Config, image) -> np.ndarray | None:
    path = cfg.images_dir / image.name
    photo = cv2.imread(str(path))
    return None if photo is None else _fit(photo)


def render_points(
    points: np.ndarray,
    colors: np.ndarray | None,
    camera,
    image,
    height: int = TILE_HEIGHT,
    dilate: int = 2,
) -> np.ndarray:
    """Project a point cloud into a registered camera.

    Painter's algorithm rather than a real z-buffer: points are drawn far to
    near so nearer ones overwrite. That is enough for a diagnostic and avoids
    meshing a cloud purely to look at it.
    """
    width_px = int(round(camera.width * height / camera.height))
    scale = height / camera.height

    cam_points = image.cam_from_world() * points
    in_front = cam_points[:, 2] > 0
    if not in_front.any():
        return np.zeros((height, width_px, 3), np.uint8)

    uv = np.asarray(camera.img_from_cam(cam_points[in_front]), dtype=float) * scale
    depth = cam_points[in_front, 2]
    rgb = (
        np.full((int(in_front.sum()), 3), 200, np.uint8)
        if colors is None or len(colors) == 0
        else (np.clip(colors[in_front], 0, 1) * 255).astype(np.uint8)[:, ::-1]  # RGB->BGR
    )

    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    inside = (u >= 0) & (u < width_px) & (v >= 0) & (v < height)
    u, v, depth, rgb = u[inside], v[inside], depth[inside], rgb[inside]

    canvas = np.zeros((height, width_px, 3), np.uint8)
    for i in np.argsort(-depth):
        canvas[v[i], u[i]] = rgb[i]
    if dilate > 0:
        canvas = cv2.dilate(canvas, np.ones((dilate + 1, dilate + 1), np.uint8))
    return canvas


def render_mesh_views(mesh, reconstruction, views: list, height: int = TILE_HEIGHT) -> list:
    """Ray-cast a mesh from each view. Headless: RaycastingScene, not Filament."""
    from .render import MeshRenderer

    renderer = MeshRenderer(mesh, background=(0, 0, 0))
    tiles = []
    for image in views:
        camera = reconstruction.cameras[image.camera_id]
        result = renderer.render(camera, image, scale=height / camera.height)
        tiles.append(_fit(result.color, height))
    return tiles


def _write(path: Path, image: np.ndarray) -> Path:
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    log.info("preview: %s", path)
    return path


def preview_point_stage(
    cfg: Config, points: np.ndarray, colors: np.ndarray | None, name: str, caption: str
) -> Path | None:
    """One preview row per viewpoint: the photograph beside the reconstruction."""
    from .metrics import load_reconstruction

    try:
        reconstruction = load_reconstruction(cfg.sparse_dir)
    except Exception as exc:
        log.warning("preview skipped (%s): %s", name, exc)
        return None

    views = select_views(reconstruction)
    if not views:
        return None

    rows = []
    for image in views:
        photo = photo_tile(cfg, image)
        rendered = render_points(
            points, colors, reconstruction.cameras[image.camera_id], image
        )
        row = [] if photo is None else [_label(photo, f"photo  {image.name}")]
        row.append(_label(rendered, caption))
        rows.append(row)
    return _write(previews_dir(cfg) / f"{name}.jpg", _grid(rows))


def preview_mesh_stage(
    cfg: Config, mesh_path: Path, name: str, caption: str
) -> Path | None:
    import open3d as o3d

    from .metrics import load_reconstruction

    if not Path(mesh_path).exists():
        return None
    try:
        reconstruction = load_reconstruction(cfg.sparse_dir)
    except Exception as exc:
        log.warning("preview skipped (%s): %s", name, exc)
        return None

    views = select_views(reconstruction)
    if not views:
        return None

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) == 0:
        log.warning("preview skipped (%s): mesh has no triangles", name)
        return None
    mesh.compute_vertex_normals()

    tiles = render_mesh_views(mesh, reconstruction, views)
    rows = []
    for image, tile in zip(views, tiles):
        photo = photo_tile(cfg, image)
        row = [] if photo is None else [_label(photo, f"photo  {image.name}")]
        row.append(_label(tile, caption))
        rows.append(row)
    return _write(previews_dir(cfg) / f"{name}.jpg", _grid(rows))


def preview_sparse(cfg: Config) -> Path | None:
    """Stage 1: the sparse cloud that fixes every camera pose downstream."""
    from .metrics import load_reconstruction

    try:
        reconstruction = load_reconstruction(cfg.sparse_dir)
    except Exception as exc:
        log.warning("sparse preview skipped: %s", exc)
        return None

    points = np.array([p.xyz for p in reconstruction.points3D.values()])
    colors = np.array([p.color for p in reconstruction.points3D.values()]) / 255.0
    if not len(points):
        return None
    return preview_point_stage(
        cfg, points, colors, "01_sfm", f"sparse SfM  {len(points)} pts"
    )


def preview_dense(cfg: Config) -> Path | None:
    """Stage 3: the fused MVS cloud, before any surface is fitted to it."""
    import open3d as o3d

    ply = cfg.dense_dir / "fused.ply"
    if not ply.exists():
        return None
    pcd = o3d.io.read_point_cloud(str(ply))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    if not len(points):
        return None
    return preview_point_stage(
        cfg, points, colors, "03_dense", f"dense MVS  {len(points)} pts"
    )


def preview_mesh(cfg: Config) -> Path | None:
    return preview_mesh_stage(cfg, cfg.mesh_dir / "mesh.ply", "04_mesh", "Poisson mesh")


def preview_texture(cfg: Config) -> Path | None:
    for candidate in ("mesh_textured.ply", "mesh_textured.obj"):
        path = cfg.mesh_dir / candidate
        if path.exists():
            return preview_mesh_stage(cfg, path, "05_texture", "textured mesh")
    return None


def progression_sheet(cfg: Config) -> Path | None:
    """Every finished stage stacked against the photograph, one row per stage.

    This is the comparison the individual previews cannot give: whether detail
    survived from dense cloud to mesh to texture, read down a column.
    """
    from .metrics import load_reconstruction

    try:
        reconstruction = load_reconstruction(cfg.sparse_dir)
    except Exception as exc:
        log.warning("progression sheet skipped: %s", exc)
        return None

    views = select_views(reconstruction)
    if not views:
        return None

    directory = previews_dir(cfg)
    photos = [photo_tile(cfg, im) for im in views]
    rows = [[_label(p, f"photograph  {im.name}") for p, im in zip(photos, views) if p is not None]]

    missing = []
    for name, title, script in STAGE_PREVIEWS:
        source = directory / f"{name}.jpg"
        if not source.exists():
            missing.append((title, script))
            continue
        sheet = cv2.imread(str(source))
        if sheet is None:
            continue
        # Each stage preview is [photo | render] per row; take the render half.
        half = sheet.shape[1] // 2
        tiles = []
        row_height = sheet.shape[0] // max(len(views), 1)
        for i in range(len(views)):
            tile = sheet[i * row_height : (i + 1) * row_height, half:]
            tiles.append(_label(_fit(tile[:-26] if tile.shape[0] > 26 else tile), title))
        rows.append(tiles)

    if not rows[0]:
        return None
    for title, script in missing:
        log.info("progression sheet: no %s preview yet (run %s)", title, script)

    return _write(directory / "pipeline_progression.jpg", _grid(rows))


_BUILDERS = {
    "sfm": preview_sparse,
    "dense": preview_dense,
    "mesh": preview_mesh,
    "texture": preview_texture,
}


def write_stage_preview(cfg: Config, stage: str) -> Path | None:
    """Render one stage's preview and refresh the progression sheet.

    Never raises. A preview is a diagnostic: failing to draw one must not fail
    a stage whose geometry and metrics were computed correctly and already
    written to disk.
    """
    builder = _BUILDERS.get(stage)
    if builder is None:
        return None
    try:
        path = builder(cfg)
        progression_sheet(cfg)
        return path
    except Exception as exc:  # noqa: BLE001 -- diagnostics must not break a stage
        log.warning("could not render the %s preview: %s", stage, exc)
        return None


def refresh(cfg: Config) -> dict[str, Any]:
    """Rebuild every preview that has data behind it, then the summary sheet."""
    written = {
        "01_sfm": preview_sparse(cfg),
        "03_dense": preview_dense(cfg),
        "04_mesh": preview_mesh(cfg),
        "05_texture": preview_texture(cfg),
    }
    written["progression"] = progression_sheet(cfg)
    return {k: str(v) for k, v in written.items() if v is not None}
