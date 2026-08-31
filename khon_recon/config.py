"""Typed configuration for the reconstruction pipeline.

A run is fully described by its config, and every stage copies the resolved
config into its run directory. That is what makes the ablations in
``scripts/07_ablations.py`` a loop over configs rather than a set of forked
scripts, and it is what makes the numbers in the report traceable back to the
settings that produced them.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Stage configs
# --------------------------------------------------------------------------


@dataclass
class PathsConfig:
    """Where images come from and where run outputs go."""

    data_root: str = "data"
    subject: str = "sample"  # names the capture set under data/raw/<subject>
    run_id: str = "default"  # names the output dir under data/runs/<run_id>


@dataclass
class PrepareConfig:
    """Image ingest: folder, video frames, or the COLMAP sample set."""

    # Frames per second to sample when the input is a video. 2 fps over a slow
    # ~60 s orbit yields ~120 frames, in the range the proposal asks for.
    video_fps: float = 2.0
    # Longest image edge after downscaling. 1600 keeps CPU SfM tractable while
    # preserving the crown detail; set to 0 to keep the original resolution.
    max_dim: int = 1600
    jpeg_quality: int = 95
    # Cap on how many frames to keep, evenly spaced. 0 = keep all.
    max_images: int = 0


@dataclass
class QCConfig:
    """Capture quality control thresholds."""

    # Images whose variance-of-Laplacian falls in this bottom fraction are
    # flagged as soft. Relative rather than absolute because the scale depends
    # on resolution and texture.
    blur_bottom_fraction: float = 0.10
    # Absolute floor as well; below this an image is blurry on any scale.
    blur_abs_min: float = 40.0
    # Luminance drift tolerance across the set (0-255 scale). The proposal
    # requires fixed exposure, so a large drift means the camera was on auto.
    exposure_max_drift: float = 25.0
    # Consecutive-pair inlier matches below this suggest insufficient overlap.
    min_pair_matches: int = 100
    # Feature count used for the cheap overlap probe.
    overlap_features: int = 2000
    # Fraction of the frame that must stay still between frames before the set
    # is called a static-background (turntable) capture.
    static_bg_ratio: float = 0.55


@dataclass
class MaskConfig:
    """Foreground masking.

    Essential for turntable capture: with a rotating object and a static
    camera the background is the rigid scene, so COLMAP will happily register
    the background and smear the object. Masks confine features to the object.
    """

    enabled: bool = True
    # "rembg" (u2net segmentation) or "grabcut" (OpenCV, no model download).
    method: str = "rembg"
    # Morphological erosion of the mask border, in pixels. Shrinking slightly
    # is safer than growing: a mask that leaks background reintroduces exactly
    # the features we are trying to exclude.
    erode_px: int = 4
    # Fraction of the frame the foreground must occupy before a mask is
    # trusted; outside this range the mask is reported as suspect.
    min_area_fraction: float = 0.02
    max_area_fraction: float = 0.95
    # GrabCut fallback: initial foreground rectangle as a margin fraction.
    grabcut_margin: float = 0.08
    grabcut_iterations: int = 5


@dataclass
class SfMConfig:
    """Sparse reconstruction (feature extraction, matching, mapping)."""

    camera_model: str = "SIMPLE_RADIAL"
    single_camera: bool = True  # one lens for the whole set
    max_num_features: int = 8192
    use_gpu: bool = False  # no CUDA on Apple Silicon; SIFT runs on CPU
    matcher: str = "exhaustive"  # "exhaustive" | "sequential"
    # Keep only every k-th image. 1 = use everything. This is the knob the
    # reduced-overlap ablation turns: dropping images widens the baseline
    # between neighbours and directly tests the proposal's 60-70% overlap claim.
    subsample_every: int = 1
    # Hold out every k-th image from mapping to serve as a genuine novel-view
    # test set. The proposal only promised eyeballed comparison; this makes it
    # a measurement. 0 disables the split.
    holdout_every: int = 5
    # Bundle adjustment controls -- the knobs the "no BA" ablation turns down.
    ba_global_max_refinements: int = 5
    ba_local_max_refinements: int = 2
    # Save a pre-final-BA snapshot so the ablation can quantify BA's effect.
    snapshot_before_final_ba: bool = True
    min_num_matches: int = 15
    # Let the mapper start over from a different seed pair when a reconstruction
    # stalls, then keep the largest model. Disabling this makes a single bad
    # initial pair fatal: on this capture it turned a 48/49 reconstruction into
    # 5/49. The subject is specular and its features are weak, so the choice of
    # seed pair matters far more than it would on an easy scene.
    multiple_models: bool = True
    # Threads for mapping. Incremental SfM is order-sensitive and multi-threaded
    # bundle adjustment does not reduce deterministically, so >1 makes runs
    # irreproducible. Set to 1 when a result has to be exactly repeatable.
    mapper_num_threads: int = -1


@dataclass
class DenseConfig:
    """Multi-view stereo.

    COLMAP's patch_match_stereo is CUDA-only and this project is developed on
    Apple Silicon, so densification runs on a Colab GPU. These settings are
    written into the hand-off bundle and read by the notebook.
    """

    max_image_size: int = 1600
    geom_consistency: bool = True
    window_radius: int = 5
    num_samples: int = 15
    # stereo_fusion: a point must be consistent across at least this many views.
    fusion_min_num_pixels: int = 5
    fusion_max_reproj_error: float = 2.0


@dataclass
class MeshConfig:
    """Poisson surface reconstruction."""

    # Normal estimation. Poisson quality is dominated by normal orientation,
    # so these get their own knobs.
    normal_knn: int = 30
    normal_orient_knn: int = 30
    # Prefer normals already on the cloud: COLMAP's stereo_fusion writes
    # correctly outward-facing normals derived from the depth maps, and
    # re-estimating them throws that information away.
    use_existing_normals: bool = True
    # Statistical outlier removal on the dense cloud before meshing.
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    poisson_depth: int = 10
    poisson_scale: float = 1.1
    # Drop the lowest-density vertices: this is what removes the balloon
    # artifacts Poisson invents over under-observed regions (eye sockets).
    density_trim_quantile: float = 0.03
    keep_largest_component: bool = True
    # 0 disables decimation; otherwise the target triangle count.
    target_triangles: int = 0


@dataclass
class TextureConfig:
    """Colour recovery."""

    # "vertex"  : per-vertex colour from the fused cloud (always works)
    # "uv"      : xatlas unwrap + photo projection into a UV texture map
    mode: str = "vertex"
    texture_size: int = 4096
    # Reject observations seen at more than this angle from the surface
    # normal; grazing views smear the gilded ornament.
    max_view_angle_deg: float = 70.0
    # Blend across views with a median rather than a mean. Deliberate: the
    # median is the cheap defence against the specular highlights the proposal
    # flags as the main failure mode.
    blend: str = "median"


@dataclass
class EvalConfig:
    """Evaluation and the specularity study."""

    # Nearest-neighbour spacing sample size for the density metric.
    density_sample: int = 50000
    render_width: int = 1200
    # Specular pixel detection in HSV: bright and unsaturated.
    specular_v_min: int = 230
    specular_s_max: int = 60
    # Render/photo comparison ignores pixels outside the rendered silhouette
    # so that empty background cannot inflate PSNR/SSIM.
    masked_compare: bool = True


@dataclass
class Config:
    """Top-level pipeline config."""

    name: str = "default"
    paths: PathsConfig = field(default_factory=PathsConfig)
    prepare: PrepareConfig = field(default_factory=PrepareConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    sfm: SfMConfig = field(default_factory=SfMConfig)
    dense: DenseConfig = field(default_factory=DenseConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    texture: TextureConfig = field(default_factory=TextureConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ---- derived paths -------------------------------------------------

    @property
    def data_root(self) -> Path:
        root = Path(self.paths.data_root)
        return root if root.is_absolute() else REPO_ROOT / root

    @property
    def raw_dir(self) -> Path:
        """Source images for this subject, as captured/prepared."""
        return self.data_root / "raw" / self.paths.subject

    @property
    def images_dir(self) -> Path:
        return self.raw_dir / "images"

    @property
    def masks_dir(self) -> Path:
        return self.raw_dir / "masks"

    @property
    def run_dir(self) -> Path:
        return self.data_root / "runs" / self.paths.run_id

    @property
    def sparse_dir(self) -> Path:
        return self.run_dir / "sparse"

    @property
    def dense_dir(self) -> Path:
        return self.run_dir / "dense"

    @property
    def mesh_dir(self) -> Path:
        return self.run_dir / "mesh"

    @property
    def eval_dir(self) -> Path:
        return self.run_dir / "eval"

    @property
    def figures_dir(self) -> Path:
        return self.run_dir / "figures"

    def make_run_dirs(self) -> Path:
        for d in (
            self.run_dir,
            self.sparse_dir,
            self.dense_dir,
            self.mesh_dir,
            self.eval_dir,
            self.figures_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        return self.run_dir

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: Path | None = None) -> Path:
        """Write the resolved config beside the run outputs."""
        path = path or (self.run_dir / "config.yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        return path


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Build a (possibly nested) dataclass from a plain dict.

    Unknown keys are rejected rather than ignored: a typo in a config is a
    silent wrong-experiment bug, which is exactly what an ablation table must
    never contain.
    """
    if not is_dataclass(cls):
        return data
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(
            f"unknown config key(s) for {cls.__name__}: {sorted(unknown)}. "
            f"valid keys: {sorted(fields)}"
        )
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        ftype = fields[key].type
        # Dataclass field types may arrive as strings under
        # ``from __future__ import annotations``; resolve via the default.
        if isinstance(value, dict):
            default = fields[key].default_factory  # type: ignore[union-attr]
            if default is not dataclasses.MISSING:
                sub_cls = type(default())
                if is_dataclass(sub_cls):
                    kwargs[key] = _from_dict(sub_cls, value)
                    continue
            if is_dataclass(ftype) and get_origin(ftype) is None:
                kwargs[key] = _from_dict(ftype, value)  # type: ignore[arg-type]
                continue
        kwargs[key] = value
    return cls(**kwargs)


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load a YAML config layered on top of the dataclass defaults.

    ``overrides`` uses dotted keys, e.g. ``{"mesh.poisson_depth": 9}``, which is
    what the ``--set`` CLI flag produces.
    """
    data: dict[str, Any] = {}
    if path is not None:
        path = Path(path)
        if not path.is_absolute() and not path.exists():
            candidate = REPO_ROOT / path
            if candidate.exists():
                path = candidate
        loaded = yaml.safe_load(path.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"config {path} must contain a YAML mapping")
        # A config may inherit from another via `extends:`.
        parent = loaded.pop("extends", None)
        if parent:
            parent_path = (path.parent / parent).resolve()
            base = load_config(parent_path).to_dict()
            loaded = _deep_merge(base, loaded)
        data = loaded

    for dotted, value in (overrides or {}).items():
        node = data
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot override into non-mapping key: {dotted}")
        node[parts[-1]] = value

    return _from_dict(Config, data)


def parse_set_overrides(pairs: list[str] | None) -> dict[str, Any]:
    """Turn ``["mesh.poisson_depth=9", "sfm.use_gpu=false"]`` into a dict.

    Values go through the YAML scalar parser so ints, floats and bools arrive
    with the right type instead of as strings.
    """
    out: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--set expects key=value, got: {pair!r}")
        key, _, raw = pair.partition("=")
        out[key.strip()] = yaml.safe_load(raw)
    return out
