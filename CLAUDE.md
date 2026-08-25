# Khon mask 3D reconstruction

Classical photogrammetry pipeline (SfM → MVS → Poisson → evaluation) implementing
`khon-3d-reconstruction-paper.docx`, a Computer Vision course paper. Results here
become figures and tables in that paper, so **metrics must be correct and
reproducible, not merely plausible**.

## Environment

Always activate the env first — nothing imports without it:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate khon
```

Python is pinned to **3.11**. Open3D 0.19 ships macOS wheels for cp310–cp312
only; on the system Python (3.14) the meshing stage cannot be installed at all.
Do not "upgrade" the env to a newer Python.

## IMPORTANT: dense MVS cannot run on this machine

This is an Apple Silicon Mac. `colmap --help` reports `4.1.1 ... without CUDA`,
and COLMAP's `patch_match_stereo` is CUDA-only. Densification runs on Colab via
`notebooks/colmap_dense_colab.ipynb`; `scripts/02_dense_export.py` and
`scripts/03_dense_import.py` are the hand-off. Never add a local dense-stereo
call, and never debug a "missing fused.ply" as if it were a local failure.

## Commands

```bash
# smoke test — no photographs needed, ~40 s, expect 25/25 registered, <1 px error
python scripts/00_prepare_images.py --fetch-sample south-building --sample-limit 25 \
    -s paths.subject=sample -s paths.run_id=smoke -s mask.enabled=false
python scripts/01_sfm.py -s paths.subject=sample -s paths.run_id=smoke -s mask.enabled=false

# real pipeline — run stages in order, one at a time
python scripts/00_prepare_images.py -c configs/mask01.yaml --input ~/khon_photos
python scripts/01_sfm.py            -c configs/mask01.yaml
python scripts/02_dense_export.py   -c configs/mask01.yaml   # → Colab → fused.ply
python scripts/03_dense_import.py ~/Downloads/fused.ply -c configs/mask01.yaml
python scripts/04_mesh.py           -c configs/mask01.yaml
python scripts/05_texture.py        -c configs/mask01.yaml
python scripts/06_evaluate.py       -c configs/mask01.yaml
python scripts/07_ablations.py      -c configs/mask01.yaml
python scripts/08_report.py         -c configs/mask01.yaml
```

Override any config value inline rather than editing YAML:
`-s mesh.poisson_depth=9`. Unknown keys raise — a typo in an ablation config is
a silently wrong experiment, so it fails loudly instead.

## Architecture

`khon_recon/` holds all logic; `scripts/NN_*.py` are thin CLI wrappers. Keep it
that way — logic in a script cannot be reused by the ablations.

Every stage writes to `data/runs/<run_id>/` alongside its resolved config and a
manifest. **Ablations are different configs, not different code**
(`khon_recon/ablations.py`). If a change requires forking a stage to support an
experiment, add a config field instead.

## Verification

Changes to geometry or metrics must be verified numerically, not by eye:

- Run the smoke test above. Expect **25/25 images, ~0.36–0.43 px** error.
- `metrics.verify_against_builtin()` must still report `agrees=True` on
  `data/runs/*/sparse/train/0` (a model untouched since bundle adjustment).
- After meshing, `outward_normal_fraction` in `mesh_stats.json` must be > 0.5.
- After texturing, a sudden jump in `unseen_fraction` means normals broke.

A rendered image looking "roughly right" is not verification — the pipeline
produced a confident, completely inside-out mesh that still rendered plausibly.

## Gotchas that have already caused real bugs

**COLMAP's mean reprojection error is stale.** `compute_mean_reprojection_error()`
returns a *cached* per-point value written during BA and never refreshed. After
held-out views are registered it under-reports (0.362 px vs the true 0.402 px).
Always use `metrics.reprojection_errors()`. Never quote the built-in in the paper.

**Normal orientation is not cosmetic.** Tangent-plane orientation is consistent
only up to a global sign and picked *inward*, making every surface point test as
self-occluded and texturing colour ~nothing. Normals must be oriented with the
reconstructed camera centres, and COLMAP's own `fused.ply` normals are preferred
over re-estimating (`mesh.estimate_normals`).

**Turntable capture needs masks.** Object rotating + camera fixed means the
*background* is the rigid scene, and SfM will reconstruct the room with an
excellent reprojection error. `mask.enabled` must stay true for such captures;
`capture_qc.static_background_probe` detects the regime.

**Held-out views must stay held out.** `sfm.holdout_every` withholds every k-th
image from mapping; they are registered afterwards with `fix_existing_frames`.
Do not "simplify" this into mapping all images — it silently turns the novel-view
metric into a training metric.

**"No bundle adjustment" is not runnable.** COLMAP cannot triangulate with BA
removed. The ablation compares minimal vs full refinement, and
`ablations.bundle_adjustment_isolation()` isolates BA on a fixed model. Describe
it that way in the paper; do not claim BA was disabled.

## Conventions

- Comments explain *why*, especially where a parameter choice defends against a
  known failure (specular highlights, invented Poisson surface). Do not add
  comments restating what the line does.
- Reuse `io_utils` (`list_images`, `write_json`, `timed`, `update_manifest`) and
  `render.camera_to_opencv` / `extrinsic_matrix` rather than re-deriving camera
  conversions.
- Rendering must stay headless: use `open3d.t.geometry.RaycastingScene`, never
  the Filament `OffscreenRenderer`, which is unreliable on macOS.
- Figures go through `report.py` at IEEE column width, saved as PDF **and** PNG.
- `data/` is gitignored and never committed — it holds images, clouds and meshes.
  Everything in it is reproducible from a capture set plus a config.
