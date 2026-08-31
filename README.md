# 3D Reconstruction of Khon Masks from Multi-View Photography

Implementation of the pipeline described in `khon-3d-reconstruction-paper.docx`:
a classical structure-from-motion and multi-view-stereo photogrammetry pipeline
that turns a set of overlapping photographs of a Khon mask into a textured 3D
mesh, plus the evaluation and ablations the proposal commits to.

```text
photographs → SfM poses (COLMAP) → dense MVS cloud → Poisson mesh + texture → evaluation
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate khon
brew install colmap          # 4.1.1, arm64 bottle
```

Python is pinned to **3.11** on purpose: Open3D 0.19 publishes macOS wheels for
cp310–cp312 only, so the meshing stage is uninstallable on 3.13/3.14.

Verify:

```bash
python -c "import open3d, pycolmap, cv2; print(open3d.__version__, pycolmap.__version__)"
colmap --help | head -1
```

### The dense stage runs on Colab, not locally

COLMAP's `patch_match_stereo` requires CUDA. On Apple Silicon the build reports
`COLMAP 4.1.1 (... without CUDA)` and dense stereo cannot run at all. Sparse SfM,
meshing, texturing and evaluation all run locally; only densification is
offloaded to a free Colab GPU via `notebooks/colmap_dense_colab.ipynb`.

---

## Running the pipeline

```bash
# 0. ingest, QC the capture, build foreground masks
python scripts/00_prepare_images.py -c configs/sample.yaml --input data/raw/sample/images

# 1. sparse reconstruction (poses + sparse cloud + Eq. (1) error)
python scripts/01_sfm.py -c configs/sample.yaml

# 2. package for the GPU stage
python scripts/02_dense_export.py -c configs/sample.yaml
#    -> run notebooks/colmap_dense_colab.ipynb on Colab, download fused.ply

# 3. bring the dense cloud back (checks it matches the sparse model)
python scripts/03_dense_import.py ~/Downloads/fused.ply -c configs/sample.yaml

# 4-5. surface reconstruction and colour recovery
python scripts/04_mesh.py    -c configs/sample.yaml
python scripts/05_texture.py -c configs/sample.yaml

# 6-8. evaluation, ablations, paper figures
python scripts/06_evaluate.py  -c configs/sample.yaml
python scripts/07_ablations.py -c configs/sample.yaml
python scripts/08_report.py    -c configs/sample.yaml
```

Any config value can be overridden per-run without editing files:

```bash
python scripts/04_mesh.py -c configs/sample.yaml -s mesh.poisson_depth=9
```

### No photographs yet?

Both of these exercise the full code path before the real shoot:

```bash
# deterministic smoke test on COLMAP's sample data
python scripts/00_prepare_images.py --fetch-sample south-building --sample-limit 25 \
    -s paths.subject=sample -s paths.run_id=smoke -s mask.enabled=false
python scripts/01_sfm.py -s paths.subject=sample -s paths.run_id=smoke -s mask.enabled=false

# rehearsal on a phone video of any household object
python scripts/00_prepare_images.py --video ~/orbit.mov -s paths.subject=dryrun
```

### Guides

- **[CAPTURE_GUIDE.md](CAPTURE_GUIDE.md)** — read before photographing the mask.
- **[RUNBOOK.md](RUNBOOK.md)** — step by step, with checkpoints, for the day the
  photographs arrive. Start here once you have images.

---

## Layout

```text
khon_recon/           importable package -- all logic
  config.py           typed config, YAML + `extends:` + dotted overrides
  prepare.py          folder / video / sample-dataset ingest
  capture_qc.py       blur, exposure, overlap, static-background checks
  masking.py          foreground masks (rembg, GrabCut fallback)
  sfm.py              pycolmap: extract -> match -> map, hold-out split
  dense.py            Colab hand-off: export bundle, import fused.ply
  mesh.py             normals -> Poisson -> density trim -> clean
  texture.py          multi-view projection + median blending, UV bake
  metrics.py          Eq. (1), tracks, coverage, density, completeness
  render.py           headless ray-cast rendering from estimated poses
  compare.py          masked PSNR/SSIM, render vs photograph
  specularity.py      does gilding actually break MVS?
  ablations.py        reduced overlap, minimal BA, no masks
  report.py           figures + LaTeX tables at IEEE column width
scripts/              numbered CLI stages, 00 -> 08
configs/              default.yaml, sample.yaml
notebooks/            the Colab dense stage
data/                 gitignored: raw/<subject>/, runs/<run_id>/
```

Every stage writes into `data/runs/<run_id>/` alongside the resolved config and
a manifest recording package versions and timings, so any number in the report
traces back to the run that produced it. Ablations are therefore just different
configs rather than forked code.

---

## What the evaluation reports

Since no ground-truth 3D scan of the mask exists, quality is measured from the
pipeline itself:

| Metric | Where |
| --- | --- |
| Mean reprojection error, Eq. (1) | `metrics.reprojection_errors` |
| Track lengths, camera coverage | `metrics.track_statistics`, `metrics.camera_coverage` |
| Dense density, holes, watertightness | `metrics.point_cloud_density`, `metrics.mesh_completeness` |
| Novel-view PSNR/SSIM on held-out views | `compare.evaluate_views` |
| Specularity vs reconstruction density | `specularity.run_specularity_study` |

Three implementation notes worth carrying into the write-up:

**Eq. (1) is computed from scratch, not read from COLMAP.** COLMAP's
`compute_mean_reprojection_error()` returns a *cached* per-point error written
during bundle adjustment and never refreshed. After held-out views are
registered — which adds observations to existing tracks — it silently reports
the stale pre-registration figure. On the smoke-test model it reports 0.362 px
where the true value is 0.402 px. `metrics.verify_against_builtin` confirms the
two agree exactly on an unmodified model, then uses the recomputed value.

**Held-out views are genuinely held out.** Every 5th image is withheld from
mapping and registered afterwards with the existing geometry fixed
(`sfm.holdout_every`), so novel-view scores measure reconstruction quality
rather than self-consistency.

**The "no bundle adjustment" ablation is really minimal-vs-full refinement.**
COLMAP cannot triangulate with BA removed entirely. To isolate BA's actual
contribution, `ablations.bundle_adjustment_isolation` takes one minimally
refined model and runs a full global BA over it, changing nothing else.

---

## Known constraints

- Dense MVS requires the Colab round trip (no CUDA locally). A local OpenMVS
  backend can be added behind the interface in `dense.py`.
- **Runs are only reproducible with `sfm.multiple_models=true` and
  `sfm.mapper_num_threads=1`.** At the defaults, five identical runs on this
  capture registered between 5 and 46 of 49 images — see
  `data/repeatability_study.json`. Both settings are already set in
  `configs/sample.yaml`.
- **Masking hurts this capture and is off.** The gilded surface yields almost no
  matchable features, so the background carries the geometry; masking took
  usable image pairs from 21.9% to 1.7%. Masking remains correct for a true
  turntable capture — verify with the match graph, not by assumption.
- Gilded and mirrored surfaces are the expected failure mode; the specularity
  study measures it rather than assuming it.
- The current capture never shows the mask's underside, so the result is an open
  shell rather than a closed model.
