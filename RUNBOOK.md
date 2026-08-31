# Runbook — from Khon mask photographs to paper results

Follow this top to bottom the day you get the photos. Each step says what to
run, **what a good result looks like**, and what to do when it doesn't.

Do not skip the checkpoints. Every one of them catches a failure that is cheap
to fix now and expensive to discover three stages later.

> Before the shoot, use [CAPTURE_GUIDE.md](CAPTURE_GUIDE.md) instead. This file
> assumes the photographs already exist.

**Rough timings** for 60–100 photos on this Mac, extrapolated from a 25-image
test — treat as estimates, not promises.

---

## Step 0 — Set up (5 min)

```bash
cd "/Users/bsan/Workspace/Computer Vision/khon-mask"
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate khon
```

Copy the photos somewhere stable and **keep the originals untouched** — the
pipeline never writes to your source folder, but a second copy costs nothing:

```bash
mkdir -p data/raw/sample/images
cp /Volumes/SDCARD/DCIM/*.JPG data/raw/sample/images/
ls data/raw/sample/images | wc -l          # expect 60–100
```

Pick a name for this mask and use it everywhere. `configs/sample.yaml` already
exists for the first subject. For a second mask, copy it:

```bash
sed 's/sample/mask02/g' configs/sample.yaml > configs/mask02.yaml
```

---

## Step 1 — Ingest, QC and masks (5–15 min)

```bash
python scripts/00_prepare_images.py -c configs/sample.yaml --input data/raw/sample/images
```

This copies and downscales the photos, runs the capture checks, and builds a
foreground mask per image.

### ✅ Checkpoint 1a — read the QC report

It prints a block like this:

```text
CAPTURE QC  --  84 images
sharpness (var of Laplacian): median  1644.0  min  1068.2
exposure drift (luminance range): 12.3
consecutive-pair overlap: median 412 inliers, weakest 180
static background: 71% of frame (border 88%)
```

| Warning | What it means | What to do |
| --- | --- | --- |
| *below the absolute sharpness floor* | Some frames are blurry | Delete those files from `data/raw/sample/images`, re-run with `--overwrite` |
| *luminance varies by N levels* | Exposure was not locked | Usable, but expect texture seams. Note it as a limitation |
| *N consecutive pair(s) below … inlier matches* | **Gaps in coverage** | The most serious warning. If you can still reshoot, fill those angles now |
| *background is static … turntable regime* | Object rotated, camera fixed | Only then turn masking on. For a walk-around capture leave it off |

### ✅ Checkpoint 1b — masking (off by default here)

`configs/sample.yaml` sets `mask.enabled: false`, and that is a **measured**
choice, not an oversight. A gilded mask yields very few matchable features, so
the background is what actually carries the geometry. Masking it away collapsed
the match graph from 21.9% usable image pairs to 1.7%, and registration to 3/66.

Turn masking on **only** if QC reports the turntable regime — object rotating,
camera fixed — where the background genuinely is a competing rigid scene:

```bash
python scripts/00_prepare_images.py -c configs/sample.yaml \
    --input data/raw/sample/images -s mask.enabled=true
open data/runs/sample/figures/mask_previews/
```

Then check the previews: the green tint must cover the whole mask **including
the crown tips**. If ornament is being cut off, add `-s mask.erode_px=1`; if
`rembg` struggles on the gilding, try `-s mask.method=grabcut`.

Whichever you choose, confirm it with the match-graph numbers in Step 2 rather
than by eye.

---

## Step 2 — Structure from motion (10–40 min)

```bash
python scripts/01_sfm.py -c configs/sample.yaml
```

### ✅ Checkpoint 2 — the most important gate in the pipeline

Look for the final line:

```text
registered 84/84 images, 61234 3D points, mean reprojection error 0.412 px
```

| Criterion | Good | If not |
| --- | --- | --- |
| Registered images | ≥ 90% of input | See fixes below |
| **Models** | **1** | 2+ means parts of the capture never connected |
| Reprojection error | < 1.0 px | > 2 px means bad matches |
| Coverage warnings | none | Note gaps; they become holes |

> ⚠️ **Run it twice before believing it.** Reprojection error alone does not
> indicate success — it stayed at 0.955 px even on a run that registered just
> 5 of 49 images. With `sfm.mapper_num_threads=1` (already set) two runs must
> agree exactly. If they differ, determinism has broken and no number below is
> trustworthy.

Re-mapping is cheap: delete `sparse/` and `sfm_stats.json` but keep
`database.db`, and stage 1 reuses the features — ~25 s instead of ~15 min.

**If many images failed to register:**

1. Re-read the QC overlap warnings — gaps are the usual cause.
2. Inspect the mask previews again; over-aggressive masks starve SfM of features.
3. Try more features: `-s sfm.max_num_features=20000 --overwrite`
4. If the mask barely moved between shots, the capture may not have enough
   parallax — that is a reshoot, not a settings problem.

**If it registered the room instead of the mask** (huge point count, mask tiny
in the middle): masking failed. Fix Step 1b and re-run with `--overwrite`.

---

## Step 3 — Package for the GPU (2–5 min)

```bash
python scripts/02_dense_export.py -c configs/sample.yaml
```

Produces `data/runs/sample/dense_bundle_sample.zip`. Dense stereo needs CUDA
and **cannot run on this Mac** — this is expected, not an error.

---

## Step 4 — Dense reconstruction on Colab (20–60 min)

1. Open <https://colab.research.google.com> → **Upload** →
   `notebooks/colmap_dense_colab.ipynb`
2. **Runtime → Change runtime type → T4 GPU.** Do this first; the notebook
   stops immediately without it.
3. Run the cells in order. Cell 1 checks the GPU, cells 2–3 install a
   CUDA-enabled COLMAP and *assert* it has CUDA.
4. When prompted, upload `dense_bundle_sample.zip`. Over ~200 MB, use the
   Google Drive route commented in that cell instead — the upload widget is
   unreliable at size.
5. Patch-match stereo is the long step. Leave the tab open and **interact with
   the page occasionally** or Colab will disconnect the runtime.
6. The last cells sanity-check and download `fused.ply`.

If Colab hands you a session limit or no GPU, wait and retry — the bundle is
self-contained, so nothing is lost.

---

## Step 5 — Bring the dense cloud back (2 min)

```bash
python scripts/03_dense_import.py ~/Downloads/fused.ply -c configs/sample.yaml
```

### ✅ Checkpoint 5 — alignment

Expect `dense/sparse alignment OK (1234567 dense points vs 61234 sparse)`.

If it reports a failure, you almost certainly downloaded `fused.ply` from a
different run. Re-download and re-run. **Do not continue past a failed
alignment check** — every later stage would be built on the wrong geometry.

---

## Step 6 — Surface reconstruction (5–20 min)

```bash
python scripts/04_mesh.py -c configs/sample.yaml
```

### ✅ Checkpoint 6 — normals and holes

Two lines matter:

```text
92.8% of mesh normals face the observing cameras
mesh: 412034 vertices, 821553 triangles, 3 hole(s), watertight=False
```

- **Outward normals must be > 50%.** Below that the mesh is inside-out and
  texturing will colour almost nothing.
- A few holes are normal and expected in the eye sockets and mouth. Dozens of
  holes means the dense cloud is too thin.

Tuning, in order of usefulness:

| Symptom | Fix |
| --- | --- |
| Mesh looks lumpy / crown detail lost | `-s mesh.poisson_depth=11` |
| Mesh is noisy and spiky | `-s mesh.poisson_depth=9` |
| Smooth "balloons" over eye sockets | `-s mesh.density_trim_quantile=0.06` |
| Holes too aggressive | `-s mesh.density_trim_quantile=0.01` |

Inspect it before moving on. macOS has no built-in `.ply` viewer, so use MeshLab
(<https://www.meshlab.net>), or view it from Python:

```bash
python -c "import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_triangle_mesh('data/runs/sample/mesh/mesh.ply')])"
```

---

## Step 7 — Texture (10–30 min)

```bash
python scripts/05_texture.py -c configs/sample.yaml
```

`configs/sample.yaml` sets `texture.mode: uv`, producing
`mesh/mesh_textured.obj` plus a texture map — the deliverable model.

### ✅ Checkpoint 7

`unseen_fraction` should be small. A warning that >5% of the surface was never
seen tells you which viewpoints the capture missed — worth quoting in the paper's
limitations. If it reports something extreme like 99%, the normals are inverted:
go back to Checkpoint 6.

If the UV bake misbehaves, fall back to `-s texture.mode=vertex`, which always
works and is fine for the report's renders.

---

## Step 8 — Evaluation (10–30 min)

```bash
python scripts/06_evaluate.py -c configs/sample.yaml
```

### 📊 Numbers to copy into the paper

All of these land in `data/runs/sample/evaluation.json`:

| Paper claim | Field |
| --- | --- |
| Eq. (1) mean reprojection error | `reprojection.mean_px` |
| Registered images | `sfm.n_registered_images` |
| Dense point count / spacing | `density.n_points`, `density.median_spacing` |
| Holes and completeness | `completeness.n_holes`, `is_watertight` |
| Novel-view quality | `views.holdout.psnr_mean`, `ssim_mean` |
| Gilding prediction | `specularity.correlation_study.correlation` |

Quote `reprojection.mean_px`, **not** COLMAP's own figure — see
[CLAUDE.md](CLAUDE.md) for why the built-in value is stale.

The specularity correlation is the interesting one. Negative means gilded
regions did reconstruct worse, supporting the proposal's prediction. If it comes
out near zero or positive, **report that honestly** — a disproved expectation is
still a result, and a more interesting one.

If it is slow, `--skip-views` postpones the render comparison.

---

## Step 9 — Ablations (30–90 min)

```bash
python scripts/07_ablations.py -c configs/sample.yaml
```

Runs the reduced-overlap and bundle-adjustment comparisons the proposal
promises, and prints a summary table. Expect the overlap variants to register
fewer images and recover fewer points — that is the finding.

If a variant reconstructs *better* than the full run, the script warns you.
Investigate rather than reporting it at face value.

---

## Step 10 — Figures and tables (2 min)

```bash
python scripts/08_report.py -c configs/sample.yaml --all-runs
```

| Output | Paper section |
| --- | --- |
| `figures/reprojection_error.pdf` | Evaluation metrics |
| `figures/camera_coverage.pdf` | Image acquisition |
| `figures/track_lengths.pdf` | Camera pose estimation |
| `figures/specularity_vs_density.pdf` | Expected outcomes and challenges |
| `eval/views/compare_*.jpg` | Qualitative novel-view comparison |
| `tables/summary.tex`, `tables/ablations.tex` | Results — paste straight into LaTeX |

Figures are already at IEEE column width, as PDF for LaTeX and PNG for viewing.

---

## Final checklist

- [ ] QC ran and every warning is either fixed or written down as a limitation
- [ ] Mask previews inspected by eye
- [ ] ≥90% of images registered, reprojection error < 1 px
- [ ] Dense/sparse alignment check passed
- [ ] Outward normal fraction > 0.5
- [ ] Textured mesh opens and looks like the mask
- [ ] `evaluation.json` saved; numbers copied into the paper
- [ ] Ablations run; the table makes sense
- [ ] Figures regenerated
- [ ] `data/runs/sample/` backed up — it holds every number in the paper

---

## Quick reference

```bash
# re-run one stage from scratch
python scripts/0N_*.py -c configs/sample.yaml --overwrite

# override any setting without editing YAML
python scripts/04_mesh.py -c configs/sample.yaml -s mesh.poisson_depth=11

# what was actually run, with versions and timings
cat data/runs/sample/manifest.json

# start over completely
rm -rf data/runs/sample
```

Stuck? `data/runs/sample/pipeline.log` holds the full log of every stage.
