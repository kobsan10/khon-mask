# 3D Reconstruction of a Gilded Khon Mask

A classical photogrammetry pipeline — structure from motion, multi-view
stereo, Poisson surface reconstruction — that turns a set of phone photographs
of a Thai Khon mask into a textured 3D mesh. Built for a Computer Vision course
project, with an emphasis on checking that every number is actually correct
rather than just plausible.

The mask is gilded, which makes it a genuinely hard subject: gold leaf is
specular, so its appearance changes with viewing angle, which breaks the
assumption feature matching relies on (that a point looks roughly the same
from different views). A lot of what's interesting below comes from that.

```text
photographs → SfM poses (COLMAP) → dense MVS cloud → Poisson mesh + texture → evaluation
```

| | |
| --- | --- |
| Input | Phone photographs, iPhone 15 Pro Max, two lenses |
| Output | 255,925-vertex textured mesh, 0.98% of surface unseen |
| Eq. (1) reprojection error | 1.0514 px over 14,653 observations |
| Stack | COLMAP 4.1.1 / pycolmap, Open3D 0.19, OpenCV, NumPy — Python 3.11 |
| Compute | Sparse reconstruction and meshing on a Mac; dense MVS on a Colab T4 GPU |

Dense stereo needs CUDA, which isn't available on Apple Silicon, so the
pipeline splits across two machines with a bundle export/import step in
between (`scripts/02_dense_export.py` / `scripts/03_dense_import.py`).

---

## A few things worth knowing before the results

**Three of my assumptions turned out to be wrong, and I kept the wrong
answers in below rather than dropping them.** I expected masking the
background to help, expected the gilding to correlate with reconstruction
failure, and treated reprojection error as a quality signal. Measured:
masking collapsed the usable-pair rate from 21.9% to 1.7%, the specularity
correlation came out at r = −0.034 (essentially nothing), and reprojection
error actually *improves* as the reconstruction falls apart.

**Reproducibility isn't automatic, and I checked it rather than assumed it.**
Five identical runs at the defaults registered anywhere from 10% to 94% of the
same set of images, which turned out to be a non-deterministic seed-pair choice
plus multi-threaded bundle adjustment. After fixing both, the result is
*mostly* reproducible — from a fixed feature database it's bit-identical, but
a fresh feature extraction still shifts the registered count by around 10
images. So results below are quoted as a spread, not a single number.

**Two bugs produced confident, wrong output that looked fine at a glance.**
A Poisson mesh that rendered normally in a viewer was actually inside-out,
which meant texturing had almost nothing to project onto. And Open3D's
Poisson step fails on roughly a third of runs while still exiting with
status zero, so a plain return-code check reports success on a mesh that was
never written.

---

## Results on the sample capture

Full run recorded in `data/runs/sample/` (not checked in — it's large, but
reproducible from the photographs plus `configs/sample.yaml`).

Viewed from every 45° around the mesh, between roughly 90° and 270° you're
looking into the model rather than at it — the capture only covered 143.8° of
azimuth, so there's no back surface. That's the main limitation here, and no
amount of processing fixes it after the fact; it needs a fuller capture.

| | |
| --- | --- |
| Registered images | ~82% (ranging ~71–92% across identical runs) |
| Sparse points, mean track length | 2,940, 4.98 |
| Eq. (1) mean reprojection error | 1.0514 px over 14,653 observations |
| Dense cloud | 105,545 points, median spacing 0.0054 |
| Mesh | 255,925 vertices, 506,860 triangles, 55 holes, not watertight |
| Outward normals / unseen surface | 98.5% / 0.98% |
| Novel-view (held-out, n=7) | PSNR 16.73 dB, SSIM 0.364 |
| Azimuth coverage | 143.8° (a 216° gap) |
| Specularity correlation | r = −0.034 |

**Masking hurt this capture.** The gilded surface gives almost no matchable
features on its own, so the background actually carries most of the
geometry — only around 40% of sparse points land on the mask. Masking it out
dropped usable image pairs from 21.9% to 1.7% and registration to 3/66, so
`mask.enabled: false` is a measured choice, not a default I forgot to flip.

**Reprojection error falls as the reconstruction gets worse**, because it's
only computed over the images that did register. In the ablations below,
`overlap_third` has the *best* error in the table (0.837 px) while
registering just 6 of 17 images — the full run scores 1.177 px. I never quote
this number without the registered count next to it (for reference, the full
distribution is median 0.757 px, p95 2.779 px).

**The specularity correlation doesn't support the hypothesis.** r = −0.034 is
close to zero, though specular blocks did recover about 0.78x the points of
matte ones. Only 0.8% of blocks were specular enough to register at all, so
the study is underpowered on this capture — an open question, not a
confirmation.

**Reducing overlap collapses the reconstruction rather than gradually
degrading it.** Half the images registered 13 of 25; a third registered 6 of
17. With only 143.8° of azimuth to begin with, there wasn't much overlap
margin to give up.

Bundle adjustment, isolated on one fixed model so nothing else changes,
improved Eq. (1) from 1.2804 to 1.0590 px — a 17.3% improvement, the cleanest
single measurement in the study.

### Are the holes real, or just an aggressive trim?

Poisson reconstruction is watertight by construction — it will invent
surface anywhere no camera looked. Trimming low-density vertices removes
that, but trim too much and you cut real geometry, so I checked which one
was happening rather than guessing. A mesh vertex further than 3x the dense
cloud's median point spacing from any observed point counts as invented:

| depth | trim | vertices | holes | invented |
| --- | --- | --- | --- | --- |
| 9 | 2% | 131,118 | 2 | 12.9% |
| 10 | 2% | 268,867 | 20 | 5.4% |
| 10 | 6% | 255,925 | 55 | 1.8% |
| 11 | 6% | 285,766 | 68 | 1.4% |

Raising the trim from 2% to 6% triples the hole count but cuts invented
surface from 5.4% to 1.8% — those extra holes are fabricated geometry being
removed, not real detail being lost. So the 55 holes in the final mesh are
genuinely missing data, and I can report them as that instead of hedging.

---

## Setup

```bash
conda env create -f environment.yml
conda activate khon
brew install colmap          # 4.1.1, arm64 bottle
```

Python is pinned to 3.11: Open3D 0.19 only publishes macOS wheels for
cp310–cp312, so the meshing stage won't install on newer Pythons.

```bash
python -c "import open3d, pycolmap, cv2; print(open3d.__version__, pycolmap.__version__)"
colmap --help | head -1
```

### Dense stereo runs on Colab, not locally

`patch_match_stereo` needs CUDA. On Apple Silicon COLMAP reports itself as
built "without CUDA," so dense stereo can't run at all here. Sparse SfM,
meshing, texturing, and evaluation all run locally; only densification goes
to a free Colab GPU via `notebooks/colmap_dense_colab.ipynb`.

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

# 6-8. evaluation, ablations, figures
python scripts/06_evaluate.py  -c configs/sample.yaml
python scripts/07_ablations.py -c configs/sample.yaml
python scripts/08_report.py    -c configs/sample.yaml
```

Any config value can be overridden per-run without editing files, e.g.
`python scripts/04_mesh.py -c configs/sample.yaml -s mesh.poisson_depth=9`.

Every stage renders itself from the same three registered cameras (photograph,
sparse SfM at 2,940 points, dense MVS at 105,545 points, Poisson mesh, textured
mesh), so the stages can be compared directly — renders from different angles
can't be.

### No photographs yet?

Both of these exercise the full pipeline before you have real photos:

```bash
# deterministic smoke test on COLMAP's own sample dataset
python scripts/00_prepare_images.py --fetch-sample south-building --sample-limit 25 \
    -s paths.subject=sample -s paths.run_id=smoke -s mask.enabled=false
python scripts/01_sfm.py -s paths.subject=sample -s paths.run_id=smoke -s mask.enabled=false

# rehearsal on a phone video of any household object
python scripts/00_prepare_images.py --video ~/orbit.mov -s paths.subject=dryrun
```

See [CAPTURE_GUIDE.md](CAPTURE_GUIDE.md) for how to actually shoot the
photos with a phone.

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
  previews.py         per-stage renders from fixed viewpoints, for comparison
  report.py           figures + LaTeX tables
scripts/              numbered CLI stages, 00 -> 08
configs/              default.yaml, sample.yaml
notebooks/            the Colab dense stage
data/                 gitignored: raw/<subject>/, runs/<run_id>/
```

Every stage writes into `data/runs/<run_id>/` alongside its resolved config
and a manifest of package versions and timings, so any number here traces
back to the run that produced it. Ablations are just different configs
against the same code, rather than forked scripts.

---

## What the evaluation actually measures

There's no ground-truth 3D scan to compare against, so quality has to come
from the reconstruction pipeline itself:

| Metric | Where |
| --- | --- |
| Mean reprojection error, Eq. (1) | `metrics.reprojection_errors` |
| Track lengths, camera coverage | `metrics.track_statistics`, `metrics.camera_coverage` |
| Dense density, holes, watertightness | `metrics.point_cloud_density`, `metrics.mesh_completeness` |
| Novel-view PSNR/SSIM on held-out views | `compare.evaluate_views` |
| Specularity vs reconstruction density | `specularity.run_specularity_study` |

A few implementation details worth knowing:

**Eq. (1) is computed from scratch, not read from COLMAP's own function.**
`compute_mean_reprojection_error()` returns a cached per-point value from the
last bundle adjustment and doesn't refresh it — after held-out views are
registered (which adds observations to existing tracks), it quietly reports
the stale figure. `metrics.verify_against_builtin` checks the two agree
exactly on an unmodified model, then the recomputed value is what's used
everywhere else.

**Held-out views are genuinely held out.** Every 5th image is left out of
mapping and registered afterward with the geometry fixed
(`sfm.holdout_every`), so the novel-view scores measure actual reconstruction
quality instead of how well the model fits images it already saw.

**The "no bundle adjustment" ablation is really minimal-vs-full
refinement.** COLMAP can't triangulate at all with BA fully removed, so
`ablations.bundle_adjustment_isolation` instead takes one minimally refined
model and runs a full global BA over it, changing nothing else.

---

## Known limitations

- Dense MVS needs the Colab round trip since there's no CUDA locally. A local
  OpenMVS backend could slot in behind the interface in `dense.py`.
- SfM is only reproducible from a fixed feature database. With
  `sfm.multiple_models=true` and `sfm.mapper_num_threads=1` (both set in
  `configs/sample.yaml`), re-mapping an existing `database.db` is
  bit-identical. A fresh feature extraction is not — four identical runs
  registered between about 71% and 92% of the images. So a single registration count
  should never be quoted alone, and ablations should be read against their
  own control run rather than the main one, since each builds its own
  database.
- Open3D 0.19's Poisson step fails intermittently (`Failed to close loop`)
  while exiting with status zero, so `mesh.py` runs it in a subprocess and
  checks for the actual output file rather than the exit code, retrying on
  failure.
- The mesh is written in two orientations. The canonical file stays in
  COLMAP's world frame (+Y down) because the evaluation stage renders it from
  the estimated camera poses; a `*_upright.ply` copy is rotated and centred
  for viewing. Open the upright one in a mesh viewer, or it'll look upside
  down and facing away.
- Masking is off for this capture because it measurably hurts it — the
  gilded surface has too few matchable features on its own, so the
  background is doing most of the work. Masking is still the right call for
  a genuine turntable capture; check the match graph rather than assuming.
- The capture spans only 143.8° of azimuth, all from above, so the result is
  an open shell with no back. The 55 holes in the mesh were checked and are
  genuinely unobserved regions, not a processing artifact — see
  [CAPTURE_GUIDE.md](CAPTURE_GUIDE.md) for what a fuller shoot would need.
- Dense MVS runs at `max_image_size: 1600` by default, about a third of the
  photographs' actual resolution. Raising it is the cheapest quality
  improvement available and needs no reshoot, just another Colab run.
