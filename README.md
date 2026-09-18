# 3D Reconstruction of a Gilded Khon Mask

A classical photogrammetry pipeline — structure from motion, multi-view
stereo, Poisson surface reconstruction — that turns phone photographs of a
Thai Khon mask into a textured 3D mesh. Built for a Computer Vision course
project, with an emphasis on checking that every number is actually correct
rather than just plausible.

The mask is gilded, which makes it a genuinely hard subject: gold leaf is
specular, so its appearance changes with viewing angle, which breaks the
assumption feature matching relies on (that a point looks roughly the same
from different views). Two captures were shot over the course of the
project — a pilot that exposed exactly where a partial walk-around fails,
and a full reshoot that fixed it — and both are reported below rather than
just the second, cleaner one.

```text
photographs → SfM poses (COLMAP) → dense MVS cloud → Poisson mesh + texture → evaluation
```

| | |
| --- | --- |
| Input | 199 phone photographs (iPhone 17), walk-around capture — 145 registered |
| Output | 684,705-vertex textured mesh, 6.9% of surface unseen |
| Azimuth coverage | 358.2° (a 9.0° gap) — full 360° minus one small notch |
| Eq. (1) reprojection error | 0.757 px over 89,926 observations |
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
masking collapsed the usable-pair rate from 21.9% to 1.7% on the pilot
capture, the specularity correlation came out at r ≈ −0.03 on the pilot and
r ≈ −0.002 on the final capture (essentially nothing, both times, in the
same direction), and reprojection error actually *improves* as the
reconstruction falls apart.

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

**A capture that covers the full circle can still fail to reconstruct as one
piece.** The final capture walked a full loop above eye level and a full
loop below it — but as two separate passes, not one continuous sweep. COLMAP
registered each loop internally but found zero features in common between
them: two disconnected models, not one, confirmed at both the putative and
RANSAC-verified match level across every possible image pair. The mask was
physically damaged partway through processing, ruling out a reshoot to
bridge the gap, so the fix had to be a decision (keep the larger loop) rather
than a code change. `CAPTURE_GUIDE.md` documents the shooting rule this
taught: shoot elevation as one spiral, never as separate top/bottom sessions.

**A background job that reported success hadn't actually finished.** Baking
a UV-mapped texture — the `.obj` format the course asked for — needs every
camera view held in memory at once for the median blend that defends against
specular highlights. At this capture's scale that's 40GB+ against a 16GB
machine; the OS silently killed the process, and the task runner then
reported "exit code 0," indistinguishable from success unless the actual
output file is checked. The mesh ships as a vertex-colored `.ply` instead —
a verified, deliberate substitution, not a silently missing deliverable. The
underlying bug is unfixed and documented in `CLAUDE.md` for anyone who wants
to raise the texture resolution or photo count further.

---

## Results on the final capture (`full_capture`)

Full run recorded in `data/runs/full_capture/` (not checked in — it's large,
but reproducible from the photographs plus `configs/full_capture.yaml`).

This capture fixed the pilot's main flaw — it covers 358.2° of azimuth, a
9.0° gap instead of a 216° one — by walking the full circle at two heights
instead of one. But the two heights were shot as separate passes (see above),
which cost the capture its below-eye-level coverage: SfM produced two
disconnected models, a 117-image loop from above and a 42-image loop from
below, with zero shared features between them. With no reshoot possible, the
larger loop was kept; the smaller one's 42 images were dropped from this
model rather than merged into it.

| | |
| --- | --- |
| Registered images | 145/199 (72.9%), including 28/39 held-out views registered afterward |
| SfM models found | 2, disconnected — 117-image loop kept, 42-image loop dropped |
| Sparse points, mean track length | 15,788, 5.70 |
| Eq. (1) mean reprojection error | 0.757 px over 89,926 observations |
| Dense cloud | 563,497 points (242k kept after removing background/pedestal fragments), median spacing 0.0026 |
| Mesh | 684,705 vertices, 1,363,343 triangles, 684 holes, not watertight |
| Outward normals / unseen surface | 83.5% / 6.9% |
| Novel-view (held-out, n=28) | PSNR 15.28 ± 1.51 dB, SSIM 0.350 ± 0.101 |
| Azimuth / elevation coverage | 358.2° (9.0° gap) / 11.5°–42.2° (30.6° span, above only) |
| Specularity correlation | r = −0.0020 |

**The 30.6° elevation span is the honest cost of the dropped loop.** Azimuth
coverage is essentially solved, but every registered camera is still above
eye level — the below-eye-level views exist as photographs but never became
part of this model. That is a coverage gap, not a mesh defect: nothing in
post-processing can recover geometry from cameras that never joined the
reconstruction.

**Overlap ablations show the same "error stays flat while geometry
collapses" pattern as the pilot, just less severe.** Halving overlap still
registered its whole reduced set (63/63) but pushed the azimuth gap to
53.7°; cutting to a third of overlap dropped registration to 34 images and
the gap to 109.7°. Reprojection error barely moved (0.793 px, 0.858 px) —
this capture had more overlap margin to spend before anything visibly broke.

**Masking made no measurable difference here either** — `no_mask` registered
the same 145/145 images at 0.756 px against the masked full run's 0.758 px —
consistent with the pilot's finding that this subject's low-texture gilded
surface needs the background for SIFT to have anything to match.
`mask.enabled: false` throughout.

**Bundle adjustment mattered far less here than on the pilot.** Isolated on
a fixed model, it improved Eq. (1) by 3.0% (0.7655 → 0.7423 px), against
17.3% on `sample`. A larger, better-overlapped capture needs less correction
to begin with.

**The background itself densified more than expected.** Unlike the pilot's
plain backdrop, this session was shot in a real room — windows, walls, a
patterned floor — and dense MVS reconstructed a lot of it: the stand and
background edges came to 34% of the dense cloud, above the pipeline's 25%
default discard limit. Verified by eye (projected the discarded cluster and
confirmed it was the stand and wall edges, not mask fragments) before
raising `mesh.cluster_max_removed_fraction` to 0.65 in `configs/full_capture.yaml`.

---

## Results on the pilot capture (`sample`)

Full run recorded in `data/runs/sample/` (not checked in — it's large, but
reproducible from the photographs plus `configs/sample.yaml`). This capture
came first and directly motivated the reshoot above: its azimuth gap and
above-only elevation are exactly the two problems `full_capture` set out to
fix.

Viewed from every 45° around the mesh, between roughly 90° and 270° you're
looking into the model rather than at it — the capture only covered 143.8° of
azimuth, so there's no back surface. No amount of processing fixes that after
the fact; it needs a fuller capture, which is what the reshoot above was.

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
only computed over the images that did register. In the ablations,
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
removed, not real detail being lost. So the 55 holes in this mesh are
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
python scripts/00_prepare_images.py -c configs/full_capture.yaml --input data/raw/full_capture/originals

# 1. sparse reconstruction (poses + sparse cloud + Eq. (1) error)
python scripts/01_sfm.py -c configs/full_capture.yaml

# 2. package for the GPU stage
python scripts/02_dense_export.py -c configs/full_capture.yaml
#    -> run notebooks/colmap_dense_colab.ipynb on Colab, download fused.ply

# 3. bring the dense cloud back (checks it matches the sparse model)
python scripts/03_dense_import.py ~/Downloads/fused.ply -c configs/full_capture.yaml

# 4-5. surface reconstruction and colour recovery
python scripts/04_mesh.py    -c configs/full_capture.yaml
python scripts/05_texture.py -c configs/full_capture.yaml

# 6-8. evaluation, ablations, figures
python scripts/06_evaluate.py  -c configs/full_capture.yaml
python scripts/07_ablations.py -c configs/full_capture.yaml
python scripts/08_report.py    -c configs/full_capture.yaml
```

`configs/sample.yaml` runs the same eight stages over the smaller pilot
capture instead — useful as a faster end-to-end check while iterating.

Any config value can be overridden per-run without editing files, e.g.
`python scripts/04_mesh.py -c configs/full_capture.yaml -s mesh.poisson_depth=9`.

Every stage renders itself from the same registered cameras (photograph,
sparse SfM, dense MVS, Poisson mesh, textured mesh), so the stages can be
compared directly — renders from different angles can't be.

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
photos with a phone — including a full log of what went right and wrong on
both captures above.

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
configs/              default.yaml, sample.yaml, full_capture.yaml
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

**Held-out views are genuinely held out.** Every k-th image is left out of
mapping and registered afterward with the geometry fixed
(`sfm.holdout_every`), so the novel-view scores measure actual reconstruction
quality instead of how well the model fits images it already saw.

**Camera coverage is measured in the reconstruction's own up/right/front
frame, not raw world Z.** COLMAP's world frame inherits the camera
convention (+Y points down), so azimuth/elevation must be decomposed against
`mesh.upright_transform`'s recovered frame — using raw Z understated the
pilot capture's true elevation span by 3x before this was caught.

**The "no bundle adjustment" ablation is really minimal-vs-full
refinement.** COLMAP can't triangulate at all with BA fully removed, so
`ablations.bundle_adjustment_isolation` instead takes one minimally refined
model and runs a full global BA over it, changing nothing else.

---

## Known limitations

- Dense MVS needs the Colab round trip since there's no CUDA locally. A local
  OpenMVS backend could slot in behind the interface in `dense.py`.
- SfM is only reproducible from a fixed feature database. With
  `sfm.multiple_models=true` and `sfm.mapper_num_threads=1`, re-mapping an
  existing `database.db` is bit-identical. A fresh feature extraction is
  not — registration counts vary by roughly 10% between otherwise-identical
  runs on both captures. So a single registration count should never be
  quoted alone, and ablations should be read against their own control run
  rather than the main one, since each builds its own database.
- Open3D 0.19's Poisson step fails intermittently (`Failed to close loop`)
  while exiting with status zero, so `mesh.py` runs it in a subprocess and
  checks for the actual output file rather than the exit code, retrying on
  failure.
- The mesh is written in two orientations. The canonical file stays in
  COLMAP's world frame (+Y down) because the evaluation stage renders it from
  the estimated camera poses; a `*_upright.ply` copy is rotated and centred
  for viewing. Open the upright one in a mesh viewer, or it'll look upside
  down and facing away.
- Masking is off for both captures because it measurably hurts them — the
  gilded surface has too few matchable features on its own, so the
  background is doing most of the work. Masking is still the right call for
  a genuine turntable capture; check the match graph rather than assuming.
- `full_capture`'s elevation coverage stops at eye level upward: the
  below-eye-level loop shot for this capture never merged into the model
  (see "known failure modes" in `CAPTURE_GUIDE.md`), so recessed
  underside geometry is missing data, not a meshing defect.
- `texture.mode: uv` — the UV-mapped `.obj` output — is unverified at this
  capture's scale: `gather_multiview_colors` needs every camera view in
  memory at once for its median blend, which exceeded 16GB and was silently
  killed by the OS. `texture.mode: vertex` has no equivalent failure mode
  and is what both captures shipped with; see `CLAUDE.md` before trying
  `uv` mode on a large capture.
- Dense MVS runs at `max_image_size: 1600` by default, well under the
  photographs' actual resolution. Raising it is the cheapest quality
  improvement available and needs no reshoot, just another Colab run.
