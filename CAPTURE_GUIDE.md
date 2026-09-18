# Khon mask capture guide

A checklist for shooting a subject like this with a phone, for classical
photogrammetry (SfM → MVS → Poisson → texture). Everything here follows from
how the pipeline actually works, not from generic photography advice — and
two real attempts on this project (logged at the bottom) back up every rule.

---

## The two things that matter most

1. **Walk all the way around — a full 360°, not a partial arc.** Any azimuth
   gap becomes a hole in the model that no processing step can fill in
   afterward: it's missing data, not a defect to clean up.
2. **Cover your elevation range (above and below eye level) as ONE
   continuous spiral, not separate disconnected passes.** If you shoot "the
   top" and "the bottom" as two unrelated loops with no shared viewpoints
   between them, SfM can fail to merge them into a single model — even
   though each loop reconstructs perfectly on its own. See "known failure
   modes" below; this is not a hypothetical, it happened on this project's
   second capture and could not be fixed without a reshoot.

---

## Walk around it — don't put it on a turntable

Structure from motion assumes the scene stays still. If you rotate the mask
in front of a fixed camera, the *background* is what stays rigid, and COLMAP
will happily reconstruct the room instead of the mask — with a low
reprojection error, looking completely fine, and being completely wrong.
This failure does not show up unless you check for it.

Walking around a stationary mask avoids the problem entirely and needs no
special setup. If you do need to rotate the object instead (e.g. it has to
stay clamped down), set `mask.enabled: true` and check the mask previews
before trusting the result — `scripts/00_prepare_images.py` will warn you if
it detects a static background, which is the tell.

---

## Before you shoot

- [ ] Lock focus and exposure — tap and hold on the phone screen until it locks
      (AE/AF lock), so it doesn't refocus or re-expose between shots.
- [ ] Turn the flash off — it moves the highlight with the camera, which is
      the worst case for multi-view matching.
- [ ] Wipe the lens.
- [ ] Decide up front whether the mask can be reshot if the first pass comes
      up short. If not — e.g. it's fragile, borrowed, or on a deadline —
      shoot more than you think you need and run the QC (below) before
      packing up, not after.

## Light and background

- [ ] Even, diffuse light — daylight near a window or open shade outside is
      enough. Avoid a single hard light source or direct sun.
- [ ] Gilded or shiny decoration will throw specular highlights under point
      light sources — this is the main thing that breaks feature matching on
      this subject, so soft light matters more than usual.
- [ ] A plain-ish background helps but doesn't need to be a studio backdrop —
      a wall or a table works. Keep anything shiny out of frame. If your
      background is a real room (windows, furniture, textured walls) rather
      than a plain backdrop, expect the dense cloud to include a lot of it —
      raise `mesh.cluster_max_removed_fraction` above the default rather than
      assume something broke (verify by eye first: project the discarded
      cluster and confirm it's background, not object).

## The shoot

- [ ] **60–100 photos** for a single-band walk-around; more (150–250) if
      you're covering multiple heights, since each height band needs its own
      overlapping sequence.
- [ ] **Small steps.** Consecutive photos should overlap by 60–70% — if the
      mask visibly "jumps" between two shots, you moved too far.
- [ ] **One continuous spiral across heights, not separate passes.** Walk the
      full circle at eye level, then keep circling while gradually raising or
      lowering the camera, rather than doing a full loop up high and then a
      second, entirely separate full loop down low. The transition shots are
      what let SfM tie the height bands into one model — without them, two
      well-shot loops can fail to merge at all, however good each looks on
      its own.
- [ ] **Three-plus heights**: eye level, above looking down (for the crown),
      and below looking up (for the jaw and underside) — connected by the
      spiral above, not shot as isolated sessions.
- [ ] Extra close-ups of fine detail (crown ornament, painted features),
      still overlapping their neighbors.
- [ ] Shoot into recesses — eye sockets, an open mouth — from a few angles.
      These become holes if under-photographed, and more viewpoints is the
      only fix.
- [ ] Keep a roughly constant distance from the mask.
- [ ] Don't switch lenses or zoom mid-shoot. If you do, set
      `sfm.single_camera: false`.

## Before you're done

Run the QC — it takes under a minute and tells you whether to keep shooting
while the mask is still set up:

```bash
python scripts/00_prepare_images.py -c configs/<your-config>.yaml --input /path/to/photos
```

Check that it reports:

- [ ] no images below the sharpness floor
- [ ] exposure drift within tolerance
- [ ] no consecutive pair below the overlap threshold
- [ ] masking status matches whether you walked around or used a turntable

If there's time, run SfM before packing up:

```bash
python scripts/01_sfm.py -c configs/<your-config>.yaml
```

- [ ] nearly all images registered, and **only one connected model** — check
      the log for "mapping produced N disconnected models"; if N > 1, you
      have a coverage gap between height bands or viewpoints *right now*,
      while you can still fix it with a few more photos
- [ ] **`largest_azimuth_gap_deg` under ~40°**. Printed by stage 6 and stored
      in `evaluation.json`; a large gap means walk further around, right now.
- [ ] elevation span of at least ~30°, ideally including some views from below
- [ ] mean reprojection error below ~1 px

Don't read a low reprojection error alone as success — it's only computed over
the images that registered, so it actually *improves* as reconstruction falls
apart (a run registering 6 of 17 images scored 0.837 px; the full run scored
1.177 px). Always read it next to the registered count.

Registration also varies a bit between identical runs on the same photos, so
one so-so number isn't proof the shoot failed — re-run before deciding to
reshoot.

---

## Known failure modes

### Azimuth gap (partial walk-around)

A capture that only covers part of the circle reconstructs an **open shell
with no back** — no amount of processing fixes that afterward, because the
data for the missing side was never taken. Fix: walk the full circle;
`largest_azimuth_gap_deg` catches this before you leave.

### Disconnected loops (separate height passes that never overlap)

If two sets of photos never share a viewpoint — e.g. a full loop shot high
above the object, then a full loop shot low below it, with nothing bridging
the two — SfM can register each loop internally but find **zero matching
features between them**. The mapper then reports multiple disconnected
models and keeps only the largest; the rest of the images are simply dropped,
however sharp and well-overlapped they are individually. This is invisible
until you check `n_models` in the SfM stats — reprojection error and
per-image registration both look completely normal inside each loop.

Fix: shoot elevation as one continuous spiral (see "the shoot" above), not as
separate top/bottom sessions. If you only discover the split after the fact,
the only recovery is more photos connecting the two bands — there is no
software fix for a viewpoint that was never taken.

---

## Attempts on this project

Two real captures were shot for this project. Both are logged here so the
numbers above aren't abstract — this is what actually happened.

### Attempt 1 — `sample`, 2025-08-30

49 photos, iPhone 15 Pro Max (two lenses), single-band walk-around at roughly
one height, no turntable.

| Measured | Result |
| --- | --- |
| Azimuth coverage | 143.8° (a 216° gap) |
| Elevation range | −67° to −8° — all from above eye level |
| Registered | ~82% (71–92% across identical reruns) |
| Reprojection error | ~0.86 px (from a fixed database) |

**Verdict:** reconstructed cleanly but is an open shell — the 216° azimuth
gap and above-only elevation meant no back of the mask and nothing from
below. The 55 holes in the final mesh were checked and are genuinely
unobserved regions, not a processing artifact
(`data/runs/sample/mesh_sweep.json`). This is the run the "walk all the way
around" rule above was written to prevent on the next attempt.

### Attempt 2 — `full_capture`, 2026-09-15

199 photos (after discarding 19 that caught the mask mid-damage — see
below), iPhone 17, single lens, walk-around covering both above and below eye
level, shot as two separate passes (a full loop from above, then a full loop
from below) rather than one continuous spiral.

Partway through post-processing, **the physical mask was damaged and no
reshoot was possible** — every decision from that point on had to work with
the existing photos only.

| Measured | Result |
| --- | --- |
| SfM models produced | **2, disconnected** — 117 images ("above" loop), 42 images ("below" loop) |
| Shared features between loops | **zero**, confirmed at both the putative and RANSAC-verified level across all possible image pairs |
| Registered (after proceeding with the larger loop only) | 145/199 (72.9%), including 28/39 held-out views registered afterward |
| Azimuth coverage | 358.2° (9.0° gap) — excellent, because the loop itself was a full circle |
| Elevation range | 11.5°–42.2° (30.6° span) — all from above; the "below" loop never made it into the model |
| Eq. (1) reprojection error | 0.758 px (`full` ablation variant) |
| Held-out view PSNR | 15.3 ± 1.5 dB |

**What went wrong:** shooting "above" and "below" as two separate full loops,
with no transition shots connecting them, meant the two passes shared no
viewpoints an SfM matcher could use to tie them together — each is
individually a perfectly good 360° reconstruction, but they could not be
merged into one. With reshoots off the table, the only viable path was to
proceed with the larger loop alone: full azimuth coverage, but no genuine
below-eye-level geometry despite having shot it.

**Also encountered on this capture** (separate from the loop issue, noted
here since they'd otherwise surprise a repeat attempt):

- 19 of the original photos caught the mask mid-damage during the session and
  had to be discarded — not a processing artifact, confirmed against
  neighboring frames.
- The real-room background (unlike `sample`'s plain backdrop) was
  texture-rich enough that dense MVS reconstructed a lot of it — the stand
  and background edges came to 34% of the dense cloud, requiring
  `mesh.cluster_max_removed_fraction` to be raised from the 25% default
  (verified by eye before raising it, not assumed).
- HEIC photos needed converting to JPEG before ingest (`khon_recon/io_utils.py`
  doesn't recognize `.heic`), and the conversion needs to preserve EXIF
  (Model, FocalLength, Orientation) or COLMAP loses camera grouping and the
  Colab undistorter aborts.

**Lesson for the next attempt:** shoot elevation as one continuous spiral,
never as separate top/bottom sessions — this is now the "the shoot" rule
above, learned directly from this run.
