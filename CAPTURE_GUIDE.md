# Khon mask capture guide

A checklist for shooting the mask with a phone. Everything here follows from
how the pipeline actually works — the numbers are from the real capture used
in this project.

---

## What the first capture got wrong

The first capture reconstructed fine but is limited by one thing worth fixing
on a reshoot:

| Measured | Got | Should be |
| --- | --- | --- |
| **Azimuth coverage** | **143.8°** (a 216° gap) | 360° |
| Elevation range | −67° to −8°, all from above | above *and* below |
| Registered | ~82% (ranging ~71–92% across identical runs) | >90% |

The 216° gap means the model is an **open shell with no back** — no amount of
processing fixes that afterward. The 55 holes in the final mesh were checked
and are genuinely unobserved regions, not a processing artifact
(`data/runs/sample/mesh_sweep.json`).

So the single most useful thing to do differently: **walk all the way around**
the mask, and get some shots from below as well as above.

---

## Walk around it — don't put it on a turntable

Structure from motion assumes the scene stays still. If you rotate the mask in
front of a fixed camera, the *background* is what stays rigid, and COLMAP will
happily reconstruct the room instead of the mask — with a low reprojection
error, looking completely fine, and being completely wrong. This failure does
not show up unless you check for it.

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

## Light and background

- [ ] Even, diffuse light — daylight near a window or open shade outside is
      enough. Avoid a single hard light source or direct sun.
- [ ] Gilded or shiny decoration will throw specular highlights under point
      light sources — this is the main thing that breaks feature matching on
      this subject, so soft light matters more than usual.
- [ ] A plain-ish background helps but doesn't need to be a studio backdrop —
      a wall or a table works. Keep anything shiny out of frame.

## The shoot

- [ ] **60–100 photos**, walking in a full circle around the mask.
- [ ] **Small steps.** Consecutive photos should overlap by 60–70% — if the
      mask visibly "jumps" between two shots, you moved too far.
- [ ] **Three heights**: eye level, above looking down (for the crown), and
      below looking up (for the jaw and underside).
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
python scripts/00_prepare_images.py -c configs/sample.yaml --input /path/to/photos
```

Check that it reports:

- [ ] no images below the sharpness floor
- [ ] exposure drift within tolerance
- [ ] no consecutive pair below the overlap threshold
- [ ] masking status matches whether you walked around or used a turntable

If there's time, run SfM before packing up:

```bash
python scripts/01_sfm.py -c configs/sample.yaml
```

- [ ] nearly all images registered
- [ ] **`largest_azimuth_gap_deg` under ~40°** — the check the first capture
      would have failed at 216°. Printed by stage 6 and stored in
      `evaluation.json`; a large gap means walk further around, right now.
- [ ] elevation span of at least ~30°, ideally including some views from below
- [ ] mean reprojection error below ~1 px

Don't read a low reprojection error alone as success — it's only computed over
the images that registered, so it actually *improves* as reconstruction falls
apart (a run registering 6 of 17 images scored 0.837 px; the full run scored
1.177 px). Always read it next to the registered count.

Registration also varies a bit between identical runs on the same photos
(ranging roughly 71–92% across four runs here), so one so-so number isn't
proof the shoot failed — re-run before deciding to reshoot.
