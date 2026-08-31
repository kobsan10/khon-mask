# Khon mask capture guide

A checklist for the photo session. Museum or collection access is the one
irreplaceable resource in this project — a reshoot may not be possible, so the
pipeline should already have been rehearsed on a stand-in object before the
mask is in front of you.

Requirements below come from the project proposal; the rest is what the code
checks for.

---

## The one decision that matters most: turntable or walk-around?

**Do not rotate the mask in front of a fixed camera unless you also mask the
background.**

Structure from motion assumes the scene is rigid. If the mask turns while the
room stays still, the *room* is the rigid scene, and COLMAP will reconstruct it
instead — producing a confident, low-reprojection-error, completely wrong
result. The failure is silent.

Two safe options:

| Regime | What to do | Notes |
| --- | --- | --- |
| **Walk-around** (preferred) | Mask stationary, you move around it | No ambiguity. Works without masks. |
| **Turntable** | Mask rotates, camera fixed | Requires `mask.enabled: true`. Check `figures/mask_previews/` before trusting anything. |

`scripts/00_prepare_images.py` detects a static background automatically and
warns you. Believe it.

---

## Before you start

- [ ] Charge batteries; bring a spare card. 100 RAW+JPEG frames fill space fast.
- [ ] **Lock focus, exposure and ISO.** Auto mode varies exposure between
      frames, which produces visible seams when the photographs are blended
      into a texture. On a phone: tap-and-hold to lock AE/AF.
- [ ] Set white balance to a fixed preset, not auto.
- [ ] Turn the flash off. It moves the highlights with the camera, which is the
      worst case for multi-view stereo.
- [ ] Clean the lens.

## Lighting

- [ ] Diffuse, even light — a lightbox, a softbox, or open shade outdoors.
- [ ] No direct point sources. Gilded and mirrored decoration turns them into
      specular highlights, which is the failure mode the project predicts.
- [ ] Keep the lighting **fixed relative to the mask** if the mask is moving on
      a turntable; keep it fixed relative to the room if you are moving.
- [ ] Avoid coloured bounce from nearby walls.

## Background

- [ ] Plain, matte, non-reflective. Mid-grey or black cloth works well.
- [ ] Nothing shiny in frame.
- [ ] For walk-around capture, a *textured* background actually helps SfM —
      it is only turntable capture that needs a plain one plus masking.

## The shoot

- [ ] **60–100 photographs.** Fewer than ~40 and coverage suffers badly.
- [ ] **60–70% overlap between consecutive frames.** Rule of thumb: the object
      should not appear to jump between two successive photos. Small steps.
- [ ] **Three elevations**, a full circle at each:
  - [ ] eye level — the face and its relief
  - [ ] high angle, looking down — the crown (*mongkut*)
  - [ ] low angle, looking up — the jaw and the underside
- [ ] Extra close-ups of fine crown ornament, still overlapping their neighbours.
- [ ] Shoot the deep recesses — eye sockets, an open fanged mouth — from several
      angles. These are the regions that end up as holes; more viewpoints is the
      only fix.
- [ ] Keep roughly constant distance so scale and sharpness stay uniform.
- [ ] Do not change lenses or zoom mid-set. If you must, set
      `sfm.single_camera: false`.

## Before you pack up

Run the QC on the spot — it takes under a minute and tells you whether to
reshoot while you still can:

```bash
python scripts/00_prepare_images.py -c configs/sample.yaml --input /path/to/photos
```

Check that it reports:

- [ ] no images below the sharpness floor
- [ ] exposure drift within tolerance (otherwise exposure was not locked)
- [ ] no consecutive pair below the overlap threshold — each one is a gap
- [ ] masking status matches your capture regime

Then, if time allows, run SfM before leaving:

```bash
python scripts/01_sfm.py -c configs/sample.yaml
```

- [ ] nearly all images registered
- [ ] mean reprojection error below ~1 px
- [ ] no large azimuth gap and an elevation span of at least ~30°

A failure at this point is recoverable while you are still in the room. It is
not recoverable afterwards.

---

## Second subject, if time permits

A monkey-role mask with an open, fanged mouth — different geometry, and its
deep mouth cavity is a harder test of whether the pipeline generalises beyond
one object.
