---
paths:
  - "khon_recon/**/*.py"
  - "scripts/**/*.py"
---

# pycolmap 4.1.1 and Open3D 0.19 API notes

Verified against the installed versions in the `khon` env. These are the calls
that are easy to get wrong because the property/method split is inconsistent and
the errors are late and confusing.

## pycolmap: property vs method

| Call | Form |
| --- | --- |
| `image.has_pose` | **property** — `image.has_pose()` raises `'bool' object is not callable` |
| `image.cam_from_world()` | **method**, returns `Rigid3d` |
| `image.projection_center()` | method |
| `image.project_point(xyz)` | method |
| `point3D.track.length()` | method |
| `point3D.track.elements` | property (a list) |
| `reconstruction.num_reg_images()` | method |
| `reconstruction.point3D_ids()` | method returning a **set** — not subscriptable; `sorted()` it first |

Projecting a world point (this is Eq. (1) in the paper):

```python
xyz_cam = image.cam_from_world() * xyz      # Rigid3d * ndarray works
if xyz_cam[2] <= 0:                          # behind the camera
    continue
uv = camera.img_from_cam(xyz_cam)            # applies intrinsics + distortion
```

## pycolmap: options

- `FeatureExtractionOptions` wraps SIFT settings under `.sift`
  (`options.sift.max_num_features`), while `use_gpu` / `max_image_size` sit on
  the outer object.
- `ImageReaderOptions.mask_path` is how masks reach COLMAP. Mask filenames keep
  the full image name plus `.png`: `IMG_0001.jpg` → `IMG_0001.jpg.png`, black =
  ignore. Use `io_utils.mask_path_for`.
- `IncrementalPipelineOptions.image_names` restricts which images are mapped;
  this is what implements the hold-out split.
- Set `random_seed = 0`. Ablations are only comparable if mapping is deterministic.

## Open3D naming

- `o3d.geometry.KDTreeSearchParamKNN` — capital `KNN`, not `Knn`.
- Poisson returns a `(mesh, densities)` tuple; the densities are needed for the
  low-density trim that removes invented surface.
- `mesh.is_watertight()`, `get_volume()`, `get_surface_area()` are methods.
  `get_volume()` throws unless the mesh is watertight — guard it.

## Camera conversion

`render.camera_to_opencv` handles SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL
and OPENCV. Anything else raises rather than silently mis-projecting. The
conversion is validated by projecting tracks with `cv2.projectPoints` and
checking sub-pixel agreement with COLMAP's own projection — if you touch it,
re-run that check.

Ray casting is a pinhole projection and ignores distortion, so **undistort the
photograph** (`render.undistort_photo`) rather than distorting the render when
comparing the two.
