"""Khon mask 3D reconstruction from multi-view photography.

A classical structure-from-motion / multi-view-stereo pipeline:

    images -> SfM poses (COLMAP) -> dense cloud (MVS) -> Poisson mesh -> evaluation

The package holds all logic; ``scripts/`` contains thin numbered CLI wrappers
that call into it. Every stage reads and writes a run directory under
``data/runs/<run_id>/`` so that a run is fully described by its config.
"""

__version__ = "0.1.0"
